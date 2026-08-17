from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import ElasticNet
from .losses import attribution_total_variation, masked_huber, spatial_block_safety
from .simulation import SpatialSimulation, simulate_spatial_decomposition

@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: list[dict]

def simulation_to_tensors(sim: SpatialSimulation, visible_mask, device="cpu", include_gene_context=False):
    visible=np.asarray(visible_mask,bool)&sim.tissue_mask
    if include_gene_context:
        channel_0=sim.gene_map*visible
        channel_1=visible.astype("float32")
    else:
        yy,xx=np.mgrid[0:sim.gene_map.shape[0],0:sim.gene_map.shape[1]]
        channel_0=(xx/max(sim.gene_map.shape[1]-1,1)).astype("float32")
        channel_1=(yy/max(sim.gene_map.shape[0]-1,1)).astype("float32")
    context=np.stack([channel_0,channel_1,sim.tissue_mask.astype("float32")])
    design=sim.enhancer_maps[:,visible].T
    global_model=ElasticNet(alpha=1e-3,l1_ratio=.5,positive=True,max_iter=5000,random_state=0)
    global_model.fit(design,sim.gene_map[visible])
    coefficients=global_model.coef_.astype("float32")
    intercept=np.asarray([global_model.intercept_],dtype="float32")
    return {"enhancers":torch.as_tensor(sim.enhancer_maps[None,:,None],device=device),
            "gene_context":torch.as_tensor(context[None],device=device),
            "target":torch.as_tensor(sim.gene_map[None,None],device=device),
            "truth_attribution":torch.as_tensor(sim.attribution[None],device=device),
            "tissue_mask":torch.as_tensor(sim.tissue_mask[None,None],device=device),
            "visible_mask":torch.as_tensor(visible[None,None],device=device),
            "enhancer_mask":torch.ones((1,len(sim.enhancer_maps)),dtype=torch.bool,device=device),
            "global_coefficients":torch.as_tensor(coefficients[None],device=device),
            "global_intercept":torch.as_tensor(intercept[None],device=device)}

def make_simulation_batch(scenario,seeds,shape,device="cpu"):
    items=[]; xx=np.indices(shape)[1]
    for offset,seed in enumerate(seeds):
        sim=simulate_spatial_decomposition(scenario,shape=shape,seed=seed)
        start=(offset%5)*shape[1]//5; end=(offset%5+1)*shape[1]//5
        items.append(simulation_to_tensors(sim,~((xx>=start)&(xx<end)),device))
    return {k:torch.cat([x[k] for x in items]) for k in items[0]}

def _loss(model,batch,tv_weight,entropy_weight,safety_weight=.5,n_spatial_blocks=5,
          attribution_weight=.2):
    out=model(batch["enhancers"],batch["gene_context"],batch["enhancer_mask"],batch["global_coefficients"],batch["global_intercept"])
    # Simulation RNA is never an input, so all tissue pixels are valid supervision.
    rec=masked_huber(out["gene_prediction"],batch["target"],batch["tissue_mask"])
    tv=attribution_total_variation(out["attribution_maps"],batch["tissue_mask"])
    contribution_penalty=out["enhancer_contribution_maps"].abs().mean()
    safety=spatial_block_safety(out["gene_prediction"],out["global_prediction"],
                                batch["target"],batch["tissue_mask"],n_spatial_blocks)
    truth=batch.get("truth_attribution")
    if truth is None:
        attribution_loss=rec.new_zeros(())
    else:
        weights=batch["tissue_mask"].to(out["attribution_logits"].dtype)
        positives=(truth*weights).sum(dim=(0,2,3))
        negatives=((1-truth)*weights).sum(dim=(0,2,3))
        positive_weight=torch.where(positives>0,negatives/positives.clamp_min(1),torch.ones_like(positives))
        pixel_loss=F.binary_cross_entropy_with_logits(out["attribution_logits"],truth,reduction="none",
                                                       pos_weight=positive_weight[None,:,None,None])
        attribution_loss=(pixel_loss*weights).sum()/(weights.sum()*truth.shape[1]).clamp_min(1)
    total=(rec+tv_weight*tv+entropy_weight*contribution_penalty+
           safety_weight*safety+attribution_weight*attribution_loss)
    return total,out,{"reconstruction":rec,"safety":safety,"attribution":attribution_loss}

def train_model(model,train_batch,validation_batch,epochs=100,learning_rate=3e-4,
                weight_decay=1e-4,tv_weight=.02,entropy_weight=.005,safety_weight=.5,
                n_spatial_blocks=5,attribution_weight=.2,attribution_warmup_epochs=50,patience=20):
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    best_state=deepcopy(model.state_dict()); best=float("inf"); best_epoch=-1; stale=0; history=[]
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(); loss,_,train_parts=_loss(model,train_batch,tv_weight,entropy_weight,safety_weight,n_spatial_blocks,attribution_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        model.eval()
        with torch.no_grad():
            val,_,val_parts=_loss(model,validation_batch,tv_weight,entropy_weight,safety_weight,n_spatial_blocks,attribution_weight)
            if epoch < attribution_warmup_epochs and "truth_attribution" in validation_batch:
                val=attribution_weight*val_parts["attribution"]
        row={"epoch":epoch,"train_loss":float(loss),"validation_loss":float(val),
             "train_reconstruction":float(train_parts["reconstruction"]),
             "train_safety":float(train_parts["safety"]),
             "train_attribution":float(train_parts["attribution"]),
             "validation_reconstruction":float(val_parts["reconstruction"]),
             "validation_safety":float(val_parts["safety"]),
             "validation_attribution":float(val_parts["attribution"])}; history.append(row)
        if epoch == attribution_warmup_epochs:
            best=float("inf"); stale=0
        if epoch < attribution_warmup_epochs:
            continue
        if row["validation_loss"]<best-1e-7:
            best=row["validation_loss"]; best_epoch=epoch; best_state=deepcopy(model.state_dict()); stale=0
        else: stale+=1
        if stale>=patience: break
    model.load_state_dict(best_state)
    return TrainingResult(best_epoch,best,history)

