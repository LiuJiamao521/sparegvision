from __future__ import annotations
import numpy as np
import torch

def _masked_mse(pred,target,mask):
    d=(pred-target).square(); w=mask.to(d.dtype)
    return float((d*w).sum()/w.sum().clamp_min(1))

@torch.no_grad()
def enhancer_ablation(model,batch,domain_masks=None):
    strength=batch.get("spatial_strength",1.0)
    model.eval(); base=model(batch["enhancers"],batch["gene_context"],batch["enhancer_mask"],batch["global_coefficients"],batch["global_intercept"],strength)
    test=batch["tissue_mask"]&~batch["visible_mask"]
    base_loss=_masked_mse(base["gene_prediction"],batch["target"],test); rows=[]
    for enhancer in range(batch["enhancers"].shape[1]):
        keep=batch["enhancer_mask"].clone(); keep[:,enhancer]=False
        out=model(batch["enhancers"],batch["gene_context"],keep,batch["global_coefficients"],batch["global_intercept"],strength)
        loss=_masked_mse(out["gene_prediction"],batch["target"],test)
        rows.append({"enhancer":enhancer,"domain":"all_test","baseline_loss":base_loss,
                     "ablated_loss":loss,"delta_loss":loss-base_loss})
        for domain,dm in enumerate(domain_masks or []):
            region=test&torch.as_tensor(dm,device=test.device,dtype=torch.bool)[None,None]
            b=_masked_mse(base["gene_prediction"],batch["target"],region)
            a=_masked_mse(out["gene_prediction"],batch["target"],region)
            rows.append({"enhancer":enhancer,"domain":domain,"baseline_loss":b,
                         "ablated_loss":a,"delta_loss":a-b})
    return rows,base

def attribution_mass_by_domain(attribution,domain_masks):
    a=np.asarray(attribution,float); rows=[]
    for i in range(a.shape[0]):
        for d,m in enumerate(domain_masks):
            v=a[i][np.asarray(m,bool)]
            rows.append({"enhancer":i,"domain":d,"attribution_mass":float(v.mean()) if len(v) else np.nan})
    return rows
