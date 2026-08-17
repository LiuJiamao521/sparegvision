from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from .ablation import enhancer_ablation
from .baselines import best_single_enhancer,global_nonnegative_elastic_net,spatial_mean_baseline
from .metrics import prediction_metrics
from .model import SpatialAttributionSetNetwork
from .gating import calibrate_spatial_gate
from .calibration import nested_spatial_feature_gate
from .simulation import simulate_spatial_decomposition
from .training import make_simulation_batch,simulation_to_tensors,train_model

def main():
    p=argparse.ArgumentParser(); p.add_argument("--scenario",default="complementary")
    p.add_argument("--output",type=Path,default=Path("results/simulation/complementary_smoke"))
    p.add_argument("--epochs",type=int,default=30); p.add_argument("--height",type=int,default=24)
    p.add_argument("--width",type=int,default=32); p.add_argument("--train-samples",type=int,default=8)
    p.add_argument("--seed",type=int,default=20260814); p.add_argument("--test-seed",type=int,default=None); p.add_argument("--test-fold",type=int,default=4)
    p.add_argument("--gate-policy",choices=("strict","majority","robust"),default="robust")
    p.add_argument("--calibration-mode",choices=("nested_feature","scalar"),default="nested_feature")
    p.add_argument("--internal-fold-layout",choices=("checkerboard","vertical"),default="checkerboard")
    p.add_argument("--gate-min-folds",type=int,default=3)
    p.add_argument("--gate-min-mean-relative-gain",type=float,default=0.002)
    p.add_argument("--gate-max-fold-relative-loss",type=float,default=0.01)
    p.add_argument("--attribution-weight",type=float,default=0.2)
    p.add_argument("--safety-weight",type=float,default=0.5)
    p.add_argument("--attribution-warmup-epochs",type=int,default=50)
    args=p.parse_args()
    if not 0 <= args.test_fold < 5: p.error("--test-fold must be in [0, 4]")
    torch.manual_seed(args.seed); np.random.seed(args.seed); shape=(args.height,args.width)
    train=make_simulation_batch(args.scenario,list(range(args.seed,args.seed+args.train_samples)),shape)
    valid=make_simulation_batch(args.scenario,[args.seed+100],shape)
    model=SpatialAttributionSetNetwork(hidden_dim=16,attention_heads=4,set_layers=1,dropout=0)
    result=train_model(model,train,valid,epochs=args.epochs,patience=max(5,args.epochs//4),
                       attribution_weight=args.attribution_weight,safety_weight=args.safety_weight,
                       attribution_warmup_epochs=args.attribution_warmup_epochs)
    test_seed=args.seed+200 if args.test_seed is None else args.test_seed
    sim=simulate_spatial_decomposition(args.scenario,shape=shape,seed=test_seed)
    xx=np.indices(shape)[1]; start=args.test_fold*args.width//5; end=(args.test_fold+1)*args.width//5
    visible=~((xx>=start)&(xx<end)); batch=simulation_to_tensors(sim,visible)
    model_args=(batch["enhancers"],batch["gene_context"],batch["enhancer_mask"],
                batch["global_coefficients"],batch["global_intercept"])
    with torch.no_grad():
        base=model(*model_args,spatial_strength=0.)["gene_prediction"]
        full_out=model(*model_args,spatial_strength=1.)
        full=full_out["gene_prediction"]
        fold_map_np=(xx*5//args.width).clip(max=4)
        calibration_test_fold=args.test_fold
        if args.calibration_mode=="nested_feature" and args.internal_fold_layout=="checkerboard":
            yy=np.indices(shape)[0]
            fold_map_np=((xx//max(args.width//8,1))+2*(yy//max(args.height//6,1)))%4
            calibration_test_fold=4
        if args.calibration_mode=="nested_feature":
            learned_gate=torch.sigmoid(full_out["attribution_logits"])[0].cpu().numpy()
            spatial_features=sim.enhancer_maps*learned_gate
            gate=nested_spatial_feature_gate(sim.enhancer_maps,spatial_features,sim.gene_map,
                sim.tissue_mask&visible,fold_map_np,calibration_test_fold,min_folds=args.gate_min_folds,
                min_mean_relative_gain=args.gate_min_mean_relative_gain,
                max_fold_relative_loss=args.gate_max_fold_relative_loss)
            strength=torch.as_tensor([float(gate.enabled)],dtype=base.dtype,device=base.device)
        else:
            fold_map=torch.as_tensor(fold_map_np,device=base.device)
            gate=calibrate_spatial_gate(base,full,batch["target"],batch["visible_mask"],fold_map,
                args.test_fold,policy=args.gate_policy,min_folds=args.gate_min_folds,
                min_mean_relative_gain=args.gate_min_mean_relative_gain,
                max_fold_relative_loss=args.gate_max_fold_relative_loss)
            strength=gate.strength
        batch["spatial_strength"]=strength
    domain_masks=[sim.attribution[i]>.5 for i in range(sim.attribution.shape[0])]
    domain_masks=[m for m in domain_masks if m.any()]
    rows,out=enhancer_ablation(model,batch,domain_masks=domain_masks); test=sim.tissue_mask&~visible
    pred=(gate.prediction if args.calibration_mode=="nested_feature" else out["gene_prediction"][0,0].cpu().numpy())
    metrics=[{"model":"spatial_attribution",**prediction_metrics(sim.gene_map[test],pred[test])}]
    q,y=spatial_mean_baseline(sim.gene_map,sim.tissue_mask&visible,test)
    metrics.append({"model":"spatial_mean",**prediction_metrics(y,q)})
    q,y,selected=best_single_enhancer(sim.enhancer_maps,sim.gene_map,sim.tissue_mask&visible,test)
    metrics.append({"model":"best_single","selected_enhancer":selected,**prediction_metrics(y,q)})
    q,y,coef=global_nonnegative_elastic_net(sim.enhancer_maps,sim.gene_map,sim.tissue_mask&visible,test)
    metrics.append({"model":"global_elastic_net","coefficients":json.dumps(coef.tolist()),**prediction_metrics(y,q)})
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(result.history).to_csv(args.output/"training_history.tsv",sep="\t",index=False)
    pd.DataFrame(metrics).to_csv(args.output/"metrics.tsv",sep="\t",index=False)
    pd.DataFrame(rows).to_csv(args.output/"regional_ablation.tsv",sep="\t",index=False)
    pd.DataFrame(gate.diagnostics).to_csv(args.output/"gate_diagnostics.tsv",sep="\t",index=False)
    np.savez_compressed(args.output/"spatial_outputs.npz",gene=sim.gene_map,prediction=pred,
        attribution=out["attribution_maps"][0].cpu().numpy(),
        raw_spatial_attribution=full_out["attribution_maps"][0].cpu().numpy(),
        raw_spatial_prediction=full_out["gene_prediction"][0,0].cpu().numpy(),
        contribution=out["enhancer_contribution_maps"][0].cpu().numpy(),
        background=out["background_map"][0,0].cpu().numpy(),
        truth_attribution=sim.attribution,tissue_mask=sim.tissue_mask,test_mask=test)
    torch.save(model.state_dict(),args.output/"best_model.pt")
    manifest={"scenario":args.scenario,"seed":args.seed,"test_seed":test_seed,"shape":list(shape),"best_epoch":result.best_epoch,
              "best_validation_loss":result.best_validation_loss,"test_fold":args.test_fold,
              "spatial_strength":float(strength[0]),"gate_enabled":gate.enabled,
              "calibration_mode":args.calibration_mode,
              "internal_fold_layout":args.internal_fold_layout,
              "calibrated_coefficients":(gate.coefficients.tolist() if args.calibration_mode=="nested_feature" else None),
              "calibrated_intercept":(gate.intercept if args.calibration_mode=="nested_feature" else None),
              "gate_policy":args.gate_policy,"gate_min_folds":args.gate_min_folds,
              "gate_min_mean_relative_gain":args.gate_min_mean_relative_gain,
              "gate_max_fold_relative_loss":args.gate_max_fold_relative_loss,
              "gate_mean_relative_gain":gate.mean_relative_gain,
              "gate_worst_relative_gain":gate.worst_relative_gain,
              "attribution_weight":args.attribution_weight,"safety_weight":args.safety_weight,
              "attribution_warmup_epochs":args.attribution_warmup_epochs,"device":"cpu"}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(metrics).to_string(index=False))
if __name__=="__main__": main()
