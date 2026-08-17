from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sparegvision.io import open_h5ad, paired_observations, read_feature_matrix
from sparegvision.rasterize import rasterize
from sparegvision.gsps import gene_spatial_score

def main():
    root=Path('results/gw12_domain_scan_batched')
    candidates=pd.read_csv(root/'domain_complexity_candidates.tsv',sep='\t')
    r=open_h5ad('data/GW12_spatial_RNA.h5ad'); a=open_h5ad('data/GW12_spatial_ATAC.h5ad'); obs,coords=paired_observations(r,a)
    matrix=r.X.to_memory() if hasattr(r.X,'to_memory') else None
    y=read_feature_matrix(r,candidates.gene.tolist(),obs,matrix=matrix,matrix_var_names=r.var_names)
    rows=[]
    for j,g in enumerate(candidates.gene):
        maps,mask,_=rasterize(y[:,j],coords,pixel_size=30,scales=(0.,1.,2.))
        s=gene_spatial_score(y[:,j],coords,maps,mask)
        rows.append({'gene':g,**s})
    gs=pd.DataFrame(rows); out=candidates.merge(gs,on='gene')
    # Moran I is the key anti-artifact gate: the two positive controls are
    # strongly positive whereas housekeeping-like artifacts are near zero.
    out['validated_discovery_score']=out.discovery_score*(out.Moran_I.clip(lower=0)/.2).clip(upper=1)
    out=out.sort_values('validated_discovery_score',ascending=False)
    out.to_csv(root/'domain_complexity_candidates_validated.tsv',sep='\t',index=False)
    print('candidates',len(out),'Moran>0.1',int((out.Moran_I>0.1).sum()))
    print(out[['gene','Moran_I','GSPS','n_expression_domains','n_domain_specific_enhancers','domain_specificity_score','validated_discovery_score']].head(30).to_string(index=False))
    r.file.close();a.file.close()

if __name__=='__main__': main()
