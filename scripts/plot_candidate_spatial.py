from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.io import open_h5ad, paired_observations, read_feature_matrix, read_links

# Match the plotting convention in gw12.ipynb: RNA is blue, enhancer
# accessibility is magenta/pink.
RNA_COLORS=['#EDEDED','#FFF7F3','#F7FCF0','#E0F3DB','#CCEBC5','#7BCCC4','#4EB3D3','#2B8CBE','#0868AC','#084081']
ATAC_COLORS=['#EDEDED','#FFF7F3','#FDE0DD','#FCC5C0','#FA9FB5','#F768A1','#DD3497','#AE017E']
RNA_CMAP=LinearSegmentedColormap.from_list('expression',RNA_COLORS,N=256)
ATAC_CMAP=LinearSegmentedColormap.from_list('accessibility',ATAC_COLORS,N=256)

mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,
                     'axes.spines.right':False,'axes.spines.top':False})

def _scale(values):
    v=np.asarray(values,float); return float(np.nanpercentile(v[v>0],99)) if np.any(v>0) else 1.

def _panel(ax,x,y,values,title,cmap):
    vmax=_scale(values); norm=Normalize(vmin=0,vmax=vmax,clip=True)
    ax.scatter(x,y,c=np.asarray(values),s=2.5,cmap=cmap,norm=norm,linewidths=0,rasterized=True)
    ax.set_title(title,fontsize=8,loc='left',pad=3); ax.set_aspect('equal'); ax.axis('off')
    return vmax

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rna',default='data/GW12_spatial_RNA.h5ad')
    p.add_argument('--enhancer',default='data/GW12_spatial_ATAC.h5ad')
    p.add_argument('--links',required=True)
    p.add_argument('--candidate-table',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--n-genes',type=int,default=10)
    p.add_argument('--n-enhancers',type=int,default=10)
    p.add_argument('--genes',nargs='*')
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    candidates=pd.read_csv(args.candidate_table,sep='\t').head(args.n_genes)
    genes=(args.genes if args.genes else candidates.gene.tolist())
    for g in ['NEFM','NEFL']:
        if g not in genes: genes.append(g)
    rna=open_h5ad(args.rna); atac=open_h5ad(args.enhancer); obs,coords=paired_observations(rna,atac)
    rna_matrix=rna.X.to_memory() if hasattr(rna.X,'to_memory') else None
    atac_matrix=atac.raw.X.to_memory() if getattr(atac,'raw',None) is not None and hasattr(atac.raw.X,'to_memory') else None
    atac_vars=atac.raw.var_names if atac_matrix is not None else atac.var_names
    links=read_links(args.links,rna.var_names,atac.var_names,top_k=20)
    domain_path=Path(args.candidate_table).with_name('enhancer_domain_specificity.tsv')
    domain=pd.read_csv(domain_path,sep='\t') if domain_path.exists() else pd.DataFrame()
    overview=[]
    for gene in genes:
        es=links[links.gene==gene].copy()
        if len(es)==0: continue
        if len(domain):
            dd=domain[domain.gene==gene].groupby('enhancer').local_enrichment.agg(['min','max'])
            if len(dd):
                dd['specificity']=dd['max']-dd['min']
                # Prefer enhancers enriched in exactly one expression domain.
                # The .05 threshold matches the domain-discovery scan.
                raw=domain[domain.gene==gene].pivot_table(index='enhancer',columns='domain',values='local_enrichment',aggfunc='max').fillna(0)
                specific=raw.gt(.05).sum(axis=1)==1
                chosen=dd[specific.reindex(dd.index,fill_value=False)].sort_values('specificity',ascending=False).head(args.n_enhancers).index.tolist()
                if len(chosen)<args.n_enhancers:
                    extra=dd.drop(index=chosen,errors='ignore').sort_values('specificity',ascending=False).head(args.n_enhancers-len(chosen)).index.tolist()
                    chosen+=extra
                es=pd.concat([es[es.enhancer.isin(chosen)],es[~es.enhancer.isin(chosen)]],ignore_index=True).drop_duplicates('enhancer').head(args.n_enhancers)
        else: es=es.sort_values('prior_score',ascending=False).head(args.n_enhancers)
        enh=es.enhancer.tolist()
        y=read_feature_matrix(rna,[gene],obs,matrix=rna_matrix,matrix_var_names=rna.var_names)[:,0]
        X=read_feature_matrix(atac,enh,obs,matrix=atac_matrix,matrix_var_names=atac_vars)
        ncols=4; n_panels=1+len(enh); nrows=int(np.ceil(n_panels/ncols))
        fig,axes=plt.subplots(nrows,ncols,figsize=(9.0,2.9*nrows),constrained_layout=True)
        axes=np.asarray(axes).reshape(-1)
        _panel(axes[0],coords[:,0],coords[:,1],y,gene+' expression',RNA_CMAP)
        for ax,j in zip(axes[1:],range(len(enh))):
            _panel(ax,coords[:,0],coords[:,1],X[:,j],enh[j],ATAC_CMAP)
        for ax in axes[1+len(enh):]: ax.axis('off')
        fig.suptitle(f'{gene}: spatial expression and enhancer accessibility',fontsize=10,x=.02,ha='left')
        fig.savefig(out/f'{gene}_spatial.png',dpi=300,bbox_inches='tight')
        fig.savefig(out/f'{gene}_spatial.pdf',bbox_inches='tight'); plt.close(fig)
        overview.append((gene,y))
    if overview:
        fig,axes=plt.subplots(3,4,figsize=(8.4,6.2),constrained_layout=True)
        for ax,(gene,y) in zip(axes.flat,overview): _panel(ax,coords[:,0],coords[:,1],y,gene,RNA_CMAP)
        for ax in axes.flat[len(overview):]: ax.axis('off')
        fig.savefig(out/'candidate_expression_overview.png',dpi=300,bbox_inches='tight')
        fig.savefig(out/'candidate_expression_overview.pdf',bbox_inches='tight'); plt.close(fig)
    rna.file.close(); atac.file.close()
    print('plotted genes:',len(overview))

if __name__=='__main__': main()
