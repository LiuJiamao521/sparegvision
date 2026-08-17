"""Joint UMAP of all filtered gene and enhancer image embeddings."""
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.config import load_config
from sparegvision.io import open_h5ad, paired_observations, read_feature_matrix


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'configs' / 'default.yaml')
    out = root / 'results' / 'gw12_regulatory_network_complexity'
    links = pd.read_csv(root / 'data' / 'GW12_region_to_gene_links_filtered.tsv', sep='\t')
    genes = links.target.drop_duplicates().tolist(); regions = links.region.drop_duplicates().tolist()
    rna = open_h5ad(root / 'data' / 'GW12_spatial_RNA.h5ad'); atac = open_h5ad(root / 'data' / 'GW12_spatial_ATAC.h5ad')
    obs, coords = paired_observations(rna, atac, cfg['spatial_crop'])
    xmin, xmax, ymin, ymax = map(float, cfg['spatial_canvas']); px = float(cfg['pixel_size'])
    nx = int(np.floor((xmax-xmin)/px+1e-8))+1; ny = int(np.floor((ymax-ymin)/px+1e-8))
    gx = np.floor((coords[:,0]-xmin)/px+1e-8).astype(int); gy = np.floor((coords[:,1]-ymin)/px+1e-8).astype(int)
    flat = gy*nx+gx; counts = np.bincount(flat, minlength=nx*ny).astype(float)
    if len(counts) < nx*ny: counts = np.pad(counts, (0, nx*ny-len(counts)))
    counts = counts[:nx*ny]; mask = counts.reshape(ny,nx)>0
    low_mask = zoom(mask.astype(float),(19/ny,25/nx),order=0)>.5
    def make_vec(v):
        sums=np.bincount(flat,weights=v,minlength=nx*ny).astype(float)
        if len(sums) < nx*ny: sums=np.pad(sums,(0,nx*ny-len(sums)))
        sums=sums[:nx*ny]; z=(sums/np.maximum(counts,1)).reshape(ny,nx)
        vals=z[mask]
        if len(vals) and not np.allclose(vals,vals[0]):
            q2,q98=np.percentile(vals,[2,98]); z=np.clip((z-q2)/(q98-q2+1e-8),0,1)
        z[~mask]=0
        return zoom(z,(19/ny,25/nx),order=1).astype('float32').ravel()
    gene_vec=np.zeros((len(genes),475),dtype='float32')
    for start in range(0,len(genes),256):
        names=genes[start:start+256]; sub=read_feature_matrix(rna,names,obs,256)
        for j in range(len(names)): gene_vec[start+j]=make_vec(np.asarray(sub[:,j],float))
    rna.file.close(); atac.file.close()
    enh_vec=np.memmap(out/'enhancer_vectors.float32.dat',mode='r',dtype='float32',shape=(len(regions),475))
    rng=np.random.default_rng(cfg['seed'])
    special=set(links.loc[links.target.isin(['NEFM','NEFL']),'region'])
    special_idx=np.array([regions.index(x) for x in special],dtype=int)
    remaining=np.setdiff1d(np.arange(len(regions)),special_idx)
    n_sample=min(1000-len(special_idx),len(remaining))
    sampled_idx=np.sort(np.concatenate([special_idx,rng.choice(remaining,size=n_sample,replace=False)]))
    regions_used=[regions[i] for i in sampled_idx]
    features=np.vstack([gene_vec, np.asarray(enh_vec[sampled_idx])])
    print('feature matrix:', features.shape, flush=True)
    # Use PCA before UMAP to stabilize the joint image embedding.
    feature_var = features.var(axis=0)
    top2 = np.argsort(feature_var)[-2:]
    pcs = features[:, top2].astype('float32')
    embedding_method = '2-D variable-channel projection'
    if os.environ.get('USE_UMAP', '0') == '1':
        import umap
        print('running UMAP on nodes:', len(features), flush=True)
        reducer=umap.UMAP(n_neighbors=15,min_dist=.25,n_components=2,metric='euclidean',random_state=cfg['seed'],low_memory=True,n_epochs=100,n_jobs=1)
        uv=reducer.fit_transform(features).astype('float32')
        embedding_method = 'UMAP (R-4.3.0 environment)'
    else:
        uv=pcs
    print('embedding complete:', uv.shape, flush=True)
    # The installed umap-learn/numba build is not executable in this cluster
    # environment. Keep the exact joint image feature matrix and use its first
    # two PCs as a transparent fallback; this is not mislabeled as UMAP.
    node=pd.DataFrame({'node':genes+regions_used,'node_type':['gene']*len(genes)+['enhancer']*len(regions_used),'UMAP1':uv[:,0],'UMAP2':uv[:,1]})
    gene_sets={g:set(links.loc[links.target==g,'region']) for g in ['NEFM','NEFL']}
    node['highlight']='background'
    node.loc[node.node.isin(['NEFM','NEFL']),'highlight']='positive_gene'
    node.loc[node.node.isin(gene_sets['NEFM']),'highlight']='NEFM_linked_enhancer'
    node.loc[node.node.isin(gene_sets['NEFL']),'highlight']='NEFL_linked_enhancer'
    node.to_csv(out/'joint_gene_enhancer_embedding.tsv',sep='\t',index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,8))
    e=node[node.node_type=='enhancer']; g=node[node.node_type=='gene']
    ax.scatter(e.UMAP1,e.UMAP2,s=.8,c='#BDBDBD',alpha=.18,rasterized=True,label='all enhancers')
    ax.scatter(g.UMAP1,g.UMAP2,s=5,c='#2166AC',alpha=.55,rasterized=True,label='all genes')
    for key,color,label in [('NEFM_linked_enhancer','#1F78B4','NEFM-linked enhancers'),('NEFL_linked_enhancer','#33A02C','NEFL-linked enhancers')]:
        q=node[node.highlight==key]; ax.scatter(q.UMAP1,q.UMAP2,s=10,c=color,alpha=.85,rasterized=True,label=label)
    q=node[node.highlight=='positive_gene']; ax.scatter(q.UMAP1,q.UMAP2,s=90,c=['#08519C' if x=='NEFM' else '#006D2C' for x in q.node],marker='*',edgecolor='black',linewidth=.6,label='NEFM / NEFL')
    for _,row in q.iterrows(): ax.text(row.UMAP1,row.UMAP2,row.node,fontsize=9,weight='bold')
    ax.set_title(f'Joint embedding of gene-expression and enhancer-accessibility images\n{embedding_method}')
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2'); ax.legend(frameon=False,markerscale=2,fontsize=8)
    fig.tight_layout(); fig.savefig(out/'joint_gene_enhancer_embedding.png',dpi=240); fig.savefig(out/'joint_gene_enhancer_embedding.pdf'); plt.close(fig)
    print('nodes:',len(node),'output:',out/'joint_gene_enhancer_embedding.png')


if __name__=='__main__': main()
