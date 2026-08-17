from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.config import load_config
from sparegvision.io import open_h5ad, read_links

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rna',required=True); p.add_argument('--enhancer',required=True)
    p.add_argument('--links',required=True); p.add_argument('--config',required=True)
    p.add_argument('--output',required=True); p.add_argument('--batch-size',type=int,default=50)
    p.add_argument('--start',type=int,default=0); p.add_argument('--end',type=int,default=None)
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(args.config); r=open_h5ad(args.rna); a=open_h5ad(args.enhancer)
    links=read_links(args.links,r.var_names,a.var_names,cfg['candidate_top_k'])
    genes=(links.groupby('gene')['prior_score'].sum().sort_values(ascending=False).index.tolist())
    r.file.close(); a.file.close()
    manifest=[]
    range_end=len(genes) if args.end is None else min(args.end,len(genes))
    for start in range(args.start,range_end,args.batch_size):
        end=min(start+args.batch_size,range_end); batch=out/f'batch_{start:05d}_{end:05d}'
        if (batch/'gene_spatial_scores.tsv').exists():
            manifest.append(str(batch)); continue
        code=("from sparegvision.pipeline import run_mvp; "
              "from sparegvision.config import load_config; "
              f"c=load_config({args.config!r}); c['gene_start']={start}; c['gene_end']={end}; "
              f"c['max_genes']={end-start}; c['domain_only']=True; "
              f"run_mvp({args.rna!r},{args.enhancer!r},{args.links!r},{str(batch)!r},c)")
        print(f'RUNNING {start}:{end}',flush=True)
        subprocess.run([sys.executable,'-c',code],check=True)
        manifest.append(str(batch))
    frames={name:[] for name in ('gene_spatial_scores','combinatorial_scores','enhancer_domain_specificity')}
    for b in manifest:
        for name in frames:
            q=Path(b)/(name+'.tsv')
            if q.exists() and q.stat().st_size: frames[name].append(pd.read_csv(q,sep='\t'))
    for name,parts in frames.items():
        if parts: pd.concat(parts,ignore_index=True).to_csv(out/(name+'.tsv'),sep='\t',index=False)
    (out/f'manifest_{args.start:05d}_{range_end:05d}.json').write_text(json.dumps({'n_genes':len(genes),'start':args.start,'end':range_end,'batch_size':args.batch_size,'batches':manifest},indent=2))
    print(f'COMPLETE range={args.start}:{range_end} batches={len(manifest)}',flush=True)

if __name__=='__main__': main()
