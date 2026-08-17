"""Filter GW12 ScenicPLUS region-to-gene links for regulatory-network analysis."""
from pathlib import Path
import json
import anndata as ad
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    source = Path('/cluster2/huanglab/jiamao/Project/HumanSpinalCord/Work/P3/ATAC/Scenicplus/GW12/outs/region_to_gene_adj.tsv')
    out = root / 'data' / 'GW12_region_to_gene_links_filtered.tsv'
    meta_out = root / 'data' / 'GW12_region_to_gene_links_filtered.metadata.json'

    links = pd.read_csv(source, sep='\t')
    links = links.dropna(subset=['target', 'region', 'importance', 'rho', 'importance_x_abs_rho']).copy()
    rna = ad.read_h5ad(root / 'data' / 'GW12_spatial_RNA.h5ad', backed='r')
    atac = ad.read_h5ad(root / 'data' / 'GW12_spatial_ATAC.h5ad', backed='r')
    rna_genes = set(map(str, rna.var_names))
    atac_regions = set(map(str, atac.raw.var_names if atac.raw is not None else atac.var_names))
    rna.file.close(); atac.file.close()

    before = len(links)
    links = links[links.target.astype(str).isin(rna_genes) & links.region.astype(str).isin(atac_regions)]
    links = links[(links.importance > 0) & (links.rho.abs() >= 0.05)].copy()
    counts = links.groupby('target').size()
    retained_genes = counts[counts >= 5].index
    links = links[links.target.isin(retained_genes)].copy()
    links.to_csv(out, sep='\t', index=False)
    meta = {
        'source': str(source), 'filter': {
            'finite_required': True, 'importance_gt': 0, 'abs_rho_gte': 0.05,
            'max_links_per_gene': None, 'min_links_per_gene': 5,
        }, 'source_rows': int(before), 'filtered_rows': int(len(links)),
        'retained_genes': int(links.target.nunique()), 'retained_regions': int(links.region.nunique()),
        'links_per_gene_summary': counts[counts >= 5].describe().to_dict(),
    }
    meta_out.write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps(meta, indent=2, default=float))
    for gene in ['NEFM', 'NEFL']:
        x = links[links.target == gene]
        print(gene, 'links:', len(x), 'top score:', float(x.importance_x_abs_rho.max()) if len(x) else None)


if __name__ == '__main__':
    main()
