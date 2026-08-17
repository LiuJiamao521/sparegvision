from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from sparegvision.io import read_links
from .complexity import enhancer_evidence_table, gene_complexity_summary


DEFAULT_ROOT = Path("/cluster3/labData/jiamao/SpaRegVision/GW12")


def _grid(values, rows, cols, shape):
    result = np.full(shape, np.nan, dtype=float)
    result[rows, cols] = np.asarray(values, dtype=float)
    return result


def _read_columns(adata, names):
    indices=adata.var_names.get_indexer(names)
    if np.any(indices<0):
        raise KeyError(f"missing features: {[names[i] for i in np.flatnonzero(indices<0)[:5]]}")
    columns=[]
    for index in indices:
        column=adata.X[:,int(index)]
        if hasattr(column,"to_memory"):
            column=column.to_memory()
        if hasattr(column,"toarray"):
            column=column.toarray()
        columns.append(np.asarray(column,dtype=float).reshape(-1))
    return np.column_stack(columns)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", type=Path, default=DEFAULT_ROOT / "rectified_h/GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad")
    parser.add_argument("--atac", type=Path, default=DEFAULT_ROOT / "rectified_h/GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad")
    parser.add_argument("--links", type=Path, default=DEFAULT_ROOT / "region_to_gene_adj.tsv")
    parser.add_argument("--output", type=Path, default=Path("results/GW12/complexity_pilot_v1"))
    parser.add_argument("--genes", nargs="+", default=["NEFM", "NEFL", "VIM", "SLC1A3", "HOXB8"])
    parser.add_argument("--screen-ranking", type=Path, default=None)
    parser.add_argument("--top-screen", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--core-threshold",type=float,default=0.35)
    parser.add_argument("--regional-local-threshold",type=float,default=0.20)
    parser.add_argument("--evidence-threshold",type=float,default=0.005)
    parser.add_argument("--qvalue-threshold",type=float,default=0.10)
    parser.add_argument("--atac-memmap",type=Path,default=Path("/tmp/gw12_rectified_h_atac_log1p_float32_transposed.memmap"))
    args = parser.parse_args()

    rna = ad.read_h5ad(args.rna, backed="r")
    atac = ad.read_h5ad(args.atac, backed="r")
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC observations are not aligned")
    rows = rna.obs["grid_row"].to_numpy(int)
    cols = rna.obs["grid_col"].to_numpy(int)
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    observed = np.zeros(shape, dtype=bool)
    observed[rows, cols] = True

    atac_cache=np.memmap(args.atac_memmap,dtype="float32",mode="r",shape=(atac.shape[1],atac.shape[0]))
    links = read_links(args.links, rna.var_names, atac.var_names, top_k=args.top_k)
    if args.screen_ranking is not None:
        screen = pd.read_csv(args.screen_ranking, sep="\t")
        args.genes = screen.nsmallest(args.top_screen, "screen_rank")["gene"].tolist()
    genes = [gene for gene in args.genes if gene in set(links["gene"])]
    all_evidence = []
    all_complexity = []
    for gene_index, gene in enumerate(genes):
        gene_links = links[links["gene"] == gene].copy()
        enhancer_names = gene_links["enhancer"].tolist()
        gene_values = _read_columns(rna,[gene])[:,0]
        enhancer_indices=atac.var_names.get_indexer(enhancer_names)
        if np.any(enhancer_indices<0):
            raise KeyError("candidate enhancer missing from ATAC")
        enhancer_values=np.asarray(atac_cache[enhancer_indices,:],dtype=float).T
        gene_map = _grid(gene_values, rows, cols, shape)
        enhancer_maps = np.stack([
            _grid(enhancer_values[:, j], rows, cols, shape)
            for j in range(len(enhancer_names))
        ])
        mask = observed & np.isfinite(gene_map)
        enhancer_maps = np.nan_to_num(enhancer_maps)
        gene_map = np.nan_to_num(gene_map)
        evidence = enhancer_evidence_table(
            enhancer_maps,
            gene_map,
            mask,
            gene=gene,
            n_permutations=args.permutations,
            seed=args.seed + gene_index,
            core_threshold=args.core_threshold,
            regional_local_threshold=args.regional_local_threshold,
            evidence_threshold=args.evidence_threshold,
            qvalue_threshold=args.qvalue_threshold,
        )
        evidence["enhancer"] = enhancer_names
        prior = gene_links.set_index("enhancer")
        evidence["prior_score"] = evidence["enhancer"].map(prior["prior_score"])
        evidence["prior_signed_score"] = evidence["enhancer"].map(prior["prior_signed_score"])
        all_evidence.append(evidence)
        summary = gene_complexity_summary(evidence, gene=gene)
        summary["mean_core_concordance"] = evidence.loc[
            evidence["predicted_class"] == "core_concordant", "global_concordance"
        ].mean()
        all_complexity.append(summary)

    evidence_table = pd.concat(all_evidence, ignore_index=True) if all_evidence else pd.DataFrame()
    complexity_table = pd.concat(all_complexity, ignore_index=True) if all_complexity else pd.DataFrame()
    if not complexity_table.empty:
        complexity_table = complexity_table.sort_values("complexity_score", ascending=False)
        complexity_table["complexity_rank"] = np.arange(1, len(complexity_table) + 1)

    args.output.mkdir(parents=True, exist_ok=True)
    evidence_table.to_csv(args.output / "enhancer_evidence.tsv", sep="\t", index=False)
    complexity_table.to_csv(args.output / "gene_complexity.tsv", sep="\t", index=False)
    manifest = {
        "dataset": "GW12_rectified_h",
        "rna": str(args.rna),
        "atac": str(args.atac),
        "links": str(args.links),
        "genes_requested": args.genes,
        "genes_analyzed": genes,
        "top_k": args.top_k,
        "permutations": args.permutations,
        "seed": args.seed,
        "atac_memmap":str(args.atac_memmap),
        "thresholds":{"core":args.core_threshold,"regional_local":args.regional_local_threshold,
                      "evidence":args.evidence_threshold,"qvalue":args.qvalue_threshold},
        "shape": list(shape),
        "n_observations": int(observed.sum()),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(complexity_table.to_string(index=False))
    rna.file.close()
    atac.file.close()


if __name__ == "__main__":
    main()
