from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sparegvision.io import read_links
from .complexity import enhancer_evidence_matrix, gene_complexity_summary


ROOT = Path("/cluster3/labData/jiamao/SpaRegVision/GW12")
RNA_PATH = ROOT / "rectified_h/GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
ATAC_PATH = ROOT / "rectified_h/GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad"
LINK_PATH = ROOT / "region_to_gene_adj.tsv"
ATAC_CACHE = Path("/tmp/gw12_rectified_h_atac_log1p_float32_transposed.memmap")


def _load_rna_csc(path):
    with h5py.File(path, "r") as handle:
        group = handle["X"]
        shape = tuple(int(x) for x in group.attrs["shape"])
        matrix = csr_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
        )
    return matrix.tocsc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/GW12/genomewide_complexity_screen_v1"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-genes", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--core-threshold", type=float, default=0.35)
    parser.add_argument("--regional-local-threshold", type=float, default=0.20)
    parser.add_argument("--evidence-threshold", type=float, default=0.005)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    parts = args.output / "parts"
    parts.mkdir(exist_ok=True)
    rna = ad.read_h5ad(RNA_PATH, backed="r")
    atac = ad.read_h5ad(ATAC_PATH, backed="r")
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC observations are not aligned")
    rows = rna.obs["grid_row"].to_numpy(int)
    cols = rna.obs["grid_col"].to_numpy(int)
    n_rows, n_cols = int(rows.max()) + 1, int(cols.max()) + 1
    domain_labels = np.minimum(rows * 2 // n_rows, 1) * 3 + np.minimum(cols * 3 // n_cols, 2)

    links = read_links(LINK_PATH, rna.var_names, atac.var_names, top_k=args.top_k)
    gene_order = links.groupby("gene")["prior_score"].sum().sort_values(ascending=False).index.tolist()
    stop = len(gene_order) if args.max_genes <= 0 else min(len(gene_order), args.start + args.max_genes)
    genes = gene_order[args.start:stop]
    links = links[links["gene"].isin(genes)].copy()
    rna_matrix = _load_rna_csc(RNA_PATH)
    atac_cache = np.memmap(ATAC_CACHE, dtype="float32", mode="r", shape=(atac.shape[1], atac.shape[0]))
    rna_index = {name: i for i, name in enumerate(rna.var_names)}
    atac_index = {name: i for i, name in enumerate(atac.var_names)}

    for chunk_start in range(0, len(genes), args.chunk_size):
        chunk_genes = genes[chunk_start:chunk_start + args.chunk_size]
        part_id = args.start + chunk_start
        evidence_path = parts / f"enhancer_evidence_{part_id:06d}.tsv"
        gene_path = parts / f"gene_complexity_{part_id:06d}.tsv"
        if evidence_path.exists() and gene_path.exists():
            print("skip completed part", part_id, flush=True)
            continue
        evidence_parts = []
        gene_parts = []
        for gene in chunk_genes:
            gene_links = links[links["gene"] == gene]
            names = gene_links["enhancer"].tolist()
            y = np.asarray(rna_matrix[:, rna_index[gene]].toarray()).reshape(-1)
            indices = [atac_index[name] for name in names]
            X = np.asarray(atac_cache[indices, :], dtype=float).T
            evidence = enhancer_evidence_matrix(
                X, y, domain_labels, gene=gene, enhancer_names=names,
                core_threshold=args.core_threshold,
                regional_local_threshold=args.regional_local_threshold,
                evidence_threshold=args.evidence_threshold,
            )
            prior = gene_links.set_index("enhancer")
            evidence["prior_score"] = evidence["enhancer"].map(prior["prior_score"])
            evidence["prior_signed_score"] = evidence["enhancer"].map(prior["prior_signed_score"])
            evidence_parts.append(evidence)
            gene_parts.append(gene_complexity_summary(evidence, gene=gene))
        pd.concat(evidence_parts, ignore_index=True).to_csv(evidence_path, sep="\t", index=False)
        pd.concat(gene_parts, ignore_index=True).to_csv(gene_path, sep="\t", index=False)
        print("completed", min(chunk_start + len(chunk_genes), len(genes)), "/", len(genes), flush=True)

    evidence_files = sorted(parts.glob("enhancer_evidence_*.tsv"))
    gene_files = sorted(parts.glob("gene_complexity_*.tsv"))
    all_evidence = pd.concat([pd.read_csv(path, sep="\t") for path in evidence_files], ignore_index=True)
    ranking = pd.concat([pd.read_csv(path, sep="\t") for path in gene_files], ignore_index=True)
    ranking = ranking.sort_values(["complexity_score", "n_regional_candidates"], ascending=False)
    ranking["screen_rank"] = np.arange(1, len(ranking) + 1)
    all_evidence.to_csv(args.output / "enhancer_evidence_screen.tsv", sep="\t", index=False)
    ranking.to_csv(args.output / "gene_complexity_screen.tsv", sep="\t", index=False)
    manifest = {
        "dataset": "GW12_rectified_h",
        "stage": "genomewide_screen_without_permutation",
        "n_genes": len(ranking),
        "n_enhancer_rows": len(all_evidence),
        "top_k": args.top_k,
        "thresholds": {
            "core": args.core_threshold,
            "regional_local": args.regional_local_threshold,
            "evidence": args.evidence_threshold,
        },
        "domain_layout": "2x3_grid",
        "requires_confirmatory_permutation": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(ranking.head(30).to_string(index=False))
    rna.file.close()
    atac.file.close()


if __name__ == "__main__":
    main()
