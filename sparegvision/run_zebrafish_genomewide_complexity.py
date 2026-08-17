from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .complexity import enhancer_evidence_matrix, gene_complexity_summary


DATA_ROOT = Path("/cluster3/labData/jiamao/SpaRegVision/weMERFISH")
RNA_PATH = DATA_ROOT / "weMERFISH_measured_C_6s_E1_rescaled_z.h5ad"
ATAC_PATH = DATA_ROOT / "weMERFISH_spatial_ATAC_C_6s_E1.h5ad"
LINK_PATH = Path("results/weMERFISH/wemerfish_measured_enhancer_divergence_genomewide.tsv")


def _decode(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _read_h5_columns(dataset, indices):
    indices = np.asarray(indices, dtype=int)
    order = np.argsort(indices)
    sorted_values = np.asarray(dataset[:, indices[order]], dtype=float)
    inverse = np.argsort(order)
    return sorted_values[:, inverse]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/zebrafish/genomewide_complexity_screen_v1"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-domains", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-genes", type=int, default=0)
    parser.add_argument("--core-threshold", type=float, default=0.35)
    parser.add_argument("--regional-local-threshold", type=float, default=0.20)
    parser.add_argument("--evidence-threshold", type=float, default=0.005)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    parts = args.output / "parts"
    parts.mkdir(exist_ok=True)
    links = pd.read_csv(LINK_PATH, sep="\t")
    links = (links.sort_values(["gene", "initial_rho"], ascending=[True, False])
             .groupby("gene", sort=False).head(args.top_k).copy())
    gene_order = (links.groupby("gene")["initial_rho"].sum()
                  .sort_values(ascending=False).index.tolist())
    if args.max_genes > 0:
        gene_order = gene_order[:args.max_genes]

    with h5py.File(RNA_PATH, "r") as rna, h5py.File(ATAC_PATH, "r") as atac:
        genes = _decode(rna["var"]["_index"][:])
        peaks = _decode(atac["var"]["_index"][:])
        gene_index = {name: i for i, name in enumerate(genes)}
        peak_index = {name: i for i, name in enumerate(peaks)}
        coords_all = np.asarray(atac["obsm"]["spatial_rescaled_z"][:], dtype=float)
        valid3d = np.isfinite(coords_all).all(axis=1)
        coords = coords_all[valid3d]
        scaled = (coords - coords.mean(axis=0)) / np.maximum(coords.std(axis=0), 1e-8)
        domains = KMeans(n_clusters=args.n_domains, random_state=args.seed,
                         n_init=20).fit_predict(scaled)
        domain_sizes = np.bincount(domains, minlength=args.n_domains)

        for chunk_start in range(0, len(gene_order), args.chunk_size):
            chunk = gene_order[chunk_start:chunk_start + args.chunk_size]
            evidence_path = parts / f"enhancer_evidence_{chunk_start:05d}.tsv"
            gene_path = parts / f"gene_complexity_{chunk_start:05d}.tsv"
            if evidence_path.exists() and gene_path.exists():
                print("skip", chunk_start, flush=True)
                continue
            evidence_parts = []
            gene_parts = []
            for gene in chunk:
                sub = links[links["gene"] == gene].copy()
                names = sub["interval_id"].astype(str).tolist()
                y = np.asarray(rna["X"][:, gene_index[gene]], dtype=float)[valid3d]
                indices = [peak_index[name] for name in names]
                X = np.log1p(np.maximum(_read_h5_columns(atac["X"], indices)[valid3d], 0.0))
                evidence = enhancer_evidence_matrix(
                    X, y, domains, gene=gene, enhancer_names=names,
                    core_threshold=args.core_threshold,
                    regional_local_threshold=args.regional_local_threshold,
                    evidence_threshold=args.evidence_threshold,
                )
                prior = sub.set_index("interval_id")
                evidence["initial_spearman_rho"] = evidence["enhancer"].map(prior["initial_rho"])
                evidence["link_pvalue"] = evidence["enhancer"].map(prior["pvalue_one_tailed_positive"])
                evidence_parts.append(evidence)
                gene_parts.append(gene_complexity_summary(evidence, gene=gene))
            pd.concat(evidence_parts, ignore_index=True).to_csv(evidence_path, sep="\t", index=False)
            pd.concat(gene_parts, ignore_index=True).to_csv(gene_path, sep="\t", index=False)
            print("completed", min(chunk_start + len(chunk), len(gene_order)), "/", len(gene_order), flush=True)

    evidence = pd.concat([pd.read_csv(path, sep="\t") for path in sorted(parts.glob("enhancer_evidence_*.tsv"))], ignore_index=True)
    ranking = pd.concat([pd.read_csv(path, sep="\t") for path in sorted(parts.glob("gene_complexity_*.tsv"))], ignore_index=True)
    ranking = ranking.sort_values(["complexity_score", "n_regional_candidates"], ascending=False)
    ranking["screen_rank"] = np.arange(1, len(ranking) + 1)
    evidence.to_csv(args.output / "enhancer_evidence_screen.tsv", sep="\t", index=False)
    ranking.to_csv(args.output / "gene_complexity_screen.tsv", sep="\t", index=False)
    manifest = {
        "dataset": "zebrafish_weMERFISH_6s_E1",
        "n_cells_total": int(len(valid3d)), "n_cells_valid_3d": int(valid3d.sum()),
        "n_cells_excluded_invalid_3d": int((~valid3d).sum()), "n_genes": int(len(ranking)),
        "n_enhancer_rows": int(len(evidence)), "top_k": args.top_k,
        "spatial_domains": "KMeans on standardized 3D spatial_rescaled_z",
        "n_domains": args.n_domains, "domain_sizes": domain_sizes.tolist(),
        "seed": args.seed, "stage": "screen_without_spatial_permutation",
        "requires_confirmatory_permutation": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(ranking.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
