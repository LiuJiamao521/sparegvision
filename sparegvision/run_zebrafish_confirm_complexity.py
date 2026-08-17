from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from .complexity import _column_correlations, enhancer_evidence_matrix
from .run_zebrafish_genomewide_complexity import (
    ATAC_PATH, LINK_PATH, RNA_PATH, _decode, _read_h5_columns,
)


def _bh(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def _toroidal_mappings(coords, n_permutations, seed):
    lower = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - lower, 1e-8)
    unit = (coords - lower) / span
    unit = np.minimum(unit, np.nextafter(1.0, 0.0))
    tree = cKDTree(unit, boxsize=1.0)
    rng = np.random.default_rng(seed)
    mappings = []
    for _ in range(n_permutations):
        shift = rng.uniform(0.1, 0.9, size=3)
        shifted = (unit + shift) % 1.0
        mappings.append(tree.query(shifted, k=1, workers=-1)[1])
    return np.asarray(mappings, dtype=np.int32)


def _null_max_local(x, y, domains, mappings):
    maxima = np.full(len(mappings), -np.inf, dtype=float)
    shifted = x[mappings]
    for domain in np.unique(domains):
        mask = domains == domain
        values = shifted[:, mask]
        target = y[mask]
        values = values - values.mean(axis=1, keepdims=True)
        target = target - target.mean()
        numerator = values @ target
        denominator = np.sqrt((values * values).sum(axis=1) * (target * target).sum())
        corr = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                         where=denominator > 1e-12)
        maxima = np.maximum(maxima, corr)
    return maxima


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, default=Path(
        "results/zebrafish/genomewide_complexity_screen_v1/mixed_threshold_sensitivity/mixed_complexity_consensus.tsv"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/zebrafish/mixed_complexity_confirm_top50_v1"))
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-domains", type=int, default=6)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.screen, sep="\t")
    rank_column = "consensus_rank" if "consensus_rank" in candidates else "screen_rank"
    candidates = candidates.nsmallest(args.top, rank_column)
    selected_genes = candidates["gene"].tolist()
    links = pd.read_csv(LINK_PATH, sep="\t")
    links = (links[links["gene"].isin(selected_genes)]
             .sort_values(["gene", "initial_rho"], ascending=[True, False])
             .groupby("gene", sort=False).head(args.top_k).copy())

    all_evidence = []
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
        mappings = _toroidal_mappings(coords, args.permutations, args.seed)

        for gene_number, gene in enumerate(selected_genes, start=1):
            sub = links[links["gene"] == gene].copy()
            names = sub["interval_id"].astype(str).tolist()
            indices = [peak_index[name] for name in names]
            y = np.asarray(rna["X"][:, gene_index[gene]], dtype=float)[valid3d]
            X = np.log1p(np.maximum(_read_h5_columns(atac["X"], indices)[valid3d], 0.0))
            evidence = enhancer_evidence_matrix(
                X, y, domains, gene=gene, enhancer_names=names,
                core_threshold=0.35, regional_local_threshold=0.20,
                evidence_threshold=0.005,
            )
            pvalues = []
            for column in range(X.shape[1]):
                null = _null_max_local(X[:, column], y, domains, mappings)
                observed = evidence.loc[column, "max_regional_concordance"]
                pvalues.append((1 + np.sum(null >= observed)) / (args.permutations + 1))
            evidence["permutation_pvalue"] = pvalues
            evidence["permutation_qvalue"] = _bh(pvalues)
            prior = sub.set_index("interval_id")
            evidence["initial_spearman_rho"] = evidence["enhancer"].map(prior["initial_rho"])
            evidence["link_pvalue"] = evidence["enhancer"].map(prior["pvalue_one_tailed_positive"])
            all_evidence.append(evidence)
            print("confirmed", gene_number, "/", len(selected_genes), gene, flush=True)

    evidence = pd.concat(all_evidence, ignore_index=True)
    evidence.to_csv(args.output / "enhancer_evidence_confirmed.tsv", sep="\t", index=False)
    manifest = {
        "dataset": "zebrafish_weMERFISH_6s_E1", "n_genes": len(selected_genes),
        "genes": selected_genes, "top_k": args.top_k, "n_domains": args.n_domains,
        "permutations": args.permutations, "seed": args.seed,
        "null": "3D toroidal coordinate shift followed by periodic nearest-neighbor mapping",
        "multiple_testing": "BH within each gene across candidate enhancers",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
