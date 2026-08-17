from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .complexity import enhancer_evidence_table, gene_complexity_summary
from .complexity_simulation import simulate_mixed_regulatory_complexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, default=Path("results/simulation/mixed_complexity_smoke"))
    args = parser.parse_args()
    sim = simulate_mixed_regulatory_complexity(seed=args.seed)
    evidence = enhancer_evidence_table(
        sim.enhancer_maps,
        sim.gene_map,
        sim.tissue_mask,
        gene=sim.gene_id,
        truth_classes=sim.enhancer_classes,
        truth_domains=sim.enhancer_domains,
    )
    complexity = gene_complexity_summary(evidence, gene=sim.gene_id)
    truth_regional = evidence["truth_class"].isin(
        ["regional_specific", "regional_redundant"]
    ).to_numpy()
    predicted_regional = (evidence["predicted_class"] == "regional_candidate").to_numpy()
    tp = int((truth_regional & predicted_regional).sum())
    fp = int((~truth_regional & predicted_regional).sum())
    fn = int((truth_regional & ~predicted_regional).sum())
    metrics = {
        "seed": args.seed,
        "regional_average_precision": float(
            average_precision_score(truth_regional, evidence["regional_evidence_score"])
        ),
        "regional_precision": tp / max(tp + fp, 1),
        "regional_recall": tp / max(tp + fn, 1),
        "regional_false_discovery_rate": fp / max(tp + fp, 1),
        "core_recall": float(
            np.mean(
                evidence.loc[evidence["truth_class"] == "core_concordant", "predicted_class"]
                == "core_concordant"
            )
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(args.output / "enhancer_evidence.tsv", sep="\t", index=False)
    complexity.to_csv(args.output / "gene_complexity.tsv", sep="\t", index=False)
    pd.DataFrame([metrics]).to_csv(args.output / "benchmark_metrics.tsv", sep="\t", index=False)
    np.savez_compressed(
        args.output / "mixed_complexity_maps.npz",
        gene=sim.gene_map,
        enhancers=sim.enhancer_maps,
        tissue_mask=sim.tissue_mask,
        enhancer_domains=sim.enhancer_domains,
    )
    (args.output / "manifest.json").write_text(
        json.dumps({"scenario": "mixed_regulatory_complexity", **metrics}, indent=2)
    )
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(evidence.sort_values("regional_evidence_score", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
