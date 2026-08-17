from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_TSV = ROOT / "plot" / "gene_batch_rectified_h_gene_divergence_all15038.tsv"
OUT_TSV = ROOT / "plot" / "gene_batch_rectified_h_gene_divergence_absolute_all15038.tsv"
OUT_TARGET_TSV = ROOT / "plot" / "gene_batch_rectified_h_nefm_nefl_rank_absolute_all15038.tsv"


def main():
    df = pd.read_csv(IN_TSV, sep="\t")
    df["rank_eligible"] = df["rank_eligible"].fillna(False).astype(bool)

    df["absolute_divergence_score"] = (
        0.6 * df["median_difference_structure_score"].astype(float)
        + 0.4 * df["top3_mean_difference_structure_score"].astype(float)
    )

    eligible = df[df["rank_eligible"]].copy()
    eligible = eligible.sort_values(
        ["absolute_divergence_score", "median_difference_structure_score", "top3_mean_difference_structure_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    eligible["absolute_rank"] = np.arange(1, len(eligible) + 1)
    df = df.merge(eligible[["gene", "absolute_rank"]], on="gene", how="left")
    df.to_csv(OUT_TSV, sep="\t", index=False)

    targets = df[df["gene"].isin(["NEFM", "NEFL"])].copy()
    targets.to_csv(OUT_TARGET_TSV, sep="\t", index=False)
    print(targets[[
        "gene", "absolute_rank", "n_enhancers_retained",
        "median_difference_structure_score",
        "top3_mean_difference_structure_score",
        "absolute_divergence_score"
    ]].to_csv(sep="\t", index=False))


if __name__ == "__main__":
    main()
