import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
IN_TSV = ROOT / "plot" / "gene_batch_rectified_h_gene_divergence_absolute_all15038.tsv"
OUT_PNG = ROOT / "plot" / "gene_divergence_absolute_all15038_distribution.png"
OUT_PDF = ROOT / "plot" / "gene_divergence_absolute_all15038_distribution.pdf"

TARGETS = ["NEFM", "NEFL"]

BLUE = "#2171B5"
PINK = "#DD3497"
GRAY = "#BDBDBD"
DARK = "#333333"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams.update({
    "font.size": 7.0,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def main():
    df = pd.read_csv(IN_TSV, sep="\t")
    df = df[df["rank_eligible"] == True].copy()
    df = df.sort_values("absolute_rank").reset_index(drop=True)
    df["absolute_rank"] = df["absolute_rank"].astype(int)
    df["absolute_divergence_score"] = df["absolute_divergence_score"].astype(float)

    targets = df[df["gene"].isin(TARGETS)].copy()
    if len(targets) != len(TARGETS):
        missing = sorted(set(TARGETS) - set(targets["gene"]))
        raise RuntimeError(f"Missing targets in eligible table: {missing}")

    q90 = float(df["absolute_divergence_score"].quantile(0.90))
    q99 = float(df["absolute_divergence_score"].quantile(0.99))

    fig = plt.figure(figsize=(10.8, 4.6), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Panel 1: rank curve
    x = df["absolute_rank"].to_numpy()
    y = df["absolute_divergence_score"].to_numpy()
    ax1.plot(x, y, color=GRAY, linewidth=1.0, alpha=0.9)
    ax1.scatter(x, y, s=6, color=GRAY, alpha=0.35, linewidth=0)
    ax1.axhline(q90, color="#9E9E9E", linestyle="--", linewidth=0.9)
    ax1.axhline(q99, color="#636363", linestyle=":", linewidth=0.9)

    color_map = {"NEFM": BLUE, "NEFL": PINK}
    for _, row in targets.iterrows():
        gx = int(row["absolute_rank"])
        gy = float(row["absolute_divergence_score"])
        ax1.scatter([gx], [gy], s=42, color=color_map[row["gene"]], edgecolor="black", linewidth=0.5, zorder=5)
        dy = 0.008 if row["gene"] == "NEFM" else -0.010
        ax1.text(
            gx + 180, gy + dy,
            f"{row['gene']}  rank={gx}\nscore={gy:.3f}",
            fontsize=7.0, color=DARK, va="center"
        )

    ax1.set_xlabel("Gene rank by absolute divergence score")
    ax1.set_ylabel("absolute_divergence_score")
    ax1.set_title("All 15,038 scorable genes", loc="left", fontsize=9.2, pad=4)
    ax1.set_xlim(1, len(df))
    ax1.set_ylim(bottom=0)

    # Panel 2: distribution
    bins = np.linspace(0, float(y.max()) * 1.02, 70)
    ax2.hist(y, bins=bins, color=GRAY, alpha=0.85, edgecolor="white", linewidth=0.35)
    ax2.axvline(q90, color="#9E9E9E", linestyle="--", linewidth=0.9)
    ax2.axvline(q99, color="#636363", linestyle=":", linewidth=0.9)
    for _, row in targets.iterrows():
        gx = float(row["absolute_divergence_score"])
        ax2.axvline(gx, color=color_map[row["gene"]], linewidth=1.4)
        ax2.text(
            gx, ax2.get_ylim()[1] * (0.92 if row["gene"] == "NEFM" else 0.78),
            row["gene"], color=color_map[row["gene"]], rotation=90,
            ha="center", va="top", fontsize=7.0, fontweight="bold"
        )

    ax2.set_xlabel("absolute_divergence_score")
    ax2.set_ylabel("Gene count")
    ax2.set_title("Score distribution", loc="left", fontsize=9.2, pad=4)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.19, wspace=0.28)
    fig.suptitle(
        "Genome-wide gene divergence ranking from rectified_h (absolute score)",
        x=0.08, ha="left", fontsize=11.0, fontweight="bold",
    )
    fig.text(
        0.08, 0.075,
        "Ranking uses absolute_divergence_score = 0.6 × median_difference_structure_score + "
        "0.4 × top3_mean_difference_structure_score. "
        "Only genes with at least 5 retained enhancers are ranked. NEFM and NEFL are highlighted.",
        fontsize=6.3, color="#444444",
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(targets[[
        "gene", "eligible_rank", "n_enhancers_retained",
        "absolute_rank",
        "median_difference_structure_score", "top3_mean_difference_structure_score",
        "absolute_divergence_score"
    ]].to_csv(sep="\t", index=False))


if __name__ == "__main__":
    main()
