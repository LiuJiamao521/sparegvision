from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "plot" / "NEFM_region_to_gene_adj_filtered_spatial_difference_scores.tsv"
OUT_PNG = ROOT / "plot" / "NEFM_genomic_difference_structure_scores.png"
OUT_PDF = ROOT / "plot" / "NEFM_genomic_difference_structure_scores.pdf"
OUT_TSV = ROOT / "plot" / "NEFM_genomic_difference_structure_scores.tsv"

BLUE = "#2C7FB8"
PINK = "#C51B8A"
GREY = "#D9D9D9"
TEXT = "#222222"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


def parse_locus(locus: str):
    m = re.fullmatch(r"(chr[^:]+):(\d+)-(\d+)", locus)
    if m is None:
        raise ValueError(f"Invalid locus: {locus}")
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    return chrom, start, end


def repel_offsets(n: int):
    pattern = [0.012, 0.028, 0.044, 0.060]
    signs = [1, -1]
    offsets = []
    for i in range(n):
        mag = pattern[(i // 2) % len(pattern)]
        sign = signs[i % 2]
        offsets.append(sign * mag)
    return offsets


def main():
    df = pd.read_csv(INFILE, sep="\t")
    parsed = df["enhancer"].map(parse_locus)
    df["chrom"] = [x[0] for x in parsed]
    df["start"] = [x[1] for x in parsed]
    df["end"] = [x[2] for x in parsed]
    df["mid"] = (df["start"] + df["end"]) / 2
    df = df.sort_values(["start", "end"]).reset_index(drop=True)
    df["rank_by_score"] = df["difference_structure_score"].rank(ascending=False, method="dense").astype(int)

    chrom = df["chrom"].iloc[0]
    xmin = df["start"].min() - 2500
    xmax = df["end"].max() + 2500
    ymin = 0
    ymax = max(0.36, float(df["difference_structure_score"].max()) + 0.06)

    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)

    norm = mpl.colors.Normalize(
        vmin=float(df["difference_structure_score"].min()),
        vmax=float(df["difference_structure_score"].max()),
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "nefm_diff", [GREY, "#F4A6C7", PINK]
    )

    for _, row in df.iterrows():
        color = cmap(norm(row["difference_structure_score"]))
        ax.hlines(
            y=row["difference_structure_score"],
            xmin=row["start"],
            xmax=row["end"],
            color=color,
            linewidth=2.2,
            alpha=0.95,
            zorder=2,
        )

    pts = ax.scatter(
        df["mid"],
        df["difference_structure_score"],
        c=df["difference_structure_score"],
        cmap=cmap,
        norm=norm,
        s=18,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )

    label_order = np.argsort(df["difference_structure_score"].to_numpy())[::-1]
    offsets = repel_offsets(len(df))
    for idx, off in zip(label_order, offsets):
        row = df.iloc[idx]
        ax.text(
            row["mid"],
            row["difference_structure_score"] + off,
            row["panel_id"],
            ha="center",
            va="bottom" if off > 0 else "top",
            fontsize=6.2,
            color=TEXT,
            zorder=4,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("difference_structure_score")
    ax.set_xlabel(f"{chrom} genomic position (Mb)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1e6:.2f}"))
    ax.set_title(
        "NEFM retained enhancers: genomic distribution of difference scores",
        fontsize=8.5,
        pad=6,
    )

    cbar = fig.colorbar(pts, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("difference score", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.text(
        0.01,
        -0.02,
        "Each interval is one retained enhancer from region_to_gene_adj. "
        "Horizontal span shows enhancer coordinates; y-value is the final difference_structure_score.",
        ha="left",
        va="top",
        fontsize=6.3,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    out_df = df[
        [
            "panel_id",
            "enhancer",
            "chrom",
            "start",
            "end",
            "mid",
            "difference_structure_score",
            "diff_structure_score",
            "logfc_structure_score",
            "residual_structure_score",
            "rank_by_score",
        ]
    ].copy()
    out_df.to_csv(OUT_TSV, sep="\t", index=False)


if __name__ == "__main__":
    main()
