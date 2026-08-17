from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .run_zebrafish_genomewide_complexity import (
    ATAC_PATH, RNA_PATH, _decode, _read_h5_columns,
)


GTF_PATH = Path("/cluster3/labData/jiamao/Genome/zebrafish/Danio_rerio.GRCz11.113.gtf")
CORE_COLOR = "#258C86"
REGIONAL_COLOR = "#C85A78"
GENE_COLOR = "#315C88"
BACKGROUND_COLOR = "#D8DDE2"

CORE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "core", ["#F2F5F4", "#B9DDD7", "#258C86", "#0B4F4B"])
REGIONAL_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "regional", ["#F7F3F4", "#EDBCC8", "#C85A78", "#7A2440"])
GENE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "gene", ["#F2F4F7", "#B8C9DB", "#315C88", "#142E4A"])

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _gtf_gene_records(path):
    pattern = re.compile(r'gene_name "([^"]+)"')
    records = {}
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            match = pattern.search(fields[8])
            if not match:
                continue
            name = match.group(1)
            start, end = int(fields[3]), int(fields[4])
            records.setdefault(name, {
                "chrom": f"chr{fields[0]}", "start": start, "end": end,
                "strand": fields[6], "tss": end if fields[6] == "-" else start,
            })
    return records


def _display_values(values):
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return np.zeros_like(values)
    cap = max(float(np.percentile(positive, 99)), 1e-8)
    return np.clip(values, 0, cap) / cap


def _spatial_panel(ax, coords, values, projection, cmap):
    first, second = projection
    valid = np.isfinite(coords[:, first]) & np.isfinite(coords[:, second]) & np.isfinite(values)
    xy = coords[valid][:, [first, second]]
    color = values[valid]
    order = np.argsort(color)
    ax.scatter(xy[:, 0], xy[:, 1], s=0.10, c=BACKGROUND_COLOR,
               rasterized=True, linewidths=0)
    active = order[color[order] > 0]
    if len(active):
        ax.scatter(xy[active, 0], xy[active, 1], s=0.32, c=color[active],
                   cmap=cmap, vmin=0, vmax=1, rasterized=True, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _parse_interval(interval):
    chrom, start, end = str(interval).split("-")
    return f"chr{chrom}", int(start), int(end), (int(start) + int(end)) // 2


def _locus_panel(ax, gene, record, features):
    centers = [item["center"] for item in features]
    xmin = min([record["start"], record["end"], *centers]) - 2500
    xmax = max([record["start"], record["end"], *centers]) + 2500
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.8, 0.8)
    ax.axhline(0, color="#7C8389", lw=0.7)
    ax.add_patch(Rectangle((record["start"], 0.17),
                           max(record["end"] - record["start"], 200), 0.18,
                           color=GENE_COLOR))
    ax.text((record["start"] + record["end"]) / 2, 0.45, gene,
            ha="center", va="bottom", color=GENE_COLOR, fontsize=7, fontweight="bold")
    for item in features:
        color = CORE_COLOR if item["class"] == "core" else REGIONAL_COLOR
        ax.vlines(item["center"], -0.31, 0.02, color=color, lw=1.2)
        ax.scatter(item["center"], -0.34, s=12, color=color, zorder=3)
        ax.text(item["center"], -0.48, item["label"], ha="center", va="top",
                fontsize=5.5, color=color)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, position: f"{value / 1e6:.3f}"))
    ax.text(1.0, 0.94, f"{record['chrom']} position (Mb)", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color="#59636D")
    ax.tick_params(axis="x", labelsize=5.5, length=2)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#9AA0A6")


def _plot_gene(gene, rank_row, evidence, rna, atac, gene_index, peak_index,
               coords, records, output):
    gene_evidence = evidence[evidence["gene"] == gene].copy()
    core = (gene_evidence[gene_evidence["global_concordance"] >= 0.35]
            .nlargest(3, "global_concordance").copy())
    regional = gene_evidence[
        (gene_evidence["global_concordance"] < 0.35)
        & (gene_evidence["max_regional_concordance"] >= 0.20)
        & (gene_evidence["regional_evidence_score"] >= 0.005)
        & (gene_evidence["permutation_qvalue"] <= 0.10)
    ].nlargest(3, "regional_evidence_score").copy()
    selected = pd.concat([core.assign(plot_class="core"),
                          regional.assign(plot_class="regional")], ignore_index=True)
    if selected.empty:
        return None
    selected["label"] = ([f"C{i + 1}" for i in range(len(core))]
                          + [f"R{i + 1}" for i in range(len(regional))])
    interval_info = selected["enhancer"].map(_parse_interval)
    selected["chrom"] = [item[0] for item in interval_info]
    selected["start"] = [item[1] for item in interval_info]
    selected["end"] = [item[2] for item in interval_info]
    selected["center"] = [item[3] for item in interval_info]

    gene_values = np.asarray(rna["X"][:, gene_index[gene]], dtype=float)
    indices = [peak_index[name] for name in selected["enhancer"]]
    enhancer_values = np.log1p(np.maximum(_read_h5_columns(atac["X"], indices), 0))
    columns = [{"label": gene, "class": "gene", "values": gene_values,
                "title": f"{gene}\nRNA", "cmap": GENE_CMAP}]
    for position, row in selected.iterrows():
        if row["plot_class"] == "core":
            title = f"{row['label']} core\nglobal r={row['global_concordance']:.2f}"
            cmap = CORE_CMAP
        else:
            title = (f"{row['label']} regional · D{int(row['best_domain'])}\n"
                     f"local r={row['max_regional_concordance']:.2f}, q={row['permutation_qvalue']:.3f}")
            cmap = REGIONAL_CMAP
        columns.append({"label": row["label"], "class": row["plot_class"],
                        "values": enhancer_values[:, position], "title": title, "cmap": cmap})

    ncols = len(columns)
    fig = plt.figure(figsize=(max(7.2, 1.48 * ncols), 6.3), facecolor="white")
    grid = fig.add_gridspec(4, ncols, height_ratios=[0.72, 1.65, 1.65, 0.95],
                           hspace=0.28, wspace=0.08, left=0.055, right=0.985,
                           top=0.93, bottom=0.09)
    locus = fig.add_subplot(grid[0, :])
    locus_features = [{"center": int(row.center), "label": row.label,
                       "class": row.plot_class} for row in selected.itertuples()]
    _locus_panel(locus, gene, records[gene], locus_features)

    for column, item in enumerate(columns):
        values = _display_values(item["values"])
        ax_xy = fig.add_subplot(grid[1, column])
        ax_xz = fig.add_subplot(grid[2, column])
        _spatial_panel(ax_xy, coords, values, (0, 1), item["cmap"])
        _spatial_panel(ax_xz, coords, values, (0, 2), item["cmap"])
        ax_xy.set_title(item["title"], fontsize=6.3, pad=3,
                        color={"gene": GENE_COLOR, "core": CORE_COLOR,
                               "regional": REGIONAL_COLOR}[item["class"]])
        if column == 0:
            ax_xy.text(-0.08, 0.5, "XY", transform=ax_xy.transAxes,
                       ha="right", va="center", fontsize=7, fontweight="bold")
            ax_xz.text(-0.08, 0.5, "XZ", transform=ax_xz.transAxes,
                       ha="right", va="center", fontsize=7, fontweight="bold")

    ax_bar = fig.add_subplot(grid[3, :])
    x = np.arange(len(selected))
    width = 0.36
    ax_bar.bar(x - width / 2, selected["global_concordance"], width,
               color="#718096", label="global correlation")
    bar_colors = [CORE_COLOR if value == "core" else REGIONAL_COLOR
                  for value in selected["plot_class"]]
    ax_bar.bar(x + width / 2, selected["max_regional_concordance"], width,
               color=bar_colors, label="best-domain correlation")
    ax_bar.axhline(0, color="#777777", lw=0.6)
    ax_bar.set_xticks(x, selected["label"])
    ax_bar.set_ylabel("Pearson r")
    ax_bar.set_ylim(min(-0.05, float(selected[["global_concordance",
                    "max_regional_concordance"]].min().min()) - 0.05), 1.0)
    ax_bar.legend(frameon=False, ncol=2, loc="upper right", fontsize=6.2)
    ax_bar.grid(axis="y", color="#E6E9ED", lw=0.5)

    fig.suptitle(f"Rank {int(rank_row.priority_rank)} · {gene}: core and regional enhancer architecture",
                 x=0.055, ha="left", fontsize=10, fontweight="bold")
    fig.text(0.985, 0.02,
             "RNA and ATAC are independently log1p/p99-scaled for spatial shape comparison; "
             "regional q: 199-shift 3D spatial null, within-gene BH.",
             ha="right", va="bottom", fontsize=5.8, color="#59636D")
    stem = output / f"rank{int(rank_row.priority_rank):02d}_{gene}_core_regional"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    selected.to_csv(stem.with_suffix(".source.tsv"), sep="\t", index=False)
    return {"gene": gene, "priority_rank": int(rank_row.priority_rank),
            "n_core_shown": len(core), "n_regional_shown": len(regional),
            "png": str(stem.with_suffix('.png')), "pdf": str(stem.with_suffix('.pdf'))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, default=Path(
        "results/zebrafish/mixed_complexity_confirm_top50_v1/threshold_sensitivity/final_priority_ranking.tsv"))
    parser.add_argument("--evidence", type=Path, default=Path(
        "results/zebrafish/mixed_complexity_confirm_top50_v1/enhancer_evidence_confirmed.tsv"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/zebrafish/complexity_gene_enhancer_figures_v1"))
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--rank-start", type=int, default=1,
                        help="First one-based rank to plot (inclusive).")
    parser.add_argument("--rank-column", default="priority_rank",
                        help="Column containing the one-based rank.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(args.ranking, sep="\t")
    if args.rank_column not in ranking.columns:
        raise ValueError(f"Rank column not found: {args.rank_column}")
    rank_end = args.rank_start + args.top - 1
    ranking = ranking[
        ranking[args.rank_column].between(args.rank_start, rank_end)
    ].sort_values(args.rank_column).copy()
    ranking["priority_rank"] = ranking[args.rank_column].astype(int)
    evidence = pd.read_csv(args.evidence, sep="\t")
    records = _gtf_gene_records(GTF_PATH)
    outputs = []
    with h5py.File(RNA_PATH, "r") as rna, h5py.File(ATAC_PATH, "r") as atac:
        gene_names = _decode(rna["var"]["_index"][:])
        peak_names = _decode(atac["var"]["_index"][:])
        gene_index = {name: index for index, name in enumerate(gene_names)}
        peak_index = {name: index for index, name in enumerate(peak_names)}
        coords = np.asarray(atac["obsm"]["spatial_rescaled_z"][:], dtype=float)
        for row in ranking.itertuples(index=False):
            result = _plot_gene(row.gene, row, evidence, rna, atac, gene_index,
                                peak_index, coords, records, args.output)
            if result is not None:
                outputs.append(result)
                print("plotted", row.gene, flush=True)
    pd.DataFrame(outputs).to_csv(args.output / "figure_manifest.tsv", sep="\t", index=False)
    (args.output / "figure_contract.json").write_text(json.dumps({
        "claim": "high-complexity genes combine globally concordant core enhancers with spatially restricted regional enhancers",
        "archetype": "image plate + quant", "formats": ["png", "pdf"],
        "core_threshold": 0.35, "regional_q_threshold": 0.10,
        "spatial_views": ["XY", "XZ"], "normalization": "per-feature log1p and positive p99 clipping",
    }, indent=2))


if __name__ == "__main__":
    main()
