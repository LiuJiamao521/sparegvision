from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .complexity import gene_complexity_summary
from .run_zebrafish_genomewide_complexity import RNA_PATH, _decode


GREY = "#C6CCD2"
MIXED = "#D9A0AF"
SELECTED = "#A83E5C"
HDLBPA = "#D98B2B"
TEAL = "#248A83"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def _summaries_at_threshold(evidence, threshold):
    frame = evidence.copy()
    core = frame["global_concordance"] >= threshold
    regional = (
        (~core)
        & (frame["max_regional_concordance"] >= 0.20)
        & (frame["regional_evidence_score"] >= 0.005)
        & (frame["permutation_qvalue"] <= 0.10)
    )
    frame["predicted_class"] = np.where(
        core, "core_concordant", np.where(regional, "regional_candidate", "unsupported")
    )
    rows = []
    for gene, group in frame.groupby("gene", sort=False):
        summary = gene_complexity_summary(group, gene=gene).iloc[0].to_dict()
        summary["core_threshold"] = threshold
        eligible = summary["n_core"] >= 1 and summary["n_regional_candidates"] >= 1
        summary["mixed_eligible"] = eligible
        summary["mixed_score"] = (
            summary["complexity_score"] * np.log1p(1 + summary["n_core"])
            if eligible else 0.0
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def _expression_audit(genes):
    with h5py.File(RNA_PATH, "r") as rna:
        names = _decode(rna["var"]["_index"][:])
        index = {name: i for i, name in enumerate(names)}
        rows = []
        for gene in genes:
            values = np.asarray(rna["X"][:, index[gene]], dtype=float)
            rows.append({
                "gene": gene,
                "rna_mean": float(values.mean()),
                "rna_detected_fraction": float((values > 0).mean()),
            })
    return pd.DataFrame(rows)


def _elbow_rank(scores):
    positive = np.asarray(scores, dtype=float)
    positive = positive[positive > 0]
    if len(positive) < 3:
        return len(positive)
    transformed = np.log10(positive + 1e-6)
    x = np.linspace(0, 1, len(transformed))
    y = (transformed - transformed[-1]) / max(transformed[0] - transformed[-1], 1e-12)
    chord = 1 - x
    distance = chord - y
    return int(np.argmax(distance) + 1)


def _build_source(evidence):
    thresholds = [0.20, 0.25, 0.30, 0.35]
    threshold_tables = [_summaries_at_threshold(evidence, threshold) for threshold in thresholds]
    long = pd.concat(threshold_tables, ignore_index=True)
    score = long.pivot(index="gene", columns="core_threshold", values="mixed_score").fillna(0)
    source = pd.DataFrame({
        "gene": score.index,
        "srci": score.mean(axis=1),
        "threshold_stability": (score > 0).sum(axis=1),
    }).reset_index(drop=True)
    canonical = long[long["core_threshold"] == 0.35].copy()
    canonical = canonical.rename(columns={
        "n_core": "n_core_strict",
        "n_regional_candidates": "n_regional_strict",
        "n_supported_domains": "n_regional_domains_strict",
        "regional_evidence_topk_sum": "regional_evidence_strict",
        "domain_diversity": "domain_diversity_strict",
    })
    keep = ["gene", "n_enhancers", "n_core_strict", "n_regional_strict",
            "n_regional_domains_strict", "regional_evidence_strict",
            "domain_diversity_strict"]
    source = source.merge(canonical[keep], on="gene", how="left")
    qsummary = evidence.groupby("gene").agg(
        n_q01_enhancers=("permutation_qvalue", lambda values: int((values <= 0.10).sum())),
        min_enhancer_q=("permutation_qvalue", "min"),
    ).reset_index()
    source = source.merge(qsummary, on="gene", how="left")
    source = source.merge(_expression_audit(source["gene"]), on="gene", how="left")
    source = source.sort_values(["srci", "n_regional_strict", "n_core_strict"],
                                ascending=False).reset_index(drop=True)
    source["srci_rank"] = np.arange(1, len(source) + 1)
    source["confirmed_mixed"] = (
        (source["n_core_strict"] >= 1)
        & (source["n_regional_strict"] >= 2)
        & (source["threshold_stability"] >= 3)
        & (source["rna_detected_fraction"] >= 0.10)
    )
    source["threshold_stable_mixed"] = (
        (source["threshold_stability"] >= 2)
        & (source["rna_detected_fraction"] >= 0.10)
    )
    elbow = _elbow_rank(source["srci"].to_numpy())
    source["above_elbow"] = source["srci_rank"] <= elbow
    return source, elbow


def _label_rank_curve(ax, row, color, dx=5, dy=0.12):
    y = np.log10(row.srci + 1e-6)
    ax.annotate(row.gene, (row.srci_rank, y), xytext=(dx, dy),
                textcoords="offset points", fontsize=6.0, color=color,
                fontweight="bold", arrowprops=dict(arrowstyle="-", lw=0.45, color=color))


def _plot(source, elbow, output):
    fig = plt.figure(figsize=(7.2, 7.0), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[1.03, 1.17],
                           left=0.09, right=0.97, top=0.89, bottom=0.09,
                           hspace=0.38, wspace=0.46)
    ax_rank = fig.add_subplot(grid[0, :])
    x = source["srci_rank"].to_numpy()
    raw_y = np.log10(source["srci"].to_numpy() + 1e-6)
    positive_floor = float(raw_y[source["srci"].to_numpy() > 0].min() - 0.28)
    y = np.where(source["srci"].to_numpy() > 0, raw_y, positive_floor)
    colors = np.where(source["threshold_stable_mixed"], MIXED, GREY).astype(object)
    colors[source["above_elbow"] & source["confirmed_mixed"]] = SELECTED
    colors[source["above_elbow"] & ~source["confirmed_mixed"]] = "#667C91"
    hmask = source["gene"].str.lower() == "hdlbpa"
    colors[hmask] = HDLBPA
    ax_rank.scatter(x, y, s=11, c=colors, edgecolors="white", linewidths=0.25, zorder=3)
    ax_rank.plot(x, y, color="#9AA2A9", lw=0.55, zorder=1)
    cutoff_score = float(source.loc[source["srci_rank"] == elbow, "srci"].iloc[0])
    cutoff_y = np.log10(cutoff_score + 1e-6)
    ax_rank.axvline(elbow, color=SELECTED, lw=0.8, ls="--")
    ax_rank.axhline(cutoff_y, color=SELECTED, lw=0.6, ls=":")
    ax_rank.text(elbow + 5, y.max() - 0.12, f"elbow = rank {elbow}",
                 color=SELECTED, fontsize=6.5, va="top")
    labels = list(source.nsmallest(5, "srci_rank")["gene"])
    label_x = [5, 35, 70, 115, 160]
    for i, gene in enumerate(labels):
        row = source[source["gene"].str.lower() == gene.lower()].iloc[0]
        color = SELECTED if row.confirmed_mixed else "#667C91"
        ax_rank.annotate(row.gene, (row.srci_rank, np.log10(row.srci + 1e-6)),
                         xytext=(label_x[i], raw_y.max() + 0.10), textcoords="data",
                         ha="center", fontsize=6.0, color=color, fontweight="bold",
                         arrowprops=dict(arrowstyle="-", lw=0.45, color=color))
    hrow = source[hmask].iloc[0]
    _label_rank_curve(ax_rank, hrow, HDLBPA, dx=5, dy=8)
    ax_rank.set_xlabel("Genes ranked by spatial regulatory complexity")
    ax_rank.set_ylabel(r"log$_{10}$(SRCI + 10$^{-6}$)")
    ax_rank.set_xlim(-5, len(source) + 7)
    ax_rank.set_ylim(positive_floor - 0.10, raw_y.max() + 0.30)
    ax_rank.grid(axis="y", color="#E7EAED", lw=0.5)
    ax_rank.set_title("a   Genome-wide spatial regulatory complexity ranking",
                      loc="left", fontsize=8.5, fontweight="bold")

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY, markeredgecolor="none", label="other genes"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MIXED, markeredgecolor="none", label="threshold-stable mixed"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SELECTED, markeredgecolor="none", label="strict mixed above elbow"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#667C91", markeredgecolor="none", label="threshold-dependent above elbow"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HDLBPA, markeredgecolor="none", label="hdlbpa"),
    ]
    ax_rank.legend(handles=legend, ncol=3, loc="lower left", fontsize=5.7, handletextpad=0.3, columnspacing=0.8)

    ax_arch = fig.add_subplot(grid[1, 0])
    sizes = 10 + 12 * source["n_regional_domains_strict"].to_numpy()
    score_log = np.log10(source["srci"].to_numpy() + 1e-6)
    scatter = ax_arch.scatter(source["n_core_strict"], source["n_regional_strict"],
                              c=score_log, s=sizes, cmap="magma", alpha=0.78,
                              edgecolors="white", linewidths=0.35)
    for label_index, gene in enumerate(list(source.head(5)["gene"]) + ["hdlbpa"]):
        row = source[source["gene"].str.lower() == gene.lower()].iloc[0]
        ax_arch.annotate(row.gene, (row.n_core_strict, row.n_regional_strict),
                         xytext=(3, 3 if gene == "hdlbpa" or label_index % 2 == 0 else -9), textcoords="offset points", fontsize=5.7,
                         color=HDLBPA if gene.lower() == "hdlbpa" else "#3B4147")
    ax_arch.set_xlabel("Strict core enhancers (r ≥ 0.35)")
    ax_arch.set_ylabel("Confirmed regional enhancers")
    ax_arch.set_title("b   Core–regional architecture", loc="left",
                      fontsize=8.5, fontweight="bold")
    ax_arch.grid(color="#E7EAED", lw=0.45)
    cax = inset_axes(ax_arch, width="38%", height="4%", loc="upper right", borderpad=0.7)
    cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.set_label(r"log$_{10}$(SRCI + 10$^{-6}$)", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)

    ax_heat = fig.add_subplot(grid[1, 1])
    confirmed_fraction = source["n_q01_enhancers"] / source["n_enhancers"].clip(lower=1)
    matrix = np.vstack([
        source["n_core_strict"] / source["n_enhancers"].clip(lower=1),
        source["n_regional_strict"] / source["n_enhancers"].clip(lower=1),
        source["n_regional_domains_strict"] / 6.0,
        source["threshold_stability"] / 4.0,
        confirmed_fraction,
        source["rna_detected_fraction"],
    ])
    cmap = LinearSegmentedColormap.from_list("composition", ["#F4F5F6", "#BBCDD0", TEAL, "#123F43"])
    ax_heat.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax_heat.axvline(elbow - 0.5, color=SELECTED, lw=0.8, ls="--")
    ax_heat.set_yticks(np.arange(6), ["core fraction", "regional fraction",
                    "domain coverage", "threshold stability", "q≤0.1 fraction",
                    "RNA detection"])
    ax_heat.set_xticks([0, elbow - 1, 99, 199, 299, 399, len(source) - 1],
                       [1, elbow, 100, 200, 300, 400, len(source)])
    ax_heat.set_xlabel("SRCI rank")
    ax_heat.set_title("c   Regulatory-complexity composition", loc="left",
                      fontsize=8.5, fontweight="bold")
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    fig.suptitle("Zebrafish 6-somite spatial regulatory complexity landscape",
                 x=0.09, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.09, 0.025,
             "SRCI: mean mixed-regulation score across core thresholds 0.20/0.25/0.30/0.35; "
             "regional enhancers require local r≥0.20, evidence≥0.005 and within-gene BH q≤0.10 "
             "from 199 three-dimensional spatial shifts. Point size in b denotes spatial-domain count.",
             ha="left", va="bottom", fontsize=5.7, color="#586169")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=Path(
        "results/zebrafish/genomewide_complexity_confirm_all_v1/enhancer_evidence_confirmed.tsv"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/zebrafish/genomewide_srci_v1/zebrafish_genomewide_srci"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(args.evidence, sep="\t")
    source, elbow = _build_source(evidence)
    source.to_csv(args.output.parent / "zebrafish_genomewide_srci_source.tsv", sep="\t", index=False)
    _plot(source, elbow, args.output)
    metadata = {
        "n_genes": int(len(source)), "elbow_rank": int(elbow),
        "elbow_gene": source.loc[source["srci_rank"] == elbow, "gene"].iloc[0],
        "n_confirmed_mixed": int(source["confirmed_mixed"].sum()),
        "n_above_elbow": int(source["above_elbow"].sum()),
        "hdlbpa_rank": int(source.loc[source["gene"].str.lower() == "hdlbpa", "srci_rank"].iloc[0]),
        "figure_contract": "asymmetric quantitative grid; ranked hero panel plus architecture and composition",
    }
    (args.output.parent / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(source.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
