from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .complexity import gene_complexity_summary
from .run_gw12_genomewide_complexity import RNA_PATH, _load_rna_csc


GREY = "#C6CCD2"
MIXED = "#D9A0AF"
SELECTED = "#A83E5C"
ACCENT = "#D98B2B"
SLATE = "#667C91"
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


def _summaries(evidence, threshold):
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
        result = gene_complexity_summary(group, gene=gene).iloc[0].to_dict()
        result["core_threshold"] = threshold
        eligible = result["n_core"] >= 1 and result["n_regional_candidates"] >= 1
        result["mixed_score"] = (
            result["complexity_score"] * np.log1p(1 + result["n_core"])
            if eligible else 0.0
        )
        rows.append(result)
    return pd.DataFrame(rows)


def _expression_audit(genes):
    rna = ad.read_h5ad(RNA_PATH, backed="r")
    matrix = _load_rna_csc(RNA_PATH)
    means = np.asarray(matrix.mean(axis=0)).ravel()
    detected = np.diff(matrix.indptr) / matrix.shape[0]
    lookup = pd.DataFrame({"gene": rna.var_names.astype(str), "rna_mean": means,
                           "rna_detected_fraction": detected})
    rna.file.close()
    return pd.DataFrame({"gene": genes}).merge(lookup, on="gene", how="left")


def _elbow(scores):
    positive = np.asarray(scores, dtype=float)
    positive = positive[positive > 0]
    if len(positive) < 3:
        return len(positive)
    transformed = np.log10(positive + 1e-6)
    x = np.linspace(0, 1, len(transformed))
    y = (transformed - transformed[-1]) / max(transformed[0] - transformed[-1], 1e-12)
    return int(np.argmax((1 - x) - y) + 1)


def build_source(screen, confirmed):
    keys = ["gene", "enhancer"]
    qvalues = confirmed[keys + ["permutation_pvalue", "permutation_qvalue"]].copy()
    evidence = screen.drop(columns=["permutation_pvalue", "permutation_qvalue"],
                           errors="ignore").merge(qvalues, on=keys, how="left")
    evidence["permutation_pvalue"] = evidence["permutation_pvalue"].fillna(1.0)
    evidence["permutation_qvalue"] = evidence["permutation_qvalue"].fillna(1.0)
    thresholds = [0.20, 0.25, 0.30, 0.35]
    long = pd.concat([_summaries(evidence, value) for value in thresholds], ignore_index=True)
    pivot = long.pivot(index="gene", columns="core_threshold", values="mixed_score").fillna(0)
    source = pd.DataFrame({"gene": pivot.index, "srci": pivot.mean(axis=1),
                           "threshold_stability": (pivot > 0).sum(axis=1)}).reset_index(drop=True)
    strict = long[long["core_threshold"] == 0.35].rename(columns={
        "n_core": "n_core_strict", "n_regional_candidates": "n_regional_strict",
        "n_supported_domains": "n_regional_domains_strict",
        "regional_evidence_topk_sum": "regional_evidence_strict",
        "domain_diversity": "domain_diversity_strict",
    })
    keep = ["gene", "n_enhancers", "n_core_strict", "n_regional_strict",
            "n_regional_domains_strict", "regional_evidence_strict",
            "domain_diversity_strict"]
    source = source.merge(strict[keep], on="gene", how="left")
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
        (source["n_core_strict"] >= 1) & (source["n_regional_strict"] >= 2)
        & (source["threshold_stability"] >= 3)
        & (source["rna_detected_fraction"] >= 0.10)
    )
    source["threshold_stable_mixed"] = (
        (source["threshold_stability"] >= 2) & (source["rna_detected_fraction"] >= 0.10)
    )
    elbow = _elbow(source["srci"])
    source["above_elbow"] = source["srci_rank"] <= elbow
    return source, evidence, elbow


def plot(source, elbow, output):
    fig = plt.figure(figsize=(7.2, 7.0), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[1.03, 1.17],
                           left=0.09, right=0.97, top=0.89, bottom=0.09,
                           hspace=0.38, wspace=0.46)
    ax = fig.add_subplot(grid[0, :])
    x = source["srci_rank"].to_numpy()
    raw_y = np.log10(source["srci"].to_numpy() + 1e-6)
    positive = source["srci"].to_numpy() > 0
    floor = float(raw_y[positive].min() - 0.28) if positive.any() else -1
    y = np.where(positive, raw_y, floor)
    colors = np.where(source["threshold_stable_mixed"], MIXED, GREY).astype(object)
    colors[source["above_elbow"] & source["confirmed_mixed"]] = SELECTED
    colors[source["above_elbow"] & ~source["confirmed_mixed"]] = SLATE
    ax.scatter(x, y, s=7, c=colors, edgecolors="white", linewidths=0.15, zorder=3)
    ax.plot(x, y, color="#9AA2A9", lw=0.5, zorder=1)
    cutoff = float(source.loc[source["srci_rank"] == elbow, "srci"].iloc[0])
    cutoff_y = np.log10(cutoff + 1e-6)
    ax.axvline(elbow, color=SELECTED, ls="--", lw=0.8)
    ax.axhline(cutoff_y, color=SELECTED, ls=":", lw=0.6)
    ax.text(elbow + max(3, len(source) * 0.005), raw_y.max() - 0.1,
            f"elbow = rank {elbow}", color=SELECTED, fontsize=6.5)
    label_rows = source[source["srci"] > 0].head(min(8, int(positive.sum())))
    label_x = np.linspace(50, min(5000, len(source) * 0.34), len(label_rows))
    for position, row in enumerate(label_rows.itertuples(index=False)):
        color = SELECTED if row.confirmed_mixed else SLATE
        ax.annotate(row.gene, (row.srci_rank, np.log10(row.srci + 1e-6)),
                    xytext=(label_x[position], raw_y.max() + 0.10), textcoords="data",
                    ha="center", fontsize=5.8, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=color))
    ax.set_xlabel("Genes ranked by spatial regulatory complexity")
    ax.set_ylabel(r"log$_{10}$(SRCI + 10$^{-6}$)")
    ax.set_xlim(-100, len(source) + 150)
    ax.set_ylim(floor - 0.1, raw_y.max() + 0.3)
    ax.grid(axis="y", color="#E7EAED", lw=0.5)
    ax.set_title("a   Genome-wide spatial regulatory complexity ranking",
                 loc="left", fontsize=8.5, fontweight="bold")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY,
               markeredgecolor="none", label="other genes"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MIXED,
               markeredgecolor="none", label="threshold-stable mixed"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SELECTED,
               markeredgecolor="none", label="strict mixed above elbow"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SLATE,
               markeredgecolor="none", label="threshold-dependent above elbow"),
    ]
    ax.legend(handles=legend, ncol=2, loc="lower left", fontsize=5.7,
              handletextpad=0.3, columnspacing=0.8)

    arch = fig.add_subplot(grid[1, 0])
    sizes = 8 + 12 * source["n_regional_domains_strict"].to_numpy()
    score_log = np.log10(source["srci"].to_numpy() + 1e-6)
    scatter = arch.scatter(source["n_core_strict"], source["n_regional_strict"],
                           c=score_log, s=sizes, cmap="magma", alpha=0.75,
                           edgecolors="white", linewidths=0.25)
    for index, row in source[source["srci"] > 0].head(8).iterrows():
        arch.annotate(row["gene"], (row["n_core_strict"], row["n_regional_strict"]),
                      xytext=(3, 3 if index % 2 == 0 else -9), textcoords="offset points",
                      fontsize=5.5, color="#3B4147")
    arch.set_xlabel("Strict core enhancers (r ≥ 0.35)")
    arch.set_ylabel("Confirmed regional enhancers")
    arch.set_title("b   Core–regional architecture", loc="left",
                   fontsize=8.5, fontweight="bold")
    arch.grid(color="#E7EAED", lw=0.45)
    cax = inset_axes(arch, width="38%", height="4%", loc="upper right", borderpad=0.7)
    cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.set_label(r"log$_{10}$(SRCI + 10$^{-6}$)", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)

    heat = fig.add_subplot(grid[1, 1])
    matrix = np.vstack([
        source["n_core_strict"] / source["n_enhancers"].clip(lower=1),
        source["n_regional_strict"] / source["n_enhancers"].clip(lower=1),
        source["n_regional_domains_strict"] / 6.0,
        source["threshold_stability"] / 4.0,
        source["n_q01_enhancers"] / source["n_enhancers"].clip(lower=1),
        source["rna_detected_fraction"],
    ])
    cmap = LinearSegmentedColormap.from_list("composition", ["#F4F5F6", "#BBCDD0", TEAL, "#123F43"])
    heat.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    heat.axvline(elbow - 0.5, color=SELECTED, ls="--", lw=0.8)
    heat.set_yticks(np.arange(6), ["core fraction", "regional fraction", "domain coverage",
                    "threshold stability", "q≤0.1 fraction", "RNA detection"])
    ticks = [0, max(elbow - 1, 0), 999, 4999, 9999, len(source) - 1]
    heat.set_xticks(ticks, [1, elbow, 1000, 5000, 10000, len(source)])
    heat.set_xlabel("SRCI rank")
    heat.set_title("c   Regulatory-complexity composition", loc="left",
                   fontsize=8.5, fontweight="bold")
    for spine in heat.spines.values():
        spine.set_visible(False)

    fig.suptitle("GW12 spatial regulatory complexity landscape", x=0.09,
                 ha="left", fontsize=11, fontweight="bold")
    fig.text(0.09, 0.025,
             "SRCI: mean mixed-regulation score across core thresholds 0.20/0.25/0.30/0.35. "
             "All potential non-zero candidates were evaluated using 99 two-dimensional spatial shifts; "
             "regional enhancers require local r≥0.20, evidence≥0.005 and within-gene BH q≤0.10.",
             ha="left", va="bottom", fontsize=5.7, color="#586169")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, default=Path(
        "results/GW12/genomewide_complexity_screen_v1/enhancer_evidence_screen.tsv"))
    parser.add_argument("--confirmed", type=Path, default=Path(
        "results/GW12/mixed_complexity_confirm_candidates_v1/enhancer_evidence.tsv"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/GW12/genomewide_srci_v1/gw12_genomewide_srci"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    screen = pd.read_csv(args.screen, sep="\t")
    confirmed = pd.read_csv(args.confirmed, sep="\t")
    source, evidence, elbow = build_source(screen, confirmed)
    source.to_csv(args.output.parent / "gw12_genomewide_srci_source.tsv", sep="\t", index=False)
    evidence.to_csv(args.output.parent / "gw12_enhancer_evidence_with_confirmation.tsv", sep="\t", index=False)
    plot(source, elbow, args.output)
    positive = int((source["srci"] > 0).sum())
    metadata = {
        "n_genes": int(len(source)), "n_positive_srci": positive,
        "elbow_rank": int(elbow),
        "elbow_gene": source.loc[source["srci_rank"] == elbow, "gene"].iloc[0],
        "n_confirmed_mixed_strict": int(source["confirmed_mixed"].sum()),
        "n_genes_with_permutation": int(confirmed["gene"].nunique()),
        "permutations": 99,
        "confirmation_scope": "all genes capable of non-zero SRCI in the genome-wide screen",
    }
    (args.output.parent / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(source.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
