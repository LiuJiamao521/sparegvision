import os
from pathlib import Path
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
ATAC_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad"
LINK_PATH = ROOT / "data" / "region_to_gene_adj.tsv"
ATAC_CACHE = Path("/tmp/rectified_h_atac_log1p_float32_dense_test.memmap")
EPS = 1e-3

RNA_CMAP = LinearSegmentedColormap.from_list(
    "rna_shape", ["#F7F7F7", "#DCEAF7", "#6BAED6", "#2171B5", "#08306B"], N=256
)
ATAC_CMAP = LinearSegmentedColormap.from_list(
    "atac_shape", ["#EDEDED", "#FFF7F3", "#FDE0DD", "#FCC5C0", "#FA9FB5", "#F768A1", "#DD3497", "#AE017E"], N=256
)
LOGFC_CMAP = LinearSegmentedColormap.from_list(
    "logfc_div", ["#2166AC", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#B2182B"], N=256
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams.update({
    "font.size": 6.2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def dense_to_grid(obs, values):
    grid = np.full((76, 101), np.nan, dtype=float)
    rows = obs["grid_row"].to_numpy(dtype=int)
    cols = obs["grid_col"].to_numpy(dtype=int)
    grid[rows, cols] = np.asarray(values, dtype=float)
    return grid


def normalize01(x):
    x = np.asarray(x, dtype=float)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    return (x - xmin) / max(xmax - xmin, 1e-8)


def fit_two_state_mask(gene_norm, n_iter=25):
    vals = gene_norm.ravel()
    c0, c1 = np.quantile(vals, [0.25, 0.75])
    for _ in range(n_iter):
        d0 = np.abs(vals - c0)
        d1 = np.abs(vals - c1)
        z = d1 < d0
        if np.any(~z):
            c0 = vals[~z].mean()
        if np.any(z):
            c1 = vals[z].mean()
    active_label = 1 if c1 > c0 else 0
    mask = z.reshape(gene_norm.shape) if active_label == 1 else (~z).reshape(gene_norm.shape)
    return mask.astype(np.uint8), float(min(c0, c1)), float(max(c0, c1))


def majority_regularize(mask, n_rounds=3, threshold=5):
    out = mask.copy().astype(np.uint8)
    for _ in range(n_rounds):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        votes = np.zeros_like(out, dtype=int)
        for dy in range(3):
            for dx in range(3):
                votes += padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
        out = (votes >= threshold).astype(np.uint8)
    return out


def per_panel_p99_normalize(grid, percentile=99.0):
    arr = np.asarray(grid, dtype=float).copy()
    valid = np.isfinite(arr)
    pos = arr[valid & (arr > 0)]
    if pos.size == 0:
        out = np.zeros_like(arr, dtype=float)
        out[~valid] = np.nan
        return out, 0.0
    p = float(np.percentile(pos, percentile))
    p = max(p, 1e-8)
    arr[valid] = np.clip(arr[valid], 0, p) / p
    return arr, p


def corr_valid(a, b):
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(aa) & np.isfinite(bb)
    if m.sum() < 3:
        return np.nan
    aa = aa[m]
    bb = bb[m]
    if np.std(aa) < 1e-8 or np.std(bb) < 1e-8:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def add_panel(ax, grid, title, cmap, norm=None, vmin=None, vmax=None, label=None, title_fontsize=6.0):
    ax.imshow(grid, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper", aspect="auto")
    ax.set_title(title, loc="left", fontsize=title_fontsize, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label:
        ax.text(-0.06, 1.04, label, transform=ax.transAxes, fontsize=7.8, fontweight="bold")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--initial-rho-threshold", type=float, default=0.0)
    ap.add_argument("--rectified-raw-rho-threshold", type=float, default=0.1)
    return ap.parse_args()


def main():
    args = parse_args()
    gene = args.gene.upper()
    prefix = ROOT / "plot" / f"{gene}_region_to_gene_adj_rectified_h"
    out_shape = prefix.with_name(prefix.name + "_filtered_gene_enhancer_p99.png")
    out_logfc = prefix.with_name(prefix.name + "_filtered_logfc_vs_control.png")
    out_tsv = prefix.with_name(prefix.name + "_filtered_analysis.tsv")

    rna = ad.read_h5ad(RNA_PATH, backed="r")
    atac = ad.read_h5ad(ATAC_PATH, backed="r")
    links = pd.read_csv(LINK_PATH, sep="\t")

    sub = (
        links.loc[links["target"].astype(str) == gene]
        .drop_duplicates(subset=["region"])
        .sort_values("importance_x_abs_rho", ascending=False)
        .reset_index(drop=True)
        .copy()
    )
    if sub.empty:
        raise RuntimeError(f"{gene} absent in region_to_gene_adj.tsv")
    sub["panel_id"] = [f"E{i+1}" for i in range(len(sub))]

    expr_idx = pd.Index(rna.var_names.astype(str)).get_indexer([gene])
    if expr_idx[0] < 0:
        raise RuntimeError(f"{gene} absent in rectified_h RNA")
    expr = rna.X[:, expr_idx[0]]
    if sparse.issparse(expr):
        expr = expr.toarray().reshape(-1)
    else:
        expr = np.asarray(expr).reshape(-1)
    expr = np.log1p(np.maximum(expr, 0))
    expr_grid = dense_to_grid(rna.obs, expr)

    gene_norm0 = normalize01(expr_grid)
    raw_mask, c_low, c_high = fit_two_state_mask(gene_norm0, n_iter=25)
    latent_mask = majority_regularize(raw_mask, n_rounds=3, threshold=5)
    control_grid = expr_grid * latent_mask.astype(float)
    control_norm, control_p99 = per_panel_p99_normalize(control_grid, percentile=99.0)

    if not ATAC_CACHE.exists():
        raise RuntimeError(f"ATAC cache missing: {ATAC_CACHE}")
    atac_mm = np.memmap(ATAC_CACHE, dtype=np.float32, mode="r", shape=(atac.n_obs, atac.n_vars))
    atac_var_index = pd.Index(atac.var_names.astype(str))
    enhancer_idx = atac_var_index.get_indexer(sub["region"].astype(str))
    keep_feature = enhancer_idx >= 0
    sub = sub.loc[keep_feature].reset_index(drop=True).copy()
    enhancer_idx = enhancer_idx[keep_feature]
    enhancer_vals = np.asarray(atac_mm[:, enhancer_idx], dtype=np.float32)

    rows = []
    shape_grids = []
    logfc_grids = []
    row_idx = atac.obs["grid_row"].to_numpy(dtype=int)
    col_idx = atac.obs["grid_col"].to_numpy(dtype=int)
    control_spot = control_norm[row_idx, col_idx]
    valid_template = np.isfinite(control_norm) & (latent_mask > 0)
    valid_spot = valid_template[row_idx, col_idx]

    for i, (_, link_row) in enumerate(sub.iterrows()):
        raw_grid = dense_to_grid(atac.obs, enhancer_vals[:, i])
        rectified_raw_rho = corr_valid(raw_grid, expr_grid)
        if float(link_row["rho"]) < args.initial_rho_threshold:
            continue
        if not np.isfinite(rectified_raw_rho) or rectified_raw_rho < args.rectified_raw_rho_threshold:
            continue

        latent_grid = raw_grid * latent_mask.astype(float)
        enh_norm, enh_p99 = per_panel_p99_normalize(latent_grid, percentile=99.0)
        shape_grids.append(enh_norm)

        lg = np.full_like(enh_norm, np.nan, dtype=float)
        valid = valid_template & np.isfinite(enh_norm)
        lg[valid] = np.log2((enh_norm[valid] + EPS) / (control_norm[valid] + EPS))
        logfc_grids.append(lg)

        mean_logfc = float(np.nanmean(lg[valid])) if valid.sum() else np.nan
        latent_rho = corr_valid(latent_grid, expr_grid)
        rows.append({
            "gene": gene,
            "panel_id": link_row["panel_id"],
            "enhancer": link_row["region"],
            "importance": float(link_row["importance"]),
            "initial_rho": float(link_row["rho"]),
            "rectified_raw_rho": rectified_raw_rho,
            "latent_state_rho": latent_rho,
            "importance_x_abs_rho": float(link_row["importance_x_abs_rho"]),
            "distance": link_row["Distance"],
            "control_p99": control_p99,
            "enhancer_p99": enh_p99,
            "mean_logfc_active": mean_logfc,
        })

    if not rows:
        raise RuntimeError(f"{gene}: no enhancer passed filters")

    out = pd.DataFrame(rows)
    out.to_csv(out_tsv, sep="\t", index=False)

    expr_norm, expr_p99 = per_panel_p99_normalize(control_grid, percentile=99.0)
    n_panels = 1 + len(rows)
    ncols = 8
    nrows = int(np.ceil(n_panels / ncols))

    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(14.4, 1.72 * nrows), constrained_layout=False)
    axes1 = np.atleast_1d(axes1).ravel()
    add_panel(axes1[0], expr_norm, f"{gene} RNA\np99 norm", RNA_CMAP, vmin=0, vmax=1, label="A", title_fontsize=7.0)
    for i, row in out.iterrows():
        title = f"{row['panel_id']}\nrawρ={row['initial_rho']:.2f}\nlatentρ={row['latent_state_rho']:.2f}"
        add_panel(axes1[i + 1], shape_grids[i], title, ATAC_CMAP, vmin=0, vmax=1, title_fontsize=6.5)
    for ax in axes1[n_panels:]:
        ax.axis("off")
    fig1.subplots_adjust(left=0.03, right=0.935, top=0.92, bottom=0.07, wspace=0.08, hspace=0.24)
    cax_rna = fig1.add_axes([0.942, 0.72, 0.012, 0.16])
    cax_atac = fig1.add_axes([0.942, 0.34, 0.012, 0.30])
    cb_rna = mpl.colorbar.ColorbarBase(cax_rna, cmap=RNA_CMAP, norm=Normalize(0, 1.0), orientation="vertical")
    cb_rna.ax.tick_params(labelsize=5.8, length=2)
    cb_rna.set_label("RNA control\np99 norm", fontsize=6.0)
    cb_atac = mpl.colorbar.ColorbarBase(cax_atac, cmap=ATAC_CMAP, norm=Normalize(0, 1.0), orientation="vertical")
    cb_atac.ax.tick_params(labelsize=5.8, length=2)
    cb_atac.set_label("latent-state ATAC\nper-enhancer p99", fontsize=6.0)
    fig1.suptitle(
        f"Rectified-H {gene} filtered enhancer maps (gene+enhancer p99 normalized)",
        x=0.03, ha="left", fontsize=11, fontweight="bold",
    )
    fig1.text(
        0.03, 0.022,
        f"Original links from region_to_gene_adj.tsv. Unified filters: initial rho >= {args.initial_rho_threshold:.2f}, rectified raw rho >= {args.rectified_raw_rho_threshold:.2f}. "
        f"Retained n={len(out)} of {len(sub)} linked peaks. Gene and enhancer maps were independently clipped at their own 99th percentile and rescaled to 0–1.",
        fontsize=5.8, color="#444444",
    )
    fig1.savefig(out_shape, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    vals = np.concatenate([g[np.isfinite(g)] for g in logfc_grids if np.isfinite(g).any()])
    vmax = float(np.percentile(np.abs(vals), 97.5)) if vals.size else 2.0
    vmax = max(vmax, 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(14.4, 1.72 * nrows), constrained_layout=False)
    axes2 = np.atleast_1d(axes2).ravel()
    add_panel(axes2[0], expr_norm, f"{gene} control\nmask × RNA p99 norm", RNA_CMAP, vmin=0, vmax=1, label="A", title_fontsize=6.8)
    for i, row in out.iterrows():
        title = f"{row['panel_id']}\nmean logFC={row['mean_logfc_active']:.2f}"
        add_panel(axes2[i + 1], logfc_grids[i], title, LOGFC_CMAP, norm=norm, title_fontsize=6.4)
    for ax in axes2[n_panels:]:
        ax.axis("off")
    fig2.subplots_adjust(left=0.03, right=0.935, top=0.92, bottom=0.07, wspace=0.08, hspace=0.24)
    cax_rna2 = fig2.add_axes([0.942, 0.72, 0.012, 0.16])
    cax_logfc = fig2.add_axes([0.942, 0.34, 0.012, 0.30])
    cb_rna2 = mpl.colorbar.ColorbarBase(cax_rna2, cmap=RNA_CMAP, norm=Normalize(0, 1.0), orientation="vertical")
    cb_rna2.ax.tick_params(labelsize=5.8, length=2)
    cb_rna2.set_label("control RNA\np99 norm", fontsize=6.0)
    cb_logfc = mpl.colorbar.ColorbarBase(cax_logfc, cmap=LOGFC_CMAP, norm=norm, orientation="vertical")
    cb_logfc.ax.tick_params(labelsize=5.8, length=2)
    cb_logfc.set_label("spot-wise log2FC\nenhancer / control", fontsize=6.0)
    fig2.suptitle(
        f"Rectified-H {gene} filtered enhancers: spot-wise log2FC versus {gene} control",
        x=0.03, ha="left", fontsize=11, fontweight="bold",
    )
    fig2.text(
        0.03, 0.022,
        f"Control was defined as {gene} RNA within the gene-specific latent-state mask. Gene and enhancer maps were independently clipped at their own 99th percentile and rescaled to 0–1 before computing "
        f"log2((enhancer+{EPS})/(control+{EPS})).",
        fontsize=5.8, color="#444444",
    )
    fig2.savefig(out_logfc, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print(out[["panel_id", "enhancer", "initial_rho", "rectified_raw_rho", "latent_state_rho", "mean_logfc_active"]].head(10).to_csv(sep="\t", index=False))
    rna.file.close()
    atac.file.close()


if __name__ == "__main__":
    main()
