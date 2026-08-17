import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.io import read_feature_matrix


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
ATAC_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad"
LINK_PATH = ROOT / "data" / "region_to_gene_adj.tsv"
OUT_PNG = ROOT / "plot" / "NEFM_region_to_gene_adj_rectified_h_filtered_logfc_vs_control.png"
OUT_TSV = ROOT / "plot" / "NEFM_region_to_gene_adj_rectified_h_filtered_logfc_vs_control.tsv"

GENE = "NEFM"
EPS = 1e-3
EXCLUDE_PANEL_IDS = {
    "E13", "E14", "E29", "E30", "E32", "E34", "E35", "E36",
    "E38", "E39", "E40", "E41", "E42", "E43", "E44", "E45",
}

RNA_CMAP = LinearSegmentedColormap.from_list(
    "rna_nefm_control", ["#F7F7F7", "#DCEAF7", "#6BAED6", "#2171B5", "#08306B"], N=256
)
LOGFC_CMAP = LinearSegmentedColormap.from_list(
    "logfc_div",
    ["#2166AC", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#B2182B"],
    N=256,
)
MASK_CMAP = LinearSegmentedColormap.from_list("latent_mask_demo", ["#EDEDED", "#0F4D92"], N=2)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
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


def latent_state_select(atac_grid, mask):
    return atac_grid * mask.astype(float)


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


def add_panel(ax, grid, title, cmap, norm=None, vmin=None, vmax=None, label=None, title_fontsize=6.0):
    ax.imshow(grid, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper", aspect="auto")
    ax.set_title(title, loc="left", fontsize=title_fontsize, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label:
        ax.text(-0.06, 1.04, label, transform=ax.transAxes, fontsize=7.8, fontweight="bold")


def main():
    rna = ad.read_h5ad(RNA_PATH, backed="r")
    atac = ad.read_h5ad(ATAC_PATH, backed="r")
    if not np.array_equal(rna.obs_names.to_numpy(), atac.obs_names.to_numpy()):
        raise ValueError("Rectified RNA and ATAC obs order differs")

    links = pd.read_csv(LINK_PATH, sep="\t")
    nefm_links = (
        links.loc[links["target"].astype(str) == GENE]
        .drop_duplicates(subset=["region"])
        .sort_values("importance_x_abs_rho", ascending=False)
        .reset_index(drop=True)
    )
    nefm_links["panel_id"] = [f"E{i+1}" for i in range(len(nefm_links))]
    nefm_links = nefm_links.loc[~nefm_links["panel_id"].isin(EXCLUDE_PANEL_IDS)].reset_index(drop=True)
    enhancers = nefm_links["region"].astype(str).tolist()

    expr = read_feature_matrix(rna, [GENE], obs_names=rna.obs_names, batch_size=1)[:, 0]
    expr = np.log1p(np.maximum(expr, 0))
    expr_grid = dense_to_grid(rna.obs, expr)
    gene_norm = normalize01(expr_grid)
    raw_mask, c_low, c_high = fit_two_state_mask(gene_norm, n_iter=25)
    latent_mask = majority_regularize(raw_mask, n_rounds=3, threshold=5)

    expr_control_grid = expr_grid * latent_mask.astype(float)
    expr_control_norm, expr_p99 = per_panel_p99_normalize(expr_control_grid, percentile=99.0)

    enhancer_vals = read_feature_matrix(atac, enhancers, obs_names=atac.obs_names, batch_size=64)
    enhancer_vals = np.log1p(np.maximum(enhancer_vals, 0))

    logfc_grids = []
    summary_rows = []
    for i, enh in enumerate(enhancers):
        raw_grid = dense_to_grid(atac.obs, enhancer_vals[:, i])
        latent_grid = latent_state_select(raw_grid, latent_mask)
        enh_norm_grid, enh_p99 = per_panel_p99_normalize(latent_grid, percentile=99.0)

        valid = np.isfinite(expr_control_norm) & np.isfinite(enh_norm_grid) & (latent_mask > 0)
        logfc_grid = np.full_like(enh_norm_grid, np.nan, dtype=float)
        logfc_grid[valid] = np.log2((enh_norm_grid[valid] + EPS) / (expr_control_norm[valid] + EPS))
        logfc_grids.append(logfc_grid)

        active_vals = logfc_grid[valid]
        summary_rows.append({
            "gene": GENE,
            "enhancer": enh,
            "panel_id": nefm_links.loc[i, "panel_id"],
            "initial_rho": float(nefm_links.loc[i, "rho"]),
            "importance": float(nefm_links.loc[i, "importance"]),
            "importance_x_abs_rho": float(nefm_links.loc[i, "importance_x_abs_rho"]),
            "distance": nefm_links.loc[i, "Distance"],
            "gene_control_p99": expr_p99,
            "enhancer_p99": enh_p99,
            "mean_logfc_active": float(np.nanmean(active_vals)) if active_vals.size else np.nan,
            "median_logfc_active": float(np.nanmedian(active_vals)) if active_vals.size else np.nan,
            "min_logfc_active": float(np.nanmin(active_vals)) if active_vals.size else np.nan,
            "max_logfc_active": float(np.nanmax(active_vals)) if active_vals.size else np.nan,
        })

    vals = np.concatenate([g[np.isfinite(g)] for g in logfc_grids if np.isfinite(g).any()])
    vmax = float(np.percentile(np.abs(vals), 97.5)) if vals.size else 2.0
    vmax = max(vmax, 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n_panels = 1 + len(enhancers)
    ncols = 8
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.4, 1.72 * nrows), constrained_layout=False)
    axes = np.atleast_1d(axes).ravel()

    add_panel(axes[0], expr_control_norm, "NEFM control\nmask × RNA p99 norm", RNA_CMAP, vmin=0, vmax=1, label="A", title_fontsize=6.8)
    for i, row in nefm_links.iterrows():
        item = summary_rows[i]
        title = f"{row['panel_id']}\nmean logFC={item['mean_logfc_active']:.2f}"
        add_panel(axes[i + 1], logfc_grids[i], title, LOGFC_CMAP, norm=norm, title_fontsize=6.4)
    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.subplots_adjust(left=0.03, right=0.935, top=0.92, bottom=0.07, wspace=0.08, hspace=0.24)
    cax_rna = fig.add_axes([0.942, 0.72, 0.012, 0.16])
    cax_logfc = fig.add_axes([0.942, 0.34, 0.012, 0.30])
    cb_rna = mpl.colorbar.ColorbarBase(cax_rna, cmap=RNA_CMAP, norm=Normalize(0, 1.0), orientation="vertical")
    cb_rna.ax.tick_params(labelsize=5.8, length=2)
    cb_rna.set_label("control RNA\np99 norm", fontsize=6.0)
    cb_logfc = mpl.colorbar.ColorbarBase(cax_logfc, cmap=LOGFC_CMAP, norm=norm, orientation="vertical")
    cb_logfc.ax.tick_params(labelsize=5.8, length=2)
    cb_logfc.set_label("spot-wise log2FC\nenhancer / control", fontsize=6.0)

    fig.suptitle(
        "Rectified-H NEFM filtered enhancers: spot-wise log2FC versus NEFM control",
        x=0.03, ha="left", fontsize=11, fontweight="bold",
    )
    fig.text(
        0.03, 0.022,
        f"Control was defined as NEFM RNA within the latent-state mask. Gene and enhancer maps were independently clipped at their own 99th percentile and rescaled to 0–1 before computing "
        f"log2((enhancer+{EPS})/(control+{EPS})). Panels retain original enhancer IDs after filtering.",
        fontsize=5.8, color="#444444",
    )
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(OUT_TSV, sep="\t", index=False)
    rna.file.close()
    atac.file.close()


if __name__ == "__main__":
    main()
