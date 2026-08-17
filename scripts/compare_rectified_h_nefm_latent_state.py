import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.io import read_feature_matrix


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
ATAC_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad"
LINK_PATH = ROOT / "data" / "GW12_region_to_gene_links_filtered.tsv"
OUT_PNG = ROOT / "plot" / "NEFM_requested_enhancers_rectified_h_latent_state_compare.png"
OUT_TSV = ROOT / "plot" / "NEFM_requested_enhancers_rectified_h_latent_state_compare.tsv"
OUT_MASK = ROOT / "plot" / "NEFM_latent_active_state_mask.png"

GENE = "NEFM"
REQUESTED_ENHANCERS = [
    "chr8:25021594-25022095",
    "chr8:25000555-25001056",
    "chr8:25034051-25034552",
    "chr8:25065472-25065973",
    "chr8:25033468-25033969",
    "chr8:24886821-24887322",
    "chr8:24999050-24999551",
]

RNA_CMAP = LinearSegmentedColormap.from_list(
    "rna_rectified_h_latent", ["#F7F7F7", "#DCEAF7", "#6BAED6", "#2171B5", "#08306B"], N=256
)
ATAC_CMAP = LinearSegmentedColormap.from_list(
    "atac_rectified_h_latent", ["#EDEDED", "#FFF7F3", "#FDE0DD", "#FCC5C0", "#FA9FB5", "#F768A1", "#DD3497", "#AE017E"], N=256
)
MASK_CMAP = LinearSegmentedColormap.from_list("latent_mask", ["#EDEDED", "#0F4D92"], N=2)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update({
    "font.size": 6.0,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def robust_vmax(values, percentile=99.5):
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(positive, percentile)) if positive.size else 1.0


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


def corr(a, b):
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    if np.std(aa) < 1e-8 or np.std(bb) < 1e-8:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def add_panel(ax, grid, title, cmap, vmax):
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", origin="upper", aspect="auto")
    ax.set_title(title, loc="left", fontsize=5.5, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    rna = ad.read_h5ad(RNA_PATH, backed="r")
    atac = ad.read_h5ad(ATAC_PATH, backed="r")
    if not np.array_equal(rna.obs_names.to_numpy(), atac.obs_names.to_numpy()):
        raise ValueError("Rectified RNA and ATAC obs order differs")

    links = pd.read_csv(LINK_PATH, sep="\t")
    nefm_links = (
        links.loc[links["target"] == GENE]
        .drop_duplicates(subset=["region"])
        .sort_values("importance_x_abs_rho", ascending=False)
        .reset_index(drop=True)
    )
    nefm_links["panel_id"] = [f"E{i+1}" for i in range(len(nefm_links))]
    link_map = nefm_links.set_index("region")

    expr = read_feature_matrix(rna, [GENE], obs_names=rna.obs_names, batch_size=1)[:, 0]
    expr = np.log1p(np.maximum(expr, 0))
    expr_grid = dense_to_grid(rna.obs, expr)
    gene_norm = normalize01(expr_grid)

    raw_mask, c_low, c_high = fit_two_state_mask(gene_norm, n_iter=25)
    latent_mask = majority_regularize(raw_mask, n_rounds=3, threshold=5)

    atac_vals = read_feature_matrix(atac, REQUESTED_ENHANCERS, obs_names=atac.obs_names, batch_size=len(REQUESTED_ENHANCERS))
    atac_vals = np.log1p(np.maximum(atac_vals, 0))
    raw_grids = [dense_to_grid(atac.obs, atac_vals[:, i]) for i in range(len(REQUESTED_ENHANCERS))]
    latent_grids = [latent_state_select(g, latent_mask) for g in raw_grids]

    expr_vmax = robust_vmax(expr_grid)
    atac_vmax = robust_vmax(np.stack(raw_grids))

    summary_rows = []
    fig, axes = plt.subplots(3, 8, figsize=(16.2, 6.8), constrained_layout=False)
    add_panel(axes[0, 0], expr_grid, "A  NEFM RNA", RNA_CMAP, expr_vmax)
    add_panel(axes[1, 0], latent_mask, "latent active\nstate", MASK_CMAP, 1.0)
    axes[2, 0].axis("off")
    axes[2, 0].text(0.5, 0.5, "latent state\nselection", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#555555")

    for i, enh in enumerate(REQUESTED_ENHANCERS):
        panel_id = link_map.loc[enh, "panel_id"] if enh in link_map.index else "NA"
        raw_r = corr(raw_grids[i], expr_grid)
        latent_r = corr(latent_grids[i], expr_grid)
        summary_rows.append({
            "enhancer": enh,
            "panel_id": panel_id,
            "raw_r": raw_r,
            "latent_state_r": latent_r,
            "latent_minus_raw": latent_r - raw_r,
        })
        col = i + 1
        add_panel(axes[0, col], raw_grids[i], f"{panel_id}  raw\nr={raw_r:.2f}", ATAC_CMAP, atac_vmax)
        add_panel(axes[1, col], latent_grids[i], f"latent\nr={latent_r:.2f}", ATAC_CMAP, atac_vmax)
        diff = np.where(raw_grids[i] > 0, latent_grids[i], 0)
        add_panel(axes[2, col], diff, "selected signal", ATAC_CMAP, atac_vmax)

    fig.subplots_adjust(left=0.03, right=0.93, top=0.92, bottom=0.08, wspace=0.06, hspace=0.12)
    cax_rna = fig.add_axes([0.935, 0.70, 0.012, 0.15])
    cax_atac = fig.add_axes([0.935, 0.26, 0.012, 0.28])
    cb_rna = mpl.colorbar.ColorbarBase(cax_rna, cmap=RNA_CMAP, norm=Normalize(0, expr_vmax), orientation="vertical")
    cb_rna.ax.tick_params(labelsize=5.5, length=2)
    cb_rna.set_label("log1p RNA", fontsize=5.7)
    cb_atac = mpl.colorbar.ColorbarBase(cax_atac, cmap=ATAC_CMAP, norm=Normalize(0, atac_vmax), orientation="vertical")
    cb_atac.ax.tick_params(labelsize=5.5, length=2)
    cb_atac.set_label("log1p ATAC", fontsize=5.7)
    fig.suptitle(
        "Rectified-H latent-state selection of ATAC using NEFM expression",
        x=0.03, ha="left", fontsize=11, fontweight="bold"
    )
    fig.text(
        0.03, 0.03,
        f"NEFM expression was converted into a binary latent active state by a two-state fit ({c_low:.2f} vs {c_high:.2f}) and spatial regularization. "
        "ATAC was then retained only inside the inferred active state.",
        fontsize=5.8, color="#444444"
    )
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig_mask, ax_mask = plt.subplots(1, 2, figsize=(6.2, 2.8), constrained_layout=False)
    add_panel(ax_mask[0], expr_grid, "NEFM RNA", RNA_CMAP, expr_vmax)
    add_panel(ax_mask[1], latent_mask, "latent active state", MASK_CMAP, 1.0)
    fig_mask.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.10, wspace=0.12)
    fig_mask.savefig(OUT_MASK, dpi=300, bbox_inches="tight")
    plt.close(fig_mask)

    summary = pd.DataFrame(summary_rows).round(4)
    summary.to_csv(OUT_TSV, sep="\t", index=False)
    rna.file.close()
    atac.file.close()
    print(summary.to_csv(sep="\t", index=False))


if __name__ == "__main__":
    main()
