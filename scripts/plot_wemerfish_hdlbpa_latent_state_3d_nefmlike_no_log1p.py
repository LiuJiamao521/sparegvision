from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = Path("/cluster3/labData/jiamao/SpaRegVision/weMERFISH/weMERFISH_combined_C_6s_E1_rescaled_z.h5ad")
OUT_PNG = ROOT / "results" / "weMERFISH" / "hdlbpa_latent_state_3d_nefmlike_no_log1p.png"
OUT_PDF = ROOT / "results" / "weMERFISH" / "hdlbpa_latent_state_3d_nefmlike_no_log1p.pdf"
OUT_TSV = ROOT / "results" / "weMERFISH" / "hdlbpa_latent_state_3d_nefmlike_no_log1p.summary.tsv"
OUT_CELL_TSV = ROOT / "results" / "weMERFISH" / "hdlbpa_latent_state_3d_nefmlike_no_log1p.cells.tsv.gz"
GENE = "hdlbpa"
K_NEIGHBORS = 8
N_ROUNDS = 3
MAJORITY_THRESHOLD = 5

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
})

def normalize01(x):
    x = np.asarray(x, dtype=float)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    return (x - xmin) / max(xmax - xmin, 1e-8)

def fit_two_state_mask(values_norm, n_iter=25):
    vals = np.asarray(values_norm, dtype=float).ravel()
    c0, c1 = np.quantile(vals, [0.25, 0.75])
    z = np.zeros(vals.shape, dtype=bool)
    for _ in range(n_iter):
        d0 = np.abs(vals - c0)
        d1 = np.abs(vals - c1)
        z = d1 < d0
        if np.any(~z):
            c0 = vals[~z].mean()
        if np.any(z):
            c1 = vals[z].mean()
    active_label = 1 if c1 > c0 else 0
    mask = z if active_label == 1 else ~z
    return mask.astype(np.uint8), float(min(c0, c1)), float(max(c0, c1))

def majority_regularize_knn(mask, coords, k=8, n_rounds=3, threshold=5):
    out = np.asarray(mask, dtype=np.uint8).copy()
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(coords)
    neigh = nn.kneighbors(return_distance=False)
    for _ in range(n_rounds):
        votes = out[neigh].sum(axis=1)
        out = (votes >= threshold).astype(np.uint8)
    return out

def identity_scale(values):
    v = np.asarray(values, dtype=float).copy()
    vmax = float(np.nanmax(v)) if np.isfinite(v).any() else 0.0
    return v, vmax

def add_scatter(ax, xy, values, cmap, title, vmin=None, vmax=None):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=0.2, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return sc

def main():
    adata = ad.read_h5ad(RNA_PATH)
    if GENE not in adata.var_names:
        raise KeyError(f"{GENE} not found")

    gene_idx = int(np.where(adata.var_names == GENE)[0][0])
    expr = np.asarray(adata.X[:, gene_idx]).reshape(-1)
    expr_raw = np.clip(expr, 0, None)
    expr_norm = normalize01(expr_raw)

    coords3d = np.asarray(adata.obsm["spatial_rescaled_z"], dtype=float)
    valid3d = np.isfinite(coords3d).all(axis=1)
    xy = coords3d[:, [0, 1]]
    xz = coords3d[:, [0, 2]]

    raw_mask, c_low, c_high = fit_two_state_mask(expr_norm, n_iter=25)
    latent_mask = np.full(raw_mask.shape, 0, dtype=np.uint8)
    latent_mask[valid3d] = majority_regularize_knn(raw_mask[valid3d], coords3d[valid3d], k=K_NEIGHBORS, n_rounds=N_ROUNDS, threshold=MAJORITY_THRESHOLD)

    expr_plot, expr_vmax = identity_scale(expr_raw)
    raw_active_n = int(raw_mask.sum())
    raw_active_frac = float(raw_mask.mean())
    latent_active_n = int(latent_mask.sum())
    latent_active_frac = float(latent_mask.mean())
    valid3d_n = int(valid3d.sum())

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
    blue = mpl.cm.Blues
    pink = mpl.colors.ListedColormap(["#E6E6E6", "#C51B8A"])

    sc1 = add_scatter(axes[0, 0], xy, expr_plot, blue, "HDLBPA RNA (xy)\nraw, no clipping")
    sc2 = add_scatter(axes[0, 1], xy, latent_mask, pink, f"3D latent state projected to xy\nactive={latent_active_n}/{adata.n_obs} ({latent_active_frac:.2%})", vmin=0, vmax=1)
    sc3 = add_scatter(axes[1, 0], xz, expr_plot, blue, "HDLBPA RNA (xz)\nraw, no clipping")
    sc4 = add_scatter(axes[1, 1], xz, latent_mask, pink, "3D latent state projected to xz", vmin=0, vmax=1)

    cbar1 = fig.colorbar(sc1, ax=[axes[0, 0], axes[1, 0]], fraction=0.03, pad=0.02)
    cbar1.set_label("HDLBPA raw RNA", fontsize=7)
    cbar1.ax.tick_params(labelsize=6)
    cbar2 = fig.colorbar(sc2, ax=[axes[0, 1], axes[1, 1]], fraction=0.03, pad=0.02, ticks=[0, 1])
    cbar2.set_label("latent state", fontsize=7)
    cbar2.ax.tick_params(labelsize=6)

    fig.suptitle("weMERFISH HDLBPA 3D NEFM-like latent state (no log1p)", fontsize=10, y=0.98)
    fig.text(
        0.01,
        0.01,
        f"Computation followed the 3D NEFM-like pipeline except for no log1p: raw RNA -> min-max normalization -> two-state fit ({c_low:.3f} vs {c_high:.3f}) -> spatial regularization. "
        f"Regularization used {N_ROUNDS} rounds of 3D kNN majority voting on spatial_rescaled_z with {K_NEIGHBORS}+self neighbors and threshold {MAJORITY_THRESHOLD}. "
        f"Cells with valid 3D coordinates: {valid3d_n}/{adata.n_obs}. Expression vmax={expr_vmax:.3f}. Raw active fraction={raw_active_frac:.2%}; regularized active fraction={latent_active_frac:.2%}.",
        ha="left", va="bottom", fontsize=6.3, color="#555555"
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    pd.DataFrame([{
        "gene": GENE,
        "n_cells": int(adata.n_obs),
        "expr_vmax_raw": expr_vmax,
        "mask_low_center": c_low,
        "mask_high_center": c_high,
        "raw_active_n": raw_active_n,
        "raw_active_fraction": raw_active_frac,
        "latent_active_n": latent_active_n,
        "latent_active_fraction": latent_active_frac,
        "knn_neighbors": K_NEIGHBORS,
        "regularization_rounds": N_ROUNDS,
        "majority_threshold": MAJORITY_THRESHOLD,
        "spatial_basis": "spatial_rescaled_z",
        "regularization_space": "3D",
        "valid_3d_cells": valid3d_n,
    }]).to_csv(OUT_TSV, sep="\t", index=False)

    pd.DataFrame({
        "cell_id": adata.obs["cell_id"].astype(str).to_numpy() if "cell_id" in adata.obs.columns else adata.obs_names.astype(str),
        "spatial_x": coords3d[:, 0],
        "spatial_y": coords3d[:, 1],
        "spatial_z_rescaled": coords3d[:, 2],
        "valid_3d": valid3d.astype(int),
        "hdlbpa_expr": expr,
        "hdlbpa_raw_nonneg": expr_raw,
        "hdlbpa_norm01": expr_norm,
        "hdlbpa_raw_state": raw_mask.astype(int),
        "hdlbpa_latent_state_3d": latent_mask.astype(int),
    }).to_csv(OUT_CELL_TSV, sep="\t", index=False)

if __name__ == "__main__":
    main()
