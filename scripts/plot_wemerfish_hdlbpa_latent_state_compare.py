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
OUT_PNG = ROOT / "results" / "weMERFISH" / "hdlbpa_latent_state_compare_xy_no_log1p.png"
OUT_PDF = ROOT / "results" / "weMERFISH" / "hdlbpa_latent_state_compare_xy_no_log1p.pdf"
OUT_TSV = ROOT / "results" / "weMERFISH" / "hdlbpa_latent_state_compare_xy_no_log1p.summary.tsv"
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

def p99_clip(values):
    v = np.asarray(values, dtype=float).copy()
    pos = v[v > 0]
    if pos.size == 0:
        return np.zeros_like(v), 0.0
    p99 = float(np.percentile(pos, 99))
    p99 = max(p99, 1e-8)
    return np.clip(v, 0, p99) / p99, p99

def add_scatter(ax, xy, values, cmap, title, vmin=None, vmax=None):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=5, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return sc

def main():
    adata = ad.read_h5ad(RNA_PATH)
    gene_idx = int(np.where(adata.var_names == GENE)[0][0])
    expr = np.asarray(adata.X[:, gene_idx]).reshape(-1)
    expr_raw = np.clip(expr, 0, None)
    expr_norm = normalize01(expr_raw)
    coords_xy = np.asarray(adata.obsm["spatial_rescaled_z"], dtype=float)[:, [0, 1]]

    raw_mask, c_low, c_high = fit_two_state_mask(expr_norm, n_iter=25)
    reg_mask = majority_regularize_knn(raw_mask, coords_xy, k=K_NEIGHBORS, n_rounds=N_ROUNDS, threshold=MAJORITY_THRESHOLD)
    expr_plot, expr_p99 = p99_clip(expr_raw)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    blue = mpl.cm.Blues
    pink = mpl.colors.ListedColormap(["#E6E6E6", "#C51B8A"])

    sc1 = add_scatter(axes[0], coords_xy, expr_plot, blue, "HDLBPA raw expression\nraw value, p99 clipped")
    sc2 = add_scatter(axes[1], coords_xy, raw_mask, pink, f"raw two-state mask\nactive={int(raw_mask.sum())}/{adata.n_obs} ({raw_mask.mean():.2%})", vmin=0, vmax=1)
    sc3 = add_scatter(axes[2], coords_xy, reg_mask, pink, f"regularized mask\nactive={int(reg_mask.sum())}/{adata.n_obs} ({reg_mask.mean():.2%})", vmin=0, vmax=1)

    cbar1 = fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.02)
    cbar1.set_label("HDLBPA raw RNA", fontsize=7)
    cbar1.ax.tick_params(labelsize=6)
    cbar2 = fig.colorbar(sc2, ax=axes[1:], fraction=0.025, pad=0.02, ticks=[0, 1])
    cbar2.set_label("state", fontsize=7)
    cbar2.ax.tick_params(labelsize=6)

    fig.suptitle("weMERFISH HDLBPA: raw expression vs latent-state variants (xy, no log1p)", fontsize=10, y=0.98)
    fig.text(0.01, 0.01,
             f"Preprocessing here: raw RNA -> min-max normalization -> two-state fit ({c_low:.3f} vs {c_high:.3f}). "
             f"The third panel adds xy kNN majority regularization ({N_ROUNDS} rounds, {K_NEIGHBORS}+self neighbors, threshold {MAJORITY_THRESHOLD}). "
             f"Expression p99={expr_p99:.3f}.",
             ha="left", va="bottom", fontsize=6.4, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    pd.DataFrame([{
        "gene": GENE,
        "n_cells": int(adata.n_obs),
        "expr_p99_raw": expr_p99,
        "mask_low_center": c_low,
        "mask_high_center": c_high,
        "raw_active_n": int(raw_mask.sum()),
        "raw_active_fraction": float(raw_mask.mean()),
        "regularized_active_n": int(reg_mask.sum()),
        "regularized_active_fraction": float(reg_mask.mean()),
        "knn_neighbors": K_NEIGHBORS,
        "regularization_rounds": N_ROUNDS,
        "majority_threshold": MAJORITY_THRESHOLD,
    }]).to_csv(OUT_TSV, sep="\t", index=False)

if __name__ == "__main__":
    main()
