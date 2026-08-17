import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse
from scipy.ndimage import gaussian_filter, label, binary_dilation, binary_erosion
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
OUT_PNG = ROOT / "plot" / "NEFM_NEFL_vision_masks.png"
OUT_TSV = ROOT / "plot" / "NEFM_NEFL_vision_masks.tsv"

GENES = ["NEFM", "NEFL"]
GRID_SHAPE = (76, 101)
SIGMA = 0.8
MIN_COMPONENT_SIZE = 25
MAX_DOMAINS = 2

RNA_CMAP = LinearSegmentedColormap.from_list(
    "rna_shape", ["#F7F7F7", "#DCEAF7", "#6BAED6", "#2171B5", "#08306B"], N=256
)
MASK_CMAP = LinearSegmentedColormap.from_list("mask_demo", ["#EDEDED", "#0F4D92"], N=2)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams.update({
    "font.size": 5.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def dense_to_grid(obs, values):
    grid = np.full(GRID_SHAPE, np.nan, dtype=float)
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


def keep_largest_components(mask, min_size=25, max_domains=2):
    labels, n = label(mask.astype(bool), structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8), []
    sizes = np.bincount(labels.ravel())[1:]
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_size]
    keep = sorted(keep, key=lambda i: sizes[i - 1], reverse=True)[:max_domains]
    out = np.isin(labels, keep).astype(np.uint8)
    kept_sizes = [int(sizes[i - 1]) for i in keep]
    return out, kept_sizes


def robust_vmax(values, percentile=99.5):
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(positive, percentile)) if positive.size else 1.0


def compactness_score(mask):
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    center = padded[1:-1, 1:-1]
    perimeter = (
        np.abs(center - padded[:-2, 1:-1]).sum()
        + np.abs(center - padded[2:, 1:-1]).sum()
        + np.abs(center - padded[1:-1, :-2]).sum()
        + np.abs(center - padded[1:-1, 2:]).sum()
    )
    return float(area / (perimeter + 1e-8))


def mask_stats(mask, old_mask):
    union = ((old_mask > 0) | (mask > 0)).sum()
    jacc = float(((old_mask > 0) & (mask > 0)).sum() / (union + 1e-8))
    return int(mask.sum()), float(mask.sum() / max(old_mask.sum(), 1)), jacc


def hysteresis_mask(smooth, valid, q_hi=0.94, q_lo=0.88):
    vals = smooth[valid]
    hi = float(np.quantile(vals, q_hi))
    lo = float(np.quantile(vals, q_lo))
    high = np.zeros_like(smooth, dtype=bool)
    low = np.zeros_like(smooth, dtype=bool)
    high[valid] = smooth[valid] >= hi
    low[valid] = smooth[valid] >= lo
    grown = high.copy()
    while True:
        new = binary_dilation(grown, structure=np.ones((3, 3), dtype=bool)) & low
        if np.array_equal(new, grown):
            break
        grown = new
    mask, sizes = keep_largest_components(grown.astype(np.uint8), MIN_COMPONENT_SIZE, MAX_DOMAINS)
    return mask, {"param": f"{q_hi:.2f}/{q_lo:.2f}", "sizes": sizes, "thr_hi": hi, "thr_lo": lo}


def random_walker_style_mask(smooth, valid, q_fg=0.95, q_bg=0.55, beta=25.0):
    vals = smooth[valid]
    fg_thr = float(np.quantile(vals, q_fg))
    bg_thr = float(np.quantile(vals, q_bg))
    fg = valid & (smooth >= fg_thr)
    bg = valid & (smooth <= bg_thr)
    idx = -np.ones_like(smooth, dtype=int)
    coords = np.argwhere(valid)
    for k, (i, j) in enumerate(coords):
        idx[i, j] = k
    n = len(coords)
    if n == 0:
        return np.zeros_like(smooth, dtype=np.uint8), {"param": f"{q_fg:.2f}/{q_bg:.2f}", "sizes": []}
    L = lil_matrix((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    for k, (i, j) in enumerate(coords):
        if fg[i, j]:
            L[k, k] = 1.0
            b[k] = 1.0
            continue
        if bg[i, j]:
            L[k, k] = 1.0
            b[k] = 0.0
            continue
        nbrs = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < smooth.shape[0] and 0 <= nj < smooth.shape[1] and valid[ni, nj]:
                nbrs.append((ni, nj))
        if not nbrs:
            L[k, k] = 1.0
            b[k] = 0.0
            continue
        wsum = 0.0
        for ni, nj in nbrs:
            kk = idx[ni, nj]
            diff = smooth[i, j] - smooth[ni, nj]
            w = np.exp(-beta * diff * diff)
            L[k, k] += w
            L[k, kk] -= w
            wsum += w
        if wsum == 0:
            L[k, k] = 1.0
    p = spsolve(L.tocsr(), b)
    prob = np.zeros_like(smooth, dtype=float)
    for k, (i, j) in enumerate(coords):
        prob[i, j] = p[k]
    raw = valid & (prob >= 0.70)
    raw &= smooth >= np.quantile(vals, 0.82)
    mask, sizes = keep_largest_components(raw.astype(np.uint8), MIN_COMPONENT_SIZE, MAX_DOMAINS)
    return mask, {"param": f"{q_fg:.2f}/{q_bg:.2f}", "sizes": sizes, "fg_thr": fg_thr, "bg_thr": bg_thr}


def active_contour_style_mask(smooth, valid, q_seed=0.95, q_cap=0.84, edge_q=0.70, n_iter=20):
    vals = smooth[valid]
    seed_thr = float(np.quantile(vals, q_seed))
    cap_thr = float(np.quantile(vals, q_cap))
    gx, gy = np.gradient(smooth)
    edge = np.sqrt(gx * gx + gy * gy)
    edge_thr = float(np.quantile(edge[valid], edge_q))
    seed = valid & (smooth >= seed_thr)
    candidate = valid & (smooth >= cap_thr)
    mask = seed.copy()
    for _ in range(n_iter):
        border = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool)) & (~mask)
        grow = border & candidate & (edge <= edge_thr)
        if not np.any(grow):
            break
        mask |= grow
        # mild smoothing to avoid spikes
        mask = binary_erosion(binary_dilation(mask, structure=np.ones((3, 3), dtype=bool)), structure=np.ones((3, 3), dtype=bool))
        mask &= candidate
    mask, sizes = keep_largest_components(mask.astype(np.uint8), MIN_COMPONENT_SIZE, MAX_DOMAINS)
    return mask, {"param": f"{q_seed:.2f}/{q_cap:.2f}", "sizes": sizes, "seed_thr": seed_thr, "cap_thr": cap_thr}


def diffusion_cluster_mask(smooth, valid, alpha=0.55):
    fill = np.zeros_like(smooth, dtype=float)
    fill[valid] = smooth[valid]
    diffuse = gaussian_filter(fill, sigma=1.4, mode="nearest")
    yy, xx = np.indices(smooth.shape)
    feat = np.column_stack([
        normalize01(diffuse[valid]),
        alpha * normalize01(yy[valid]),
        alpha * normalize01(xx[valid]),
    ])
    km = KMeans(n_clusters=3, random_state=0, n_init=20)
    lab = km.fit_predict(feat)
    means = [float(np.mean(diffuse[valid][lab == k])) for k in range(3)]
    best = int(np.argmax(means))
    raw = np.zeros_like(smooth, dtype=np.uint8)
    raw[valid] = (lab == best).astype(np.uint8)
    mask, sizes = keep_largest_components(raw, MIN_COMPONENT_SIZE, MAX_DOMAINS)
    return mask, {"param": alpha, "sizes": sizes, "cluster_means": means}


def prototype_similarity_mask(smooth, valid, q_seed=0.95, q_keep=0.86):
    vals = smooth[valid]
    seed_thr = float(np.quantile(vals, q_seed))
    keep_thr = float(np.quantile(vals, q_keep))
    yy, xx = np.indices(smooth.shape)
    y = normalize01(yy.astype(float))
    x = normalize01(xx.astype(float))
    seed = valid & (smooth >= seed_thr)
    if seed.sum() == 0:
        return np.zeros_like(smooth, dtype=np.uint8), {"param": f"{q_seed:.2f}/{q_keep:.2f}", "sizes": []}
    proto = np.array([smooth[seed].mean(), y[seed].mean(), x[seed].mean()])
    feat = np.stack([smooth, y, x], axis=-1)
    scale = np.array([1.0, 0.35, 0.35])
    dist = np.sqrt(np.sum(((feat - proto) / scale) ** 2, axis=-1))
    sim = np.zeros_like(smooth, dtype=float)
    sim[valid] = np.exp(-0.5 * dist[valid] ** 2)
    sim_thr = float(np.quantile(sim[valid & (smooth >= keep_thr)], 0.35)) if np.any(valid & (smooth >= keep_thr)) else 0.5
    raw = valid & (smooth >= keep_thr) & (sim >= sim_thr)
    raw |= seed
    mask, sizes = keep_largest_components(raw.astype(np.uint8), MIN_COMPONENT_SIZE, MAX_DOMAINS)
    return mask, {"param": f"{q_seed:.2f}/{q_keep:.2f}", "sizes": sizes, "seed_thr": seed_thr, "keep_thr": keep_thr}


def add_panel(ax, grid, title, cmap, vmin=None, vmax=None):
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper", aspect="auto")
    ax.set_title(title, loc="left", fontsize=5.8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    rna = ad.read_h5ad(RNA_PATH, backed="r")
    var_index = {g: i for i, g in enumerate(rna.var_names.astype(str))}
    rows = []
    plot_rows = []
    methods = [
        ("old", None),
        ("H", hysteresis_mask),
        ("R", random_walker_style_mask),
        ("C", active_contour_style_mask),
        ("D", diffusion_cluster_mask),
        ("P", prototype_similarity_mask),
    ]

    for gene in GENES:
        x = rna.X[:, var_index[gene]]
        if sparse.issparse(x):
            x = x.toarray().reshape(-1)
        else:
            x = np.asarray(x).reshape(-1)
        expr = np.log1p(np.maximum(x, 0))
        expr_grid = dense_to_grid(rna.obs, expr)
        expr_norm = normalize01(expr_grid)
        old_raw, c_low, c_high = fit_two_state_mask(expr_norm, n_iter=25)
        old_mask = majority_regularize(old_raw, n_rounds=3, threshold=5)

        valid = np.isfinite(expr_grid)
        fill = np.zeros_like(expr_grid, dtype=float)
        fill[valid] = expr_grid[valid]
        smooth = gaussian_filter(fill, sigma=SIGMA, mode="nearest")
        gene_masks = {"old": old_mask}
        rows.append({
            "gene": gene, "method": "old", "area": int(old_mask.sum()), "area_vs_old": 1.0,
            "jaccard_vs_old": 1.0, "param": "", "component_sizes": "", "old_center_low": c_low, "old_center_high": c_high,
        })
        for name, fn in methods[1:]:
            mask, meta = fn(smooth, valid)
            gene_masks[name] = mask
            area, area_ratio, jacc = mask_stats(mask, old_mask)
            rows.append({
                "gene": gene,
                "method": name,
                "area": area,
                "area_vs_old": area_ratio,
                "jaccard_vs_old": jacc,
                "param": meta.get("param", ""),
                "component_sizes": ",".join(map(str, meta.get("sizes", []))),
                "old_center_low": c_low,
                "old_center_high": c_high,
            })
        plot_rows.append((gene, expr_grid, gene_masks))

    pd.DataFrame(rows).to_csv(OUT_TSV, sep="\t", index=False)

    fig, axes = plt.subplots(len(GENES), 7, figsize=(14.8, 3.3 * len(GENES)), constrained_layout=False)
    axes = np.atleast_2d(axes)
    for r, (gene, expr_grid, masks) in enumerate(plot_rows):
        expr_vmax = robust_vmax(expr_grid)
        add_panel(axes[r, 0], expr_grid, f"{gene} RNA", RNA_CMAP, vmin=0, vmax=expr_vmax)
        add_panel(axes[r, 1], masks["old"], f"{gene} old\n{int(masks['old'].sum())}", MASK_CMAP, vmin=0, vmax=1)
        add_panel(axes[r, 2], masks["H"], f"{gene} H\n{int(masks['H'].sum())}", MASK_CMAP, vmin=0, vmax=1)
        add_panel(axes[r, 3], masks["R"], f"{gene} R\n{int(masks['R'].sum())}", MASK_CMAP, vmin=0, vmax=1)
        add_panel(axes[r, 4], masks["C"], f"{gene} C\n{int(masks['C'].sum())}", MASK_CMAP, vmin=0, vmax=1)
        add_panel(axes[r, 5], masks["D"], f"{gene} D\n{int(masks['D'].sum())}", MASK_CMAP, vmin=0, vmax=1)
        add_panel(axes[r, 6], masks["P"], f"{gene} P\n{int(masks['P'].sum())}", MASK_CMAP, vmin=0, vmax=1)

    fig.subplots_adjust(left=0.035, right=0.94, top=0.92, bottom=0.08, wspace=0.08, hspace=0.16)
    cax = fig.add_axes([0.945, 0.66, 0.012, 0.18])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=RNA_CMAP, norm=mpl.colors.Normalize(0, 1), orientation="vertical")
    cb.set_label("RNA (relative)", fontsize=6.0)
    cb.ax.tick_params(labelsize=5.8, length=2)
    fig.suptitle(
        "NEFM / NEFL machine-vision-style compact mask alternatives",
        x=0.035, ha="left", fontsize=11, fontweight="bold",
    )
    fig.text(
        0.035, 0.025,
        "H: hysteresis. R: random-walker-style harmonic diffusion. C: active-contour-style constrained growth. D: diffusion plus clustering. "
        "P: prototype similarity. All methods are gene-wise, unsupervised, and keep only the largest 1–2 connected domains.",
        fontsize=5.8, color="#444444",
    )
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    rna.file.close()


if __name__ == "__main__":
    main()
