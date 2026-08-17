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
from scipy.ndimage import gaussian_filter, label


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
OUT_PNG = ROOT / "plot" / "NEFM_NEFL_compact_mask_methods.png"
OUT_TSV = ROOT / "plot" / "NEFM_NEFL_compact_mask_methods.tsv"

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
    "font.size": 6.0,
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


def top_quantile_mask(smooth, valid, q=0.92):
    vals = smooth[valid]
    thr = float(np.quantile(vals, q))
    raw = np.zeros_like(smooth, dtype=np.uint8)
    raw[valid] = (smooth[valid] >= thr).astype(np.uint8)
    mask, sizes = keep_largest_components(raw, min_size=MIN_COMPONENT_SIZE, max_domains=MAX_DOMAINS)
    return mask, {"method_param": q, "threshold": thr, "component_sizes": sizes}


def adaptive_quantile_mask(smooth, valid):
    vals = smooth[valid]
    best = None
    for q in np.linspace(0.82, 0.98, 17):
        thr = float(np.quantile(vals, q))
        raw = np.zeros_like(smooth, dtype=np.uint8)
        raw[valid] = (smooth[valid] >= thr).astype(np.uint8)
        mask, sizes = keep_largest_components(raw, min_size=MIN_COMPONENT_SIZE, max_domains=MAX_DOMAINS)
        area = float(mask.sum())
        if area < 80 or area > 2200:
            continue
        labels, n = label(mask.astype(bool), structure=np.ones((3, 3), dtype=int))
        largest_frac = (max(sizes) / area) if sizes else 0.0
        compact = compactness_score(mask)
        area_frac = area / float(valid.sum())
        score = 1.2 * largest_frac + 1.0 * compact - 0.8 * area_frac - 0.15 * max(n - 2, 0)
        item = (score, q, thr, mask, sizes, largest_frac, compact, area_frac, n)
        if best is None or item[0] > best[0]:
            best = item
    if best is None:
        return top_quantile_mask(smooth, valid, q=0.94)[0], {"method_param": 0.94, "threshold": float(np.quantile(vals, 0.94)), "component_sizes": []}
    _, q, thr, mask, sizes, largest_frac, compact, area_frac, n = best
    return mask, {
        "method_param": q,
        "threshold": thr,
        "component_sizes": sizes,
        "largest_frac": largest_frac,
        "compactness": compact,
        "area_frac": area_frac,
        "n_components": n,
    }


def mad_mask(smooth, valid, k=2.5):
    vals = smooth[valid]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)) + 1e-8)
    thr = med + k * 1.4826 * mad
    raw = np.zeros_like(smooth, dtype=np.uint8)
    raw[valid] = (smooth[valid] >= thr).astype(np.uint8)
    mask, sizes = keep_largest_components(raw, min_size=MIN_COMPONENT_SIZE, max_domains=MAX_DOMAINS)
    return mask, {"method_param": k, "threshold": thr, "component_sizes": sizes, "median": med, "mad": mad}


def robust_vmax(values, percentile=99.5):
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(positive, percentile)) if positive.size else 1.0


def add_panel(ax, grid, title, cmap, vmin=None, vmax=None):
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper", aspect="auto")
    ax.set_title(title, loc="left", fontsize=6.0, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    rna = ad.read_h5ad(RNA_PATH, backed="r")
    var_index = {g: i for i, g in enumerate(rna.var_names.astype(str))}
    rows = []
    plot_rows = []
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

        q_mask, q_meta = top_quantile_mask(smooth, valid, q=0.92)
        a_mask, a_meta = adaptive_quantile_mask(smooth, valid)
        m_mask, m_meta = mad_mask(smooth, valid, k=2.5)

        methods = {
            "old": (old_mask, {"threshold": np.nan, "method_param": np.nan, "component_sizes": []}),
            "Q": (q_mask, q_meta),
            "A": (a_mask, a_meta),
            "M": (m_mask, m_meta),
        }

        for name, (mask, meta) in methods.items():
            union = ((old_mask > 0) | (mask > 0)).sum()
            jacc = float(((old_mask > 0) & (mask > 0)).sum() / (union + 1e-8))
            rows.append({
                "gene": gene,
                "method": name,
                "area": int(mask.sum()),
                "area_vs_old": float(mask.sum() / max(old_mask.sum(), 1)),
                "jaccard_vs_old": jacc,
                "threshold": meta.get("threshold", np.nan),
                "param": meta.get("method_param", np.nan),
                "component_sizes": ",".join(map(str, meta.get("component_sizes", []))),
                "old_center_low": c_low,
                "old_center_high": c_high,
            })
        plot_rows.append((gene, expr_grid, old_mask, q_mask, a_mask, m_mask))

    pd.DataFrame(rows).to_csv(OUT_TSV, sep="\t", index=False)

    fig, axes = plt.subplots(len(GENES), 5, figsize=(12.4, 3.6 * len(GENES)), constrained_layout=False)
    axes = np.atleast_2d(axes)
    for r, (gene, expr_grid, old_mask, q_mask, a_mask, m_mask) in enumerate(plot_rows):
        expr_vmax = robust_vmax(expr_grid)
        add_panel(axes[r, 0], expr_grid, f"{gene} RNA", RNA_CMAP, vmin=0, vmax=expr_vmax)
        add_panel(axes[r, 1], old_mask, f"{gene} old\narea={int(old_mask.sum())}", MASK_CMAP, vmin=0, vmax=1)
        add_panel(axes[r, 2], q_mask, f"{gene} Q\narea={int(q_mask.sum())}", MASK_CMAP, vmin=0, vmax=1)
        add_panel(axes[r, 3], a_mask, f"{gene} A\narea={int(a_mask.sum())}", MASK_CMAP, vmin=0, vmax=1)
        add_panel(axes[r, 4], m_mask, f"{gene} M\narea={int(m_mask.sum())}", MASK_CMAP, vmin=0, vmax=1)

    fig.subplots_adjust(left=0.04, right=0.94, top=0.92, bottom=0.08, wspace=0.10, hspace=0.18)
    cax = fig.add_axes([0.95, 0.66, 0.012, 0.20])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=RNA_CMAP, norm=mpl.colors.Normalize(0, 1), orientation="vertical")
    cb.set_label("RNA (relative)", fontsize=6.0)
    cb.ax.tick_params(labelsize=5.8, length=2)
    fig.suptitle(
        "NEFM / NEFL compact-mask alternatives",
        x=0.04, ha="left", fontsize=11, fontweight="bold",
    )
    fig.text(
        0.04, 0.025,
        "Q: fixed high quantile (q=0.92). A: adaptive quantile sweep optimized for compact dominant components. M: robust median+2.5*MAD threshold. "
        "All methods are gene-wise, unsupervised, and keep only the largest 1–2 connected domains.",
        fontsize=5.8, color="#444444",
    )
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    rna.file.close()


if __name__ == "__main__":
    main()
