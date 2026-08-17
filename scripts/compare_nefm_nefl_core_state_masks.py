import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse
from scipy.ndimage import gaussian_filter, label


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
OUT_PNG = ROOT / "plot" / "NEFM_NEFL_active_vs_core_state_masks.png"
OUT_TSV = ROOT / "plot" / "NEFM_NEFL_active_vs_core_state_masks.tsv"

GENES = ["NEFM", "NEFL"]
GRID_SHAPE = (76, 101)
SIGMA = 1.0
POSTERIOR_THRESHOLD = 0.9
MIN_COMPONENT_SIZE = 25
MAX_DOMAINS = 2

RNA_CMAP = LinearSegmentedColormap.from_list(
    "rna_shape", ["#F7F7F7", "#DCEAF7", "#6BAED6", "#2171B5", "#08306B"], N=256
)
MASK_CMAP = LinearSegmentedColormap.from_list("mask_demo", ["#EDEDED", "#0F4D92"], N=2)
POST_CMAP = LinearSegmentedColormap.from_list(
    "posterior_demo", ["#F7F7F7", "#DCEAF7", "#9ECAE1", "#4292C6", "#084594"], N=256
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


def normal_pdf(x, mu, sigma):
    sigma = max(float(sigma), 1e-6)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2 * np.pi))


def fit_gmm_1d(x, n_iter=100):
    x = np.asarray(x, dtype=float)
    q1, q2 = np.quantile(x, [0.25, 0.75])
    mu1, mu2 = float(q1), float(q2)
    s1 = s2 = float(np.std(x) + 1e-4)
    w1 = w2 = 0.5
    for _ in range(n_iter):
        p1 = w1 * normal_pdf(x, mu1, s1)
        p2 = w2 * normal_pdf(x, mu2, s2)
        denom = p1 + p2 + 1e-12
        r1 = p1 / denom
        r2 = p2 / denom
        w1 = float(r1.mean())
        w2 = float(r2.mean())
        mu1 = float((r1 * x).sum() / (r1.sum() + 1e-12))
        mu2 = float((r2 * x).sum() / (r2.sum() + 1e-12))
        s1 = float(np.sqrt(((r1 * (x - mu1) ** 2).sum() / (r1.sum() + 1e-12)) + 1e-8))
        s2 = float(np.sqrt(((r2 * (x - mu2) ** 2).sum() / (r2.sum() + 1e-12)) + 1e-8))
    if mu1 <= mu2:
        return {"mu_low": mu1, "sd_low": s1, "w_low": w1, "mu_high": mu2, "sd_high": s2, "w_high": w2}
    return {"mu_low": mu2, "sd_low": s2, "w_low": w2, "mu_high": mu1, "sd_high": s1, "w_high": w1}


def posterior_high(x, pars):
    p_low = pars["w_low"] * normal_pdf(x, pars["mu_low"], pars["sd_low"])
    p_high = pars["w_high"] * normal_pdf(x, pars["mu_high"], pars["sd_high"])
    return p_high / (p_low + p_high + 1e-12)


def keep_largest_components(mask, min_size=25, max_domains=2):
    labels, n = label(mask.astype(bool), structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return mask.astype(np.uint8), []
    sizes = np.bincount(labels.ravel())[1:]
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_size]
    keep = sorted(keep, key=lambda i: sizes[i - 1], reverse=True)[:max_domains]
    out = np.isin(labels, keep).astype(np.uint8)
    kept_sizes = [int(sizes[i - 1]) for i in keep]
    return out, kept_sizes


def make_core_state(expr_grid):
    valid = np.isfinite(expr_grid)
    fill = np.zeros_like(expr_grid, dtype=float)
    fill[valid] = expr_grid[valid]
    smooth = gaussian_filter(fill, sigma=SIGMA, mode="nearest")
    values = smooth[valid]
    pars = fit_gmm_1d(values, n_iter=100)
    bg_sub = np.zeros_like(smooth)
    bg_sub[valid] = np.maximum(smooth[valid] - pars["mu_low"], 0.0)
    post = np.full_like(smooth, np.nan, dtype=float)
    post[valid] = posterior_high(bg_sub[valid], fit_gmm_1d(bg_sub[valid], n_iter=100))
    seed = np.zeros_like(smooth, dtype=np.uint8)
    seed[valid] = (post[valid] >= POSTERIOR_THRESHOLD).astype(np.uint8)
    core_mask, kept_sizes = keep_largest_components(seed, min_size=MIN_COMPONENT_SIZE, max_domains=MAX_DOMAINS)
    return {
        "smooth": smooth,
        "background_subtracted": bg_sub,
        "posterior": post,
        "core_mask": core_mask,
        "kept_component_sizes": kept_sizes,
        "mu_low": pars["mu_low"],
        "mu_high": pars["mu_high"],
    }


def robust_vmax(values, percentile=99.5):
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(positive, percentile)) if positive.size else 1.0


def add_panel(ax, grid, title, cmap, vmin=None, vmax=None, label=None):
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper", aspect="auto")
    ax.set_title(title, loc="left", fontsize=6.3, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label:
        ax.text(-0.06, 1.04, label, transform=ax.transAxes, fontsize=7.8, fontweight="bold")


def main():
    rna = ad.read_h5ad(RNA_PATH, backed="r")
    var_index = {g: i for i, g in enumerate(rna.var_names.astype(str))}
    summaries = []
    results = []
    for gene in GENES:
        idx = var_index[gene]
        x = rna.X[:, idx]
        if sparse.issparse(x):
            x = x.toarray().reshape(-1)
        else:
            x = np.asarray(x).reshape(-1)
        expr = np.log1p(np.maximum(x, 0))
        expr_grid = dense_to_grid(rna.obs, expr)
        expr_norm = normalize01(expr_grid)
        old_mask_raw, c_low, c_high = fit_two_state_mask(expr_norm, n_iter=25)
        old_mask = majority_regularize(old_mask_raw, n_rounds=3, threshold=5)
        core = make_core_state(expr_grid)
        overlap = float(((old_mask > 0) & (core["core_mask"] > 0)).sum() / (((old_mask > 0) | (core["core_mask"] > 0)).sum() + 1e-8))
        summaries.append({
            "gene": gene,
            "old_mask_area": int(old_mask.sum()),
            "core_mask_area": int(core["core_mask"].sum()),
            "area_ratio_core_vs_old": float(core["core_mask"].sum() / max(old_mask.sum(), 1)),
            "mask_jaccard": overlap,
            "old_center_low": c_low,
            "old_center_high": c_high,
            "core_component_sizes": ",".join(map(str, core["kept_component_sizes"])) if core["kept_component_sizes"] else "",
            "core_mu_low_smooth": float(core["mu_low"]),
            "core_mu_high_smooth": float(core["mu_high"]),
        })
        results.append((gene, expr_grid, old_mask, core["posterior"], core["core_mask"]))

    pd = __import__("pandas")
    pd.DataFrame(summaries).to_csv(OUT_TSV, sep="\t", index=False)

    fig, axes = plt.subplots(len(GENES), 4, figsize=(10.8, 4.0 * len(GENES)), constrained_layout=False)
    axes = np.atleast_2d(axes)
    for r, (gene, expr_grid, old_mask, post, core_mask) in enumerate(results):
        expr_vmax = robust_vmax(expr_grid)
        add_panel(axes[r, 0], expr_grid, f"{gene} RNA", RNA_CMAP, vmin=0, vmax=expr_vmax, label="AB"[r] if r < 2 else None)
        add_panel(axes[r, 1], old_mask, f"{gene} old active mask\narea={int(old_mask.sum())}", MASK_CMAP, vmin=0, vmax=1)
        add_panel(axes[r, 2], post, f"{gene} core posterior", POST_CMAP, vmin=0, vmax=1)
        add_panel(axes[r, 3], core_mask, f"{gene} core-state mask\narea={int(core_mask.sum())}", MASK_CMAP, vmin=0, vmax=1)

    fig.subplots_adjust(left=0.04, right=0.92, top=0.93, bottom=0.08, wspace=0.10, hspace=0.18)
    cax_rna = fig.add_axes([0.93, 0.72, 0.012, 0.16])
    cax_post = fig.add_axes([0.93, 0.36, 0.012, 0.24])
    cb_rna = mpl.colorbar.ColorbarBase(cax_rna, cmap=RNA_CMAP, norm=mpl.colors.Normalize(0, 1), orientation="vertical")
    cb_rna.set_label("RNA (relative)", fontsize=6.0)
    cb_rna.ax.tick_params(labelsize=5.8, length=2)
    cb_post = mpl.colorbar.ColorbarBase(cax_post, cmap=POST_CMAP, norm=mpl.colors.Normalize(0, 1), orientation="vertical")
    cb_post.set_label("core posterior", fontsize=6.0)
    cb_post.ax.tick_params(labelsize=5.8, length=2)
    fig.suptitle(
        "NEFM / NEFL active-state versus core-state masks",
        x=0.04, ha="left", fontsize=11, fontweight="bold",
    )
    fig.text(
        0.04, 0.025,
        "Core-state mask: Gaussian smoothing, unsupervised 2-component mixture, posterior >= 0.9, remove small components, keep largest 1–2 domains. "
        "This is designed to suppress diffuse low-level tails while preserving the dominant expression core.",
        fontsize=5.8, color="#444444",
    )
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    rna.file.close()


if __name__ == "__main__":
    main()
