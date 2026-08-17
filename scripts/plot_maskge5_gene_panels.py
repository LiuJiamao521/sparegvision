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
from scipy import sparse
from scipy.ndimage import label

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.gsps import moran_i


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
ATAC_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad"
ATAC_CACHE = Path("/tmp/rectified_h_atac_log1p_float32_dense_test.memmap")
LINK_PATH = ROOT / "data" / "region_to_gene_adj.tsv"
GENE_TABLE = ROOT / "plot" / "gene_batch_rectified_h_gene_divergence_absolute_all15038_maskge5.tsv"
OUT_DIR = ROOT / "plot" / "maskge5_gene_divergence_panels"

INITIAL_RHO_THRESHOLD = 0.0
RECTIFIED_RAW_RHO_THRESHOLD = 0.1
EPS = 1e-3

RNA_CMAP = LinearSegmentedColormap.from_list(
    "rna_maskge5", ["#F7F7F7", "#DCEAF7", "#6BAED6", "#2171B5", "#08306B"], N=256
)
ATAC_CMAP = LinearSegmentedColormap.from_list(
    "atac_maskge5",
    ["#EDEDED", "#FFF7F3", "#FDE0DD", "#FCC5C0", "#FA9FB5", "#F768A1", "#DD3497", "#AE017E"],
    N=256,
)
MASK_CMAP = LinearSegmentedColormap.from_list("mask_maskge5", ["#EDEDED", "#0F4D92"], N=2)

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


def component_stats(score_grid, valid_mask, q=0.9):
    vals = np.abs(score_grid[valid_mask])
    if vals.size == 0 or np.all(~np.isfinite(vals)):
        return 0, 0.0
    thr = float(np.nanquantile(vals, q))
    hot = valid_mask & np.isfinite(score_grid) & (np.abs(score_grid) >= thr)
    if hot.sum() == 0:
        return 0, 0.0
    labels, n = label(hot, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return 0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max()) if sizes.size else 0
    return int(n), float(largest / max(hot.sum(), 1))


def positive(x):
    return float(max(float(x), 0.0))


def score_enhancer(control_norm, latent_mask, coords, row_idx, col_idx, raw_grid):
    latent_grid = raw_grid * latent_mask.astype(float)
    enh_norm, _ = per_panel_p99_normalize(latent_grid, percentile=99.0)
    valid_template = np.isfinite(control_norm) & (latent_mask > 0)
    valid = valid_template & np.isfinite(enh_norm)
    if valid.sum() == 0:
        return np.nan

    diff_grid = np.full_like(enh_norm, np.nan, dtype=float)
    diff_grid[valid] = enh_norm[valid] - control_norm[valid]
    logfc_grid = np.full_like(enh_norm, np.nan, dtype=float)
    logfc_grid[valid] = np.log2((enh_norm[valid] + EPS) / (control_norm[valid] + EPS))

    x = control_norm[valid].ravel()
    y = enh_norm[valid].ravel()
    if np.std(x) < 1e-8:
        slope, intercept = 0.0, float(np.mean(y))
    else:
        slope, intercept = np.polyfit(x, y, deg=1)
    pred = intercept + slope * control_norm
    residual_grid = np.full_like(enh_norm, np.nan, dtype=float)
    residual_grid[valid] = enh_norm[valid] - pred[valid]

    diff_vals = diff_grid[valid]
    logfc_vals = logfc_grid[valid]
    resid_vals = residual_grid[valid]

    control_spot = control_norm[row_idx, col_idx]
    enh_spot = enh_norm[row_idx, col_idx]
    valid_spot = valid_template[row_idx, col_idx] & np.isfinite(control_spot) & np.isfinite(enh_spot)
    spot_coords = coords[valid_spot]
    diff_spot = enh_spot[valid_spot] - control_spot[valid_spot]
    absdiff_spot = np.abs(diff_spot)
    logfc_spot = np.log2((enh_spot[valid_spot] + EPS) / (control_spot[valid_spot] + EPS))
    resid_spot = enh_spot[valid_spot] - (intercept + slope * control_spot[valid_spot])
    absresid_spot = np.abs(resid_spot)

    _, diff_hot = component_stats(diff_grid, valid, q=0.9)
    _, logfc_hot = component_stats(logfc_grid, valid, q=0.9)
    _, resid_hot = component_stats(residual_grid, valid, q=0.9)

    diff_structure_score = (
        np.nanmean(np.abs(diff_vals))
        * (1.0 + positive(moran_i(absdiff_spot, spot_coords, k=6)))
        * diff_hot
    )
    logfc_structure_score = (
        np.nanmean(np.abs(logfc_vals))
        * (1.0 + positive(moran_i(np.abs(logfc_spot), spot_coords, k=6)))
        * logfc_hot
    )
    residual_structure_score = (
        np.nanstd(resid_vals)
        * (1.0 + positive(moran_i(absresid_spot, spot_coords, k=6)))
        * resid_hot
    )
    return float(0.4 * diff_structure_score + 0.25 * logfc_structure_score + 0.35 * residual_structure_score)


def robust_vmax(values, percentile=99.5):
    values = np.asarray(values, dtype=float)
    pos = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(pos, percentile)) if pos.size else 1.0


def add_panel(ax, grid, title, cmap, vmin=None, vmax=None, label=None):
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper", aspect="auto")
    ax.set_title(title, loc="left", fontsize=5.8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label:
        ax.text(-0.06, 1.04, label, transform=ax.transAxes, fontsize=7.4, fontweight="bold")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    genes_df = pd.read_csv(GENE_TABLE, sep="\t").sort_values("absolute_rank_maskge5")
    genes = genes_df["gene"].astype(str).tolist()

    rna = ad.read_h5ad(RNA_PATH, backed="r")
    atac = ad.read_h5ad(ATAC_PATH, backed="r")
    links = pd.read_csv(LINK_PATH, sep="\t")
    links["target"] = links["target"].astype(str)
    grouped = {
        gene: grp.drop_duplicates(subset=["region"]).sort_values("importance_x_abs_rho", ascending=False).reset_index(drop=True).copy()
        for gene, grp in links.groupby("target", sort=False)
    }
    rna_var_index = pd.Index(rna.var_names.astype(str))
    atac_var_index = pd.Index(atac.var_names.astype(str))
    atac_mm = np.memmap(ATAC_CACHE, dtype=np.float32, mode="r", shape=(atac.n_obs, atac.n_vars))
    coords = atac.obsm["spatial"]
    row_idx = atac.obs["grid_row"].to_numpy(dtype=int)
    col_idx = atac.obs["grid_col"].to_numpy(dtype=int)

    for rank, gene in enumerate(genes, start=1):
        sub = grouped[gene].copy()
        sub["panel_id"] = [f"E{i+1}" for i in range(len(sub))]

        expr_idx = rna_var_index.get_indexer([gene])[0]
        expr = rna.X[:, expr_idx]
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
        control_norm, _ = per_panel_p99_normalize(control_grid, percentile=99.0)

        enhancers = sub["region"].astype(str).tolist()
        enhancer_idx = atac_var_index.get_indexer(enhancers)
        keep_feature = enhancer_idx >= 0
        sub = sub.loc[keep_feature].reset_index(drop=True).copy()
        enhancer_idx = enhancer_idx[keep_feature]
        enhancer_vals = np.asarray(atac_mm[:, enhancer_idx], dtype=np.float32)

        rows = []
        raw_grids = []
        latent_grids = []
        for i, row in enumerate(sub.itertuples(index=False)):
            raw_grid = dense_to_grid(atac.obs, enhancer_vals[:, i])
            rect_r = corr_valid(raw_grid, expr_grid)
            if float(row.rho) < INITIAL_RHO_THRESHOLD or (not np.isfinite(rect_r)) or rect_r < RECTIFIED_RAW_RHO_THRESHOLD:
                continue
            latent_grid = raw_grid * latent_mask.astype(float)
            lat_r = corr_valid(latent_grid, expr_grid)
            diff_score = score_enhancer(control_norm, latent_mask, coords, row_idx, col_idx, raw_grid)
            rows.append({
                "panel_id": row.panel_id,
                "enhancer": row.region,
                "initial_rho": float(row.rho),
                "rectified_raw_rho": rect_r,
                "latent_state_rho": lat_r,
                "difference_structure_score": diff_score,
                "importance": float(row.importance),
                "distance": row.Distance,
            })
            raw_grids.append(raw_grid)
            latent_grids.append(latent_grid)

        if not rows:
            continue

        order = np.argsort([-r["difference_structure_score"] for r in rows])
        rows = [rows[i] for i in order]
        raw_grids = [raw_grids[i] for i in order]
        latent_grids = [latent_grids[i] for i in order]

        expr_vmax = robust_vmax(expr_grid)
        atac_vmax = robust_vmax(np.stack(raw_grids + latent_grids))

        n_pairs = len(rows)
        ncols = 8
        pair_cols = ncols // 2
        nrows = 1 + int(np.ceil(n_pairs / pair_cols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 2.05 * nrows), constrained_layout=False)
        axes = np.atleast_2d(axes)

        add_panel(axes[0, 0], expr_grid, f"{gene} RNA", RNA_CMAP, vmin=0, vmax=expr_vmax, label="A")
        add_panel(axes[0, 1], latent_mask, f"{gene} latent mask\narea={int(latent_mask.sum())}", MASK_CMAP, vmin=0, vmax=1)
        for c in range(2, ncols):
            axes[0, c].axis("off")

        for idx, item in enumerate(rows):
            rr = 1 + idx // pair_cols
            cc = (idx % pair_cols) * 2
            raw_title = f"{item['panel_id']} raw\ninitρ={item['initial_rho']:.2f}, rectρ={item['rectified_raw_rho']:.2f}"
            lat_title = f"{item['panel_id']} latent\nlatentρ={item['latent_state_rho']:.2f}, diffS={item['difference_structure_score']:.2f}"
            add_panel(axes[rr, cc], raw_grids[idx], raw_title, ATAC_CMAP, vmin=0, vmax=atac_vmax)
            add_panel(axes[rr, cc + 1], latent_grids[idx], lat_title, ATAC_CMAP, vmin=0, vmax=atac_vmax)

        for rr in range(1, nrows):
            for cc in range(ncols):
                pair_index = (rr - 1) * pair_cols + (cc // 2)
                if pair_index >= n_pairs:
                    axes[rr, cc].axis("off")

        fig.subplots_adjust(left=0.03, right=0.945, top=0.92, bottom=0.08, wspace=0.10, hspace=0.26)
        cax_rna = fig.add_axes([0.95, 0.72, 0.012, 0.16])
        cax_atac = fig.add_axes([0.95, 0.36, 0.012, 0.26])
        cb_rna = mpl.colorbar.ColorbarBase(cax_rna, cmap=RNA_CMAP, norm=Normalize(0, expr_vmax), orientation="vertical")
        cb_rna.ax.tick_params(labelsize=5.6, length=2)
        cb_rna.set_label("log1p RNA", fontsize=5.8)
        cb_atac = mpl.colorbar.ColorbarBase(cax_atac, cmap=ATAC_CMAP, norm=Normalize(0, atac_vmax), orientation="vertical")
        cb_atac.ax.tick_params(labelsize=5.6, length=2)
        cb_atac.set_label("log1p ATAC", fontsize=5.8)
        fig.suptitle(
            f"Mask>=5% rank {rank} {gene}: rectified_h gene / latent mask / raw and latent enhancer panels",
            x=0.03, ha="left", fontsize=10.5, fontweight="bold",
        )
        fig.text(
            0.03, 0.03,
            f"Genes were prefiltered by latent mask area >= 5% of the 101×76 canvas. Enhancers were then filtered by initial rho >= {INITIAL_RHO_THRESHOLD:.1f} "
            f"and rectified raw rho >= {RECTIFIED_RAW_RHO_THRESHOLD:.1f}, and ordered by difference_structure_score.",
            fontsize=5.8, color="#444444",
        )

        prefix = OUT_DIR / f"{rank:02d}_{gene}_rectified_h_gene_enhancer_panels"
        fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(rows).to_csv(prefix.with_suffix(".tsv"), sep="\t", index=False)
        print(f"done {rank:02d} {gene} n={len(rows)}")

    rna.file.close()
    atac.file.close()


if __name__ == "__main__":
    main()
