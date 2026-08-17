import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import anndata as ad
import numpy as np
import pandas as pd
from scipy.ndimage import label

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.io import read_feature_matrix
from sparegvision.gsps import moran_i
from sparegvision.metrics import pearson, spearman, hotspot_dice, gradient_similarity, ssim


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
ATAC_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad"
LINK_PATH = ROOT / "data" / "region_to_gene_adj.tsv"
OUT_TSV = ROOT / "plot" / "NEFM_region_to_gene_adj_filtered_spatial_difference_scores.tsv"

GENE = "NEFM"
EPS = 1e-3
EXCLUDE_PANEL_IDS = {
    "E13", "E14", "E29", "E30", "E32", "E34", "E35", "E36",
    "E38", "E39", "E40", "E41", "E42", "E43", "E44", "E45",
}


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


def component_stats(score_grid, valid_mask, q=0.9):
    vals = np.abs(score_grid[valid_mask])
    if vals.size == 0 or np.all(~np.isfinite(vals)):
        return 0, 0.0, 0.0
    thr = float(np.nanquantile(vals, q))
    hot = valid_mask & np.isfinite(score_grid) & (np.abs(score_grid) >= thr)
    if hot.sum() == 0:
        return 0, 0.0, thr
    labels, n = label(hot, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return 0, 0.0, thr
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max()) if sizes.size else 0
    return int(n), float(largest / max(hot.sum(), 1)), thr


def finite_corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    xv = x[m]
    yv = y[m]
    if np.std(xv) < 1e-8 or np.std(yv) < 1e-8:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def positive(x):
    return float(max(float(x), 0.0))


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
    gene_norm0 = normalize01(expr_grid)
    raw_mask, _, _ = fit_two_state_mask(gene_norm0, n_iter=25)
    latent_mask = majority_regularize(raw_mask, n_rounds=3, threshold=5)
    control_grid = expr_grid * latent_mask.astype(float)
    control_norm, control_p99 = per_panel_p99_normalize(control_grid, percentile=99.0)

    coords = atac.obsm["spatial"]
    row_idx = atac.obs["grid_row"].to_numpy(dtype=int)
    col_idx = atac.obs["grid_col"].to_numpy(dtype=int)

    enhancer_vals = read_feature_matrix(atac, enhancers, obs_names=atac.obs_names, batch_size=64)
    enhancer_vals = np.log1p(np.maximum(enhancer_vals, 0))

    rows = []
    valid_template = np.isfinite(control_norm) & (latent_mask > 0)
    control_spot = control_norm[row_idx, col_idx]
    valid_spot_mask = valid_template[row_idx, col_idx]

    for i, enh in enumerate(enhancers):
        raw_grid = dense_to_grid(atac.obs, enhancer_vals[:, i])
        latent_grid = latent_state_select(raw_grid, latent_mask)
        enh_norm, enh_p99 = per_panel_p99_normalize(latent_grid, percentile=99.0)

        valid = valid_template & np.isfinite(enh_norm)
        if valid.sum() == 0:
            continue

        diff_grid = np.full_like(enh_norm, np.nan, dtype=float)
        diff_grid[valid] = enh_norm[valid] - control_norm[valid]

        logfc_grid = np.full_like(enh_norm, np.nan, dtype=float)
        logfc_grid[valid] = np.log2((enh_norm[valid] + EPS) / (control_norm[valid] + EPS))

        # residual after linear fit enhancer ~ control
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

        enh_spot = enh_norm[row_idx, col_idx]
        spot_valid = valid_spot_mask & np.isfinite(enh_spot) & np.isfinite(control_spot)
        spot_coords = coords[spot_valid]
        diff_spot = enh_spot[spot_valid] - control_spot[spot_valid]
        absdiff_spot = np.abs(diff_spot)
        logfc_spot = np.log2((enh_spot[spot_valid] + EPS) / (control_spot[spot_valid] + EPS))
        resid_spot = enh_spot[spot_valid] - (intercept + slope * control_spot[spot_valid])
        absresid_spot = np.abs(resid_spot)

        n_hot_diff, largest_hot_diff, diff_thr = component_stats(diff_grid, valid, q=0.9)
        n_hot_logfc, largest_hot_logfc, logfc_thr = component_stats(logfc_grid, valid, q=0.9)
        n_hot_resid, largest_hot_resid, resid_thr = component_stats(residual_grid, valid, q=0.9)

        similarity_mask = valid
        spatial_corr = pearson(control_norm, enh_norm, similarity_mask)
        spatial_spearman = spearman(control_norm, enh_norm, similarity_mask)
        spatial_ssim = ssim(control_norm, enh_norm, similarity_mask)
        spatial_grad = gradient_similarity(control_norm, enh_norm, similarity_mask)
        hotspot_overlap = hotspot_dice(control_norm, enh_norm, similarity_mask, quantile=0.8)

        diff_structure_score = (
            np.nanmean(np.abs(diff_vals))
            * (1.0 + positive(moran_i(absdiff_spot, spot_coords, k=6)))
            * largest_hot_diff
        )
        logfc_structure_score = (
            np.nanmean(np.abs(logfc_vals))
            * (1.0 + positive(moran_i(np.abs(logfc_spot), spot_coords, k=6)))
            * largest_hot_logfc
        )
        residual_structure_score = (
            np.nanstd(resid_vals)
            * (1.0 + positive(moran_i(absresid_spot, spot_coords, k=6)))
            * largest_hot_resid
        )
        difference_structure_score = float(
            0.4 * diff_structure_score
            + 0.25 * logfc_structure_score
            + 0.35 * residual_structure_score
        )

        rows.append({
            "gene": GENE,
            "panel_id": nefm_links.loc[i, "panel_id"],
            "enhancer": enh,
            "importance": float(nefm_links.loc[i, "importance"]),
            "initial_rho": float(nefm_links.loc[i, "rho"]),
            "importance_x_abs_rho": float(nefm_links.loc[i, "importance_x_abs_rho"]),
            "distance": nefm_links.loc[i, "Distance"],
            "control_p99": control_p99,
            "enhancer_p99": enh_p99,
            "spatial_corr": spatial_corr,
            "spatial_spearman": spatial_spearman,
            "spatial_ssim": spatial_ssim,
            "spatial_gradient_similarity": spatial_grad,
            "hotspot_overlap_q80": hotspot_overlap,
            "mean_abs_diff": float(np.nanmean(np.abs(diff_vals))),
            "mean_diff": float(np.nanmean(diff_vals)),
            "std_diff": float(np.nanstd(diff_vals)),
            "diff_abs_moran_i": float(moran_i(absdiff_spot, spot_coords, k=6)),
            "diff_hotspots_q90": n_hot_diff,
            "diff_largest_hotspot_fraction": largest_hot_diff,
            "diff_abs_threshold_q90": diff_thr,
            "mean_abs_logfc": float(np.nanmean(np.abs(logfc_vals))),
            "mean_logfc": float(np.nanmean(logfc_vals)),
            "std_logfc": float(np.nanstd(logfc_vals)),
            "logfc_abs_moran_i": float(moran_i(np.abs(logfc_spot), spot_coords, k=6)),
            "logfc_hotspots_q90": n_hot_logfc,
            "logfc_largest_hotspot_fraction": largest_hot_logfc,
            "logfc_abs_threshold_q90": logfc_thr,
            "residual_slope": float(slope),
            "residual_intercept": float(intercept),
            "residual_mean": float(np.nanmean(resid_vals)),
            "residual_std": float(np.nanstd(resid_vals)),
            "residual_abs_moran_i": float(moran_i(absresid_spot, spot_coords, k=6)),
            "residual_hotspots_q90": n_hot_resid,
            "residual_largest_hotspot_fraction": largest_hot_resid,
            "residual_abs_threshold_q90": resid_thr,
            "diff_structure_score": float(diff_structure_score),
            "logfc_structure_score": float(logfc_structure_score),
            "residual_structure_score": float(residual_structure_score),
            "difference_structure_score": difference_structure_score,
        })

    out = pd.DataFrame(rows).sort_values(
        ["difference_structure_score", "residual_structure_score", "diff_structure_score"],
        ascending=False,
    ).reset_index(drop=True)
    out.to_csv(OUT_TSV, sep="\t", index=False)
    print(out.head(15).to_csv(sep="\t", index=False))
    rna.file.close()
    atac.file.close()


if __name__ == "__main__":
    main()
