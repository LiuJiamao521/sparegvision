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
OUT_ENH_TSV = ROOT / "plot" / "gene_subset_rectified_h_enhancer_divergence_test.tsv"
OUT_GENE_TSV = ROOT / "plot" / "gene_subset_rectified_h_gene_divergence_test.tsv"
ATAC_CACHE = Path("/tmp/rectified_h_atac_log1p_float32_dense_test.memmap")

GENES = ["NEFM", "NEFL", "GFAP", "SHH"]
EPS = 1e-3
INITIAL_RHO_THRESHOLD = 0.0
RECTIFIED_RAW_RHO_THRESHOLD = 0.1
MIN_ENHANCERS_FOR_GENE_RANK = 5


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


def positive(x):
    return float(max(float(x), 0.0))


def load_atac_cache(atac):
    shape = (atac.n_obs, atac.n_vars)
    if not ATAC_CACHE.exists():
        raise FileNotFoundError(f"ATAC cache missing: {ATAC_CACHE}")
    mm = np.memmap(ATAC_CACHE, dtype=np.float32, mode="r", shape=shape)
    var_index = pd.Index(atac.var_names.astype(str))
    return mm, var_index


def score_one_enhancer(control_norm, latent_mask, coords, row_idx, col_idx, raw_grid, link_row):
    latent_grid = latent_state_select(raw_grid, latent_mask)
    enh_norm, enh_p99 = per_panel_p99_normalize(latent_grid, percentile=99.0)
    valid_template = np.isfinite(control_norm) & (latent_mask > 0)
    valid = valid_template & np.isfinite(enh_norm)
    if valid.sum() == 0:
        return None

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

    diff_hot_n, diff_hot_frac, diff_thr = component_stats(diff_grid, valid, q=0.9)
    logfc_hot_n, logfc_hot_frac, logfc_thr = component_stats(logfc_grid, valid, q=0.9)
    resid_hot_n, resid_hot_frac, resid_thr = component_stats(residual_grid, valid, q=0.9)

    spatial_corr = pearson(control_norm, enh_norm, valid)
    spatial_spearman = spearman(control_norm, enh_norm, valid)
    spatial_ssim = ssim(control_norm, enh_norm, valid)
    spatial_grad = gradient_similarity(control_norm, enh_norm, valid)
    hotspot_overlap = hotspot_dice(control_norm, enh_norm, valid, quantile=0.8)

    diff_structure_score = (
        np.nanmean(np.abs(diff_vals))
        * (1.0 + positive(moran_i(absdiff_spot, spot_coords, k=6)))
        * diff_hot_frac
    )
    logfc_structure_score = (
        np.nanmean(np.abs(logfc_vals))
        * (1.0 + positive(moran_i(np.abs(logfc_spot), spot_coords, k=6)))
        * logfc_hot_frac
    )
    residual_structure_score = (
        np.nanstd(resid_vals)
        * (1.0 + positive(moran_i(absresid_spot, spot_coords, k=6)))
        * resid_hot_frac
    )
    difference_structure_score = float(
        0.4 * diff_structure_score
        + 0.25 * logfc_structure_score
        + 0.35 * residual_structure_score
    )

    out = {
        "panel_id": link_row["panel_id"],
        "enhancer": link_row["region"],
        "importance": float(link_row["importance"]),
        "initial_rho": float(link_row["rho"]),
        "importance_x_abs_rho": float(link_row["importance_x_abs_rho"]),
        "distance": link_row["Distance"],
        "enhancer_p99": enh_p99,
        "spatial_corr": spatial_corr,
        "spatial_spearman": spatial_spearman,
        "spatial_ssim": spatial_ssim,
        "spatial_gradient_similarity": spatial_grad,
        "hotspot_overlap_q80": hotspot_overlap,
        "rectified_raw_rho": float(pearson(dense_to_grid_coords(raw_grid, valid_template), dense_to_grid_coords(control_norm, valid_template), valid_template)) if False else np.nan,
        "mean_abs_diff": float(np.nanmean(np.abs(diff_vals))),
        "mean_diff": float(np.nanmean(diff_vals)),
        "std_diff": float(np.nanstd(diff_vals)),
        "diff_abs_moran_i": float(moran_i(absdiff_spot, spot_coords, k=6)),
        "diff_hotspots_q90": diff_hot_n,
        "diff_largest_hotspot_fraction": diff_hot_frac,
        "diff_abs_threshold_q90": diff_thr,
        "mean_abs_logfc": float(np.nanmean(np.abs(logfc_vals))),
        "mean_logfc": float(np.nanmean(logfc_vals)),
        "std_logfc": float(np.nanstd(logfc_vals)),
        "logfc_abs_moran_i": float(moran_i(np.abs(logfc_spot), spot_coords, k=6)),
        "logfc_hotspots_q90": logfc_hot_n,
        "logfc_largest_hotspot_fraction": logfc_hot_frac,
        "logfc_abs_threshold_q90": logfc_thr,
        "residual_slope": float(slope),
        "residual_intercept": float(intercept),
        "residual_mean": float(np.nanmean(resid_vals)),
        "residual_std": float(np.nanstd(resid_vals)),
        "residual_abs_moran_i": float(moran_i(absresid_spot, spot_coords, k=6)),
        "residual_hotspots_q90": resid_hot_n,
        "residual_largest_hotspot_fraction": resid_hot_frac,
        "residual_abs_threshold_q90": resid_thr,
        "diff_structure_score": float(diff_structure_score),
        "logfc_structure_score": float(logfc_structure_score),
        "residual_structure_score": float(residual_structure_score),
        "difference_structure_score": difference_structure_score,
    }
    return out


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


def aggregate_gene(group, global_threshold):
    scores = group["difference_structure_score"].to_numpy(dtype=float)
    n = len(group)
    topk = int(min(3, n))
    top3_mean = float(np.mean(np.sort(scores)[-topk:])) if n else np.nan
    divergent_fraction = float(np.mean(scores >= global_threshold)) if n else np.nan
    gene_divergence_score = (
        0.4 * float(np.nanmedian(scores))
        + 0.35 * divergent_fraction
        + 0.25 * top3_mean
    )
    return pd.Series({
        "n_enhancers_retained": n,
        "median_difference_structure_score": float(np.nanmedian(scores)),
        "mean_difference_structure_score": float(np.nanmean(scores)),
        "top3_mean_difference_structure_score": top3_mean,
        "max_difference_structure_score": float(np.nanmax(scores)),
        "std_difference_structure_score": float(np.nanstd(scores)),
        "divergent_fraction_top10pct": divergent_fraction,
        "gene_divergence_score": gene_divergence_score,
        "rank_eligible": bool(n >= MIN_ENHANCERS_FOR_GENE_RANK),
    })


def main():
    rna = ad.read_h5ad(RNA_PATH, backed="r")
    atac = ad.read_h5ad(ATAC_PATH, backed="r")
    if not np.array_equal(rna.obs_names.to_numpy(), atac.obs_names.to_numpy()):
        raise ValueError("Rectified RNA and ATAC obs order differs")

    links = pd.read_csv(LINK_PATH, sep="\t")
    coords = atac.obsm["spatial"]
    row_idx = atac.obs["grid_row"].to_numpy(dtype=int)
    col_idx = atac.obs["grid_col"].to_numpy(dtype=int)
    atac_mm, atac_var_index = load_atac_cache(atac)

    enhancer_rows = []
    gene_meta = []

    for gene in GENES:
        sub = (
            links.loc[links["target"].astype(str) == gene]
            .drop_duplicates(subset=["region"])
            .sort_values("importance_x_abs_rho", ascending=False)
            .reset_index(drop=True)
            .copy()
        )
        if sub.empty:
            gene_meta.append({"gene": gene, "n_links_raw": 0, "n_links_pass_filter": 0, "note": "gene absent in region_to_gene_adj.tsv"})
            continue
        sub["panel_id"] = [f"E{i+1}" for i in range(len(sub))]
        enhancers = sub["region"].astype(str).tolist()
        enhancer_idx = atac_var_index.get_indexer(enhancers)
        keep_feature = enhancer_idx >= 0
        if not np.all(keep_feature):
            sub = sub.loc[keep_feature].reset_index(drop=True).copy()
            enhancer_idx = enhancer_idx[keep_feature]
            enhancers = sub["region"].astype(str).tolist()
        if sub.empty:
            gene_meta.append({"gene": gene, "n_links_raw": len(enhancers), "n_links_pass_filter": 0, "note": "no linked peaks found in rectified_h ATAC"})
            continue

        expr = read_feature_matrix(rna, [gene], obs_names=rna.obs_names, batch_size=1)[:, 0]
        expr = np.log1p(np.maximum(expr, 0))
        expr_grid = dense_to_grid(rna.obs, expr)
        gene_norm0 = normalize01(expr_grid)
        raw_mask, c_low, c_high = fit_two_state_mask(gene_norm0, n_iter=25)
        latent_mask = majority_regularize(raw_mask, n_rounds=3, threshold=5)
        control_grid = expr_grid * latent_mask.astype(float)
        control_norm, control_p99 = per_panel_p99_normalize(control_grid, percentile=99.0)

        enhancer_vals = np.asarray(atac_mm[:, enhancer_idx], dtype=np.float32)

        passed = []
        for i, (_, link_row) in enumerate(sub.iterrows()):
            raw_grid = dense_to_grid(atac.obs, enhancer_vals[:, i])
            rectified_raw_rho = corr_valid(raw_grid, expr_grid)
            if float(link_row["rho"]) < INITIAL_RHO_THRESHOLD:
                continue
            if not np.isfinite(rectified_raw_rho) or rectified_raw_rho < RECTIFIED_RAW_RHO_THRESHOLD:
                continue
            scored = score_one_enhancer(control_norm, latent_mask, coords, row_idx, col_idx, raw_grid, link_row)
            if scored is None:
                continue
            scored.update({
                "gene": gene,
                "n_links_raw_gene": len(sub),
                "control_p99": control_p99,
                "mask_low_center": c_low,
                "mask_high_center": c_high,
                "rectified_raw_rho": rectified_raw_rho,
            })
            enhancer_rows.append(scored)
            passed.append(scored)

        gene_meta.append({
            "gene": gene,
            "n_links_raw": len(sub),
            "n_links_pass_filter": len(passed),
            "note": "",
        })

    enh_df = pd.DataFrame(enhancer_rows)
    if enh_df.empty:
        raise RuntimeError("No enhancer passed the standardized filter in the test subset")

    global_threshold = float(np.nanquantile(enh_df["difference_structure_score"], 0.9))
    gene_df = (
        enh_df.groupby("gene", as_index=False)
        .apply(lambda g: aggregate_gene(g, global_threshold))
        .reset_index()
    )
    if "level_1" in gene_df.columns:
        gene_df = gene_df.drop(columns=["level_1"])
    gene_df = gene_df.merge(pd.DataFrame(gene_meta), on="gene", how="outer")
    gene_df["global_divergent_threshold_top10pct"] = global_threshold
    gene_df = gene_df.sort_values(
        ["rank_eligible", "gene_divergence_score", "median_difference_structure_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    enh_df = enh_df.sort_values(["gene", "difference_structure_score"], ascending=[True, False]).reset_index(drop=True)
    enh_df.to_csv(OUT_ENH_TSV, sep="\t", index=False)
    gene_df.to_csv(OUT_GENE_TSV, sep="\t", index=False)

    print("Gene summary:")
    print(gene_df.to_string(index=False))
    print("\nTop enhancers per gene:")
    print(
        enh_df.groupby("gene", as_index=False)
        .head(3)[["gene", "panel_id", "enhancer", "difference_structure_score", "spatial_corr", "mean_abs_diff", "mean_abs_logfc", "residual_std", "rectified_raw_rho", "initial_rho"]]
        .to_string(index=False)
    )

    rna.file.close()
    atac.file.close()


if __name__ == "__main__":
    main()
