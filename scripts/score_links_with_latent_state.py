import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.io import read_feature_matrix


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
ATAC_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad"
LINKS_PATH = Path("/cluster2/huanglab/jiamao/Project/HumanSpinalCord/Work/P3/ATAC/Scenicplus/GW12/outs/region_to_gene_adj.tsv")
DEFAULT_OUT = ROOT / "data" / "GW12_region_to_gene_links_rectified_h_latent_state.tsv"
DEFAULT_META = ROOT / "data" / "GW12_region_to_gene_links_rectified_h_latent_state.metadata.json"
DEFAULT_CACHE = Path("/tmp/rectified_h_atac_log1p_float32_dense.memmap")


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


def corr(a, b):
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    if np.std(aa) < 1e-8 or np.std(bb) < 1e-8:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def corr_vec(y, X):
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    yc = y - y.mean()
    Xc = X - X.mean(axis=0, keepdims=True)
    num = yc @ Xc
    den = np.sqrt((yc ** 2).sum()) * np.sqrt((Xc ** 2).sum(axis=0))
    out = np.full((X.shape[1],), np.nan, dtype=np.float64)
    valid = den > 1e-8
    out[valid] = num[valid] / den[valid]
    return out


def infer_mask_from_gene(rna, gene: str):
    y = read_feature_matrix(rna, [gene], obs_names=rna.obs_names, batch_size=1)[:, 0]
    y = np.log1p(np.maximum(y, 0))
    y_grid = dense_to_grid(rna.obs, y)
    y_norm = normalize01(y_grid)
    raw_mask, c_low, c_high = fit_two_state_mask(y_norm, n_iter=25)
    mask = majority_regularize(raw_mask, n_rounds=3, threshold=5)
    mask_vec = mask[
        rna.obs["grid_row"].to_numpy(dtype=int),
        rna.obs["grid_col"].to_numpy(dtype=int),
    ].astype(np.uint8)
    return y, mask_vec, c_low, c_high


def build_dense_log1p_cache(atac_path: Path, cache_path: Path):
    atac = ad.read_h5ad(atac_path, backed="r")
    n_obs, n_vars = atac.shape
    atac.file.close()
    expected_bytes = n_obs * n_vars * np.dtype(np.float32).itemsize
    if cache_path.exists() and cache_path.stat().st_size == expected_bytes:
        return np.memmap(cache_path, dtype=np.float32, mode="r", shape=(n_obs, n_vars))
    if cache_path.exists():
        cache_path.unlink()
    mm = np.memmap(cache_path, dtype=np.float32, mode="w+", shape=(n_obs, n_vars))
    with h5py.File(atac_path, "r") as f:
        xg = f["X"]
        data = xg["data"]
        indices = xg["indices"]
        indptr = xg["indptr"][:]
        for i in range(n_obs):
            if i % 256 == 0:
                print(f"dense rows {i}:{min(i+256, n_obs)} / {n_obs}", flush=True)
            s = int(indptr[i]); e = int(indptr[i + 1])
            row_idx = indices[s:e]
            row_data = np.log1p(np.maximum(np.asarray(data[s:e], dtype=np.float32), 0))
            if len(row_idx) == n_vars:
                mm[i, :] = row_data
            else:
                mm[i, :] = 0.0
                mm[i, row_idx] = row_data
    mm.flush()
    return mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--metadata", default=str(DEFAULT_META))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--gene-start", type=int, default=0)
    ap.add_argument("--max-genes", type=int, default=None)
    args = ap.parse_args()

    out_path = Path(args.output)
    meta_path = Path(args.metadata)
    cache_path = Path(args.cache)

    links = pd.read_csv(LINKS_PATH, sep="\t")
    links["target"] = links["target"].astype(str)
    links["region"] = links["region"].astype(str)

    rna = ad.read_h5ad(RNA_PATH, backed="r")
    atac = ad.read_h5ad(ATAC_PATH, backed="r")

    valid_genes = set(rna.var_names.astype(str))
    valid_regions = set(atac.var_names.astype(str))
    region_to_col = {r: i for i, r in enumerate(atac.var_names.astype(str))}
    links = links[links["target"].isin(valid_genes) & links["region"].isin(valid_regions)].copy()
    atac.file.close()

    genes = links["target"].drop_duplicates().tolist()
    genes = genes[args.gene_start:]
    if args.max_genes is not None:
        genes = genes[:args.max_genes]
    links = links[links["target"].isin(genes)].copy()

    Xmm = build_dense_log1p_cache(ATAC_PATH, cache_path)

    out_rows = []
    for gi, gene in enumerate(genes, start=1):
        sub = links[links["target"] == gene].copy()
        enhancers = sub["region"].tolist()
        cols = np.array([region_to_col[e] for e in enhancers], dtype=np.int64)
        y, mask_vec, c_low, c_high = infer_mask_from_gene(rna, gene)
        X = np.asarray(Xmm[:, cols], dtype=np.float32)
        active_spots = int(mask_vec.sum())
        active_fraction = float(mask_vec.mean())

        raw_r = corr_vec(y, X)
        latent_X = X * mask_vec[:, None]
        latent_r = corr_vec(y, latent_X)
        raw_sum = X.sum(axis=0)
        latent_sum = latent_X.sum(axis=0)
        retained_fraction = np.divide(latent_sum, np.maximum(raw_sum, 1e-8))

        sub["rectified_raw_rho"] = raw_r
        sub["latent_state_rho"] = latent_r
        sub["rectified_raw_importance_x_rho"] = sub["importance"].to_numpy(float) * raw_r
        sub["rectified_raw_importance_x_abs_rho"] = sub["importance"].to_numpy(float) * np.abs(raw_r)
        sub["latent_importance_x_rho"] = sub["importance"].to_numpy(float) * latent_r
        sub["latent_importance_x_abs_rho"] = sub["importance"].to_numpy(float) * np.abs(latent_r)
        sub["active_spots"] = active_spots
        sub["active_fraction"] = active_fraction
        sub["latent_state_center_inactive"] = c_low
        sub["latent_state_center_active"] = c_high
        sub["raw_signal_sum"] = raw_sum
        sub["latent_signal_sum"] = latent_sum
        sub["latent_signal_retained_fraction"] = retained_fraction
        out_rows.append(sub)

        if gi % 100 == 0:
            print(f"processed genes: {gi} / {len(genes)}", flush=True)

    out = pd.concat(out_rows, ignore_index=True) if out_rows else links.iloc[:0].copy()
    out.to_csv(out_path, sep="\t", index=False)

    meta = {
        "source_links": str(LINKS_PATH),
        "source_rna": str(RNA_PATH.relative_to(ROOT)),
        "source_atac": str(ATAC_PATH.relative_to(ROOT)),
        "output": str(out_path),
        "n_input_rows_after_feature_filter": int(len(links)),
        "n_output_rows": int(len(out)),
        "n_genes": int(len(genes)),
        "cache_path": str(cache_path),
        "method": "gene-conditioned latent-state selection on rectified_h RNA, scored against rectified_h ATAC",
        "new_columns": [
            "rectified_raw_rho",
            "latent_state_rho",
            "rectified_raw_importance_x_rho",
            "rectified_raw_importance_x_abs_rho",
            "latent_importance_x_rho",
            "latent_importance_x_abs_rho",
            "active_spots",
            "active_fraction",
            "latent_state_center_inactive",
            "latent_state_center_active",
            "raw_signal_sum",
            "latent_signal_sum",
            "latent_signal_retained_fraction",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    rna.file.close()
    del Xmm
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
