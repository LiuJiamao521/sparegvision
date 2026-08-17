import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import Delaunay, cKDTree


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "selected_rectangle" / "GW12_spatial_RNA_rectangle_6863spots.h5ad"
OUT_DIR = ROOT / "data" / "rectified_h"
OUT_H5AD = OUT_DIR / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
OUT_SPOTS = OUT_DIR / "rectified_spots.tsv"
OUT_META = OUT_DIR / "README.json"
CHUNK_SIZE = 512


def fill_1d_missing(values):
    values = np.asarray(values, dtype=float)
    idx = np.arange(values.size, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        raise ValueError("Boundary series has no valid values")
    if valid.sum() == 1:
        return np.full(values.shape, values[valid][0], dtype=float)
    return np.interp(idx, idx[valid], values[valid])


def compute_row_col_bounds(index_grid):
    n_rows, n_cols = index_grid.shape
    left = np.full(n_rows, np.nan, dtype=float)
    right = np.full(n_rows, np.nan, dtype=float)
    top = np.full(n_cols, np.nan, dtype=float)
    bottom = np.full(n_cols, np.nan, dtype=float)
    for row_idx in range(n_rows):
        valid_cols = np.flatnonzero(index_grid[row_idx] >= 0)
        if valid_cols.size:
            left[row_idx] = float(valid_cols.min())
            right[row_idx] = float(valid_cols.max())
    for col_idx in range(n_cols):
        valid_rows = np.flatnonzero(index_grid[:, col_idx] >= 0)
        if valid_rows.size:
            top[col_idx] = float(valid_rows.min())
            bottom[col_idx] = float(valid_rows.max())
    return fill_1d_missing(left), fill_1d_missing(right), fill_1d_missing(top), fill_1d_missing(bottom)


def build_warp_geometry(coords):
    x_unique = np.sort(np.unique(coords[:, 0]))
    y_unique = np.sort(np.unique(coords[:, 1]))
    x_to_idx = {value: idx for idx, value in enumerate(x_unique)}
    y_to_idx = {value: idx for idx, value in enumerate(y_unique)}
    index_grid = np.full((len(y_unique), len(x_unique)), -1, dtype=int)
    for obs_idx, (x, y) in enumerate(coords):
        index_grid[y_to_idx[y], x_to_idx[x]] = obs_idx
    if index_grid.shape != (76, 101):
        raise ValueError(f"Expected index grid shape (76, 101), got {index_grid.shape}")

    left, right, top, bottom = compute_row_col_bounds(index_grid)
    warped_points = []
    source_obs_idx = []
    for row_idx in range(index_grid.shape[0]):
        valid_cols = np.flatnonzero(index_grid[row_idx] >= 0)
        for col_idx in valid_cols:
            row_span = max(right[row_idx] - left[row_idx], 1.0)
            col_span = max(bottom[col_idx] - top[col_idx], 1.0)
            u = (col_idx - left[row_idx]) / row_span * (index_grid.shape[1] - 1)
            v = (row_idx - top[col_idx]) / col_span * (index_grid.shape[0] - 1)
            warped_points.append([u, v])
            source_obs_idx.append(index_grid[row_idx, col_idx])

    warped_points = np.asarray(warped_points, dtype=np.float64)
    source_obs_idx = np.asarray(source_obs_idx, dtype=np.int32)
    target_x, target_y = np.meshgrid(
        np.arange(index_grid.shape[1], dtype=np.float64),
        np.arange(index_grid.shape[0], dtype=np.float64),
        indexing="xy",
    )
    target_points = np.column_stack([target_x.ravel(), target_y.ravel()])
    return index_grid.shape, warped_points, source_obs_idx, target_points


def build_linear_weights(warped_points, source_obs_idx, target_points):
    tri = Delaunay(warped_points)
    simplex = tri.find_simplex(target_points)
    rows = np.full((target_points.shape[0], 3), -1, dtype=np.int32)
    weights = np.zeros((target_points.shape[0], 3), dtype=np.float32)

    inside = simplex >= 0
    inside_idx = np.flatnonzero(inside)
    inside_simplex = simplex[inside]
    transform = tri.transform[inside_simplex]
    delta = target_points[inside] - transform[:, 2, :]
    bary_xy = np.einsum("nij,nj->ni", transform[:, :2, :], delta)
    bary = np.column_stack([bary_xy, 1.0 - bary_xy.sum(axis=1)])
    vertices = tri.simplices[inside_simplex]
    rows[inside_idx] = source_obs_idx[vertices]
    weights[inside_idx] = bary.astype(np.float32)

    outside_idx = np.flatnonzero(~inside)
    if outside_idx.size:
        tree = cKDTree(warped_points)
        _, nearest = tree.query(target_points[outside_idx], k=1)
        rows[outside_idx, 0] = source_obs_idx[nearest]
        weights[outside_idx, 0] = 1.0

    return rows, weights


def read_sparse_chunk(rna, start, end):
    chunk = rna[:, start:end].X
    if hasattr(chunk, "to_memory"):
        chunk = chunk.to_memory()
    if sp.issparse(chunk):
        return chunk.tocsr()
    return sp.csr_matrix(np.asarray(chunk))


def interpolate_chunk(chunk_dense, rows, weights):
    output = np.zeros((rows.shape[0], chunk_dense.shape[1]), dtype=np.float32)
    for k in range(3):
        valid = weights[:, k] > 0
        if not np.any(valid):
            continue
        output[valid] += weights[valid, k, None] * chunk_dense[rows[valid, k]]
    return sp.csr_matrix(output)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rna = ad.read_h5ad(RNA_PATH, backed="r")
    coords = np.asarray(rna.obsm["spatial"])

    grid_shape, warped_points, source_obs_idx, target_points = build_warp_geometry(coords)
    rows, weights = build_linear_weights(warped_points, source_obs_idx, target_points)

    chunks = []
    for start in range(0, rna.n_vars, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, rna.n_vars)
        chunk = read_sparse_chunk(rna, start, end)
        chunk_dense = chunk.toarray().astype(np.float32, copy=False)
        chunks.append(interpolate_chunk(chunk_dense, rows, weights))
    X_rectified = sp.hstack(chunks, format="csr")

    target_rows = target_points[:, 1].astype(int)
    target_cols = target_points[:, 0].astype(int)
    obs = pd.DataFrame({
        "grid_row": target_rows,
        "grid_col": target_cols,
        "source_obs_index_0": rows[:, 0],
        "source_obs_index_1": rows[:, 1],
        "source_obs_index_2": rows[:, 2],
        "weight_0": weights[:, 0],
        "weight_1": weights[:, 1],
        "weight_2": weights[:, 2],
        "source_obs_name_0": np.where(rows[:, 0] >= 0, rna.obs_names.to_numpy()[rows[:, 0]], ""),
        "source_obs_name_1": np.where(rows[:, 1] >= 0, rna.obs_names.to_numpy()[np.clip(rows[:, 1], 0, None)], ""),
        "source_obs_name_2": np.where(rows[:, 2] >= 0, rna.obs_names.to_numpy()[np.clip(rows[:, 2], 0, None)], ""),
    }, index=[f"h_r{row:02d}_c{col:03d}" for row, col in zip(target_rows, target_cols)])

    rectified = ad.AnnData(
        X=X_rectified,
        obs=obs,
        var=rna.var.copy(),
        obsm={"spatial": np.column_stack([target_cols, target_rows]).astype(np.float32)},
    )
    rectified.uns["rectification"] = {
        "method": "H",
        "description": "2D warp using row/column boundaries, then linear resampling with nearest fallback to a 101x76 grid",
        "source_h5ad": str(RNA_PATH.relative_to(ROOT)),
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "n_source_spots": int(rna.n_obs),
        "n_rectified_spots": int(rectified.n_obs),
        "n_genes": int(rectified.n_vars),
    }

    rectified.write_h5ad(OUT_H5AD, compression="gzip")
    obs.to_csv(OUT_SPOTS, sep="\t", index=True)

    metadata = {
        "method": "H",
        "description": "2D warp + linear resampling with nearest fallback",
        "source": "data/selected_rectangle/GW12_spatial_RNA_rectangle_6863spots.h5ad",
        "output_h5ad": str(OUT_H5AD.relative_to(ROOT)),
        "output_spots": str(OUT_SPOTS.relative_to(ROOT)),
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "n_source_spots": int(rna.n_obs),
        "n_rectified_spots": int(rectified.n_obs),
        "n_genes": int(rectified.n_vars),
        "chunk_size": CHUNK_SIZE,
    }
    OUT_META.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    rna.file.close()
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
