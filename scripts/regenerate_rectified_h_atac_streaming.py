import json
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
ATAC_PATH = ROOT / "data" / "selected_8k" / "GW12_spatial_ATAC_8065spots.h5ad"
SPOTS_PATH = ROOT / "data" / "rectified_h" / "rectified_spots.tsv"
OUT_DIR = ROOT / "data" / "rectified_h"
OUT_H5AD = OUT_DIR / "GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad"
OUT_META = OUT_DIR / "README_ATAC.json"

ROW_CHUNK = 512
VAR_CHUNK = 32768


def resolve_source_indices(atac_obs_names: np.ndarray, rectified_spots: pd.DataFrame):
    obs_to_idx = {name: idx for idx, name in enumerate(atac_obs_names)}
    rows = np.full((len(rectified_spots), 3), -1, dtype=np.int32)
    for k in range(3):
        names = rectified_spots[f"source_obs_name_{k}"].astype(str).to_numpy()
        rows[:, k] = np.array([obs_to_idx.get(name, -1) for name in names], dtype=np.int32)
    weights = rectified_spots[["weight_0", "weight_1", "weight_2"]].to_numpy(dtype=np.float32)
    return rows, weights


def prepare_skeleton(
    obs: pd.DataFrame,
    var: pd.DataFrame,
    coords: np.ndarray,
    temp_path: Path,
    n_source_spots: int,
):
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    zero = sp.csr_matrix((len(obs), len(var)), dtype=np.float32)
    rectified = ad.AnnData(X=zero, obs=obs.copy(), var=var.copy(), obsm={"spatial": coords.astype(np.float32)})
    rectified.uns["rectification"] = {
        "method": "H",
        "description": "2D warp using row/column boundaries, then linear resampling with nearest fallback to a 101x76 grid",
        "source_h5ad": str(ATAC_PATH.relative_to(ROOT)),
        "weights_from": str(SPOTS_PATH.relative_to(ROOT)),
        "grid_shape": [76, 101],
        "n_source_spots": int(n_source_spots),
        "n_rectified_spots": int(len(obs)),
        "n_peaks": int(len(var)),
        "implementation": "streaming dense chunked weighted remap",
    }
    rectified.write_h5ad(temp_path, compression="gzip")


def _read_weighted_block(source_x, src_idx: np.ndarray, src_w: np.ndarray, var_start: int, var_end: int):
    valid = (src_idx >= 0) & (src_w > 0)
    if not np.any(valid):
        return None, valid
    chosen_idx = src_idx[valid]
    unique_idx, inverse = np.unique(chosen_idx, return_inverse=True)
    src_block = np.asarray(source_x[unique_idx.tolist(), var_start:var_end], dtype=np.float32)
    weighted = src_block[inverse] * src_w[valid][:, None]
    return weighted, valid


def replace_x_dense_weighted(source_path: Path, temp_path: Path, rows: np.ndarray, weights: np.ndarray):
    source = ad.read_h5ad(source_path, backed="r")
    source_x = source.file["X"]
    n_out = rows.shape[0]
    n_vars = source.n_vars

    with h5py.File(temp_path, "a") as target:
        if "X" in target:
            del target["X"]
        x_out = target.create_dataset(
            "X",
            shape=(n_out, n_vars),
            dtype=np.float32,
            chunks=(min(ROW_CHUNK, n_out), min(VAR_CHUNK, n_vars)),
        )
        x_out.attrs["encoding-type"] = "array"
        x_out.attrs["encoding-version"] = "0.2.0"

        for row_start in range(0, n_out, ROW_CHUNK):
            row_end = min(row_start + ROW_CHUNK, n_out)
            row_idx_block = rows[row_start:row_end]
            weight_block = weights[row_start:row_end]
            print(f"rows {row_start}:{row_end} / {n_out}", flush=True)

            for var_start in range(0, n_vars, VAR_CHUNK):
                var_end = min(var_start + VAR_CHUNK, n_vars)
                out_block = np.zeros((row_end - row_start, var_end - var_start), dtype=np.float32)

                for k in range(3):
                    weighted, valid = _read_weighted_block(
                        source_x,
                        row_idx_block[:, k],
                        weight_block[:, k],
                        var_start,
                        var_end,
                    )
                    if weighted is not None:
                        out_block[valid] += weighted

                x_out[row_start:row_end, var_start:var_end] = out_block

    source.file.close()


def build_metadata(n_source_spots: int, n_rectified_spots: int, n_peaks: int):
    return {
        "method": "H",
        "description": "2D warp + linear resampling with nearest fallback",
        "source": str(ATAC_PATH.relative_to(ROOT)),
        "weights_from": str(SPOTS_PATH.relative_to(ROOT)),
        "output_h5ad": str(OUT_H5AD.relative_to(ROOT)),
        "grid_shape": [76, 101],
        "n_source_spots": int(n_source_spots),
        "n_rectified_spots": int(n_rectified_spots),
        "n_peaks": int(n_peaks),
        "implementation": "streaming dense chunked weighted remap",
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rectified_spots = pd.read_csv(SPOTS_PATH, sep="\t", index_col=0)
    atac = ad.read_h5ad(ATAC_PATH, backed="r")

    rows, weights = resolve_source_indices(atac.obs_names.to_numpy(), rectified_spots)
    coords = np.column_stack([
        rectified_spots["grid_col"].to_numpy(),
        rectified_spots["grid_row"].to_numpy(),
    ]).astype(np.float32)
    obs = rectified_spots.copy()
    var = atac.var.copy()

    temp_h5ad = OUT_H5AD.with_suffix(OUT_H5AD.suffix + ".tmp")
    if temp_h5ad.exists():
        temp_h5ad.unlink()

    prepare_skeleton(obs, var, coords, temp_h5ad, atac.n_obs)
    replace_x_dense_weighted(ATAC_PATH, temp_h5ad, rows, weights)
    os.replace(temp_h5ad, OUT_H5AD)

    metadata = build_metadata(atac.n_obs, len(obs), atac.n_vars)
    OUT_META.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    atac.file.close()
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
