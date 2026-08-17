import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
ATAC_PATH = ROOT / "data" / "selected_8k" / "GW12_spatial_ATAC_8065spots.h5ad"
SPOTS_PATH = ROOT / "data" / "rectified_h" / "rectified_spots.tsv"
OUT_DIR = ROOT / "data" / "rectified_h"
OUT_H5AD = OUT_DIR / "GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad"
OUT_META = OUT_DIR / "README_ATAC.json"
def resolve_source_indices(atac, rectified_spots):
    obs_names = atac.obs_names.to_numpy()
    obs_to_idx = {name: idx for idx, name in enumerate(obs_names)}
    rows = np.full((len(rectified_spots), 3), -1, dtype=np.int32)
    for k in range(3):
        names = rectified_spots[f"source_obs_name_{k}"].astype(str).to_numpy()
        rows[:, k] = np.array([obs_to_idx.get(name, -1) for name in names], dtype=np.int32)
    weights = rectified_spots[["weight_0", "weight_1", "weight_2"]].to_numpy(dtype=np.float32)
    return rows, weights


def interpolate_rows(source_matrix, rows, weights):
    out_rows = []
    for i in range(rows.shape[0]):
        if i % 500 == 0:
            print(f"row {i} / {rows.shape[0]}", flush=True)
        row_acc = None
        for k in range(3):
            source_idx = int(rows[i, k])
            weight = float(weights[i, k])
            if source_idx < 0 or weight <= 0:
                continue
            src_row = source_matrix[source_idx, :]
            src_row = src_row.astype(np.float32)
            weighted = src_row.multiply(weight) if sp.issparse(src_row) else sp.csr_matrix(src_row * weight)
            row_acc = weighted if row_acc is None else (row_acc + weighted)
        if row_acc is None:
            row_acc = sp.csr_matrix((1, source_matrix.shape[1]), dtype=np.float32)
        out_rows.append(row_acc.tocsr())
    return sp.vstack(out_rows, format="csr")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rectified_spots = pd.read_csv(SPOTS_PATH, sep="\t", index_col=0)
    atac = ad.read_h5ad(ATAC_PATH, backed="r")
    rows, weights = resolve_source_indices(atac, rectified_spots)
    source = atac.raw.X if getattr(atac, "raw", None) is not None else atac.X
    X_rectified = interpolate_rows(source, rows, weights)

    obs = rectified_spots.copy()
    rectified = ad.AnnData(
        X=X_rectified,
        obs=obs,
        var=(atac.raw.var.copy() if getattr(atac, "raw", None) is not None else atac.var.copy()),
        obsm={"spatial": np.column_stack([obs["grid_col"].to_numpy(), obs["grid_row"].to_numpy()]).astype(np.float32)},
    )
    rectified.uns["rectification"] = {
        "method": "H",
        "description": "2D warp using row/column boundaries, then linear resampling with nearest fallback to a 101x76 grid",
        "source_h5ad": str(ATAC_PATH.relative_to(ROOT)),
        "weights_from": str(SPOTS_PATH.relative_to(ROOT)),
        "grid_shape": [76, 101],
        "n_source_spots": int(atac.n_obs),
        "n_rectified_spots": int(rectified.n_obs),
        "n_peaks": int(rectified.n_vars),
    }

    temp_h5ad = OUT_H5AD.with_suffix(OUT_H5AD.suffix + ".tmp")
    if temp_h5ad.exists():
        temp_h5ad.unlink()
    rectified.write_h5ad(temp_h5ad, compression="gzip")
    os.replace(temp_h5ad, OUT_H5AD)

    metadata = {
        "method": "H",
        "description": "2D warp + linear resampling with nearest fallback",
        "source": "data/selected_8k/GW12_spatial_ATAC_8065spots.h5ad",
        "weights_from": "data/rectified_h/rectified_spots.tsv",
        "output_h5ad": str(OUT_H5AD.relative_to(ROOT)),
        "grid_shape": [76, 101],
        "n_source_spots": int(atac.n_obs),
        "n_rectified_spots": int(rectified.n_obs),
        "n_peaks": int(rectified.n_vars),
        "implementation": "row-wise weighted sum from sparse source rows",
    }
    OUT_META.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    atac.file.close()
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
