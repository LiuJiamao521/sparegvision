import json
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "GW12_spatial_ATAC.h5ad"
SELECTED_8K_PATH = ROOT / "data" / "selected_8k" / "GW12_spatial_ATAC_8065spots.h5ad"
SELECTED_RECT_PATH = ROOT / "data" / "selected_rectangle" / "GW12_spatial_ATAC_rectangle_6863spots.h5ad"
SELECTED_8K_META = ROOT / "data" / "selected_8k" / "README.json"
SELECTED_RECT_META = ROOT / "data" / "selected_rectangle" / "README.json"
SELECTED_8K_SPOTS = ROOT / "data" / "selected_8k" / "spots.tsv"
SELECTED_RECT_SPOTS = ROOT / "data" / "selected_rectangle" / "spots.tsv"

RECT = {"xmin": 4200.0, "xmax": 7200.0, "ymin": 5100.0, "ymax": 7350.0}
ROW_CHUNK = 128


def prepare_skeleton(obs: pd.DataFrame, var: pd.DataFrame, coords: np.ndarray, temp_path: Path):
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    zero = sp.csr_matrix((len(obs), len(var)), dtype=np.float32)
    subset = ad.AnnData(X=zero, obs=obs.copy(), var=var.copy(), obsm={"spatial": coords.astype(np.float32)})
    subset.write_h5ad(temp_path, compression="gzip")


def replace_x_dense(source_path: Path, temp_path: Path, row_idx: np.ndarray):
    source = ad.read_h5ad(source_path, backed="r")
    source_x = source.file["X"]
    with h5py.File(temp_path, "a") as target:
        if "X" in target:
            del target["X"]
        x_out = target.create_dataset(
            "X",
            shape=(len(row_idx), source.n_vars),
            dtype=source_x.dtype,
            compression="gzip",
            chunks=(min(ROW_CHUNK, len(row_idx)), min(2048, source.n_vars)),
        )
        for key, value in source_x.attrs.items():
            x_out.attrs[key] = value
        for start in range(0, len(row_idx), ROW_CHUNK):
            end = min(start + ROW_CHUNK, len(row_idx))
            x_out[start:end, :] = source_x[row_idx[start:end], :]
    source.file.close()


def write_subset(source: ad.AnnData, row_idx: np.ndarray, output_path: Path):
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    coords = np.asarray(source.obsm["spatial"])[row_idx]
    obs = source.obs.iloc[row_idx].copy()
    prepare_skeleton(obs, source.var, coords, temp_path)
    replace_x_dense(SOURCE_PATH, temp_path, row_idx)
    os.replace(temp_path, output_path)


def main():
    source = ad.read_h5ad(SOURCE_PATH, backed="r")
    coords = np.asarray(source.obsm["spatial"])
    valid_mask = np.isfinite(coords).all(axis=1)
    if "valid_spot" in source.obs:
        valid_mask &= source.obs["valid_spot"].to_numpy(bool)

    valid_idx = np.flatnonzero(valid_mask)
    rect_mask = (
        (coords[:, 0] >= RECT["xmin"]) & (coords[:, 0] <= RECT["xmax"]) &
        (coords[:, 1] >= RECT["ymin"]) & (coords[:, 1] <= RECT["ymax"])
    )
    rect_idx = np.flatnonzero(valid_mask & rect_mask)

    if len(valid_idx) != 8065:
        raise ValueError(f"Expected 8065 valid spots, got {len(valid_idx)}")
    if len(rect_idx) != 6863:
        raise ValueError(f"Expected 6863 rectangle spots, got {len(rect_idx)}")

    write_subset(source, valid_idx, SELECTED_8K_PATH)
    write_subset(source, rect_idx, SELECTED_RECT_PATH)

    valid_spots = pd.DataFrame({
        "spot": source.obs_names[valid_idx].to_numpy(),
        "x": coords[valid_idx, 0],
        "y": coords[valid_idx, 1],
    })
    rect_spots = pd.DataFrame({
        "spot": source.obs_names[rect_idx].to_numpy(),
        "x": coords[rect_idx, 0],
        "y": coords[rect_idx, 1],
    })
    valid_spots.to_csv(SELECTED_8K_SPOTS, sep="\t", index=False)
    rect_spots.to_csv(SELECTED_RECT_SPOTS, sep="\t", index=False)

    selected_8k_meta = {
        "n_spots": int(len(valid_idx)),
        "selection": "valid_spot == True in current full ATAC AnnData object",
        "source_atac": str(SOURCE_PATH.relative_to(ROOT)),
        "coordinate_key": 'obsm["spatial"]',
        "coordinate_range": {
            "xmin": float(coords[valid_idx, 0].min()),
            "xmax": float(coords[valid_idx, 0].max()),
            "ymin": float(coords[valid_idx, 1].min()),
            "ymax": float(coords[valid_idx, 1].max()),
        },
        "output_h5ad": str(SELECTED_8K_PATH.relative_to(ROOT)),
    }
    SELECTED_8K_META.write_text(json.dumps(selected_8k_meta, indent=2, ensure_ascii=False) + "\n")

    selected_rect_meta = {
        "n_spots": int(len(rect_idx)),
        "selection": "valid_spot == True and x in [4200,7200], y in [5100,7350] on current full ATAC AnnData object",
        "rectangle": RECT,
        "source_atac": str(SOURCE_PATH.relative_to(ROOT)),
        "source_selected_8k": str(SELECTED_8K_PATH.relative_to(ROOT)),
        "output_h5ad": str(SELECTED_RECT_PATH.relative_to(ROOT)),
    }
    SELECTED_RECT_META.write_text(json.dumps(selected_rect_meta, indent=2, ensure_ascii=False) + "\n")

    source.file.close()
    print(json.dumps({
        "source": str(SOURCE_PATH.relative_to(ROOT)),
        "selected_8k": str(SELECTED_8K_PATH.relative_to(ROOT)),
        "selected_rectangle": str(SELECTED_RECT_PATH.relative_to(ROOT)),
        "n_valid_spots": int(len(valid_idx)),
        "n_rectangle_spots": int(len(rect_idx)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
