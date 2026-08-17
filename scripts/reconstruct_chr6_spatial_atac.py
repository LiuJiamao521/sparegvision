from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import sparse


BASE_URL = "https://single-cell-data-yinan.s3.us-west-2.amazonaws.com/6s/atac"
GENES_URL = f"{BASE_URL}/genes.csv"
CHROM = "6"
START = 26_800_000
END = 27_000_000
TARGET = "6-26925631-26926131"

ROOT = Path("/cluster2/huanglab/jiamao/Project/SpaRegVision")
DATA_DIR = ROOT / "data" / "weMERFISH"
INPUT_H5AD = DATA_DIR / "weMERFISH_combined_C_6s_E1_rescaled_z.h5ad"
OUT_DIR = DATA_DIR / "chr6_atac_recon"


def fetch_text(url: str) -> str:
    response = requests.get(url, verify=False, timeout=120)
    response.raise_for_status()
    return response.text


def parse_intervals(text: str) -> list[str]:
    values = [x for x in text.split(",") if x and x != "genes"]
    kept = []
    for value in values:
        match = re.match(r"^(\d+)-(\d+)-(\d+)$", value)
        if not match:
            continue
        chrom, start, end = match.groups()
        start_i = int(start)
        end_i = int(end)
        if chrom == CHROM and start_i >= START and end_i <= END:
            kept.append(value)
    return kept


def fetch_interval_vector(interval: str, expected_cells: int) -> np.ndarray:
    text = fetch_text(f"{BASE_URL}/{interval}.csv")
    values = [x for x in text.split(",") if x]
    if not values or values[0] != interval:
        raise ValueError(f"Unexpected header for {interval}: {values[:3]}")
    vector = np.asarray(values[1:], dtype=np.float32)
    if vector.shape[0] != expected_cells:
        raise ValueError(
            f"{interval} has {vector.shape[0]} cells, expected {expected_cells}"
        )
    return vector


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(INPUT_H5AD)
    spatial = adata.obsm["spatial"].copy()
    cell_names = adata.obs_names.astype(str).to_numpy()
    n_cells = adata.n_obs

    intervals = parse_intervals(fetch_text(GENES_URL))
    if TARGET not in intervals:
        raise ValueError(f"Target interval {TARGET} not found in window")

    matrix = np.empty((n_cells, len(intervals)), dtype=np.float32)
    for idx, interval in enumerate(intervals):
        matrix[:, idx] = fetch_interval_vector(interval, n_cells)

    var = pd.DataFrame(index=[f"chr{interval.replace('-', ':', 1).replace('-', '-')}" for interval in intervals])
    var["interval_id"] = intervals
    parsed = [re.match(r"^(\d+)-(\d+)-(\d+)$", interval).groups() for interval in intervals]
    var["chrom"] = [f"chr{x[0]}" for x in parsed]
    var["start"] = [int(x[1]) for x in parsed]
    var["end"] = [int(x[2]) for x in parsed]
    var["width"] = var["end"] - var["start"]

    obs = adata.obs.copy()
    atac_adata = ad.AnnData(
        X=sparse.csr_matrix(matrix),
        obs=obs,
        var=var,
        obsm={"spatial": spatial, "spatial_rescaled_z": adata.obsm["spatial_rescaled_z"].copy()},
        uns={
            "source": "MERFISHEYES public S3 6s/atac",
            "window": "chr6:26800000-27000000",
            "value_type": "imputed accessibility over 500 bp bins",
        },
    )
    atac_adata.obs_names = cell_names

    out_h5ad = OUT_DIR / "chr6_26800000_27000000_spatial_atac_6s_E1.h5ad"
    atac_adata.write_h5ad(out_h5ad)

    target_idx = intervals.index(TARGET)
    target_values = matrix[:, target_idx]
    np.save(OUT_DIR / "chr6_26925631_26926131_values.npy", target_values)

    coords = spatial
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    scatter_kwargs = dict(c=target_values, cmap="magma", s=4, linewidths=0)
    axes[0].scatter(coords[:, 1], coords[:, 2], **scatter_kwargs)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Spatial x-y")
    axes[0].set_aspect("equal")

    sc = axes[1].scatter(coords[:, 1], coords[:, 0], **scatter_kwargs)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    axes[1].set_title("Spatial x-z")
    axes[1].set_aspect("equal")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Imputed accessibility")
    fig.suptitle("chr6:26925631-26926131 in 6s E1 spatial cells")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chr6_26925631_26926131_spatial.png", bbox_inches="tight")
    plt.close(fig)

    summary = OUT_DIR / "README.txt"
    summary.write_text(
        "\n".join(
            [
                "Local reconstruction from MERFISHEYES public 6s/atac S3 vectors.",
                f"Cells: {n_cells}",
                f"Intervals in chr6:26800000-27000000: {len(intervals)}",
                f"Target interval: {TARGET}",
                f"H5AD: {out_h5ad.name}",
                "Values represent imputed accessibility over 500 bp bins.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved_h5ad\t{out_h5ad}")
    print(f"n_cells\t{n_cells}")
    print(f"n_intervals\t{len(intervals)}")
    print(f"target_min\t{float(target_values.min())}")
    print(f"target_max\t{float(target_values.max())}")
    print(f"target_mean\t{float(target_values.mean())}")


if __name__ == "__main__":
    main()
