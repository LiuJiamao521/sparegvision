from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .run_zebrafish_genomewide_complexity import ATAC_PATH, _decode


OUTPUT = Path("results/zebrafish/spatial_domains_v1")
SEED = 20260814
N_DOMAINS = 6
COLORS = ["#315C88", "#248A83", "#D49A45", "#B75B73", "#7B6FA8", "#77945C"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def _projection(ax, coords, labels, dims, title, axis_labels):
    for domain in range(N_DOMAINS):
        mask = labels == domain
        ax.scatter(coords[mask, dims[0]], coords[mask, dims[1]], s=0.55,
                   color=COLORS[domain], alpha=0.78, linewidths=0,
                   rasterized=True, label=f"D{domain}")
        center = np.median(coords[mask][:, dims], axis=0)
        ax.text(center[0], center[1], str(domain), ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white",
                bbox={"boxstyle": "circle,pad=0.18", "facecolor": COLORS[domain],
                      "edgecolor": "white", "linewidth": 0.6})
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(ATAC_PATH, "r") as atac:
        coords_all = np.asarray(atac["obsm"]["spatial_rescaled_z"][:], dtype=float)
        obs_index = atac["obs"].attrs.get("_index", "_index")
        if isinstance(obs_index, bytes):
            obs_index = obs_index.decode()
        obs_names = _decode(atac["obs"][str(obs_index)][:])

    valid = np.isfinite(coords_all).all(axis=1)
    coords = coords_all[valid]
    scaled = (coords - coords.mean(axis=0)) / np.maximum(coords.std(axis=0), 1e-8)
    labels = KMeans(n_clusters=N_DOMAINS, random_state=SEED, n_init=20).fit_predict(scaled)
    sizes = np.bincount(labels, minlength=N_DOMAINS)

    source = pd.DataFrame({
        "cell": np.asarray(obs_names)[valid],
        "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
        "x_standardized": scaled[:, 0], "y_standardized": scaled[:, 1],
        "z_standardized": scaled[:, 2], "domain": labels,
    })
    source.to_csv(OUTPUT / "zebrafish_spatial_domain_labels.tsv", sep="\t", index=False)
    pd.DataFrame({"domain": np.arange(N_DOMAINS), "n_cells": sizes,
                  "fraction": sizes / sizes.sum()}).to_csv(
        OUTPUT / "zebrafish_spatial_domain_sizes.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(9.0, 7.4), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.06, right=0.97, bottom=0.075,
                           top=0.90, wspace=0.16, hspace=0.23)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    for domain in range(N_DOMAINS):
        mask = labels == domain
        ax3d.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                     s=0.42, color=COLORS[domain], alpha=0.65,
                     linewidths=0, rasterized=True)
        center = np.median(coords[mask], axis=0)
        ax3d.text(center[0], center[1], center[2], str(domain), color="white",
                  fontsize=7, fontweight="bold", ha="center", va="center",
                  bbox={"boxstyle": "circle,pad=0.20", "facecolor": COLORS[domain],
                        "edgecolor": "white", "linewidth": 0.6})
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_xlabel("X", labelpad=-6)
    ax3d.set_ylabel("Y", labelpad=-6)
    ax3d.set_zlabel("Z", labelpad=-6)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    ax3d.set_box_aspect(np.ptp(coords, axis=0))
    ax3d.grid(False)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#D8DDE2")
    ax3d.set_title("a   Three-dimensional domain map", loc="left",
                   fontsize=8.5, fontweight="bold")

    ax_xy = fig.add_subplot(grid[0, 1])
    ax_xz = fig.add_subplot(grid[1, 0])
    ax_yz = fig.add_subplot(grid[1, 1])
    _projection(ax_xy, coords, labels, (0, 1), "b   XY projection", ("X", "Y"))
    _projection(ax_xz, coords, labels, (0, 2), "c   XZ projection", ("X", "Z"))
    _projection(ax_yz, coords, labels, (1, 2), "d   YZ projection", ("Y", "Z"))

    handles = [mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=5,
                               color=COLORS[d], label=f"Domain {d}  n={sizes[d]:,}")
               for d in range(N_DOMAINS)]
    fig.legend(handles=handles, ncol=6, loc="upper center", bbox_to_anchor=(0.52, 0.955),
               fontsize=6.5, columnspacing=0.9, handletextpad=0.35)
    fig.suptitle("Zebrafish spatial domains used for regional enhancer analysis",
                 x=0.06, y=0.985, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.06, 0.022,
             "Domains: K-means (k=6, n_init=20, seed=20260814) on standardized 3D spatial_rescaled_z; "
             "22,281 cells with finite coordinates. Labels denote computational spatial partitions, not anatomical annotations.",
             ha="left", va="bottom", fontsize=5.8, color="#586169")
    stem = OUTPUT / "zebrafish_spatial_domains"
    fig.savefig(Path(f"{stem}.png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(f"{stem}.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "dataset": "zebrafish_weMERFISH_6s_E1",
        "n_cells_total": int(len(coords_all)),
        "n_cells_valid_3d": int(valid.sum()),
        "n_cells_excluded": int((~valid).sum()),
        "method": "KMeans on per-axis standardized spatial_rescaled_z",
        "n_domains": N_DOMAINS, "domain_sizes": sizes.tolist(),
        "seed": SEED, "n_init": 20,
        "figure_contract": "3D overview plus XY/XZ/YZ orthogonal projections",
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
