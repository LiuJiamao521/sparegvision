import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.io import read_feature_matrix


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
ATAC_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad"
LINK_PATH = ROOT / "data" / "GW12_region_to_gene_links_filtered.tsv"
OUT_PATH = ROOT / "plot" / "NEFM_requested_enhancers_rectified_h.png"

GENE = "NEFM"
REQUESTED_ENHANCERS = [
    "chr8:25021594-25022095",
    "chr8:25000555-25001056",
    "chr8:25034051-25034552",
    "chr8:25065472-25065973",
    "chr8:25033468-25033969",
    "chr8:24886821-24887322",
    "chr8:24999050-24999551",
]

RNA_CMAP = LinearSegmentedColormap.from_list(
    "rna_rectified_h", ["#F7F7F7", "#DCEAF7", "#6BAED6", "#2171B5", "#08306B"], N=256
)
ATAC_CMAP = LinearSegmentedColormap.from_list(
    "atac_rectified_h", ["#EDEDED", "#FFF7F3", "#FDE0DD", "#FCC5C0", "#FA9FB5", "#F768A1", "#DD3497", "#AE017E"], N=256
)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.6,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def robust_vmax(values, percentile=99.5):
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(positive, percentile)) if positive.size else 1.0


def dense_to_grid(obs, values):
    grid = np.full((76, 101), np.nan, dtype=float)
    rows = obs["grid_row"].to_numpy(dtype=int)
    cols = obs["grid_col"].to_numpy(dtype=int)
    grid[rows, cols] = np.asarray(values, dtype=float)
    return grid


def draw_panel(ax, grid, title, cmap, vmax):
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", origin="upper", aspect="auto")
    ax.set_title(title, loc="left", fontsize=6.0, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    rna = ad.read_h5ad(RNA_PATH, backed="r")
    atac = ad.read_h5ad(ATAC_PATH, backed="r")
    if not np.array_equal(rna.obs_names.to_numpy(), atac.obs_names.to_numpy()):
        raise ValueError("rectified_h RNA and ATAC obs order differs")

    links = pd.read_csv(LINK_PATH, sep="\t")
    link_map = (
        links.loc[links["target"] == GENE]
        .drop_duplicates(subset=["region"])
        .sort_values("importance_x_abs_rho", ascending=False)
        .reset_index(drop=True)
    )
    link_map["panel_id"] = [f"E{i+1}" for i in range(len(link_map))]
    link_map = link_map.set_index("region")

    expr = read_feature_matrix(rna, [GENE], obs_names=rna.obs_names, batch_size=1)[:, 0]
    expr = np.log1p(np.maximum(expr, 0))
    enh = read_feature_matrix(atac, REQUESTED_ENHANCERS, obs_names=atac.obs_names, batch_size=len(REQUESTED_ENHANCERS))
    enh = np.log1p(np.maximum(enh, 0))

    expr_vmax = robust_vmax(expr)
    enh_vmax = robust_vmax(enh)
    expr_grid = dense_to_grid(rna.obs, expr)
    enh_grids = [dense_to_grid(atac.obs, enh[:, i]) for i in range(len(REQUESTED_ENHANCERS))]

    fig, axes = plt.subplots(2, 4, figsize=(11.3, 5.9), constrained_layout=False)
    axes = axes.ravel()
    draw_panel(axes[0], expr_grid, "A  NEFM RNA", RNA_CMAP, expr_vmax)
    panel_labels = ["B", "C", "D", "E", "F", "G", "H"]
    for i, enhancer in enumerate(REQUESTED_ENHANCERS):
        if enhancer in link_map.index:
            row = link_map.loc[enhancer]
            title = f"{panel_labels[i]}  {row['panel_id']}  rho={float(row['rho']):.2f}"
        else:
            title = f"{panel_labels[i]}  not in filtered NEFM links"
        draw_panel(axes[i + 1], enh_grids[i], title, ATAC_CMAP, enh_vmax)

    fig.subplots_adjust(left=0.035, right=0.92, top=0.92, bottom=0.08, wspace=0.08, hspace=0.12)
    cax_rna = fig.add_axes([0.93, 0.60, 0.015, 0.23])
    cax_atac = fig.add_axes([0.93, 0.22, 0.015, 0.23])
    cb_rna = mpl.colorbar.ColorbarBase(cax_rna, cmap=RNA_CMAP, norm=Normalize(0, expr_vmax), orientation="vertical")
    cb_rna.ax.tick_params(labelsize=5.8, length=2)
    cb_rna.set_label("log1p RNA", fontsize=6.0)
    cb_atac = mpl.colorbar.ColorbarBase(cax_atac, cmap=ATAC_CMAP, norm=Normalize(0, enh_vmax), orientation="vertical")
    cb_atac.ax.tick_params(labelsize=5.8, length=2)
    cb_atac.set_label("log1p ATAC", fontsize=6.0)
    fig.suptitle("Rectified-H spatial maps for NEFM and the requested enhancers", x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    rna.file.close()
    atac.file.close()


if __name__ == "__main__":
    main()
