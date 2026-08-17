from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "gw12_slice_selection"
    out.mkdir(parents=True, exist_ok=True)

    rna = ad.read_h5ad(root / "data" / "GW12_spatial_RNA.h5ad", backed="r")
    xy = np.asarray(rna.obsm["spatial"])
    valid = rna.obs["valid_spot"].astype(bool).to_numpy()

    # Central crop candidate: retain the main tissue body while removing
    # sparse outer rows/columns and large technical-empty margins.
    crop = dict(xmin=4200.0, xmax=7200.0, ymin=5100.0, ymax=7350.0)
    inside = (
        valid
        & (xy[:, 0] >= crop["xmin"])
        & (xy[:, 0] <= crop["xmax"])
        & (xy[:, 1] >= crop["ymin"])
        & (xy[:, 1] <= crop["ymax"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, mode in zip(axes, ["all", "valid"]):
        mask = np.ones(len(xy), dtype=bool) if mode == "all" else valid
        ax.scatter(
            xy[mask, 0], xy[mask, 1], s=5, c="#BDBDBD", alpha=0.55,
            linewidths=0, rasterized=True,
        )
        ax.scatter(
            xy[inside, 0], xy[inside, 1], s=5, c="#2166AC", alpha=0.7,
            linewidths=0, rasterized=True,
        )
        ax.add_patch(
            plt.Rectangle(
                (crop["xmin"], crop["ymin"]),
                crop["xmax"] - crop["xmin"],
                crop["ymax"] - crop["ymin"],
                fill=False, linewidth=2, edgecolor="#D73027",
            )
        )
        ax.set_title(f"GW12 {mode} spots")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.invert_yaxis()

    fig.suptitle(
        "Central crop candidate: "
        f"x={crop['xmin']:.0f}–{crop['xmax']:.0f}, "
        f"y={crop['ymin']:.0f}–{crop['ymax']:.0f} "
        f"({inside.sum()} valid spots)",
        fontsize=13,
    )
    fig.savefig(out / "GW12_slice_crop_overview.png", dpi=220)
    fig.savefig(out / "GW12_slice_crop_overview.pdf")

    np.savetxt(
        out / "selected_crop.tsv",
        np.array([[crop["xmin"], crop["xmax"], crop["ymin"], crop["ymax"], inside.sum()]]),
        delimiter="\t",
        header="xmin\txmax\tymin\tymax\tn_valid_spots",
        comments="",
    )
    print("output:", out / "GW12_slice_crop_overview.png")
    print("selected:", crop, "valid spots:", int(inside.sum()))


if __name__ == "__main__":
    main()
