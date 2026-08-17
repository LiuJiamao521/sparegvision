import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad"
ATAC_PATH = ROOT / "data" / "rectified_h" / "GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad"
LINKS_PATH = ROOT / "data" / "GW12_region_to_gene_links_filtered.tsv"
OUT_DIR = ROOT / "data" / "rectified_h_downstream"


def replace_symlink(link_path: Path, target_path: Path):
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    relative_target = os.path.relpath(target_path, start=link_path.parent)
    link_path.symlink_to(relative_target)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rna = ad.read_h5ad(RNA_PATH, backed="r")
    atac = ad.read_h5ad(ATAC_PATH, backed="r")
    if not np.array_equal(rna.obs_names.to_numpy(), atac.obs_names.to_numpy()):
        raise ValueError("RNA and ATAC obs_names differ")
    if not np.array_equal(np.asarray(rna.obsm["spatial"]), np.asarray(atac.obsm["spatial"])):
        raise ValueError("RNA and ATAC spatial coordinates differ")

    replace_symlink(OUT_DIR / "rna.h5ad", RNA_PATH)
    replace_symlink(OUT_DIR / "enhancer.h5ad", ATAC_PATH)

    shared_obs = rna.obs.copy()
    spatial = np.asarray(rna.obsm["spatial"])
    shared_obs.insert(0, "spot", rna.obs_names.to_numpy())
    shared_obs["x"] = spatial[:, 0]
    shared_obs["y"] = spatial[:, 1]
    shared_obs.to_csv(OUT_DIR / "shared_obs.tsv", sep="\t", index=False)

    rna.var.reset_index().rename(columns={"index": "gene"}).to_csv(OUT_DIR / "genes.tsv", sep="\t", index=False)
    atac.var.reset_index().rename(columns={"index": "enhancer"}).to_csv(OUT_DIR / "enhancers.tsv", sep="\t", index=False)

    links = pd.read_csv(LINKS_PATH, sep="\t")
    links = links[links["target"].isin(set(rna.var_names)) & links["region"].isin(set(atac.var_names))].copy()
    links = links.drop_duplicates(subset=["target", "region"]).reset_index(drop=True)
    links.to_csv(OUT_DIR / "links.tsv", sep="\t", index=False)

    manifest = {
        "dataset": "rectified_h",
        "description": "Standardized paired downstream inputs for SpaRegVision using the H-rectified 101x76 RNA/ATAC canvas.",
        "files": {
            "rna_h5ad": "rna.h5ad",
            "enhancer_h5ad": "enhancer.h5ad",
            "links_tsv": "links.tsv",
            "shared_obs_tsv": "shared_obs.tsv",
            "genes_tsv": "genes.tsv",
            "enhancers_tsv": "enhancers.tsv",
        },
        "sources": {
            "rna": str(RNA_PATH.relative_to(ROOT)),
            "atac": str(ATAC_PATH.relative_to(ROOT)),
            "links": str(LINKS_PATH.relative_to(ROOT)),
        },
        "paired_obs": {
            "n_spots": int(rna.n_obs),
            "obs_names_equal": True,
            "spatial_equal": True,
            "grid_shape": [76, 101],
        },
        "features": {
            "n_genes": int(rna.n_vars),
            "n_enhancers": int(atac.n_vars),
            "n_links_retained": int(len(links)),
        },
        "recommended_cli": (
            "sparegvision run --rna data/rectified_h_downstream/rna.h5ad "
            "--enhancer data/rectified_h_downstream/enhancer.h5ad "
            "--links data/rectified_h_downstream/links.tsv "
            "--config configs/default.yaml --output results_rectified_h"
        ),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    rna.file.close()
    atac.file.close()
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
