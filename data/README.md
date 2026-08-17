# Data Location And Formats

The workspace-local `data/` directory is no longer the canonical dataset store.
The canonical data root is:

`/cluster3/labData/jiamao/SpaRegVision`

Agents and scripts should read data from that path by default.

## Canonical directory layout

### 1. `GW12/`
Human GW12 reference dataset used as a paired spatial RNA and spatial ATAC example.

Main files:
- `GW12_spatial_RNA.h5ad`
- `GW12_spatial_ATAC.h5ad`
- `GW12_region_to_gene_links_rectified_h_latent_state.tsv`
- `GW12_region_to_gene_links_rectified_h_latent_state.metadata.json`
- `region_to_gene_adj.tsv`
- `gw12.ipynb`

Subdirectories:
- `rectified_h/`
- `selected_8k/`
- `selected_rectangle/`

Formats:
- `*.h5ad`: AnnData objects for RNA or ATAC matrices.
- `*.tsv`: tab-separated text tables.
- `*.json`: metadata/configuration for linked outputs.
- `*.ipynb`: notebook analysis.

### 2. `weMERFISH/`
Zebrafish 6-somite (`C_6s`) E1 dataset, including transcriptome, reconstructed spatial ATAC, raw ATAC interval exports, and a small number of visualization outputs.

Main files:
- `weMERFISH_measured_C_6s_E1_rescaled_z.h5ad`
- `weMERFISH_combined_C_6s_E1_rescaled_z.h5ad`
- `weMERFISH_spatial_ATAC_C_6s_E1.h5ad`
- `README.md`
- `hdlbpa_enhancer_chr6_26925631_26926131_xy.png`
- `hdlbpa_enhancer_chr6_26925631_26926131_xz.png`

Subdirectories:
- `atac_raw_6s/`
  - `csv/`: one CSV per ATAC interval.
  - `logs/`: download, retry, validation, and reconstruction logs.

Formats:
- `weMERFISH_measured_C_6s_E1_rescaled_z.h5ad`
  - measured single-cell transcriptome AnnData.
  - spatial coordinates are stored in `adata.obsm['spatial']` and `adata.obsm['spatial_rescaled_z']`.
- `weMERFISH_combined_C_6s_E1_rescaled_z.h5ad`
  - transcriptome AnnData with measured genes in `adata.X` and imputed genes in `adata.obsm['X_imputed']`.
  - imputed gene names are stored in `adata.uns['imputed_gene_names']`.
- `weMERFISH_spatial_ATAC_C_6s_E1.h5ad`
  - reconstructed spatial ATAC AnnData aligned to the same cell order as the transcriptome objects.
  - accessibility matrix is in `adata.X`.
  - interval metadata are in `adata.var` with fields such as `chrom`, `start`, `end`, and `width`.
- `atac_raw_6s/csv/*.csv`
  - raw per-interval ATAC exports.
  - each file contains one interval ID followed by one accessibility value per cell.
- `*.png`
  - static visualization outputs.

## Practical guidance for agents

- Treat `/cluster3/labData/jiamao/SpaRegVision` as the source of truth.
- Do not assume the local workspace `data/` directory contains full datasets.
- If a task asks for RNA or ATAC data loading, start from the `.h5ad` files under the canonical root.
- If a task asks for raw interval-level ATAC reconstruction or validation, use `weMERFISH/atac_raw_6s/csv/` under the canonical root.
- Ignore `.DS_Store` and `._*` files if they appear; they are filesystem artifacts, not scientific data.
