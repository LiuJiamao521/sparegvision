from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import pandas as pd

_COLUMN_MAP_CACHE = {}

def open_h5ad(path, backed="r"):
    import anndata as ad
    return ad.read_h5ad(path, backed=backed)

def paired_observations(rna, atac, spatial_crop=None):
    """Return common observations with finite coordinates and optional crop.

    ``spatial_crop`` is ``(xmin, xmax, ymin, ymax)`` in the original spatial
    coordinate system.  Filtering is done after intersecting RNA/ATAC spots so
    every downstream gene/enhancer image uses exactly the same canvas points.
    """
    common = rna.obs_names.intersection(atac.obs_names)
    coords = np.asarray(rna.obsm["spatial"])[rna.obs_names.get_indexer(common)]
    valid = np.isfinite(coords).all(axis=1)
    if "valid_spot" in rna.obs:
        valid &= rna.obs.loc[common, "valid_spot"].to_numpy(bool)
    if "valid_spot" in atac.obs:
        valid &= atac.obs.loc[common, "valid_spot"].to_numpy(bool)
    if spatial_crop is not None:
        xmin, xmax, ymin, ymax = map(float, spatial_crop)
        valid &= (
            (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax)
            & (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
        )
    return common[valid], coords[valid]

def read_feature_matrix(adata, feature_names, obs_names=None, batch_size=256, matrix=None, matrix_var_names=None):
    """Read selected features in chunks, preserving feature order."""
    names = list(feature_names)
    source = adata
    if matrix is not None:
        names = list(feature_names)
        var_names = list(matrix_var_names if matrix_var_names is not None else adata.var_names)
        key=id(matrix)
        col_map = _COLUMN_MAP_CACHE.get(key)
        if col_map is None:
            col_map = {x:i for i,x in enumerate(var_names)}
            _COLUMN_MAP_CACHE[key] = col_map
        cols = [col_map[x] for x in names]
        sub = matrix[:, cols]
        if hasattr(sub, "toarray"): sub = sub.toarray()
        sub = np.asarray(sub, dtype=float)
        if obs_names is not None:
            sub = sub[adata.obs_names.get_indexer(obs_names), :]
        return sub
    # GW12 ATAC X is an unchunked 25-GB dense array, while raw/X is sparse.
    # Prefer the sparse representation when available to avoid an accidental
    # full-file read for a single candidate peak.
    if getattr(adata, "raw", None) is not None:
        try:
            if hasattr(adata.raw.X, "to_memory") or hasattr(adata.raw.X, "toarray"):
                source = adata.raw
        except Exception:
            pass
    obs_idx = slice(None) if obs_names is None else adata.obs_names.get_indexer(obs_names)
    present = set(source.var_names)
    missing = [x for x in names if x not in present]
    if missing:
        raise KeyError("Features not found: %s" % missing[:5])
    chunks = []
    for start in range(0, len(names), batch_size):
        # Select columns on the backed object first, then select observations
        # in memory. h5py permits only one fancy index per dataset operation.
        chunk_names = names[start:start + batch_size]
        if source is not adata and hasattr(source.X, "__getitem__"):
            # Raw backed matrices expose efficient sparse column indexing;
            # going through Raw.__getitem__ would return a lazy dataset view
            # and can materialize the complete matrix.
            col_idx = [source.var_names.get_loc(x) for x in chunk_names]
            # Scalar-column access is reliable across anndata versions and
            # avoids CSRDataset materializing a large fancy-index selection.
            import scipy.sparse
            cols = [source.X[:, j] for j in col_idx]
            sub = scipy.sparse.hstack(cols, format="csr")
        else:
            sub = source[:, chunk_names].X
        if hasattr(sub, "to_memory"):
            sub = sub.to_memory()
        if hasattr(sub, "toarray"):
            sub = sub.toarray()
        sub = np.asarray(sub, dtype=float)
        if obs_names is not None:
            sub = sub[obs_idx, :]
        chunks.append(sub)
    n_obs = adata.n_obs if obs_names is None else len(obs_names)
    return np.concatenate(chunks, axis=1) if chunks else np.empty((n_obs, 0))

def read_links(path, rna_names=None, atac_names=None, top_k=20):
    links = pd.read_csv(path, sep="\t")
    required = {"target", "region"}
    missing = required - set(links.columns)
    if missing:
        raise ValueError("Link table missing columns: %s" % sorted(missing))
    links = links.rename(columns={"target": "gene", "region": "enhancer"})
    links["prior_score"] = links.get("importance_x_abs_rho", 0.0).astype(float)
    links["prior_signed_score"] = links.get("importance_x_rho", 0.0).astype(float)
    if rna_names is not None:
        links = links[links.gene.isin(set(rna_names))]
    if atac_names is not None:
        links = links[links.enhancer.isin(set(atac_names))]
    links = links.drop_duplicates(["gene", "enhancer"])
    links = links.sort_values(["gene", "prior_score"], ascending=[True, False])
    return links.groupby("gene", sort=False, group_keys=False).head(top_k).reset_index(drop=True)

def audit_h5ad(path):
    a = open_h5ad(path, backed="r")
    coords = np.asarray(a.obsm.get("spatial")) if "spatial" in a.obsm else None
    valid = np.isfinite(coords).all(1) if coords is not None else np.zeros(a.n_obs, bool)
    if "valid_spot" in a.obs:
        valid &= a.obs.valid_spot.to_numpy(bool)
    out = {"path": str(path), "n_obs": int(a.n_obs), "n_vars": int(a.n_vars),
           "obs_columns": list(a.obs.columns), "var_columns": list(a.var.columns),
           "has_spatial": coords is not None, "valid_spots": int(valid.sum())}
    a.file.close()
    return out
