"""Genome-wide gene-image scan on the fixed cropped GW12 canvas."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.stats import entropy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.config import load_config
from sparegvision.io import open_h5ad, paired_observations


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    out = root / "results" / "gw12_all_gene_cropped_features"
    out.mkdir(parents=True, exist_ok=True)
    rna = open_h5ad(root / "data" / "GW12_spatial_RNA.h5ad")
    atac = open_h5ad(root / "data" / "GW12_spatial_ATAC.h5ad")
    obs, coords = paired_observations(rna, atac, cfg["spatial_crop"])
    obs_idx = rna.obs_names.get_indexer(obs)
    xmin, xmax, ymin, ymax = cfg["spatial_canvas"]
    px = cfg["pixel_size"]
    nx = int(np.floor((xmax - xmin) / px + 1e-8)) + 1
    ny = int(np.floor((ymax - ymin) / px + 1e-8)) + 1
    grid_x = np.floor((coords[:, 0] - xmin) / px + 1e-8).astype(int)
    grid_y = np.floor((coords[:, 1] - ymin) / px + 1e-8).astype(int)
    flat = grid_y * nx + grid_x
    counts = np.bincount(flat, minlength=nx * ny).astype(float)
    observed = counts > 0
    img_mask = observed.reshape(ny, nx)

    # Valid adjacent-pixel pairs define a mask-aware spatial smoothness score.
    pair_masks = []
    for dy, dx in [(0, 1), (1, 0)]:
        a = np.zeros_like(img_mask); b = np.zeros_like(img_mask)
        if dy == 0:
            a[:, :-1], b[:, 1:] = img_mask[:, :-1], img_mask[:, 1:]
        else:
            a[:-1, :], b[1:, :] = img_mask[:-1, :], img_mask[1:, :]
        pair_masks.append((a & b, (dy, dx)))

    def metrics(values):
        sums = np.bincount(flat, weights=values, minlength=nx * ny)
        base = sums / np.maximum(counts, 1)
        base = base.reshape(ny, nx)
        vals = base[img_mask]
        if len(vals) == 0 or np.allclose(vals, vals[0]):
            z = np.zeros_like(base)
            return 0., 0., 0., 0., 0., z
        q2, q98 = np.percentile(vals, [2, 98])
        z = np.clip((base - q2) / (q98 - q2 + 1e-8), 0, 1)
        z[~img_mask] = 0
        hist, _ = np.histogram(z[img_mask], bins=32, range=(0, 1))
        p = hist[hist > 0].astype(float); p /= p.sum()
        ent = float(entropy(p) / np.log(32))
        corrs = []
        for pm, (dy, dx) in pair_masks:
            if pm.sum() < 3: continue
            if dy == 0: a, b = z[:, :-1][pm[:, :-1]], z[:, 1:][pm[:, :-1]]
            else: a, b = z[:-1, :][pm[:-1, :]], z[1:, :][pm[:-1, :]]
            if np.std(a) > 0 and np.std(b) > 0: corrs.append(np.corrcoef(a, b)[0, 1])
        spatial_corr = float(np.mean(corrs)) if corrs else 0.
        threshold = np.quantile(z[img_mask], .8)
        active = img_mask & (z >= threshold)
        labs, n = label(active, structure=np.ones((3, 3), int))
        sizes = np.bincount(labs.ravel())[1:]
        keep = np.sort(sizes[sizes >= 25])[::-1][:2]
        n_domains = int(len(keep))
        balance = float(keep[-1] / keep[0]) if len(keep) == 2 else 0.
        return ent, spatial_corr, n_domains, balance, float(vals.mean()), z

    rows = []
    names = list(rna.var_names)
    X = rna.X[obs_idx, :]
    if hasattr(X, "to_memory"):
        X = X.to_memory()
    batch = 512
    for start in range(0, len(names), batch):
        sub = X[:, start:start + batch]
        if hasattr(sub, "toarray"): sub = sub.toarray()
        sub = np.asarray(sub, dtype=float)
        for j, gene in enumerate(names[start:start + batch]):
            v = sub[:, j]
            detected = v > 0
            ent, sc, nd, bal, mean_image, _ = metrics(v)
            rows.append({"gene": gene, "n_detected": int(detected.sum()),
                         "detection_fraction": float(detected.mean()),
                         "mean_count": float(v.mean()), "image_entropy": ent,
                         "spatial_neighbor_corr": sc, "n_expression_domains": nd,
                         "domain_balance": bal, "mean_image_signal": mean_image})
        if (start + batch) % 4096 < batch:
            print("processed genes:", min(start + batch, len(names)), flush=True)
    result = pd.DataFrame(rows)
    result["two_domain_score"] = result["n_expression_domains"].eq(2) * result["domain_balance"] * result["spatial_neighbor_corr"].clip(lower=0)
    result["spatial_complexity_proxy"] = result["image_entropy"] * result["spatial_neighbor_corr"].clip(lower=0) * (1 + result["n_expression_domains"].eq(2))
    result.sort_values("spatial_complexity_proxy", ascending=False).to_csv(out / "all_gene_image_features.tsv", sep="\t", index=False)
    result[(result.detection_fraction >= .01) & (result.spatial_neighbor_corr > .1)].sort_values("spatial_complexity_proxy", ascending=False).to_csv(out / "spatially_structured_genes.tsv", sep="\t", index=False)
    (out / "metadata.json").write_text(pd.Series({"n_genes": len(result), "n_paired_spots": len(obs), "canvas_shape": [ny, nx], "observed_fraction": float(img_mask.mean()), "note": "No formal DEG list was present; structured-gene subset uses detection_fraction >= 0.01 and spatial_neighbor_corr > 0.1."}).to_json())
    print("done:", len(result), "genes; output:", out)


if __name__ == "__main__":
    main()
