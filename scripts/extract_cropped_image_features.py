"""Pilot extraction of mask-aware image features for NEFM/NEFL."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import entropy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.config import load_config
from sparegvision.io import open_h5ad, paired_observations, read_feature_matrix
from sparegvision.rasterize import rasterize
from sparegvision.metrics import rvs_components


def image_entropy(maps, mask, bins=32):
    vals = maps[0][mask]
    if len(vals) == 0 or np.allclose(vals, vals[0]):
        return 0.0
    hist, _ = np.histogram(vals, bins=bins, range=(0, 1), density=False)
    p = hist[hist > 0].astype(float)
    p /= p.sum()
    return float(entropy(p) / np.log(bins))


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    out = root / "results" / "gw12_cropped_image_features"
    out.mkdir(parents=True, exist_ok=True)
    genes = ["NEFM", "NEFL"]

    rna = open_h5ad(root / "data" / "GW12_spatial_RNA.h5ad")
    atac = open_h5ad(root / "data" / "GW12_spatial_ATAC.h5ad")
    obs, coords = paired_observations(rna, atac, cfg["spatial_crop"])
    canvas = cfg["spatial_canvas"]
    feature_rows, edge_rows, node_vectors = [], [], []

    for gene in genes:
        links_path = root / "results" / f"gw12_{gene}_domains" / "enhancer_gene_rvs.tsv"
        links = pd.read_csv(links_path, sep="\t")
        enhancers = links.enhancer.drop_duplicates().tolist()
        names = [gene] + enhancers
        vals = read_feature_matrix(rna, [gene], obs, cfg["max_features_per_batch"])[:, 0]
        gene_maps, mask, meta = rasterize(vals, coords, cfg["pixel_size"], cfg["scales"], canvas=canvas)
        feature_rows.append({"gene": gene, "node": gene, "node_type": "gene",
                             "entropy": image_entropy(gene_maps, mask),
                             "observed_fraction": meta["observed_fraction"]})
        node_vectors.append((gene, gene, gene_maps[:, mask].ravel()))

        X = read_feature_matrix(atac, enhancers, obs, cfg["max_features_per_batch"])
        for j, enhancer in enumerate(enhancers):
            em, emask, _ = rasterize(X[:, j], coords, cfg["pixel_size"], cfg["scales"], canvas=canvas)
            edge = rvs_components(gene_maps, em, mask, cfg["scale_weights"])
            edge_rows.append({"gene": gene, "enhancer": enhancer, **edge})
            feature_rows.append({"gene": gene, "node": enhancer, "node_type": "enhancer",
                                 "entropy": image_entropy(em, mask),
                                 "observed_fraction": float(emask.mean())})
            node_vectors.append((gene, enhancer, em[:, mask].ravel()))

    pd.DataFrame(feature_rows).to_csv(out / "node_image_features.tsv", sep="\t", index=False)
    pd.DataFrame(edge_rows).to_csv(out / "gene_enhancer_edges.tsv", sep="\t", index=False)

    # PCA is used as the quantitative feature space. UMAP is only a view.
    genes_for_nodes = [x[0] for x in node_vectors]
    names = [x[1] for x in node_vectors]
    matrix = np.vstack([x[2] for x in node_vectors])
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    z = PCA(n_components=min(15, matrix.shape[0] - 1, matrix.shape[1]), random_state=cfg["seed"]).fit_transform(StandardScaler().fit_transform(matrix))
    emb = pd.DataFrame(z, columns=[f"PC{i+1}" for i in range(z.shape[1])])
    emb.insert(0, "node", names)
    emb.insert(1, "gene", genes_for_nodes)
    emb.insert(2, "node_type", ["gene" if n in genes else "enhancer" for n in names])
    embedding_method = "UMAP"
    try:
        import umap
        u = umap.UMAP(n_components=2, n_neighbors=min(10, len(names)-1), min_dist=.2, random_state=cfg["seed"]).fit_transform(z)
        emb["UMAP1"], emb["UMAP2"] = u[:, 0], u[:, 1]
    except Exception:
        # Keep the plot usable if optional umap-learn is unavailable.
        embedding_method = "PCA"
        emb["UMAP1"], emb["UMAP2"] = z[:, 0], z[:, 1]
    emb.to_csv(out / "node_embedding.tsv", sep="\t", index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"NEFM": "#2166AC", "NEFL": "#1B7837"}
    for gene in genes:
        sub = emb[emb.gene == gene]
        enh = sub[sub.node_type == "enhancer"]
        g = sub[sub.node_type == "gene"].iloc[0]
        ax.scatter(enh.UMAP1, enh.UMAP2, s=28, alpha=.75, c=colors[gene], label=f"{gene} enhancers")
        ax.scatter(g.UMAP1, g.UMAP2, s=150, marker="*", c=colors[gene], edgecolor="black", linewidth=.7,
                   label=f"{gene} expression")
        for _, row in enh.iterrows():
            ax.plot([g.UMAP1, row.UMAP1], [g.UMAP2, row.UMAP2], color=colors[gene], alpha=.12, linewidth=.6)
    ax.set_title(f"Mask-aware cropped gene–enhancer image embedding ({embedding_method})")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gene_enhancer_umap_network.png", dpi=220)
    fig.savefig(out / "gene_enhancer_umap_network.pdf")
    plt.close(fig)
    pd.Series({"n_paired_spots": len(obs), "canvas": str(canvas), "canvas_shape": str(meta["shape"]),
               "observed_fraction": meta["observed_fraction"]}).to_json(out / "metadata.json", indent=2)
    print("paired spots:", len(obs), "canvas:", meta["shape"], "observed fraction:", meta["observed_fraction"])
    print("nodes:", len(feature_rows), "edges:", len(edge_rows), "output:", out)


if __name__ == "__main__":
    main()
