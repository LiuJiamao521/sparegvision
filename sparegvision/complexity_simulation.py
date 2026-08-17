from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class MixedComplexitySimulation:
    enhancer_maps: np.ndarray
    gene_map: np.ndarray
    tissue_mask: np.ndarray
    enhancer_classes: tuple[str, ...]
    enhancer_domains: np.ndarray
    gene_id: str = "sim_gene"


def _blob(xx, yy, x0, y0, sigma):
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))


def simulate_mixed_regulatory_complexity(
    shape=(48, 64), noise=0.04, seed=20260814
):
    """One gene with 10 core, 3 unique regional, 2 redundant regional and 5 null enhancers."""
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = xx / max(w - 1, 1)
    y = yy / max(h - 1, 1)
    tissue = (((x - 0.5) / 0.48) ** 2 + ((y - 0.5) / 0.43) ** 2 <= 1)
    latent = (
        0.85 * _blob(x, y, 0.28, 0.52, 0.17)
        + 0.72 * _blob(x, y, 0.70, 0.52, 0.16)
        + 0.48 * _blob(x, y, 0.50, 0.25, 0.13)
    ) * tissue
    gene_noise = gaussian_filter(rng.normal(size=shape), 1.1)
    gene_noise /= np.std(gene_noise[tissue]) + 1e-8
    gene = np.clip(latent + noise * gene_noise * tissue, 0, None)

    domains = np.stack([
        (x < 0.40) & tissue,
        (x > 0.60) & tissue,
        (y < 0.42) & tissue,
    ])
    enhancers = np.zeros((20, h, w), dtype=np.float32)
    classes = ["core_concordant"] * 10
    classes += ["regional_specific"] * 3
    classes += ["regional_redundant"] * 2
    classes += ["unsupported"] * 5
    enhancer_domains = np.full(20, -1, dtype=int)

    for i in range(10):
        local_noise = gaussian_filter(rng.normal(size=shape), 0.8 + 0.15 * (i % 3))
        enhancers[i] = np.clip((0.75 + 0.05 * i) * gene + 0.05 * local_noise * tissue, 0, None)

    regional_assignment = [0, 1, 2, 0, 1]
    for j, domain_id in enumerate(regional_assignment, start=10):
        domain = domains[domain_id]
        off_domain = gaussian_filter(rng.random(shape), 2.2) * tissue
        off_domain = off_domain / (off_domain[tissue].max() + 1e-8)
        local_noise = gaussian_filter(rng.normal(size=shape), 0.8)
        signal = np.where(domain, gene + 0.025 * local_noise, off_domain)
        enhancers[j] = np.clip(signal * tissue, 0, None)
        enhancer_domains[j] = domain_id

    for j in range(15, 20):
        null = gaussian_filter(rng.random(shape), 2.0 + 0.25 * (j - 15)) * tissue
        enhancers[j] = null / (null[tissue].max() + 1e-8)

    enhancers += noise * 0.25 * rng.normal(size=enhancers.shape) * tissue
    enhancers = np.clip(enhancers, 0, None)
    return MixedComplexitySimulation(
        enhancer_maps=enhancers.astype(np.float32),
        gene_map=(gene * tissue).astype(np.float32),
        tissue_mask=tissue.astype(bool),
        enhancer_classes=tuple(classes),
        enhancer_domains=enhancer_domains,
    )
