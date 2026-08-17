from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class SpatialSimulation:
    enhancer_maps: np.ndarray  # [n_enhancers, H, W]
    gene_map: np.ndarray  # [H, W]
    attribution: np.ndarray  # [n_enhancers, H, W]
    tissue_mask: np.ndarray  # [H, W]
    scenario: str


def _blob(xx, yy, x0, y0, sigma):
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))


def simulate_spatial_decomposition(
    scenario: str = "complementary",
    shape: tuple[int, int] = (48, 64),
    n_enhancers: int = 4,
    noise: float = 0.04,
    seed: int = 20260814,
) -> SpatialSimulation:
    """Simulate gene maps with known enhancer-specific spatial contributions."""
    allowed = {
        "single", "redundant", "complementary", "spatial_complementary", "partial_switching",
        "unrelated_missing_factor",
    }
    if scenario not in allowed:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {sorted(allowed)}")
    if n_enhancers < 2:
        raise ValueError("n_enhancers must be at least 2")
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx / max(w - 1, 1)
    yy = yy / max(h - 1, 1)
    tissue = (((xx - 0.5) / 0.48) ** 2 + ((yy - 0.5) / 0.43) ** 2 <= 1)
    left = _blob(xx, yy, 0.30, 0.50, 0.16) * tissue
    right = _blob(xx, yy, 0.70, 0.50, 0.16) * tissue
    top = _blob(xx, yy, 0.50, 0.28, 0.15) * tissue
    enh = np.zeros((n_enhancers, h, w), dtype=np.float32)
    attr = np.zeros_like(enh)

    if scenario == "single":
        enh[0] = left
        enh[1] = right
        attr[0] = tissue
    elif scenario == "redundant":
        enh[0] = left
        enh[1] = np.clip(left + rng.normal(0, 0.025, shape), 0, None) * tissue
        attr[0] = 0.5 * tissue
        attr[1] = 0.5 * tissue
    elif scenario == "complementary":
        enh[0] = left
        enh[1] = right
        attr[0] = (xx <= 0.5) * tissue
        attr[1] = (xx > 0.5) * tissue
    elif scenario == "spatial_complementary":
        # Both candidates are active in both domains; only spatially varying
        # weights can suppress each enhancers off-domain cross-activity.
        enh[0] = left + 0.80 * right
        enh[1] = 0.80 * left + 0.40 * right
        attr[0] = (xx <= 0.5) * tissue
        attr[1] = (xx > 0.5) * tissue
    elif scenario == "partial_switching":
        enh[0] = left + 0.25 * top
        enh[1] = right + 0.25 * top
        blend = np.clip((0.58 - xx) / 0.16, 0, 1) * tissue
        attr[0] = blend
        attr[1] = tissue - blend
    else:
        enh[0] = right
        enh[1] = gaussian_filter(rng.random(shape), 2) * tissue
        # The true driver is intentionally absent from the candidate set.

    for i in range(2, n_enhancers):
        enh[i] = gaussian_filter(rng.random(shape), 2.5) * tissue
    if scenario == "unrelated_missing_factor":
        gene = left
    else:
        gene = np.sum(attr * enh, axis=0)
    spatial_noise = gaussian_filter(rng.normal(size=shape), 1.2)
    spatial_noise /= np.std(spatial_noise[tissue]) + 1e-8
    gene = np.clip(gene + noise * spatial_noise * tissue, 0, None)
    enh = np.clip(enh + noise * 0.5 * rng.normal(size=enh.shape) * tissue, 0, None)
    return SpatialSimulation(
        enhancer_maps=enh.astype(np.float32),
        gene_map=(gene * tissue).astype(np.float32),
        attribution=attr.astype(np.float32),
        tissue_mask=tissue.astype(bool),
        scenario=scenario,
    )

