from __future__ import annotations

import numpy as np


def contiguous_grid_blocks(
    coords: np.ndarray, n_folds: int = 5, axis: int = 0
) -> np.ndarray:
    """Assign observations to deterministic contiguous coordinate slabs.

    Quantile boundaries produce spatially continuous test regions and avoid
    random spot leakage. Duplicate boundary coordinates remain in one fold.
    """
    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("coords must have shape [n_observations, >=2]")
    if not 2 <= n_folds <= len(xy):
        raise ValueError("n_folds must be between 2 and n_observations")
    if axis < 0 or axis >= xy.shape[1]:
        raise ValueError("axis is outside the coordinate dimensions")
    order = np.argsort(xy[:, axis], kind="stable")
    labels = np.empty(len(xy), dtype=int)
    labels[order] = np.minimum(
        np.arange(len(xy), dtype=int) * n_folds // len(xy), n_folds - 1
    )
    # Never split observations sharing the same coordinate along the cut axis.
    values = xy[:, axis]
    for value in np.unique(values):
        idx = np.flatnonzero(values == value)
        labels[idx] = int(np.min(labels[idx]))
    # Reindex after duplicate-boundary merging.
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(int)


def spatial_folds(labels: np.ndarray):
    """Yield non-overlapping train/test indices for each spatial block."""
    labels = np.asarray(labels)
    for fold in np.unique(labels):
        test = np.flatnonzero(labels == fold)
        train = np.flatnonzero(labels != fold)
        if len(train) and len(test):
            yield train, test


def mask_heldout_context(
    gene_map: np.ndarray, fold_map: np.ndarray, heldout_fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """Zero held-out gene context and return a binary visibility channel."""
    gene = np.asarray(gene_map, dtype=float)
    folds = np.asarray(fold_map)
    if gene.shape != folds.shape:
        raise ValueError("gene_map and fold_map must have identical shapes")
    visible = folds != heldout_fold
    context = np.where(visible, gene, 0.0)
    return context, visible.astype(np.float32)

