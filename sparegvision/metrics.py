from __future__ import annotations

import numpy as np


def prediction_metrics(observed, predicted):
    y = np.asarray(observed, float).ravel()
    p = np.asarray(predicted, float).ravel()
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) < 2:
        return {"mse": np.nan, "r2": np.nan, "pearson": np.nan}
    mse = float(np.mean((y - p) ** 2))
    den = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - p) ** 2) / (den + 1e-12))
    corr = float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else 0.0
    return {"mse": mse, "r2": r2, "pearson": corr}


def attribution_overlap(contributions, mask=None):
    """Mean pairwise soft Dice; high values indicate redundant support."""
    c = np.maximum(np.asarray(contributions, float), 0)
    if c.ndim != 3:
        raise ValueError("contributions must have shape [N,H,W]")
    m = np.ones(c.shape[1:], bool) if mask is None else np.asarray(mask, bool)
    scores = []
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            a, b = c[i][m], c[j][m]
            scores.append(2 * np.minimum(a, b).sum() / (a.sum() + b.sum() + 1e-8))
    return float(np.mean(scores)) if scores else 0.0


def unique_contribution_coverage(contributions, mask=None, threshold=0.5):
    c = np.maximum(np.asarray(contributions, float), 0)
    m = np.ones(c.shape[1:], bool) if mask is None else np.asarray(mask, bool)
    total = c.sum(axis=0) + 1e-8
    dominance = c.max(axis=0) / total
    return float(np.mean(dominance[m] >= threshold)) if m.any() else 0.0

