from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression


def _flat(enhancer_maps, gene_map, train_mask, test_mask):
    x = np.asarray(enhancer_maps, dtype=float)
    y = np.asarray(gene_map, dtype=float)
    if x.ndim != 3 or x.shape[1:] != y.shape:
        raise ValueError("enhancer_maps must be [N,H,W] matching gene_map [H,W]")
    tr = np.asarray(train_mask, bool).ravel()
    te = np.asarray(test_mask, bool).ravel()
    design = x.reshape(x.shape[0], -1).T
    return design[tr], y.ravel()[tr], design[te], y.ravel()[te]


def spatial_mean_baseline(gene_map, train_mask, test_mask):
    y = np.asarray(gene_map, float)
    mean = float(np.mean(y[np.asarray(train_mask, bool)]))
    return np.full(int(np.sum(test_mask)), mean), y[np.asarray(test_mask, bool)]


def best_single_enhancer(enhancer_maps, gene_map, train_mask, test_mask):
    xtr, ytr, xte, yte = _flat(enhancer_maps, gene_map, train_mask, test_mask)
    best = None
    for j in range(xtr.shape[1]):
        model = LinearRegression(positive=True).fit(xtr[:, j:j + 1], ytr)
        loss = float(np.mean((ytr - model.predict(xtr[:, j:j + 1])) ** 2))
        if best is None or loss < best[0]:
            best = (loss, j, model)
    _, selected, model = best
    return model.predict(xte[:, selected:selected + 1]), yte, selected


def global_nonnegative_elastic_net(
    enhancer_maps, gene_map, train_mask, test_mask, alpha=1e-3, l1_ratio=0.5
):
    xtr, ytr, xte, yte = _flat(enhancer_maps, gene_map, train_mask, test_mask)
    model = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, positive=True, max_iter=5000,
        random_state=0,
    ).fit(xtr, ytr)
    return model.predict(xte), yte, model.coef_.copy()

