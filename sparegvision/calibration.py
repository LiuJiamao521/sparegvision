from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import ElasticNet


@dataclass(frozen=True)
class FeatureGateDecision:
    prediction: np.ndarray
    enabled: bool
    coefficients: np.ndarray
    intercept: float
    diagnostics: list[dict[str, float | int | bool]]
    mean_relative_gain: float
    worst_relative_gain: float


def _fit_map(features, target, fit_mask, alpha=1e-3):
    model = ElasticNet(
        alpha=alpha, l1_ratio=0.5, positive=True, max_iter=5000, random_state=0
    )
    model.fit(features[:, fit_mask].T, target[fit_mask])
    prediction = model.predict(features.reshape(features.shape[0], -1).T).reshape(target.shape)
    return prediction, model.coef_.astype(float), float(model.intercept_)


def nested_spatial_feature_gate(
    global_features,
    spatial_features,
    target,
    visible_mask,
    fold_map,
    test_fold,
    min_folds=3,
    min_mean_relative_gain=0.002,
    max_fold_relative_loss=0.01,
    alpha=1e-3,
):
    """Compare global and learned spatial features with symmetric nested refits."""
    global_features = np.asarray(global_features, dtype=float)
    spatial_features = np.asarray(spatial_features, dtype=float)
    target = np.asarray(target, dtype=float)
    visible_mask = np.asarray(visible_mask, dtype=bool)
    fold_map = np.asarray(fold_map)
    diagnostics = []
    for validation_fold in sorted(np.unique(fold_map)):
        validation_fold = int(validation_fold)
        if validation_fold == test_fold:
            continue
        fit_mask = visible_mask & (fold_map != validation_fold)
        validation_mask = visible_mask & (fold_map == validation_fold)
        global_prediction, _, _ = _fit_map(global_features, target, fit_mask, alpha)
        spatial_prediction, _, _ = _fit_map(spatial_features, target, fit_mask, alpha)
        global_mse = float(np.mean((global_prediction[validation_mask] - target[validation_mask]) ** 2))
        spatial_mse = float(np.mean((spatial_prediction[validation_mask] - target[validation_mask]) ** 2))
        relative_gain = (global_mse - spatial_mse) / max(global_mse, 1e-12)
        diagnostics.append({
            "validation_fold": validation_fold,
            "global_mse": global_mse,
            "spatial_mse": spatial_mse,
            "relative_gain": relative_gain,
            "improved": relative_gain > 0,
        })
    gains = np.asarray([row["relative_gain"] for row in diagnostics])
    enabled = (
        int((gains > 0).sum()) >= min_folds
        and float(gains.mean()) >= min_mean_relative_gain
        and float(gains.min()) >= -max_fold_relative_loss
    )
    selected_features = spatial_features if enabled else global_features
    prediction, coefficients, intercept = _fit_map(
        selected_features, target, visible_mask, alpha
    )
    return FeatureGateDecision(
        prediction=prediction,
        enabled=enabled,
        coefficients=coefficients,
        intercept=intercept,
        diagnostics=diagnostics,
        mean_relative_gain=float(gains.mean()),
        worst_relative_gain=float(gains.min()),
    )
