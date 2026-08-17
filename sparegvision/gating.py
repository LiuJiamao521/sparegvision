from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GateDecision:
    strength: torch.Tensor
    enabled: bool
    diagnostics: list[dict[str, float | int | bool]]
    mean_relative_gain: float
    worst_relative_gain: float


def calibrate_spatial_gate(
    base: torch.Tensor,
    full: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    fold_map: torch.Tensor,
    test_fold: int,
    policy: str = "robust",
    min_folds: int = 3,
    min_mean_relative_gain: float = 0.002,
    max_fold_relative_loss: float = 0.01,
) -> GateDecision:
    """Calibrate spatial modulation using only blocks visible at final testing."""
    if policy not in {"strict", "majority", "robust"}:
        raise ValueError(f"unknown gate policy: {policy}")
    delta = full - base
    candidates: list[torch.Tensor] = []
    diagnostics: list[dict[str, float | int | bool]] = []
    for validation_fold in torch.unique(fold_map).tolist():
        validation_fold = int(validation_fold)
        if validation_fold == test_fold:
            continue
        fit_mask = visible_mask & (fold_map[None, None] != validation_fold)
        validation_mask = visible_mask & (fold_map[None, None] == validation_fold)
        weight = fit_mask.to(delta.dtype)
        numerator = ((target - base) * delta * weight).sum()
        denominator = (delta.square() * weight).sum().clamp_min(1e-12)
        candidate = (numerator / denominator).clamp(0, 1).reshape(1)
        calibrated = base + candidate.reshape(-1, 1, 1, 1) * delta
        vm = validation_mask.to(delta.dtype)
        base_loss = (((base - target) ** 2) * vm).sum() / vm.sum().clamp_min(1)
        calibrated_loss = (((calibrated - target) ** 2) * vm).sum() / vm.sum().clamp_min(1)
        relative_gain = (base_loss - calibrated_loss) / base_loss.clamp_min(1e-12)
        candidates.append(candidate)
        diagnostics.append({
            "validation_fold": validation_fold,
            "candidate_strength": float(candidate.item()),
            "base_mse": float(base_loss.item()),
            "calibrated_mse": float(calibrated_loss.item()),
            "relative_gain": float(relative_gain.item()),
            "improved": bool(relative_gain.item() > 0),
        })
    if not candidates:
        raise ValueError("no visible validation folds available")
    candidate = torch.median(torch.cat(candidates)).reshape(1)
    gains = np.asarray([d["relative_gain"] for d in diagnostics], dtype=float)
    improved = int((gains > 0).sum())
    if policy == "strict":
        enabled = improved == len(gains)
    elif policy == "majority":
        enabled = improved >= min_folds
    else:
        enabled = (improved >= min_folds and float(gains.mean()) >= min_mean_relative_gain
                   and float(gains.min()) >= -max_fold_relative_loss)
    strength = candidate if enabled else torch.zeros_like(candidate)
    return GateDecision(strength, enabled, diagnostics, float(gains.mean()), float(gains.min()))
