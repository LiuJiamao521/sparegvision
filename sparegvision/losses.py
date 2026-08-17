from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_huber(prediction, target, mask, delta=1.0):
    loss = F.huber_loss(prediction, target, reduction="none", delta=delta)
    weights = mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1)


def spatial_block_safety(prediction, baseline, target, mask, n_blocks=5):
    """Penalize spatial residuals that increase MSE in any contiguous x block."""
    width=prediction.shape[-1]
    penalties=[]
    for block in range(n_blocks):
        start=block*width//n_blocks; end=(block+1)*width//n_blocks
        block_mask=mask[...,start:end].to(prediction.dtype)
        denominator=block_mask.sum().clamp_min(1)
        spatial_mse=(((prediction[...,start:end]-target[...,start:end])**2)*block_mask).sum()/denominator
        baseline_mse=(((baseline[...,start:end]-target[...,start:end])**2)*block_mask).sum()/denominator
        penalties.append(torch.relu(spatial_mse-baseline_mse))
    return torch.stack(penalties).mean()

def attribution_total_variation(attribution, tissue_mask=None):
    dx = torch.abs(attribution[..., :, 1:] - attribution[..., :, :-1])
    dy = torch.abs(attribution[..., 1:, :] - attribution[..., :-1, :])
    if tissue_mask is not None:
        m = tissue_mask.to(dx.dtype)
        dx = dx * (m[..., :, 1:] * m[..., :, :-1])
        dy = dy * (m[..., 1:, :] * m[..., :-1, :])
    return dx.mean() + dy.mean()


def attribution_entropy(attribution, background, tissue_mask=None):
    probs = torch.cat([background, attribution], dim=1).clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=1, keepdim=True)
    if tissue_mask is None:
        return entropy.mean()
    weights = tissue_mask.to(entropy.dtype)
    return (entropy * weights).sum() / weights.sum().clamp_min(1)

