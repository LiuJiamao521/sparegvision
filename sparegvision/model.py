from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SpatialAttributionSetNetwork(nn.Module):
    """Permutation-equivariant enhancer-set model with explicit spatial output."""

    def __init__(
        self, enhancer_channels: int = 1, gene_context_channels: int = 3,
        hidden_dim: int = 64, attention_heads: int = 4, set_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.enhancer_encoder = ConvEncoder(enhancer_channels, hidden_dim)
        self.gene_encoder = ConvEncoder(gene_context_channels, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=attention_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.set_interaction = nn.TransformerEncoder(layer, num_layers=set_layers)
        self.attr_head = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.global_head = nn.Linear(hidden_dim, 1)
        self.background_head = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.normal_(self.attr_head[-1].weight,mean=0.0,std=0.02)
        nn.init.zeros_(self.attr_head[-1].bias)
        nn.init.constant_(self.background_head.bias, -4.0)

    def forward(self, enhancer_maps, gene_context, enhancer_mask=None, global_coefficients=None,
                global_intercept=None, spatial_strength=1.0):
        """
        enhancer_maps: [B,N,C,H,W]; gene_context: [B,Cg,H,W].
        enhancer_mask: [B,N], True for real and False for padded enhancers.
        """
        if enhancer_maps.ndim != 5:
            raise ValueError("enhancer_maps must have shape [B,N,C,H,W]")
        b, n, c, h, w = enhancer_maps.shape
        if enhancer_mask is None:
            enhancer_mask = torch.ones((b, n), dtype=torch.bool, device=enhancer_maps.device)
        if enhancer_mask.shape != (b, n):
            raise ValueError("enhancer_mask must have shape [B,N]")
        ef = self.enhancer_encoder(enhancer_maps.reshape(b * n, c, h, w))
        ef = ef.reshape(b, n, -1, h, w)
        gf = self.gene_encoder(gene_context)
        tokens = ef.mean(dim=(-1, -2))
        tokens = self.set_interaction(tokens, src_key_padding_mask=~enhancer_mask)
        token_maps = tokens[..., None, None].expand(-1, -1, -1, h, w)
        gene_maps = gf[:, None].expand(-1, n, -1, -1, -1)
        joint = torch.cat([ef + token_maps, gene_maps], dim=2).reshape(b * n, -1, h, w)
        local_logits = self.attr_head(joint).reshape(b, n, h, w)
        raw_activity = enhancer_maps[:, :, 0].clamp_min(0)
        learned_global_scale = F.softplus(self.global_head(tokens)).squeeze(-1)
        global_scale = learned_global_scale if global_coefficients is None else global_coefficients
        # With zero local modulation this is a global additive model.
        local_modulation = torch.tanh(local_logits)
        strength = torch.as_tensor(spatial_strength,dtype=enhancer_maps.dtype,device=enhancer_maps.device).reshape(-1,1,1,1)
        global_attribution = global_scale[..., None, None]
        # The spatial branch may revive a candidate rejected by the global fit.
        # strength=0 remains exactly the fixed Elastic Net baseline.
        candidate_scale = learned_global_scale
        spatial_attribution = candidate_scale[..., None, None] * (1.0 + local_modulation)
        attribution = global_attribution + strength * (spatial_attribution - global_attribution)
        attribution = attribution * enhancer_mask[..., None, None].to(attribution.dtype)
        contributions = raw_activity * attribution
        background = strength * F.softplus(self.background_head(gf))
        intercept = 0.0 if global_intercept is None else global_intercept[..., None, None]
        mixture = contributions.sum(dim=1, keepdim=True) + intercept + background
        prediction = mixture
        global_prediction = (raw_activity * global_scale[..., None, None]).sum(dim=1, keepdim=True) + intercept
        return {
            "gene_prediction": prediction,
            "global_prediction": global_prediction,
            "attribution_maps": attribution,
            "background_map": background,
            "global_coefficients": global_scale,
            "local_modulation_maps": local_modulation,
            "attribution_logits": local_logits,
            "enhancer_value_maps": raw_activity,
            "enhancer_contribution_maps": contributions,
        }

