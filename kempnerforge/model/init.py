"""Weight initialization strategies for KempnerForge models."""

from __future__ import annotations

import math

import torch.nn as nn

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.moe import MoEMLP
from kempnerforge.model.mot import MoTAttention, MoTBlock


def init_weights(model: nn.Module, config: ModelConfig) -> None:
    """Apply standard initialization to all parameters in a model.

    Strategy (following GPT-2/Llama conventions):

    - Linear layers: normal(0, 0.02)
    - Embedding layers: normal(0, 0.02)
    - Residual output projections (o_proj, down_proj): scaled by 1/sqrt(2 * n_layers)
    - Cross-attention block residuals: zero-initialized (identity-at-init warm-start)
    - MoT per-modality residual projections: zero-initialized (identity-at-construction)
    - Norm layers: weight=1 (already default)
    """
    std = config.init_std
    residual_scale = 1.0 / math.sqrt(2.0 * config.n_layers)

    for name, param in model.named_parameters():
        if param.is_meta:
            continue
        if param.dim() < 2:
            # Bias and norm parameters: leave at default (zeros / ones)
            continue

        # Cross-attention block residuals get zero-init so the block is
        # identity at construction; downstream training learns a
        # non-zero contribution from there.
        if name.startswith("cross_attention_layers.") and name.endswith(
            ("o_proj.weight", "down_proj.weight")
        ):
            nn.init.zeros_(param)
        # Residual projections elsewhere get scaled init to prevent signal growth
        elif name.endswith(("o_proj.weight", "down_proj.weight")):
            nn.init.normal_(param, mean=0.0, std=std * residual_scale)
        else:
            nn.init.normal_(param, mean=0.0, std=std)

    # MoT-specific zero-init for identity-at-construction residual.
    # These FQNs do not match endswith("o_proj.weight") (per-modality
    # ModuleDict nesting puts {modality} after o_proj), so they are
    # handled by a second pass over modules.
    for module in model.modules():
        if isinstance(module, MoTAttention):
            for m in module.modalities:
                nn.init.zeros_(module.o_proj[m].weight)  # type: ignore[reportArgumentType]
        elif isinstance(module, MoTBlock):
            for m in module.modalities:
                mlp_m = module.mlp[m]
                if isinstance(mlp_m, MoEMLP):
                    continue
                if hasattr(mlp_m, "down_proj"):
                    nn.init.zeros_(mlp_m.down_proj.weight)  # type: ignore[union-attr]
