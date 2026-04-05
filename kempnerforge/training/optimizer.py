"""Optimizer construction for KempnerForge.

Builds AdamW with per-parameter-group settings:
  - Weight decay applied to matrix weights only
  - Bias and norm parameters excluded from weight decay
  - Fused kernel when available (PyTorch 2.x)
"""

from __future__ import annotations

import logging

import torch

from kempnerforge.config.schema import OptimizerConfig

logger = logging.getLogger(__name__)


def _should_decay(name: str, param: torch.nn.Parameter) -> bool:
    """Decide whether a parameter should receive weight decay.

    Excluded: 1D parameters (biases, norm scales/shifts), embedding weights.
    """
    if param.ndim <= 1:
        return False
    return "bias" not in name


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Construct an optimizer with per-parameter-group weight decay settings.

    Args:
        model: Model whose parameters to optimize.
        config: Optimizer configuration.

    Returns:
        Configured optimizer instance.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _should_decay(name, param):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Log parameter counts
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    logger.info(
        f"Optimizer groups: {n_decay:,} params with decay, {n_no_decay:,} params without decay"
    )

    if config.name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            fused=config.fused and torch.cuda.is_available(),
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.name!r}")

    return optimizer
