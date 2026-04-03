"""Learning rate schedulers for KempnerForge.

All schedulers are step-based (not epoch-based) and compose a warmup
phase with a decay phase:

  - Cosine: warmup → cosine decay to min_lr
  - Linear: warmup → linear decay to min_lr
  - WSD:    warmup → stable (constant LR) → cosine decay to min_lr
"""

from __future__ import annotations

import math

import torch

from kempnerforge.config.schema import SchedulerConfig, SchedulerType


def _warmup_factor(step: int, warmup_steps: int) -> float:
    """Linear warmup from 0 to 1 over warmup_steps."""
    if warmup_steps == 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def _cosine_decay(step: int, total_steps: int, min_ratio: float) -> float:
    """Cosine decay from 1 to min_ratio over total_steps."""
    if total_steps <= 0:
        return 1.0
    progress = min(1.0, step / total_steps)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def _linear_decay(step: int, total_steps: int, min_ratio: float) -> float:
    """Linear decay from 1 to min_ratio over total_steps."""
    if total_steps <= 0:
        return 1.0
    progress = min(1.0, step / total_steps)
    return 1.0 - (1.0 - min_ratio) * progress


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a LR scheduler from config.

    Args:
        optimizer: Optimizer to schedule.
        config: Scheduler configuration.
        max_steps: Total training steps (used to compute decay length).

    Returns:
        PyTorch LambdaLR scheduler.
    """
    warmup = config.warmup_steps
    min_ratio = config.min_lr_ratio

    if config.name == SchedulerType.cosine:
        decay_steps = config.decay_steps or (max_steps - warmup)

        def lr_fn(step: int) -> float:
            if step < warmup:
                return _warmup_factor(step, warmup)
            return _cosine_decay(step - warmup, decay_steps, min_ratio)

    elif config.name == SchedulerType.linear:
        decay_steps = config.decay_steps or (max_steps - warmup)

        def lr_fn(step: int) -> float:
            if step < warmup:
                return _warmup_factor(step, warmup)
            return _linear_decay(step - warmup, decay_steps, min_ratio)

    elif config.name == SchedulerType.wsd:
        stable = config.stable_steps or 0
        decay_steps = config.decay_steps or (max_steps - warmup - stable)

        def lr_fn(step: int) -> float:
            if step < warmup:
                return _warmup_factor(step, warmup)
            if step < warmup + stable:
                return 1.0
            return _cosine_decay(step - warmup - stable, decay_steps, min_ratio)

    else:
        raise ValueError(f"Unknown scheduler: {config.name!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
