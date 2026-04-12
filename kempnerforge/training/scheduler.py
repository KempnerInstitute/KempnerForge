"""Learning rate schedulers for KempnerForge.

All schedulers are step-based (not epoch-based) and compose a warmup
phase with a decay phase:

  - Cosine:    warmup → cosine decay to min_lr
  - Linear:    warmup → linear decay to min_lr
  - WSD:       warmup → stable → decay to min_lr (cosine/linear/sqrt cooldown)
  - Constant:  warmup → flat LR (for ablations)
  - REX:       warmup → polynomial decay (1 - t/T)^alpha
  - None:      constant factor=1.0 (for schedule-free optimizers)
"""

from __future__ import annotations

import math

import torch

from kempnerforge.config.registry import registry
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


def _sqrt_decay(step: int, total_steps: int, min_ratio: float) -> float:
    """Sqrt decay from 1 to min_ratio over total_steps.

    Stays higher than linear for longer, then drops more steeply near the end.
    Shape: ``min_ratio + (1 - min_ratio) * (1 - progress)^0.5``
    """
    if total_steps <= 0:
        return 1.0
    progress = min(1.0, step / total_steps)
    return min_ratio + (1.0 - min_ratio) * (1.0 - progress) ** 0.5


@registry.register_scheduler("cosine")
def _build_cosine(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = config.warmup_steps
    min_ratio = config.min_lr_ratio
    decay_steps = config.decay_steps or (max_steps - warmup)

    def lr_fn(step: int) -> float:
        if step < warmup:
            return _warmup_factor(step, warmup)
        return _cosine_decay(step - warmup, decay_steps, min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


@registry.register_scheduler("linear")
def _build_linear(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = config.warmup_steps
    min_ratio = config.min_lr_ratio
    decay_steps = config.decay_steps or (max_steps - warmup)

    def lr_fn(step: int) -> float:
        if step < warmup:
            return _warmup_factor(step, warmup)
        return _linear_decay(step - warmup, decay_steps, min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


@registry.register_scheduler("wsd")
def _build_wsd(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = config.warmup_steps
    min_ratio = config.min_lr_ratio
    stable = config.stable_steps or 0
    decay_steps = config.decay_steps or (max_steps - warmup - stable)

    decay_fn = {
        "cosine": _cosine_decay,
        "linear": _linear_decay,
        "sqrt": _sqrt_decay,
    }[config.wsd_decay_type]

    def lr_fn(step: int) -> float:
        if step < warmup:
            return _warmup_factor(step, warmup)
        if step < warmup + stable:
            return 1.0
        return decay_fn(step - warmup - stable, decay_steps, min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


@registry.register_scheduler("constant")
def _build_constant(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = config.warmup_steps

    def lr_fn(step: int) -> float:
        if step < warmup:
            return _warmup_factor(step, warmup)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


@registry.register_scheduler("rex")
def _build_rex(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = config.warmup_steps
    min_ratio = config.min_lr_ratio
    alpha = config.rex_alpha
    decay_steps = config.decay_steps or (max_steps - warmup)

    def lr_fn(step: int) -> float:
        if step < warmup:
            return _warmup_factor(step, warmup)
        progress = min(1.0, (step - warmup) / max(decay_steps, 1))
        return max(min_ratio, (1.0 - progress) ** alpha)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


@registry.register_scheduler("none")
def _build_none(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)


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
    name = config.name.value if isinstance(config.name, SchedulerType) else config.name
    builder = registry.get_scheduler(name)
    return builder(optimizer, config, max_steps)
