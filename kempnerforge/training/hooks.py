"""Training loop hooks for extensibility without forking train.py.

Researchers subclass ``TrainingHook`` and override only the methods they need.
Hooks run at defined points in the training loop; when no hooks are registered,
the overhead is a single empty-list check per call site.

Usage:
    from kempnerforge.training.hooks import TrainingHook, StepContext

    class GradHistogramHook(TrainingHook):
        def on_step_end(self, ctx: StepContext) -> None:
            for name, p in ctx.model.named_parameters():
                if p.grad is not None:
                    wandb.log({f"grad_norm/{name}": p.grad.norm().item()}, step=ctx.step)

    hooks = [GradHistogramHook()]
    runner = HookRunner(hooks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from kempnerforge.config.job import JobConfig


@dataclass
class StepContext:
    """Per-step context passed to hooks after each training step."""

    step: int
    loss: float
    grad_norm: float
    lr: float
    tokens_seen: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer


class TrainingHook:
    """Base class for training hooks. Override only the methods you need."""

    def on_train_begin(self, config: JobConfig) -> None:
        """Called once after setup, before the training loop starts."""

    def on_step_end(self, ctx: StepContext) -> None:
        """Called after each optimizer step + metrics logging."""

    def on_eval_end(self, metrics: dict[str, float], step: int) -> None:
        """Called after each evaluation round completes."""

    def on_checkpoint_save(self, step: int, path: str) -> None:
        """Called after a checkpoint is saved."""

    def on_train_end(self, step: int, tokens_seen: int) -> None:
        """Called after the training loop exits."""


class HookRunner:
    """Dispatches hook calls to registered hooks. Zero cost when empty."""

    def __init__(self, hooks: list[TrainingHook] | None = None) -> None:
        self.hooks: list[TrainingHook] = hooks or []

    def on_train_begin(self, config: JobConfig) -> None:
        if not self.hooks:
            return
        for hook in self.hooks:
            hook.on_train_begin(config)

    def on_step_end(self, ctx: StepContext) -> None:
        if not self.hooks:
            return
        for hook in self.hooks:
            hook.on_step_end(ctx)

    def on_eval_end(self, metrics: dict[str, float], step: int) -> None:
        if not self.hooks:
            return
        for hook in self.hooks:
            hook.on_eval_end(metrics, step)

    def on_checkpoint_save(self, step: int, path: str) -> None:
        if not self.hooks:
            return
        for hook in self.hooks:
            hook.on_checkpoint_save(step, path)

    def on_train_end(self, step: int, tokens_seen: int) -> None:
        if not self.hooks:
            return
        for hook in self.hooks:
            hook.on_train_end(step, tokens_seen)
