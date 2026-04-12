"""Example training hooks for KempnerForge.

Demonstrates how to extend the training loop without forking train.py.
Subclass ``TrainingHook`` and override only the methods you need.

Usage:
    # In your training script or config setup:
    from examples.custom_hook import GradNormHistogramHook, LearningDynamicsHook

    hooks = [GradNormHistogramHook(), LearningDynamicsHook()]
    hook_runner = HookRunner(hooks)
"""

from __future__ import annotations

import torch

from kempnerforge.training.hooks import StepContext, TrainingHook


class GradNormHistogramHook(TrainingHook):
    """Log per-layer gradient norms to WandB every N steps.

    Useful for diagnosing vanishing/exploding gradients, layer-specific
    learning dynamics, and verifying that all layers receive gradients.
    """

    def __init__(self, interval: int = 100) -> None:
        self.interval = interval

    def on_step_end(self, ctx: StepContext) -> None:
        if ctx.step % self.interval != 0:
            return

        grad_norms: dict[str, float] = {}
        for name, param in ctx.model.named_parameters():
            if param.grad is not None:
                grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(grad_norms, step=ctx.step)
        except ImportError:
            pass


class LearningDynamicsHook(TrainingHook):
    """Track weight norms and gradient signal-to-noise ratio.

    Reports per-layer weight norms and gradient SNR (mean/std of gradients),
    which helps identify layers that are undertrained or have noisy gradients.
    """

    def __init__(self, interval: int = 100) -> None:
        self.interval = interval

    def on_step_end(self, ctx: StepContext) -> None:
        if ctx.step % self.interval != 0:
            return

        metrics: dict[str, float] = {}
        for name, param in ctx.model.named_parameters():
            metrics[f"weight_norm/{name}"] = param.data.norm().item()
            if param.grad is not None:
                grad = param.grad
                mean = grad.mean().item()
                std = grad.std().item()
                metrics[f"grad_snr/{name}"] = abs(mean) / (std + 1e-8)

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=ctx.step)
        except ImportError:
            pass


class EarlyStoppingHook(TrainingHook):
    """Stop training if eval loss hasn't improved for ``patience`` evaluations."""

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.evals_without_improvement = 0

    def on_eval_end(self, metrics: dict[str, float], step: int) -> None:
        loss = metrics.get("eval/loss", float("inf"))
        if loss < self.best_loss:
            self.best_loss = loss
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1

        if self.evals_without_improvement >= self.patience:
            raise KeyboardInterrupt(
                f"Early stopping: no improvement for {self.patience} evals "
                f"(best={self.best_loss:.4f})"
            )


class ExpertLoadBalanceHook(TrainingHook):
    """Log MoE expert utilization statistics for load balance debugging."""

    def __init__(self, interval: int = 50) -> None:
        self.interval = interval

    def on_step_end(self, ctx: StepContext) -> None:
        if ctx.step % self.interval != 0:
            return

        if not hasattr(ctx.model, "get_expert_counts"):
            return

        counts = ctx.model.get_expert_counts()
        if not counts:
            return

        all_counts = torch.stack(list(counts.values())).float()
        metrics = {
            "moe_hook/expert_max": all_counts.max().item(),
            "moe_hook/expert_min": all_counts.min().item(),
            "moe_hook/expert_std": all_counts.std().item(),
            "moe_hook/expert_balance": (all_counts.min() / (all_counts.max() + 1e-8)).item(),
        }

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=ctx.step)
        except ImportError:
            pass
