"""Gradient utilities for distributed training.

Handles gradient accumulation with correct loss scaling, no_sync context
for skipping redundant all-reduces, and distributed gradient clipping.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator

import torch
import torch.distributed as dist


@contextlib.contextmanager
def maybe_no_sync(
    model: torch.nn.Module,
    micro_step: int,
    grad_accum_steps: int,
) -> Generator[None, None, None]:
    """Skip gradient sync on intermediate accumulation steps.

    On the last micro-step, gradients are synchronized normally.
    On earlier micro-steps, sync is skipped to avoid redundant all-reduces.

    Works with FSDP2 (which implements ``no_sync()`` on the module).
    For non-distributed models, this is a no-op.
    """
    is_last_step = (micro_step + 1) == grad_accum_steps

    if is_last_step or not hasattr(model, "set_requires_gradient_sync"):
        yield
    else:
        # FSDP2 uses set_requires_gradient_sync instead of no_sync() context
        model.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            model.set_requires_gradient_sync(True)


def scale_grads_by_token_count(
    model: torch.nn.Module,
    local_token_count: int,
) -> int:
    """Scale gradients by global token count for correct accumulation.

    When using gradient accumulation with variable-length sequences,
    we need to normalize by the total number of tokens across all
    DP ranks and accumulation steps.

    Args:
        model: Model with accumulated gradients.
        local_token_count: Tokens processed by this rank in this accumulation window.

    Returns:
        Global token count (summed across all ranks).
    """
    if dist.is_initialized():
        token_tensor = torch.tensor([local_token_count], dtype=torch.long, device="cuda")
        dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
        global_tokens = token_tensor.item()
    else:
        global_tokens = local_token_count

    if global_tokens > 0:
        scale = 1.0 / global_tokens
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)

    return global_tokens
