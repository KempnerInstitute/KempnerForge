"""Gradient utilities for distributed training.

Handles gradient accumulation with no_sync context for skipping
redundant all-reduces during micro-batching.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator

import torch


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
        model.set_requires_gradient_sync(False)  # type: ignore[reportCallIssue]
        try:
            yield
        finally:
            model.set_requires_gradient_sync(True)  # type: ignore[reportCallIssue]
