"""Gradient utilities for distributed training.

Handles gradient accumulation with no_sync context for skipping
redundant all-reduces during micro-batching.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator

import torch
from torch.distributed.fsdp import FSDPModule


def _gradient_sync_switch(model: torch.nn.Module) -> Callable[[bool], None] | None:
    """The call that turns gradient sync on or off for every FSDP2 unit of ``model``.

    A model with its own ``set_requires_gradient_sync``, such as one whose root is an
    FSDP2 module, is switched through it. Any other model is switched through each FSDP2
    module it holds, one unit at a time, which reaches components sharded on their own
    and modules behind a ``torch.compile`` wrapper. None when the model holds no FSDP2
    module.
    """
    switch = getattr(model, "set_requires_gradient_sync", None)
    if switch is not None:
        return switch
    units = [module for module in model.modules() if isinstance(module, FSDPModule)]
    if not units:
        return None

    def switch_units(requires_gradient_sync: bool) -> None:
        for unit in units:
            unit.set_requires_gradient_sync(requires_gradient_sync, recurse=False)

    return switch_units


@contextlib.contextmanager
def maybe_no_sync(
    model: torch.nn.Module,
    micro_step: int,
    grad_accum_steps: int,
) -> Generator[None, None, None]:
    """Skip gradient sync on intermediate accumulation steps.

    On the last micro-step, gradients are synchronized normally. On earlier
    micro-steps, every FSDP2 unit of ``model`` skips its reduce-scatter (and,
    under HSDP, its all-reduce) and keeps its unsharded gradient until the last
    micro-step reduces the sum. A model whose root is an FSDP2 module is switched
    through its root; any other model, such as a ``VLMWrapper`` whose components
    are sharded on their own, through each FSDP2 module it holds, also behind a
    ``torch.compile`` wrapper. Sync is turned back on when the block exits.
    For a model without FSDP2 modules, this is a no-op.
    """
    is_last_step = (micro_step + 1) == grad_accum_steps
    switch = None if is_last_step else _gradient_sync_switch(model)

    if switch is None:
        yield
    else:
        switch(False)
        try:
            yield
        finally:
            switch(True)
