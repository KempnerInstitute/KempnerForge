"""Distributed training utilities.

Gradient clipping, reductions, and synchronization helpers that work
correctly with FSDP2 and multi-dimensional parallelism.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)


def clip_grad_norm_(
    model: torch.nn.Module,
    max_norm: float,
    foreach: bool = True,
) -> torch.Tensor:
    """Clip gradient norm across all parameters.

    FSDP2-compatible: ``torch.nn.utils.clip_grad_norm_`` works correctly
    with FSDP2 because gradients are exposed as unsharded DTensor views.

    Args:
        model: Model whose gradients to clip.
        max_norm: Maximum gradient norm.
        foreach: Use the faster foreach implementation.

    Returns:
        Total gradient norm (before clipping).
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        foreach=foreach,
    )


def dist_reduce(
    tensor: torch.Tensor,
    op: str = "mean",
    mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """All-reduce a tensor across ranks.

    Args:
        tensor: Tensor to reduce (modified in-place).
        op: Reduction operation — "sum", "mean", "max", or "min".
        mesh: Optional 1D DeviceMesh for mesh-aware reduction.
            If None, reduces across the default process group.

    Returns:
        The reduced tensor.
    """
    if not dist.is_initialized():
        return tensor

    reduce_ops = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    if op not in reduce_ops:
        raise ValueError(f"Unknown reduce op: {op!r}. Choose from {list(reduce_ops)}")

    group = mesh.get_group() if mesh is not None else None
    world_size = mesh.size() if mesh is not None else dist.get_world_size()

    dist.all_reduce(tensor, op=reduce_ops[op], group=group)

    if op == "mean":
        tensor.div_(world_size)

    return tensor


def barrier_with_timeout(
    timeout_sec: int = 300,
    mesh: DeviceMesh | None = None,
) -> None:
    """Distributed barrier with a configurable timeout.

    Uses ``monitored_barrier`` for the default process group (gives clear
    error messages showing which rank is stuck). Falls back to standard
    barrier for custom groups.

    Args:
        timeout_sec: Timeout in seconds.
        mesh: Optional DeviceMesh for mesh-aware barrier.
    """
    if not dist.is_initialized():
        return

    if mesh is not None:
        dist.barrier(group=mesh.get_group())
    else:
        dist.monitored_barrier(timeout=timedelta(seconds=timeout_sec))
