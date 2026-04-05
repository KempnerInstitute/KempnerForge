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

    Handles mixed DTensor meshes that arise when TP+FSDP produces parameters on
    different meshes (e.g., TP-sharded linears on (dp_shard, tp) vs FSDP-only
    norms on (dp_shard)). Groups gradients by mesh, computes per-group norms
    via stack, then combines across groups.

    Falls back to ``torch.nn.utils.clip_grad_norm_`` when there is only one mesh
    (pure FSDP or single-GPU).

    Args:
        model: Model whose gradients to clip.
        max_norm: Maximum gradient norm.
        foreach: Use the faster foreach implementation.

    Returns:
        Total gradient norm (before clipping).
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    # Check if all gradients share the same mesh (or are plain tensors).
    # If so, use the fast standard path.
    def _mesh_key(g: torch.Tensor) -> int:
        spec = getattr(g, "_spec", None)
        return id(spec.mesh) if spec is not None else 0

    mesh_keys = {_mesh_key(g) for g in grads}

    if len(mesh_keys) <= 1:
        # Single mesh — standard path works
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
            foreach=foreach,
        )

    # Mixed meshes — group by mesh, compute per-group norms, combine.
    from collections import defaultdict

    groups: dict[int, list[torch.Tensor]] = defaultdict(list)
    for g in grads:
        groups[_mesh_key(g)].append(g.detach())

    total_norm_sq = torch.tensor(0.0, device=grads[0].device)
    for group_grads in groups.values():
        norms = torch.stack([g.norm(2.0) for g in group_grads])
        group_norm_sq = norms.pow(2).sum()
        # DTensor norm is a partial sum — full_tensor() does the all-reduce
        if hasattr(group_norm_sq, "full_tensor"):
            group_norm_sq = group_norm_sq.full_tensor()
        total_norm_sq = total_norm_sq + group_norm_sq

    total_norm = total_norm_sq.sqrt()

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    for g in grads:
        g.mul_(clip_coef_clamped)

    return total_norm


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
