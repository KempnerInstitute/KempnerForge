"""Distributed training utilities.

Gradient clipping and data-parallel helpers that work correctly with
FSDP2 and multi-dimensional parallelism.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)


def get_dp_info(device_mesh: DeviceMesh | None) -> tuple[int, int]:
    """Get (dp_rank, dp_size) from the device mesh, accounting for PP/TP/EP.

    Only ``dp_replicate`` and ``dp_shard`` split the data, so ranks that differ
    only in ``pp``, ``ep`` or ``tp`` share a DP rank. Under HSDP the DP rank is
    the row-major index of the rank's ``(dp_replicate, dp_shard)`` coordinates
    and the DP size is their product; no process group is created.

    Args:
        device_mesh: Full DeviceMesh, or None for single-GPU.

    Returns:
        Tuple of (dp_rank, dp_world_size); ``(0, 1)`` when there is no mesh or
        the mesh has no DP dimension.
    """
    if device_mesh is None:
        return 0, 1
    dim_names = device_mesh.mesh_dim_names
    dp_rank, dp_size = 0, 1
    for name in ("dp_replicate", "dp_shard"):
        if name in dim_names:  # type: ignore[reportOperatorIssue]
            size = device_mesh[name].size()
            dp_rank = dp_rank * size + device_mesh.get_local_rank(name)
            dp_size *= size
    return dp_rank, dp_size


def data_parallel_groups(device_mesh: DeviceMesh | None) -> tuple[dist.ProcessGroup, ...]:
    """The process groups whose sums, taken in turn, total a value over the DP ranks.

    One group for each of ``dp_shard`` and ``dp_replicate`` with more than one rank,
    taken from the mesh, so none is created. Ranks that differ only in ``pp``, ``ep``
    or ``tp`` share a DP rank, hold the same data, and are in none of them. Empty
    without a mesh or with one DP rank.

    Args:
        device_mesh: Full DeviceMesh, or None for single-GPU.

    Returns:
        The groups, for ``dp_all_reduce_`` and ``dp_mean``.
    """
    if device_mesh is None:
        return ()
    names = device_mesh.mesh_dim_names or ()
    groups = []
    for name in ("dp_shard", "dp_replicate"):
        if name in names:
            submesh = device_mesh[name]
            if submesh.size() > 1:
                groups.append(submesh.get_group())
    return tuple(groups)


def dp_all_reduce_(tensor: torch.Tensor, groups: Sequence[dist.ProcessGroup]) -> torch.Tensor:
    """Sum ``tensor`` in place over the DP ranks that ``groups`` span, and return it.

    ``groups`` comes from ``data_parallel_groups``; every rank in them must call this.
    """
    for group in groups:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def dp_mean(value: float, groups: Sequence[dist.ProcessGroup], device: torch.device) -> float:
    """Mean of ``value`` over the DP ranks that ``groups`` span; ``value`` itself when empty.

    The sum is taken in float64. Every rank in ``groups`` must call this.
    """
    if not groups:
        return value
    total = torch.tensor([value], dtype=torch.float64, device=device)
    dp_size = math.prod(dist.get_world_size(group) for group in groups)
    return dp_all_reduce_(total, groups).item() / dp_size


def clip_grad_norm_(
    model: torch.nn.Module,
    max_norm: float,
    foreach: bool = True,
    pp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Clip gradient norm across all parameters.

    Handles mixed DTensor meshes that arise when TP+FSDP produces parameters on
    different meshes (e.g., TP-sharded linears on (dp_shard, tp) vs FSDP-only
    norms on (dp_shard)). Groups gradients by mesh, computes per-group norms
    via stack, then combines across groups.

    Falls back to ``torch.nn.utils.clip_grad_norm_`` when there is only one mesh
    (pure FSDP or single-GPU).

    Under pipeline parallelism ``model`` holds one stage, so pass the stages'
    process group as ``pp_group``: the squared stage norms are summed over it, and
    every stage returns the whole model's norm and scales by the same coefficient.
    Every stage must make the call, including one without gradients, since the
    sum is a collective.

    Args:
        model: Model whose gradients to clip.
        max_norm: Maximum gradient norm.
        foreach: Use the faster foreach implementation.
        pp_group: Process group of the pipeline stages, or None without
            pipeline parallelism.

    Returns:
        Total gradient norm (before clipping).
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if pp_group is not None:
        device = grads[0].device if grads else next(model.parameters()).device
        norm_sq = _norm_sq(grads, device)
        dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM, group=pp_group)
        return _scale_(grads, max_norm, norm_sq.sqrt())
    if not grads:
        return torch.tensor(0.0)

    mesh_keys = {_mesh_key(g) for g in grads}

    if len(mesh_keys) <= 1:
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
            foreach=foreach,
        )

    return _scale_(grads, max_norm, _norm_sq(grads, grads[0].device).sqrt())


def _mesh_key(g: torch.Tensor) -> int:
    """Identity of the mesh a DTensor gradient lives on; 0 for a plain tensor."""
    spec = getattr(g, "_spec", None)
    return id(spec.mesh) if spec is not None else 0


def _norm_sq(grads: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Squared L2 norm of ``grads`` as a plain tensor on ``device``.

    Gradients are grouped by mesh and each group's norm is computed via stack. A
    DTensor gradient's norm is partial over its shards, and ``.pow(2)``
    all-reduces it into a replicated tensor, so ``full_tensor()`` only unwraps
    the group's sum before the groups are added. Each gradient's norm is taken
    in at least fp32: squared in fp16, a norm above about 256 overflows to inf,
    and the clip coefficient would then zero every gradient. fp64 gradients keep
    fp64, since ``vector_norm`` refuses to narrow them.
    """
    groups: dict[int, list[torch.Tensor]] = defaultdict(list)
    for g in grads:
        groups[_mesh_key(g)].append(g.detach())

    total_norm_sq = torch.tensor(0.0, device=device)
    for group_grads in groups.values():
        norms = [
            torch.linalg.vector_norm(g, 2.0, dtype=torch.promote_types(g.dtype, torch.float32))
            for g in group_grads
        ]
        group_norm_sq = torch.stack(norms).pow(2).sum()
        if hasattr(group_norm_sq, "full_tensor"):
            group_norm_sq = group_norm_sq.full_tensor()  # type: ignore[reportAttributeAccessIssue]
        total_norm_sq = total_norm_sq + group_norm_sq
    return total_norm_sq


def _scale_(grads: list[torch.Tensor], max_norm: float, total_norm: torch.Tensor) -> torch.Tensor:
    """Scale ``grads`` in place to a norm of at most ``max_norm``; return ``total_norm``."""
    clip_coef_clamped = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for g in grads:
        g.mul_(clip_coef_clamped)
    return total_norm
