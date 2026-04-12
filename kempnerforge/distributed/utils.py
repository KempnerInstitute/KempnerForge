"""Distributed training utilities.

Gradient clipping and data-parallel helpers that work correctly with
FSDP2 and multi-dimensional parallelism.
"""

from __future__ import annotations

import logging

import torch
from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)


def get_dp_info(device_mesh: DeviceMesh | None) -> tuple[int, int]:
    """Get (dp_rank, dp_size) from the device mesh, accounting for PP/TP.

    Handles all DeviceMesh configurations: HSDP (dp_replicate + dp_shard),
    pure FSDP (dp_shard only), replicate-only, or no mesh (single GPU).

    Args:
        device_mesh: Full DeviceMesh, or None for single-GPU.

    Returns:
        Tuple of (dp_rank, dp_world_size).
    """
    if device_mesh is None:
        return 0, 1
    dim_names = device_mesh.mesh_dim_names
    if "dp_shard" in dim_names and "dp_replicate" in dim_names:  # type: ignore[reportOperatorIssue]
        dp_mesh = device_mesh["dp_replicate", "dp_shard"]
        return dp_mesh.get_local_rank(), dp_mesh.size()
    elif "dp_shard" in dim_names:  # type: ignore[reportOperatorIssue]
        dp_mesh = device_mesh["dp_shard"]
        return dp_mesh.get_local_rank(), dp_mesh.size()
    elif "dp_replicate" in dim_names:  # type: ignore[reportOperatorIssue]
        dp_mesh = device_mesh["dp_replicate"]
        return dp_mesh.get_local_rank(), dp_mesh.size()
    return 0, 1


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
            group_norm_sq = group_norm_sq.full_tensor()  # type: ignore[reportAttributeAccessIssue]
        total_norm_sq = total_norm_sq + group_norm_sq

    total_norm = total_norm_sq.sqrt()

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    for g in grads:
        g.mul_(clip_coef_clamped)

    return total_norm
