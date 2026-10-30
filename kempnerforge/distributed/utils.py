"""Distributed training utilities.

Gradient clipping and data-parallel helpers that work correctly with
FSDP2 and multi-dimensional parallelism.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from kempnerforge.distributed.expert_parallel import expert_parallel_params

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

    Under expert parallelism each EP rank holds its own routed experts and a copy
    of every other parameter. The experts' squared norms are summed over the EP
    group the ``MoEMLP`` layers record, and the copies' are averaged over it, so
    every EP rank returns the whole model's norm and scales by the same
    coefficient. Every EP rank must make the call.

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
    expert_params, ep_group = expert_parallel_params(model)
    if pp_group is not None or ep_group is not None:
        device = grads[0].device if grads else next(model.parameters()).device
        if ep_group is None:
            norm_sq = _norm_sq(grads, device)
        else:
            norm_sq = _ep_norm_sq(model, expert_params, ep_group, device)
        if pp_group is not None:
            dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM, group=pp_group)
        return _scale_(grads, max_norm, norm_sq.sqrt())
    if not grads:
        return torch.tensor(0.0)

    # Check if all gradients share the same mesh (or are plain tensors).
    # If so, use the fast standard path.
    mesh_keys = {_mesh_key(g) for g in grads}

    if len(mesh_keys) <= 1:
        # Single mesh — standard path works
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


def _ep_norm_sq(
    model: torch.nn.Module,
    expert_params: list[torch.nn.Parameter],
    ep_group: dist.ProcessGroup,
    device: torch.device,
) -> torch.Tensor:
    """Squared L2 norm of the whole model's gradients under expert parallelism.

    One all-reduce over ``ep_group`` sums the squared norms of each rank's routed
    experts and averages those of the parameters every EP rank holds a copy of, so
    every EP rank gets the same value even if the copies differ in the last bits.
    """
    expert_ids = {id(p) for p in expert_params}
    expert = [p.grad for p in expert_params if p.grad is not None]
    shared = [p.grad for p in model.parameters() if p.grad is not None and id(p) not in expert_ids]
    norm_sq = _norm_sq(shared, device) / dist.get_world_size(ep_group) + _norm_sq(expert, device)
    dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM, group=ep_group)
    return norm_sq


def _scale_(grads: list[torch.Tensor], max_norm: float, total_norm: torch.Tensor) -> torch.Tensor:
    """Scale ``grads`` in place to a norm of at most ``max_norm``; return ``total_norm``."""
    clip_coef_clamped = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for g in grads:
        g.mul_(clip_coef_clamped)
    return total_norm
