"""Distributed process group and DeviceMesh initialization.

Sets up torch.distributed, CUDA devices, and constructs a DeviceMesh
with named parallelism dimensions based on DistributedConfig.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from kempnerforge.config.schema import DistributedConfig

logger = logging.getLogger(__name__)


def get_world_info() -> tuple[int, int, int]:
    """Return (rank, local_rank, world_size) from environment variables.

    Works with both torchrun and SLURM launchers.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def is_rank_zero() -> bool:
    """Check if the current process is rank 0."""
    return int(os.environ.get("RANK", "0")) == 0


def _set_nccl_env() -> None:
    """Set NCCL environment variables for optimal performance."""
    # Prevent NCCL memory stacking — avoids OOM from cached NCCL buffers
    os.environ.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")


def _set_seed(seed: int, rank: int, pp_rank: int = 0) -> None:
    """Set deterministic seeds for reproducibility.

    - Same seed across data-parallel replicas (for consistent dropout)
    - Different seed across pipeline stages (for stochastic depth variation)
    """
    effective_seed = seed + pp_rank
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)


def init_distributed(config: DistributedConfig, seed: int = 42) -> DeviceMesh | None:
    """Initialize distributed training and build the DeviceMesh.

    Args:
        config: Distributed configuration with parallelism dimensions.
        seed: Random seed for reproducibility.

    Returns:
        DeviceMesh if world_size > 1, None for single-GPU.
    """
    rank, local_rank, world_size = get_world_info()

    # Single-GPU: no distributed setup needed
    if world_size == 1:
        torch.cuda.set_device(0)
        _set_seed(seed, rank=0)
        return None

    _set_nccl_env()

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=config.backend,
            timeout=timedelta(seconds=config.nccl_timeout_sec),
        )

    # Set CUDA device
    torch.cuda.set_device(local_rank)

    # Resolve parallelism dimensions
    resolved = config.resolve(world_size)

    # Build DeviceMesh — only include dimensions with size > 1
    # to avoid unnecessary process groups for degenerate dimensions
    mesh_dims: list[str] = []
    mesh_sizes: list[int] = []

    dim_map = [
        ("pp", resolved.pp),
        ("dp_replicate", resolved.dp_replicate),
        ("dp_shard", resolved.dp_shard),
        ("tp", resolved.tp),
    ]

    for name, size in dim_map:
        if size > 1:
            mesh_dims.append(name)
            mesh_sizes.append(size)

    # If all dimensions are 1 (pure single-dim), create a flat DP mesh
    if not mesh_dims:
        mesh_dims = ["dp_shard"]
        mesh_sizes = [world_size]

    if is_rank_zero():
        logger.info(
            f"DeviceMesh: dims={mesh_dims}, sizes={mesh_sizes}, world_size={world_size}"
        )

    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=tuple(mesh_sizes),
        mesh_dim_names=tuple(mesh_dims),
    )

    # Ensure all ranks have finished mesh creation before proceeding
    dist.barrier()

    # Set seed (vary by PP rank for different dropout/stochastic depth per stage)
    pp_rank = 0
    if "pp" in device_mesh.mesh_dim_names:
        pp_rank = device_mesh["pp"].get_local_rank()
    _set_seed(seed, rank=rank, pp_rank=pp_rank)

    return device_mesh


def destroy_distributed() -> None:
    """Clean up distributed process groups."""
    if dist.is_initialized():
        dist.destroy_process_group()
