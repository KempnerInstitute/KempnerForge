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

    Works with both torchrun (RANK/LOCAL_RANK/WORLD_SIZE) and direct srun
    launch (SLURM_PROCID/SLURM_LOCALID/SLURM_NTASKS). When running under
    srun, also sets RANK/LOCAL_RANK/WORLD_SIZE so that PyTorch's env://
    rendezvous can find them.
    """
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))

    # Ensure standard env vars are set for PyTorch's env:// rendezvous
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))

    return rank, local_rank, world_size


def is_rank_zero() -> bool:
    """Check if the current process is rank 0."""
    return int(os.environ.get("RANK", "0")) == 0


def _detect_ib_interface() -> str | None:
    """Detect the first UP InfiniBand interface with an IP address.

    Returns interface name (e.g., "ib0") or None if no IB interface found.
    Used to set NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME so both backends
    bind to the high-speed IB network rather than management Ethernet.
    """
    try:
        import subprocess

        result = subprocess.run(["ip", "-br", "addr"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0].startswith("ib") and parts[1] == "UP":
                return parts[0]
    except Exception:
        pass
    return None


def _set_nccl_env() -> None:
    """Set NCCL and Gloo network environment variables.

    Detects the IB interface and configures both NCCL (for RDMA data transport
    bootstrap) and Gloo (for DCP async checkpoint coordination) to bind to it.
    Without this, Gloo defaults to management Ethernet which can cause timeouts
    for multi-node collectives.
    """
    # Detect IB interface if not already set by launch script
    ib_iface = os.environ.get("NCCL_SOCKET_IFNAME") or _detect_ib_interface()
    if ib_iface:
        os.environ.setdefault("NCCL_SOCKET_IFNAME", ib_iface)
        os.environ.setdefault("GLOO_SOCKET_IFNAME", ib_iface)

    # Use InfiniBand for inter-node communication when available
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    os.environ.setdefault("NCCL_NET_GDR_LEVEL", "2")


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

    # Set MASTER_ADDR/MASTER_PORT from SLURM env vars if not already set.
    # torchrun sets these automatically, but srun-direct launch does not.
    if "MASTER_ADDR" not in os.environ and "SLURM_JOB_NODELIST" in os.environ:
        import subprocess

        result = subprocess.run(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            capture_output=True,
            text=True,
        )
        os.environ["MASTER_ADDR"] = result.stdout.strip().split("\n")[0]
    os.environ.setdefault("MASTER_ADDR", "localhost")
    # Pick a port if not set. For multi-node SLURM jobs, the launch script
    # (multinode.sh) sets MASTER_PORT before srun so all ranks get the same
    # value. This fallback handles single-node and interactive use.
    # Under SLURM, derive port deterministically from job ID so all ranks
    # on the same job agree without communication, while different jobs on
    # the same shared HPC node get different ports.
    if "MASTER_PORT" not in os.environ:
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id is not None:
            import random

            os.environ["MASTER_PORT"] = str(random.Random(int(job_id)).randint(15000, 30000))
        else:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])

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
    # to avoid unnecessary process groups for degenerate dimensions.
    # Exception: always include dp_shard when TP is active, even if dp_shard=1.
    # FSDP2 wrapping converts all params to DTensors, which the optimizer needs
    # when TP has already made some params DTensors.
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

    # When TP is active, ensure dp_shard is in the mesh so FSDP2 can wrap
    # all parameters as DTensors (required for fused optimizer compatibility).
    if resolved.tp > 1 and "dp_shard" not in mesh_dims:
        # Insert dp_shard before tp (mesh order: pp, dp_replicate, dp_shard, tp)
        tp_idx = mesh_dims.index("tp")
        mesh_dims.insert(tp_idx, "dp_shard")
        mesh_sizes.insert(tp_idx, 1)

    # If all dimensions are 1 (pure single-dim), create a flat DP mesh
    if not mesh_dims:
        mesh_dims = ["dp_shard"]
        mesh_sizes = [world_size]

    if is_rank_zero():
        logger.info(f"DeviceMesh: dims={mesh_dims}, sizes={mesh_sizes}, world_size={world_size}")

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
