"""Distributed process group and DeviceMesh initialization.

Sets up torch.distributed, CUDA devices, and constructs a DeviceMesh
with named parallelism dimensions based on DistributedConfig.
"""

from __future__ import annotations

import logging
import os
import random as _random
from datetime import timedelta

import numpy as _np
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
    ib_iface = os.environ.get("NCCL_SOCKET_IFNAME") or _detect_ib_interface()
    if ib_iface:
        os.environ.setdefault("NCCL_SOCKET_IFNAME", ib_iface)
        os.environ.setdefault("GLOO_SOCKET_IFNAME", ib_iface)

    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    os.environ.setdefault("NCCL_NET_GDR_LEVEL", "2")

    existing = os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING")
    if existing == "0":
        logger.warning(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING=0 detected — NCCL timeouts "
            "are advisory; stuck collectives can hang indefinitely."
        )
    else:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def _barrier_with_timeout(seconds: float, reason: str) -> None:
    """dist.barrier with an explicit per-op timeout and a diagnostic log.

    The process-group default timeout (``config.nccl_timeout_sec``) is sized
    for training collectives (minutes of reduce on large tensors). Init-path
    barriers should fail fast so mesh or env misconfiguration does not block
    a job for 30 minutes before surfacing a useful error.
    """
    work = dist.barrier(async_op=True)
    try:
        done = work.wait(timeout=timedelta(seconds=seconds))  # type: ignore[reportOptionalMemberAccess]
    except RuntimeError as e:
        raise RuntimeError(
            f"Barrier timed out after {seconds}s during {reason}. "
            f"Common causes: MASTER_ADDR/MASTER_PORT disagreement across ranks, "
            f"a rank missing from the job, or the IB interface unreachable. "
            f"Underlying: {e}"
        ) from e
    if done is False:
        raise RuntimeError(f"Barrier timed out after {seconds}s during {reason}.")


def _set_seed(seed: int, rank: int, pp_rank: int = 0) -> None:
    """Set deterministic seeds for reproducibility.

    - Same seed across data-parallel replicas (for consistent dropout)
    - Different seed across pipeline stages (for stochastic depth variation)
    - Covers torch (CPU + all visible CUDA devices), Python's random, and
      NumPy's legacy global RNG — matches the four generators captured by
      ``checkpoint.state.get_rng_state`` so cold start and warm resume
      seed the same set of generators.
    """
    effective_seed = seed + pp_rank
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    _random.seed(effective_seed)
    _np.random.seed(effective_seed)


def init_distributed(config: DistributedConfig, seed: int = 42) -> DeviceMesh | None:
    """Initialize distributed training and build the DeviceMesh.

    Args:
        config: Distributed configuration with parallelism dimensions.
        seed: Random seed for reproducibility.

    Returns:
        DeviceMesh if world_size > 1, None for single-GPU.
    """
    rank, local_rank, world_size = get_world_info()

    if world_size == 1:
        torch.cuda.set_device(0)
        _set_seed(seed, rank=0)
        return None

    _set_nccl_env()

    if "MASTER_ADDR" not in os.environ and "SLURM_JOB_NODELIST" in os.environ:
        import subprocess

        result = subprocess.run(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            capture_output=True,
            text=True,
        )
        os.environ["MASTER_ADDR"] = result.stdout.strip().split("\n")[0]
    os.environ.setdefault("MASTER_ADDR", "localhost")
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

    if not dist.is_initialized():
        dist.init_process_group(
            backend=config.backend,
            timeout=timedelta(seconds=config.nccl_timeout_sec),
        )

    torch.cuda.set_device(local_rank)

    resolved = config.resolve(world_size)

    mesh_dims: list[str] = []
    mesh_sizes: list[int] = []

    dim_map = [
        ("pp", resolved.pp),
        ("dp_replicate", resolved.dp_replicate),
        ("dp_shard", resolved.dp_shard),
        ("ep", resolved.ep),
        ("tp", resolved.tp),
    ]

    for name, size in dim_map:
        if size > 1:
            mesh_dims.append(name)
            mesh_sizes.append(size)

    if (resolved.tp > 1 or resolved.ep > 1) and "dp_shard" not in mesh_dims:
        for insert_before in ("ep", "tp"):
            if insert_before in mesh_dims:
                idx = mesh_dims.index(insert_before)
                mesh_dims.insert(idx, "dp_shard")
                mesh_sizes.insert(idx, 1)
                break
        else:
            mesh_dims.append("dp_shard")
            mesh_sizes.append(1)

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

    _barrier_with_timeout(60.0, reason="DeviceMesh construction")

    pp_rank = 0
    if "pp" in device_mesh.mesh_dim_names:  # type: ignore[reportOperatorIssue]
        pp_rank = device_mesh["pp"].get_local_rank()
    _set_seed(seed, rank=rank, pp_rank=pp_rank)

    return device_mesh


def destroy_distributed() -> None:
    """Clean up distributed process groups."""
    if dist.is_initialized():
        dist.destroy_process_group()
