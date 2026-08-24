"""Process-level context for a training job: ranks, device, mesh, pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from kempnerforge.config.job import JobConfig
from kempnerforge.distributed.setup import get_world_info, init_distributed
from kempnerforge.metrics.logger import get_logger
from kempnerforge.resilience.elastic import log_job_info

logger = get_logger(__name__)


@dataclass(frozen=True)
class RuntimeContext:
    """Where this process sits: ranks, device, and the mesh built for them."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    device_mesh: DeviceMesh | None


@dataclass(frozen=True)
class PipelineBundle:
    """Pipeline-parallel state. Built only when ``distributed.pp > 1``.

    The PP process group is not held here: it is derived from the mesh at the
    call sites that need it, so a job that never runs a step never slices it.
    """

    rank: int
    size: int
    schedule: Any


def pp_group(runtime: RuntimeContext) -> Any:
    """The PP process group for this rank, derived on demand."""
    mesh = runtime.device_mesh
    if mesh is None:  # unreachable: validate() rejects pp > 1 without a mesh
        raise RuntimeError("pipeline parallelism requires a device mesh")
    return mesh["pp"].get_group()


def setup_distributed(config: JobConfig) -> RuntimeContext:
    """Initialize the process group + device mesh, then validate the config against it."""
    rank, local_rank, world_size = get_world_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    device_mesh = init_distributed(config.distributed, seed=config.train.seed)
    config.validate(world_size)

    log_job_info()
    logger.info(f"Training config: {config}")

    return RuntimeContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        device_mesh=device_mesh,
    )
