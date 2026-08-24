"""Process-level context for a training job: ranks, device, mesh, pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from kempnerforge.config.job import JobConfig
from kempnerforge.distributed.setup import get_world_info, init_distributed
from kempnerforge.distributed.utils import get_dp_info
from kempnerforge.metrics.logger import get_logger
from kempnerforge.resilience.elastic import log_job_info

logger = get_logger(__name__)


@dataclass(frozen=True)
class RuntimeContext:
    """Where this process sits: ranks, device, mesh, and the DP view of it."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    device_mesh: DeviceMesh | None
    dp_rank: int
    dp_size: int


@dataclass(frozen=True)
class PipelineBundle:
    """Pipeline-parallel state. Built only when ``distributed.pp > 1``."""

    rank: int
    size: int
    schedule: Any
    group: Any


def setup_distributed(config: JobConfig) -> RuntimeContext:
    """Initialize the process group + device mesh, then validate the config against it."""
    rank, local_rank, world_size = get_world_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    device_mesh = init_distributed(config.distributed, seed=config.train.seed)
    config.validate(world_size)

    log_job_info()
    logger.info(f"Training config: {config}")

    # With PP, samplers use DP rank/size (not world size) since all PP
    # stages in the same DP group process the same batch.
    dp_rank, dp_size = get_dp_info(device_mesh)

    return RuntimeContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        device_mesh=device_mesh,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )
