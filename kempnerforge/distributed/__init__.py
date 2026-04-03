"""Distributed training infrastructure.

Public API:
  - init_distributed / destroy_distributed: Process group lifecycle
  - apply_fsdp2 / apply_ac: FSDP2 sharding and activation checkpointing
  - apply_tensor_parallel: DTensor-based tensor parallelism
  - clip_grad_norm_ / dist_reduce / barrier_with_timeout: Utilities
"""

from kempnerforge.distributed.parallel import apply_ac, apply_fsdp2, default_mp_policy, get_dp_mesh
from kempnerforge.distributed.setup import (
    destroy_distributed,
    get_world_info,
    init_distributed,
    is_rank_zero,
)
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel, get_tp_mesh
from kempnerforge.distributed.utils import barrier_with_timeout, clip_grad_norm_, dist_reduce

__all__ = [
    "apply_ac",
    "apply_fsdp2",
    "apply_tensor_parallel",
    "barrier_with_timeout",
    "clip_grad_norm_",
    "default_mp_policy",
    "destroy_distributed",
    "dist_reduce",
    "get_dp_mesh",
    "get_tp_mesh",
    "get_world_info",
    "init_distributed",
    "is_rank_zero",
]
