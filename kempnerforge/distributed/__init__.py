"""Distributed training infrastructure.

Public API:
  - init_distributed / destroy_distributed: Process group lifecycle
  - apply_fsdp2 / apply_ac: FSDP2 sharding and activation checkpointing
  - apply_tensor_parallel: DTensor-based tensor parallelism
  - Pipeline parallelism: build_stage_module, build_pipeline_stage, build_pipeline_schedule
  - clip_grad_norm_: DTensor-aware gradient clipping
"""

from kempnerforge.distributed.parallel import apply_ac, apply_fsdp2, default_mp_policy, get_dp_mesh
from kempnerforge.distributed.pipeline_parallel import (
    build_pipeline_schedule,
    build_pipeline_stage,
    build_stage_module,
    compute_layer_assignment,
    get_pp_mesh,
    get_pp_rank,
    get_pp_size,
)
from kempnerforge.distributed.setup import (
    destroy_distributed,
    get_world_info,
    init_distributed,
    is_rank_zero,
)
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel, get_tp_mesh
from kempnerforge.distributed.utils import clip_grad_norm_

__all__ = [
    "apply_ac",
    "apply_fsdp2",
    "apply_tensor_parallel",
    "build_pipeline_schedule",
    "build_pipeline_stage",
    "build_stage_module",
    "clip_grad_norm_",
    "compute_layer_assignment",
    "default_mp_policy",
    "destroy_distributed",
    "get_dp_mesh",
    "get_pp_mesh",
    "get_pp_rank",
    "get_pp_size",
    "get_tp_mesh",
    "get_world_info",
    "init_distributed",
    "is_rank_zero",
]
