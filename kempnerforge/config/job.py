"""Top-level job configuration aggregating all sub-configs."""

from __future__ import annotations

from dataclasses import dataclass, field

from kempnerforge.config.checkpoint import CheckpointConfig
from kempnerforge.config.data import DataConfig
from kempnerforge.config.distributed import DistributedConfig
from kempnerforge.config.eval import EvalConfig
from kempnerforge.config.metrics import MetricsConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.optimizer import OptimizerConfig
from kempnerforge.config.profiling import ProfilingConfig
from kempnerforge.config.scheduler import SchedulerConfig
from kempnerforge.config.training import TrainConfig


@dataclass
class JobConfig:
    """Top-level configuration aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    def validate(self, world_size: int = 1) -> None:
        """Run cross-config validations."""
        self.distributed.validate_world_size(world_size)

        if self.train.seq_len > self.model.max_seq_len:
            raise ValueError(
                f"train.seq_len ({self.train.seq_len}) exceeds "
                f"model.max_seq_len ({self.model.max_seq_len})"
            )

        if self.model.tie_embeddings and self.distributed.pp > 1:
            raise ValueError(
                "Tied embeddings are not supported with pipeline parallelism "
                "(embedding and output head must be on different stages)"
            )

        if self.eval.enabled and not self.eval.dataset_path and not self.eval.hf_dataset_name:
            raise ValueError(
                "eval.enabled is True but no eval data source is configured. "
                "Set eval.dataset_path or eval.hf_dataset_name."
            )

        if self.model.is_moe and self.train.compile_model:
            import logging

            logging.getLogger(__name__).warning(
                "torch.compile is not yet optimized for MoE dispatch (data-dependent shapes "
                "cause graph breaks). Set compile_model=false for MoE models."
            )

        if self.distributed.tp > 1:
            if self.model.n_heads % self.distributed.tp != 0:
                raise ValueError(
                    f"n_heads ({self.model.n_heads}) must be divisible by "
                    f"tp ({self.distributed.tp})"
                )
            if self.model.n_kv_heads and self.model.n_kv_heads % self.distributed.tp != 0:
                raise ValueError(
                    f"n_kv_heads ({self.model.n_kv_heads}) must be divisible by "
                    f"tp ({self.distributed.tp})"
                )

        if self.train.is_fp8 and self.distributed.tp > 1:
            raise ValueError(
                "FP8 + Tensor Parallelism is not yet supported (torchao Float8Linear "
                "does not compose with DTensor sharding). Use FP8 with FSDP only, "
                "or TP without FP8."
            )

        if self.model.is_moe and self.distributed.pp > 1:
            raise ValueError(
                "MoE + Pipeline Parallelism is not supported. MoE layers use "
                "data-dependent routing that is incompatible with pipeline stage "
                "splitting. Use FSDP, TP, or EP instead."
            )

        if self.distributed.ep > 1:
            if not self.model.is_moe:
                raise ValueError("ep > 1 requires an MoE model (num_experts > 0)")
            if self.model.num_experts % self.distributed.ep != 0:
                raise ValueError(
                    f"num_experts ({self.model.num_experts}) must be divisible by "
                    f"ep ({self.distributed.ep})"
                )
