"""Typed configuration schemas for KempnerForge.

All configuration is expressed as frozen dataclasses. Each dataclass validates its
own invariants at __post_init__ time so misconfigurations fail at startup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

# ---------------------------------------------------------------------------
# Enums for constrained choices
# ---------------------------------------------------------------------------


class NormType(StrEnum):
    rmsnorm = "rmsnorm"
    layernorm = "layernorm"


class Activation(StrEnum):
    silu = "silu"
    gelu = "gelu"
    relu = "relu"


class ActivationCheckpointing(StrEnum):
    none = "none"
    full = "full"
    selective = "selective"


class AsyncCheckpointMode(StrEnum):
    disabled = "disabled"
    async_ = "async"
    async_pinned = "async_with_pinned_mem"


class PipelineSchedule(StrEnum):
    schedule_1f1b = "1f1b"
    gpipe = "gpipe"
    interleaved_1f1b = "interleaved_1f1b"


class SchedulerType(StrEnum):
    cosine = "cosine"
    linear = "linear"
    wsd = "wsd"  # warmup-stable-decay


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Architecture hyperparameters for a transformer model."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None  # None → same as n_heads (MHA)
    vocab_size: int = 32000
    ffn_dim_multiplier: float = 1.0
    ffn_hidden_dim: int | None = None  # Override computed hidden dim
    norm_type: NormType = NormType.rmsnorm
    norm_eps: float = 1e-5
    activation: Activation = Activation.silu
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # Positivity checks first (before any division)
        if self.dim <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("dim, n_layers, and n_heads must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")

        # Divisibility checks
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def computed_ffn_hidden_dim(self) -> int:
        """FFN hidden dimension, rounded to nearest multiple of 256 for hardware efficiency."""
        if self.ffn_hidden_dim is not None:
            return self.ffn_hidden_dim
        # Llama-style: 4 * dim * (2/3) * ffn_dim_multiplier, rounded up to multiple of 256
        raw = int(4 * self.dim * (2 / 3) * self.ffn_dim_multiplier)
        return 256 * math.ceil(raw / 256)

    @property
    def num_params_estimate(self) -> int:
        """Rough parameter count estimate (excluding embedding if tied)."""
        d = self.dim
        h = self.computed_ffn_hidden_dim
        n_kv = self.n_kv_heads
        head_d = self.head_dim
        # Per layer: attention (Q + K + V + O) + MLP (gate + up + down) + 2 norms
        attn = d * (self.n_heads * head_d) + 2 * d * (n_kv * head_d) + (self.n_heads * head_d) * d
        mlp = d * h + d * h + h * d  # gate + up + down (SwiGLU has 3 matrices)
        norm = 2 * d  # 2 norms per layer
        per_layer = attn + mlp + norm
        embedding = self.vocab_size * d
        output = 0 if self.tie_embeddings else self.vocab_size * d
        final_norm = d
        return self.n_layers * per_layer + embedding + output + final_norm


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 8  # Per-device micro-batch size
    seq_len: int = 2048
    max_steps: int = 100000
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    seed: int = 42
    compile_model: bool = True
    activation_checkpointing: ActivationCheckpointing = ActivationCheckpointing.none

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")


# ---------------------------------------------------------------------------
# Optimizer configuration
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """Optimizer settings."""

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    fused: bool = True  # Use fused AdamW when available

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError("betas must be in [0, 1)")


# ---------------------------------------------------------------------------
# LR scheduler configuration
# ---------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    """Learning rate schedule settings."""

    name: SchedulerType = SchedulerType.cosine
    warmup_steps: int = 2000
    decay_steps: int | None = None  # None → decay over remaining steps
    min_lr_ratio: float = 0.1  # min_lr = lr * min_lr_ratio
    # WSD-specific
    stable_steps: int | None = None  # For WSD: steps at constant LR

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not (0 <= self.min_lr_ratio <= 1):
            raise ValueError("min_lr_ratio must be in [0, 1]")


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Data pipeline settings."""

    dataset_path: str = ""
    file_pattern: str = "*.npy"  # Glob pattern for data files (e.g., "*.npy", "tokenized_*.bin")
    tokenizer_path: str = ""
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    # For HuggingFace datasets
    hf_dataset_name: str | None = None
    hf_dataset_split: str = "train"
    hf_dataset_text_field: str = "text"

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be >= 1")


# ---------------------------------------------------------------------------
# Distributed configuration
# ---------------------------------------------------------------------------


@dataclass
class DistributedConfig:
    """Parallelism dimensions and distributed settings."""

    dp_shard: int = -1  # -1 → auto (use all remaining GPUs)
    dp_replicate: int = 1
    tp: int = 1
    pp: int = 1
    pp_schedule: PipelineSchedule = PipelineSchedule.schedule_1f1b
    cp: int = 1
    nccl_timeout_sec: int = 600
    backend: str = "nccl"

    def validate_world_size(self, world_size: int) -> None:
        """Validate that parallelism dimensions match world size."""
        dp_shard = self._resolve_dp_shard(world_size)
        expected = self.dp_replicate * dp_shard * self.tp * self.pp * self.cp
        if expected != world_size:
            raise ValueError(
                f"Parallelism dimensions ({self.dp_replicate} × {dp_shard} × "
                f"{self.tp} × {self.pp} × {self.cp} = {expected}) "
                f"do not match world_size ({world_size})"
            )

    def _resolve_dp_shard(self, world_size: int) -> int:
        """Resolve dp_shard=-1 to actual value."""
        if self.dp_shard > 0:
            return self.dp_shard
        other = self.dp_replicate * self.tp * self.pp * self.cp
        if world_size % other != 0:
            raise ValueError(
                f"world_size ({world_size}) not divisible by "
                f"dp_replicate*tp*pp*cp ({other})"
            )
        return world_size // other

    def resolve(self, world_size: int) -> DistributedConfig:
        """Return a copy with dp_shard resolved to a concrete value."""
        resolved = DistributedConfig(
            dp_shard=self._resolve_dp_shard(world_size),
            dp_replicate=self.dp_replicate,
            tp=self.tp,
            pp=self.pp,
            pp_schedule=self.pp_schedule,
            cp=self.cp,
            nccl_timeout_sec=self.nccl_timeout_sec,
            backend=self.backend,
        )
        resolved.validate_world_size(world_size)
        return resolved

    def __post_init__(self) -> None:
        if self.dp_shard == 0 or self.dp_shard < -1:
            raise ValueError("dp_shard must be -1 (auto) or positive")
        for name, val in [
            ("dp_replicate", self.dp_replicate),
            ("tp", self.tp),
            ("pp", self.pp),
            ("cp", self.cp),
        ]:
            if val < 1:
                raise ValueError(f"{name} must be >= 1")


# ---------------------------------------------------------------------------
# Checkpoint configuration
# ---------------------------------------------------------------------------


@dataclass
class CheckpointConfig:
    """Checkpointing settings."""

    dir: str = "checkpoints"
    interval: int = 1000  # Save every N steps
    async_mode: AsyncCheckpointMode = AsyncCheckpointMode.disabled
    keep_last_n: int = 3  # Number of checkpoints to retain
    load_path: str | None = None  # Path to load from (for resumption)
    export_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    exclude_from_loading: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("interval must be positive")
        if self.keep_last_n < 1:
            raise ValueError("keep_last_n must be >= 1")


# ---------------------------------------------------------------------------
# Metrics configuration
# ---------------------------------------------------------------------------


@dataclass
class MetricsConfig:
    """Logging and metrics settings."""

    log_interval: int = 10  # Log every N steps
    enable_wandb: bool = False
    enable_tensorboard: bool = False
    wandb_project: str = "kempnerforge"
    wandb_run_name: str | None = None  # None → auto-generated
    tensorboard_dir: str = "tb_logs"

    def __post_init__(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


# ---------------------------------------------------------------------------
# Profiling configuration
# ---------------------------------------------------------------------------


@dataclass
class ProfilingConfig:
    """Performance profiling settings."""

    enable: bool = False
    start_step: int = 5
    end_step: int = 8
    trace_dir: str = "profiler_traces"

    def __post_init__(self) -> None:
        if self.end_step <= self.start_step:
            raise ValueError("end_step must be greater than start_step")


# ---------------------------------------------------------------------------
# Top-level job configuration
# ---------------------------------------------------------------------------


@dataclass
class JobConfig:
    """Top-level configuration aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
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
