"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch


class ActivationCheckpointing(StrEnum):
    none = "none"
    full = "full"
    selective = "selective"


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
    mixed_precision: Literal["bf16", "fp16", "fp32", "fp8"] = "bf16"
    activation_checkpointing: ActivationCheckpointing = ActivationCheckpointing.none
    loss_fn: str = "cross_entropy"  # Registry key for loss function
    z_loss_weight: float = 0.0  # Logit magnitude regularizer (PaLM uses 1e-4, 0=disabled)
    ce_chunk_size: int = 0  # Token chunk size for chunked_cross_entropy (0=auto 4096)
    shutdown_timeout_sec: float = 600.0  # Graceful shutdown timeout before forced exit
    nccl_health_check_interval: int = 0  # Check NCCL health every N steps (0=disabled)

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

    @property
    def param_dtype(self) -> torch.dtype:
        """Resolve mixed_precision to the master weight dtype.

        FP8 uses bf16 master weights -- FP8 is a compute mode, not a storage dtype.
        """
        import torch

        _DTYPE_MAP = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "fp8": torch.bfloat16,  # FP8 compute with bf16 master weights
        }
        return _DTYPE_MAP[self.mixed_precision]

    @property
    def is_fp8(self) -> bool:
        """Whether FP8 mixed precision is enabled."""
        return self.mixed_precision == "fp8"
