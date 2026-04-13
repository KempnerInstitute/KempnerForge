"""Model architecture configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class NormType(StrEnum):
    rmsnorm = "rmsnorm"
    layernorm = "layernorm"


class Activation(StrEnum):
    silu = "silu"
    gelu = "gelu"
    relu = "relu"


@dataclass
class ModelConfig:
    """Architecture hyperparameters for a transformer model."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None  # None -> same as n_heads (MHA)
    vocab_size: int = 32000
    ffn_dim_multiplier: float = 1.0
    ffn_hidden_dim: int | None = None  # Override computed hidden dim
    norm_type: NormType = NormType.rmsnorm
    norm_eps: float = 1e-5
    activation: Activation = Activation.silu
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = False
    qk_norm: bool = False  # Apply RMSNorm to Q/K per-head before RoPE (Gemma, DeepSeek-V3)
    init_std: float = 0.02  # Std for weight initialization (GPT-2/Llama default)
    model_type: str = "transformer"  # Registry key for model builder

    # MoE (all defaults produce a dense model -- zero behavior change)
    num_experts: int = 0  # 0 = dense, >0 = MoE
    moe_top_k: int = 2  # experts selected per token
    moe_frequency: int = 1  # MoE every N layers (1=all, 2=alternating)
    moe_router: str = "softmax_topk"  # registry key for router type
    moe_shared_experts: int = 0  # shared experts that process all tokens
    moe_aux_loss_weight: float = 0.01  # aux loss coefficient in training loss
    moe_capacity_factor: float = 0.0  # 0=no drop, >0=cap tokens/expert (e.g. 1.25)
    moe_sequence_aux_loss_weight: float = 0.0  # Sequence-level balance loss (0=off)
    moe_gradient_scale: bool = False  # Per-expert gradient normalization
    moe_bias_schedule: str = "constant"  # "constant", "cosine_decay", "linear_warmup"

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
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

        # MoE validation
        if self.num_experts > 0:
            if self.moe_top_k <= 0:
                raise ValueError("moe_top_k must be positive when num_experts > 0")
            if self.moe_top_k > self.num_experts:
                raise ValueError(
                    f"moe_top_k ({self.moe_top_k}) must be <= num_experts ({self.num_experts})"
                )
            if self.moe_frequency <= 0:
                raise ValueError("moe_frequency must be positive")
            if self.moe_sequence_aux_loss_weight < 0:
                raise ValueError("moe_sequence_aux_loss_weight must be non-negative")
            if self.moe_bias_schedule not in ("constant", "cosine_decay", "linear_warmup"):
                raise ValueError(
                    f"Unknown moe_bias_schedule: '{self.moe_bias_schedule}'. "
                    "Options: 'constant', 'cosine_decay', 'linear_warmup'"
                )

    @property
    def is_moe(self) -> bool:
        """Whether this config uses Mixture-of-Experts."""
        return self.num_experts > 0

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
        """Rough total parameter count estimate (excluding embedding if tied).

        For MoE models, counts all expert parameters (total, not active).
        """
        d = self.dim
        h = self.computed_ffn_hidden_dim
        n_kv = self.n_kv_heads
        head_d = self.head_dim
        # Per layer: attention (Q + K + V + O) + 2 norms
        attn = d * (self.n_heads * head_d) + 2 * d * (n_kv * head_d) + (self.n_heads * head_d) * d  # type: ignore[reportOptionalOperand]
        mlp = d * h + d * h + h * d  # gate + up + down (SwiGLU has 3 matrices)
        norm = 2 * d  # 2 norms per layer

        if self.is_moe:
            n_moe = sum(1 for i in range(self.n_layers) if (i + 1) % self.moe_frequency == 0)
            n_dense = self.n_layers - n_moe
            router = d * self.num_experts  # gate linear per MoE layer
            shared_mlp = self.moe_shared_experts * mlp
            moe_per_layer = attn + self.num_experts * mlp + router + shared_mlp + norm
            dense_per_layer = attn + mlp + norm
            layer_params = n_moe * moe_per_layer + n_dense * dense_per_layer
        else:
            layer_params = self.n_layers * (attn + mlp + norm)

        embedding = self.vocab_size * d
        output = 0 if self.tie_embeddings else self.vocab_size * d
        final_norm = d
        return layer_params + embedding + output + final_norm
