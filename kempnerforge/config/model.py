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


FLEX_BLOCK_SIZE = 128

FLEX_MIN_HEAD_DIM = 16


@dataclass
class ModelConfig:
    """Architecture hyperparameters for a transformer model."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    head_dim_override: int = 0
    vocab_size: int = 32000
    ffn_dim_multiplier: float = 1.0
    ffn_hidden_dim: int | None = None
    norm_type: NormType = NormType.rmsnorm
    norm_eps: float = 1e-5
    activation: Activation = Activation.silu
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = False
    qk_norm: bool = False
    init_std: float = 0.02
    model_type: str = "transformer"
    sdpa_backend: str = "auto"
    attention_backend: str = "sdpa"

    num_experts: int = 0
    moe_top_k: int = 2
    moe_frequency: int = 1
    moe_router: str = "softmax_topk"
    moe_shared_experts: int = 0
    moe_aux_loss_weight: float = 0.01
    moe_capacity_factor: float = 0.0
    moe_sequence_aux_loss_weight: float = 0.0
    moe_gradient_scale: bool = False
    moe_bias_schedule: str = "constant"
    moe_packed_experts: bool = False
    moe_expert_ffn_multiplier: float = 1.0
    moe_router_z_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.dim <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("dim, n_layers, and n_heads must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if self.head_dim_override < 0:
            raise ValueError(
                f"head_dim_override must be non-negative (got {self.head_dim_override}); "
                "0 infers dim // n_heads"
            )
        if self.head_dim_override % 2:
            raise ValueError(
                "head_dim_override must be even for rotary embeddings "
                f"(got {self.head_dim_override})"
            )

        if self.head_dim_override == 0 and self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads}) "
                "unless head_dim_override is set"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

        if self.sdpa_backend not in ("auto", "flash", "efficient", "cudnn", "math"):
            raise ValueError(
                f"Unknown sdpa_backend: '{self.sdpa_backend}'. "
                "Options: 'auto', 'flash', 'efficient', 'cudnn', 'math'"
            )

        if self.attention_backend not in ("sdpa", "flex"):
            raise ValueError(
                f"Unknown attention_backend: '{self.attention_backend}'. Options: 'sdpa', 'flex'"
            )
        if self.attention_backend == "flex":
            if self.head_dim < FLEX_MIN_HEAD_DIM:
                raise ValueError(
                    f"attention_backend='flex' requires head_dim >= {FLEX_MIN_HEAD_DIM}, got "
                    f"{self.head_dim} (dim={self.dim} // n_heads={self.n_heads}). "
                    "FlexAttention's Triton template does not lower below that "
                    "(head_dim 8 fails with 'NYI: embedding dimension')."
                )
            if self.sdpa_backend != "auto":
                import logging

                logging.getLogger(__name__).warning(
                    "sdpa_backend=%r is ignored when attention_backend='flex' -- the flex "
                    "path does not route through the SDPA kernel selector.",
                    self.sdpa_backend,
                )

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
            if self.moe_expert_ffn_multiplier <= 0:
                raise ValueError("moe_expert_ffn_multiplier must be positive")
            if self.moe_shared_experts < 0:
                raise ValueError(
                    f"moe_shared_experts must be >= 0, got {self.moe_shared_experts}. Set it "
                    "to the number of shared experts every token goes through, or 0 for none."
                )
            if self.moe_router_z_loss_weight < 0:
                raise ValueError("moe_router_z_loss_weight must be non-negative")

    @property
    def is_moe(self) -> bool:
        """Whether this config uses Mixture-of-Experts."""
        return self.num_experts > 0

    @property
    def tp_sequence_parallel(self) -> bool:
        """Whether tensor parallelism runs this model with ``SequenceParallel``.

        ``apply_tensor_parallel`` shards activations along the sequence between blocks
        unless the embeddings are tied, since sharding the output head's columns would also
        shard the shared embedding weight, or the model is MoE, since boolean indexing in
        expert dispatch breaks sequence-sharded DTensors. Pipeline stages also run without
        it; that depends on the stage, not on this config.
        """
        return not self.tie_embeddings and not self.is_moe

    @property
    def head_dim(self) -> int:
        """Per-head attention width.

        Derived rather than stored so a later ``dim`` / ``n_heads`` change (a
        TOML or CLI overlay onto a default config) cannot leave a stale value
        behind. ``head_dim_override`` decouples it from ``dim``.
        """
        return self.head_dim_override or self.dim // self.n_heads

    @property
    def computed_ffn_hidden_dim(self) -> int:
        """FFN hidden dimension, rounded to nearest multiple of 256 for hardware efficiency."""
        if self.ffn_hidden_dim is not None:
            return self.ffn_hidden_dim
        raw = int(4 * self.dim * (2 / 3) * self.ffn_dim_multiplier)
        return 256 * math.ceil(raw / 256)

    @property
    def computed_expert_ffn_hidden_dim(self) -> int:
        """Per-expert FFN hidden dim = ``computed_ffn_hidden_dim`` * ``moe_expert_ffn_multiplier``.

        Rounded to a multiple of 16 for tensor-core alignment. With the default
        multiplier 1.0 this equals ``computed_ffn_hidden_dim`` (zero behavior
        change); set 0.5 for fine-grained experts so top-2 routing matches the
        dense FFN's activated FLOPs (2 * F/2 = F). Routed experts are this wide
        (build_moe and MoMa's ExpertChoiceMoE); build_moe's shared experts are one
        MLP ``moe_shared_experts`` times this wide.
        """
        raw = int(self.computed_ffn_hidden_dim * self.moe_expert_ffn_multiplier)
        return max(16, 16 * round(raw / 16))

    @property
    def mlp_matrices(self) -> int:
        """Weight matrices in each MLP the model builds, dense layers and experts alike.

        Three for ``silu``, which builds SwiGLU (gate, up and down projections), and two
        for ``gelu`` and ``relu``, which build the standard MLP (up and down). An MLP a
        plugin registered under another name is counted as three.
        """
        return 2 if self.activation in (Activation.gelu, Activation.relu) else 3

    @property
    def num_params_estimate(self) -> int:
        """Rough total parameter count estimate (excluding embedding if tied).

        For MoE models, counts all expert parameters (total, not active). Each MLP
        counts ``mlp_matrices`` weight matrices.
        """
        d = self.dim
        h = self.computed_ffn_hidden_dim
        n_kv = self.n_kv_heads
        head_d = self.head_dim
        attn = d * (self.n_heads * head_d) + 2 * d * (n_kv * head_d) + (self.n_heads * head_d) * d  # type: ignore[reportOptionalOperand]
        mlp = self.mlp_matrices * d * h
        norm = 2 * d

        if self.is_moe:
            he = self.computed_expert_ffn_hidden_dim
            expert_mlp = self.mlp_matrices * d * he
            n_moe = sum(1 for i in range(self.n_layers) if (i + 1) % self.moe_frequency == 0)
            n_dense = self.n_layers - n_moe
            router = d * self.num_experts
            shared_mlp = self.moe_shared_experts * expert_mlp
            moe_per_layer = attn + self.num_experts * expert_mlp + router + shared_mlp + norm
            dense_per_layer = attn + mlp + norm
            layer_params = n_moe * moe_per_layer + n_dense * dense_per_layer
        else:
            layer_params = self.n_layers * (attn + mlp + norm)

        embedding = self.vocab_size * d
        output = 0 if self.tie_embeddings else self.vocab_size * d
        final_norm = d
        return layer_params + embedding + output + final_norm
