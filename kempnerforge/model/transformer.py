"""Transformer model for KempnerForge.

Architecture: Llama-style pre-norm transformer.
  Token Embedding → [TransformerBlock × N] → Final Norm → Output Head

Design choices:
  - ModuleDict (not ModuleList) for layers — preserves FQNs for DCP checkpointing.
  - Embedding and output head are optional (can be None for PP middle stages).
  - Forward is a simple loop over blocks — pipeline-parallelism friendly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kempnerforge.config.registry import registry
from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.attention import Attention, KVCache
from kempnerforge.model.embedding import OutputHead, TokenEmbedding
from kempnerforge.model.init import init_weights
from kempnerforge.model.mlp import build_mlp
from kempnerforge.model.moe import MoEMLP, build_moe
from kempnerforge.model.norm import build_norm
from kempnerforge.model.position import precompute_rope_frequencies


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.

    Structure: norm → attention → residual, norm → mlp → residual
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.attention_norm = build_norm(config.norm_type, config.dim, eps=config.norm_eps)
        self.attention = Attention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,  # type: ignore[reportArgumentType]
            head_dim=config.head_dim,
            qk_norm=config.qk_norm,
        )

        self.mlp_norm = build_norm(config.norm_type, config.dim, eps=config.norm_eps)

        # MoE placement: with moe_frequency=1, all layers are MoE.
        # With moe_frequency=2, layers 1,3,5... are MoE (layer 0 stays dense).
        use_moe = config.is_moe and ((layer_idx + 1) % config.moe_frequency == 0)
        if use_moe:
            self.mlp = build_moe(
                dim=config.dim,
                hidden_dim=config.computed_ffn_hidden_dim,
                num_experts=config.num_experts,
                top_k=config.moe_top_k,
                activation=config.activation,
                router_type=config.moe_router,
                shared_experts=config.moe_shared_experts,
                capacity_factor=config.moe_capacity_factor,
                gradient_scale=config.moe_gradient_scale,
                sequence_aux_loss_weight=config.moe_sequence_aux_loss_weight,
                bias_schedule=config.moe_bias_schedule,
                packed_experts=config.moe_packed_experts,
            )
        else:
            self.mlp = build_mlp(
                dim=config.dim,
                hidden_dim=config.computed_ffn_hidden_dim,
                activation=config.activation,
            )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attention(
            self.attention_norm(x), rope_cos, rope_sin, kv_cache=kv_cache, doc_ids=doc_ids
        )
        # Pre-norm MLP with residual
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    """Full transformer model built from ModelConfig.

    Embedding → TransformerBlocks → Norm → Output Head
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding (can be None for PP middle stages)
        self.token_embedding: TokenEmbedding | None = TokenEmbedding(config.vocab_size, config.dim)

        # Transformer blocks — use ModuleDict to preserve FQNs for DCP
        self.layers = nn.ModuleDict(
            {str(i): TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)}
        )

        # Final normalization
        self.norm = build_norm(config.norm_type, config.dim, eps=config.norm_eps)

        # Output head (can be None for PP non-final stages)
        self.output_head: OutputHead | None = OutputHead(config.dim, config.vocab_size)

        # Weight tying
        can_tie = self.token_embedding is not None and self.output_head is not None
        if config.tie_embeddings and can_tie:
            self.output_head.tie_weights(self.token_embedding)

        # Precompute RoPE cos/sin tables — stored as plain attributes (not buffers)
        # so model.to(bf16) doesn't cast them from float32.
        # Skip when on meta device (no data); call init_weights_and_freqs() later.
        self._rope_cos = None
        self._rope_sin = None
        if not any(p.is_meta for p in self.parameters()):
            self._rope_cos, self._rope_sin = precompute_rope_frequencies(
                head_dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
            init_weights(self, config)

    def init_weights_and_freqs(self) -> None:
        """Initialize weights and RoPE frequencies after meta-device materialization.

        Called after ``model.to_empty(device=...)`` to fill in parameter values
        and compute RoPE frequency table. Safe to call on already-initialized models
        (skips if freqs are already computed).
        """
        if self._rope_cos is None:
            self._rope_cos, self._rope_sin = precompute_rope_frequencies(
                head_dim=self.config.head_dim,
                max_seq_len=self.config.max_seq_len,
                theta=self.config.rope_theta,
            )
        init_weights(self, self.config)

    def set_moe_step(self, step: int, max_steps: int) -> None:
        """Set training step on all MoE routers for adaptive bias scheduling."""
        from kempnerforge.model.router import SigmoidTopKRouter

        for layer in self.layers.values():
            if isinstance(layer.mlp, MoEMLP) and isinstance(layer.mlp.router, SigmoidTopKRouter):
                layer.mlp.router.set_step(step, max_steps)

    def get_moe_aux_loss(self) -> torch.Tensor:
        """Collect auxiliary losses from all MoE layers. Returns 0 if dense."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers.values():
            if isinstance(layer.mlp, MoEMLP):
                total = total + layer.mlp.aux_loss
        return total

    def get_expert_counts(self) -> dict[int, torch.Tensor]:
        """Collect per-layer expert utilization. Returns {} if dense."""
        counts = {}
        for name, layer in self.layers.items():
            if isinstance(layer.mlp, MoEMLP):
                counts[int(name)] = layer.mlp.expert_counts
        return counts

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        kv_caches: list[KVCache] | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: Integer tensor of shape (batch, seq_len).
            kv_caches: Optional list of KVCache (one per layer) for generation.
                When provided, RoPE positions are offset by the current cache
                fill level so incremental decode tokens get correct positions.
            doc_ids: Optional per-token document IDs for packed sequences,
                shape (batch, seq_len). Enables block-diagonal causal attention
                that isolates documents within packed sequences.

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        seq_len = tokens.shape[1]

        # Embed tokens (PP middle stages receive hidden states directly)
        h = self.token_embedding(tokens) if self.token_embedding is not None else tokens

        # Determine position offset from KV cache fill level
        start_pos = kv_caches[0].seq_len if kv_caches is not None else 0

        # Slice RoPE frequencies for current positions (device transfer cached)
        if self._rope_cos.device != h.device:  # type: ignore[reportOptionalMemberAccess]
            self._rope_cos = self._rope_cos.to(h.device)  # type: ignore[reportOptionalMemberAccess]
            self._rope_sin = self._rope_sin.to(h.device)  # type: ignore[reportOptionalMemberAccess]
        cos = self._rope_cos[start_pos : start_pos + seq_len]  # type: ignore[reportOptionalSubscript]
        sin = self._rope_sin[start_pos : start_pos + seq_len]  # type: ignore[reportOptionalSubscript]

        # Transformer blocks
        for i, layer in enumerate(self.layers.values()):
            cache = kv_caches[i] if kv_caches is not None else None
            h = layer(h, cos, sin, kv_cache=cache, doc_ids=doc_ids)

        # Final norm
        h = self.norm(h)

        # Output projection
        if self.output_head is not None:
            h = self.output_head(h)

        return h


@registry.register_model("transformer")
def _build_transformer(model_config: ModelConfig) -> Transformer:
    return Transformer(model_config)
