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

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.attention import Attention
from kempnerforge.model.embedding import OutputHead, TokenEmbedding
from kempnerforge.model.init import init_weights
from kempnerforge.model.mlp import build_mlp
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
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
        )

        self.mlp_norm = build_norm(config.norm_type, config.dim, eps=config.norm_eps)
        self.mlp = build_mlp(
            dim=config.dim,
            hidden_dim=config.computed_ffn_hidden_dim,
            activation=config.activation,
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attention(self.attention_norm(x), freqs_cis)
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

        # Precompute RoPE frequencies — stored as a plain attribute (not a buffer)
        # to avoid dtype casts when calling model.to(bf16). Complex tensors cannot
        # be cast to real dtypes. Device transfers are handled in forward().
        self._freqs_cis = precompute_rope_frequencies(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        # Initialize weights
        init_weights(self, config)

    @property
    def freqs_cis(self) -> torch.Tensor:
        return self._freqs_cis

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: Integer tensor of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        seq_len = tokens.shape[1]

        # Embed tokens (PP middle stages receive hidden states directly)
        h = self.token_embedding(tokens) if self.token_embedding is not None else tokens

        # Slice RoPE frequencies for current sequence length, move to device
        freqs_cis = self._freqs_cis[:seq_len].to(h.device)

        # Transformer blocks
        for layer in self.layers.values():
            h = layer(h, freqs_cis)

        # Final norm
        h = self.norm(h)

        # Output projection
        if self.output_head is not None:
            h = self.output_head(h)

        return h
