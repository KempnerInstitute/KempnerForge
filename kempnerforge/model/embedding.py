"""Token embedding and output head for KempnerForge models."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding layer.

    Can be disabled (returns input unchanged) for pipeline parallelism
    middle stages where the embedding lives on a different stage.
    """

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed token ids to vectors.

        Args:
            tokens: Integer tensor of shape (batch, seq_len).

        Returns:
            Tensor of shape (batch, seq_len, dim).
        """
        return self.embedding(tokens)


class OutputHead(nn.Module):
    """Linear output projection from hidden dim to vocab size.

    Produces logits (no softmax). Can optionally share weights with an embedding layer.
    """

    def __init__(self, dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logits.

        Args:
            x: Tensor of shape (batch, seq_len, dim).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        return self.proj(x)

    def tie_weights(self, embedding: TokenEmbedding) -> None:
        """Share the output projection weight with the embedding layer."""
        self.proj.weight = embedding.embedding.weight
