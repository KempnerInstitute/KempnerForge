"""Loss function registry for KempnerForge.

Registers loss functions and provides build_loss_fn() to compose them
with config-driven options (chunk size, z-loss). Follows the same
builder pattern as build_optimizer.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
import torch.nn.functional as F

from kempnerforge.config.registry import registry


@registry.register_loss("cross_entropy")
def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss for language modeling."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


@registry.register_loss("chunked_cross_entropy")
def chunked_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Cross-entropy computed in token-dimension chunks.

    Chunks along the token dimension and uses PyTorch's fused CE kernel per
    chunk, avoiding an explicit float32 materialization of the full logit
    tensor. For Llama-3 7B (vocab=128K, batch=4, seq=4096), the manual
    logsumexp path would create a ~8 GB float32 copy; this implementation
    avoids that entirely.

    Note: the input logit tensor (B*S, V) is still fully materialized by the
    model's output head before reaching this function. For deeper savings
    (never materializing the full logit tensor), the output projection itself
    must be chunked in the model forward pass — a future enhancement.

    Args:
        logits: (batch, seq, vocab) or (tokens, vocab).
        labels: (batch, seq) or (tokens,).
        chunk_size: Number of tokens per chunk.
    """
    num_tokens = logits.shape[0] * logits.shape[1] if logits.ndim == 3 else logits.shape[0]
    flat_logits = logits.view(num_tokens, -1)
    flat_labels = labels.view(-1)

    if num_tokens <= chunk_size:
        return F.cross_entropy(flat_logits, flat_labels)

    total_loss = torch.tensor(0.0, device=flat_logits.device, dtype=torch.float32)
    for i in range(0, num_tokens, chunk_size):
        total_loss = total_loss + F.cross_entropy(
            flat_logits[i : i + chunk_size],
            flat_labels[i : i + chunk_size],
            reduction="sum",
        )
    return total_loss / num_tokens


def z_loss(logits: torch.Tensor, weight: float) -> torch.Tensor:
    """Logit magnitude regularizer (PaLM / Gemini).

    Penalizes large logit magnitudes to prevent logit drift that causes
    NaN/divergence in long training runs. Negligible compute cost.

    Formula: weight * mean(logsumexp(logits, dim=-1) ** 2)

    Args:
        logits: Model output logits, shape (batch, seq, vocab) or (tokens, vocab).
        weight: Regularization weight (PaLM uses 1e-4).

    Returns:
        Scalar z-loss term to add to the main loss.
    """
    if weight == 0.0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    flat = logits.view(-1, logits.shape[-1]).float()
    lse = torch.logsumexp(flat, dim=-1)
    return weight * (lse**2).mean()


def build_loss_fn(config) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a composed loss function from training config.

    Follows the build_optimizer pattern: config in, callable out.
    Binds chunk_size for chunked CE and composes z-loss, so the caller
    gets a clean ``(logits, labels) -> Tensor`` interface.
    """
    base_fn = registry.get_loss(config.loss_fn)

    if config.loss_fn == "chunked_cross_entropy":
        chunk_size = config.ce_chunk_size if config.ce_chunk_size > 0 else 4096
        base_fn = functools.partial(base_fn, chunk_size=chunk_size)

    if config.z_loss_weight > 0:
        z_weight = config.z_loss_weight
        _inner = base_fn

        def composed(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return _inner(logits, labels) + z_loss(logits, z_weight)

        return composed

    return base_fn
