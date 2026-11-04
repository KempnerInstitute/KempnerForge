"""Loss function registry for KempnerForge.

Registers loss functions and provides build_loss_fn() to compose them
with config-driven options (chunk size, z-loss). Follows the same
builder pattern as build_optimizer.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

from kempnerforge.config.registry import registry

_DEFAULT_CHUNK_SIZE = 4096
_CHUNKED_LOSSES = ("cross_entropy", "chunked_cross_entropy")


def _chunk_loss(chunk: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed cross-entropy of one chunk and each row's log-sum-exp, in at least fp32."""
    x = chunk.to(torch.promote_types(chunk.dtype, torch.float32))
    lse = torch.logsumexp(x, dim=-1)
    valid = target != -100
    picked = x.gather(-1, torch.where(valid, target, 0).unsqueeze(-1)).squeeze(-1)
    return torch.where(valid, lse - picked, 0.0).sum(), lse


def _chunk_grad(
    chunk: torch.Tensor, target: torch.Tensor, lse: torch.Tensor, grad_total: torch.Tensor
) -> torch.Tensor:
    """Gradient of the summed loss with respect to one chunk, returned in the chunk's dtype."""
    probs = torch.exp(chunk.to(lse.dtype) - lse.unsqueeze(-1))
    columns = torch.arange(chunk.shape[-1], device=chunk.device)
    onehot = (columns == target.unsqueeze(-1)).to(probs.dtype)
    valid = (target != -100).unsqueeze(-1)
    return torch.where(valid, (probs - onehot) * grad_total, 0.0).to(chunk.dtype)


_CHUNK_LOSS_COMPILED = torch.compile(_chunk_loss)
_CHUNK_GRAD_COMPILED = torch.compile(_chunk_grad)


class _ChunkedCrossEntropySum(torch.autograd.Function):
    """Summed cross-entropy in at least fp32, one chunk of tokens at a time, softmax recomputed."""

    @staticmethod
    def forward(
        ctx: Any, logits: torch.Tensor, labels: torch.Tensor, chunk_size: int
    ) -> torch.Tensor:
        chunk_loss = _CHUNK_LOSS_COMPILED if logits.is_cuda else _chunk_loss
        compute_dtype = torch.promote_types(logits.dtype, torch.float32)
        lse = torch.empty(logits.shape[0], dtype=compute_dtype, device=logits.device)
        total = torch.zeros((), dtype=compute_dtype, device=logits.device)
        for start in range(0, logits.shape[0], chunk_size):
            stop = start + chunk_size
            part, chunk_lse = chunk_loss(logits[start:stop], labels[start:stop])
            lse[start:stop] = chunk_lse
            total += part
        ctx.save_for_backward(logits, labels, lse)
        ctx.chunk_size = chunk_size
        return total

    @staticmethod
    def backward(ctx: Any, grad_total: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        logits, labels, lse = ctx.saved_tensors
        chunk_grad = _CHUNK_GRAD_COMPILED if logits.is_cuda else _chunk_grad
        grad = torch.empty_like(logits)
        for start in range(0, logits.shape[0], ctx.chunk_size):
            stop = start + ctx.chunk_size
            grad[start:stop] = chunk_grad(
                logits[start:stop], labels[start:stop], lse[start:stop], grad_total
            )
        return grad, None, None


def _zero_on_graph(logits: torch.Tensor) -> torch.Tensor:
    """An exact float32 zero that stays on the autograd graph of ``logits``."""
    return logits[..., :0].sum().float()


@registry.register_loss("cross_entropy")
def cross_entropy_loss(
    logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = _DEFAULT_CHUNK_SIZE
) -> torch.Tensor:
    """Mean cross-entropy over tokens not labeled ``-100``, in fp32 one ``chunk_size`` at a time.

    A batch with every label ``-100`` returns an exact float32 zero that stays on the autograd
    graph of ``logits``.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid_tokens = (flat_labels != -100).sum()
    if valid_tokens == 0:
        return _zero_on_graph(logits)
    total = _ChunkedCrossEntropySum.apply(flat_logits, flat_labels, chunk_size)
    return total / valid_tokens


@registry.register_loss("chunked_cross_entropy")
def chunked_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Alias of :func:`cross_entropy_loss`, kept so configs naming it still load."""
    return cross_entropy_loss(logits, labels, chunk_size=chunk_size)


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
    Binds chunk_size for both cross-entropy losses and composes z-loss, so the caller
    gets a clean ``(logits, labels) -> Tensor`` interface.
    """
    base_fn = registry.get_loss(config.loss_fn)

    if config.loss_fn in _CHUNKED_LOSSES:
        chunk_size = config.ce_chunk_size if config.ce_chunk_size > 0 else _DEFAULT_CHUNK_SIZE
        base_fn = functools.partial(base_fn, chunk_size=chunk_size)

    if config.z_loss_weight > 0:
        z_weight = config.z_loss_weight
        _inner = base_fn

        def composed(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return _inner(logits, labels) + z_loss(logits, z_weight)

        return composed

    return base_fn
