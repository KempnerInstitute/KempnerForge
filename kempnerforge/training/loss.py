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
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from kempnerforge.config.registry import registry
from kempnerforge.config.training import FP32_CROSS_ENTROPY_LOSSES


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


class _Fp32CrossEntropySum(torch.autograd.Function):
    """Summed cross-entropy in at least fp32, ``chunk_size`` tokens at a time."""

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
    @once_differentiable
    def backward(ctx: Any, grad_total: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        logits, labels, lse = ctx.saved_tensors
        chunk_grad = _CHUNK_GRAD_COMPILED if logits.is_cuda else _chunk_grad
        if ctx.chunk_size >= logits.shape[0]:
            return chunk_grad(logits, labels, lse, grad_total), None, None
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


def _fp32_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, chunk_size: int | None
) -> torch.Tensor:
    """Mean fp32 cross-entropy over tokens not labeled ``-100``; ``None`` means one chunk."""
    flat_logits = logits.view(-1, logits.shape[-1])
    flat_labels = labels.view(-1)
    valid_tokens = (flat_labels != -100).sum()
    if valid_tokens == 0:
        return _zero_on_graph(logits)
    size = flat_logits.shape[0] if chunk_size is None else chunk_size
    total = _Fp32CrossEntropySum.apply(flat_logits, flat_labels, size)
    return total / valid_tokens


@registry.register_loss("cross_entropy")
def cross_entropy_loss(
    logits: torch.Tensor, labels: torch.Tensor, fp32: bool = False
) -> torch.Tensor:
    """Standard cross-entropy loss for language modeling.

    Uses ``ignore_index=-100`` so packed-sequence boundary tokens (labeled -100
    by the dataset) are excluded from the loss. When no -100 labels are present,
    this has zero overhead. A batch with every label ``-100`` returns an exact
    float32 zero that stays on the autograd graph of ``logits``.

    With ``fp32=True`` the loss and its gradient are computed in fp32 even when the
    logits are bf16 (``train.fp32_cross_entropy``).
    """
    if fp32:
        return _fp32_cross_entropy(logits, labels, None)
    flat_labels = labels.view(-1)
    if (flat_labels == -100).all():
        return _zero_on_graph(logits)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), flat_labels, ignore_index=-100)


@registry.register_loss("chunked_cross_entropy")
def chunked_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 4096,
    fp32: bool = False,
) -> torch.Tensor:
    """Cross-entropy computed in token-dimension chunks.

    Chunks along the token dimension and uses PyTorch's fused CE kernel per
    chunk, avoiding an explicit float32 materialization of the full logit
    tensor. For Llama-3 7B (vocab=128K, batch=4, seq=4096), the manual
    logsumexp path would create a ~8 GB float32 copy; this implementation
    avoids that entirely.

    Uses ``ignore_index=-100`` so packed-sequence boundary tokens are excluded.
    When no -100 labels are present, this has zero overhead. A batch with every
    label ``-100`` returns an exact float32 zero that stays on the autograd graph
    of ``logits``.

    Note: the input logit tensor (B*S, V) is still fully materialized by the
    model's output head before reaching this function. For deeper savings
    (never materializing the full logit tensor), the output projection itself
    must be chunked in the model forward pass — a future enhancement.

    Args:
        logits: (batch, seq, vocab) or (tokens, vocab).
        labels: (batch, seq) or (tokens,).
        chunk_size: Number of tokens per chunk.
        fp32: Compute the loss and its gradient in fp32 from bf16 logits, still without an
            fp32 copy of the full logits (``train.fp32_cross_entropy``).
    """
    if fp32:
        return _fp32_cross_entropy(logits, labels, chunk_size)
    num_tokens = logits.shape[0] * logits.shape[1] if logits.ndim == 3 else logits.shape[0]
    flat_logits = logits.view(num_tokens, -1)
    flat_labels = labels.view(-1)

    if (flat_labels == -100).all():
        return _zero_on_graph(flat_logits)
    if num_tokens <= chunk_size:
        return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

    valid_tokens = (flat_labels != -100).sum()

    total_loss = torch.tensor(0.0, device=flat_logits.device, dtype=torch.float32)
    for i in range(0, num_tokens, chunk_size):
        total_loss = total_loss + F.cross_entropy(
            flat_logits[i : i + chunk_size],
            flat_labels[i : i + chunk_size],
            reduction="sum",
            ignore_index=-100,
        )
    return total_loss / valid_tokens


def z_loss(logits: torch.Tensor, weight: float) -> torch.Tensor:
    """Logit magnitude regularizer (PaLM / Gemini).

    Penalizes large logit magnitudes to prevent logit drift that causes
    NaN/divergence in long training runs. Negligible compute cost.

    Formula: weight * mean(logsumexp(logits, dim=-1) ** 2)

    Args:
        logits: Model output logits, shape (batch, seq, vocab) or (tokens, vocab).
        weight: Regularization weight (PaLM uses 1e-4).

    Returns:
        Scalar z-loss term to add to the main loss; zero on the graph for logits
        with no rows (a packed micro-batch without text).
    """
    if weight == 0.0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    flat = logits.view(-1, logits.shape[-1]).float()
    if flat.shape[0] == 0:
        return flat.sum()
    lse = torch.logsumexp(flat, dim=-1)
    return weight * (lse**2).mean()


def build_loss_fn(config) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a composed loss function from training config.

    Follows the build_optimizer pattern: config in, callable out.
    Binds chunk_size for chunked CE and ``fp32`` when ``train.fp32_cross_entropy`` is set,
    and composes z-loss, so the caller
    gets a clean ``(logits, labels) -> Tensor`` interface.
    """
    base_fn = registry.get_loss(config.loss_fn)

    if config.loss_fn == "chunked_cross_entropy":
        chunk_size = config.ce_chunk_size if config.ce_chunk_size > 0 else 4096
        base_fn = functools.partial(base_fn, chunk_size=chunk_size)

    if config.fp32_cross_entropy:
        if config.loss_fn not in FP32_CROSS_ENTROPY_LOSSES:
            raise ValueError(
                f"train.fp32_cross_entropy applies to {', '.join(FP32_CROSS_ENTROPY_LOSSES)}, "
                f"not loss_fn={config.loss_fn!r}"
            )
        base_fn = functools.partial(base_fn, fp32=True)

    if config.z_loss_weight > 0:
        z_weight = config.z_loss_weight
        _inner = base_fn

        def composed(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return _inner(logits, labels) + z_loss(logits, z_weight)

        return composed

    return base_fn
