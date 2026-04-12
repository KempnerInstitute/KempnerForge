"""Autoregressive text generation with KV-cache.

Supports greedy decoding, top-k, top-p (nucleus) sampling, and temperature
scaling. Designed for single-GPU research/debug use, not production serving.
"""

from __future__ import annotations

import torch

from kempnerforge.model.attention import KVCache
from kempnerforge.model.transformer import Transformer


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample next token from logits.

    Applies temperature scaling, top-k filtering, and nucleus (top-p) sampling
    in that order. Can be called independently for custom generation loops.

    Args:
        logits: (batch, vocab_size) unnormalized log-probabilities.
        temperature: Sampling temperature. 0 = greedy.
        top_k: Keep only top-k tokens. 0 = no filtering.
        top_p: Nucleus sampling threshold. 1.0 = no filtering.

    Returns:
        Token ids, shape (batch,).
    """
    if temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        threshold = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.where(logits >= threshold, torch.full_like(logits, float("-inf")))

    if top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cumulative_probs = probs.cumsum(dim=-1)
        # Mask tokens where cumulative prob (excluding current) exceeds top_p
        mask = (cumulative_probs - probs) >= top_p
        sorted_logits[mask] = float("-inf")
        # Unsort back to original vocabulary order
        logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate tokens autoregressively with KV-cache.

    Args:
        model: Transformer model (set to eval mode during generation).
        prompt_tokens: Input token ids, shape (batch, prompt_len).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature. 0 = greedy decoding.
        top_k: Top-k filtering. 0 = disabled.
        top_p: Nucleus sampling threshold. 1.0 = disabled.
        eos_token_id: Stop when all sequences produce this token.

    Returns:
        Full sequence (prompt + generated), shape (batch, total_len).
    """
    was_training = model.training
    model.eval()

    device = prompt_tokens.device
    batch_size, prompt_len = prompt_tokens.shape
    total_len = prompt_len + max_new_tokens
    config = model.config

    if total_len > config.max_seq_len:
        raise ValueError(
            f"prompt ({prompt_len}) + max_new_tokens ({max_new_tokens}) = {total_len} "
            f"exceeds model max_seq_len ({config.max_seq_len})"
        )

    dtype = next(model.parameters()).dtype

    # Allocate one KV cache per layer
    kv_caches = [
        KVCache(batch_size, total_len, config.n_kv_heads, config.head_dim, dtype, device)  # type: ignore[reportArgumentType]
        for _ in range(config.n_layers)
    ]

    # Prefill: forward the full prompt through the model
    logits = model(prompt_tokens, kv_caches=kv_caches)
    next_logits = logits[:, -1, :]

    # Autoregressive decode loop
    generated = []
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        next_token = sample(next_logits, temperature, top_k, top_p)
        generated.append(next_token)

        if eos_token_id is not None:
            done = done | (next_token == eos_token_id)
            if done.all():
                break

        # Single-token decode step
        next_logits = model(next_token.unsqueeze(1), kv_caches=kv_caches)[:, -1, :]

    if was_training:
        model.train()

    if generated:
        return torch.cat([prompt_tokens, torch.stack(generated, dim=1)], dim=1)
    return prompt_tokens
