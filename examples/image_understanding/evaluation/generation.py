"""Greedy decoding of a batch of documents, each prompt after its images.

A row holds one document laid out the way a packed training row lays a document
out: the tokens of every view of every image, then the text, padding last. The
rows are run once into a KV cache, with the views encoded once and written over
their slots, and each step then runs only the new token of every row against the
cache. Attention is causal, and with ``view_groups`` a view's tokens also see each
other, as training let them. A row's padding stays in the cache, where its later
tokens skip it. Nothing after a row's last token reaches it, so in exact
arithmetic a row decodes to the same tokens alone or in any batch; across batch
sizes the tokens match up to floating-point (bf16) numerics, which can flip a
close argmax.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from kempnerforge.model.attention import KVCache
from kempnerforge.model.vlm import VLMWrapper, packed_tokens_per_image

if TYPE_CHECKING:
    from image_text.render import ChatRenderer

    from .inputs import ModelInputs


@dataclass(frozen=True)
class Generation:
    """The tokens decoded after one prompt.

    ``stopped`` says the decode ended on a stop token, which ``token_ids`` leaves
    out; otherwise it ran to ``max_new_tokens``.
    """

    token_ids: tuple[int, ...]
    stopped: bool


def stop_token_ids(renderer: ChatRenderer) -> frozenset[int]:
    """The tokens that end a reply: the renderer's turn end and the tokenizer's EOS, if any."""
    tokenizer = renderer.tokenizer
    stops = set(tokenizer(renderer.turn_end, add_special_tokens=False)["input_ids"])
    if tokenizer.eos_token_id is not None:
        stops.add(tokenizer.eos_token_id)
    return frozenset(stops)


def footprint(model: VLMWrapper, inputs: ModelInputs) -> int:
    """Positions a document's views and prompt take in a row."""
    return inputs.pixel_values.shape[0] * packed_tokens_per_image(model) + len(inputs.prompt_ids)


def _check_budget(
    model: VLMWrapper, rows: Sequence[ModelInputs], max_new_tokens: int, row_len: int
) -> None:
    if not rows:
        raise ValueError("no document to decode")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")
    if min(len(row.prompt_ids) for row in rows) < 1:
        raise ValueError("a prompt has no tokens to decode after")
    shapes = {tuple(row.pixel_values.shape[1:]) for row in rows}
    if len(shapes) > 1:
        raise ValueError(f"the documents' views differ in shape: {sorted(shapes)}")
    longest = max(footprint(model, row) for row in rows)
    if longest + max_new_tokens > row_len:
        raise ValueError(
            f"a document of {longest} positions and max_new_tokens={max_new_tokens} do not fit "
            f"a row of {row_len}"
        )
    max_seq_len = model.transformer.config.max_seq_len
    if row_len > max_seq_len:
        raise ValueError(f"a row of {row_len} exceeds the model's max_seq_len={max_seq_len}")


@torch.inference_mode()
def greedy_decode(
    model: VLMWrapper,
    rows: Sequence[ModelInputs],
    *,
    max_new_tokens: int,
    stop_ids: Collection[int],
    pad_id: int,
    row_len: int,
    view_groups: bool = False,
) -> list[Generation]:
    """What ``model`` decodes greedily after each document's prompt, given its images.

    Each row is one document of ``rows``: its views' token slots, then its prompt,
    right-padded with ``pad_id``. The rows run once into a KV cache, with the views
    encoded once, and each step appends the most likely next token to every row
    still decoding, running only that token; the head runs on each row's last
    token only. A row ends on a token of ``stop_ids`` or after ``max_new_tokens``
    tokens, and rows that have ended leave the batch and the cache. ``model`` is
    decoded as it is, so pass it in eval mode.

    Raises:
        ValueError: before anything is computed, when there is no row, a prompt is
            empty, the rows' views differ in shape, ``max_new_tokens`` is not
            positive, a document and ``max_new_tokens`` exceed ``row_len``, or
            ``row_len`` exceeds the model's ``max_seq_len``.
    """
    _check_budget(model, rows, max_new_tokens, row_len)
    device = next(model.transformer.parameters()).device
    cache_dtype = next(model.transformer.parameters()).dtype
    pixel_dtype = next((p.dtype for p in model.vision_encoder.parameters()), torch.float32)
    tokens = packed_tokens_per_image(model)
    batch = len(rows)
    width = max(footprint(model, row) for row in rows)

    ids = torch.full((batch, width), pad_id, dtype=torch.long)
    groups = torch.full((batch, width), -1, dtype=torch.long)
    image_rows: list[int] = []
    image_starts: list[int] = []
    start = torch.zeros(batch, dtype=torch.long)
    for index, row in enumerate(rows):
        views = row.pixel_values.shape[0]
        for view in range(views):
            groups[index, view * tokens : (view + 1) * tokens] = len(image_rows)
            image_rows.append(index)
            image_starts.append(view * tokens)
        slots = views * tokens
        ids[index, slots : slots + len(row.prompt_ids)] = torch.tensor(row.prompt_ids)
        start[index] = slots + len(row.prompt_ids)
    ids, groups, start = ids.to(device), groups.to(device), start.to(device)
    pixels = torch.cat([row.pixel_values for row in rows]).to(device=device, dtype=pixel_dtype)
    context = model.strategy.prepare(
        model,
        pixels,
        ids,
        image_rows=torch.tensor(image_rows, dtype=torch.long, device=device),
        image_starts=torch.tensor(image_starts, dtype=torch.long, device=device),
        image_tokens=tokens,
    )
    config = model.transformer.config
    caches = [
        KVCache(
            batch,
            width + max_new_tokens,
            config.n_kv_heads or config.n_heads,
            config.head_dim,
            cache_dtype,
            device,
        )
        for _ in range(config.n_layers)
    ]
    logits = model.transformer(
        tokens=ids,
        modality=replace(
            context, output_rows=torch.arange(batch, device=device), output_positions=start - 1
        ),
        attention_group_ids=groups if view_groups else None,
        kv_caches=caches,
    )
    stops = torch.tensor(sorted(stop_ids), dtype=torch.long, device=device)
    prompt_keys = torch.arange(width, device=device) < start.unsqueeze(1)
    decoded: list[list[int]] = [[] for _ in range(batch)]
    stopped = [False] * batch

    alive = torch.arange(batch, device=device)
    for step in range(max_new_tokens):
        token = logits[0].argmax(dim=-1)
        ended = torch.isin(token, stops)
        for row, value, end in zip(alive.tolist(), token.tolist(), ended.tolist(), strict=True):
            if end:
                stopped[row] = True
            else:
                decoded[row].append(value)
        if step == max_new_tokens - 1:
            break
        if ended.any():
            keep = (~ended).nonzero().squeeze(1)
            alive, token = alive[keep], token[keep]
            if alive.numel() == 0:
                break
            for cache in caches:
                cache.keep_rows(keep)
        generated = torch.ones(alive.numel(), step + 1, dtype=torch.bool, device=device)
        logits = model.transformer(
            tokens=token.unsqueeze(1),
            position_ids=(start[alive] + step).unsqueeze(1),
            kv_caches=caches,
            valid_keys=torch.cat([prompt_keys[alive], generated], dim=1),
        )
        logits = logits[:, -1].unsqueeze(0)

    return [Generation(tuple(decoded[row]), stopped[row]) for row in range(batch)]
