"""Greedy decoding of a batch of documents, each prompt after its images.

A row holds one document laid out the way a packed training row lays a document
out: the tokens of every view of every image, then the text, padding last. The
views are encoded once and written over their slots at every step. Attention is
causal, and with ``view_groups`` a view's tokens also see each other, as training
let them. Nothing after a row's last token reaches it, so in exact arithmetic a
row decodes to the same tokens alone or in any batch; across batch sizes the
tokens match up to floating-point (bf16) numerics, which can flip a close argmax.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from kempnerforge.model.modality import ModalityContext
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


def _rows_of(context: ModalityContext, alive: torch.Tensor, batch: int) -> ModalityContext:
    """``context`` with the image tokens of the rows in ``alive`` only, renumbered in order."""
    assert context.inline_rows is not None and context.inline_embeds is not None
    assert context.inline_positions is not None
    renumber = torch.full((batch,), -1, dtype=torch.long, device=alive.device)
    renumber[alive] = torch.arange(alive.numel(), device=alive.device)
    keep = renumber[context.inline_rows] >= 0
    return replace(
        context,
        inline_embeds=context.inline_embeds[keep],
        inline_rows=renumber[context.inline_rows[keep]],
        inline_positions=context.inline_positions[keep],
    )


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
    right-padded with ``pad_id``. The views are encoded once, and each step appends
    the most likely next token to every row still decoding; the head runs on each
    row's last token only. A row ends on a token of ``stop_ids`` or after
    ``max_new_tokens`` tokens, and rows that have ended leave the batch. ``model``
    is decoded as it is, so pass it in eval mode.

    Raises:
        ValueError: before anything is computed, when there is no row, a prompt is
            empty, the rows' views differ in shape, ``max_new_tokens`` is not
            positive, a document and ``max_new_tokens`` exceed ``row_len``, or
            ``row_len`` exceeds the model's ``max_seq_len``.
    """
    _check_budget(model, rows, max_new_tokens, row_len)
    device = next(model.transformer.parameters()).device
    pixel_dtype = next((p.dtype for p in model.vision_encoder.parameters()), torch.float32)
    tokens = packed_tokens_per_image(model)
    batch = len(rows)
    width = max(footprint(model, row) for row in rows) + max_new_tokens

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
    length = start.clone()
    pixels = torch.cat([row.pixel_values for row in rows]).to(device=device, dtype=pixel_dtype)
    context = model.strategy.prepare(
        model,
        pixels,
        ids,
        image_rows=torch.tensor(image_rows, dtype=torch.long, device=device),
        image_starts=torch.tensor(image_starts, dtype=torch.long, device=device),
        image_tokens=tokens,
    )
    stops = torch.tensor(sorted(stop_ids), dtype=torch.long, device=device)
    stopped = torch.zeros(batch, dtype=torch.bool, device=device)

    alive = torch.arange(batch, device=device)
    current = context
    for _ in range(max_new_tokens):
        span = int(length[alive].max())
        last = length[alive] - 1
        logits = model.transformer(
            tokens=ids[alive, :span],
            modality=replace(
                current,
                output_rows=torch.arange(alive.numel(), device=device),
                output_positions=last,
            ),
            attention_group_ids=groups[alive, :span] if view_groups else None,
        )
        token = logits[0].argmax(dim=-1)
        ended = torch.isin(token, stops)
        if ended.any():
            stopped[alive[ended]] = True
            alive, token = alive[~ended], token[~ended]
            if alive.numel() == 0:
                break
            current = _rows_of(context, alive, batch)
        ids[alive, length[alive]] = token
        length[alive] += 1

    return [
        Generation(tuple(ids[row, start[row] : length[row]].tolist()), bool(stopped[row]))
        for row in range(batch)
    ]
