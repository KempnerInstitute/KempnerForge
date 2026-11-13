"""Greedy decoding of a batch of prompts, each after its image.

A row is laid out the way training lays it out: the image tokens, then the
text, padding last. Attention is causal, so nothing after a row's last token
reaches it. In exact arithmetic a row therefore decodes to the same tokens alone
or in any batch; across batch sizes the tokens match up to floating-point (bf16)
numerics, which can flip a close argmax.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

import torch

from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.vlm import VLMWrapper

if TYPE_CHECKING:
    from image_text.render import ChatRenderer

_REUSED_FIELDS = ("prefix_embeds", "output_slice")


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


def _check_reusable(context: ModalityContext) -> None:
    """Refuses a modality context that cannot serve every decode step.

    The images are encoded once and reused while the text grows, which holds when
    the context is the image tokens and the slice that drops them from the
    output. A field laid out along the text would be stale after one step.
    """
    others = sorted(
        field.name
        for field in fields(context)
        if field.name not in _REUSED_FIELDS and getattr(context, field.name) is not None
    )
    prefix = context.prefix_embeds
    if prefix is None or others or context.output_slice != slice(prefix.shape[1], None):
        raise ValueError(
            "the decode encodes the images once and reuses them at every step, which needs a "
            "modality context of prefix_embeds and the output_slice past them; got "
            f"prefix_embeds {'set' if prefix is not None else 'unset'}, "
            f"output_slice={context.output_slice} and {others or 'no other field'}"
        )


def _check_budget(
    model: VLMWrapper,
    batch: int,
    lengths: Sequence[int],
    max_new_tokens: int,
    max_text_len: int,
) -> None:
    if not lengths or batch != len(lengths):
        raise ValueError(f"{batch} images for {len(lengths)} prompts; decode one prompt per image")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")
    if min(lengths) < 1:
        raise ValueError("a prompt has no tokens to decode after")
    if max(lengths) + max_new_tokens > max_text_len:
        raise ValueError(
            f"a prompt of {max(lengths)} tokens and max_new_tokens={max_new_tokens} do not fit "
            f"the text budget of {max_text_len} tokens"
        )
    max_seq_len = model.transformer.config.max_seq_len
    if model.num_image_tokens + max_text_len > max_seq_len:
        raise ValueError(
            f"{model.num_image_tokens} image tokens and a text budget of {max_text_len} tokens "
            f"exceed the model's max_seq_len={max_seq_len}"
        )


@torch.inference_mode()
def greedy_decode(
    model: VLMWrapper,
    pixel_values: torch.Tensor,
    prompts: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    stop_ids: Collection[int],
    pad_id: int,
    max_text_len: int,
) -> list[Generation]:
    """What ``model`` decodes greedily after each prompt, given the prompt's image.

    ``pixel_values`` is ``(batch, 3, H, W)`` and ``prompts`` holds one token
    sequence per image. The prompts are right-padded with ``pad_id`` into one
    batch, the images are encoded once, and each step appends the most likely
    next token to every row still decoding. A row ends on a token of
    ``stop_ids`` or after ``max_new_tokens`` tokens, and rows that have ended
    leave the batch. ``model`` is decoded as it is, so pass it in eval mode.

    Raises:
        ValueError: before anything is computed, when the images and prompts do
            not pair up, a prompt is empty, ``max_new_tokens`` is not positive,
            the longest prompt and ``max_new_tokens`` exceed ``max_text_len``, or
            the image tokens and ``max_text_len`` exceed the model's
            ``max_seq_len``; and when the model's modality context cannot be
            reused across steps.
    """
    lengths = [len(prompt) for prompt in prompts]
    _check_budget(model, pixel_values.shape[0], lengths, max_new_tokens, max_text_len)
    device = next(model.transformer.parameters()).device
    pixel_dtype = next((p.dtype for p in model.vision_encoder.parameters()), pixel_values.dtype)
    batch, longest = len(prompts), max(lengths)

    ids = torch.full((batch, longest + max_new_tokens), pad_id, dtype=torch.long, device=device)
    for row, prompt in enumerate(prompts):
        ids[row, : lengths[row]] = torch.tensor(prompt, dtype=torch.long, device=device)
    start = torch.tensor(lengths, dtype=torch.long, device=device)
    length = start.clone()
    stops = torch.tensor(sorted(stop_ids), dtype=torch.long, device=device)
    stopped = torch.zeros(batch, dtype=torch.bool, device=device)

    context = model.strategy.prepare(
        model, pixel_values.to(device=device, dtype=pixel_dtype), ids[:, :longest]
    )
    _check_reusable(context)
    image_tokens = context.prefix_embeds
    assert image_tokens is not None

    alive = torch.arange(batch, device=device)
    for _ in range(max_new_tokens):
        width = int(length[alive].max())
        logits = model.transformer(tokens=ids[alive, :width], modality=context)
        last = logits[torch.arange(alive.numel(), device=device), length[alive] - 1]
        token = last.argmax(dim=-1)
        ended = torch.isin(token, stops)
        if ended.any():
            stopped[alive[ended]] = True
            alive, token = alive[~ended], token[~ended]
            if alive.numel() == 0:
                break
            context = replace(context, prefix_embeds=image_tokens[alive])
        ids[alive, length[alive]] = token
        length[alive] += 1

    return [
        Generation(tuple(ids[row, start[row] : length[row]].tolist()), bool(stopped[row]))
        for row in range(batch)
    ]
