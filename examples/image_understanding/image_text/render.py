"""Render image-text documents into fixed-length VLM training samples."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset

from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    pil_to_tensor,
    resolve_pad_id,
)

from .document import Document


class DocumentSource(Protocol):
    """A map-style reader of documents."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Document: ...


def assistant_span_mask(
    ids: Sequence[int], generation_prompt: Sequence[int], turn_end_id: int
) -> list[bool]:
    """Mark every token after an assistant generation prompt, through its turn-end token."""
    mask = [False] * len(ids)
    prompt = list(generation_prompt)
    span = len(prompt)
    i = 0
    while i <= len(ids) - span:
        if list(ids[i : i + span]) != prompt:
            i += 1
            continue
        j = i + span
        while j < len(ids):
            mask[j] = True
            if ids[j] == turn_end_id:
                break
            j += 1
        i = j + 1
    return mask


class ChatRenderer:
    """Renders a single-image document with a chat template into one VLM training sample.

    The sample holds ``pixel_values``, ``input_ids`` padded to ``max_text_len``, and
    next-token ``labels`` (``labels[i]`` is token ``i + 1``) that are ``-100`` on
    padding and, when ``answer_only``, outside the assistant replies. A document
    whose replies ``max_text_len`` cuts away entirely is refused, since it would
    train nothing.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_text_len: int,
        image_size: int,
        answer_only: bool,
        image_placeholder: str = "",
        turn_start: str = "<|im_start|>",
        turn_end: str = "<|im_end|>",
        image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN,
        image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD,
    ) -> None:
        if max_text_len < 2:
            raise ValueError(f"max_text_len must be at least 2, got {max_text_len}")
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.image_size = image_size
        self.answer_only = answer_only
        self.image_placeholder = image_placeholder
        self.turn_end = turn_end
        self.image_mean = image_mean
        self.image_std = image_std
        self.pad_id = resolve_pad_id(tokenizer)
        self._generation_prompt = tokenizer(f"{turn_start}assistant\n", add_special_tokens=False)[
            "input_ids"
        ]
        self._turn_end_id = tokenizer.convert_tokens_to_ids(turn_end)
        if tokenizer(turn_end, add_special_tokens=False)["input_ids"] != [self._turn_end_id]:
            raise ValueError(f"turn_end {turn_end!r} is not a single token of the tokenizer")

    def _template(self, messages: list[dict[str, str]], *, generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation_prompt,
            enable_thinking=False,
        )

    def text(self, document: Document, answers: Sequence[str]) -> str:
        """Chat-templated text of ``document`` with ``answers[i]`` replying to turn ``i``."""
        if len(answers) != len(document.turns):
            raise ValueError(f"{len(document.turns)} turns but {len(answers)} answers")
        placeholders = self.image_placeholder * len(document.images)
        if document.kind == "qa":
            user = f"{placeholders} {document.turns[0].user}"
            prompt = self._template([{"role": "user", "content": user}], generation_prompt=True)
            return prompt + answers[0] + self.turn_end if answers[0] else prompt
        messages = []
        for i, (turn, answer) in enumerate(zip(document.turns, answers, strict=True)):
            user = f"{placeholders} {turn.user}" if i == 0 else turn.user
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": answer})
        return self._template(messages, generation_prompt=False)

    def token_ids(self, document: Document, answers: Sequence[str]) -> list[int]:
        """Token ids of :meth:`text`, before truncation."""
        return self.tokenizer(self.text(document, answers), add_special_tokens=False)["input_ids"]

    def render(self, document: Document, answers: Sequence[str]) -> dict[str, torch.Tensor]:
        """The training sample for ``document`` with the given replies."""
        if len(document.images) != 1:
            raise ValueError(
                f"only single-image documents render to a sample, got {len(document.images)} images"
            )
        full = self.token_ids(document, answers)
        ids = full[: self.max_text_len]
        if self.answer_only:
            trained = assistant_span_mask(ids, self._generation_prompt, self._turn_end_id)
        else:
            trained = [True] * len(ids)
        if not any(trained[1:]):
            raise ValueError(
                f"the document trains no tokens: its replies fall past "
                f"max_text_len={self.max_text_len} ({len(full)} tokens in all)"
            )
        n = len(ids)
        input_ids = torch.full((self.max_text_len,), self.pad_id, dtype=torch.long)
        labels = torch.full((self.max_text_len,), -100, dtype=torch.long)
        if n:
            ids_t = torch.tensor(ids, dtype=torch.long)
            input_ids[:n] = ids_t
            labels[: n - 1] = torch.where(torch.tensor(trained[1:]), ids_t[1:], -100)
        pixel_values = pil_to_tensor(
            document.images[0], self.image_size, self.image_mean, self.image_std
        )
        return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}

    def __call__(self, document: Document, rng: random.Random) -> dict[str, torch.Tensor]:
        """Render with each turn's reply drawn from its answers by ``rng``."""
        return self.render(document, [rng.choice(turn.answers) for turn in document.turns])


class RenderedDataset(Dataset):
    """Renders a reader's documents, drawing replies from a generator seeded by seed and index."""

    def __init__(self, reader: DocumentSource, renderer: ChatRenderer, seed: int | str = 0) -> None:
        self.reader = reader
        self.renderer = renderer
        self.seed = seed

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.renderer(self.reader[index], random.Random(f"{self.seed}:{index}"))
