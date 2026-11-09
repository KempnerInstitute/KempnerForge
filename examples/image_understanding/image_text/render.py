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
from .skip import MAX_ATTEMPTS, SkipLog, check_max_attempts, first_usable


class UnrenderableDocumentError(ValueError):
    """A document the renderer refuses: it would train nothing, or its replies cannot be located."""


class DocumentSource(Protocol):
    """A map-style reader of documents."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Document: ...


Span = tuple[int, int]


class ChatRenderer:
    """Renders a single-image document with a chat template into one VLM training sample.

    The sample holds ``pixel_values``, ``input_ids`` padded to ``max_text_len``, and
    next-token ``labels`` (``labels[i]`` is token ``i + 1``) that are ``-100`` on
    padding and, when ``answer_only``, outside the replies and their turn ends. The
    replies are located by character span in the templated text, so what the template
    writes before a reply (an empty think block, for instance) is never trained.
    Leading newlines of a reply are dropped, as the template drops them. A reply that
    is empty or that ``max_text_len`` cuts away entirely is refused with
    ``UnrenderableDocumentError``, since it would train nothing.
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
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("ChatRenderer needs a fast tokenizer to map replies to tokens")
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.image_size = image_size
        self.answer_only = answer_only
        self.image_placeholder = image_placeholder
        self.turn_end = turn_end
        self.image_mean = image_mean
        self.image_std = image_std
        self.pad_id = resolve_pad_id(tokenizer)
        self._header = f"{turn_start}assistant\n"
        prompt = self._template([{"role": "user", "content": ""}], generation_prompt=True)
        if self._header not in prompt:
            raise ValueError(
                f"the generation prompt {prompt!r} does not end a {self._header!r} turn"
            )
        self._reply_prefix = prompt[prompt.rindex(self._header) + len(self._header) :]
        turn = self._template(
            [{"role": "user", "content": ""}, {"role": "assistant", "content": "x"}],
            generation_prompt=False,
        )
        if f"x{turn_end}" not in turn:
            raise ValueError(f"turn_end {turn_end!r} does not end the template's assistant turns")

    def _template(self, messages: list[dict[str, str]], *, generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation_prompt,
            enable_thinking=False,
        )

    def text(self, document: Document, answers: Sequence[str]) -> str:
        """Chat-templated text of ``document`` with ``answers[i]`` replying to turn ``i``."""
        return self.text_and_reply_spans(document, answers)[0]

    def _replies(self, document: Document, answers: Sequence[str]) -> list[str]:
        if len(answers) != len(document.turns):
            raise ValueError(f"{len(document.turns)} turns but {len(answers)} answers")
        replies = [answer.lstrip("\n") for answer in answers]
        if not all(replies):
            raise UnrenderableDocumentError("an empty reply trains nothing")
        return replies

    def text_and_reply_spans(
        self, document: Document, answers: Sequence[str]
    ) -> tuple[str, list[Span]]:
        """The templated text and the character span of each reply with its turn end."""
        answers = self._replies(document, answers)
        placeholders = self.image_placeholder * len(document.images)
        first_user = document.turns[0].user
        if placeholders:
            first_user = f"{placeholders} {first_user}"
        if document.kind == "qa":
            prompt = self._template(
                [{"role": "user", "content": first_user}], generation_prompt=True
            )
            reply = answers[0] + self.turn_end
            return prompt + reply, [(len(prompt), len(prompt) + len(reply))]
        messages = []
        for i, (turn, answer) in enumerate(zip(document.turns, answers, strict=True)):
            user = first_user if i == 0 else turn.user
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": answer})
        text = self._template(messages, generation_prompt=False)
        spans: list[Span] = []
        cursor = 0
        for answer in answers:
            start = self._find_reply(text, answer + self.turn_end, cursor)
            cursor = start + len(answer) + len(self.turn_end)
            spans.append((start, cursor))
        return text, spans

    def _find_reply(self, text: str, reply: str, cursor: int) -> int:
        """Where ``reply`` starts an assistant turn of ``text`` at or after ``cursor``."""
        at = text.find(self._header, cursor)
        while at >= 0:
            start = at + len(self._header)
            if text.startswith(self._reply_prefix, start):
                start += len(self._reply_prefix)
            if text.startswith(reply, start):
                return start
            at = text.find(self._header, at + 1)
        raise UnrenderableDocumentError(
            f"the chat template did not render {reply!r} as an assistant turn"
        )

    def token_ids(self, document: Document, answers: Sequence[str]) -> list[int]:
        """Token ids of :meth:`text`, before truncation."""
        return self.tokenizer(self.text(document, answers), add_special_tokens=False)["input_ids"]

    def render(self, document: Document, answers: Sequence[str]) -> dict[str, torch.Tensor]:
        """The training sample for ``document`` with the given replies."""
        if len(document.images) != 1:
            raise ValueError(
                f"only single-image documents render to a sample, got {len(document.images)} images"
            )
        text, spans = self.text_and_reply_spans(document, answers)
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        full = encoded["input_ids"]
        ids = full[: self.max_text_len]
        if self.answer_only:
            trained = [
                any(s < end and e > start for start, end in spans)
                for s, e in encoded["offset_mapping"][: len(ids)]
            ]
        else:
            trained = [True] * len(ids)
        if not any(trained[1:]):
            raise UnrenderableDocumentError(
                f"the document trains no tokens: its replies fall past "
                f"max_text_len={self.max_text_len} ({len(full)} tokens in all)"
            )
        n = len(ids)
        input_ids = torch.full((self.max_text_len,), self.pad_id, dtype=torch.long)
        labels = torch.full((self.max_text_len,), -100, dtype=torch.long)
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
    """Renders a reader's documents, drawing replies from a generator seeded by seed and index.

    A document the renderer refuses is passed over for the next one, up to
    ``max_attempts`` in a row; ``skipped`` counts them, the first is logged.
    """

    def __init__(
        self,
        reader: DocumentSource,
        renderer: ChatRenderer,
        seed: int | str = 0,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.reader = reader
        self.renderer = renderer
        self.seed = seed
        self.max_attempts = check_max_attempts(max_attempts)
        self._skips = SkipLog("document")

    @property
    def skipped(self) -> int:
        return self._skips.count

    def __len__(self) -> int:
        return len(self.reader)

    def _render(self, index: int) -> dict[str, torch.Tensor]:
        return self.renderer(self.reader[index], random.Random(f"{self.seed}:{index}"))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} documents")
        return first_usable(
            index,
            len(self),
            self.max_attempts,
            self._render,
            self._skips,
            UnrenderableDocumentError,
        )
