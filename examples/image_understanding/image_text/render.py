"""Render image-text documents into fixed-length VLM training samples."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    image_to_views,
    resolve_pad_id,
)

from .document import Document
from .skip import MAX_ATTEMPTS, SkipLog, UnusableRowError, check_max_attempts, first_usable


class UnrenderableDocumentError(UnusableRowError, ValueError):
    """A document the renderer refuses: it would train nothing, or its replies cannot be located."""


class DocumentSource(Protocol):
    """A map-style reader of documents."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Document: ...


class DocumentStream(Protocol):
    """A resumable stream of documents, each with the position ``advance`` records."""

    def __iter__(self) -> Iterator[tuple[list[int], Document]]: ...

    def advance(self, position: list[int]) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


Span = tuple[int, int]


class ChatRenderer:
    """Renders a document with a chat template into one VLM training sample.

    The sample holds ``pixel_values``, ``input_ids`` padded to ``max_text_len``,
    next-token ``labels`` (``labels[i]`` is token ``i + 1``) that are ``-100`` on
    padding and, when ``answer_only``, outside the replies and their turn ends, and
    ``text_len``, the count of real tokens. ``max_text_len=None`` keeps the whole
    text unpadded, for packing. ``pixel_values`` follows the core dataset's shapes:
    by default (``max_images=None``, no cap) a document's images, or all their
    views, are concatenated to ``(N, 3, H, W)``, as with ``max_images`` above 1.
    With ``max_images=1`` an image is ``(3, H, W)``, or its ``V`` views
    ``(V, 3, H, W)`` with ``tile_grid``, and a text-only document is empty,
    ``(0, 3, H, W)`` or ``(0, V, 3, H, W)``. A document with more than
    ``max_images`` images is refused with ``UnrenderableDocumentError``. The
    replies are located by character span in the templated text, each where the
    template of the turns before it ends, so a user message that quotes a reply is
    not mistaken for it. What the template writes
    between the assistant header and a reply (an empty think block, for instance)
    is trained only with ``train_reply_prefix``, and only for a reply the text
    budget keeps a token of. The first user message follows the image
    placeholders after a space, and with ``lead_space`` starts with that space
    even when there are no placeholders.
    Leading newlines of a reply are dropped, as the template drops them. A document
    whose replies are all blank, a reply that carries template text (``turn_start``,
    ``turn_end`` or a think tag), or replies that ``max_text_len`` cuts away
    entirely are refused with ``UnrenderableDocumentError``. A document longer than
    ``max_text_len`` keeps its head and ends with ``turn_end``, as the core dataset
    ends a cut sample with EOS, so a cut reply still trains to stop.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_text_len: int | None,
        image_size: int,
        answer_only: bool,
        max_images: int | None = None,
        tile_grid: int = 0,
        tile_global: bool = True,
        train_reply_prefix: bool = False,
        image_placeholder: str = "",
        lead_space: bool = False,
        turn_start: str = "<|im_start|>",
        turn_end: str = "<|im_end|>",
        image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN,
        image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD,
    ) -> None:
        if max_text_len is not None and max_text_len < 2:
            raise ValueError(f"max_text_len must be at least 2 or None, got {max_text_len}")
        if max_images is not None and max_images < 1:
            raise ValueError(f"max_images must be positive or None, got {max_images}")
        if tile_grid < 0 or tile_grid == 1:
            raise ValueError(f"tile_grid must be 0 or at least 2, got {tile_grid}")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("ChatRenderer needs a fast tokenizer to map replies to tokens")
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.image_size = image_size
        self.answer_only = answer_only
        self.max_images = max_images
        self.tile_grid = tile_grid
        self.tile_global = tile_global
        self.train_reply_prefix = train_reply_prefix
        self.image_placeholder = image_placeholder
        self.lead_space = lead_space
        self.turn_end = turn_end
        self.image_mean = image_mean
        self.image_std = image_std
        self.pad_id = resolve_pad_id(tokenizer)
        self._marks = (turn_start, turn_end, "<think>", "</think>")
        end_ids = tokenizer(turn_end, add_special_tokens=False)["input_ids"]
        if len(end_ids) != 1:
            raise ValueError(f"turn_end {turn_end!r} must be one token, got {len(end_ids)}")
        self._end_id = end_ids[0]
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
        if not any(reply.strip() for reply in replies):
            raise UnrenderableDocumentError("a document whose replies are all empty trains nothing")
        if any(mark in reply for reply in replies for mark in self._marks):
            raise UnrenderableDocumentError("a reply carries template text")
        return replies

    def text_and_reply_spans(
        self, document: Document, answers: Sequence[str]
    ) -> tuple[str, list[Span]]:
        """The templated text and the character span of each reply with its turn end."""
        answers = self._replies(document, answers)
        placeholders = self.image_placeholder * len(document.images)
        first_user = document.turns[0].user
        if placeholders or self.lead_space:
            first_user = f"{placeholders} {first_user}"
        if document.kind == "qa":
            prompt = self._template(
                [{"role": "user", "content": first_user}], generation_prompt=True
            )
            reply = answers[0] + self.turn_end
            start = len(prompt)
            if self.train_reply_prefix and prompt.endswith(self._header + self._reply_prefix):
                start -= len(self._reply_prefix)
            return prompt + reply, [(start, len(prompt) + len(reply))]
        messages = []
        for i, (turn, answer) in enumerate(zip(document.turns, answers, strict=True)):
            user = first_user if i == 0 else turn.user
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": answer})
        text = self._template(messages, generation_prompt=False)
        spans: list[Span] = []
        for i, answer in enumerate(answers):
            prefix = self._template(messages[: 2 * i + 1], generation_prompt=False)
            if not text.startswith(prefix):
                raise UnrenderableDocumentError(
                    f"the chat template does not render turn {i} after the turns before it"
                )
            opened, start = self._reply_start(text, len(prefix), answer + self.turn_end)
            spans.append(
                (
                    opened if self.train_reply_prefix else start,
                    start + len(answer) + len(self.turn_end),
                )
            )
        return text, spans

    def prompt(self, document: Document) -> str:
        """The text before the first reply, the reply prefix included, as generation prompts it."""
        text, spans = self.text_and_reply_spans(document, ["x"] * len(document.turns))
        start = spans[0][0]
        if self.train_reply_prefix and text.startswith(self._reply_prefix, start):
            start += len(self._reply_prefix)
        return text[:start]

    def _reply_start(self, text: str, at: int, reply: str) -> tuple[int, int]:
        """Where the assistant turn at ``at`` ends its header, and where ``reply`` starts."""
        if text.startswith(self._header, at):
            at += len(self._header)
            opened = at
            if text.startswith(self._reply_prefix, at):
                at += len(self._reply_prefix)
            if text.startswith(reply, at):
                return opened, at
        raise UnrenderableDocumentError(
            f"the chat template did not render {reply!r} as an assistant turn"
        )

    def _kept_replies(
        self, text: str, spans: list[Span], offsets: Sequence[tuple[int, int]]
    ) -> list[Span]:
        """The reply spans of which a token past any trained reply prefix is kept."""
        prefix = self._reply_prefix if self.train_reply_prefix else ""
        kept = []
        for start, end in spans:
            reply = start + len(prefix) if prefix and text.startswith(prefix, start) else start
            if any(s < end and e > reply for s, e in offsets):
                kept.append((start, end))
        return kept

    def token_ids(self, document: Document, answers: Sequence[str]) -> list[int]:
        """Token ids of :meth:`text`, before truncation."""
        return self.tokenizer(self.text(document, answers), add_special_tokens=False)["input_ids"]

    def render(self, document: Document, answers: Sequence[str]) -> dict[str, torch.Tensor]:
        """The training sample for ``document`` with the given replies."""
        if self.max_images is not None and len(document.images) > self.max_images:
            raise UnrenderableDocumentError(
                f"the document holds {len(document.images)} images, over "
                f"max_images={self.max_images}"
            )
        text, spans = self.text_and_reply_spans(document, answers)
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        full = encoded["input_ids"]
        width = len(full) if self.max_text_len is None else self.max_text_len
        cut = len(full) > width
        kept = full[: width - 1] if cut else full
        ends_already = cut and kept[-1] == self._end_id
        ids = kept if ends_already or not cut else kept + [self._end_id]
        if self.answer_only:
            offsets = encoded["offset_mapping"][: len(kept)]
            spans = self._kept_replies(text, spans, offsets)
            trained = [any(s < end and e > start for start, end in spans) for s, e in offsets]
            if len(ids) > len(kept):
                trained.append(trained[-1])
        else:
            trained = [True] * len(ids)
        if not any(trained[1:]):
            raise UnrenderableDocumentError(
                f"the document trains no tokens: its replies fall past "
                f"max_text_len={self.max_text_len} ({len(full)} tokens in all)"
            )
        n = len(ids)
        input_ids = torch.full((width,), self.pad_id, dtype=torch.long)
        labels = torch.full((width,), -100, dtype=torch.long)
        ids_t = torch.tensor(ids, dtype=torch.long)
        input_ids[:n] = ids_t
        labels[: n - 1] = torch.where(torch.tensor(trained[1:]), ids_t[1:], -100)
        return {
            "pixel_values": self.pixels(document),
            "input_ids": input_ids,
            "labels": labels,
            "text_len": torch.tensor(n, dtype=torch.long),
        }

    def pixels(self, document: Document) -> torch.Tensor:
        """The sample's ``pixel_values`` for ``document``."""
        if not document.images:
            views = self.tile_grid**2 + int(self.tile_global) if self.tile_grid else 0
            tiled = (views,) if views and self.max_images == 1 else ()
            return torch.zeros(0, *tiled, 3, self.image_size, self.image_size)
        views = [
            image_to_views(
                image,
                self.image_size,
                self.image_mean,
                self.image_std,
                tile_grid=self.tile_grid,
                tile_global=self.tile_global,
            )
            for image in document.images
        ]
        if self.max_images == 1:
            return views[0]
        return torch.cat([v if v.dim() == 4 else v.unsqueeze(0) for v in views])

    def __call__(self, document: Document, rng: random.Random) -> dict[str, torch.Tensor]:
        """Render with each turn's reply drawn from its answers by ``rng``."""
        return self.render(document, [rng.choice(turn.answers) for turn in document.turns])


class RenderedDataset(Dataset):
    """Renders a reader's documents, drawing replies from a generator seeded by seed and index.

    A document the renderer refuses is passed over for the next one, up to
    ``max_attempts`` in a row. A reader whose ``max_images`` keeps rows with more
    images than the renderer's ``max_images`` is refused, since each such row would
    be passed over. ``skipped`` counts them with the rows the reader
    passed over, per rank: loader workers forked from the training process add to
    its count, and the first skip in each process is logged.
    """

    def __init__(
        self,
        reader: DocumentSource,
        renderer: ChatRenderer,
        seed: int | str = 0,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        kept = getattr(reader, "max_images", renderer.max_images)
        if renderer.max_images is not None and (kept is None or kept > renderer.max_images):
            raise ValueError(
                f"the reader keeps rows of up to {kept} images (None: any number), over the "
                f"renderer's max_images={renderer.max_images}; give both the same cap"
            )
        self.reader = reader
        self.renderer = renderer
        self.seed = seed
        self.max_attempts = check_max_attempts(max_attempts)
        self._skips = SkipLog("document")

    @property
    def skipped(self) -> int:
        return self._skips.count + getattr(self.reader, "skipped", 0)

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} documents")
        next_index = index + 1

        def render(row: int) -> dict[str, torch.Tensor]:
            nonlocal next_index
            next_index = row + 1
            read_with_index = getattr(self.reader, "read_with_index", None)
            actual, document = (
                read_with_index(row) if read_with_index is not None else (row, self.reader[row])
            )
            next_index = actual + 1
            return self.renderer(document, random.Random(f"{self.seed}:{row}"))

        return first_usable(
            index,
            len(self),
            self.max_attempts,
            render,
            self._skips,
            lambda row, error: next_index,
        )


class RenderedStream(IterableDataset):
    """Renders a stream of documents, passing over the ones the renderer refuses.

    Replies are drawn from a generator seeded by ``seed``, the data-parallel
    ``rank`` and the loader worker, started afresh by each pass, so a stream's draws
    do not follow a resume. ``max_attempts`` refusals in a row raise; ``skipped``
    counts them with the stream's own skips. Each sample keeps its document's
    position, and ``advance``, ``state_dict`` and ``load_state_dict`` are the
    stream's.
    """

    def __init__(
        self,
        documents: DocumentStream,
        renderer: ChatRenderer,
        seed: int | str = 0,
        rank: int = 0,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.documents = documents
        self.renderer = renderer
        self.seed = seed
        self.rank = rank
        self.max_attempts = check_max_attempts(max_attempts)
        self._skips = SkipLog("streamed document")

    @property
    def skipped(self) -> int:
        return self._skips.count + getattr(self.documents, "skipped", 0)

    def __iter__(self) -> Iterator[tuple[list[int], dict[str, torch.Tensor]]]:
        info = get_worker_info()
        rng = random.Random(f"{self.seed}:{self.rank}:{0 if info is None else info.id}")
        in_a_row = 0
        for count, (position, document) in enumerate(self.documents):
            try:
                sample = self.renderer(document, rng)
            except UnusableRowError as exc:
                self._skips(count, exc)
                in_a_row += 1
                if in_a_row >= self.max_attempts:
                    raise RuntimeError(
                        f"{self.max_attempts} streamed documents in a row cannot be rendered"
                    ) from exc
                continue
            in_a_row = 0
            yield position, sample

    def advance(self, position: list[int]) -> None:
        self.documents.advance(position)

    def state_dict(self) -> dict[str, Any]:
        return self.documents.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.documents.load_state_dict(state)
