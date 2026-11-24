"""Reads the answer to a question out of a free reply with an extractive question-answering model.

``AnswerExtractor`` returns the span of a reply (the context) that a model
fine-tuned for extractive question answering takes to answer a question, the
model loaded on the extractor's first call. ``bigbird_extractor`` makes one of
``google/bigbird-base-trivia-itc`` at a pinned revision, whose files it fetches
into the Hugging Face cache when it is made. The span is chosen as the
transformers question-answering pipeline of transformers 4 chose its single
answer, which transformers 5 no longer has, called as the recipe calls it:
windows over the context as long as the model reads (the recipe sets the
pipeline's ``max_seq_len`` to the tokenizer's 4,096 in place of its default 384)
with the pipeline's stride, a softmax over each window's context tokens, the
best spans of up to ``MAX_ANSWER_TOKENS`` tokens widened to whole words, and the
texts of equal spans, ignoring case, adding up their scores.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from kempnerforge.metrics.logger import get_logger

if TYPE_CHECKING:
    from tokenizers import Encoding, Tokenizer

logger = get_logger(__name__)

EXTRACTOR_MODEL = "google/bigbird-base-trivia-itc"
EXTRACTOR_REVISION = "b653c106e2cae13eea0d8d08b5c73101eb923fc3"
EXTRACTOR_FILES = ("config.json", "pytorch_model.bin", "spiece.model")
MAX_ANSWER_TOKENS = 15
CANDIDATE_SPANS = 12
WINDOW_STRIDE = 128
MASKED_LOGIT = -10000.0
CONTEXT = 1
CONTROL_PIECE = 3
USER_DEFINED_PIECE = 4
PIECE_MARK = "\u2581"

Span = tuple[float, int, int]


@dataclass(frozen=True)
class SpanModel:
    """An extractive question-answering model with the tokenizer it reads text with.

    ``tokenizer`` encodes a question and a context as one pair, the context cut
    into overflowing windows of the model's length; ``model`` maps a window's
    ``input_ids`` and ``attention_mask`` to ``start_logits`` and ``end_logits``
    for every token; ``cls_id`` is the token the model scores as no answer.
    """

    tokenizer: Tokenizer
    model: Callable[..., Any]
    cls_id: int


def bigbird_tokenizer(spiece_model: Path, max_length: int) -> Tokenizer:
    """BigBird's tokenizer, built from its SentencePiece model as transformers 4 converted it.

    The pieces and their scores make a Unigram model; trailing whitespace is
    stripped and runs of spaces become one piece marker before the text is split
    at spaces. The control and user-defined pieces are matched whole in the text
    before it is normalized, the control ones as special tokens, as transformers
    4 registers them; the special tokens are then added as BigBird declares them,
    and a pair is laid out as ``[CLS] question [SEP] context [SEP]``. A pair
    longer than ``max_length`` tokens is cut into windows that overlap by
    ``WINDOW_STRIDE`` tokens, the context truncated from the left. The file is
    read with ``protobuf``, so the ``sentencepiece`` package is not needed.
    """
    from tokenizers import AddedToken, Regex, Tokenizer, normalizers, pre_tokenizers, processors
    from tokenizers.models import Unigram
    from transformers.convert_slow_tokenizer import import_protobuf

    messages: Any = import_protobuf()
    proto = messages.ModelProto()
    proto.ParseFromString(spiece_model.read_bytes())
    tokenizer = Tokenizer(
        Unigram(
            [(piece.piece, piece.score) for piece in proto.pieces],
            unk_id=proto.trainer_spec.unk_id,
            byte_fallback=proto.trainer_spec.byte_fallback,
        )
    )
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Strip(left=False, right=True), normalizers.Replace(Regex(" {2,}"), PIECE_MARK)]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=PIECE_MARK, prepend_scheme="always", split=True
    )
    tokenizer.add_tokens(
        [
            AddedToken(piece.piece, normalized=False, special=piece.type == CONTROL_PIECE)
            for piece in proto.pieces
            if piece.type in (CONTROL_PIECE, USER_DEFINED_PIECE)
        ]
    )
    tokenizer.add_special_tokens(
        [
            AddedToken(token, lstrip=token == "[MASK]", normalized=True, special=True)
            for token in ("<pad>", "</s>", "<s>", "[CLS]", "[SEP]", "[MASK]", "<unk>")
        ]
    )
    special = [(token, tokenizer.token_to_id(token)) for token in ("[CLS]", "[SEP]")]
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=special,
    )
    tokenizer.enable_truncation(
        max_length, stride=WINDOW_STRIDE, strategy="only_second", direction="left"
    )
    return tokenizer


def fetch_bigbird() -> Path:
    """The local folder of ``EXTRACTOR_FILES`` of ``EXTRACTOR_MODEL`` at ``EXTRACTOR_REVISION``.

    The files are downloaded into the Hugging Face cache unless they are there.
    """
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            EXTRACTOR_MODEL, revision=EXTRACTOR_REVISION, allow_patterns=list(EXTRACTOR_FILES)
        )
    )


def load_bigbird(device: torch.device, folder: Path) -> SpanModel:
    """BigBird and its tokenizer from ``folder`` (``fetch_bigbird``'s), on ``device``.

    The model attends fully at every length: BigBird's block-sparse attention
    gives way to full attention at its first input of 704 tokens or fewer and
    keeps it, so a model that started block-sparse would read a long input
    differently before that first short one than after it. transformers logs
    only errors while the model loads, as the recipe has it: its load report
    names a position buffer the checkpoint carries and the model rebuilds.
    """
    from transformers import BigBirdForQuestionAnswering
    from transformers.utils import logging as transformers_logging

    verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        model: Any = BigBirdForQuestionAnswering.from_pretrained(
            folder, attention_type="original_full", dtype=torch.float32
        )
    finally:
        transformers_logging.set_verbosity(verbosity)
    model.to(device).eval()
    tokenizer = bigbird_tokenizer(folder / "spiece.model", model.config.max_position_embeddings)
    cls_id = tokenizer.token_to_id("[CLS]")
    assert cls_id is not None
    return SpanModel(tokenizer, model, cls_id)


def _probabilities(logits: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """A softmax of ``logits`` over the ``kept`` tokens, the others masked out."""
    masked = np.where(kept, logits, np.float32(MASKED_LOGIT))
    weights = np.exp(masked - masked.max())
    return weights / weights.sum()


def answer_spans(start: np.ndarray, end: np.ndarray, kept: np.ndarray) -> list[Span]:
    """The best answer spans of one window, best first: score, first token and last token.

    ``start`` and ``end`` are a window's start and end logits per token, and
    ``kept`` marks the tokens a softmax runs over: the context's and the
    no-answer token, which is the window's first. After the softmax the first
    token's probabilities are set to 0, and a span scores the product of its
    first token's start and its last token's end probability. Spans of up to
    ``MAX_ANSWER_TOKENS`` tokens count, and the ``CANDIDATE_SPANS`` best with a
    positive score are returned; equal scores keep the earlier span first.
    """
    first_probabilities = _probabilities(np.asarray(start), np.asarray(kept))
    last_probabilities = _probabilities(np.asarray(end), np.asarray(kept))
    first_probabilities[0] = last_probabilities[0] = 0.0
    tokens = first_probabilities.size
    spans: list[Span] = []
    for length in range(min(MAX_ANSWER_TOKENS, tokens)):
        scores = first_probabilities[: tokens - length] * last_probabilities[length:]
        spans.extend(
            (float(scores[first]), int(first), int(first) + length)
            for first in np.flatnonzero(scores > 0)
        )
    spans.sort(key=lambda span: (-span[0], span[1], span[2]))
    return spans[:CANDIDATE_SPANS]


def _characters(window: Encoding, first: int, last: int) -> tuple[int, int]:
    """The context characters of the words holding tokens ``first`` to ``last``.

    A token outside a word (an added token) falls back to the tokens' own offsets.
    """
    first_word, last_word = window.token_to_word(first), window.token_to_word(last)
    if first_word is not None and last_word is not None:
        begin = window.word_to_chars(first_word, CONTEXT)
        stop = window.word_to_chars(last_word, CONTEXT)
        if begin is not None and stop is not None:
            return begin[0], stop[1]
    return window.offsets[first][0], window.offsets[last][1]


def check_extractor_device(device: str) -> torch.device:
    """``device`` as a torch device, refused unless it is the CPU or a CUDA device this process has.

    Raises:
        ValueError: ``device`` is not a torch device, is neither ``cpu`` nor
            ``cuda``, or is a CUDA device and this process has no CUDA or no
            device of that index.
    """
    try:
        resolved = torch.device(device)
    except RuntimeError as exc:
        raise ValueError(f"extractor_device {device!r} is not a torch device: {exc}") from exc
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError(f"extractor_device {device!r} must be a cpu or cuda device")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"extractor_device {device!r} needs CUDA, which is not available")
    if resolved.type == "cuda" and (resolved.index or 0) >= torch.cuda.device_count():
        raise ValueError(
            f"extractor_device {device!r} names a CUDA device this process does not have; "
            f"it sees {torch.cuda.device_count()}"
        )
    return resolved


class AnswerExtractor:
    """Reads the answer to a question out of a reply, the span a model takes to answer it.

    ``load`` builds the model on ``device`` from local files; it runs on the
    first call, which a reply that needs no model never makes, and a load that
    raises is tried again by the next call. Calling the extractor returns the
    span of ``context`` that answers ``question``, as described in this module,
    or ``""`` when the context holds no token (only whitespace), where the
    pipeline finds no answer at all.

    Raises:
        ValueError: ``device`` is refused by ``check_extractor_device``.
    """

    def __init__(self, device: str, load: Callable[[torch.device], SpanModel]) -> None:
        self.device = check_extractor_device(device)
        self._load = load
        self._loaded: SpanModel | None = None

    def _span_model(self) -> SpanModel:
        if self._loaded is None:
            started = time.perf_counter()
            self._loaded = self._load(self.device)
            logger.info(
                f"Loaded the answer extractor on {self.device} in "
                f"{time.perf_counter() - started:.1f} s"
            )
        return self._loaded

    def _window_spans(self, span_model: SpanModel, window: Encoding) -> list[Span]:
        ids = torch.tensor([window.ids], device=self.device)
        attention = torch.tensor([window.attention_mask], device=self.device)
        with torch.inference_mode():
            output = span_model.model(input_ids=ids, attention_mask=attention)
        kept = np.array([sequence == CONTEXT for sequence in window.sequence_ids])
        kept |= np.array(window.ids) == span_model.cls_id
        return answer_spans(
            output.start_logits[0].float().cpu().numpy(),
            output.end_logits[0].float().cpu().numpy(),
            kept,
        )

    def __call__(self, question: str, context: str) -> str:
        """The span of ``context`` that answers ``question``.

        Raises:
            ValueError: ``question`` or ``context`` is empty.
        """
        if not question or not context:
            raise ValueError("an answer is read from a non-empty context for a non-empty question")
        span_model = self._span_model()
        encoding = span_model.tokenizer.encode(question, context)
        windows: Sequence[Encoding] = [encoding, *encoding.overflowing]
        totals: dict[str, float] = {}
        texts: dict[str, str] = {}
        for window in windows:
            for score, first, last in self._window_spans(span_model, window):
                begin, stop = _characters(window, first, last)
                text = context[begin:stop]
                key = text.lower()
                if key not in totals:
                    totals[key], texts[key] = 0.0, text
                totals[key] += score
        if not totals:
            return ""
        return texts[max(totals, key=totals.__getitem__)]


def bigbird_extractor(device: str = "cpu") -> AnswerExtractor:
    """An ``AnswerExtractor`` of ``EXTRACTOR_MODEL`` on ``device``.

    Its files are fetched now (``fetch_bigbird``), so a download that fails
    does so before anything is scored, and every process that makes one
    fetches at the same point; the model is loaded from them on the
    extractor's first call.

    Raises:
        ValueError: ``device`` is refused by ``check_extractor_device``, which is
            checked before anything is fetched.
    """
    check_extractor_device(device)
    return AnswerExtractor(device, partial(load_bigbird, folder=fetch_bigbird()))
