"""Tests for the extractive answer reader, with a tiny SentencePiece model and stand-in models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from evaluation import extraction
from evaluation.extraction import (
    CANDIDATE_SPANS,
    EXTRACTOR_FILES,
    EXTRACTOR_MODEL,
    EXTRACTOR_REVISION,
    MAX_ANSWER_TOKENS,
    WINDOW_STRIDE,
    AnswerExtractor,
    SpanModel,
    answer_spans,
    bigbird_extractor,
    bigbird_tokenizer,
    check_extractor_device,
    fetch_bigbird,
    load_bigbird,
)

NORMAL, UNKNOWN, CONTROL, USER_DEFINED = 1, 2, 3, 4
PIECES = (
    ("<pad>", CONTROL),
    ("</s>", CONTROL),
    ("<s>", CONTROL),
    ("<sep_0>", USER_DEFINED),
    ("[CLS]", CONTROL),
    ("[SEP]", CONTROL),
    ("[MASK]", CONTROL),
    ("<unk>", UNKNOWN),
    *(
        (piece, NORMAL)
        for piece in (
            "\u2581",
            "\u2581the",
            "\u2581answer",
            "\u2581is",
            "\u258142",
            "42",
            ".",
            "\u2581what",
            "?",
            "\u2581how",
            "\u2581many",
            "\u2581cat",
            "\u2581Cat",
            "\u2581dog",
            "\u2581a",
            "a",
            "b",
            "c",
            "d",
            "o",
            "g",
            "t",
        )
    ),
)
TOKENS = 512
FAR = "a " * MAX_ANSWER_TOKENS


@pytest.fixture(scope="module")
def spiece_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A SentencePiece model file of ``PIECES``, written with protobuf alone."""
    from transformers.convert_slow_tokenizer import import_protobuf

    proto = import_protobuf().ModelProto()
    for index, (piece, kind) in enumerate(PIECES):
        entry = proto.pieces.add()
        entry.piece, entry.type = piece, kind
        entry.score = 0.0 if kind != NORMAL else -float(index)
    proto.trainer_spec.unk_id = [piece for piece, _ in PIECES].index("<unk>")
    path = tmp_path_factory.mktemp("spiece") / "spiece.model"
    path.write_bytes(proto.SerializeToString())
    return path


def _id(tokenizer: Any, token: str) -> int:
    found = tokenizer.token_to_id(token)
    assert found is not None
    return found


class _Logits:
    """A stand-in span model: chosen start and end logits for chosen token ids, 0 elsewhere."""

    def __init__(self, starts: dict[int, float], ends: dict[int, float]) -> None:
        self.starts, self.ends, self.calls = starts, ends, 0

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        self.calls += 1
        ids = input_ids[0].tolist()
        start = torch.tensor([[self.starts.get(token, 0.0) for token in ids]])
        end = torch.tensor([[self.ends.get(token, 0.0) for token in ids]])
        return SimpleNamespace(start_logits=start, end_logits=end)


def _extractor(
    spiece_model: Path,
    starts: dict[str, float],
    ends: dict[str, float],
    max_length: int = TOKENS,
) -> tuple[AnswerExtractor, _Logits]:
    tokenizer = bigbird_tokenizer(spiece_model, max_length)
    model = _Logits(
        {_id(tokenizer, token): value for token, value in starts.items()},
        {_id(tokenizer, token): value for token, value in ends.items()},
    )
    span_model = SpanModel(tokenizer, model, _id(tokenizer, "[CLS]"))
    return AnswerExtractor("cpu", lambda device: span_model), model


def _load_nothing(device: torch.device) -> SpanModel:
    raise AssertionError("nothing is loaded")


class TestAnswerSpans:
    def test_a_span_scores_its_start_and_end_probabilities(self) -> None:
        start = np.array([0.0, 2.0, 1.0, 0.0], dtype=np.float32)
        end = np.array([0.0, 0.0, 1.0, 3.0], dtype=np.float32)
        spans = answer_spans(start, end, np.ones(4, dtype=bool))
        start_p = np.exp(start - start.max()) / np.exp(start - start.max()).sum()
        end_p = np.exp(end - end.max()) / np.exp(end - end.max()).sum()
        best, first, last = spans[0]
        assert (first, last) == (1, 3)
        assert best == pytest.approx(float(start_p[1] * end_p[3]))

    def test_only_kept_tokens_are_in_the_softmax_and_the_spans(self) -> None:
        start = np.array([0.0, 9.0, 1.0, 1.0], dtype=np.float32)
        end = np.array([0.0, 9.0, 1.0, 1.0], dtype=np.float32)
        kept = np.array([True, False, True, True])
        spans = answer_spans(start, end, kept)
        assert all(1 not in (first, last) for _, first, last in spans)
        assert sum(score for score, _, _ in spans) < 1.0

    def test_the_first_token_never_starts_or_ends_a_span(self) -> None:
        start = np.array([50.0, 1.0, 0.0], dtype=np.float32)
        end = np.array([50.0, 0.0, 1.0], dtype=np.float32)
        spans = answer_spans(start, end, np.ones(3, dtype=bool))
        assert spans and all(first > 0 and last > 0 for _, first, last in spans)
        assert spans[0][1:] == (1, 2)

    def test_a_span_is_at_most_the_answer_length(self) -> None:
        tokens = MAX_ANSWER_TOKENS + 4
        start, end = np.zeros(tokens, dtype=np.float32), np.zeros(tokens, dtype=np.float32)
        start[1], end[1 + MAX_ANSWER_TOKENS] = 40.0, 40.0
        spans = answer_spans(start, end, np.ones(tokens, dtype=bool))
        assert all(last - first < MAX_ANSWER_TOKENS for _, first, last in spans)

    def test_the_best_spans_come_first_and_ties_keep_the_earlier(self) -> None:
        start = np.ones(40, dtype=np.float32)
        end = np.ones(40, dtype=np.float32)
        spans = answer_spans(start, end, np.ones(40, dtype=bool))
        assert len(spans) == CANDIDATE_SPANS
        assert [span[1:] for span in spans[:3]] == [(1, 1), (1, 2), (1, 3)]
        assert [score for score, _, _ in spans] == sorted((s for s, _, _ in spans), reverse=True)

    def test_a_window_whose_only_kept_token_is_the_first_has_no_span(self) -> None:
        kept = np.array([True, False, False])
        assert answer_spans(np.ones(3), np.ones(3), kept) == []


class TestBigBirdTokenizer:
    def test_a_pair_is_the_question_then_the_context_between_separators(
        self, spiece_model: Path
    ) -> None:
        tokenizer = bigbird_tokenizer(spiece_model, TOKENS)
        encoding = tokenizer.encode("how many?", "the answer is 42.")
        assert encoding.tokens[0] == "[CLS]"
        assert encoding.tokens.count("[SEP]") == 2 and encoding.tokens[-1] == "[SEP]"
        sequences = encoding.sequence_ids
        first_sep = encoding.tokens.index("[SEP]")
        assert set(sequences[1:first_sep]) == {0}
        assert set(sequences[first_sep + 1 : -1]) == {1}
        assert sequences[0] is None and sequences[-1] is None

    @pytest.mark.parametrize("context", ["a cat \n ", "a   cat", "a  cat\t"])
    def test_trailing_whitespace_goes_and_a_run_of_spaces_is_one(
        self, spiece_model: Path, context: str
    ) -> None:
        tokenizer = bigbird_tokenizer(spiece_model, TOKENS)
        assert tokenizer.encode("what?", context).ids == tokenizer.encode("what?", "a cat").ids

    def test_a_user_defined_piece_is_matched_whole(self, spiece_model: Path) -> None:
        tokenizer = bigbird_tokenizer(spiece_model, TOKENS)
        assert "<sep_0>" in tokenizer.encode("what?", "a<sep_0>b").tokens

    @pytest.mark.parametrize(
        ("context", "pieces"),
        [
            ("a </s>", ["\u2581a", "</s>"]),
            ("a  </s>", ["\u2581a", "</s>"]),
            ("a <pad>", ["\u2581a", "<pad>"]),
            ("</s> a", ["</s>", "\u2581a"]),
        ],
    )
    def test_a_control_piece_is_matched_before_the_text_is_normalized(
        self, spiece_model: Path, context: str, pieces: list[str]
    ) -> None:
        tokenizer = bigbird_tokenizer(spiece_model, TOKENS)
        tokens = tokenizer.encode("what?", context).tokens
        assert tokens[tokens.index("[SEP]") + 1 : -1] == pieces

    def test_a_long_context_is_cut_into_overlapping_windows(self, spiece_model: Path) -> None:
        length = WINDOW_STRIDE + 40
        tokenizer = bigbird_tokenizer(spiece_model, length)
        encoding = tokenizer.encode("what?", " ".join(["a"] * 400))
        windows = [encoding, *encoding.overflowing]
        assert len(windows) > 1
        assert all(len(window.ids) <= length for window in windows)
        assert all(window.tokens[:3] == ["[CLS]", "\u2581what", "?"] for window in windows)


class _Window:
    """A stand-in encoding with chosen words, word characters and token offsets."""

    def __init__(self, words: list[int | None], chars: dict[int, tuple[int, int]]) -> None:
        self.words, self.chars = words, chars
        self.offsets = [(0, 2), (2, 5), (6, 9)]

    def token_to_word(self, token: int) -> int | None:
        return self.words[token]

    def word_to_chars(self, word: int, sequence: int) -> tuple[int, int] | None:
        return self.chars.get(word)


@pytest.mark.parametrize(
    ("words", "chars", "span"),
    [
        ([0, 0, 1], {0: (0, 5), 1: (6, 9)}, (0, 5)),
        ([0, None, 1], {0: (0, 5), 1: (6, 9)}, (2, 5)),
        ([0, 0, 1], {}, (2, 5)),
    ],
)
def test_a_span_covers_its_words_or_else_its_tokens(
    words: list[int | None], chars: dict[int, tuple[int, int]], span: tuple[int, int]
) -> None:
    window: Any = _Window(words, chars)
    assert extraction._characters(window, 1, 1) == span


class TestAnswerExtractor:
    def test_the_best_span_widens_to_its_words(self, spiece_model: Path) -> None:
        extract, _ = _extractor(spiece_model, {"\u258142": 8.0}, {"\u258142": 8.0})
        assert extract("how many?", "the answer is 42.") == " 42."

    def test_spans_with_the_same_text_add_up(self, spiece_model: Path) -> None:
        logits = {"\u258142": 4.2, ".": 4.2, "\u2581dog": 4.6, "\u2581a": -10.0}
        extract, _ = _extractor(spiece_model, logits, logits)
        assert extract("how many?", "42. " + FAR + "dog") == "42."

    def test_texts_that_differ_only_in_case_count_as_one(self, spiece_model: Path) -> None:
        logits = {"\u2581Cat": 4.2, "\u2581cat": 4.2, "\u2581dog": 4.4, "\u2581a": -10.0}
        extract, _ = _extractor(spiece_model, logits, logits)
        assert extract("what?", "a Cat " + FAR + "dog " + FAR + "cat") == " Cat"

    def test_an_answer_only_an_overflowing_window_holds_is_found(self, spiece_model: Path) -> None:
        extract, model = _extractor(
            spiece_model, {"\u258142": 9.0}, {"\u258142": 9.0}, max_length=WINDOW_STRIDE + 40
        )
        tokenizer = bigbird_tokenizer(spiece_model, WINDOW_STRIDE + 40)
        context = "42 " + " ".join(["a"] * 300)
        encoding = tokenizer.encode("what?", context)
        assert "\u258142" not in encoding.tokens
        assert any("\u258142" in window.tokens for window in encoding.overflowing)
        assert extract("what?", context) == "42"
        assert model.calls == 1 + len(encoding.overflowing)

    def test_a_pair_that_cannot_be_cut_into_windows_is_refused(self, spiece_model: Path) -> None:
        length = WINDOW_STRIDE + 40
        extract, model = _extractor(spiece_model, {}, {}, max_length=length)
        limit = length - 3 - WINDOW_STRIDE
        long_context = " ".join(["a"] * 300)
        assert extract(" ".join(["a"] * (limit - 1)), long_context) == " a"
        assert extract(" ".join(["a"] * (limit - 2)) + " </s>", long_context) == " a"
        assert extract(" ".join(["a"] * limit), "a cat") == "a"
        calls = model.calls
        with pytest.raises(
            ValueError, match=f"the question is {limit} tokens.*at most {limit - 1}"
        ):
            extract(" ".join(["a"] * limit), long_context)
        with pytest.raises(ValueError, match=f"the question is {length + 50} tokens"):
            extract(" ".join(["a"] * (length + 50)), "a cat")
        assert model.calls == calls

    def test_checking_a_pair_leaves_the_windows_as_they_were(self, spiece_model: Path) -> None:
        length = WINDOW_STRIDE + 40
        tokenizer = bigbird_tokenizer(spiece_model, length)
        truncation = tokenizer.truncation
        extraction._check_pair(tokenizer, "what?", " ".join(["a"] * 300))
        assert tokenizer.truncation == truncation
        with pytest.raises(ValueError):
            extraction._check_pair(tokenizer, " ".join(["a"] * length), " ".join(["a"] * 300))
        assert tokenizer.truncation == truncation

    @pytest.mark.parametrize("context", [" ", "\n", " \t\n "])
    def test_a_context_without_a_token_gives_no_answer(
        self, spiece_model: Path, context: str
    ) -> None:
        extract, _ = _extractor(spiece_model, {}, {})
        assert extract("what?", context) == ""

    @pytest.mark.parametrize(("question", "context"), [("", "a cat"), ("what?", "")])
    def test_an_empty_question_or_context_is_refused(
        self, spiece_model: Path, question: str, context: str
    ) -> None:
        extract, _ = _extractor(spiece_model, {}, {})
        with pytest.raises(ValueError, match="non-empty"):
            extract(question, context)

    def test_the_model_is_loaded_on_the_first_call_and_kept(self, spiece_model: Path) -> None:
        tokenizer = bigbird_tokenizer(spiece_model, TOKENS)
        span_model = SpanModel(tokenizer, _Logits({}, {}), _id(tokenizer, "[CLS]"))
        loads: list[torch.device] = []

        def load(device: torch.device) -> SpanModel:
            loads.append(device)
            return span_model

        extract = AnswerExtractor("cpu", load=load)
        assert loads == []
        extract("what?", "a cat")
        extract("how many?", "a dog")
        assert loads == [torch.device("cpu")]

    def test_a_load_that_fails_raises_and_is_tried_again(self, spiece_model: Path) -> None:
        tokenizer = bigbird_tokenizer(spiece_model, TOKENS)
        span_model = SpanModel(tokenizer, _Logits({}, {}), _id(tokenizer, "[CLS]"))
        attempts: list[int] = []

        def load(device: torch.device) -> SpanModel:
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("the extractor is not on this node")
            return span_model

        extract = AnswerExtractor("cpu", load)
        with pytest.raises(OSError, match="not on this node"):
            extract("what?", "a cat")
        assert extract("what?", "a cat") == "a"
        assert len(attempts) == 2

    def test_a_device_torch_cannot_read_is_refused(self) -> None:
        with pytest.raises(ValueError, match="extractor_device 'gpu0' is not a torch device"):
            AnswerExtractor("gpu0", _load_nothing)

    def test_a_cuda_device_needs_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError, match="needs CUDA"):
            check_extractor_device("cuda")
        assert check_extractor_device("cpu") == torch.device("cpu")

    @pytest.mark.parametrize("device", ["mps", "meta", "xpu:0"])
    def test_only_the_cpu_and_cuda_are_extractor_devices(self, device: str) -> None:
        with pytest.raises(ValueError, match="must be a cpu or cuda device"):
            check_extractor_device(device)

    def test_a_cuda_index_must_be_a_device_the_process_has(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        assert check_extractor_device("cuda:1") == torch.device("cuda", 1)
        assert check_extractor_device("cuda") == torch.device("cuda")
        with pytest.raises(ValueError, match="it sees 2"):
            check_extractor_device("cuda:2")


@pytest.fixture
def bigbird_snapshot(tmp_path: Path, spiece_model: Path) -> Path:
    """A folder laid out like the extractor's snapshot: a tiny random BigBird and ``PIECES``."""
    from transformers import BigBirdConfig, BigBirdForQuestionAnswering

    tokenizer = bigbird_tokenizer(spiece_model, TOKENS)
    config = BigBirdConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=TOKENS,
        block_size=4,
        num_random_blocks=1,
        sep_token_id=_id(tokenizer, "[SEP]"),
        pad_token_id=_id(tokenizer, "<pad>"),
    )
    torch.manual_seed(0)
    folder = tmp_path / "snapshot"
    BigBirdForQuestionAnswering(config).save_pretrained(folder)
    (folder / "spiece.model").write_bytes(spiece_model.read_bytes())
    return folder


class TestBigBird:
    def test_the_pinned_files_are_fetched(
        self, bigbird_snapshot: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import huggingface_hub

        requests: list[tuple[Sequence[Any], dict[str, Any]]] = []

        def download(*args: Any, **kwargs: Any) -> str:
            requests.append((args, kwargs))
            return str(bigbird_snapshot)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
        assert fetch_bigbird() == bigbird_snapshot
        assert requests == [
            (
                (EXTRACTOR_MODEL,),
                {"revision": EXTRACTOR_REVISION, "allow_patterns": list(EXTRACTOR_FILES)},
            )
        ]

    def test_the_model_loads_from_its_folder_attending_fully(self, bigbird_snapshot: Path) -> None:
        from transformers.utils import logging as transformers_logging

        verbosity = transformers_logging.get_verbosity()
        span_model = load_bigbird(torch.device("cpu"), bigbird_snapshot)
        assert transformers_logging.get_verbosity() == verbosity
        model: Any = span_model.model
        assert model.bert.attention_type == "original_full" and not model.training
        assert span_model.cls_id == span_model.tokenizer.token_to_id("[CLS]")

    def test_the_extractor_fetches_when_made_and_loads_on_its_first_call(
        self, bigbird_snapshot: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetched: list[Path] = []
        loaded: list[torch.device] = []

        def fetch() -> Path:
            fetched.append(bigbird_snapshot)
            return bigbird_snapshot

        def load(device: torch.device, folder: Path) -> SpanModel:
            loaded.append(device)
            return load_bigbird(device, folder)

        monkeypatch.setattr(extraction, "fetch_bigbird", fetch)
        monkeypatch.setattr(extraction, "load_bigbird", load)
        extract = bigbird_extractor("cpu")
        assert (fetched, loaded) == ([bigbird_snapshot], [])
        context = "the answer is 42. a cat"
        answer = extract("how many?", context)
        assert answer and answer in context
        extract("what?", "a dog")
        assert (fetched, loaded) == ([bigbird_snapshot], [torch.device("cpu")])

    def test_a_device_is_refused_before_anything_is_fetched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetched: list[str] = []
        monkeypatch.setattr(extraction, "fetch_bigbird", lambda: fetched.append("fetched"))
        with pytest.raises(ValueError, match="extractor_device 'gpu0'"):
            bigbird_extractor("gpu0")
        assert fetched == []
