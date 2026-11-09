"""Unit tests for HuggingFaceVLMDataset and VLMCollator."""

from __future__ import annotations

import itertools
import logging
import random
import re

import pytest
import torch
from PIL import Image

from kempnerforge.data import vlm_dataset
from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    HuggingFaceVLMDataset,
    PackedVLMCollator,
    VLMCollator,
    _tokenize_and_mask,
    check_prompt_fits,
    frames_to_clip_tensor,
    pack_vlm_samples,
    pil_to_tensor,
)


def _make_image(size: int = 64, color: int = 128) -> Image.Image:
    return Image.new("RGB", (size, size), color=(color, color, color))


def _dataset_over(
    rows,
    *,
    image_size: int = 16,
    max_text_len: int = 8,
    allow_text_only: bool = False,
    max_images: int = 1,
    tokenizer_path: str = "gpt2",
) -> HuggingFaceVLMDataset:
    """A ``HuggingFaceVLMDataset`` over in-memory ``rows``, skipping ``__init__``'s loading."""
    from transformers import AutoTokenizer

    ds = HuggingFaceVLMDataset.__new__(HuggingFaceVLMDataset)
    ds._ds = rows
    ds._image_field = "image"
    ds._text_field = "caption"
    ds._prompt_field = None
    ds._image_size = image_size
    ds._image_mean = DEFAULT_IMAGE_MEAN
    ds._image_std = DEFAULT_IMAGE_STD
    ds._max_text_len = max_text_len
    ds._allow_text_only = allow_text_only
    ds._max_images = max_images
    ds._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return ds


class TestPILToTensor:
    def test_shape_and_dtype(self):
        img = _make_image(48)
        t = pil_to_tensor(img, 224, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        assert t.shape == (3, 224, 224)
        assert t.dtype == torch.float32

    def test_non_square_resize(self):
        img = Image.new("RGB", (100, 50), color=(0, 128, 255))
        t = pil_to_tensor(img, 64, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        assert t.shape == (3, 64, 64)

    def test_grayscale_promoted_to_rgb(self):
        img = Image.new("L", (32, 32), color=128)
        t = pil_to_tensor(img, 32, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        assert t.shape == (3, 32, 32)

    def test_normalization(self):
        """Input with pixels at 127.5 (0.5 after /255) should normalize
        to ~0 under mean=0.5, std=0.5."""
        img = Image.new("RGB", (16, 16), color=(128, 128, 128))
        t = pil_to_tensor(img, 16, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        assert abs(float(t.mean())) < 0.02

    def test_non_pil_raises(self):
        with pytest.raises(TypeError, match="Expected a PIL.Image"):
            pil_to_tensor(torch.randn(3, 16, 16), 16, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)


class _MockTokenizer:
    """Deterministic char-level tokenizer for tests (no HF download).

    Tokens 1-26 map to a-z, 27 maps to space, 28 maps to period, 0 is the
    pad id. Unknown chars become 0. This is enough to verify ``labels``
    masking semantics without pulling a real HF checkpoint.
    """

    pad_token_id = 0
    eos_token_id = 28

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        ids = []
        for ch in text.lower():
            if ch == " ":
                ids.append(27)
            elif ch == ".":
                ids.append(28)
            elif "a" <= ch <= "z":
                ids.append(1 + ord(ch) - ord("a"))
            else:
                ids.append(0)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class TestTokenizeAndMask:
    def test_shape_and_pad(self):
        tok = _MockTokenizer()
        ids, labels, _ = _tokenize_and_mask(tok, "abc", max_text_len=8, prompt=None)
        assert ids.shape == (8,)
        assert labels.shape == (8,)
        assert ids[:4].tolist() == [1, 2, 3, 28]
        assert ids[4].item() == 0
        assert labels[:4].tolist() == [2, 3, 28, -100]
        assert labels[4].item() == -100

    def test_prompt_masks_labels(self):
        tok = _MockTokenizer()
        ids, labels, _ = _tokenize_and_mask(tok, text="xyz", max_text_len=8, prompt="ab")
        assert ids[:6].tolist() == [1, 2, 24, 25, 26, 28]
        assert labels[0].item() == -100
        assert labels[1:5].tolist() == [24, 25, 26, 28]
        assert labels[5].item() == -100

    def test_truncation(self):
        tok = _MockTokenizer()
        ids, labels, _ = _tokenize_and_mask(tok, text="abcdefghij", max_text_len=4, prompt=None)
        assert ids.tolist() == [1, 2, 3, 28]
        assert labels.tolist() == [2, 3, 28, -100]

    def test_single_token_caption_supervises_eos(self):
        """A one-token caption still yields a supervised target (the appended
        EOS), so it is never an all--100 row."""
        tok = _MockTokenizer()
        ids, labels, _ = _tokenize_and_mask(tok, "a", max_text_len=8, prompt=None)
        assert ids[:2].tolist() == [1, 28]
        assert labels[0].item() == 28
        assert (labels[1:] == -100).all()

    def test_empty_caption_all_masked(self):
        """An empty caption has no target (a lone EOS has no successor)."""
        tok = _MockTokenizer()
        _, labels, _ = _tokenize_and_mask(tok, "", max_text_len=8, prompt=None)
        assert (labels == -100).all()

    def test_no_eos_tokenizer_appends_nothing(self):
        """A tokenizer with no EOS id appends nothing and keeps up to
        max_text_len caption tokens (the no-EOS branch of the append)."""
        tok = _MockTokenizer()
        tok.eos_token_id = None
        ids, labels, _ = _tokenize_and_mask(tok, "abcde", max_text_len=4, prompt=None)
        assert ids.tolist() == [1, 2, 3, 4]
        assert labels.tolist() == [2, 3, 4, -100]

    def test_prompt_mask_with_bpe_tokenizer(self):
        """Regression: BPE (gpt2) and SentencePiece tokenizers are not
        prefix-preserving. ``tokenize(prompt) + tokenize(text)`` differs
        from ``tokenize(prompt + text)`` at the boundary (tokens can
        merge or split). The implementation must tokenize the prompt and
        text independently and concatenate the id lists, so the
        prompt/target boundary is exact and only prompt-predicting
        positions are masked. Verified end-to-end on the gpt2 tokenizer.
        """
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        prompt = "Q: "
        text = "hello"

        prompt_ids = list(tok(prompt, add_special_tokens=False)["input_ids"])
        text_ids = list(tok(text, add_special_tokens=False)["input_ids"])
        expected_ids = prompt_ids + text_ids
        n = len(expected_ids)

        ids, labels, _ = _tokenize_and_mask(tok, text, max_text_len=32, prompt=prompt)

        assert ids[:n].tolist() == expected_ids
        assert (labels[: len(prompt_ids) - 1] == -100).all()
        assert labels[len(prompt_ids) - 1 : n - 1].tolist() == text_ids

    def test_prompt_mask_under_bpe_merge_keeps_target_intact(self):
        """Concrete BPE case: ``tokenize("foo") + tokenize("bar")`` can
        differ from ``tokenize("foobar")`` when gpt2 BPE merges the joined
        string into a shorter sequence. Our implementation uses the split
        form, so the target portion of labels is exactly
        ``tokenize(text)`` regardless of whether merging would happen."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        prompt = "foo"
        text = "bar"

        prompt_ids = list(tok(prompt, add_special_tokens=False)["input_ids"])
        text_ids = list(tok(text, add_special_tokens=False)["input_ids"])

        ids, labels, _ = _tokenize_and_mask(tok, text, max_text_len=16, prompt=prompt)

        split_total = len(prompt_ids) + len(text_ids)
        assert ids[:split_total].tolist() == prompt_ids + text_ids
        assert (
            labels[len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(text_ids)].tolist() == text_ids
        )


def _expected_next_token_labels(ids: list[int], prompt_len: int, max_text_len: int) -> list[int]:
    """Index-by-index restatement of the label contract, derived from the loss.

    ``cross_entropy_loss`` does not shift, and the VLM arches hand it logits
    over the text positions, so position ``t`` scores a prediction of token
    ``t+1``. Written as an explicit loop (not tensor slices) so it cannot
    share an off-by-one with the implementation it checks.
    """
    labels = [-100] * max_text_len
    for t in range(max_text_len):
        nxt = t + 1
        if nxt >= len(ids):
            continue
        if nxt <= prompt_len - 1:
            continue
        labels[t] = ids[nxt]
    return labels


class TestNextTokenLabelContract:
    """Cross-checks ``_tokenize_and_mask`` against an independently derived
    statement of the contract, instead of against fitted literals."""

    @pytest.mark.parametrize(
        ("text", "prompt", "max_text_len"),
        [
            ("abc", None, 8),
            ("abc", "ab", 8),
            ("abcdefg", None, 8),
            ("abcdefgh", None, 8),
            ("abcdefghijkl", None, 8),
            ("abcdefgh", "qrs", 8),
            ("a", None, 8),
            ("", None, 8),
            ("abc", "", 8),
            ("ab", "a", 8),
            ("a", None, 2),
            ("xyz", "aaaaaa", 8),
            ("xyz", "aaaaaaa", 8),
            ("xyz", "aaaaaaaa", 8),
            ("xyz", "aaaaaaaaaaaa", 8),
        ],
    )
    @pytest.mark.parametrize("with_eos", [True, False])
    def test_matches_derived_contract(self, text, prompt, max_text_len, with_eos):
        tok = _MockTokenizer()
        if not with_eos:
            tok.eos_token_id = None
        ids, labels, _ = _tokenize_and_mask(tok, text, max_text_len, prompt)

        real = list(tok(text)["input_ids"])
        prompt_ids = list(tok(prompt)["input_ids"]) if prompt is not None else []
        budget = max_text_len - 1 if with_eos else max_text_len
        kept = (prompt_ids + real)[:budget]
        n = len(kept) + (1 if with_eos else 0)
        assert ids.shape == (max_text_len,)
        assert ids[: len(kept)].tolist() == kept
        if with_eos:
            assert ids[len(kept)].item() == tok.eos_token_id
        assert (ids[n:] == 0).all()

        prompt_len = min(len(prompt_ids), n)
        expected = _expected_next_token_labels(ids[:n].tolist(), prompt_len, max_text_len)
        caption_slots = max_text_len - len(prompt_ids) - (1 if with_eos else 0)
        if prompt is not None and caption_slots < 1:
            expected = [-100] * max_text_len
        assert labels.tolist() == expected

    def test_prompt_mask_boundary_both_directions(self):
        """The two indices either side of the prompt boundary, pinned.

        With shifted labels the prompt occupies ``ids[0 .. prompt_len-1]``, so
        ``labels[prompt_len-1]`` predicts the *first caption token* and must stay
        supervised, while ``labels[prompt_len-2]`` predicts the last prompt token
        and must be masked. Widening the mask by one silently drops the first
        caption token; narrowing it leaks a prompt token into the loss.
        """
        tok = _MockTokenizer()
        prompt, text = "abcd", "xyz"
        prompt_len = len(tok(prompt)["input_ids"])
        ids, labels, _ = _tokenize_and_mask(tok, text, max_text_len=16, prompt=prompt)

        first_caption_id = tok(text)["input_ids"][0]
        assert ids[prompt_len].item() == first_caption_id
        assert labels[prompt_len - 1].item() == first_caption_id
        assert labels[prompt_len - 2].item() == -100
        assert (labels[: prompt_len - 1] == -100).all()

    def test_eos_is_a_supervised_target(self):
        """The last caption token predicts EOS, so captions learn to stop."""
        tok = _MockTokenizer()
        _, labels, _ = _tokenize_and_mask(tok, "hello", max_text_len=16, prompt="ab")
        supervised = labels[labels != -100].tolist()
        assert supervised[-1] == tok.eos_token_id
        assert supervised.count(tok.eos_token_id) == 1

    def test_eos_survives_an_overlong_caption(self):
        """Truncation reserves the EOS slot, so a caption longer than the budget
        still ends in a supervised stop target rather than a dropped one."""
        tok = _MockTokenizer()
        ids, labels, _ = _tokenize_and_mask(tok, "abcdefghij", max_text_len=5, prompt=None)
        assert ids[-1].item() == tok.eos_token_id
        assert labels[-2].item() == tok.eos_token_id
        assert labels[-1].item() == -100

    def test_prompt_filling_the_budget_masks_every_label(self):
        """A prompt that leaves no room for a caption token neither raises nor
        produces a negative-index slice: the sample keeps its static shapes and
        every label is masked, so it adds no loss instead of training on EOS alone."""
        tok = _MockTokenizer()
        for prompt_chars in (7, 8, 12):
            ids, labels, _ = _tokenize_and_mask(
                tok, "xyz", max_text_len=8, prompt="a" * prompt_chars
            )
            assert ids.shape == labels.shape == (8,)
            assert (labels == -100).all()

    def test_labels_are_identical_across_ranks(self, monkeypatch):
        """Every DP rank must build byte-identical labels for a given index.

        Label construction is pure in ``(tokenizer, text, max_text_len, prompt)``,
        so a rank-dependent or RNG-dependent path here would desync the loss
        across ranks. Emulates two ranks in-process: distinct rank environment
        and distinct global RNG state, separate tokenizer instances, same index.
        """
        from transformers import AutoTokenizer

        captions = ["a red bus", "two dogs running", "a bowl of soup"]
        index = 1

        def build(rank: int) -> tuple[bytes, bytes]:
            monkeypatch.setenv("RANK", str(rank))
            monkeypatch.setenv("LOCAL_RANK", str(rank))
            monkeypatch.setenv("WORLD_SIZE", "2")
            random.seed(rank)
            torch.manual_seed(rank)
            tok = AutoTokenizer.from_pretrained("gpt2")
            ids, labels, _ = _tokenize_and_mask(
                tok, captions[index], max_text_len=16, prompt="Describe: "
            )
            return ids.numpy().tobytes(), labels.numpy().tobytes()

        ids0, labels0 = build(0)
        ids1, labels1 = build(1)
        assert ids0 == ids1
        assert labels0 == labels1


class _CountingTokenizer(_MockTokenizer):
    """``_MockTokenizer`` that records every text it tokenizes."""

    def __init__(self, eos_token_id: int | None = 28) -> None:
        self.eos_token_id = eos_token_id
        self.texts: list[str] = []

    def __call__(self, text: str, add_special_tokens: bool = False):
        self.texts.append(text)
        return super().__call__(text, add_special_tokens)


class _RecordingHandler(logging.Handler):
    """Keeps every record that passes its filters."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _prompted_dataset(
    tmp_path,
    monkeypatch,
    prompts: list,
    tokenizer: _MockTokenizer | None = None,
    prompt_dtype: str | None = None,
) -> HuggingFaceVLMDataset:
    """A real ``HuggingFaceVLMDataset`` whose rows pair caption ``"xyz"`` with ``prompts``.

    The prompt column is cast to ``Value(prompt_dtype)`` when ``prompt_dtype`` is given.
    """
    from datasets import Dataset, Value

    rows = Dataset.from_dict(
        {
            "image": [_make_image(32, 50 + i) for i in range(len(prompts))],
            "caption": ["xyz"] * len(prompts),
            "prompt": prompts,
        }
    )
    if prompt_dtype is not None:
        rows = rows.cast_column("prompt", Value(prompt_dtype))
    rows.save_to_disk(str(tmp_path / "prompted"))
    tok = tokenizer if tokenizer is not None else _MockTokenizer()
    monkeypatch.setattr(vlm_dataset, "build_tokenizer", lambda path: tok)
    return HuggingFaceVLMDataset(
        dataset_name=str(tmp_path / "prompted"),
        split="train",
        image_field="image",
        text_field="caption",
        tokenizer_path="unused",
        max_text_len=8,
        prompt_field="prompt",
        image_size=16,
    )


class TestPromptRoom:
    """A prompt must leave room in ``max_text_len`` for one caption token, which the
    last prompt token predicts, plus the EOS that caption token predicts when the
    tokenizer defines one. With ``max_text_len = 8`` the longest prompt that fits is
    6 tokens with an EOS and 7 without."""

    @pytest.mark.parametrize(
        ("eos_token_id", "prompt_tokens", "supervised"),
        [(28, 6, [24, 28]), (None, 7, [24])],
    )
    def test_longest_fitting_prompt_supervises_a_caption_token(
        self, eos_token_id, prompt_tokens, supervised
    ):
        tok = _MockTokenizer()
        tok.eos_token_id = eos_token_id
        prompt = "a" * prompt_tokens
        _, labels, _ = _tokenize_and_mask(tok, "xyz", max_text_len=8, prompt=prompt)
        assert labels[labels != -100].tolist() == supervised
        check_prompt_fits(tok, prompt, max_text_len=8)

    @pytest.mark.parametrize(("eos_token_id", "prompt_tokens"), [(28, 7), (None, 8)])
    def test_one_more_prompt_token_masks_the_sample_and_fails_the_check(
        self, eos_token_id, prompt_tokens
    ):
        tok = _MockTokenizer()
        tok.eos_token_id = eos_token_id
        prompt = "a" * prompt_tokens
        ids, labels, _ = _tokenize_and_mask(tok, "xyz", max_text_len=8, prompt=prompt)
        assert ids.shape == (8,)
        assert (labels == -100).all()
        with pytest.raises(
            ValueError,
            match=rf"prompt is {prompt_tokens} tokens.*needs max_text_len >= 9.*vlm\.max_text_len",
        ):
            check_prompt_fits(tok, prompt, max_text_len=8)

    def test_truncating_a_long_caption_is_not_masked(self, monkeypatch, caplog):
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        tok = _MockTokenizer()
        with caplog.at_level(logging.WARNING, logger="kempnerforge.data.vlm_dataset"):
            _, labels, _ = _tokenize_and_mask(tok, "xyzxyzxyz", max_text_len=8, prompt="ab")
        assert labels[labels != -100].tolist() == [24, 25, 26, 24, 25, 28]
        assert caplog.records == []

    def test_masked_samples_are_logged_at_powers_of_two(self, monkeypatch, caplog):
        monkeypatch.setattr(vlm_dataset, "_samples_masked_for_prompt", itertools.count(1))
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        tok = _MockTokenizer()
        with caplog.at_level(logging.WARNING, logger="kempnerforge.data.vlm_dataset"):
            for _ in range(5):
                _tokenize_and_mask(tok, "xyz", max_text_len=8, prompt="a" * 7)
        messages = [r.getMessage() for r in caplog.records]
        assert [re.search(r"\((\d+) such samples", m).group(1) for m in messages] == [
            "1",
            "2",
            "4",
        ]
        assert all("needs max_text_len >= 9" in m and "max_text_len=8" in m for m in messages)

    def test_masked_sample_warning_passes_the_rank_zero_filter_on_every_rank(self, monkeypatch):
        """KF's log handler drops records from every rank but 0, and a masked sample is
        seen only by the rank that reads it, so its warning is marked for every rank."""
        from kempnerforge.metrics.logger import _RankFilter

        monkeypatch.setattr(vlm_dataset, "_samples_masked_for_prompt", itertools.count(1))
        monkeypatch.setenv("RANK", "3")
        handler = _RecordingHandler()
        handler.addFilter(_RankFilter(rank=0))
        dataset_logger = logging.getLogger("kempnerforge.data.vlm_dataset")
        monkeypatch.setattr(dataset_logger, "handlers", [handler])
        monkeypatch.setattr(dataset_logger, "propagate", False)
        _tokenize_and_mask(_MockTokenizer(), "xyz", max_text_len=8, prompt="a" * 7)
        dataset_logger.warning("an unmarked record")
        assert [r.getMessage() for r in handler.records] == [
            "A prompt needs max_text_len >= 9 to leave room for a caption token, but "
            "max_text_len=8: masked every label of that sample so it adds no loss "
            "(1 such samples so far in this process, logged at powers of two)"
        ]

    def test_per_sample_prompts_are_masked_row_by_row(self, tmp_path, monkeypatch):
        """A column with a prompt that fits is admitted, and a row whose prompt leaves
        no room is masked when it is read; the rows around it are unaffected."""
        ds = _prompted_dataset(tmp_path, monkeypatch, ["ab", "a" * 7, "a" * 6])
        supervised = [ds[i]["labels"][ds[i]["labels"] != -100].tolist() for i in range(3)]
        assert supervised == [[24, 25, 26, 28], [], [24, 28]]
        assert ds[1]["input_ids"].shape == (8,)


class TestPromptColumnRoom:
    """Building the dataset rejects a prompt column in which no row leaves room for a
    caption token, since every sample would add no loss. It reads the column only up
    to the first row that trains on something, so the bound is the per-sample one
    and the rows after it are left to the per-sample masking."""

    @pytest.mark.parametrize(("eos_token_id", "prompt_tokens"), [(28, 7), (None, 8)])
    def test_a_column_with_no_prompt_that_fits_is_rejected(
        self, tmp_path, monkeypatch, eos_token_id, prompt_tokens
    ):
        prompts = ["a" * (prompt_tokens + 3), "a" * prompt_tokens, "a" * (prompt_tokens + 1)]
        tok = _CountingTokenizer(eos_token_id)
        with pytest.raises(
            ValueError,
            match=(
                rf"No prompt in column 'prompt' leaves room in max_text_len=8 .* the shortest "
                rf"is {prompt_tokens} tokens \(needs max_text_len >= 9\); raise vlm\.max_text_len"
            ),
        ):
            _prompted_dataset(tmp_path, monkeypatch, prompts, tok)
        assert tok.texts == prompts

    @pytest.mark.parametrize(("eos_token_id", "prompt_tokens"), [(28, 6), (None, 7)])
    def test_the_last_row_fitting_admits_the_column(
        self, tmp_path, monkeypatch, eos_token_id, prompt_tokens
    ):
        overlong = "a" * (prompt_tokens + 1)
        tok = _CountingTokenizer(eos_token_id)
        prompts = [overlong, overlong, "a" * prompt_tokens]
        ds = _prompted_dataset(tmp_path, monkeypatch, prompts, tok)
        assert tok.texts == prompts
        assert [bool((ds[i]["labels"] != -100).any()) for i in range(3)] == [False, False, True]

    def test_a_row_without_a_prompt_admits_the_column(self, tmp_path, monkeypatch):
        ds = _prompted_dataset(tmp_path, monkeypatch, ["a" * 7, None, "a" * 9])
        assert [bool((ds[i]["labels"] != -100).any()) for i in range(3)] == [False, True, False]

    def test_the_column_is_read_only_up_to_the_first_row_that_fits(self, tmp_path, monkeypatch):
        tok = _CountingTokenizer()
        _prompted_dataset(tmp_path, monkeypatch, ["a" * 7, "ab", "a" * 9, "a" * 12], tok)
        assert tok.texts == ["a" * 7, "ab"]

    def test_an_empty_dataset_is_not_rejected(self, tmp_path, monkeypatch):
        assert len(_prompted_dataset(tmp_path, monkeypatch, [])) == 0


class TestPromptColumnType:
    """Building the dataset rejects a prompt column whose type admits a value that is
    neither a string nor missing, whatever its rows hold, so every rank fails while
    building it instead of the rank that reads such a value failing mid-run while its
    peers wait in a collective."""

    @pytest.mark.parametrize(
        "prompts",
        [[None, ["ab"], None, None], [["ab"]], [None, 3], [None, b"ab"], [None, {"text": "ab"}]],
        ids=["list-after-a-missing-row", "list", "int", "bytes", "struct"],
    )
    def test_a_column_of_another_type_is_rejected_before_any_row_is_read(
        self, tmp_path, monkeypatch, prompts
    ):
        from datasets import Dataset

        feature = Dataset.from_dict({"prompt": prompts}).features["prompt"]
        tok = _CountingTokenizer()
        with pytest.raises(
            TypeError,
            match=rf"'prompt' must be a string column.*'null' when no row has a prompt.*got "
            rf"{re.escape(str(feature))}$",
        ):
            _prompted_dataset(tmp_path, monkeypatch, prompts, tok)
        assert tok.texts == []

    @pytest.mark.parametrize(
        ("prompts", "dtype"),
        [(["ab", None], "string"), (["ab", None], "large_string"), ([None, None], "null")],
    )
    def test_a_string_column_or_one_with_no_prompt_is_admitted(
        self, tmp_path, monkeypatch, prompts, dtype
    ):
        ds = _prompted_dataset(tmp_path, monkeypatch, prompts, prompt_dtype=dtype)
        assert ds._ds.features["prompt"].dtype == dtype
        for i, prompt in enumerate(prompts):
            _, labels, _ = _tokenize_and_mask(
                _MockTokenizer(), "xyz", max_text_len=8, prompt=prompt
            )
            assert torch.equal(ds[i]["labels"], labels)


class TestVLMCollator:
    def _sample(self, text_len: int, max_text_len: int = 20) -> dict[str, torch.Tensor]:
        ids = torch.zeros(max_text_len, dtype=torch.long)
        ids[:text_len] = torch.arange(1, text_len + 1, dtype=torch.long)
        labels = torch.full((max_text_len,), -100, dtype=torch.long)
        labels[:text_len] = torch.arange(1, text_len + 1, dtype=torch.long)
        return {
            "pixel_values": torch.randn(3, 32, 32),
            "input_ids": ids,
            "labels": labels,
        }

    def test_a_text_only_sample_gets_a_blank_image_and_a_frame_mask(self):
        collator = VLMCollator(pad_id=0, max_text_len=20, allow_text_only=True)
        text_only = self._sample(3)
        text_only["pixel_values"] = torch.zeros(0, 3, 32, 32)
        batch = collator([self._sample(5), text_only])
        assert batch["pixel_values"].shape == (2, 3, 32, 32)
        assert torch.equal(batch["pixel_values"][1], torch.zeros(3, 32, 32))
        assert batch["frame_mask"].tolist() == [[True], [False]]
        assert batch["input_ids"][1, :3].tolist() == [1, 2, 3]

    def test_the_frame_mask_is_emitted_on_every_batch_when_text_only_rows_are_allowed(self):
        """So the attention mask's shape, and the kernel it selects, never flip between batches."""
        collator = VLMCollator(pad_id=0, max_text_len=20, allow_text_only=True)
        batch = collator([self._sample(5), self._sample(6)])
        assert batch["frame_mask"].tolist() == [[True], [True]]

    def test_without_the_flag_there_is_no_frame_mask_and_a_text_only_sample_raises(self):
        collator = VLMCollator(pad_id=0, max_text_len=20)
        assert "frame_mask" not in collator([self._sample(5), self._sample(6)])
        text_only = self._sample(3)
        text_only["pixel_values"] = torch.zeros(0, 3, 32, 32)
        with pytest.raises(ValueError, match="hf_allow_text_only"):
            collator([text_only])

    def test_views_stack_on_the_frame_axis_and_a_text_only_sample_matches_them(self):
        collator = VLMCollator(pad_id=0, max_text_len=20, allow_text_only=True)
        tiled, text_only = self._sample(3), self._sample(4)
        tiled["pixel_values"] = torch.randn(2, 3, 32, 32)
        text_only["pixel_values"] = torch.zeros(0, 2, 3, 32, 32)
        batch = collator([tiled, text_only])
        assert batch["pixel_values"].shape == (2, 2, 3, 32, 32)
        assert torch.count_nonzero(batch["pixel_values"][1]) == 0
        assert batch["frame_mask"].tolist() == [[True, True], [False, False]]
        only_text = collator([text_only, text_only])
        assert only_text["pixel_values"].shape == (2, 2, 3, 32, 32)
        assert only_text["frame_mask"].shape == (2, 2)

    def test_samples_with_different_image_shapes_are_rejected(self):
        collator = VLMCollator(pad_id=0, max_text_len=20)
        tiled = self._sample(3)
        tiled["pixel_values"] = torch.randn(2, 3, 32, 32)
        with pytest.raises(ValueError, match="share one image shape"):
            collator([self._sample(5), tiled])

    def test_fixed_length_not_batch_max(self):
        collator = VLMCollator(pad_id=0, max_text_len=20)
        batch = collator([self._sample(5), self._sample(10), self._sample(7)])
        assert batch["input_ids"].shape == (3, 20)
        assert batch["labels"].shape == (3, 20)

    def test_pad_positions_masked(self):
        collator = VLMCollator(pad_id=0, max_text_len=16)
        batch = collator([self._sample(5, max_text_len=16)])
        assert (batch["input_ids"][0, 5:] == 0).all()
        assert (batch["labels"][0, 5:] == -100).all()

    def test_image_positions_emitted(self):
        collator = VLMCollator(pad_id=0, max_text_len=8)
        batch = collator([self._sample(3, max_text_len=8) for _ in range(4)])
        assert "image_positions" in batch
        assert batch["image_positions"].shape == (4,)
        assert (batch["image_positions"] == 0).all()

    def test_pixel_values_stacked(self):
        collator = VLMCollator(pad_id=0, max_text_len=8)
        batch = collator([self._sample(3, max_text_len=8) for _ in range(4)])
        assert batch["pixel_values"].shape == (4, 3, 32, 32)

    def test_empty_batch_raises(self):
        collator = VLMCollator(pad_id=0, max_text_len=8)
        with pytest.raises(ValueError, match="empty batch"):
            collator([])

    def test_max_text_len_must_be_positive(self):
        with pytest.raises(ValueError, match="max_text_len must be positive"):
            VLMCollator(pad_id=0, max_text_len=0)


@pytest.fixture
def tiny_hf_dataset(tmp_path):
    """Build a synthetic HF dataset in memory using datasets.Dataset.from_dict.

    The dataset has 3 rows; each row has a PIL image and a short caption.
    Saved to disk under tmp_path and loaded back via load_dataset so the
    code path matches production.
    """
    from datasets import Dataset

    ds = Dataset.from_dict(
        {
            "image": [_make_image(32, i * 50) for i in range(1, 4)],
            "caption": ["a cat.", "a dog runs.", "hello world."],
        }
    )
    out = tmp_path / "tiny_ds"
    ds.save_to_disk(str(out))
    return str(out)


class TestHuggingFaceVLMDataset:
    def test_getitem_shapes(self, tiny_hf_dataset):
        from datasets import load_from_disk

        ds = load_from_disk(tiny_hf_dataset)
        wrapper = _dataset_over(ds, image_size=32, max_text_len=16)
        item = wrapper[0]
        assert item["pixel_values"].shape == (3, 32, 32)
        assert item["input_ids"].shape == (16,)
        assert item["labels"].shape == (16,)
        assert item["input_ids"].dtype == torch.long
        text_len = int(item["text_len"])
        assert 0 < text_len <= 16
        assert item["labels"][text_len - 1].item() == -100
        assert (item["labels"][: text_len - 1] == item["input_ids"][1:text_len]).all()

    def test_init_load_from_disk_single_split(self, tiny_hf_dataset):
        """Real ``__init__`` on a dataset saved by ``save_to_disk`` (single
        split). Exercises the ``is_local`` branch."""
        ds = HuggingFaceVLMDataset(
            dataset_name=tiny_hf_dataset,
            split="ignored_when_single_split",
            image_field="image",
            text_field="caption",
            tokenizer_path="gpt2",
            max_text_len=8,
        )
        assert len(ds) == 3
        item = ds[0]
        assert item["pixel_values"].shape == (3, 224, 224)
        assert item["input_ids"].shape == (8,)
        assert item["labels"].shape == (8,)

    def test_init_load_from_disk_dataset_dict(self, tmp_path):
        """``load_from_disk`` returns a DatasetDict when multiple splits are
        saved together; ``__init__`` selects by split name."""
        from datasets import Dataset, DatasetDict

        train = Dataset.from_dict({"image": [_make_image(32, 100)], "caption": ["a tree."]})
        val = Dataset.from_dict({"image": [_make_image(32, 200)], "caption": ["a river."]})
        dd = DatasetDict({"train": train, "validation": val})
        out = tmp_path / "split_ds"
        dd.save_to_disk(str(out))

        ds = HuggingFaceVLMDataset(
            dataset_name=str(out),
            split="validation",
            image_field="image",
            text_field="caption",
            tokenizer_path="gpt2",
            max_text_len=4,
        )
        assert len(ds) == 1
        assert "river" in ds._ds[0]["caption"]  # type: ignore[attr-defined]

    def test_init_via_load_dataset_branch(self, monkeypatch, tiny_hf_dataset):
        """When ``dataset_name`` is not a local directory, ``__init__`` falls
        through to ``load_dataset``. We mock ``load_dataset`` to hand back a
        prebuilt Dataset and assert the call routed through the correct
        branch."""
        from datasets import load_from_disk

        from kempnerforge.data import vlm_dataset as mod

        prebuilt = load_from_disk(tiny_hf_dataset)
        called: dict[str, object] = {}

        def fake_load_dataset(name, config=None, split=None):
            called["name"] = name
            called["config"] = config
            called["split"] = split
            return prebuilt

        monkeypatch.setattr(mod, "load_dataset", fake_load_dataset, raising=False)
        import datasets

        monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

        ds = HuggingFaceVLMDataset(
            dataset_name="some/hub-id",
            split="train",
            image_field="image",
            text_field="caption",
            tokenizer_path="gpt2",
            max_text_len=8,
            dataset_config="cfg",
        )
        assert called == {"name": "some/hub-id", "config": "cfg", "split": "train"}
        assert len(ds) == 3

    def test_init_rejects_streaming_dataset(self, monkeypatch):
        """Non-map-style outputs (e.g. IterableDataset) should raise."""
        import datasets
        from datasets import IterableDataset

        def gen():
            yield {"image": _make_image(32, 50), "caption": "hi."}

        iterable = IterableDataset.from_generator(gen)
        monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: iterable)

        with pytest.raises(TypeError, match="map-style"):
            HuggingFaceVLMDataset(
                dataset_name="not/a/real/path",
                split="train",
                image_field="image",
                text_field="caption",
                tokenizer_path="gpt2",
                max_text_len=4,
            )

    def test_getitem_rejects_non_string_text(self, tiny_hf_dataset):
        """Schema guard: text field must be a string at __getitem__ time."""
        del tiny_hf_dataset
        from datasets import Dataset

        mixed = Dataset.from_dict(
            {
                "image": [_make_image(32, 100)],
                "caption": [{"not": "a string"}],  # type: ignore[list-item]
            }
        )
        wrapper = _dataset_over(mixed, image_size=32, max_text_len=4)
        with pytest.raises(TypeError, match="must be str"):
            _ = wrapper[0]

    def test_getitem_rejects_a_non_string_prompt(self, tmp_path, monkeypatch):
        """A prompt value that is not a string is rejected, not dropped as no prompt."""
        ds = _prompted_dataset(tmp_path, monkeypatch, ["ab"])
        row = {"image": _make_image(32), "caption": "xyz", "prompt": ["ab"]}
        monkeypatch.setattr(ds, "_ds", [row])
        with pytest.raises(TypeError, match="'prompt' must be str or None, got list"):
            _ = ds[0]

    def test_missing_prompt_trains_as_an_unprompted_caption(self, tmp_path, monkeypatch):
        """``None`` is how a string column stores a missing value: that row has no prompt."""
        ds = _prompted_dataset(tmp_path, monkeypatch, ["ab", None])
        _, unprompted, _ = _tokenize_and_mask(_MockTokenizer(), "xyz", max_text_len=8, prompt=None)
        assert torch.equal(ds[1]["labels"], unprompted)

    def test_collator_wired_through_stateful_dataloader(self, tiny_hf_dataset):
        """VLMCollator reaches the batch when passed via collate_fn.

        Without explicit wiring, StatefulDataLoader used the default torch
        collation, which would miss the image_positions slot. This test
        pins the wiring so a regression surfaces at unit time.
        """
        from datasets import load_from_disk

        from kempnerforge.config.schema import DataConfig
        from kempnerforge.data.dataloader import StatefulDataLoader
        from kempnerforge.data.sampler import DistributedSampler
        from kempnerforge.data.vlm_dataset import VLMCollator

        ds = load_from_disk(tiny_hf_dataset)
        wrapper = _dataset_over(ds, image_size=16, max_text_len=12)
        collator = VLMCollator(pad_id=0, max_text_len=12)
        loader = StatefulDataLoader(
            wrapper,
            batch_size=2,
            sampler=DistributedSampler(wrapper, num_replicas=1, rank=0, shuffle=False),
            config=DataConfig(num_workers=0),
            collate_fn=collator,
        )
        batch = next(iter(loader))
        assert "image_positions" in batch
        assert batch["image_positions"].shape == (2,)
        assert batch["input_ids"].shape == (2, 12)
        assert batch["labels"].shape == (2, 12)
        assert batch["pixel_values"].shape == (2, 3, 16, 16)


def _mixed_image_dataset(tmp_path, images):
    from datasets import Dataset

    ds = Dataset.from_dict({"image": images, "caption": ["a b"] * len(images)})
    out = tmp_path / "mixed_ds"
    ds.save_to_disk(str(out))
    return str(out)


class TestTextOnlyRows:
    @pytest.mark.parametrize("missing", [None, []])
    def test_a_row_without_an_image_is_text_only_when_allowed(self, missing):
        ds = _dataset_over([{"image": missing, "caption": "a b c"}], allow_text_only=True)
        item = ds[0]
        assert item["pixel_values"].shape == (0, 3, 16, 16)
        assert item["input_ids"].shape == (8,)
        assert 0 < int(item["text_len"]) <= 8

    @pytest.mark.parametrize("wrap", [lambda image: image, lambda image: [image]])
    def test_a_row_with_an_image_is_unchanged(self, wrap):
        rows = [{"image": wrap(_make_image(8)), "caption": "a b"}]
        ds = _dataset_over(rows, allow_text_only=True)
        assert ds[0]["pixel_values"].shape == (3, 16, 16)

    def test_a_row_without_an_image_raises_by_default(self):
        ds = _dataset_over([{"image": None, "caption": "a b"}], allow_text_only=False)
        with pytest.raises(ValueError, match="hf_allow_text_only"):
            ds[0]

    def test_a_dataset_with_an_imageless_row_is_rejected_when_loaded(self, tmp_path):
        path = _mixed_image_dataset(tmp_path, [_make_image(8), None, _make_image(8), None])
        kwargs = dict(
            split="train", image_field="image", text_field="caption", tokenizer_path="gpt2"
        )
        with pytest.raises(ValueError, match="2 of 4 rows have no image.*hf_allow_text_only"):
            HuggingFaceVLMDataset(path, max_text_len=8, **kwargs)
        ds = HuggingFaceVLMDataset(path, max_text_len=8, allow_text_only=True, **kwargs)
        assert ds.has_text_only_rows
        assert ds[1]["pixel_values"].shape == (0, 3, 224, 224)
        assert ds[0]["pixel_values"].shape == (3, 224, 224)

    def test_a_dataset_whose_rows_all_have_images_reports_no_text_only_rows(self, tmp_path):
        path = _mixed_image_dataset(tmp_path, [[_make_image(8)], [_make_image(8)]])
        kwargs = dict(
            split="train", image_field="image", text_field="caption", tokenizer_path="gpt2"
        )
        ds = HuggingFaceVLMDataset(path, max_text_len=8, allow_text_only=True, **kwargs)
        assert not ds.has_text_only_rows

    def test_a_row_with_several_images_keeps_its_first_at_the_default_cap(
        self, tmp_path, monkeypatch, caplog
    ):
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        path = _mixed_image_dataset(tmp_path, [[_make_image(8)] * 2, [_make_image(8)]])
        kwargs = dict(
            split="train", image_field="image", text_field="caption", tokenizer_path="gpt2"
        )
        with caplog.at_level(logging.WARNING, logger="kempnerforge.data.vlm_dataset"):
            ds = HuggingFaceVLMDataset(path, max_text_len=8, allow_text_only=True, **kwargs)
        assert "1 of 2 rows hold more than data.hf_max_images=1" in caplog.text
        assert ds[0]["pixel_values"].shape == (3, 224, 224)

    def test_images_per_row_counts_lists_and_single_images(self):
        from datasets import Dataset, Features, Image, Sequence

        singles = Dataset.from_dict({"image": [_make_image(8), None, _make_image(8)]})
        assert vlm_dataset._images_per_row(singles, "image").tolist() == [1, 0, 1]
        one = _make_image(8)
        lists = Dataset.from_dict(
            {"image": [[one] * 2, [], None, [one], [None], [one, None]]},
            features=Features({"image": Sequence(Image())}),
        )
        assert vlm_dataset._images_per_row(lists, "image").tolist() == [2, 0, 0, 1, 0, 1]


class TestMultiImageRows:
    def test_a_list_of_images_stacks_in_order(self):
        rows = [{"image": [_make_image(8, 0), _make_image(8, 255)], "caption": "a b"}]
        pixels = _dataset_over(rows, max_images=2)[0]["pixel_values"]
        assert pixels.shape == (2, 3, 16, 16)
        assert pixels[0].mean() < pixels[1].mean()

    def test_the_cap_fixes_the_sample_rank(self):
        """One image is ``(3, H, W)`` at the default cap and ``(1, 3, H, W)`` above it."""
        rows = [{"image": [_make_image(8)], "caption": "a b"}]
        assert _dataset_over(rows)[0]["pixel_values"].shape == (3, 16, 16)
        assert _dataset_over(rows, max_images=3)[0]["pixel_values"].shape == (1, 3, 16, 16)

    def test_a_longer_list_keeps_the_first_images(self):
        rows = [{"image": [_make_image(8, c) for c in (0, 128, 255)], "caption": "a b"}]
        pixels = _dataset_over(rows, max_images=2)[0]["pixel_values"]
        assert pixels.shape == (2, 3, 16, 16)
        assert pixels[0].mean() < pixels[1].mean() < 0.5

    def test_several_images_keep_the_first_at_the_default_cap(self):
        rows = [{"image": [_make_image(8, 0), _make_image(8, 255)], "caption": "a b"}]
        pixels = _dataset_over(rows)[0]["pixel_values"]
        assert pixels.shape == (3, 16, 16)
        assert pixels.mean() < 0

    def test_rows_over_the_cap_are_counted_once_when_the_dataset_is_loaded(
        self, tmp_path, monkeypatch, caplog
    ):
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        path = _mixed_image_dataset(
            tmp_path, [[_make_image(8)] * 3, [_make_image(8)], [_make_image(8)] * 4]
        )
        kwargs = dict(
            split="train", image_field="image", text_field="caption", tokenizer_path="gpt2"
        )
        with caplog.at_level(logging.WARNING, logger="kempnerforge.data.vlm_dataset"):
            HuggingFaceVLMDataset(path, max_text_len=8, **kwargs)
        assert "2 of 3 rows hold more than data.hf_max_images=1" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="kempnerforge.data.vlm_dataset"):
            ds = HuggingFaceVLMDataset(path, max_text_len=8, max_images=2, **kwargs)
        assert "2 of 3 rows hold more than data.hf_max_images=2" in caplog.text
        assert ds[2]["pixel_values"].shape == (2, 3, 224, 224)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="kempnerforge.data.vlm_dataset"):
            HuggingFaceVLMDataset(path, max_text_len=8, max_images=4, **kwargs)
        assert "hf_max_images" not in caplog.text


class TestOneDocumentPerRow:
    def test_one_document_per_fixed_row(self):
        collator = PackedVLMCollator(
            pad_id=0, seq_len=12, tokens_per_image=2, max_docs_per_row=1, allow_text_only=True
        )
        batch = collator([_doc(3, 10, views=2), _doc(4, 20), _doc(2, 30, views=0)])
        assert batch["input_ids"].shape == (3, 12)
        assert batch["pixel_values"].shape == (3, 3, 4, 4)
        assert batch["image_rows"].tolist() == [0, 0, 1]
        assert batch["image_starts"].tolist() == [0, 2, 0]
        assert batch["input_ids"][0, 4:7].tolist() == [10, 11, 12]
        assert batch["input_ids"][1, 2:6].tolist() == [20, 21, 22, 23]
        assert batch["input_ids"][2, :2].tolist() == [30, 31]
        assert batch["doc_ids"][2].tolist() == [0, 0] + [1] * 10
        assert "frame_mask" not in batch

    def test_empty_batches_and_oversized_documents_raise(self):
        collator = PackedVLMCollator(pad_id=0, seq_len=12, tokens_per_image=2, max_docs_per_row=1)
        with pytest.raises(ValueError, match="no samples"):
            collator([])
        with pytest.raises(ValueError, match="does not fit a row"):
            collator([_doc(3, 10, views=5)])


class TestFramesToClipTensor:
    """The shared frame-packing helper used by both training (``WebVidVideoDataset``)
    and the VLM evaluation adapter."""

    def test_full_clip_shape_and_dtype(self):
        frames = [_make_image(32) for _ in range(4)]
        pv, mask = frames_to_clip_tensor(frames, max_frames=4, frame_size=16)
        assert pv.shape == (4, 3, 16, 16)
        assert pv.dtype == torch.float32
        assert mask.tolist() == [True, True, True, True]

    def test_short_clip_is_zero_padded(self):
        frames = [_make_image(32) for _ in range(2)]
        pv, mask = frames_to_clip_tensor(frames, max_frames=5, frame_size=16)
        assert pv.shape == (5, 3, 16, 16)
        assert mask.tolist() == [True, True, False, False, False]
        assert torch.count_nonzero(pv[2:]) == 0

    def test_single_image_is_one_frame_clip(self):
        pv, mask = frames_to_clip_tensor([_make_image(20)], max_frames=3, frame_size=16)
        assert pv.shape == (3, 3, 16, 16)
        assert mask.tolist() == [True, False, False]

    def test_per_frame_matches_pil_to_tensor(self):
        img = _make_image(40)
        pv, _ = frames_to_clip_tensor([img], max_frames=1, frame_size=16)
        expected = pil_to_tensor(img, 16, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        assert torch.allclose(pv[0], expected)

    def test_uses_frame_size(self):
        pv, _ = frames_to_clip_tensor([_make_image(32)], max_frames=2, frame_size=24)
        assert pv.shape == (2, 3, 24, 24)

    def test_excess_frames_truncated_to_budget(self):
        frames = [_make_image(16) for _ in range(5)]
        pv, mask = frames_to_clip_tensor(frames, max_frames=3, frame_size=16)
        assert pv.shape == (3, 3, 16, 16)
        assert mask.all()


def _doc(text_len: int, first: int, views: int = 1, padded_to: int = 8) -> dict[str, torch.Tensor]:
    """A document of ``text_len`` real tokens ``first, first+1, ...`` padded to ``padded_to``."""
    ids = torch.zeros(padded_to, dtype=torch.long)
    ids[:text_len] = torch.arange(first, first + text_len)
    labels = torch.full((padded_to,), -100, dtype=torch.long)
    labels[: text_len - 1] = ids[1:text_len]
    pixels = (
        torch.full((3, 4, 4), float(first))
        if views == 1
        else torch.full((views, 3, 4, 4), float(first))
    )
    return {
        "pixel_values": pixels,
        "input_ids": ids,
        "labels": labels,
        "text_len": torch.tensor(text_len),
    }


class TestPackVLMSamples:
    def test_first_fit_layout(self):
        """Footprints 5, 6 and 7 with seq_len 12: the first two share a row, the third opens one."""
        batch = pack_vlm_samples(
            [_doc(3, 10), _doc(4, 20), _doc(5, 30)], seq_len=12, tokens_per_image=2, pad_id=0
        )
        assert batch["input_ids"].shape == (2, 12)
        assert batch["input_ids"][0].tolist() == [0, 0, 10, 11, 12, 0, 0, 20, 21, 22, 23, 0]
        assert batch["labels"][0].tolist() == [
            -100,
            -100,
            11,
            12,
            -100,
            -100,
            -100,
            21,
            22,
            23,
            -100,
            -100,
        ]
        assert batch["doc_ids"][0].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]
        assert batch["input_ids"][1].tolist() == [0, 0, 30, 31, 32, 33, 34, 0, 0, 0, 0, 0]
        assert batch["doc_ids"][1].tolist() == [0] * 7 + [1] * 5
        assert batch["position_ids"][0].tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0]
        assert batch["position_ids"][1].tolist() == list(range(7)) + list(range(5))
        assert batch["image_rows"].tolist() == [0, 0, 1]
        assert batch["image_starts"].tolist() == [0, 5, 0]
        assert batch["pixel_values"].shape == (3, 3, 4, 4)
        assert batch["pixel_values"][:, 0, 0, 0].tolist() == [10.0, 20.0, 30.0]
        assert int(batch["image_tokens"]) == 2

    def test_a_later_document_fills_an_earlier_row(self):
        batch = pack_vlm_samples(
            [_doc(6, 10), _doc(6, 20), _doc(2, 30)], seq_len=12, tokens_per_image=2, pad_id=0
        )
        assert batch["input_ids"].shape == (2, 12)
        assert batch["image_rows"].tolist() == [0, 1, 0]
        assert batch["image_starts"].tolist() == [0, 0, 8]
        assert batch["doc_ids"][0].tolist() == [0] * 8 + [1] * 4

    def test_views_take_consecutive_slots(self):
        batch = pack_vlm_samples([_doc(2, 10, views=3)], seq_len=12, tokens_per_image=2, pad_id=0)
        assert batch["pixel_values"].shape == (3, 3, 4, 4)
        assert batch["image_starts"].tolist() == [0, 2, 4]
        assert batch["input_ids"][0, 6:8].tolist() == [10, 11]

    def test_without_text_len_every_position_is_real(self):
        doc = _doc(3, 10)
        del doc["text_len"]
        batch = pack_vlm_samples([doc], seq_len=12, tokens_per_image=2, pad_id=0)
        assert batch["doc_ids"][0].tolist() == [0] * 10 + [1] * 2

    def test_oversize_document_raises(self):
        with pytest.raises(ValueError, match="does not fit a row"):
            pack_vlm_samples([_doc(8, 10)], seq_len=9, tokens_per_image=2, pad_id=0)

    def test_empty_and_bad_sizes_raise(self):
        with pytest.raises(ValueError, match="no samples"):
            pack_vlm_samples([], seq_len=8, tokens_per_image=2, pad_id=0)
        with pytest.raises(ValueError, match="must be positive"):
            pack_vlm_samples([_doc(2, 1)], seq_len=8, tokens_per_image=0, pad_id=0)

    def test_a_text_only_document_takes_no_image_slots(self):
        batch = pack_vlm_samples(
            [_doc(3, 10, views=0), _doc(4, 20)],
            seq_len=12,
            tokens_per_image=2,
            pad_id=0,
            allow_text_only=True,
        )
        assert batch["input_ids"][0].tolist() == [10, 11, 12, 0, 0, 20, 21, 22, 23, 0, 0, 0]
        assert batch["doc_ids"][0].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        assert batch["image_rows"].tolist() == [0]
        assert batch["image_starts"].tolist() == [3]
        assert batch["pixel_values"].shape == (1, 3, 4, 4)

    def test_a_batch_of_text_only_documents_carries_no_images(self):
        batch = pack_vlm_samples(
            [_doc(3, 10, views=0), _doc(4, 20, views=0)],
            seq_len=8,
            tokens_per_image=2,
            pad_id=0,
            allow_text_only=True,
        )
        assert batch["pixel_values"].shape == (0, 3, 4, 4)
        assert batch["image_rows"].shape == (0,)
        assert batch["image_starts"].shape == (0,)
        assert batch["image_rows"].dtype == torch.long
        assert batch["input_ids"][0].tolist() == [10, 11, 12, 20, 21, 22, 23, 0]

    def test_max_docs_per_row_keeps_documents_apart(self):
        batch = pack_vlm_samples(
            [_doc(3, 10), _doc(4, 20)], seq_len=12, tokens_per_image=2, pad_id=0, max_docs_per_row=1
        )
        assert batch["input_ids"].shape == (2, 12)
        assert batch["image_rows"].tolist() == [0, 1]
        assert batch["doc_ids"][0].tolist() == [0] * 5 + [1] * 7

    def test_max_docs_per_row_caps_a_row_at_the_count(self):
        docs = [_doc(2, 10), _doc(2, 20), _doc(2, 30)]
        batch = pack_vlm_samples(docs, seq_len=12, tokens_per_image=2, pad_id=0, max_docs_per_row=2)
        assert batch["input_ids"].shape == (2, 12)
        assert batch["image_rows"].tolist() == [0, 0, 1]
        assert batch["doc_ids"][0].tolist() == [0] * 4 + [1] * 4 + [2] * 4

    def test_collator_packs_the_batch(self):
        collator = PackedVLMCollator(pad_id=7, seq_len=12, tokens_per_image=2)
        batch = collator([_doc(3, 10), _doc(4, 20)])
        assert batch["input_ids"].shape == (1, 12)
        assert batch["input_ids"][0, 11].item() == 7
        with pytest.raises(ValueError, match="no samples"):
            collator([])

    def test_a_text_only_document_needs_the_flag(self):
        doc = _doc(3, 10, views=0)
        with pytest.raises(ValueError, match="hf_allow_text_only"):
            pack_vlm_samples([doc], seq_len=12, tokens_per_image=2, pad_id=0)
        collator = PackedVLMCollator(pad_id=0, seq_len=12, tokens_per_image=2)
        with pytest.raises(ValueError, match="hf_allow_text_only"):
            collator([doc])
        packed = PackedVLMCollator(pad_id=0, seq_len=12, tokens_per_image=2, allow_text_only=True)
        assert packed([doc])["pixel_values"].shape == (0, 3, 4, 4)
