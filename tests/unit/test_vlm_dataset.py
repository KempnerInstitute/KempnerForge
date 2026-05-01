"""Unit tests for HuggingFaceVLMDataset and VLMCollator."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    HuggingFaceVLMDataset,
    VLMCollator,
    _pil_to_tensor,
    _tokenize_and_mask,
)


def _make_image(size: int = 64, color: int = 128) -> Image.Image:
    return Image.new("RGB", (size, size), color=(color, color, color))


# ---------------------------------------------------------------------------
# _pil_to_tensor
# ---------------------------------------------------------------------------


class TestPILToTensor:
    def test_shape_and_dtype(self):
        img = _make_image(48)
        t = _pil_to_tensor(img, 224, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        assert t.shape == (3, 224, 224)
        assert t.dtype == torch.float32

    def test_non_square_resize(self):
        img = Image.new("RGB", (100, 50), color=(0, 128, 255))
        t = _pil_to_tensor(img, 64, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        assert t.shape == (3, 64, 64)

    def test_grayscale_promoted_to_rgb(self):
        img = Image.new("L", (32, 32), color=128)
        t = _pil_to_tensor(img, 32, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        assert t.shape == (3, 32, 32)

    def test_normalization(self):
        """Input with pixels at 127.5 (0.5 after /255) should normalize
        to ~0 under mean=0.5, std=0.5."""
        img = Image.new("RGB", (16, 16), color=(128, 128, 128))  # ~0.502
        t = _pil_to_tensor(img, 16, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        # (0.502 - 0.5) / 0.5 ~= 0.004
        assert abs(float(t.mean())) < 0.02

    def test_non_pil_raises(self):
        with pytest.raises(TypeError, match="Expected a PIL.Image"):
            _pil_to_tensor(torch.randn(3, 16, 16), 16, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)


# ---------------------------------------------------------------------------
# _tokenize_and_mask
# ---------------------------------------------------------------------------


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
        ids, labels = _tokenize_and_mask(tok, "abc", max_text_len=8, prompt=None)
        assert ids.shape == (8,)
        assert labels.shape == (8,)
        assert ids[:3].tolist() == [1, 2, 3]
        assert ids[3].item() == 0  # pad
        assert labels[3].item() == -100  # pad masked

    def test_prompt_masks_labels(self):
        tok = _MockTokenizer()
        ids, labels = _tokenize_and_mask(tok, text="xyz", max_text_len=8, prompt="ab")
        # Prompt "ab" = 2 tokens; target "xyz" = 3 tokens; total 5 tokens.
        assert ids[:5].tolist() == [1, 2, 24, 25, 26]
        assert labels[:2].tolist() == [-100, -100]  # prompt masked
        assert labels[2:5].tolist() == [24, 25, 26]  # targets not masked
        assert labels[5].item() == -100  # pad masked

    def test_truncation(self):
        tok = _MockTokenizer()
        ids, labels = _tokenize_and_mask(tok, text="abcdefghij", max_text_len=4, prompt=None)
        assert ids.tolist() == [1, 2, 3, 4]
        assert labels.tolist() == [1, 2, 3, 4]

    def test_prompt_mask_with_bpe_tokenizer(self):
        """Regression: BPE (gpt2) and SentencePiece tokenizers are not
        prefix-preserving. ``tokenize(prompt) + tokenize(text)`` differs
        from ``tokenize(prompt + text)`` at the boundary (tokens can
        merge or split). The implementation must tokenize the prompt and
        text independently and concatenate the id lists, so the
        ``labels[:len(prompt_ids)]`` mask lines up exactly with the
        prompt boundary. Verified end-to-end on the gpt2 tokenizer.
        """
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        prompt = "Q: "
        text = "hello"

        prompt_ids = list(tok(prompt, add_special_tokens=False)["input_ids"])
        text_ids = list(tok(text, add_special_tokens=False)["input_ids"])
        expected_ids = prompt_ids + text_ids
        n = len(expected_ids)

        ids, labels = _tokenize_and_mask(tok, text, max_text_len=32, prompt=prompt)

        # input_ids exactly matches the independent-concat form.
        assert ids[:n].tolist() == expected_ids
        # Prompt portion of labels is -100.
        assert (labels[: len(prompt_ids)] == -100).all()
        # Target portion is the independently-tokenized text ids, byte-for-byte.
        assert labels[len(prompt_ids) : n].tolist() == text_ids

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

        ids, labels = _tokenize_and_mask(tok, text, max_text_len=16, prompt=prompt)

        # We store the split form on input_ids (not the joined BPE form).
        split_total = len(prompt_ids) + len(text_ids)
        assert ids[:split_total].tolist() == prompt_ids + text_ids
        # Target is exactly tokenize(text) — a regression would show up
        # here as a drift in the label ids (not just their length).
        assert labels[len(prompt_ids) : len(prompt_ids) + len(text_ids)].tolist() == text_ids


# ---------------------------------------------------------------------------
# VLMCollator
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# HuggingFaceVLMDataset (in-memory, no network)
# ---------------------------------------------------------------------------


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

        # Bypass load_dataset by injecting the preloaded dataset.
        class _Wrapper(HuggingFaceVLMDataset):
            def __init__(self, ds, tokenizer_path, max_text_len):
                from transformers import AutoTokenizer

                self._ds = ds
                self._image_field = "image"
                self._text_field = "caption"
                self._prompt_field = None
                self._image_size = 32
                self._image_mean = DEFAULT_IMAGE_MEAN
                self._image_std = DEFAULT_IMAGE_STD
                self._max_text_len = max_text_len
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        ds = load_from_disk(tiny_hf_dataset)
        wrapper = _Wrapper(ds, tokenizer_path="gpt2", max_text_len=16)
        item = wrapper[0]
        assert item["pixel_values"].shape == (3, 32, 32)
        assert item["input_ids"].shape == (16,)
        assert item["labels"].shape == (16,)
        assert item["input_ids"].dtype == torch.long

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
        # Confirm we got the validation split, not train.
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
        # Patch the import target that __init__ pulls in lazily.
        import datasets

        monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

        ds = HuggingFaceVLMDataset(
            dataset_name="some/hub-id",  # not a directory -> load_dataset path
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

        # Build a tiny IterableDataset and have load_dataset hand it back.
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
        # tiny_hf_dataset fixture is parameter-bound (forces creation) but we
        # do not need to load it here; the schema-guard check uses a fresh
        # in-memory dataset with a non-string column.
        del tiny_hf_dataset
        from datasets import Dataset

        class _Wrapper(HuggingFaceVLMDataset):
            def __init__(self, ds, tokenizer_path, max_text_len):
                from transformers import AutoTokenizer

                self._ds = ds
                self._image_field = "image"
                self._text_field = "caption"
                self._prompt_field = None
                self._image_size = 32
                self._image_mean = DEFAULT_IMAGE_MEAN
                self._image_std = DEFAULT_IMAGE_STD
                self._max_text_len = max_text_len
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        mixed = Dataset.from_dict(
            {
                "image": [_make_image(32, 100)],
                "caption": [{"not": "a string"}],  # type: ignore[list-item]
            }
        )
        wrapper = _Wrapper(mixed, tokenizer_path="gpt2", max_text_len=4)
        with pytest.raises(TypeError, match="must be str"):
            _ = wrapper[0]

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

        class _Wrapper(HuggingFaceVLMDataset):
            def __init__(self, ds, tokenizer_path, max_text_len):
                from transformers import AutoTokenizer

                self._ds = ds
                self._image_field = "image"
                self._text_field = "caption"
                self._prompt_field = None
                self._image_size = 16
                self._image_mean = DEFAULT_IMAGE_MEAN
                self._image_std = DEFAULT_IMAGE_STD
                self._max_text_len = max_text_len
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        ds = load_from_disk(tiny_hf_dataset)
        wrapper = _Wrapper(ds, tokenizer_path="gpt2", max_text_len=12)
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
