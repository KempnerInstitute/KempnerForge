"""Unit tests for WebVidVideoDataset and VideoCollator.

The dataset is exercised with a stubbed decoder (no real video / no ``av``)
and a char-level mock tokenizer (no HF download), mirroring the approach in
``test_vlm_dataset.py``.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from PIL import Image

from kempnerforge.data import video_dataset as vd
from kempnerforge.data.video_dataset import VideoCollator, WebVidVideoDataset
from kempnerforge.data.vlm_dataset import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD


class _MockTokenizer:
    """Char-level tokenizer (a->1..z->26, space->27, '.'->28), pad id 0."""

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


class _StubVideoDataset(WebVidVideoDataset):
    """Bypass __init__ (no CSV/tokenizer loading); set attributes directly."""

    def __init__(
        self,
        ids: list[str],
        caps: list[str],
        split: str = "train",
        *,
        max_frames: int = 8,
        min_frames: int = 2,
        fps: float = 2.0,
        frame_size: int = 16,
        max_text_len: int = 8,
        prompt: str = "",
    ) -> None:
        self._ids = ids
        self._caps = caps
        self._split = split
        self._video_dir = f"/fake/videos/{'train' if split == 'train' else 'validation'}"
        self._tokenizer = _MockTokenizer()
        self._pad_id = 0
        self._max_text_len = max_text_len
        self._max_frames = max_frames
        self._min_frames = min_frames
        self._fps = fps
        self._frame_size = frame_size
        self._prompt = prompt
        self._image_mean = DEFAULT_IMAGE_MEAN
        self._image_std = DEFAULT_IMAGE_STD


def _frames(n: int, size: int = 16) -> list[Image.Image]:
    return [Image.new("RGB", (size, size), color=(i * 10 % 255, 0, 0)) for i in range(n)]


# ---------------------------------------------------------------------------
# Video path mapping (verified against the on-disk WebVid layout)
# ---------------------------------------------------------------------------


class TestVideoPath:
    def test_train_prefix_nesting(self):
        ds = _StubVideoDataset(["8469580"], ["x"], split="train")
        assert ds._video_path("8469580") == "/fake/videos/train/84/8469/846958/8469580.mp4"

    def test_short_id_prefix(self):
        # id shorter than 6 chars: id[:6] is the whole id.
        ds = _StubVideoDataset(["84490"], ["x"], split="train")
        assert ds._video_path("84490") == "/fake/videos/train/84/8449/84490/84490.mp4"

    def test_validation_is_flat(self):
        ds = _StubVideoDataset(["10006310"], ["x"], split="validation")
        assert ds._video_path("10006310") == "/fake/videos/validation/10006310.mp4"


# ---------------------------------------------------------------------------
# __getitem__ (stubbed decoder)
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_shapes_and_mask_full_clip(self, monkeypatch):
        monkeypatch.setattr(vd, "decode_video_frames", lambda *a, **k: _frames(8))
        ds = _StubVideoDataset(["1"], ["a cat."], max_frames=8, frame_size=16)
        item = ds[0]
        assert item["pixel_values"].shape == (8, 3, 16, 16)
        assert item["pixel_values"].dtype == torch.float32
        assert item["frame_mask"].shape == (8,)
        assert item["frame_mask"].dtype == torch.bool
        assert item["frame_mask"].all()
        assert item["input_ids"].shape == (8,)
        assert item["labels"].shape == (8,)

    def test_pads_and_masks_short_clip(self, monkeypatch):
        monkeypatch.setattr(vd, "decode_video_frames", lambda *a, **k: _frames(3))
        ds = _StubVideoDataset(["1"], ["a dog."], max_frames=8)
        item = ds[0]
        assert item["frame_mask"].tolist() == [True, True, True, False, False, False, False, False]
        # Padded frames are zeros.
        assert torch.count_nonzero(item["pixel_values"][3:]) == 0

    def test_caption_is_supervised_when_frames_present(self, monkeypatch):
        monkeypatch.setattr(vd, "decode_video_frames", lambda *a, **k: _frames(4))
        ds = _StubVideoDataset(["1"], ["abc"], max_frames=8, max_text_len=8)
        item = ds[0]
        # "abc" -> ids 1,2,3 supervised; rest -100.
        assert item["labels"][:3].tolist() == [1, 2, 3]
        assert (item["labels"][3:] == -100).all()

    def test_decode_failure_yields_zero_clip_no_loss(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("corrupt video")

        monkeypatch.setattr(vd, "decode_video_frames", _boom)
        ds = _StubVideoDataset(["1"], ["a cat."], max_frames=8)
        item = ds[0]
        assert torch.count_nonzero(item["pixel_values"]) == 0
        assert not item["frame_mask"].any()
        assert (item["labels"] == -100).all()  # no supervision for an unloadable clip

    def test_empty_decode_yields_zero_clip_no_loss(self, monkeypatch):
        monkeypatch.setattr(vd, "decode_video_frames", lambda *a, **k: [])
        ds = _StubVideoDataset(["1"], ["a cat."], max_frames=4)
        item = ds[0]
        assert not item["frame_mask"].any()
        assert (item["labels"] == -100).all()

    def test_prompt_is_masked(self, monkeypatch):
        monkeypatch.setattr(vd, "decode_video_frames", lambda *a, **k: _frames(2))
        ds = _StubVideoDataset(["1"], ["xyz"], max_frames=4, max_text_len=8, prompt="ab")
        item = ds[0]
        # prompt "ab" (2 toks) masked; "xyz" (24,25,26) supervised.
        assert item["input_ids"][:5].tolist() == [1, 2, 24, 25, 26]
        assert item["labels"][:2].tolist() == [-100, -100]
        assert item["labels"][2:5].tolist() == [24, 25, 26]

    def test_len(self):
        ds = _StubVideoDataset(["1", "2", "3"], ["a", "b", "c"])
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# VideoCollator
# ---------------------------------------------------------------------------


class TestVideoCollator:
    def _sample(self, n_frames_valid: int, max_frames: int = 4, max_text_len: int = 8):
        pv = torch.zeros(max_frames, 3, 16, 16)
        pv[:n_frames_valid] = torch.randn(n_frames_valid, 3, 16, 16)
        mask = torch.zeros(max_frames, dtype=torch.bool)
        mask[:n_frames_valid] = True
        ids = torch.zeros(max_text_len, dtype=torch.long)
        ids[:3] = torch.tensor([1, 2, 3])
        labels = torch.full((max_text_len,), -100, dtype=torch.long)
        labels[:3] = torch.tensor([1, 2, 3])
        return {"pixel_values": pv, "frame_mask": mask, "input_ids": ids, "labels": labels}

    def test_batch_shapes(self):
        collator = VideoCollator(pad_id=0, max_text_len=8)
        batch = collator([self._sample(4), self._sample(2), self._sample(3)])
        assert batch["pixel_values"].shape == (3, 4, 3, 16, 16)
        assert batch["frame_mask"].shape == (3, 4)
        assert batch["frame_mask"].dtype == torch.bool
        assert batch["input_ids"].shape == (3, 8)
        assert batch["labels"].shape == (3, 8)

    def test_frame_mask_preserved(self):
        collator = VideoCollator(pad_id=0, max_text_len=8)
        batch = collator([self._sample(2, max_frames=4)])
        assert batch["frame_mask"][0].tolist() == [True, True, False, False]

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError, match="empty batch"):
            VideoCollator(pad_id=0, max_text_len=8)([])

    def test_max_text_len_must_be_positive(self):
        with pytest.raises(ValueError, match="max_text_len must be positive"):
            VideoCollator(pad_id=0, max_text_len=0)


# ---------------------------------------------------------------------------
# Real dataset integration: build a synthetic WebVid layout (CSV manifest +
# a tiny encoded .mp4 at the prefix path) and exercise the real __init__,
# manifest load, path mapping, __getitem__ decode, and the decode-failure
# path. Uses av (a hard dependency) so it runs in CI; gpt2 tokenizer matches
# the existing VLM dataset tests.
# ---------------------------------------------------------------------------

_AV_AVAILABLE = importlib.util.find_spec("av") is not None


def _write_mp4(path, n_frames: int, size: int = 32, fps: int = 8) -> None:
    import av
    import numpy as np

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            arr = np.full((size, size, 3), (i * 17) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestRealDatasetIntegration:
    def _manifest_dir(self, root):
        d = root / "raw" / "webvid-10M" / "data" / "train" / "partitions"
        d.mkdir(parents=True)
        return d

    def test_init_getitem_and_decode(self, tmp_path):
        vid, cap = "123456", "a test clip"
        (self._manifest_dir(tmp_path) / "0000.csv").write_text(f"videoid,name\n{vid},{cap}\n")
        vdir = tmp_path / "raw" / "videos" / "train" / vid[:2] / vid[:4] / vid[:6]
        vdir.mkdir(parents=True)
        _write_mp4(vdir / f"{vid}.mp4", n_frames=16, size=32, fps=8)

        ds = WebVidVideoDataset(
            data_root=str(tmp_path),
            split="train",
            tokenizer_path="gpt2",
            max_text_len=16,
            max_frames=8,
            min_frames=4,
            fps=2.0,
            frame_size=32,
        )
        assert len(ds) == 1
        item = ds[0]
        assert item["pixel_values"].shape == (8, 3, 32, 32)
        assert item["frame_mask"].any()  # real frames decoded
        assert (item["labels"] != -100).any()  # caption supervised

    def test_decode_failure_is_masked(self, tmp_path):
        # Manifest points at a videoid with no .mp4 on disk -> decode raises,
        # __getitem__ catches it and yields a zero clip with no loss.
        (self._manifest_dir(tmp_path) / "0000.csv").write_text("videoid,name\n999999,missing\n")
        ds = WebVidVideoDataset(
            data_root=str(tmp_path),
            split="train",
            tokenizer_path="gpt2",
            max_text_len=8,
            max_frames=4,
            min_frames=2,
            fps=2.0,
            frame_size=16,
        )
        item = ds[0]
        assert not item["frame_mask"].any()
        assert (item["labels"] == -100).all()

    def test_empty_manifest_raises(self, tmp_path):
        self._manifest_dir(tmp_path)  # dir exists but no CSVs
        with pytest.raises(FileNotFoundError, match="No partition CSVs"):
            WebVidVideoDataset(
                data_root=str(tmp_path),
                split="train",
                tokenizer_path="gpt2",
                max_text_len=8,
                max_frames=4,
                min_frames=2,
                fps=2.0,
            )
