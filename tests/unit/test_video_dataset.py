"""Unit tests for WebVidVideoDataset and VideoCollator.

The dataset is exercised with a stubbed decoder (no real video / no ``av``)
and a char-level mock tokenizer (no HF download), mirroring the approach in
``test_vlm_dataset.py``.
"""

from __future__ import annotations

import collections
import importlib.util

import pytest
import torch
from PIL import Image

from kempnerforge.data import video_dataset as vd
from kempnerforge.data.frame_selection import CandidatePoolSpec
from kempnerforge.data.video_dataset import VideoCollator, WebVidVideoDataset
from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    pil_to_uint8_tensor,
)


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
        self._sampling_policy = "uniform"
        self._image_mean = DEFAULT_IMAGE_MEAN
        self._image_std = DEFAULT_IMAGE_STD


def _frames(n: int, size: int = 16) -> list[Image.Image]:
    return [Image.new("RGB", (size, size), color=(i * 10 % 255, 0, 0)) for i in range(n)]


def _decoded(n: int, size: int = 16) -> tuple[list[Image.Image], list[float]]:
    """Mimic decode_video_frames: (frames, per-frame presentation seconds)."""
    return _frames(n, size), [float(i) for i in range(n)]


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
        monkeypatch.setattr(vd, "decode_video_frames", lambda *a, **k: _decoded(8))
        ds = _StubVideoDataset(["1"], ["a cat."], max_frames=8, frame_size=16)
        item = ds[0]
        assert item["pixel_values"].shape == (8, 3, 16, 16)
        assert item["pixel_values"].dtype == torch.float32
        assert item["frame_mask"].shape == (8,)
        assert item["frame_mask"].dtype == torch.bool
        assert item["frame_mask"].all()
        assert item["frame_times"].shape == (8,)
        assert item["frame_times"].dtype == torch.float32
        assert item["frame_times"].tolist() == [float(i) for i in range(8)]
        assert item["input_ids"].shape == (8,)
        assert item["labels"].shape == (8,)

    def test_pads_and_masks_short_clip(self, monkeypatch):
        monkeypatch.setattr(vd, "decode_video_frames", lambda *a, **k: _decoded(3))
        ds = _StubVideoDataset(["1"], ["a dog."], max_frames=8)
        item = ds[0]
        assert item["frame_mask"].tolist() == [True, True, True, False, False, False, False, False]
        # Padded frames are zeros.
        assert torch.count_nonzero(item["pixel_values"][3:]) == 0
        # Real frames carry their times; pad frames are 0.0.
        assert item["frame_times"][:3].tolist() == [0.0, 1.0, 2.0]
        assert item["frame_times"][3:].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_caption_is_supervised_when_frames_present(self, monkeypatch):
        monkeypatch.setattr(vd, "decode_video_frames", lambda *a, **k: _decoded(4))
        ds = _StubVideoDataset(["1"], ["abc"], max_frames=8, max_text_len=8)
        item = ds[0]
        # "abc" -> [1,2,3] + EOS(28); next-token labels [2,3,28,-100], rest -100.
        assert item["labels"][:4].tolist() == [2, 3, 28, -100]
        assert (item["labels"][4:] == -100).all()

    def test_decode_failure_yields_zero_clip_no_loss(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("corrupt video")

        monkeypatch.setattr(vd, "decode_video_frames", _boom)
        ds = _StubVideoDataset(["1"], ["a cat."], max_frames=8)
        item = ds[0]
        assert torch.count_nonzero(item["pixel_values"]) == 0
        assert not item["frame_mask"].any()
        assert torch.count_nonzero(item["frame_times"]) == 0
        assert (item["labels"] == -100).all()  # no supervision for an unloadable clip


class TestProbeChecksClipPresence:
    def test_missing_clip_not_counted_as_supervised(self, tmp_path):
        # A record with text but an absent clip decodes to zero frames at
        # __getitem__ -> all labels masked. probe_supervision must therefore not
        # count it, so a partially-downloaded corpus (require_video_file off) is
        # reported honestly instead of at false-high supervision.
        from kempnerforge.data.video_dataset import VideoRecord

        present = tmp_path / "present.mp4"
        present.write_bytes(b"x")

        class _DS(_StubVideoDataset):
            def _record(self, idx):
                path = str(present) if idx == 0 else str(tmp_path / "missing.mp4")
                return VideoRecord(path, "", "a caption.")

        ds = _DS(["a", "b"], ["a caption.", "a caption."], max_text_len=16)
        assert ds.probe_supervision() == 0.5  # present supervised, missing skipped

    def test_empty_decode_yields_zero_clip_no_loss(self, monkeypatch):
        monkeypatch.setattr(vd, "decode_video_frames", lambda *a, **k: ([], []))
        ds = _StubVideoDataset(["1"], ["a cat."], max_frames=4)
        item = ds[0]
        assert not item["frame_mask"].any()
        assert (item["labels"] == -100).all()

    def test_prompt_is_masked(self, monkeypatch):
        monkeypatch.setattr(vd, "decode_video_frames", lambda *a, **k: _decoded(2))
        ds = _StubVideoDataset(["1"], ["xyz"], max_frames=4, max_text_len=8, prompt="ab")
        item = ds[0]
        # prompt "ab": last prompt token predicts first target (24); caption + EOS(28).
        assert item["input_ids"][:6].tolist() == [1, 2, 24, 25, 26, 28]
        assert item["labels"][0].item() == -100
        assert item["labels"][1:5].tolist() == [24, 25, 26, 28]
        assert item["labels"][5].item() == -100

    def test_len(self):
        ds = _StubVideoDataset(["1", "2", "3"], ["a", "b", "c"])
        assert len(ds) == 3

    def test_pixel_values_match_shared_helper(self, monkeypatch):
        """The dataset's clip is exactly ``frames_to_clip_tensor``'s output (the
        shared helper), guaranteeing training/eval frame-packing parity by
        construction."""
        from kempnerforge.data.vlm_dataset import frames_to_clip_tensor

        frames = _frames(3)
        monkeypatch.setattr(
            vd, "decode_video_frames", lambda *a, **k: (frames, [0.0] * len(frames))
        )
        ds = _StubVideoDataset(["1"], ["a cat."], max_frames=8, frame_size=16)
        item = ds[0]
        expected_pv, expected_mask = frames_to_clip_tensor(
            frames,
            max_frames=8,
            frame_size=16,
            image_mean=DEFAULT_IMAGE_MEAN,
            image_std=DEFAULT_IMAGE_STD,
        )
        assert torch.equal(item["pixel_values"], expected_pv)
        assert torch.equal(item["frame_mask"], expected_mask)


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
        times = torch.zeros(max_frames, dtype=torch.float32)
        times[:n_frames_valid] = torch.arange(n_frames_valid, dtype=torch.float32)
        return {
            "pixel_values": pv,
            "frame_mask": mask,
            "frame_times": times,
            "input_ids": ids,
            "labels": labels,
        }

    def test_batch_shapes(self):
        collator = VideoCollator(pad_id=0, max_text_len=8)
        batch = collator([self._sample(4), self._sample(2), self._sample(3)])
        assert batch["pixel_values"].shape == (3, 4, 3, 16, 16)
        assert batch["frame_mask"].shape == (3, 4)
        assert batch["frame_mask"].dtype == torch.bool
        assert batch["frame_times"].shape == (3, 4)
        assert batch["frame_times"].dtype == torch.float32
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

    def _pool_sample(self, count: int, max_text_len: int = 8, tag: str = "a"):
        # Ragged emission: exactly ``count`` frames, no worker-side padding.
        pixels = torch.randint(1, 255, (count, 3, 16, 16), dtype=torch.uint8)
        times = torch.arange(count, dtype=torch.float32)
        ids = torch.zeros(max_text_len, dtype=torch.long)
        labels = torch.full((max_text_len,), -100, dtype=torch.long)
        return {
            "candidate_pixels": pixels,
            "candidate_count": torch.tensor(count, dtype=torch.long),
            "candidate_times": times,
            "query": f"question {tag}?",
            "seed_key": f"/clip/{tag}.mp4|question {tag}?",
            "input_ids": ids,
            "labels": labels,
        }

    def test_pool_mode_batch(self):
        collator = VideoCollator(pad_id=0, max_text_len=8)
        batch = collator([self._pool_sample(6, tag="a"), self._pool_sample(2, tag="b")])
        # C = batch-max candidate count (6 here), not a fixed candidate_frames.
        assert batch["candidate_pixels"].shape == (2, 6, 3, 16, 16)
        assert batch["candidate_pixels"].dtype == torch.uint8
        assert batch["candidate_count"].tolist() == [6, 2]
        assert batch["candidate_times"].shape == (2, 6)
        assert batch["query"] == ["question a?", "question b?"]
        assert batch["seed_key"] == ["/clip/a.mp4|question a?", "/clip/b.mp4|question b?"]
        assert batch["input_ids"].shape == (2, 8)
        assert batch["labels"].shape == (2, 8)
        assert "pixel_values" not in batch  # produced downstream by apply_frame_selection

    def test_pool_mode_pads_ragged_samples_to_batch_max(self):
        collator = VideoCollator(pad_id=0, max_text_len=8)
        s_full, s_short = self._pool_sample(5, tag="a"), self._pool_sample(2, tag="b")
        batch = collator([s_full, s_short])
        assert batch["candidate_pixels"].shape == (2, 5, 3, 16, 16)
        # Real prefixes are copied verbatim...
        assert torch.equal(batch["candidate_pixels"][0], s_full["candidate_pixels"])
        assert torch.equal(batch["candidate_pixels"][1, :2], s_short["candidate_pixels"])
        assert batch["candidate_times"][0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert batch["candidate_times"][1, :2].tolist() == [0.0, 1.0]
        # ...and the padded tail past each count is zeros.
        assert (batch["candidate_pixels"][1, 2:] == 0).all()
        assert (batch["candidate_times"][1, 2:] == 0).all()

    def test_pool_mode_all_failed_decodes_pads_to_one(self):
        # Every sample in the batch failed to decode (count 0, empty pixel
        # tensors): cmax floors at 1 so the batch still has a valid shape.
        collator = VideoCollator(pad_id=0, max_text_len=8)
        batch = collator([self._pool_sample(0, tag="a"), self._pool_sample(0, tag="b")])
        assert batch["candidate_pixels"].shape == (2, 1, 3, 16, 16)
        assert (batch["candidate_pixels"] == 0).all()
        assert batch["candidate_times"].shape == (2, 1)
        assert batch["candidate_count"].tolist() == [0, 0]

    def test_pool_mode_forwards_dataset_idx(self):
        collator = VideoCollator(pad_id=0, max_text_len=8)
        s0, s1 = self._pool_sample(3, tag="a"), self._pool_sample(3, tag="b")
        s0["dataset_idx"] = 0
        s1["dataset_idx"] = 1
        batch = collator([s0, s1])
        assert batch["dataset_idx"].tolist() == [0, 1]


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
        assert item["frame_times"].shape == (8,)
        assert (item["frame_times"][item["frame_mask"]] >= 0).all()  # real-frame times set
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

    def test_build_video_dataset_with_custom_dataset_name(self, tmp_path):
        # De-hardcoded: a custom dataset_name reads raw/<name>/data, and
        # build_video_dataset dispatches via the registry (no hardcoded class).
        from kempnerforge.config.video import VideoConfig
        from kempnerforge.data.video_dataset import VideoDataset, build_video_dataset

        name = "my-webvid"
        mdir = tmp_path / "raw" / name / "data" / "train" / "partitions"
        mdir.mkdir(parents=True)
        vid, cap = "654321", "a clip"
        (mdir / "0000.csv").write_text(f"videoid,name\n{vid},{cap}\n")
        vdir = tmp_path / "raw" / "videos" / "train" / vid[:2] / vid[:4] / vid[:6]
        vdir.mkdir(parents=True)
        _write_mp4(vdir / f"{vid}.mp4", n_frames=16, size=32, fps=8)

        cfg = VideoConfig(
            data_root=str(tmp_path),
            dataset_name=name,
            split="train",
            max_frames=8,
            min_frames=4,
            fps=2.0,
            frame_size=32,
        )
        ds = build_video_dataset(cfg, "gpt2", max_text_len=16)
        assert isinstance(ds, VideoDataset)
        assert len(ds) == 1
        assert ds[0]["frame_mask"].any()


class TestVideoDatasetRegistry:
    """Registry + build_video_dataset make the dataset style config-switchable."""

    def test_webvid_registered(self):
        from kempnerforge.config.registry import registry

        assert "webvid" in registry.list_video_datasets()

    def test_is_video_dataset_subclass(self):
        from kempnerforge.data.video_dataset import VideoDataset

        assert issubclass(WebVidVideoDataset, VideoDataset)

    def test_unknown_dataset_type_raises(self):
        from kempnerforge.config.registry import registry

        with pytest.raises(KeyError, match="video_dataset"):
            registry.get_video_dataset("bogus")


# ---------------------------------------------------------------------------
# Candidate-pool mode (base-class capability, dataset-agnostic)
# ---------------------------------------------------------------------------


class TestGetItemPoolMode:
    def test_pool_contract_ragged_and_pixels(self, monkeypatch):
        frames = _frames(4)
        monkeypatch.setattr(
            vd, "decode_candidate_pool", lambda path, **k: (frames, [0.0, 1.0, 2.0, 3.0])
        )
        # The query is the sample's prompt (the input-time question/instruction),
        # never the caption/target; the seed is (clip path, query) so multiple
        # questions per clip draw independently (see the per-question test below).
        # max_text_len leaves room for prompt + target so labels stay supervised.
        ds = _StubVideoDataset(
            ["7788"], ["a caption"], max_frames=8, prompt="describe:", max_text_len=24
        )
        ds._candidate_spec = CandidatePoolSpec(candidate_frames=6)
        item = ds[0]
        # Ragged emission: exactly the decoded frames, no padding to
        # candidate_frames (batch padding moved to VideoCollator).
        assert item["candidate_pixels"].shape == (4, 3, 16, 16)
        assert item["candidate_pixels"].dtype == torch.uint8
        assert item["candidate_count"].item() == 4
        for j in range(4):  # shipped pixels are the worker-side resize, exactly
            assert torch.equal(item["candidate_pixels"][j], pil_to_uint8_tensor(frames[j], 16))
        assert item["candidate_times"].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert item["query"] == "describe:"  # the prompt, not the caption
        assert item["seed_key"] == f"{ds._video_path('7788')}|describe:"
        assert "pixel_values" not in item  # selection happens downstream
        assert not (item["labels"] == -100).all()  # supervised: real target + frames

    def test_empty_query_fast_path_pre_strides_pool(self, monkeypatch):
        # An empty query can never be scored: the worker pre-applies the exact
        # uniform stride FrameSelector._fallback_indices would take, shipping
        # only max_frames frames instead of the full pool.
        from kempnerforge.data.frame_selection import uniform_stride_indices

        frames = _frames(6)
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        monkeypatch.setattr(vd, "decode_candidate_pool", lambda path, **k: (frames, times))
        ds = _StubVideoDataset(["1"], ["a caption"], max_frames=4)  # prompt="" -> no query
        ds._candidate_spec = CandidatePoolSpec(candidate_frames=8)
        item = ds[0]
        sel = uniform_stride_indices(6, 4)
        assert sel == [0, 1, 3, 4]  # [i * n // k], pinned to _fallback_indices
        assert item["candidate_count"].item() == 4
        assert item["candidate_pixels"].shape == (4, 3, 16, 16)
        for out_j, src_j in enumerate(sel):
            assert torch.equal(
                item["candidate_pixels"][out_j], pil_to_uint8_tensor(frames[src_j], 16)
            )
        assert item["candidate_times"].tolist() == [times[j] for j in sel]
        assert not (item["labels"] == -100).all()  # still supervised (n > 0)

    def test_empty_query_small_pool_ships_all(self, monkeypatch):
        # n <= max_frames: nothing to stride, ship the whole pool (mirrors the
        # selector's take-all branch, checked BEFORE its empty-query stride).
        frames = _frames(3)
        monkeypatch.setattr(
            vd, "decode_candidate_pool", lambda path, **k: (frames, [0.0, 1.0, 2.0])
        )
        ds = _StubVideoDataset(["1"], ["a caption"], max_frames=4)  # prompt="" -> no query
        ds._candidate_spec = CandidatePoolSpec(candidate_frames=8)
        item = ds[0]
        assert item["candidate_count"].item() == 3
        assert item["candidate_pixels"].shape == (3, 3, 16, 16)
        assert item["candidate_times"].tolist() == [0.0, 1.0, 2.0]

    def test_whitespace_only_prompt_takes_fast_path(self, monkeypatch):
        # " " strips to nothing: _fallback_indices would uniform-stride it, so
        # the worker must pre-stride too (same strip semantics) instead of
        # shipping the full pool across IPC for a deterministic no-op.
        frames = _frames(6)
        times = [float(i) for i in range(6)]
        monkeypatch.setattr(vd, "decode_candidate_pool", lambda path, **k: (frames, times))
        ds = _StubVideoDataset(
            ["1"], ["a caption"], max_frames=4, prompt=" ", max_text_len=24
        )
        ds._candidate_spec = CandidatePoolSpec(candidate_frames=8)
        item = ds[0]
        assert item["candidate_count"].item() == 4  # pre-strided, not 6
        assert item["candidate_times"].tolist() == [0.0, 1.0, 3.0, 4.0]

    def test_selection_query_prefers_bare_question(self, monkeypatch):
        # A QA record threads the bare question via VideoRecord.query; scoring
        # conditions on it (not the rendered options/boilerplate prompt), and
        # the seed key folds it in via make_seed_key.
        from kempnerforge.data.frame_selection import make_seed_key
        from kempnerforge.data.video_dataset import VideoRecord

        monkeypatch.setattr(
            vd, "decode_candidate_pool", lambda path, **k: (_frames(2), [0.0, 1.0])
        )

        class _BareQuestionDataset(_StubVideoDataset):
            def _record(self, idx):
                return VideoRecord(
                    "/clip.mp4",
                    "Question: what happens?\nA. x\nB. y\nAnswer with the option's letter.\nAnswer:",
                    " A",
                    query="what happens?",
                )

        ds = _BareQuestionDataset(["1"], ["x"], max_frames=4, max_text_len=48)
        ds._candidate_spec = CandidatePoolSpec(candidate_frames=4)
        item = ds[0]
        assert item["query"] == "what happens?"
        assert item["seed_key"] == make_seed_key("/clip.mp4", "what happens?")

        # Without a bare question, selection falls back to the rendered prompt.
        class _PromptOnlyDataset(_StubVideoDataset):
            def _record(self, idx):
                return VideoRecord("/clip.mp4", "describe the video", "a caption")

        ds2 = _PromptOnlyDataset(["1"], ["x"], max_frames=4, max_text_len=48)
        ds2._candidate_spec = CandidatePoolSpec(candidate_frames=4)
        assert ds2[0]["query"] == "describe the video"

    def test_non_empty_query_ships_full_pool(self, monkeypatch):
        # A real query must NOT be pre-strided: the scorer needs the full pool.
        frames = _frames(6)
        monkeypatch.setattr(
            vd, "decode_candidate_pool", lambda path, **k: (frames, [float(i) for i in range(6)])
        )
        ds = _StubVideoDataset(
            ["1"], ["an answer"], max_frames=4, prompt="what happens?", max_text_len=24
        )
        ds._candidate_spec = CandidatePoolSpec(candidate_frames=8)
        item = ds[0]
        assert item["candidate_count"].item() == 6
        assert item["candidate_pixels"].shape == (6, 3, 16, 16)

    def test_fast_path_identity_with_full_pool_selection(self, monkeypatch):
        # The #10 identity proof: a worker pre-strided (empty-query) sample fed
        # through VideoCollator + apply_frame_selection yields exactly the
        # pixel_values / frame_times the full-pool path yields for the same
        # inputs — the fast path only moves the stride, never changes frames.
        from kempnerforge.data.frame_selection import TopKSelector, apply_frame_selection

        frames = _frames(6)
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        monkeypatch.setattr(vd, "decode_candidate_pool", lambda path, **k: (frames, times))
        ds = _StubVideoDataset(["1"], ["a caption"], max_frames=4)  # prompt="" -> no query
        ds._candidate_spec = CandidatePoolSpec(candidate_frames=8)
        strided_sample = ds[0]  # worker fast path: ships 4 pre-strided frames

        # Full-pool sample for the same clip (what a worker without the fast
        # path would ship): all 6 frames + count 6.
        full_sample = dict(strided_sample)
        full_sample["candidate_pixels"] = torch.stack(
            [pil_to_uint8_tensor(f, 16) for f in frames]
        )
        full_sample["candidate_count"] = torch.tensor(6, dtype=torch.long)
        full_sample["candidate_times"] = torch.tensor(times, dtype=torch.float32)

        collator = VideoCollator(pad_id=0, max_text_len=8)
        # Empty query -> selection never touches the scorer (fallback paths).
        selector = TopKSelector(scorer=None)
        out_strided = apply_frame_selection(
            collator([strided_sample]), selector, max_frames=4, device="cpu"
        )
        out_full = apply_frame_selection(
            collator([full_sample]), selector, max_frames=4, device="cpu"
        )
        assert torch.equal(out_strided["pixel_values"], out_full["pixel_values"])
        assert torch.equal(out_strided["frame_times"], out_full["frame_times"])
        assert torch.equal(out_strided["frame_mask"], out_full["frame_mask"])
        assert out_full["frame_mask"].all()  # all 4 slots filled in both paths

    def test_decode_kwargs_come_from_spec(self, monkeypatch):
        captured = {}

        def _decode(path, *, candidate_frames, candidate_fps, sampling_policy):
            captured.update(cf=candidate_frames, cfps=candidate_fps, policy=sampling_policy)
            return _frames(2), [0.0, 1.0]

        monkeypatch.setattr(vd, "decode_candidate_pool", _decode)
        ds = _StubVideoDataset(["1"], ["a cat."], max_frames=4)
        ds._candidate_spec = CandidatePoolSpec(candidate_frames=12, candidate_fps=1.5)
        ds[0]
        assert captured == {"cf": 12, "cfps": 1.5, "policy": "uniform"}

    def test_seed_key_folds_in_query_for_per_question_independence(self, monkeypatch):
        # A QA corpus emits many questions per clip. Seeding on the path alone
        # would draw identical Gumbel noise for every question; the seed must fold
        # in the query so same-clip questions are independent yet reproducible.
        monkeypatch.setattr(vd, "decode_candidate_pool", lambda path, **k: (_frames(2), [0.0, 1.0]))

        class _TwoQuestionsOneClip(_StubVideoDataset):
            # Same clip path, different prompt per index (mimics a QA corpus).
            def _record(self, idx):
                from kempnerforge.data.video_dataset import VideoRecord

                return VideoRecord("/same/clip.mp4", ["q one?", "q two?"][idx], "answer")

        ds = _TwoQuestionsOneClip(["a", "b"], ["x", "y"], max_frames=4)
        ds._candidate_spec = CandidatePoolSpec(candidate_frames=4)
        seeds = [ds[0]["seed_key"], ds[1]["seed_key"]]
        assert seeds == ["/same/clip.mp4|q one?", "/same/clip.mp4|q two?"]
        assert seeds[0] != seeds[1]  # independent draws for the two questions

    def test_empty_prompt_ships_empty_query(self, monkeypatch):
        # A captioning sample with no prompt has no query to condition on (the
        # deliberate expectation that query-aware selection cannot help it).
        monkeypatch.setattr(vd, "decode_candidate_pool", lambda path, **k: (_frames(2), [0.0, 1.0]))
        ds = _StubVideoDataset(["1"], ["the caption"], max_frames=4)  # prompt=""
        ds._candidate_spec = CandidatePoolSpec(candidate_frames=4)
        assert ds[0]["query"] == ""  # empty prompt -> no query, not the caption

    def test_decode_error_degrades_to_empty_pool(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("decode blew up")

        monkeypatch.setattr(vd, "decode_candidate_pool", _boom)
        ds = _StubVideoDataset(["1"], ["a cat."], max_frames=4)
        ds._candidate_spec = CandidatePoolSpec(candidate_frames=4)
        item = ds[0]
        assert item["candidate_count"].item() == 0
        assert (item["candidate_pixels"] == 0).all()
        assert (item["labels"] == -100).all()  # trains on nothing

    def test_custom_mean_std_rejected_with_spec(self, monkeypatch):
        import transformers

        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda p: _MockTokenizer())
        )
        kwargs = dict(
            records=[],
            tokenizer_path="mock",
            max_text_len=8,
            max_frames=4,
            min_frames=2,
            fps=1.0,
            candidate_spec=CandidatePoolSpec(candidate_frames=8),
        )
        # Pool-mode frames are normalized downstream with the SigLIP defaults;
        # a custom mean/std would silently diverge, so it must be rejected.
        with pytest.raises(ValueError, match="image_mean"):
            vd.VideoQADataset(image_mean=(0.4, 0.4, 0.4), **kwargs)
        vd.VideoQADataset(**kwargs)  # defaults are fine


class TestBuilderThreadsSpec:
    def test_backward_compatible_without_kwarg(self, monkeypatch):
        # Existing call sites pass no candidate_spec; the builder still
        # dispatches and receives None (no pool mode).
        captured = {}

        def _fake_builder(video_config, tokenizer_path, max_text_len, candidate_spec=None):
            captured["spec"] = candidate_spec
            return object()

        from kempnerforge.config.registry import registry
        from kempnerforge.config.video import VideoConfig
        from kempnerforge.data.video_dataset import build_video_dataset

        monkeypatch.setattr(registry, "get_video_dataset", lambda name: _fake_builder)
        build_video_dataset(VideoConfig(), "gpt2", 16)  # no candidate_spec arg
        assert captured["spec"] is None

    def test_build_video_dataset_threads_spec(self, monkeypatch):
        captured = {}

        def _fake_builder(video_config, tokenizer_path, max_text_len, candidate_spec=None):
            captured["spec"] = candidate_spec
            return object()

        from kempnerforge.config.registry import registry

        monkeypatch.setattr(registry, "get_video_dataset", lambda name: _fake_builder)
        from kempnerforge.config.video import VideoConfig
        from kempnerforge.data.video_dataset import build_video_dataset

        spec = CandidatePoolSpec(candidate_frames=32)
        build_video_dataset(VideoConfig(), "gpt2", 16, spec)
        assert captured["spec"] is spec

    def test_build_video_data_derives_spec_from_config(self, monkeypatch):
        # build_video_data reduces the [frame_selector] config to a
        # CandidatePoolSpec — the dataset side never sees a selector object.
        captured = {}

        def _fake_builder(video_config, tokenizer_path, max_text_len, candidate_spec=None):
            captured["spec"] = candidate_spec
            return object()

        from kempnerforge.config.frame_selector import FrameSelectorConfig
        from kempnerforge.config.registry import registry
        from kempnerforge.config.video import VideoConfig

        monkeypatch.setattr(registry, "get_video_dataset", lambda name: _fake_builder)
        vd.build_video_data(
            VideoConfig(data_root="/fake/root"),
            "gpt2",
            16,
            frame_selector_config=FrameSelectorConfig(candidate_frames=64, candidate_fps=2.0),
        )
        assert captured["spec"] == CandidatePoolSpec(candidate_frames=64, candidate_fps=2.0)


class TestBaseHookGenerality:
    def test_second_dataset_style_gets_pool_mode(self, monkeypatch):
        # A genuine non-WebVid style (own record scheme: flat qa_<idx>.mp4
        # paths + a per-sample question) adopts pool mode with ZERO
        # frame-selection code of its own — it only threads candidate_spec
        # through super().__init__ and inherits the full pool contract.
        import transformers

        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda p: _MockTokenizer())
        )
        frames = _frames(3)
        monkeypatch.setattr(
            vd, "decode_candidate_pool", lambda path, **k: (frames, [0.0, 0.5, 1.0])
        )

        class _JsonlQADataset(vd.VideoQADataset):
            """Second style: parallel question/answer lists, flat path layout."""

            def __init__(self, questions, answers, video_dir, **kwargs):
                self._questions = questions
                self._answers = answers
                self._qa_video_dir = video_dir
                super().__init__(**kwargs)

            def __len__(self):
                return len(self._questions)

            def _record(self, idx):
                return vd.VideoRecord(
                    f"{self._qa_video_dir}/qa_{idx}.mp4",
                    self._questions[idx],
                    self._answers[idx],
                )

        ds = _JsonlQADataset(
            ["what happens?"],
            ["a dog runs."],
            "/qa/videos",
            tokenizer_path="mock",
            max_text_len=32,
            max_frames=8,
            min_frames=2,
            fps=1.0,
            frame_size=16,
            candidate_spec=CandidatePoolSpec(candidate_frames=6),
        )
        item = ds[0]
        # Full pool contract under the ragged shape (n, 3, S, S) uint8.
        assert item["candidate_pixels"].shape == (3, 3, 16, 16)
        assert item["candidate_pixels"].dtype == torch.uint8
        for j in range(3):  # per-frame uint8 conversion, exactly
            assert torch.equal(item["candidate_pixels"][j], pil_to_uint8_tensor(frames[j], 16))
        assert item["candidate_count"].item() == 3
        assert item["candidate_times"].tolist() == [0.0, 0.5, 1.0]
        assert item["query"] == "what happens?"  # the per-sample prompt
        assert item["seed_key"] == "/qa/videos/qa_0.mp4|what happens?"
        assert "pixel_values" not in item  # selection happens downstream

        # Without a spec the same style stays in the plain decode path.
        monkeypatch.setattr(
            vd, "decode_video_frames", lambda *a, **k: (frames, [0.0, 0.5, 1.0])
        )
        ds_off = _JsonlQADataset(
            ["what happens?"],
            ["a dog runs."],
            "/qa/videos",
            tokenizer_path="mock",
            max_text_len=32,
            max_frames=8,
            min_frames=2,
            fps=1.0,
            frame_size=16,
        )
        assert ds_off._candidate_spec is None  # off by default
        assert "candidate_pixels" not in ds_off[0]


class TestPromptPool:
    """``[video].prompt_pool``: one paraphrase per example, varying by epoch.

    A single fixed instruction teaches the model to tie that exact wording to
    "produce a caption"; a pool keeps the behaviour general. The draw must be a
    pure function of (index, epoch) so every rank and every resume agrees.
    """

    POOL = [f"prompt {i}" for i in range(8)]

    def _ds(self, pool):
        ds = vd.VideoQADataset.__new__(vd.VideoQADataset)
        ds._prompt = "fixed"
        ds._prompt_pool = list(pool)
        return ds

    def test_empty_pool_falls_back_to_scalar_prompt(self):
        assert self._ds([])._prompt_for(0) == "fixed"

    def test_pool_overrides_scalar_prompt(self):
        assert self._ds(self.POOL)._prompt_for(0) in self.POOL

    def test_varies_across_examples(self):
        ds = self._ds(self.POOL)
        assert len({ds._prompt_for(i) for i in range(200)}) == len(self.POOL)

    def test_same_index_same_epoch_is_stable(self):
        a, b = self._ds(self.POOL), self._ds(self.POOL)
        assert [a._prompt_for(i) for i in range(50)] == [b._prompt_for(i) for i in range(50)]

    def test_varies_across_epochs(self):
        ds = self._ds(self.POOL)
        e0 = [ds._prompt_for(i) for i in range(50)]
        ds.set_epoch(1)
        e1 = [ds._prompt_for(i) for i in range(50)]
        assert e0 != e1
        ds.set_epoch(0)
        assert [ds._prompt_for(i) for i in range(50)] == e0  # returning is stable

    def test_draw_is_roughly_uniform(self):
        ds = self._ds(self.POOL)
        counts = collections.Counter(ds._prompt_for(i) for i in range(4000))
        expected = 4000 / len(self.POOL)
        assert all(0.7 * expected < c < 1.3 * expected for c in counts.values())
