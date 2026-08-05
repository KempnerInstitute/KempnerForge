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
# Query-aware frame selection (base-class hook, dataset-agnostic)
# ---------------------------------------------------------------------------


class _FakeSelector:
    """Records the query/k/seed_key and returns a fixed subset of indices."""

    def __init__(self, indices):
        self._indices = indices
        self.calls = []

    def select(self, frames, times, query, k, *, seed_key=None):
        self.calls.append({"n": len(frames), "query": query, "k": k, "seed_key": seed_key})
        return self._indices


class TestGetItemWithSelector:
    def _select_stub(self, indices):
        # Mirror select_video_frames: decode a pool, subset to the given indices.
        def _fn(path, query, selector, k, *, sampling_policy="uniform", seed_key=None):
            selector.select(_dummy(len(indices) + 2), [], query, k, seed_key=seed_key)
            frames = _frames(len(indices))
            return frames, [float(i) for i in range(len(indices))]

        return _fn

    def test_uses_decode_clip_and_records_query_seed(self, monkeypatch):
        sel = _FakeSelector([0, 1, 2])
        captured = {}

        def _select(path, query, selector, k, *, sampling_policy="uniform", seed_key=None):
            captured.update(query=query, k=k, seed_key=seed_key, policy=sampling_policy)
            return _frames(3), [0.0, 1.0, 2.0]

        monkeypatch.setattr(vd, "select_video_frames", _select)
        # The query is the sample's prompt (the input-time question/instruction),
        # never the caption/target; the seed is (clip path, query) so multiple
        # questions per clip draw independently (see the per-question test below).
        ds = _StubVideoDataset(["7788"], ["a caption"], max_frames=8, prompt="describe:")
        ds._frame_selector = sel
        item = ds[0]
        assert captured["query"] == "describe:"  # the prompt, not the caption
        assert captured["k"] == 8  # max_frames
        assert captured["seed_key"] == f"{ds._video_path('7788')}|describe:"
        assert item["frame_mask"][:3].all() and not item["frame_mask"][3:].any()
        assert item["frame_times"][:3].tolist() == [0.0, 1.0, 2.0]

    def test_seed_key_folds_in_query_for_per_question_independence(self, monkeypatch):
        # A QA corpus emits many questions per clip. Seeding on the path alone
        # would draw identical Gumbel noise for every question; the seed must fold
        # in the query so same-clip questions are independent yet reproducible.
        seeds: list[str] = []

        def _select(path, query, selector, k, *, sampling_policy="uniform", seed_key=None):
            seeds.append(seed_key)
            return _frames(2), [0.0, 1.0]

        monkeypatch.setattr(vd, "select_video_frames", _select)

        class _TwoQuestionsOneClip(_StubVideoDataset):
            # Same clip path, different prompt per index (mimics a QA corpus).
            def _record(self, idx):
                from kempnerforge.data.video_dataset import VideoRecord

                return VideoRecord("/same/clip.mp4", ["q one?", "q two?"][idx], "answer")

        ds = _TwoQuestionsOneClip(["a", "b"], ["x", "y"], max_frames=4)
        ds._frame_selector = _FakeSelector([0, 1])
        ds[0]
        ds[1]
        assert seeds == ["/same/clip.mp4|q one?", "/same/clip.mp4|q two?"]
        assert seeds[0] != seeds[1]  # independent draws for the two questions

    def test_empty_prompt_gives_no_query(self, monkeypatch):
        # A captioning sample with no prompt has no query to condition on (the
        # deliberate expectation that query-aware selection cannot help it).
        captured = {}

        def _select(path, query, selector, k, *, sampling_policy="uniform", seed_key=None):
            captured["query"] = query
            return _frames(2), [0.0, 1.0]

        monkeypatch.setattr(vd, "select_video_frames", _select)
        ds = _StubVideoDataset(["1"], ["the caption"], max_frames=4)  # prompt=""
        ds._frame_selector = _FakeSelector([0, 1])
        ds[0]
        assert captured["query"] is None  # empty prompt -> no query, not the caption

    def test_selection_error_degrades_to_empty_clip(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("scoring blew up")

        monkeypatch.setattr(vd, "select_video_frames", _boom)
        ds = _StubVideoDataset(["1"], ["a cat."], max_frames=4)
        ds._frame_selector = _FakeSelector([0])
        item = ds[0]
        assert not item["frame_mask"].any()
        assert (item["labels"] == -100).all()

    def test_scorer_unavailable_propagates(self, monkeypatch):
        from kempnerforge.data.frame_selection import ScorerUnavailableError

        def _unavail(*a, **k):
            raise ScorerUnavailableError("prefetch the model")

        monkeypatch.setattr(vd, "select_video_frames", _unavail)
        ds = _StubVideoDataset(["1"], ["a cat."], max_frames=4)
        ds._frame_selector = _FakeSelector([0])
        with pytest.raises(ScorerUnavailableError):
            ds[0]  # systemic misconfig must not be silently masked


class TestBuilderThreadsSelector:
    def test_backward_compatible_without_kwarg(self, monkeypatch):
        # Existing call sites pass no frame_selector; the builder still
        # dispatches and receives None (no selector).
        captured = {}

        def _fake_builder(video_config, tokenizer_path, max_text_len, frame_selector=None):
            captured["sel"] = frame_selector
            return object()

        from kempnerforge.config.registry import registry
        from kempnerforge.config.video import VideoConfig
        from kempnerforge.data.video_dataset import build_video_dataset

        monkeypatch.setattr(registry, "get_video_dataset", lambda name: _fake_builder)
        build_video_dataset(VideoConfig(), "gpt2", 16)  # no frame_selector arg
        assert captured["sel"] is None

    def test_build_video_dataset_threads_prebuilt_selector(self, monkeypatch):
        # build_video_dataset forwards the prebuilt frame_selector object to the
        # builder (the object is now constructed once at the build_video_data seam).
        captured = {}

        def _fake_builder(video_config, tokenizer_path, max_text_len, frame_selector=None):
            captured["sel"] = frame_selector
            return object()

        from kempnerforge.config.registry import registry

        monkeypatch.setattr(registry, "get_video_dataset", lambda name: _fake_builder)
        from kempnerforge.config.video import VideoConfig
        from kempnerforge.data.video_dataset import build_video_dataset

        sentinel = object()
        build_video_dataset(VideoConfig(), "gpt2", 16, frame_selector=sentinel)
        assert captured["sel"] is sentinel


class _StubQADataset(vd.VideoDataset):
    """A non-WebVid dataset that adopts selection via the base hook alone.

    Proves the base-class ``_init_frame_selector`` / ``_decode_clip`` path is
    dataset-agnostic: this class has no frame-selection code of its own, uses a
    question (not a caption) as its query, and a different seed key.
    """

    def __init__(self, question, frame_selector=None):
        self._question = question
        self._init_frame_selector(frame_selector)

    def probe(self, path, seed_key):
        # The question is the query (the input-time prompt), never a target.
        return self._decode_clip(
            path,
            self._question,
            fps=1.0,
            min_frames=2,
            max_frames=4,
            sampling_policy="uniform",
            seed_key=seed_key,
        )


class TestBaseHookGenerality:
    def test_second_dataset_style_gets_selection(self, monkeypatch):
        captured = {}

        def _select(path, query, selector, k, *, sampling_policy="uniform", seed_key=None):
            captured.update(query=query, k=k, seed_key=seed_key)
            return _frames(3), [0.0, 1.0, 2.0]

        monkeypatch.setattr(vd, "select_video_frames", _select)
        ds = _StubQADataset("what color is the car?")
        ds._frame_selector = _FakeSelector([0, 1, 2])  # simulate a built selector
        frames, times = ds.probe("clip.mp4", seed_key="qa-42")
        assert captured == {"query": "what color is the car?", "k": 4, "seed_key": "qa-42"}
        assert len(frames) == 3

    def test_no_selector_falls_back_to_plain_decode(self, monkeypatch):
        monkeypatch.setattr(vd, "decode_video_frames", lambda *a, **k: _decoded(4))
        ds = _StubQADataset("q")  # no selector configured
        frames, times = ds.probe("clip.mp4", seed_key="x")
        assert len(frames) == 4  # plain uniform decode path


def _dummy(n: int) -> list[int]:
    return list(range(n))
