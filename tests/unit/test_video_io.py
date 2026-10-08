"""Unit tests for video frame sampling and decoding."""

from __future__ import annotations

import importlib.util
import os

import pytest

from kempnerforge.data.video_io import sample_timestamps

# A known-good WebVid clip on the Kempner testbed; the decode integration test
# is skipped when ``av`` or the data are unavailable (CI without either).
_WEBVID_CLIP = (
    "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/video/webvid-10m/"
    "raw/videos/train/21/2117/211794/21179416.mp4"
)
_AV_AVAILABLE = importlib.util.find_spec("av") is not None


def _encoder_available(name: str) -> bool:
    """Whether the installed PyAV build bundles the encoder ``name``."""
    if not _AV_AVAILABLE:
        return False
    import av

    return name in av.codecs_available


_H264_AVAILABLE = _encoder_available("libx264")
_VP9_AVAILABLE = _encoder_available("libvpx-vp9")


# ---------------------------------------------------------------------------
# sample_timestamps (pure policy, no decoder)
# ---------------------------------------------------------------------------


class TestSampleTimestamps:
    def test_zero_duration_returns_single_start(self):
        assert sample_timestamps(0.0, fps=2.0, min_frames=4, max_frames=16) == [0.0]

    def test_negative_duration_returns_single_start(self):
        assert sample_timestamps(-3.0, fps=2.0, min_frames=4, max_frames=16) == [0.0]

    def test_includes_first_and_last_frame(self):
        ts = sample_timestamps(10.0, fps=2.0, min_frames=4, max_frames=16)
        assert ts[0] == 0.0
        assert ts[-1] == pytest.approx(10.0)

    def test_strictly_increasing(self):
        ts = sample_timestamps(7.5, fps=2.0, min_frames=4, max_frames=16)
        assert all(b > a for a, b in zip(ts, ts[1:], strict=False))

    def test_caps_at_max_frames(self):
        # 100s * 2fps = 200 desired, capped to 16, uniformly over [0, 100].
        ts = sample_timestamps(100.0, fps=2.0, min_frames=4, max_frames=16)
        assert len(ts) == 16
        assert ts[-1] == pytest.approx(100.0)

    def test_target_rate_when_under_cap(self):
        # 2s * 2fps = 4 frames, within [4, 16].
        ts = sample_timestamps(2.0, fps=2.0, min_frames=4, max_frames=16)
        assert len(ts) == 4
        assert ts == pytest.approx([0.0, 2 / 3, 4 / 3, 2.0])

    def test_floors_at_min_frames(self):
        # 1s * 2fps = 2 desired, raised to min_frames=4.
        ts = sample_timestamps(1.0, fps=2.0, min_frames=4, max_frames=16)
        assert len(ts) == 4

    def test_single_frame_when_max_is_one(self):
        ts = sample_timestamps(5.0, fps=2.0, min_frames=1, max_frames=1)
        assert ts == [0.0]

    @pytest.mark.parametrize("fps", [0.0, -1.0])
    def test_bad_fps_raises(self, fps):
        with pytest.raises(ValueError, match="fps must be positive"):
            sample_timestamps(10.0, fps=fps, min_frames=4, max_frames=16)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="must be <="):
            sample_timestamps(10.0, fps=2.0, min_frames=8, max_frames=4)

    def test_min_below_one_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            sample_timestamps(10.0, fps=2.0, min_frames=0, max_frames=4)


# ---------------------------------------------------------------------------
# decode_video_frames (integration; needs av + the testbed data)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _AV_AVAILABLE or not os.path.exists(_WEBVID_CLIP),
    reason="requires the 'av' package and the WebVid testbed clip",
)
class TestDecodeVideoFramesIntegration:
    def test_decodes_pil_frames(self):
        from PIL import Image

        from kempnerforge.data.video_io import decode_video_frames

        frames = decode_video_frames(_WEBVID_CLIP, fps=2.0, min_frames=4, max_frames=8)
        assert 1 <= len(frames) <= 8
        assert all(isinstance(f, Image.Image) and f.mode == "RGB" for f in frames)

    def test_respects_max_frames(self):
        from kempnerforge.data.video_io import decode_video_frames

        frames = decode_video_frames(_WEBVID_CLIP, fps=8.0, min_frames=4, max_frames=4)
        assert len(frames) == 4

    def test_missing_file_raises(self):
        from kempnerforge.data.video_io import decode_video_frames

        with pytest.raises(Exception):  # noqa: B017,PT011 - any av/OS error is acceptable
            decode_video_frames("/no/such/video.mp4", fps=2.0, min_frames=4, max_frames=8)


def _write_mp4(path, n_frames: int, size: int = 32, fps: int = 10) -> None:
    """Encode a tiny solid-color clip with PyAV (av is a hard dependency)."""
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
        for packet in stream.encode():  # flush
            container.mux(packet)


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestDecodeSynthetic:
    """Decode a synthetic clip (no external data) — runs in CI since av is a dep."""

    def test_decodes_rgb_frames(self, tmp_path):
        from PIL import Image

        from kempnerforge.data.video_io import decode_video_frames

        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=20, fps=10)  # ~2s
        frames = decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8)
        assert 1 <= len(frames) <= 8
        assert all(isinstance(f, Image.Image) and f.mode == "RGB" for f in frames)

    def test_respects_max_frames(self, tmp_path):
        from kempnerforge.data.video_io import decode_video_frames

        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=40, fps=10)  # ~4s
        frames = decode_video_frames(str(path), fps=8.0, min_frames=4, max_frames=4)
        assert len(frames) == 4

    def test_short_clip_returns_frames(self, tmp_path):
        from kempnerforge.data.video_io import decode_video_frames

        path = tmp_path / "short.mp4"
        _write_mp4(path, n_frames=3, fps=10)  # shorter than min_frames request
        frames = decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8)
        assert len(frames) >= 1


def _index_frame(i: int, size: int = 64):
    """An RGB frame encoding ``i`` in binary: 8-px stripe ``b`` is white when bit ``b`` is set."""
    import numpy as np

    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for b in range(8):
        if (i >> b) & 1:
            arr[:, 8 * b : 8 * b + 8] = 255
    return arr


def _frame_index(img) -> int:
    """The index ``_index_frame`` encoded; stripe interiors survive lossy coding."""
    import numpy as np

    luma = np.asarray(img.convert("L"), dtype=np.float64)
    return sum(1 << b for b in range(8) if luma[:, 8 * b + 2 : 8 * b + 6].mean() > 128)


def _write_indexed_clip(
    path,
    n_frames: int,
    fps: int = 10,
    *,
    codec: str = "mpeg4",
    fmt: str | None = None,
    codec_options: dict | None = None,
    container_options: dict | None = None,
) -> None:
    """Encode ``n_frames`` frames, frame ``i`` showing ``_index_frame(i)``."""
    import av

    with av.open(str(path), mode="w", format=fmt, options=container_options or {}) as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        if codec_options:
            stream.codec_context.options = codec_options
        for i in range(n_frames):
            frame = av.VideoFrame.from_ndarray(_index_frame(i), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush
            container.mux(packet)


def _remux_shifted(src, dst, offset_s: float) -> None:
    """Remux ``src`` into ``dst`` (container from its suffix), timestamps ``offset_s`` later."""
    import av

    with av.open(str(src)) as ic, av.open(str(dst), mode="w") as oc:
        istream = ic.streams.video[0]
        ostream = oc.add_stream_from_template(istream)
        shift = round(offset_s / istream.time_base)
        for packet in ic.demux(istream):
            if packet.pts is None:
                continue
            packet.pts += shift
            if packet.dts is not None:
                packet.dts += shift
            packet.stream = ostream
            oc.mux(packet)


def _first_frame_time(path) -> float | None:
    """Presentation time of the first decoded frame (``None`` without a timestamp)."""
    import av

    with av.open(str(path)) as container:
        return next(container.decode(container.streams.video[0])).time


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestDecodeStartOffset:
    """Frame times count from the first decoded frame, since sample targets start at 0 s."""

    @pytest.mark.parametrize("offset_s", [0.0, 0.5, 5.0])
    @pytest.mark.parametrize(
        ("suffix", "codec"),
        [("mp4", "mpeg4"), ("mkv", "mpeg4"), ("webm", "libvpx-vp9"), ("flv", "flv")],
    )
    def test_start_offset_selects_same_frames(self, tmp_path, suffix, codec, offset_s):
        """MP4 reports a stream duration; MKV and WebM a container duration that ends at
        the last frame's end time, FLV one that spans the clip."""
        from kempnerforge.data.video_io import decode_video_frames

        if codec == "libvpx-vp9" and not _VP9_AVAILABLE:
            pytest.skip("requires the libvpx-vp9 encoder")
        src = tmp_path / f"src.{suffix}"
        base = tmp_path / f"base.{suffix}"
        shifted = tmp_path / f"shifted.{suffix}"
        _write_indexed_clip(src, n_frames=20, fps=10, codec=codec)  # 2 s
        _remux_shifted(src, base, 0.0)
        _remux_shifted(src, shifted, offset_s)
        assert _first_frame_time(shifted) == pytest.approx(offset_s)
        got = decode_video_frames(str(shifted), fps=2.0, min_frames=4, max_frames=4)
        ref = decode_video_frames(str(base), fps=2.0, min_frames=4, max_frames=4)
        assert [f.tobytes() for f in got] == [f.tobytes() for f in ref]
        if suffix != "flv":  # a remuxed FLV's span ends at its last frame's time
            # Targets [0, 2/3, 4/3, 2] s; the last sits past the final frame (1.9 s).
            assert [_frame_index(f) for f in got] == [0, 7, 14, 19]

    def test_b_frame_delay_keeps_selection(self, tmp_path):
        """MPEG-4 B-frames in AVI: the stream starts at 0 s but its first frame at 33 ms."""
        import av

        from kempnerforge.data.video_io import decode_video_frames

        path = tmp_path / "bframes.avi"
        _write_indexed_clip(path, n_frames=60, fps=30, fmt="avi", codec_options={"bf": "2"})
        with av.open(str(path)) as container:
            assert container.streams.video[0].start_time == 0
        assert _first_frame_time(path) == pytest.approx(1 / 30)
        frames = decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=4)
        assert [_frame_index(f) for f in frames] == [0, 20, 40, 59]

    @pytest.mark.skipif(not _H264_AVAILABLE, reason="requires the libx264 encoder")
    def test_h264_b_frame_delay_keeps_selection(self, tmp_path):
        """H.264 B-frames in MP4 without an edit list: the first frame is at 67 ms."""
        from kempnerforge.data.video_io import decode_video_frames

        path = tmp_path / "bframes.mp4"
        _write_indexed_clip(
            path,
            n_frames=60,
            fps=30,
            codec="libx264",
            codec_options={"x264-params": "bframes=2:b-adapt=0"},
            container_options={"use_editlist": "0"},
        )
        assert _first_frame_time(path) == pytest.approx(2 / 30)
        frames = decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=4)
        assert [_frame_index(f) for f in frames] == [0, 20, 40, 59]

    def test_zero_start_matches_absolute_times(self, tmp_path):
        """With the first frame at 0 s, each target takes the first frame whose
        absolute time is within 1 ms of or after it, as before."""
        import av

        from kempnerforge.data.video_io import _video_duration_seconds, decode_video_frames

        path = tmp_path / "clip.mp4"
        _write_indexed_clip(path, n_frames=45, fps=30, codec_options={"bf": "2"})
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            targets = sample_timestamps(_video_duration_seconds(stream, container), 8.0, 12, 12)
            decoded = [(f.time, f.to_image().tobytes()) for f in container.decode(stream)]
        assert decoded[0][0] == 0.0
        last = decoded[-1][1]
        expected = [next((b for t, b in decoded if t + 1e-3 >= tgt), last) for tgt in targets]
        frames = decode_video_frames(str(path), fps=8.0, min_frames=12, max_frames=12)
        assert [f.tobytes() for f in frames] == expected

    @pytest.mark.skipif(not _H264_AVAILABLE, reason="requires the libx264 encoder")
    def test_frames_without_timestamps_count_as_zero(self, tmp_path, monkeypatch):
        """A raw H.264 stream has no timestamps: the first frame takes the 0 s target and
        later targets fall back to the last frame."""
        from types import SimpleNamespace

        import kempnerforge.data.video_io as video_io

        path = tmp_path / "clip.h264"
        _write_indexed_clip(path, n_frames=20, fps=10, codec="libx264", fmt="h264")
        assert _first_frame_time(path) is None
        fixed = SimpleNamespace(get_sampling_policy=lambda name: lambda *args: [0.0, 0.5, 1.0])
        monkeypatch.setattr(video_io, "registry", fixed)
        frames = video_io.decode_video_frames(str(path), fps=2.0, min_frames=1, max_frames=4)
        assert [_frame_index(f) for f in frames] == [0, 19, 19]


class TestVideoDuration:
    """The clip span runs from the stream's start, where frame times start."""

    @pytest.mark.parametrize(
        ("format_name", "start_s", "span_s"),
        [
            ("matroska,webm", 5.0, 2.0),  # the segment ends at 7 s
            ("nut", 5.0, 2.0),
            ("flv", 5.0, 7.0),  # already a span
            ("matroska,webm", None, 7.0),  # no start to subtract
        ],
    )
    def test_container_duration(self, format_name, start_s, span_s):
        from fractions import Fraction
        from types import SimpleNamespace

        from kempnerforge.data.video_io import _video_duration_seconds

        stream = SimpleNamespace(
            duration=None,
            time_base=Fraction(1, 1000),
            start_time=None if start_s is None else round(start_s * 1000),
        )
        container = SimpleNamespace(duration=7_000_000, format=SimpleNamespace(name=format_name))
        assert _video_duration_seconds(stream, container) == pytest.approx(span_s)

    def test_stream_duration_is_used_as_is(self):
        from fractions import Fraction
        from types import SimpleNamespace

        from kempnerforge.data.video_io import _video_duration_seconds

        stream = SimpleNamespace(duration=2000, time_base=Fraction(1, 1000), start_time=5000)
        assert _video_duration_seconds(stream, SimpleNamespace(duration=7_000_000)) == 2.0


class TestSamplingPolicyRegistry:
    """The sampling-policy registry makes frame selection config-switchable."""

    def test_uniform_registered(self):
        from kempnerforge.config.registry import registry

        assert "uniform" in registry.list_sampling_policies()
        assert registry.get_sampling_policy("uniform") is sample_timestamps

    def test_unknown_policy_raises(self):
        from kempnerforge.config.registry import registry

        with pytest.raises(KeyError, match="sampling_policy"):
            registry.get_sampling_policy("bogus")
