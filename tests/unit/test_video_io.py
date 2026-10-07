"""Unit tests for video frame sampling and decoding."""

from __future__ import annotations

import importlib.util
import logging
import os
from fractions import Fraction
from types import SimpleNamespace

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
_HEVC_AVAILABLE = _encoder_available("libx265")
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


def _write_mp4(
    path, n_frames: int, size: int = 32, fps: int = 10, gop_size: int | None = None
) -> None:
    """Encode a tiny solid-color clip with PyAV (av is a hard dependency).

    ``gop_size`` pins the keyframe cadence (frame 0, then every ``gop_size``
    frames), which the seek tests rely on; ``None`` keeps encoder defaults.
    """
    import av
    import numpy as np

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        if gop_size is not None:
            # Scene-change detection must be suppressed alongside gop_size:
            # the changing gray otherwise promotes every frame to a keyframe.
            stream.codec_context.gop_size = gop_size
            stream.codec_context.options = {"sc_threshold": "1000000000"}
        for i in range(n_frames):
            arr = np.full((size, size, 3), (i * 17) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush
            container.mux(packet)


def _write_h264_mp4(
    path, n_frames: int, size: int = 64, fps: int = 10, gop_size: int = 12, open_gop: bool = False
) -> None:
    """Encode an H.264 clip with B-frames (and optionally open GOPs).

    Produces the codec features the mpeg4 fixture cannot: presentation
    reordering (pts != dts) and, with ``open_gop``, leading B-frames that
    reference across GOP boundaries. ``b-adapt=0`` forces a fixed B-frame
    pattern and a moving stripe gives the encoder real motion.
    """
    import av
    import numpy as np

    params = f"keyint={gop_size}:min-keyint={gop_size}:scenecut=0:bframes=2:b-adapt=0"
    if open_gop:
        params += ":open-gop=1"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.codec_context.options = {"x264-params": params}
        for i in range(n_frames):
            arr = np.full((size, size, 3), (i * 7) % 200, dtype=np.uint8)
            arr[:, (i * 5) % size] = 255  # moving stripe: motion for B-frames
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush
            container.mux(packet)


def _serial_reference(path, fps: float, min_frames: int, max_frames: int) -> list:
    """Ground-truth frames via the serial reference pass (bypasses seeking)."""
    import av

    from kempnerforge.data.video_io import _decode_serial, _video_duration_seconds

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration_s = _video_duration_seconds(stream, container)
        targets = sample_timestamps(duration_s, fps, min_frames, max_frames)
        return _decode_serial(container, stream, targets)


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


def _write_shifted_mp4(src, dst, offset_s: float) -> None:
    """Remux ``src`` with every packet timestamp shifted later by ``offset_s``."""
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
    def test_start_offset_selects_same_frames(self, tmp_path, offset_s):
        from kempnerforge.data.video_io import decode_video_frames

        src = tmp_path / "src.mp4"
        shifted = tmp_path / "shifted.mp4"
        _write_indexed_clip(src, n_frames=20, fps=10)  # 2 s
        _write_shifted_mp4(src, shifted, offset_s)
        assert _first_frame_time(shifted) == pytest.approx(offset_s)
        got = decode_video_frames(str(shifted), fps=2.0, min_frames=4, max_frames=4)
        ref = decode_video_frames(str(src), fps=2.0, min_frames=4, max_frames=4)
        # Targets [0, 2/3, 4/3, 2] s; the last sits past the final frame (1.9 s).
        assert [_frame_index(f) for f in got] == [0, 7, 14, 19]
        assert [f.tobytes() for f in got] == [f.tobytes() for f in ref]

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


# ---------------------------------------------------------------------------
# seek-based decoding (parity with the serial reference, guards, fallback)
# ---------------------------------------------------------------------------


def _assert_seek_parity(path, monkeypatch, *, fps, min_frames, max_frames):
    """Seek-path frames equal the serial reference, with no fallback to serial.

    A silent fallback would make parity trivially true, so ``_decode_serial``
    is spied on. Frames are compared by bytes, not identity: the seek path
    reuses one image for targets that share a frame.
    """
    import kempnerforge.data.video_io as video_io

    expected = _serial_reference(path, fps, min_frames, max_frames)
    fell_back = []
    original = video_io._decode_serial

    def _spy(*args, **kwargs):
        fell_back.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(video_io, "_decode_serial", _spy)
    got = video_io.decode_video_frames(
        str(path), fps=fps, min_frames=min_frames, max_frames=max_frames
    )
    assert not fell_back, "seek path unexpectedly fell back to serial decode"
    assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]
    return got


class _CountingContainer:
    """Wraps an ``av`` container, counting seeks and decoded frames."""

    def __init__(self, container) -> None:
        self._container = container
        self.seeks = 0
        self.decoded = 0

    def __enter__(self):
        self._container.__enter__()
        return self

    def __exit__(self, *exc):
        return self._container.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._container, name)

    def seek(self, *args, **kwargs):
        self.seeks += 1
        return self._container.seek(*args, **kwargs)

    def decode(self, *args, **kwargs):
        for frame in self._container.decode(*args, **kwargs):
            self.decoded += 1
            yield frame


def _run_direct(path, fn, targets):
    """Run ``_decode_seek`` or ``_decode_serial`` on explicit targets; no fallback."""
    import av

    with av.open(str(path)) as raw:
        container = _CountingContainer(raw)
        stream = raw.streams.video[0]
        stream.thread_type = "AUTO"
        return fn(container, stream, list(targets)), container


def _mark_every_packet_key(src, dst) -> None:
    """Remux ``src`` with every packet flagged as a keyframe, as an index without a sync table."""
    import av

    with av.open(str(src)) as ic, av.open(str(dst), mode="w") as oc:
        istream = ic.streams.video[0]
        ostream = oc.add_stream_from_template(istream)
        for packet in ic.demux(istream):
            if packet.pts is None:
                continue
            packet.is_keyframe = True
            packet.stream = ostream
            oc.mux(packet)


def _damage_packet(src, dst, at_s: float) -> None:
    """Copy ``src`` to ``dst`` with the packet presented at ``at_s`` overwritten by 0xFF."""
    import shutil

    import av

    shutil.copyfile(src, dst)
    with av.open(str(src)) as container:
        stream = container.streams.video[0]
        packets = [p for p in container.demux(stream) if p.size]
        packet = min(packets, key=lambda p: abs(p.pts * stream.time_base - at_s))
        pos, size = packet.pos, packet.size
    with open(dst, "r+b") as fh:
        fh.seek(pos)
        fh.write(b"\xff" * size)


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestKeyframeFixture:
    """gop_size must actually control the fixture's keyframe cadence."""

    def test_gop_size_controls_keyframe_cadence(self, tmp_path):
        import av

        path = tmp_path / "gop.mp4"
        _write_mp4(path, n_frames=40, gop_size=12)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            packets = [p for p in container.demux(stream) if p.pts is not None]
        assert [i for i, p in enumerate(packets) if p.is_keyframe] == [0, 12, 24, 36]


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestSeekMatchesSerial:
    """The seek path must select byte-identical frames to the serial pass."""

    def test_sparse_targets_long_clip(self, tmp_path, monkeypatch):
        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=200, gop_size=12)  # 20s, 17 GOPs
        _assert_seek_parity(path, monkeypatch, fps=2.0, min_frames=1, max_frames=4)

    def test_dense_targets_min_eq_max(self, tmp_path, monkeypatch):
        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=40, gop_size=12)  # 4s
        _assert_seek_parity(path, monkeypatch, fps=8.0, min_frames=16, max_frames=16)

    def test_duplicate_targets_dense_short_clip(self, tmp_path, monkeypatch):
        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=10, gop_size=5)  # 1s; 16 targets over 10 frames
        _assert_seek_parity(path, monkeypatch, fps=2.0, min_frames=16, max_frames=16)

    def test_single_keyframe_clip(self, tmp_path, monkeypatch):
        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=60, gop_size=999)  # one group: never seeks
        _assert_seek_parity(path, monkeypatch, fps=2.0, min_frames=4, max_frames=8)

    def test_short_clip_tail_fill(self, tmp_path, monkeypatch):
        path = tmp_path / "short.mp4"
        _write_mp4(path, n_frames=3)  # shorter than min_frames -> tail-fill
        _assert_seek_parity(path, monkeypatch, fps=2.0, min_frames=4, max_frames=8)

    @pytest.mark.parametrize("offset_s", [0.5, 5.0])
    def test_start_time_shifted_clip(self, tmp_path, monkeypatch, offset_s):
        src = tmp_path / "src.mp4"
        dst = tmp_path / "shifted.mp4"
        _write_mp4(src, n_frames=200, gop_size=12)
        _write_shifted_mp4(src, dst, offset_s=offset_s)
        # Seeks go to the first frame's time plus the target.
        got = _assert_seek_parity(dst, monkeypatch, fps=2.0, min_frames=1, max_frames=4)
        ref = _serial_reference(src, 2.0, 1, 4)
        assert [f.tobytes() for f in got] == [f.tobytes() for f in ref]


@pytest.mark.skipif(not _H264_AVAILABLE, reason="requires a PyAV build with the libx264 encoder")
class TestSeekMatchesSerialH264:
    """Parity on the codec shape of real data: H.264 with B-frames.

    The mpeg4 fixtures above never exercise presentation reordering; real
    clips are mostly H.264, where it always occurs. ``open_gop`` adds
    leading B-frames that are dropped after a mid-stream seek.
    """

    def test_fixture_has_b_frames_and_reordering(self, tmp_path):
        import av
        from av.video.frame import PictureType

        path = tmp_path / "h264.mp4"
        _write_h264_mp4(path, n_frames=60, open_gop=True)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            packets = [p for p in container.demux(stream) if p.pts is not None]
        # B-frames are stored out of presentation order: dts != pts somewhere.
        assert any(p.dts is not None and p.dts != p.pts for p in packets)
        with av.open(str(path)) as container:
            types = {
                PictureType(int(f.pict_type)).name
                for f in container.decode(container.streams.video[0])
            }
        assert "B" in types

    @pytest.mark.parametrize("open_gop", [False, True])
    def test_sparse_targets_long_clip(self, tmp_path, monkeypatch, open_gop):
        path = tmp_path / "clip.mp4"
        _write_h264_mp4(path, n_frames=200, open_gop=open_gop)  # 20s, 1.2s GOPs
        _assert_seek_parity(path, monkeypatch, fps=2.0, min_frames=1, max_frames=4)

    @pytest.mark.parametrize("open_gop", [False, True])
    def test_dense_targets_min_eq_max(self, tmp_path, monkeypatch, open_gop):
        path = tmp_path / "clip.mp4"
        _write_h264_mp4(path, n_frames=60, open_gop=open_gop)  # 6s
        _assert_seek_parity(path, monkeypatch, fps=8.0, min_frames=16, max_frames=16)

    @pytest.mark.parametrize("open_gop", [False, True])
    def test_explicit_targets_straddle_gop_boundaries(self, tmp_path, open_gop):
        # Targets just after each keyframe time (keyint=12 @ 10fps -> 1.2s
        # GOPs), where open-GOP leading B-frames sit; drives _decode_seek
        # directly so a serial fallback cannot mask a divergence.
        from kempnerforge.data.video_io import _decode_seek, _decode_serial

        path = tmp_path / "clip.mp4"
        _write_h264_mp4(path, n_frames=120, open_gop=open_gop)
        targets = [0.05, 2.45, 4.85, 7.25, 9.65, 11.9]
        got, _ = _run_direct(path, _decode_seek, targets)
        expected, _ = _run_direct(path, _decode_serial, targets)
        assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]

    @pytest.mark.parametrize("pre_s", [0.05, 0.1, 0.15, 0.2, 0.25])
    def test_open_gop_targets_before_a_keyframe(self, tmp_path, monkeypatch, pre_s):
        """A target in the leading B-frames before an open-GOP keyframe: those frames
        are dropped after a seek, so the seek must either match or report itself."""
        from kempnerforge.data.video_io import _decode_seek, _decode_serial, _SeekUnreliableError

        path = tmp_path / "clip.mp4"
        _write_h264_mp4(path, n_frames=200, open_gop=True)
        targets = [0.0] + [3.6 * m - pre_s for m in (1, 2, 3, 4)]  # 3 GOPs apart: seeks
        expected, _ = _run_direct(path, _decode_serial, targets)
        try:
            got, counter = _run_direct(path, _decode_seek, targets)
        except _SeekUnreliableError as e:
            assert "landed" in str(e)
        else:
            assert counter.seeks > 0
            assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]
        _assert_seek_parity(path, monkeypatch, fps=2.0, min_frames=1, max_frames=4)


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestSeekFrameIdentity:
    """Targets must map to the expected source frames, not just the right count."""

    def test_picks_expected_source_frames(self, tmp_path):
        import numpy as np

        from kempnerforge.data.video_io import decode_video_frames

        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=20, fps=10)  # 2s; frame i is solid (i*17)%256 gray
        frames = decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=4)
        assert len(frames) == 4
        # Targets [0, 2/3, 4/3, 2.0] -> first frame at/after each: 0, 7, 14;
        # the final target sits past the last PTS -> tail-fills with frame 19.
        expected = [0, 7 * 17, 14 * 17, (19 * 17) % 256]
        means = [float(np.asarray(f.convert("L")).mean()) for f in frames]
        for got, want in zip(means, expected, strict=True):
            assert abs(got - want) <= 4.0  # mpeg4 encode/decode roundtrip tolerance


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestSeekFirstGroup:
    """The first targets are decoded from the start of the stream, never after a seek."""

    def test_raw_mpeg4_with_b_frames(self, tmp_path):
        """A raw MPEG-4 stream with B-frames: seeks go by decode time, and the keyframe
        shown at 0.3 s is stored at decode time 0, so a seek to 0 s lands there."""
        import av

        from kempnerforge.data.video_io import _decode_seek, _decode_serial, decode_video_frames

        path = tmp_path / "clip.m4v"
        options = {"g": "3", "bf": "2", "sc_threshold": "1000000000"}
        _write_indexed_clip(path, n_frames=60, fmt="m4v", codec_options=options)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(0, stream=stream, backward=True, any_frame=False)
            assert next(container.decode(stream)).time == pytest.approx(0.3)
        targets = [0.0, 0.15, 0.25]
        got, counter = _run_direct(path, _decode_seek, targets)
        expected, _ = _run_direct(path, _decode_serial, targets)
        assert [_frame_index(f) for f in got] == [0, 2, 3]
        assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]
        assert counter.seeks == 0
        frames = decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=16)
        assert [_frame_index(f) for f in frames] == [0]


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestSeekCodecShapes:
    """Parity on stream shapes where seeking is hard; a guard may route them to serial."""

    @pytest.mark.skipif(not _HEVC_AVAILABLE, reason="requires the libx265 encoder")
    @pytest.mark.parametrize("pre_s", [0.05, 0.1, 0.15, 0.2])
    def test_hevc_open_gop_targets_before_a_keyframe(self, tmp_path, monkeypatch, pre_s):
        from kempnerforge.data.video_io import _decode_seek, _decode_serial, _SeekUnreliableError

        path = tmp_path / "hevc.mp4"
        params = "keyint=12:min-keyint=12:scenecut=0:bframes=2:b-adapt=0:open-gop=1:log-level=error"
        _write_indexed_clip(
            path, n_frames=200, codec="libx265", codec_options={"x265-params": params}
        )
        targets = [0.0] + [3.6 * m - pre_s for m in (1, 2, 3, 4)]
        expected, _ = _run_direct(path, _decode_serial, targets)
        try:
            got, counter = _run_direct(path, _decode_seek, targets)
        except _SeekUnreliableError as e:
            assert "landed" in str(e)
        else:
            assert counter.seeks > 0
            assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]
        _assert_seek_parity(path, monkeypatch, fps=2.0, min_frames=1, max_frames=4)

    @pytest.mark.skipif(not _H264_AVAILABLE, reason="requires the libx264 encoder")
    def test_intra_refresh(self, tmp_path, monkeypatch):
        """Periodic intra refresh instead of keyframes: no clean point to seek to."""
        path = tmp_path / "intra_refresh.mp4"
        params = "keyint=30:intra-refresh=1:bframes=0:scenecut=0"
        _write_indexed_clip(
            path, n_frames=150, codec="libx264", codec_options={"x264-params": params}
        )
        _assert_seek_parity(path, monkeypatch, fps=2.0, min_frames=1, max_frames=4)
        _assert_seek_parity(path, monkeypatch, fps=2.0, min_frames=4, max_frames=16)

    @pytest.mark.skipif(not _H264_AVAILABLE, reason="requires the libx264 encoder")
    def test_mpegts_falls_back_on_inexact_seek(self, tmp_path, monkeypatch):
        """MPEG-TS has no index, so a seek lands near its target, here past it."""
        import kempnerforge.data.video_io as video_io

        path = tmp_path / "clip.ts"
        params = "keyint=12:min-keyint=12:scenecut=0:bframes=2"
        _write_indexed_clip(
            path, n_frames=200, codec="libx264", fmt="mpegts", codec_options={"x264-params": params}
        )
        expected = _serial_reference(path, 2.0, 4, 8)
        reasons = []
        monkeypatch.setattr(video_io, "_log_fallback_once", reasons.append)
        got = video_io.decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8)
        assert reasons == ["_SeekUnreliableError"]
        assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]

    def test_index_listing_every_frame_as_a_seek_point(self, tmp_path, monkeypatch):
        """A seek then lands on a P-frame, which decodes without its reference frame."""
        import kempnerforge.data.video_io as video_io

        src = tmp_path / "src.mp4"
        dst = tmp_path / "every_frame_key.mp4"
        options = {"g": "30", "sc_threshold": "1000000000"}
        _write_indexed_clip(src, n_frames=300, codec_options=options)  # 30 s
        _mark_every_packet_key(src, dst)
        with pytest.raises(video_io._SeekUnreliableError):
            _run_direct(dst, video_io._decode_seek, [0.0, 10.0])
        expected = _serial_reference(dst, 2.0, 1, 4)
        reasons = []
        monkeypatch.setattr(video_io, "_log_fallback_once", reasons.append)
        got = video_io.decode_video_frames(str(dst), fps=2.0, min_frames=1, max_frames=4)
        assert reasons == ["_SeekUnreliableError"]
        assert [_frame_index(f) for f in got] == [0, 100, 200, 299 % 256]
        assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]

    @pytest.mark.skipif(not _H264_AVAILABLE, reason="requires the libx264 encoder")
    def test_raw_h264_falls_back(self, tmp_path, monkeypatch):
        """A raw H.264 stream has no timestamps to seek by."""
        import kempnerforge.data.video_io as video_io

        path = tmp_path / "clip.h264"
        _write_indexed_clip(path, n_frames=30, codec="libx264", fmt="h264")
        reasons = []
        monkeypatch.setattr(video_io, "_log_fallback_once", reasons.append)
        frames = video_io.decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8)
        assert reasons == ["_SeekUnreliableError"]
        assert [_frame_index(f) for f in frames] == [0]

    @pytest.mark.skipif(not _H264_AVAILABLE, reason="requires the libx264 encoder")
    def test_variable_frame_rate(self, tmp_path, monkeypatch):
        from fractions import Fraction

        import av
        import numpy as np

        path = tmp_path / "vfr.mp4"
        gaps = np.random.default_rng(0).choice([16, 33, 50, 100, 250], size=150)
        pts = np.concatenate([[0], np.cumsum(gaps)[:-1]]).tolist()
        params = "keyint=20:min-keyint=20:scenecut=0:bframes=2"
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width = stream.height = 64
            stream.pix_fmt = "yuv420p"
            stream.codec_context.time_base = Fraction(1, 1000)
            stream.time_base = Fraction(1, 1000)
            stream.codec_context.options = {"x264-params": params}
            for i, t in enumerate(pts):
                frame = av.VideoFrame.from_ndarray(_index_frame(i), format="rgb24")
                frame.pts = int(t)
                frame.time_base = Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        for cfg in ((2.0, 1, 4), (2.0, 4, 16), (4.0, 2, 48)):
            _assert_seek_parity(path, monkeypatch, fps=cfg[0], min_frames=cfg[1], max_frames=cfg[2])

    @pytest.mark.skipif(not _VP9_AVAILABLE, reason="requires the libvpx-vp9 encoder")
    def test_vp9_hidden_alt_ref_frames(self, tmp_path, monkeypatch):
        path = tmp_path / "clip.webm"
        options = {"auto-alt-ref": "1", "lag-in-frames": "16", "g": "30"}
        options |= {"deadline": "realtime", "cpu-used": "8"}
        _write_indexed_clip(path, n_frames=150, codec="libvpx-vp9", codec_options=options)
        _assert_seek_parity(path, monkeypatch, fps=2.0, min_frames=1, max_frames=4)
        _assert_seek_parity(path, monkeypatch, fps=2.0, min_frames=4, max_frames=16)


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestSeekDecodeWork:
    """Seeking skips whole groups for sparse targets and never adds work for dense ones."""

    def _decode_counting(self, path, monkeypatch, **cfg):
        import av

        from kempnerforge.data.video_io import decode_video_frames

        real_open = av.open
        opened = []

        def _open(*args, **kwargs):
            opened.append(_CountingContainer(real_open(*args, **kwargs)))
            return opened[-1]

        monkeypatch.setattr(av, "open", _open)
        frames = decode_video_frames(str(path), **cfg)
        assert len(opened) == 1  # no fallback reopen
        return frames, opened[0]

    def test_sparse_targets_decode_a_fraction(self, tmp_path, monkeypatch):
        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=200, gop_size=12)  # 20 s, keyframes every 1.2 s
        frames, counter = self._decode_counting(
            path, monkeypatch, fps=2.0, min_frames=1, max_frames=4
        )
        assert len(frames) == 4
        # Each target costs at most its group plus the frame that ends it.
        assert counter.seeks == 3
        assert counter.decoded <= 4 * (12 + 1)

    def test_dense_targets_never_seek(self, tmp_path, monkeypatch):
        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=200, gop_size=12)
        frames, counter = self._decode_counting(
            path, monkeypatch, fps=2.0, min_frames=4, max_frames=16
        )
        assert len(frames) == 16
        assert counter.seeks == 0
        assert counter.decoded == 200


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestSeekDamage:
    """Damage in a stretch the seek skips is never decoded, unlike in a serial pass."""

    def _clips(self, tmp_path, damage_at_s):
        clean = tmp_path / "clean.mp4"
        damaged = tmp_path / "damaged.mp4"
        _write_mp4(clean, n_frames=300, gop_size=12)  # 30 s; targets 0, 10, 20, 30 s
        _damage_packet(clean, damaged, damage_at_s)
        return clean, damaged

    def test_damage_in_skipped_stretch_returns_frames(self, tmp_path):
        import av

        from kempnerforge.data.video_io import decode_video_frames

        clean, damaged = self._clips(tmp_path, damage_at_s=5.0)  # between the 0 s and 10 s groups
        with pytest.raises(av.FFmpegError):
            _serial_reference(damaged, 2.0, 1, 4)
        got = decode_video_frames(str(damaged), fps=2.0, min_frames=1, max_frames=4)
        ref = decode_video_frames(str(clean), fps=2.0, min_frames=1, max_frames=4)
        assert [f.tobytes() for f in got] == [f.tobytes() for f in ref]

    def test_damage_in_sampled_group_raises(self, tmp_path):
        import av

        from kempnerforge.data.video_io import decode_video_frames

        _, damaged = self._clips(tmp_path, damage_at_s=9.8)  # the 9.6 s group holds 10 s
        with pytest.raises(av.FFmpegError):
            decode_video_frames(str(damaged), fps=2.0, min_frames=1, max_frames=4)


class _FakeFrame:
    """Decoded-frame stand-in; ``to_image`` returns the frame's index."""

    def __init__(self, index: int, time: float | None, key_frame: bool) -> None:
        self.index = index
        self.time = time
        self.key_frame = key_frame

    def to_image(self) -> int:
        return self.index


class _FakeContainer:
    """A 10 fps stream with a keyframe every ``gop`` frames.

    ``seek`` moves to the last keyframe at or before the requested time, then
    ``land_late`` keyframes further; ``land_on_any_frame`` moves to the frame at
    that time instead, and ``empty_after_seek`` leaves nothing to decode.
    """

    def __init__(
        self,
        n_frames,
        gop,
        *,
        land_late=0,
        land_on_any_frame=False,
        empty_after_seek=False,
        timed=True,
    ):
        self.frames = [
            _FakeFrame(i, i / 10 if timed else None, i % gop == 0) for i in range(n_frames)
        ]
        self.gop = gop
        self.land_late = land_late
        self.land_on_any_frame = land_on_any_frame
        self.empty_after_seek = empty_after_seek
        self.pos = 0
        self.seeks: list[float] = []
        self.decoded = 0

    def seek(self, offset, *, stream, backward, any_frame):
        t = float(offset * stream.time_base)
        self.seeks.append(t)
        index = int(t * 10 + 1e-6)
        key = index if self.land_on_any_frame else index // self.gop * self.gop
        self.pos = len(self.frames) if self.empty_after_seek else key + self.land_late * self.gop

    def decode(self, stream):
        while self.pos < len(self.frames):
            self.pos += 1
            self.decoded += 1
            yield self.frames[self.pos - 1]


class TestSeekCursor:
    """The seek decisions and guards, on a simulated stream."""

    _STREAM = SimpleNamespace(time_base=Fraction(1, 1000))

    def test_seeks_only_over_whole_groups(self):
        from kempnerforge.data.video_io import _decode_seek, _decode_serial

        targets = [0.0, 0.5, 1.5, 5.0, 9.9]
        seek = _FakeContainer(100, gop=10)  # 10 s, a keyframe every second
        serial = _FakeContainer(100, gop=10)
        assert _decode_seek(seek, self._STREAM, targets) == [0, 5, 15, 50, 99]
        assert _decode_serial(serial, self._STREAM, targets) == [0, 5, 15, 50, 99]
        # 1.5 s is reached by decoding on through the 1 s keyframe; 5 s and 9.9 s
        # lie over a whole group away, so the decode seeks to their keyframes.
        assert seek.seeks == [5.0, 9.9]
        assert seek.decoded == 21 + 11 + 10
        assert serial.decoded == 100

    def test_seek_landing_past_its_target_raises(self):
        from kempnerforge.data.video_io import _decode_seek, _SeekUnreliableError

        container = _FakeContainer(100, gop=10, land_late=1)
        with pytest.raises(_SeekUnreliableError, match="seek to 5.000s landed at 6.000s"):
            _decode_seek(container, self._STREAM, [0.0, 5.0])

    def test_seek_landing_on_a_non_keyframe_raises(self):
        from kempnerforge.data.video_io import _decode_seek, _SeekUnreliableError

        container = _FakeContainer(100, gop=10, land_on_any_frame=True)
        with pytest.raises(_SeekUnreliableError, match="seek to 5.500s landed on a non-keyframe"):
            _decode_seek(container, self._STREAM, [0.0, 5.5])

    def test_nothing_decodable_after_a_seek_raises(self):
        from kempnerforge.data.video_io import _decode_seek, _SeekUnreliableError

        container = _FakeContainer(100, gop=10, empty_after_seek=True)
        with pytest.raises(_SeekUnreliableError, match="no frames decodable at or after 5.000s"):
            _decode_seek(container, self._STREAM, [0.0, 5.0])

    def test_frame_without_timestamp_raises(self):
        from kempnerforge.data.video_io import _decode_seek, _SeekUnreliableError

        container = _FakeContainer(10, gop=5, timed=False)
        with pytest.raises(_SeekUnreliableError, match="without a timestamp"):
            _decode_seek(container, self._STREAM, [0.0])

    def test_stream_without_time_base_raises(self):
        from kempnerforge.data.video_io import _decode_seek, _SeekUnreliableError

        with pytest.raises(_SeekUnreliableError, match="no time_base"):
            _decode_seek(_FakeContainer(10, gop=5), SimpleNamespace(time_base=None), [0.0])


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestSeekFallback:
    """Unreliable seeks must degrade to the serial pass, never to wrong frames."""

    def test_falls_back_to_serial_on_unreliable_seek(self, tmp_path, monkeypatch):
        import kempnerforge.data.video_io as video_io

        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=40, gop_size=12)
        expected = _serial_reference(path, 2.0, 4, 8)

        def _raise(container, stream, targets):
            raise video_io._SeekUnreliableError("test")

        monkeypatch.setattr(video_io, "_decode_seek", _raise)
        got = video_io.decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8)
        assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]

    def test_fallback_logged_once_per_cause(self, tmp_path, monkeypatch, caplog):
        import kempnerforge.data.video_io as video_io

        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=40, gop_size=12)

        def _raise(container, stream, targets):
            raise video_io._SeekUnreliableError("test")

        monkeypatch.setattr(video_io, "_decode_seek", _raise)
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        video_io._log_fallback_once.cache_clear()
        with caplog.at_level(logging.DEBUG, logger="kempnerforge.data.video_io"):
            for _ in range(3):
                video_io.decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8)
        fallback_lines = [r for r in caplog.records if "falling back" in r.message]
        assert len(fallback_lines) == 1

    def test_fallback_reopen_without_video_stream_returns_empty(self, tmp_path, monkeypatch):
        import av

        import kempnerforge.data.video_io as video_io

        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=40, gop_size=12)

        def _raise(container, stream, targets):
            raise video_io._SeekUnreliableError("test")

        # The second open (the fallback reopen) yields a container whose video
        # stream has vanished — the guard must return [] rather than crash.
        real_open = av.open
        opens = {"n": 0}

        class _NoVideoStreams:
            video = ()

        class _NoVideoContainer:
            streams = _NoVideoStreams()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def _flaky_open(p, *args, **kwargs):
            opens["n"] += 1
            if opens["n"] == 2:
                return _NoVideoContainer()
            return real_open(p, *args, **kwargs)

        monkeypatch.setattr(video_io, "_decode_seek", _raise)
        monkeypatch.setattr(av, "open", _flaky_open)
        assert video_io.decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8) == []

    @pytest.mark.skipif(not _H264_AVAILABLE, reason="requires the libx264 encoder")
    def test_landing_check_raises_on_dropped_leading_frames(self, tmp_path):
        """Seeking to 9.45 s lands on the open-GOP keyframe at 9.6 s: the leading
        B-frames before it, which serial selects from, are dropped."""
        from kempnerforge.data.video_io import _decode_seek, _decode_serial, _SeekUnreliableError

        path = tmp_path / "clip.mp4"
        _write_h264_mp4(path, n_frames=120, open_gop=True)
        frames, _ = _run_direct(path, _decode_serial, [0.0, 9.45, 9.55])
        # Serial takes the 9.5 s B-frame for 9.45 s, not the keyframe the seek reaches.
        assert frames[1].tobytes() != frames[2].tobytes()
        with pytest.raises(_SeekUnreliableError, match="seek to 9.450s landed at 9.600s"):
            _run_direct(path, _decode_seek, [0.0, 9.45])

    def test_audio_only_returns_empty(self, tmp_path):
        import av
        import numpy as np

        from kempnerforge.data.video_io import decode_video_frames

        path = tmp_path / "audio.wav"
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("pcm_s16le", rate=8000)
            samples = np.zeros((1, 800), dtype=np.int16)
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            frame.sample_rate = 8000
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        assert decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8) == []


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
