"""Unit tests for video frame sampling and decoding."""

from __future__ import annotations

import importlib.util
import logging
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


def _libx264_available() -> bool:
    """Whether this PyAV build bundles the libx264 encoder (H.264 fixtures)."""
    if not _AV_AVAILABLE:
        return False
    import av

    return "libx264" in av.codecs_available


_H264_AVAILABLE = _libx264_available()


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


def _write_shifted_mp4(src, dst, offset_s: float = 5.0) -> None:
    """Remux ``src`` with every packet timestamp shifted forward by ``offset_s``."""
    import av

    with av.open(str(src)) as ic, av.open(str(dst), mode="w") as oc:
        istream = ic.streams.video[0]
        ostream = oc.add_stream_from_template(istream)
        shift = int(offset_s / istream.time_base)
        for packet in ic.demux(istream):
            if packet.pts is None:
                continue
            packet.pts += shift
            if packet.dts is not None:
                packet.dts += shift
            packet.stream = ostream
            oc.mux(packet)


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


# ---------------------------------------------------------------------------
# seek-based decoding (parity with the serial reference, guards, fallback)
# ---------------------------------------------------------------------------


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

    def _assert_parity(self, path, *, fps, min_frames, max_frames):
        from kempnerforge.data.video_io import decode_video_frames

        expected = _serial_reference(path, fps, min_frames, max_frames)
        got = decode_video_frames(str(path), fps=fps, min_frames=min_frames, max_frames=max_frames)
        assert len(got) == len(expected)
        # Compare bytes, not identity: the seek path reuses one PIL object for
        # an eps-group of duplicate targets where serial makes distinct copies.
        assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]

    def test_sparse_targets_long_clip(self, tmp_path):
        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=200, gop_size=12)  # 20s, 17 GOPs
        self._assert_parity(path, fps=2.0, min_frames=1, max_frames=4)

    def test_dense_targets_min_eq_max(self, tmp_path):
        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=40, gop_size=12)  # 4s
        self._assert_parity(path, fps=8.0, min_frames=16, max_frames=16)

    def test_duplicate_targets_dense_short_clip(self, tmp_path):
        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=10, gop_size=5)  # 1s; 16 targets over 10 frames
        self._assert_parity(path, fps=2.0, min_frames=16, max_frames=16)

    def test_single_keyframe_clip(self, tmp_path):
        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=60, gop_size=999)  # every seek lands on frame 0
        self._assert_parity(path, fps=2.0, min_frames=4, max_frames=8)

    def test_short_clip_tail_fill(self, tmp_path):
        path = tmp_path / "short.mp4"
        _write_mp4(path, n_frames=3)  # shorter than min_frames -> tail-fill
        self._assert_parity(path, fps=2.0, min_frames=4, max_frames=8)

    def test_start_time_shifted_clip(self, tmp_path):
        src = tmp_path / "src.mp4"
        dst = tmp_path / "shifted.mp4"
        _write_mp4(src, n_frames=20, gop_size=5)
        _write_shifted_mp4(src, dst, offset_s=5.0)
        self._assert_parity(dst, fps=2.0, min_frames=4, max_frames=8)


@pytest.mark.skipif(not _H264_AVAILABLE, reason="requires a PyAV build with the libx264 encoder")
class TestSeekMatchesSerialH264:
    """Parity on the codec shape of real data: H.264 with B-frames.

    The mpeg4 fixtures above never exercise presentation reordering; real
    clips (e.g. WebVid) are H.264, where it always occurs. ``open_gop`` adds
    leading B-frames that are dropped after a mid-stream seek.
    """

    def _assert_parity_seek_ran(self, path, *, fps, min_frames, max_frames, monkeypatch):
        # Parity as in TestSeekMatchesSerial, plus a spy proving the seek path
        # actually ran: a silent serial fallback would make parity trivially true.
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
        assert len(got) == len(expected)
        assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]

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
        self._assert_parity_seek_ran(
            path, fps=2.0, min_frames=1, max_frames=4, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize("open_gop", [False, True])
    def test_dense_targets_min_eq_max(self, tmp_path, monkeypatch, open_gop):
        path = tmp_path / "clip.mp4"
        _write_h264_mp4(path, n_frames=60, open_gop=open_gop)  # 6s
        self._assert_parity_seek_ran(
            path, fps=8.0, min_frames=16, max_frames=16, monkeypatch=monkeypatch
        )

    @pytest.mark.parametrize("open_gop", [False, True])
    def test_explicit_targets_straddle_gop_boundaries(self, tmp_path, open_gop):
        # Targets just after each keyframe time (keyint=12 @ 10fps -> 1.2s
        # GOPs), where open-GOP leading B-frames sit; drives _decode_seek
        # directly so a serial fallback cannot mask a divergence.
        import av

        from kempnerforge.data.video_io import _decode_seek, _decode_serial

        path = tmp_path / "clip.mp4"
        _write_h264_mp4(path, n_frames=120, open_gop=open_gop)
        targets = [0.05, 2.45, 4.85, 7.25, 9.65, 11.9]

        def _run(fn):
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                return fn(container, stream, list(targets))

        got = _run(_decode_seek)
        expected = _run(_decode_serial)
        assert len(got) == len(expected)
        assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]


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

    def test_landing_check_raises_on_shifted_stream(self, tmp_path):
        import av

        from kempnerforge.data.video_io import _decode_seek, _SeekUnreliableError

        src = tmp_path / "src.mp4"
        dst = tmp_path / "shifted.mp4"
        _write_mp4(src, n_frames=20, gop_size=5)
        _write_shifted_mp4(src, dst, offset_s=5.0)
        with av.open(str(dst)) as container:
            stream = container.streams.video[0]
            # Target 2.0s precedes the shifted stream's first keyframe (5.0s):
            # the seek must land past its target and be flagged unreliable.
            with pytest.raises(_SeekUnreliableError, match="landed"):
                _decode_seek(container, stream, [2.0])


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
