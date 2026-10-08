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


def _remux_shifted(src, dst, offset_s: float, *, from_keyframe: int = 0) -> None:
    """Remux ``src``'s video into ``dst`` (container from its suffix), timestamps ``offset_s``
    later; ``from_keyframe=k`` drops the packets before the k-th keyframe in decode order."""
    import av

    with av.open(str(src)) as ic, av.open(str(dst), mode="w") as oc:
        istream = ic.streams.video[0]
        ostream = oc.add_stream_from_template(istream)
        shift = round(offset_s / istream.time_base)
        keyframes = 0
        for packet in ic.demux(istream):
            if packet.pts is None:
                continue
            keyframes += packet.is_keyframe
            if keyframes <= from_keyframe:
                continue
            packet.pts += shift
            if packet.dts is not None:
                packet.dts += shift
            packet.stream = ostream
            oc.mux(packet)


def _write_audio_first_clip(path, n_frames: int, fps: int, video_start_s: float) -> None:
    """An indexed clip from ``video_start_s`` with PCM audio from 0.5 s before to 0.5 s after."""
    from fractions import Fraction

    import av
    import numpy as np

    rate = 8000
    with av.open(str(path), mode="w") as container:
        audio = container.add_stream("pcm_s16le", rate=rate)
        video = container.add_stream("mpeg4", rate=fps)
        video.width = video.height = 64
        video.pix_fmt = "yuv420p"
        first = round((video_start_s - 0.5) * rate)
        last = round((video_start_s + n_frames / fps + 0.5) * rate)
        for pts in range(first, last, rate // 10):
            samples = av.AudioFrame.from_ndarray(
                np.zeros((1, rate // 10), np.int16), format="s16", layout="mono"
            )
            samples.sample_rate, samples.pts, samples.time_base = rate, pts, Fraction(1, rate)
            for packet in audio.encode(samples):
                container.mux(packet)
        for i in range(n_frames):
            frame = av.VideoFrame.from_ndarray(_index_frame(i), format="rgb24")
            frame.pts, frame.time_base = round(video_start_s * fps) + i, Fraction(1, fps)
            for packet in video.encode(frame):
                container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode():
                container.mux(packet)


def _end_edit_list_at(src, dst, end_s: float) -> None:
    """Copy an MP4 whose edit list holds one edit, ending that edit at ``end_s``."""
    import struct

    data = bytearray(src.read_bytes())
    elst, mvhd = data.index(b"elst"), data.index(b"mvhd")
    assert data[elst + 4] == 0 and struct.unpack_from(">I", data, elst + 8)[0] == 1
    assert data[mvhd + 4] == 0  # version 0: the movie timescale follows two 32-bit times
    timescale = struct.unpack_from(">I", data, mvhd + 16)[0]
    struct.pack_into(">I", data, elst + 12, round(end_s * timescale))
    dst.write_bytes(bytes(data))


def _first_frame_time(path) -> float | None:
    """Presentation time of the first decoded frame (``None`` without a timestamp)."""
    import av

    with av.open(str(path)) as container:
        return next(container.decode(container.streams.video[0])).time


def _full_extent(path) -> tuple[float, float]:
    """Start and span from every presented packet, the reference for the bounded read."""
    import itertools
    import statistics

    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        shown = [
            (p.pts, p.duration or 0)
            for p in container.demux(stream)
            if p.pts is not None and not p.is_discard
        ]
    pts = sorted(p for p, _ in shown)
    last, duration = max(shown)
    duration = duration or statistics.median(b - a for a, b in itertools.pairwise(pts) if b > a)
    return float(pts[0] * time_base), float((last + duration - pts[0]) * time_base)


def _rule_indices(path, start: float, span: float) -> list[int]:
    """The 4-frame selection with frame times counted from ``start`` over ``span``."""
    import av

    with av.open(str(path)) as container:
        decoded = [
            (f.time - start, _frame_index(f.to_image()))
            for f in container.decode(container.streams.video[0])
        ]
    last = decoded[-1][1]
    targets = sample_timestamps(span, 2.0, 4, 4)
    return [next((i for t, i in decoded if t + 1e-3 >= tgt), last) for tgt in targets]


def _indices(path, **sampling) -> list[int]:
    from kempnerforge.data.video_io import decode_video_frames

    sampling = sampling or dict(fps=2.0, min_frames=4, max_frames=4)
    return [_frame_index(f) for f in decode_video_frames(str(path), **sampling)]


class _CountingContainer:
    """Forwards to a PyAV input container, counting its seeks and demuxed packets."""

    def __init__(self, container, counts):
        self._container, self._counts = container, counts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._container.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._container, name)

    def seek(self, *args, **kwargs):
        self._counts["seeks"] += 1
        return self._container.seek(*args, **kwargs)

    def demux(self, *args, **kwargs):
        for packet in self._container.demux(*args, **kwargs):
            self._counts["packets"] += 1
            yield packet


@pytest.fixture
def counted_open(monkeypatch):
    """Patch ``av.open`` to count opens, seeks and demuxed packets."""
    import av

    counts = {"opens": 0, "seeks": 0, "packets": 0}
    real_open = av.open

    def _open(*args, **kwargs):
        counts["opens"] += 1
        return _CountingContainer(real_open(*args, **kwargs), counts)

    monkeypatch.setattr(av, "open", _open)
    return counts


# (suffix, encoder, encoder options): the containers and codecs a stream offset is checked in.
_X264_BFRAMES = {"x264-params": "bframes=2:b-adapt=0"}
_OFFSET_CASES = {
    "mp4": ("mp4", "mpeg4", None),
    "mov-bframes": ("mov", "mpeg4", {"bf": "2"}),  # an edit list holds the B-frame delay
    "mkv": ("mkv", "mpeg4", None),
    "webm": ("webm", "libvpx-vp9", None),
    "nut-bframes": ("nut", "mpeg4", {"bf": "2"}),
    "flv": ("flv", "flv", None),  # packets carry no duration
    "asf": ("asf", "wmv2", None),
    "asf-bframes": ("asf", "mpeg4", {"bf": "2"}),  # no stream start time
    "mpegts": ("ts", "mpeg2video", {"bf": "2"}),
    "mpegps": ("mpg", "mpeg2video", {"bf": "2"}),
    "ogg": ("ogv", "libvpx", None),
    "mp4-h264-bframes": ("mp4", "libx264", _X264_BFRAMES),
    "mkv-h264-bframes": ("mkv", "libx264", _X264_BFRAMES),
}


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestDecodeStartOffset:
    """Frame times and the sampled span come from the video packets' timestamps, so a
    stream that starts after 0 s is sampled like one that starts at 0 s."""

    @pytest.mark.parametrize("offset_s", [0.0, 0.5, 5.0])
    @pytest.mark.parametrize("case", list(_OFFSET_CASES))
    def test_start_offset_selects_same_frames(self, tmp_path, case, offset_s):
        """A copy whose timestamps start ``offset_s`` later selects the same frames."""
        suffix, codec, options = _OFFSET_CASES[case]
        if not _encoder_available(codec):
            pytest.skip(f"requires the {codec} encoder")
        src = tmp_path / f"src.{suffix}"
        base = tmp_path / f"base.{suffix}"
        shifted = tmp_path / f"shifted.{suffix}"
        _write_indexed_clip(src, n_frames=20, fps=10, codec=codec, codec_options=options)  # 2 s
        _remux_shifted(src, base, 0.0)
        _remux_shifted(src, shifted, offset_s)
        assert _first_frame_time(shifted) == pytest.approx(_first_frame_time(base) + offset_s)
        # Targets [0, 2/3, 4/3, 2] s; the last sits past the final frame (1.9 s).
        assert _indices(shifted) == _indices(base) == [0, 7, 14, 19]

    def test_b_frame_delay_keeps_selection(self, tmp_path):
        """MPEG-4 B-frames in AVI: the stream starts at 0 s but its first frame at 33 ms."""
        import av

        path = tmp_path / "bframes.avi"
        _write_indexed_clip(path, n_frames=60, fps=30, fmt="avi", codec_options={"bf": "2"})
        with av.open(str(path)) as container:
            assert container.streams.video[0].start_time == 0
        assert _first_frame_time(path) == pytest.approx(1 / 30)
        assert _indices(path) == [0, 20, 40, 59]

    @pytest.mark.skipif(not _H264_AVAILABLE, reason="requires the libx264 encoder")
    def test_h264_b_frame_delay_keeps_selection(self, tmp_path):
        """H.264 B-frames in MP4 without an edit list: the first frame is at 67 ms."""
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
        assert _indices(path) == [0, 20, 40, 59]

    @pytest.mark.parametrize("suffix", ["mkv", "ts"])
    def test_open_gop_start_selects_same_frames(self, tmp_path, suffix):
        """Cut at an open-GOP keyframe, a stream starts with frames that reference the
        dropped group: the decoder skips them, yet they start the stream at any offset."""
        src = tmp_path / f"src.{suffix}"
        options = {"bf": "2", "g": "10"}  # MPEG-2 GOPs are open by default
        _write_indexed_clip(src, n_frames=40, fps=10, codec="mpeg2video", codec_options=options)
        selections = []
        for offset_s in (0.0, 0.5, 5.0):
            cut = tmp_path / f"cut_{offset_s}.{suffix}"
            _remux_shifted(src, cut, offset_s, from_keyframe=1)
            start, span = _full_extent(cut)
            assert start < _first_frame_time(cut)  # the leading frames are skipped
            selections.append(_indices(cut))
            assert selections[-1] == _rule_indices(cut, start, span)
        assert selections[0] == selections[1] == selections[2]

    def test_audio_outside_the_video_is_ignored(self, tmp_path):
        """Audio that starts before the video and ends after it moves neither the stream's
        start nor its end: the selection matches a video-only copy's."""
        _write_indexed_clip(tmp_path / "video.mkv", n_frames=20, fps=10)
        assert _indices(tmp_path / "video.mkv") == [0, 7, 14, 19]
        for offset_s in (0.0, 0.5, 5.0):
            path = tmp_path / f"av_{offset_s}.mkv"
            _write_audio_first_clip(path, n_frames=20, fps=10, video_start_s=offset_s + 0.5)
            assert _indices(path) == [0, 7, 14, 19]

    def test_discarded_preroll_does_not_start_the_stream(self, tmp_path):
        """Frames ahead of an edit list's start are decoded but not shown: the stream
        starts at the first shown frame."""
        from kempnerforge.data.video_io import _video_extent

        src = tmp_path / "src.mov"
        _write_indexed_clip(src, n_frames=20, fps=10, codec_options={"g": "10"})
        trimmed = tmp_path / "trimmed.mov"
        _remux_shifted(src, trimmed, -0.3)  # frames 0-2 fall before 0 s
        assert _video_extent(str(trimmed)) == pytest.approx((0.0, 1.7))
        assert _indices(trimmed) == [3, 9, 15, 19]

    def test_zero_start_matches_absolute_times(self, tmp_path):
        """With the first frame at 0 s, each target takes the first frame whose
        absolute time is within 1 ms of or after it, as before."""
        import av

        from kempnerforge.data.video_io import decode_video_frames

        path = tmp_path / "clip.mp4"
        _write_indexed_clip(path, n_frames=45, fps=30, codec_options={"bf": "2"})
        start, span = _full_extent(path)
        assert start == 0.0
        targets = sample_timestamps(span, 8.0, 12, 12)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
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
        assert video_io._video_extent(str(path)) is None
        fixed = SimpleNamespace(get_sampling_policy=lambda name: lambda *args: [0.0, 0.5, 1.0])
        monkeypatch.setattr(video_io, "registry", fixed)
        frames = video_io.decode_video_frames(str(path), fps=2.0, min_frames=1, max_frames=4)
        assert [_frame_index(f) for f in frames] == [0, 19, 19]

    def test_origin_is_the_first_timestamped_frame(self, monkeypatch):
        """Without packet timestamps, an untimed frame ahead of the first timestamp counts
        as 0 s; later times count from the first timestamped frame (5.0 s here)."""
        from fractions import Fraction
        from types import SimpleNamespace

        import av

        import kempnerforge.data.video_io as video_io

        class _Frame:
            def __init__(self, index, time):
                self.index, self.time = index, time

            def to_image(self):
                return self.index

        class _Container:
            duration = None
            streams = SimpleNamespace(
                video=[
                    SimpleNamespace(
                        duration=None, time_base=Fraction(1, 10), frames=0, average_rate=None
                    )
                ]
            )

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def demux(self, stream):
                return iter([SimpleNamespace(pts=None, dts=None, is_discard=False)] * 5)

            def decode(self, stream):
                return (_Frame(i, t) for i, t in enumerate([None, 5.0, 5.1, 5.2, 5.3]))

        monkeypatch.setattr(av, "open", lambda path: _Container())
        fixed = SimpleNamespace(get_sampling_policy=lambda name: lambda *args: [0.0, 0.15, 0.25])
        monkeypatch.setattr(video_io, "registry", fixed)
        frames = video_io.decode_video_frames("clip", fps=2.0, min_frames=1, max_frames=4)
        assert frames == [0, 3, 4]


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestVideoExtent:
    """``_video_extent`` reads the start and the end of the file, or every packet when a
    seek to the end cannot be trusted."""

    @pytest.mark.parametrize(
        ("suffix", "options"),
        [
            ("mp4", {"bf": "2", "g": "10"}),
            ("mkv", {"g": "10"}),
            ("nut", {"bf": "2", "g": "10"}),
            ("flv", {"g": "10"}),
            ("avi", {"bf": "2", "g": "10"}),
        ],
    )
    def test_reads_only_the_first_and_last_groups(self, tmp_path, counted_open, suffix, options):
        """On a 300-frame clip with a keyframe every 10 frames, the extent matches a full
        read after reading under a tenth of the packets: the first few and the last group."""
        from kempnerforge.data.video_io import _video_extent

        path = tmp_path / f"clip.{suffix}"
        codec = "flv" if suffix == "flv" else "mpeg4"
        _write_indexed_clip(path, n_frames=300, fps=10, codec=codec, codec_options=options)
        expected = _full_extent(path)
        counted_open.update(opens=0, seeks=0, packets=0)
        assert _video_extent(str(path)) == pytest.approx(expected)
        assert expected[1] == pytest.approx(30.0)
        assert counted_open["opens"] == 1
        assert counted_open["seeks"] == 1
        assert counted_open["packets"] <= 30

    def test_stream_within_the_first_reordered_packets(self, tmp_path):
        """A one-frame clip ends before the start is settled: that read is the whole stream."""
        from kempnerforge.data.video_io import _video_extent

        path = tmp_path / "frame.mp4"
        _write_indexed_clip(path, n_frames=1, fps=10)
        assert _video_extent(str(path)) == pytest.approx((0.0, 0.1))

    def test_one_frame_last_group(self, tmp_path):
        """Intra-only FLV: the end read holds only the last frame."""
        from kempnerforge.data.video_io import _video_extent

        path = tmp_path / "intra.flv"
        _write_indexed_clip(path, n_frames=20, fps=10, codec="flv", codec_options={"g": "1"})
        assert _video_extent(str(path)) == pytest.approx((0.0, 2.0))

    def test_one_end_packet_without_duration_takes_the_step_from_the_start(self, monkeypatch):
        """When the end read is one packet without a duration, the step between the packets
        read at the start stands in for it; the start read keeps two packets for that."""
        from fractions import Fraction
        from types import SimpleNamespace

        import av

        from kempnerforge.data.video_io import _video_extent

        def packet(i):
            return SimpleNamespace(
                pts=i, dts=i, duration=0, is_keyframe=i in (0, 9), is_discard=False
            )

        class _Container:
            size = 1000
            streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 10))])

            def __init__(self):
                self.packets = [packet(i) for i in range(10)]

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def demux(self, stream):
                return iter(self.packets)

            def seek(self, offset, stream):
                self.packets = self.packets[9:]  # the last keyframe group: one packet

        monkeypatch.setattr(av, "open", lambda path: _Container())
        assert _video_extent("clip") == pytest.approx((0.0, 1.0))

    @pytest.mark.parametrize(
        ("end_s", "span_s", "indices"), [(1.5, 1.5, [0, 5, 10, 14]), (0.95, 1.0, [0, 4, 7, 9])]
    )
    def test_frames_past_the_edit_list_end_are_not_shown(self, tmp_path, end_s, span_s, indices):
        """An MP4 edit list that ends early leaves the later frames decode-only: the span
        ends with the last shown frame, also when the last keyframe lies past the edit."""
        from kempnerforge.data.video_io import _video_extent

        src = tmp_path / "src.mp4"
        _write_indexed_clip(src, n_frames=20, fps=10, codec_options={"bf": "2", "g": "10"})
        clip = tmp_path / "trimmed.mp4"
        _end_edit_list_at(src, clip, end_s)
        assert _video_extent(str(clip)) == pytest.approx((0.0, span_s))
        assert _indices(clip) == indices

    @pytest.mark.parametrize("suffix", ["ts", "mpg"])
    def test_landing_off_a_keyframe_reads_every_packet(self, tmp_path, counted_open, suffix):
        """MPEG-TS and MPEG-PS seek by bisecting timestamps and land on any packet, where
        rebuilt timestamps need not match a read from the start: every packet is read."""
        from kempnerforge.data.video_io import _video_extent

        path = tmp_path / f"clip.{suffix}"
        options = {"bf": "2", "g": "10"}
        _write_indexed_clip(path, n_frames=40, fps=10, codec="mpeg2video", codec_options=options)
        expected = _full_extent(path)
        counted_open.update(opens=0, seeks=0, packets=0)
        assert _video_extent(str(path)) == pytest.approx(expected)
        assert expected[1] == pytest.approx(4.0)
        assert counted_open["opens"] == 2

    def test_failed_seek_reads_every_packet(self, tmp_path, counted_open):
        """SWF cannot seek: the file is reopened and every packet read."""
        from kempnerforge.data.video_io import _video_extent

        path = tmp_path / "clip.swf"
        _write_indexed_clip(path, n_frames=40, fps=10, codec="flv")
        counted_open.update(opens=0, seeks=0, packets=0)
        assert _video_extent(str(path)) == pytest.approx((0.0, 4.0))
        assert counted_open["opens"] == 2

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires named pipes")
    def test_pipe_is_read_through(self, tmp_path, monkeypatch):
        """A pipe has no size to seek within: its packets are read in a single pass."""
        import threading

        import av

        from kempnerforge.data.video_io import _video_extent

        clip = tmp_path / "clip.mkv"
        _write_indexed_clip(clip, n_frames=40, fps=10, codec_options={"g": "10"})
        pipe = tmp_path / "pipe"
        os.mkfifo(pipe)
        data = clip.read_bytes()

        def _feed():
            with open(pipe, "wb") as writer:
                writer.write(data)

        counts = {"opens": 0, "seeks": 0, "packets": 0}
        real_open = av.open

        def _open_once(*args, **kwargs):
            counts["opens"] += 1
            assert counts["opens"] == 1, "a pipe can be read only once"
            return _CountingContainer(real_open(*args, **kwargs), counts)

        monkeypatch.setattr(av, "open", _open_once)
        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()
        assert _video_extent(str(pipe)) == pytest.approx((0.0, 4.0))
        feeder.join(timeout=10)
        assert counts["seeks"] == 0

    def test_stream_without_time_base(self, monkeypatch):
        """A stream without a time base has no usable timestamps."""
        from types import SimpleNamespace

        import av

        from kempnerforge.data.video_io import _video_extent

        class _Container:
            streams = SimpleNamespace(video=[SimpleNamespace(time_base=None)])

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(av, "open", lambda path: _Container())
        assert _video_extent("clip") is None


class TestSpan:
    """``_span`` measures from the start to where the last presented packet ends."""

    def test_last_packet_duration(self):
        from fractions import Fraction

        from kempnerforge.data.video_io import _span

        runs = [[(10, 1), (13, 1), (11, 1), (12, 2)]]
        assert _span(10, runs, Fraction(1, 10)) == pytest.approx((1.0, 0.4))

    def test_missing_duration_is_the_median_step_within_runs(self):
        """Steps 1, 2 and 6 give 2; the gap between the two reads (9 -> 20) is not a step."""
        from fractions import Fraction

        from kempnerforge.data.video_io import _span

        runs = [[(0, 0), (3, 0), (1, 0), (9, 0)], [(20, 0)]]
        assert _span(0, runs, Fraction(1, 1)) == (0.0, 22.0)

    def test_single_packet_without_duration(self):
        from fractions import Fraction

        from kempnerforge.data.video_io import _span

        assert _span(5, [[(5, 0)]], Fraction(1, 2)) == (2.5, 0.0)


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
