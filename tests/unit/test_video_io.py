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

    from kempnerforge.data.video_io import _decode_serial, _video_extent

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        (start, span), packets, _ = _video_extent(container, stream, str(path))
        targets = sample_timestamps(span, fps, min_frames, max_frames)
        return _decode_serial(packets, targets, start)


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


def _remux_shifted(
    src, dst, offset_s: float, *, from_keyframe: int = 0, skip_packets: int = 0
) -> None:
    """Remux ``src``'s video and audio into ``dst`` (container from its suffix), timestamps
    ``offset_s`` later; ``from_keyframe=k`` drops the video packets before the k-th keyframe
    in decode order, and ``skip_packets=n`` the first n of them, so the copy starts off a
    keyframe. Only video packets are counted, so an audio track is copied whole."""
    import av

    with av.open(str(src)) as ic, av.open(str(dst), mode="w") as oc:
        streams = [s for s in ic.streams if s.type in ("video", "audio")]
        outputs = {s.index: oc.add_stream_from_template(s) for s in streams}
        keyframes = seen = 0
        for packet in ic.demux(streams):
            if packet.pts is None:
                continue
            if packet.stream.type == "video":
                keyframes += packet.is_keyframe
                seen += 1
                if keyframes <= from_keyframe or seen <= skip_packets:
                    continue
            shift = round(offset_s / packet.stream.time_base)
            packet.pts += shift
            if packet.dts is not None:
                packet.dts += shift
            packet.stream = outputs[packet.stream.index]
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


def _write_clip_with_audio(
    path, n_frames: int, fps: int, *, codec, codec_options, tail_s=1.0
) -> None:
    """An indexed clip with a silent MP2 track that runs ``tail_s`` past the last frame."""
    from fractions import Fraction

    import av
    import numpy as np

    rate = 48000
    with av.open(str(path), mode="w") as container:
        video = container.add_stream(codec, rate=fps)
        video.width = video.height = 64
        video.pix_fmt = "yuv420p"
        video.codec_context.time_base = Fraction(1, fps)
        video.codec_context.options = codec_options
        audio = container.add_stream("mp2", rate=rate)
        audio.layout = "stereo"
        for i in range(n_frames):
            frame = av.VideoFrame.from_ndarray(_index_frame(i), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for packet in video.encode(frame):
                container.mux(packet)
        samples = audio.codec_context.frame_size or 1152
        for pts in range(0, round((n_frames / fps + tail_s) * rate), samples):
            block = av.AudioFrame.from_ndarray(
                np.zeros((1, 2 * samples), np.int16), format="s16", layout="stereo"
            )
            block.sample_rate, block.pts, block.time_base = rate, pts, Fraction(1, rate)
            for packet in audio.encode(block):
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


def _extent_of(path) -> tuple[float, float] | None:
    """``_video_extent`` of a clip, read from its own open container."""
    import av

    from kempnerforge.data.video_io import _video_extent

    with av.open(str(path)) as container:
        return _video_extent(container, container.streams.video[0], str(path))[0]


def _decoded_pts(packets) -> list[int]:
    return [frame.pts for packet in packets for frame in packet.decode()]


def _fresh_pts(path) -> list[int]:
    import av

    with av.open(str(path)) as container:
        return [frame.pts for frame in container.decode(container.streams.video[0])]


class _FakeContainer:
    """An open input over a list of fake packets, as ``_video_extent`` reads one: ``demux``
    reads on from the current position and ``seek`` moves it to the last keyframe at or
    before the target (decode time, else presentation time), like a backward seek."""

    def __init__(self, packets, time_base, size=1000):
        from types import SimpleNamespace

        self.packets, self.size, self.at, self.seeks = packets, size, 0, []
        self.streams = SimpleNamespace(video=[SimpleNamespace(time_base=time_base)])

    def demux(self, stream):
        return iter(self.packets[self.at :])

    def seek(self, offset, stream):
        self.seeks.append(offset)
        keys = [
            i
            for i, p in enumerate(self.packets)
            if p.is_keyframe and (p.dts if p.dts is not None else p.pts) <= offset
        ]
        self.at = keys[-1] if keys else 0


def _fake_packet(pts, dts, *, key=False, duration=0):
    from types import SimpleNamespace

    return SimpleNamespace(
        pts=pts, dts=dts, duration=duration, is_keyframe=key, is_discard=False, pos=dts, size=1
    )


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


# (suffix, encoder, encoder options, packets cut from the start): the containers and codecs a
# stream offset is checked in. A cut copy starts off a keyframe, as a stream copy cut can.
_X264_BFRAMES = {"x264-params": "bframes=2:b-adapt=0"}
_X264_GOP10 = {"x264-params": "keyint=10:min-keyint=10:scenecut=0:bframes=0"}
_OFFSET_CASES = {
    "mp4": ("mp4", "mpeg4", None, 0),
    "mov-bframes": ("mov", "mpeg4", {"bf": "2"}, 0),  # an edit list holds the B-frame delay
    "mkv": ("mkv", "mpeg4", None, 0),
    "webm": ("webm", "libvpx-vp9", None, 0),
    "nut-bframes": ("nut", "mpeg4", {"bf": "2"}, 0),
    "flv": ("flv", "flv", None, 0),  # packets carry no duration
    "asf": ("asf", "wmv2", None, 0),
    "asf-bframes": ("asf", "mpeg4", {"bf": "2"}, 0),  # no stream start time
    "mpegts": ("ts", "mpeg2video", {"bf": "2"}, 0),
    "mpegps": ("mpg", "mpeg2video", {"bf": "2"}, 0),
    "ogg": ("ogv", "libvpx", None, 0),
    "mp4-h264-bframes": ("mp4", "libx264", _X264_BFRAMES, 0),
    "mkv-h264-bframes": ("mkv", "libx264", _X264_BFRAMES, 0),
    "mkv-h264-cut": ("mkv", "libx264", _X264_GOP10, 3),  # its duration is an end time
    "mp4-h264-cut": ("mp4", "libx264", _X264_GOP10, 3),
    "mpegts-h264-cut": ("ts", "libx264", _X264_GOP10, 3),
}


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestDecodeStartOffset:
    """Frame times and the sampled span come from the video packets' timestamps, so a
    stream that starts after 0 s is sampled like one that starts at 0 s."""

    @pytest.mark.parametrize("offset_s", [0.0, 0.5, 5.0])
    @pytest.mark.parametrize("case", list(_OFFSET_CASES))
    def test_start_offset_selects_same_frames(self, tmp_path, case, offset_s):
        """A copy whose timestamps start ``offset_s`` later selects the same frames."""
        suffix, codec, options, cut = _OFFSET_CASES[case]
        if not _encoder_available(codec):
            pytest.skip(f"requires the {codec} encoder")
        src = tmp_path / f"src.{suffix}"
        base = tmp_path / f"base.{suffix}"
        shifted = tmp_path / f"shifted.{suffix}"
        n_frames = 40 if cut else 20
        _write_indexed_clip(src, n_frames=n_frames, fps=10, codec=codec, codec_options=options)
        _remux_shifted(src, base, 0.0, skip_packets=cut)
        _remux_shifted(src, shifted, offset_s, skip_packets=cut)
        assert _first_frame_time(shifted) == pytest.approx(_first_frame_time(base) + offset_s)
        if cut:  # frames 3-9 start the stream but the decoder skips them up to the keyframe
            expected = _rule_indices(base, *_full_extent(base))
            assert expected[0] == 10
        else:  # targets [0, 2/3, 4/3, 2] s; the last sits past the final frame (1.9 s)
            expected = [0, 7, 14, 19]
        assert _indices(shifted) == _indices(base) == expected

    @pytest.mark.skipif(not _H264_AVAILABLE, reason="requires the libx264 encoder")
    @pytest.mark.skipif(not _encoder_available("mp2"), reason="requires the mp2 encoder")
    @pytest.mark.parametrize("offset_s", [0.0, 0.5, 5.0])
    @pytest.mark.parametrize("suffix", ["ts", "mkv"])
    def test_cut_with_audio_measures_the_video_only(self, tmp_path, suffix, offset_s):
        """A cut whose audio runs a second past the last frame is sampled over the video's
        own span. MPEG-TS gets there by reading every packet, so a read that took in the
        other streams would stretch the span; Matroska seeks to its last keyframe instead."""
        src = tmp_path / f"src.{suffix}"
        base = tmp_path / f"base.{suffix}"
        shifted = tmp_path / f"shifted.{suffix}"
        _write_clip_with_audio(src, 40, 10, codec="libx264", codec_options=_X264_GOP10)
        _remux_shifted(src, base, 0.0, skip_packets=3)
        _remux_shifted(src, shifted, offset_s, skip_packets=3)
        start, span = _full_extent(base)  # the video stream's own extent
        assert span == pytest.approx(_full_extent(shifted)[1])
        assert _indices(shifted) == _indices(base) == _rule_indices(base, start, span)

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
        src = tmp_path / "src.mov"
        _write_indexed_clip(src, n_frames=20, fps=10, codec_options={"g": "10"})
        trimmed = tmp_path / "trimmed.mov"
        _remux_shifted(src, trimmed, -0.3)  # frames 0-2 fall before 0 s
        assert _extent_of(trimmed) == pytest.approx((0.0, 1.7))
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
        assert _extent_of(path) is None
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

        def packet(index, time):
            frame = SimpleNamespace(time=time, to_image=lambda: index)
            return SimpleNamespace(pts=None, dts=None, is_keyframe=True, decode=lambda: [frame])

        class _Container:
            duration, size = None, 1000
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
                return (packet(i, t) for i, t in enumerate([None, 5.0, 5.1, 5.2, 5.3]))

        monkeypatch.setattr(av, "open", lambda path: _Container())
        fixed = SimpleNamespace(get_sampling_policy=lambda name: lambda *args: [0.0, 0.15, 0.25])
        monkeypatch.setattr(video_io, "registry", fixed)
        frames = video_io.decode_video_frames("clip", fps=2.0, min_frames=1, max_frames=4)
        assert frames == [0, 3, 4]


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestVideoExtent:
    """``_video_extent`` reads the start and the end of the stream in the open container,
    or every packet when the seek to the end cannot be trusted, then returns the stream's
    packets from the first one."""

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
        read after reading under a tenth of the packets, and decoding resumes at the start."""
        import av

        from kempnerforge.data.video_io import _video_extent

        path = tmp_path / f"clip.{suffix}"
        codec = "flv" if suffix == "flv" else "mpeg4"
        _write_indexed_clip(path, n_frames=300, fps=10, codec=codec, codec_options=options)
        expected, fresh = _full_extent(path), _fresh_pts(path)
        counted_open.update(opens=0, seeks=0, packets=0)
        with av.open(str(path)) as container:
            extent, packets, _ = _video_extent(container, container.streams.video[0], str(path))
            assert extent == pytest.approx(expected)
            assert (counted_open["opens"], counted_open["seeks"]) == (1, 2)
            assert counted_open["packets"] <= 30
            assert _decoded_pts(packets) == fresh
        assert expected[1] == pytest.approx(30.0)

    def test_stream_within_the_first_reordered_packets(self, tmp_path):
        """A one-frame clip ends before the start is settled: that read is the whole stream."""
        path = tmp_path / "frame.mp4"
        _write_indexed_clip(path, n_frames=1, fps=10)
        assert _extent_of(path) == pytest.approx((0.0, 0.1))
        assert _indices(path) == [0, 0, 0, 0]

    def test_one_frame_last_group(self, tmp_path):
        """Intra-only FLV: the end read holds only the last frame."""
        path = tmp_path / "intra.flv"
        _write_indexed_clip(path, n_frames=20, fps=10, codec="flv", codec_options={"g": "1"})
        assert _extent_of(path) == pytest.approx((0.0, 2.0))

    @pytest.mark.parametrize(
        ("end_s", "span_s", "indices"), [(1.5, 1.5, [0, 5, 10, 14]), (0.95, 1.0, [0, 4, 7, 9])]
    )
    def test_frames_past_the_edit_list_end_are_not_shown(self, tmp_path, end_s, span_s, indices):
        """An MP4 edit list that ends early leaves the later frames decode-only: the span
        ends with the last shown frame, also when the last keyframe lies past the edit."""
        src = tmp_path / "src.mp4"
        _write_indexed_clip(src, n_frames=20, fps=10, codec_options={"bf": "2", "g": "10"})
        clip = tmp_path / "trimmed.mp4"
        _end_edit_list_at(src, clip, end_s)
        assert _extent_of(clip) == pytest.approx((0.0, span_s))
        assert _indices(clip) == indices

    def test_seek_back_goes_by_decode_time(self, tmp_path):
        """Raw MPEG-4 with B-frames stores the keyframe shown at 0.3 s at decode time 0, so
        seeking back by the first packet's presentation time (0) would land there; seeking
        by its decode time lands on the first packet."""
        path = tmp_path / "clip.m4v"
        options = {"g": "3", "bf": "2", "sc_threshold": "1000000000"}
        _write_indexed_clip(path, n_frames=60, fmt="m4v", codec_options=options)
        assert _extent_of(path) == pytest.approx(_full_extent(path))
        assert _indices(path) == [0, 20, 40, 59]

    @pytest.mark.parametrize("suffix", ["ts", "mpg"])
    def test_landing_off_a_keyframe_reads_every_packet(self, tmp_path, counted_open, suffix):
        """MPEG-TS and MPEG-PS seek by bisecting timestamps and land on any packet, where
        rebuilt timestamps need not match a read from the start: every packet is read
        from the start, which the container seeks back to twice."""
        import av

        from kempnerforge.data.video_io import _video_extent

        path = tmp_path / f"clip.{suffix}"
        options = {"bf": "2", "g": "10"}
        _write_indexed_clip(path, n_frames=40, fps=10, codec="mpeg2video", codec_options=options)
        expected, fresh = _full_extent(path), _fresh_pts(path)
        counted_open.update(opens=0, seeks=0, packets=0)
        with av.open(str(path)) as container:
            extent, packets, _ = _video_extent(container, container.streams.video[0], str(path))
            assert extent == pytest.approx(expected)
            assert (counted_open["opens"], counted_open["seeks"]) == (1, 3)
            assert _decoded_pts(packets) == fresh
        assert expected[1] == pytest.approx(4.0)

    def test_stream_starting_off_a_keyframe_is_read_from_a_second_open(
        self, tmp_path, counted_open
    ):
        """SWF marks no keyframes, so its container could not seek back to the first
        packet: the extent is read from a second open (and, as SWF cannot seek, a third
        for every packet), and the decoding container stays at the first packet."""
        import av

        from kempnerforge.data.video_io import _video_extent

        path = tmp_path / "clip.swf"
        _write_indexed_clip(path, n_frames=40, fps=10, codec="flv")
        expected, fresh = _full_extent(path), _fresh_pts(path)
        counted_open.update(opens=0, seeks=0, packets=0)
        with av.open(str(path)) as container:
            extent, packets, _ = _video_extent(container, container.streams.video[0], str(path))
            assert extent == pytest.approx(expected)
            assert counted_open["opens"] == 3
            assert _decoded_pts(packets) == fresh

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires named pipes")
    @pytest.mark.parametrize(
        ("suffix", "indices"),
        [("mkv", [0, 7, 14, 19]), ("ts", [0]), ("mp4", [0, 7, 14, 19])],
    )
    def test_pipe_is_decoded_from_one_open(self, tmp_path, monkeypatch, suffix, indices):
        """A pipe can be read once and has no size: ``decode_video_frames`` opens it once
        and decodes it without the probe, with the metadata span (none for MPEG-TS read
        from a pipe, so one frame) and times from the first frame, as before."""
        import threading

        import av

        from kempnerforge.data.video_io import decode_video_frames

        clip = tmp_path / f"clip.{suffix}"
        _write_indexed_clip(clip, n_frames=20, fps=10, codec_options={"g": "10"})
        pipe = tmp_path / "pipe"
        os.mkfifo(pipe)
        data = clip.read_bytes()
        opens = []
        real_open = av.open

        def _open_once(*args, **kwargs):
            opens.append(args[0])
            assert len(opens) == 1, "the pipe was opened twice"
            return real_open(*args, **kwargs)

        def _feed():
            with open(pipe, "wb") as writer:
                writer.write(data)

        result = {}

        def _decode():
            try:
                result["frames"] = decode_video_frames(
                    str(pipe), fps=2.0, min_frames=4, max_frames=4
                )
            except BaseException as e:  # noqa: BLE001 - reported below
                result["error"] = e

        monkeypatch.setattr(av, "open", _open_once)
        feeder = threading.Thread(target=_feed, daemon=True)
        worker = threading.Thread(target=_decode, daemon=True)
        feeder.start()
        worker.start()
        worker.join(timeout=30)
        if worker.is_alive():  # release a reader blocked on a second open of the pipe
            os.close(os.open(pipe, os.O_WRONLY | os.O_NONBLOCK))
            pytest.fail("decoding the pipe hung")
        assert "error" not in result, result.get("error")
        assert len(opens) == 1
        assert [_frame_index(f) for f in result["frames"]] == indices

    def test_start_settles_on_decode_timestamps(self):
        """A B-pyramid whose earliest frame is the third packet decoded, (PTS, DTS) =
        I(3, -2), B(1, -1), b(0, 0): the start is settled only when a decode timestamp
        reaches the earliest presentation timestamp seen, here at b, so it is 0."""
        from fractions import Fraction

        from kempnerforge.data.video_io import _video_extent

        order = [(3, -2), (1, -1), (0, 0), (2, 1), (7, 2), (5, 3), (4, 4), (6, 5)]
        order += [(11, 6), (9, 7), (8, 8), (10, 9)]
        packets = [_fake_packet(pts, dts, key=pts in (3, 11), duration=1) for pts, dts in order]
        container = _FakeContainer(packets, Fraction(1, 10))
        extent, rest, _ = _video_extent(container, container.streams.video[0], "clip")
        assert extent == pytest.approx((0.0, 1.2))
        assert list(rest) == packets

    def test_one_end_packet_without_duration_takes_the_step_from_the_start(self):
        """When the end read is one packet without a duration, the step between the packets
        read at the start stands in for it; the start read keeps two packets for that."""
        from fractions import Fraction

        from kempnerforge.data.video_io import _video_extent

        packets = [_fake_packet(i, i, key=i in (0, 9)) for i in range(10)]
        container = _FakeContainer(packets, Fraction(1, 10))
        extent, rest, _ = _video_extent(container, container.streams.video[0], "clip")
        assert extent == pytest.approx((0.0, 1.0))
        assert list(rest) == packets
        assert container.seeks[-1] == 0  # back to the first packet's decode time

    def test_failed_end_seek_reads_every_packet(self):
        """A seek past the end that fails leaves the start to seek back to: every packet is
        read from there."""
        import errno
        from fractions import Fraction

        import av

        from kempnerforge.data.video_io import _video_extent

        class _NoFarSeek(_FakeContainer):
            def seek(self, offset, stream):
                if offset > 100:
                    raise av.error.PermissionError(errno.EPERM, "cannot seek there")
                super().seek(offset, stream)

        packets = [_fake_packet(i, i, key=i % 5 == 0, duration=1) for i in range(12)]
        container = _NoFarSeek(packets, Fraction(1, 10))
        extent, rest, _ = _video_extent(container, container.streams.video[0], "clip")
        assert extent == pytest.approx((0.0, 1.2))
        assert list(rest) == packets
        assert container.seeks == [0, 0]

    def test_seek_back_landing_elsewhere_raises(self):
        """Decoding from another packet than the first would skip frames: it raises."""
        from fractions import Fraction

        from kempnerforge.data.video_io import _video_extent

        class _LandsLate(_FakeContainer):
            def seek(self, offset, stream):
                super().seek(offset, stream)
                self.at = max(self.at, 5)

        packets = [_fake_packet(i, i, key=i % 5 == 0, duration=1) for i in range(12)]
        container = _LandsLate(packets, Fraction(1, 10))
        with pytest.raises(RuntimeError, match="another packet"):
            _video_extent(container, container.streams.video[0], "clip")

    @pytest.mark.parametrize(("pos", "size"), [(9, 1), (0, 7)])
    def test_seek_back_checks_position_and_size(self, pos, size):
        """A landing packet with the first one's PTS but another byte position or size is
        another packet; a first packet timed only by its decode timestamp has no PTS to
        tell them apart (both None here)."""
        from fractions import Fraction

        from kempnerforge.data.video_io import _rewind

        first = _fake_packet(None, 0, key=True)  # pos 0, size 1
        landing = _fake_packet(None, 0, key=True)
        landing.pos, landing.size = pos, size
        container = _FakeContainer([landing], Fraction(1, 10))
        with pytest.raises(RuntimeError, match="another packet"):
            _rewind(container, container.streams.video[0], first)

    def test_stream_without_time_base(self):
        """A stream without a time base has no usable timestamps."""
        from kempnerforge.data.video_io import _video_extent

        container = _FakeContainer([_fake_packet(0, 0, key=True)], None)
        extent, rest, _ = _video_extent(container, container.streams.video[0], "clip")
        assert extent is None and len(list(rest)) == 1 and container.seeks == []


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


class _WorkCountingContainer:
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

    def demux(self, *args, **kwargs):
        for packet in self._container.demux(*args, **kwargs):
            yield _CountingPacket(packet, self)


class _CountingPacket:
    """Forwards to a packet, counting the frames its ``decode`` returns."""

    def __init__(self, packet, counter):
        self._packet, self._counter = packet, counter

    def __getattr__(self, name):
        return getattr(self._packet, name)

    def decode(self):
        frames = self._packet.decode()
        self._counter.decoded += len(frames)
        return frames


def _run_direct(path, fn, targets):
    """Run ``_decode_seek`` or ``_decode_serial`` on explicit targets; no fallback.

    The extent probe's own reads are not counted: the counters start at the decode."""
    import av

    from kempnerforge.data.video_io import _decode_seek, _video_extent

    with av.open(str(path)) as raw:
        container = _WorkCountingContainer(raw)
        stream = raw.streams.video[0]
        stream.thread_type = "AUTO"
        extent, packets, _ = _video_extent(container, stream, str(path))
        start = 0.0 if extent is None else extent[0]
        container.seeks = container.decoded = 0
        if fn is _decode_seek:
            return fn(container, stream, packets, list(targets), start), container
        return fn(packets, list(targets), start), container


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
        _remux_shifted(src, dst, offset_s=offset_s)
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
        targets = [0.0] + [4.8 * m - pre_s for m in (1, 2, 3, 4)]  # 4 GOPs apart: seeks
        expected, _ = _run_direct(path, _decode_serial, targets)
        try:
            got, counter = _run_direct(path, _decode_seek, targets)
        except _SeekUnreliableError as e:
            assert "landed" in str(e)
        else:
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
        assert [f.tobytes() for f in frames] == [
            f.tobytes() for f in _serial_reference(path, 2.0, 4, 16)
        ]


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
        targets = [0.0] + [4.8 * m - pre_s for m in (1, 2, 3, 4)]
        expected, _ = _run_direct(path, _decode_serial, targets)
        try:
            got, counter = _run_direct(path, _decode_seek, targets)
        except _SeekUnreliableError as e:
            assert "landed" in str(e)
        else:
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
        expected = _serial_reference(path, 2.0, 1, 4)
        reasons = []
        monkeypatch.setattr(video_io, "_log_fallback_once", reasons.append)
        got = video_io.decode_video_frames(str(path), fps=2.0, min_frames=1, max_frames=4)
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
    def test_raw_h264_is_decoded_without_seeking(self, tmp_path, monkeypatch):
        """A raw H.264 stream has no packet timestamps to seek by, so it is decoded
        serially from the start: no seek is attempted and none has to be undone."""
        import kempnerforge.data.video_io as video_io

        path = tmp_path / "clip.h264"
        _write_indexed_clip(path, n_frames=30, codec="libx264", fmt="h264")
        reasons = []
        monkeypatch.setattr(video_io, "_log_fallback_once", reasons.append)
        monkeypatch.setattr(
            video_io, "_decode_seek", lambda *a: pytest.fail("seeking a stream without timestamps")
        )
        frames = video_io.decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8)
        assert reasons == []
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

    def _decode_counting(self, path, _monkeypatch, threads, **cfg):
        """Frames, plus the seeks and frames the decoder itself spent.

        The counters start once the extent probe has run, whose own bounded reads are
        measured in ``TestVideoExtent``, and ``threads`` pins the decoder's depth, which
        the seek decision weighs and which would otherwise follow the host's core count.
        """
        import av

        import kempnerforge.data.video_io as video_io

        real_open, real_extent = av.open, video_io._video_extent
        opened = []

        def _open(*args, **kwargs):
            opened.append(_WorkCountingContainer(real_open(*args, **kwargs)))
            return opened[-1]

        def _extent(container, stream, path_):
            stream.codec_context.thread_count = threads
            result = real_extent(container, stream, path_)
            container.seeks = container.decoded = 0
            return result

        with pytest.MonkeyPatch.context() as patch:  # undone per call, so calls do not nest
            patch.setattr(av, "open", _open)
            patch.setattr(video_io, "_video_extent", _extent)
            frames = video_io.decode_video_frames(str(path), **cfg)
        assert len(opened) == 1  # no fallback reopen
        return frames, opened[0]

    # 20 s at 10 fps with a keyframe every 12 frames: a group is 1.2 s. ``max_frames``
    # sets the spacing, and with it how many whole groups a decode could skip between
    # one target's keyframe and the next target: none at 16 targets (1.33 s apart), one
    # at 8 (2.86 s) and four at 4 (6.67 s), so 0, 12 and 48 frames. A seek throws away
    # the frames the decoder holds in flight and decodes them again, twice ``threads - 1``
    # here, so it pays above 0 frames single-threaded and above 30 at 16 threads.
    @pytest.mark.parametrize(
        ("max_frames", "threads", "seeks", "decoded"),
        [
            (16, 1, 0, 200),
            (16, 16, 0, 200),
            (8, 1, 7, 99),
            (8, 16, 0, 200),
            (4, 1, 3, 47),
            (4, 16, 3, 47),
        ],
    )
    def test_seeks_when_the_groups_skipped_outweigh_the_decoder_flush(
        self, tmp_path, monkeypatch, max_frames, threads, seeks, decoded
    ):
        """Whatever it decides, the frames are the serial pass's."""
        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=200, gop_size=12)
        min_frames = 4 if max_frames == 16 else max_frames
        frames, counter = self._decode_counting(
            path, monkeypatch, threads, fps=2.0, min_frames=min_frames, max_frames=max_frames
        )
        assert [f.tobytes() for f in frames] == [
            f.tobytes() for f in _serial_reference(path, 2.0, min_frames, max_frames)
        ]
        assert (counter.seeks, counter.decoded) == (seeks, decoded)

    @pytest.mark.parametrize(
        ("suffix", "codec", "options"),
        [
            ("mp4", "mpeg4", {"g": "10", "bf": "2"}),
            ("mkv", "mpeg4", {"g": "10"}),
            ("webm", "libvpx-vp9", {"g": "10"}),
            (
                "mkv",
                "libx265",
                {"x265-params": "keyint=10:min-keyint=10:scenecut=0:log-level=error"},
            ),
            ("mp4", "libx264", {"x264-params": "keyint=20:min-keyint=20:scenecut=0:bframes=2"}),
        ],
    )
    @pytest.mark.parametrize("sparse", [True, False])
    def test_the_decision_holds_across_codecs_and_containers(
        self, tmp_path, monkeypatch, suffix, codec, options, sparse
    ):
        """Over 20 s, four targets sit 6.67 s apart and sixteen 1.33 s apart, so the first
        leave whole groups behind and the second at most one. A 16-deep decoder loses 30
        frames to a flush, so it seeks only for the sparse ones, whatever the codec, the
        container or the keyframe spacing; a decoder holding nothing in flight has less to
        lose, so it never seeks less. Both return the serial pass's frames."""
        if not _encoder_available(codec):
            pytest.skip(f"requires the {codec} encoder")
        path = tmp_path / f"clip.{suffix}"
        _write_indexed_clip(path, n_frames=200, fps=10, codec=codec, codec_options=options)
        max_frames = 4 if sparse else 16
        min_frames = 4
        expected = [f.tobytes() for f in _serial_reference(path, 2.0, min_frames, max_frames)]
        counts = {}
        for threads in (16, 1):
            frames, counter = self._decode_counting(
                path, monkeypatch, threads, fps=2.0, min_frames=min_frames, max_frames=max_frames
            )
            assert [f.tobytes() for f in frames] == expected
            counts[threads] = counter.seeks
        assert (counts[16] > 0) is sparse
        assert counts[1] >= counts[16]


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


class _FakeSeekFrame:
    """Decoded-frame stand-in; ``to_image`` returns the frame's index."""

    def __init__(self, index: int, time: float | None, key_frame: bool) -> None:
        self.index = index
        self.time = time
        self.key_frame = key_frame

    def to_image(self) -> int:
        return self.index


class _FakeSeekPacket:
    """Packet stand-in holding one frame."""

    def __init__(self, frame):
        self._frame = frame

    def decode(self):
        return [self._frame]


class _FakeSeekContainer:
    """A 10 fps stream with a keyframe every ``gop`` frames.

    ``seek`` moves to the last keyframe at or before the requested time, then
    ``land_late`` keyframes further; ``land_at_start`` lands on the first frame, as a
    container whose index holds a single entry; ``land_on_any_frame`` moves to the frame at
    that time instead, and ``empty_after_seek`` leaves nothing to decode.
    ``first_frame_key=False`` leaves frame 0 unflagged.
    """

    def __init__(
        self,
        n_frames,
        gop,
        *,
        land_late=0,
        land_at_start=False,
        land_on_any_frame=False,
        empty_after_seek=False,
        timed=True,
        first_frame_key=True,
    ):
        self.frames = [
            _FakeSeekFrame(
                i, i / 10 if timed else None, i % gop == 0 and (i > 0 or first_frame_key)
            )
            for i in range(n_frames)
        ]
        self.gop = gop
        self.land_late = land_late
        self.land_at_start = land_at_start
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
        landed = 0 if self.land_at_start else max(0, key + self.land_late * self.gop)
        self.pos = len(self.frames) if self.empty_after_seek else landed

    def demux(self, stream):
        while self.pos < len(self.frames):
            self.pos += 1
            self.decoded += 1
            yield _FakeSeekPacket(self.frames[self.pos - 1])


def _fake_seek_stream(delay: int = 0):
    """A stream stand-in whose decoder reports ``delay`` frames in flight."""
    return SimpleNamespace(time_base=Fraction(1, 1000), codec_context=SimpleNamespace(delay=delay))


def _run_fake(container, targets, delay: int = 0):
    """``_decode_seek`` over a simulated stream, from its first packet."""
    from kempnerforge.data.video_io import _decode_seek

    stream = _fake_seek_stream(delay)
    return _decode_seek(container, stream, container.demux(stream), list(targets), 0.0)


class TestSeekCursor:
    """The seek decisions and guards, on a simulated stream."""

    def test_seeks_only_over_whole_groups(self):
        from kempnerforge.data.video_io import _decode_serial

        targets = [0.0, 0.5, 1.5, 5.0, 9.9]
        seek = _FakeSeekContainer(100, gop=10)  # 10 s, a keyframe every second
        serial = _FakeSeekContainer(100, gop=10)
        assert _run_fake(seek, targets) == [0, 5, 15, 50, 99]
        assert _decode_serial(serial.demux(None), targets, 0.0) == [0, 5, 15, 50, 99]
        # 1.5 s sits inside the group the decode is already in, so nothing whole can be
        # skipped and it runs on; 5 s and 9.9 s leave whole groups behind, so it seeks.
        assert seek.seeks == [5.0, 9.9]
        assert seek.decoded == 21 + 11 + 10
        assert serial.decoded == 100

    @pytest.mark.parametrize(
        ("delay", "target", "seeks", "decoded"),
        [(0, 2.5, [2.5], 17), (5, 2.5, [], 26), (5, 3.5, [3.5], 17), (10, 3.5, [], 36)],
    )
    def test_seeks_when_the_groups_skipped_outweigh_the_decoder_flush(
        self, delay, target, seeks, decoded
    ):
        """At the 1 s keyframe, 2.5 s leaves one whole group (1-2 s) to skip and 3.5 s two,
        so 10 and 20 frames; a seek costs the twice-``delay`` frames the flush throws away
        and decodes again, so each target is skipped only by a decoder shallow enough."""
        container = _FakeSeekContainer(100, gop=10)
        assert _run_fake(container, [0.0, target], delay=delay) == [0, round(target * 10)]
        assert container.seeks == seeks
        assert container.decoded == decoded

    def test_seeks_without_a_measured_keyframe_interval(self):
        """No keyframe precedes the first one met after a match, so there is no interval
        to compare against: the decode seeks."""
        container = _FakeSeekContainer(100, gop=10, first_frame_key=False)
        assert _run_fake(container, [0.0, 2.5], delay=1000) == [0, 25]
        assert container.seeks == [2.5]

    def test_seek_landing_past_its_target_raises(self):
        from kempnerforge.data.video_io import _SeekUnreliableError

        container = _FakeSeekContainer(100, gop=10, land_late=1)
        with pytest.raises(_SeekUnreliableError, match="seek to 5.000s landed at 6.000s"):
            _run_fake(container, [0.0, 5.0])

    def test_seek_landing_on_a_non_keyframe_raises(self):
        from kempnerforge.data.video_io import _SeekUnreliableError

        container = _FakeSeekContainer(100, gop=10, land_on_any_frame=True)
        with pytest.raises(_SeekUnreliableError, match="seek to 5.500s landed on a non-keyframe"):
            _run_fake(container, [0.0, 5.5])

    def test_nothing_decodable_after_a_seek_raises(self):
        from kempnerforge.data.video_io import _SeekUnreliableError

        container = _FakeSeekContainer(100, gop=10, empty_after_seek=True)
        with pytest.raises(_SeekUnreliableError, match="no frames decodable at or after 5.000s"):
            _run_fake(container, [0.0, 5.0])

    def test_frame_without_timestamp_raises(self):
        from kempnerforge.data.video_io import _SeekUnreliableError

        container = _FakeSeekContainer(10, gop=5, timed=False)
        with pytest.raises(_SeekUnreliableError, match="without a timestamp"):
            _run_fake(container, [0.0])

    def test_a_seek_landing_where_it_decided_stops_seeking(self):
        """A container that indexes fewer seek points than the stream has keyframes lands
        back where the decode already was: it decodes the rest without seeking again."""
        container = _FakeSeekContainer(100, gop=10, land_at_start=True)
        assert _run_fake(container, [0.0, 5.0, 9.9]) == [0, 50, 99]
        assert len(container.seeks) == 1


@pytest.mark.skipif(not _AV_AVAILABLE, reason="requires the 'av' package")
class TestSeekFallback:
    """Unreliable seeks must degrade to the serial pass, never to wrong frames."""

    def test_falls_back_to_serial_on_unreliable_seek(self, tmp_path, monkeypatch):
        import kempnerforge.data.video_io as video_io

        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=40, gop_size=12)
        expected = _serial_reference(path, 2.0, 4, 8)

        def _raise(container, stream, packets, targets, start):
            raise video_io._SeekUnreliableError("test")

        monkeypatch.setattr(video_io, "_decode_seek", _raise)
        got = video_io.decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8)
        assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]

    def test_fallback_logged_once_per_cause(self, tmp_path, monkeypatch, caplog):
        import kempnerforge.data.video_io as video_io

        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=40, gop_size=12)

        def _raise(container, stream, packets, targets, start):
            raise video_io._SeekUnreliableError("test")

        monkeypatch.setattr(video_io, "_decode_seek", _raise)
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        video_io._log_fallback_once.cache_clear()
        with caplog.at_level(logging.DEBUG, logger="kempnerforge.data.video_io"):
            for _ in range(3):
                video_io.decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8)
        fallback_lines = [r for r in caplog.records if "falling back" in r.message]
        assert len(fallback_lines) == 1

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires named pipes")
    @pytest.mark.parametrize("suffix", ["mkv", "ts", "mp4"])
    def test_pipe_never_reaches_the_seeking_decoder(self, tmp_path, monkeypatch, suffix):
        """A pipe is read once and cannot be rewound, so there is no fallback to make: it
        is decoded serially without a seek, which would otherwise hang on the drained pipe."""
        import threading

        import av

        import kempnerforge.data.video_io as video_io

        clip = tmp_path / f"clip.{suffix}"
        _write_indexed_clip(clip, n_frames=20, fps=10, codec_options={"g": "10"})
        pipe = tmp_path / "pipe"
        os.mkfifo(pipe)
        data = clip.read_bytes()
        counts = {"opens": 0, "seeks": 0}
        real_open = av.open

        class _NoSeek:
            def __init__(self, container):
                self._container = container

            def __enter__(self):
                self._container.__enter__()
                return self

            def __exit__(self, *exc):
                return self._container.__exit__(*exc)

            def __getattr__(self, name):
                return getattr(self._container, name)

            def seek(self, *args, **kwargs):
                counts["seeks"] += 1
                raise AssertionError("seeking a pipe")

        def _open(*args, **kwargs):
            counts["opens"] += 1
            assert counts["opens"] == 1, "the pipe was opened twice"
            return _NoSeek(real_open(*args, **kwargs))

        monkeypatch.setattr(av, "open", _open)
        monkeypatch.setattr(
            video_io, "_decode_seek", lambda *a: pytest.fail("seeking a stream read once")
        )
        result = {}

        def _feed():
            with open(pipe, "wb") as writer:
                writer.write(data)

        def _decode():
            try:
                result["frames"] = video_io.decode_video_frames(
                    str(pipe), fps=2.0, min_frames=4, max_frames=4
                )
            except BaseException as e:  # noqa: BLE001 - reported below
                result["error"] = e

        feeder = threading.Thread(target=_feed, daemon=True)
        worker = threading.Thread(target=_decode, daemon=True)
        feeder.start()
        worker.start()
        worker.join(timeout=30)
        if worker.is_alive():  # release a reader blocked on a second open of the pipe
            os.close(os.open(pipe, os.O_WRONLY | os.O_NONBLOCK))
            pytest.fail("decoding the pipe hung")
        assert "error" not in result, result.get("error")
        assert (counts["opens"], counts["seeks"]) == (1, 0)
        assert _frame_index(result["frames"][0]) == 0

    def test_fallback_replays_inside_the_same_container(self, tmp_path, monkeypatch):
        """The serial fallback re-reads the stream through the container already open, so
        the input is opened once; reopening it would hang on a stream that is not a file."""
        import av

        import kempnerforge.data.video_io as video_io

        path = tmp_path / "clip.mp4"
        _write_mp4(path, n_frames=40, gop_size=12)
        expected = _serial_reference(path, 2.0, 4, 8)

        def _raise(container, stream, packets, targets, start):
            raise video_io._SeekUnreliableError("test")

        opens = []
        real_open = av.open
        monkeypatch.setattr(video_io, "_decode_seek", _raise)
        monkeypatch.setattr(
            av, "open", lambda p, *a, **k: (opens.append(p), real_open(p, *a, **k))[1]
        )
        got = video_io.decode_video_frames(str(path), fps=2.0, min_frames=4, max_frames=8)
        assert len(opens) == 1
        assert [f.tobytes() for f in got] == [f.tobytes() for f in expected]

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
