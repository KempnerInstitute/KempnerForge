"""Video frame sampling and decoding for the VLM video path.

A clip is reduced to an ordered set of still frames that the VLM pipeline
treats like a sequence of images. Two concerns live here:

1. ``sample_timestamps`` — *which* timestamps to sample. This is the policy
   from the Molmo2 paper (§3.1, §A): sample at a target frame-rate ``fps``,
   cap the total at ``max_frames`` (uniformly subsampling longer clips), and
   always include the first and last frame. Sampling is expressed in
   *seconds* rather than frame indices so it is robust to variable-fps video.
   This function is pure (no decoder dependency) and unit-tested directly.

2. ``decode_video_frames`` — *how* to read those frames. Decoding uses PyAV
   (``av``), whose manylinux wheel bundles FFmpeg, so no system FFmpeg or
   matching CUDA libraries are required (torchcodec needs both). ``av`` is
   imported lazily so this module imports cleanly without it; only actual
   decoding requires the package.

Returned frames are ``PIL.Image`` objects so the caller can reuse the exact
image preprocessing (``pil_to_tensor``) used on the single-image path.
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

from kempnerforge.config.registry import registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fractions import Fraction

    from PIL.Image import Image as PILImage

# AV_TIME_BASE: container.duration is expressed in microseconds.
_AV_TIME_BASE = 1_000_000.0

# A seek target this far (about 68 years) past a stream's start lies beyond its end, yet
# stays within int64 when a demuxer rescales it to nanoseconds.
_PAST_END_S = 1 << 31


@registry.register_sampling_policy("uniform")
def sample_timestamps(
    duration_s: float, fps: float, min_frames: int, max_frames: int
) -> list[float]:
    """Timestamps (seconds) to sample from a clip of length ``duration_s``.

    Policy (Molmo2 §3.1/§A): aim for ``fps`` frames per second, clamp the
    count to ``[min_frames, max_frames]``, and lay the samples out uniformly
    over ``[0, duration_s]`` so the first frame (``0.0``) and last frame
    (``duration_s``) are always included. A non-positive duration (unknown or
    instantaneous) yields a single timestamp at the start.

    Returns a strictly increasing list of length in ``[1, max_frames]``.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive (got {fps})")
    if min_frames < 1 or max_frames < 1:
        raise ValueError(f"min_frames and max_frames must be >= 1 (got {min_frames}, {max_frames})")
    if min_frames > max_frames:
        raise ValueError(f"min_frames ({min_frames}) must be <= max_frames ({max_frames})")
    if duration_s <= 0.0:
        return [0.0]
    desired = round(duration_s * fps)
    desired = max(min_frames, min(max_frames, desired))
    if desired <= 1:
        return [0.0]
    step = duration_s / (desired - 1)
    return [step * i for i in range(desired)]


def _video_duration_seconds(stream: Any, container: Any) -> float:
    """Best-effort clip duration in seconds from PyAV stream/container metadata."""
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration) / _AV_TIME_BASE
    if stream.frames and stream.average_rate:
        return float(stream.frames) / float(stream.average_rate)
    return 0.0


def _presented(packets: Iterable[Any]) -> list[tuple[int, int]]:
    """``(pts, duration)`` of the presented packets, an unknown duration as 0.

    A packet is presented when it has a presentation timestamp and is not flagged
    discard (decoded only as a reference, e.g. ahead of an edit list's start).
    """
    return [(p.pts, p.duration or 0) for p in packets if p.pts is not None and not p.is_discard]


def _span(
    start: int, runs: list[list[tuple[int, int]]], time_base: Fraction
) -> tuple[float, float]:
    """``(start, span)`` in seconds, the span ending where the last presented packet ends.

    ``runs`` are ``_presented`` lists, each from one contiguous read. A last packet
    without a duration lasts the median step between presentation timestamps, taken
    within runs so the gap between two reads is not a step.
    """
    end, duration = max(max(run) for run in runs)
    if not duration:
        pts = (sorted(p for p, _ in run) for run in runs)
        steps = [b - a for run in pts for a, b in itertools.pairwise(run) if b > a]
        duration = statistics.median(steps) if steps else 0
    return float(start * time_base), float((end + duration - start) * time_base)


def _rewind(container: Any, stream: Any, first: Any) -> Iterator[Any]:
    """The stream's packets again from ``first``, its first packet, after seeking back.

    Raises ``RuntimeError`` if the seek lands on another packet: decoding from there
    would skip frames.
    """
    container.seek(first.dts if first.dts is not None else first.pts, stream=stream)
    packets = container.demux(stream)
    again = next(packets)
    if (again.pts, again.pos, again.size) != (first.pts, first.pos, first.size):
        raise RuntimeError("seeking back to the first video packet landed on another packet")
    return itertools.chain([again], packets)


def _all_packets(path: str) -> Iterator[Any]:
    """Every packet of the first video stream, from a fresh open of ``path``."""
    import av

    with av.open(path) as container:
        yield from container.demux(container.streams.video[0])


def _read_extent(
    container: Any, stream: Any, packets: Iterator[Any], restart: Callable[[], Iterator[Any]]
) -> tuple[float, float] | None:
    """Start and span in seconds of ``stream``, reading ``packets`` from its first and
    then its end; ``None`` when no packet has a presentation timestamp.

    The start is the earliest presentation timestamp (PTS) of a presented packet
    (``_presented``); the span runs from it to the end of the last presented packet
    (``_span``). The reads are bounded. A packet is presented no earlier than it is
    decoded, and decode timestamps never decrease, so the start is settled once a
    packet's decode timestamp reaches the earliest PTS seen: only the first reordered
    packets are read. For the end, a backward seek past the stream's end lands on its
    last seek point; when that is a presented keyframe, the packets from it to the end
    of the file hold the last presented one, since no packet decoded before a keyframe
    is presented after it. Every packet is read instead (``restart`` yields them from
    the first) when that seek fails or lands elsewhere, as in containers that seek by
    bisecting timestamps, where timestamps rebuilt from the landing need not match a
    read from the start.
    """
    import av

    time_base = stream.time_base
    head: list[tuple[int, int]] = []
    start = None
    for packet in packets:
        pts, dts = packet.pts, packet.dts
        if pts is not None and not packet.is_discard:
            head.append((pts, packet.duration or 0))
            start = pts if start is None else min(start, pts)
        if start is not None and dts is not None and dts >= start and len(head) > 1:
            break
    else:  # the whole stream was read
        return None if start is None else _span(start, [head], time_base)
    try:
        container.seek(start + int(_PAST_END_S / time_base), stream=stream)
        tail = container.demux(stream)
        landing = next(tail)
        ends = _presented(itertools.chain([landing], tail))
    except av.FFmpegError:
        landing, ends = None, []
    if landing is not None and landing.is_keyframe and _presented([landing]):
        return _span(start, [head, ends], time_base)
    return _span(start, [_presented(restart())], time_base)


def _video_extent(
    container: Any, stream: Any, path: str
) -> tuple[tuple[float, float] | None, Iterator[Any]]:
    """Start and span in seconds of a video stream (``_read_extent``), and its packets
    from the first, for decoding from the open ``container`` of ``path``.

    The extent is ``None`` for a pipe, which can be read only once (an input without a
    size), and for a stream without presentation timestamps. When the first packet is a
    keyframe with a timestamp, the extent is read in ``container``, which then seeks
    back to that packet (``_rewind``). Otherwise the container could not seek back to
    it, so the extent is read from a second open of the file, and ``container`` is
    left at its first packet.
    """
    import av

    packets = container.demux(stream)
    if stream.time_base is None or container.size <= 0:
        return None, packets
    first = next(packets)
    if first.is_keyframe and (first.pts is not None or first.dts is not None):

        def rewind() -> Iterator[Any]:
            return _rewind(container, stream, first)

        return _read_extent(container, stream, itertools.chain([first], packets), rewind), rewind()
    with av.open(path) as probe:
        video = probe.streams.video[0]
        extent = _read_extent(probe, video, probe.demux(video), lambda: _all_packets(path))
    return extent, itertools.chain([first], packets)


def decode_video_frames(
    path: str, *, fps: float, min_frames: int, max_frames: int, sampling_policy: str = "uniform"
) -> list[PILImage]:
    """Decode a clip into a list of sampled ``PIL.Image`` frames (RGB).

    Frames are chosen by the registered ``sampling_policy`` (default
    ``"uniform"`` = ``sample_timestamps``) and read in a single decode pass: each
    target timestamp is mapped to the first decoded frame at or after it
    (timestamps past the last frame map to the last frame, so the final frame is
    always returned). Frame times and the sampled span come from the video stream's
    packet timestamps, read from the same open container (``_video_extent``): times
    count from the stream's start, so a stream whose timestamps start after zero is
    sampled like one starting at zero, and the span ends where its last frame ends,
    whatever the container's duration covers. For a pipe, which can be read only once,
    and a stream without timestamps (a raw elementary stream), the span comes from the
    container metadata and times count from the first frame that has a timestamp; a
    frame without a timestamp counts as time zero. The returned list has length equal
    to the number of sampled timestamps (``<= max_frames``), or is empty when the file
    has no decodable video stream.

    Raises whatever ``av`` raises on a missing/corrupt file, and ``RuntimeError`` if
    the container cannot seek back to the stream's first packet after reading its end;
    callers that train over noisy data should catch and substitute an empty clip.
    """
    try:
        import av  # lazy: bundled-FFmpeg decoder, optional (the "video" dep group)
    except ImportError as e:  # pragma: no cover - only triggered without PyAV installed
        raise ImportError(
            "Video decoding requires PyAV, an optional dependency. "
            "Install the video extra: `uv sync --group video`."
        ) from e

    sample = registry.get_sampling_policy(sampling_policy)
    images: list[PILImage] = []
    with av.open(path) as container:
        if not container.streams.video:
            return images
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        extent, packets = _video_extent(container, stream, path)
        if extent is None:
            start, duration_s = None, _video_duration_seconds(stream, container)
        else:
            start, duration_s = extent
        targets = sample(duration_s, fps, min_frames, max_frames)

        j = 0
        eps = 1e-3
        last_frame = None
        for frame in (frame for packet in packets for frame in packet.decode()):
            ft = frame.time
            if ft is None:
                t = 0.0
            else:
                if start is None:
                    start = ft
                t = ft - start
            while j < len(targets) and t + eps >= targets[j]:
                images.append(frame.to_image())
                j += 1
            last_frame = frame
            if j >= len(targets):
                break
        # Trailing targets (e.g. the final ``duration_s`` timestamp, which sits
        # just past the last frame's PTS) map to the last decoded frame.
        if j < len(targets) and last_frame is not None:
            tail = last_frame.to_image()
            images.extend(tail for _ in range(len(targets) - j))
    return images
