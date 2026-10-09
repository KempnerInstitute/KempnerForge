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
   decoding requires the package. Decoding seeks over the keyframe groups
   that hold no sampled timestamp instead of decoding the whole clip, so cost
   scales with the frames kept rather than clip length; streams that cannot
   seek reliably fall back to a single serial pass with identical selection.

Returned frames are ``PIL.Image`` objects so the caller can reuse the exact
image preprocessing (``pil_to_tensor``) used on the single-image path.
"""

from __future__ import annotations

import functools
import itertools
import logging
import statistics
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

from kempnerforge.config.registry import registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fractions import Fraction

    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

# AV_TIME_BASE: container.duration is expressed in microseconds.
_AV_TIME_BASE = 1_000_000.0

# A seek target this far (about 68 years) past a stream's start lies beyond its end, yet
# stays within int64 when a demuxer rescales it to nanoseconds.
_PAST_END_S = 1 << 31

# Slack (seconds) when matching a decoded frame's timestamp against a target.
_MATCH_EPS_S = 1e-3


class _SeekUnreliableError(Exception):
    """Raised when seek-based decoding cannot guarantee serial-identical output."""


@functools.cache
def _log_fallback_once(reason: str) -> None:
    """Log a seek-to-serial fallback once per distinct cause for this process.

    On a corpus where seeking systematically fails, a per-clip log line would
    bury the training log, so fallbacks are deduplicated by ``reason`` (the
    exception type name): ``functools.cache`` runs the body, and thus the log
    call, only on each cause's first occurrence.
    """
    logger.debug(
        "seek decode failed (%s); falling back to serial decode "
        "(further fallbacks with this cause are not logged)",
        reason,
    )


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
    start: int, runs: list[list[tuple[int, int]]], time_base: Fraction, metadata: float
) -> tuple[float, float]:
    """``(start, span)`` in seconds, the span ending where the last presented frame ends.

    ``runs`` are ``_presented`` lists, each from one contiguous read. The packets place
    the last frame's start exactly, and when that packet carries a duration they place
    its end too, which is the span.

    A packet without a duration leaves the end unknown: the packets bound the span from
    below, by the extent from the first timestamp to the last, but only the container's
    ``metadata`` duration knows how long that last frame is shown, which on
    variable-rate video is not the step between any two timestamps and on a one-frame
    stream has no step at all. A duration is reported either as a length or as an end
    time on the stream clock, and the two differ by the stream's start, so a reading is
    taken only where it reaches the last timestamp and leaves the frames a positive
    time to be shown in; a one-frame stream, whose extent is zero, is what the second
    rules out. Where no duration is reported at all it arrives here as zero, which says
    nothing and is not an end time. With neither reading left the last frame is given
    the median step between the timestamps read, taken within runs so the gap between
    two reads is not a step.
    """
    end, duration = max(max(run) for run in runs)
    if duration:
        return float(start * time_base), float((end + duration - start) * time_base)
    begins = float(start * time_base)
    extent = float((end - start) * time_base)
    if metadata > 0:
        as_end = metadata - begins  # read as an end time on the stream clock
        if as_end >= extent and as_end > 0:
            return begins, as_end
        if metadata >= extent:  # read as a length
            return begins, metadata
    pts = (sorted(p for p, _ in run) for run in runs)
    steps = [b - a for run in pts for a, b in itertools.pairwise(run) if b > a]
    shown = statistics.median(steps) if steps else 0
    return begins, float((end + shown - start) * time_base)


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
    (``_presented``); the span runs from it to the end of the last presented frame
    (``_span``, which falls back to the container's duration where the packets do not
    give that end). The reads are bounded. A packet is presented no earlier than it is
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
    metadata = _video_duration_seconds(stream, container)
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
        return None if start is None else _span(start, [head], time_base, metadata)
    try:
        container.seek(start + int(_PAST_END_S / time_base), stream=stream)
        tail = container.demux(stream)
        landing = next(tail)
        ends = _presented(itertools.chain([landing], tail))
    except av.FFmpegError:
        landing, ends = None, []
    if landing is not None and landing.is_keyframe and _presented([landing]):
        return _span(start, [head, ends], time_base, metadata)
    return _span(start, [_presented(restart())], time_base, metadata)


def _video_extent(
    container: Any, stream: Any, path: str
) -> tuple[tuple[float, float] | None, Iterator[Any], Callable[[], Iterator[Any]] | None]:
    """Start and span in seconds of a video stream (``_read_extent``), its packets from
    the first, and a way to read them again, for decoding from the open ``container`` of
    ``path``.

    The extent is ``None`` for a pipe, which can be read only once (an input without a
    size), and for a stream without presentation timestamps. When the first packet is a
    keyframe with a timestamp, the extent is read in ``container``, which then seeks
    back to that packet (``_rewind``), and that seek is also what replays the stream.
    Otherwise the container could not seek back to it, so the extent is read from a
    second open of the file, ``container`` is left at its first packet, and the stream
    cannot be replayed: the replay is ``None``, which is what keeps a pipe, and a stream
    that starts off a keyframe, out of the seeking decoder.
    """
    import av

    packets = container.demux(stream)
    if stream.time_base is None or container.size <= 0:
        return None, packets, None
    first = next(packets)
    if first.is_keyframe and (first.pts is not None or first.dts is not None):

        def rewind() -> Iterator[Any]:
            return _rewind(container, stream, first)

        extent = _read_extent(container, stream, itertools.chain([first], packets), rewind)
        return extent, rewind(), rewind
    with av.open(path) as probe:
        video = probe.streams.video[0]
        extent = _read_extent(probe, video, probe.demux(video), lambda: _all_packets(path))
    return extent, itertools.chain([first], packets), None


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
    sampled like one starting at zero, and the span ends where its last frame ends.
    Where the last packet carries no duration, the container's duration supplies that
    end, read as a length or as an end time, so a duration that covers other streams
    can extend it; for a pipe, which can be read only once, and a stream without
    timestamps (a raw elementary stream), the whole span comes from the container
    metadata and times count from the first frame that has a timestamp; a frame
    without a timestamp counts as time zero. The returned list has length equal
    to the number of sampled timestamps (``<= max_frames``), or is empty when the file
    has no decodable video stream.

    Decoding seeks over the keyframe groups that hold no target (``_decode_seek``), so
    cost scales with frames kept rather than clip length. If a seek cannot guarantee the
    same selection, the stream is replayed from its first packet in the same container
    and decoded in a single pass (``_decode_serial``), so seeking changes which frames
    are decoded, never which are returned. A stream that cannot be replayed, a pipe or
    one that starts off a keyframe, is decoded serially from the start without seeking.
    One consequence of seeking: damage confined to a skipped stretch is never decoded,
    so such a clip returns frames where a serial pass would raise; damage in a group
    that holds a target still raises.

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
    with av.open(path) as container:
        if not container.streams.video:
            return []
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        extent, packets, rewind = _video_extent(container, stream, path)
        if extent is None:  # no packet timestamps: the metadata span, times from frame one
            targets = sample(
                _video_duration_seconds(stream, container), fps, min_frames, max_frames
            )
            return _decode_serial(packets, targets, None)
        start, span = extent
        targets = sample(span, fps, min_frames, max_frames)
        if rewind is None:  # the stream cannot be read twice: one serial pass, no seeking
            return _decode_serial(packets, targets, start)
        try:
            return _decode_seek(container, stream, packets, targets, start)
        except (av.FFmpegError, _SeekUnreliableError) as e:
            _log_fallback_once(type(e).__name__)
            return _decode_serial(rewind(), targets, start)


def _decode_seek(
    container: Any, stream: Any, packets: Iterator[Any], targets: list[float], start: float
) -> list[PILImage]:
    """Decode forward like ``_decode_serial``, seeking over keyframe groups with no target.

    Selection matches ``_decode_serial`` byte-for-byte: frame times count from ``start``,
    each target takes the first frame with ``time + _MATCH_EPS_S >= target`` (one frame
    may satisfy several targets), and targets past the last frame take that frame. Where
    several targets take the same frame they are handed one image rather than one each,
    which a serial pass builds separately; the pixels are the same, the object is not.

    The first targets are decoded from ``packets``, the stream from its first packet,
    without a seek, so the first frames are exactly serial's. After a match the same
    decode keeps running. When it reaches a keyframe with a target still pending, it
    decides between decoding on and seeking. Both end up decoding the frames from the
    keyframe at or before that target up to the target, so they differ only in what
    comes before: decoding on pays for the whole groups in between, while a seek pays
    for the decoder's flush. The decoder holds ``codec_context.delay`` frames in flight
    (the frame-threading latency, which only a decode populates, hence reading it here);
    a seek discards them and must decode as many again before output resumes, so it
    costs about twice that. Seeking therefore pays when the whole groups in between hold
    more frames than that, counted at the longest keyframe interval seen so a stream
    whose keyframes are irregularly spaced, as a scene cut leaves them, is never credited
    with more groups than it has. With no interval measured yet it seeks. A seek that
    lands no further than where it was decided means the
    container indexes fewer seek points than the stream has keyframes, so the rest of
    the clip is decoded without seeking. Seeking once per target instead would re-decode
    a group once for every target inside it.

    Raises ``_SeekUnreliableError`` whenever identical selection cannot be guaranteed: a
    frame without a timestamp, a seek that lands past its target (frames may have been
    skipped) or on a frame that is not a keyframe (the container's index named a seek
    point the stream does not have, so frames decode without their references), or a
    seek after which nothing decodes.
    """
    time_base = stream.time_base
    images: list[PILImage] = []
    j = 0
    seeked = False
    seeking = True
    decided_at = 0.0
    while j < len(targets):
        tgt = targets[j]
        if seeked:
            container.seek(
                int((start + tgt) / time_base), stream=stream, backward=True, any_frame=False
            )
            packets = container.demux(stream)
        last = None
        matched = False
        prev_key_t: float | None = None
        gop_s = 0.0  # the longest keyframe interval seen, so groups are never over-counted
        group = 0  # frames decoded since the last keyframe
        skip_ahead = False
        for frame in (frame for packet in packets for frame in packet.decode()):
            ft = frame.time
            if ft is None:
                raise _SeekUnreliableError("frame without a timestamp")
            if last is None and seeked:
                if ft - start > tgt + _MATCH_EPS_S:
                    raise _SeekUnreliableError(f"seek to {tgt:.3f}s landed at {ft - start:.3f}s")
                if not frame.key_frame:
                    raise _SeekUnreliableError(f"seek to {tgt:.3f}s landed on a non-keyframe")
                if ft - start <= decided_at:
                    seeking = False
            t = ft - start
            if t + _MATCH_EPS_S >= tgt:
                img = frame.to_image()
                while j < len(targets) and t + _MATCH_EPS_S >= targets[j]:
                    images.append(img)
                    j += 1
                if j == len(targets):
                    return images
                tgt = targets[j]
                matched = True
            elif matched and frame.key_frame and seeking:
                longest = gop_s if prev_key_t is None else max(gop_s, t - prev_key_t)
                skippable = None if not longest or longest < 0 else (tgt - t) // longest * group
                if skippable is None or skippable > 2 * (stream.codec_context.delay or 0):
                    skip_ahead = True
                    decided_at = t
                    break
            if frame.key_frame:
                if prev_key_t is not None:
                    gop_s = max(gop_s, t - prev_key_t)
                prev_key_t = t
                group = 0
            group += 1
            last = frame
        if skip_ahead:
            seeked = True
            continue
        # EOF: the remaining targets sit past the last frame.
        if last is None:
            raise _SeekUnreliableError(f"no frames decodable at or after {tgt:.3f}s")
        tail = last.to_image()
        images.extend(tail for _ in range(len(targets) - j))
        j = len(targets)
    return images


def _decode_serial(
    packets: Iterator[Any], targets: list[float], start: float | None
) -> list[PILImage]:
    """Single decode pass over ``packets``; the frame-selection reference.

    Frame times count from ``start``, or, where no packet timestamp gave one, from the
    first frame that has a timestamp; a frame without a timestamp counts as time zero.
    Kept as the fallback for streams where seeking is unavailable or unreliable;
    ``_decode_seek`` must match its selection byte-for-byte.
    """
    images: list[PILImage] = []
    j = 0
    last_frame = None
    for frame in (frame for packet in packets for frame in packet.decode()):
        ft = frame.time
        if ft is None:
            t = 0.0
        else:
            if start is None:
                start = ft
            t = ft - start
        while j < len(targets) and t + _MATCH_EPS_S >= targets[j]:
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
