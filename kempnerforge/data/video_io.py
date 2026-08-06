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
   decoding requires the package. Frames are read by seeking to the keyframe
   before each sampled timestamp instead of decoding the whole clip, so cost
   scales with frames kept rather than clip length; containers that cannot
   seek reliably fall back to a single serial pass with identical selection.

Returned frames are ``PIL.Image`` objects so the caller can reuse the exact
image preprocessing (``pil_to_tensor``) used on the single-image path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from kempnerforge.config.registry import registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

# AV_TIME_BASE: container.duration is expressed in microseconds.
_AV_TIME_BASE = 1_000_000.0

# Slack (seconds) when matching a decoded frame's timestamp against a target.
_MATCH_EPS_S = 1e-3


class _SeekUnreliableError(Exception):
    """Raised when seek-based decoding cannot guarantee serial-identical output."""


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


def decode_video_frames(
    path: str, *, fps: float, min_frames: int, max_frames: int, sampling_policy: str = "uniform"
) -> list[PILImage]:
    """Decode a clip into a list of sampled ``PIL.Image`` frames (RGB).

    Frames are chosen by the registered ``sampling_policy`` (default
    ``"uniform"`` = ``sample_timestamps``): each target timestamp is mapped to
    the first decoded frame at or after it (timestamps past the last frame map
    to the last frame, so the final frame is always returned). The returned
    list has length equal to the number of sampled timestamps
    (``<= max_frames``), or is empty when the file has no decodable video
    stream.

    Reading seeks to the keyframe at or before each target and decodes forward
    (``_decode_seek``), so cost scales with frames kept rather than clip
    length. If a seek cannot guarantee the same selection (see
    ``_decode_seek``), the clip is reopened and decoded in a single serial
    pass (``_decode_serial``) with identical selection, so seeking only ever
    changes speed.

    Raises whatever ``av`` raises on a missing/corrupt file; callers that train
    over noisy data should catch and substitute an empty clip.
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
        duration_s = _video_duration_seconds(stream, container)
        targets = sample(duration_s, fps, min_frames, max_frames)
        try:
            return _decode_seek(container, stream, targets)
        except (av.FFmpegError, _SeekUnreliableError) as e:
            reason = f"{type(e).__name__}: {e}"
    logger.debug("seek decode failed for %s (%s); falling back to serial decode", path, reason)
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        return _decode_serial(container, stream, targets)


def _decode_seek(container: Any, stream: Any, targets: list[float]) -> list[PILImage]:
    """Seek to the keyframe at or before each target, then decode forward to it.

    Selection matches ``_decode_serial`` byte-for-byte: each target takes the
    first frame with ``time + _MATCH_EPS_S >= target`` (one frame may satisfy
    several targets), and targets past the last frame tail-fill with that
    frame. Raises ``_SeekUnreliableError`` whenever identical selection cannot
    be guaranteed: no ``time_base``, a frame without PTS, a seek landing past
    its target (frames may have been skipped; exempt for targets near 0.0,
    where the landing is the stream's first frame — what serial selects too),
    or EOF with nothing decoded after a seek.
    """
    tb = stream.time_base
    if tb is None:
        raise _SeekUnreliableError("stream has no time_base")
    images: list[PILImage] = []
    j = 0
    while j < len(targets):
        tgt = targets[j]
        container.seek(int(tgt / tb), stream=stream, backward=True, any_frame=False)
        last = None
        matched = False
        first = True
        for frame in container.decode(stream):  # fresh generator after every seek
            if frame.time is None:
                raise _SeekUnreliableError("frame without pts")
            t = frame.time
            if first:
                first = False
                if t > tgt + _MATCH_EPS_S and tgt > _MATCH_EPS_S:
                    raise _SeekUnreliableError(f"seek to {tgt:.3f}s landed at {t:.3f}s")
            if t + _MATCH_EPS_S >= tgt:
                img = frame.to_image()
                while j < len(targets) and t + _MATCH_EPS_S >= targets[j]:
                    images.append(img)
                    j += 1
                matched = True
                break
            last = frame
        if not matched:  # EOF: the remaining targets sit past the last frame
            if last is None:
                raise _SeekUnreliableError(f"no frames decodable at or after {tgt:.3f}s")
            tail = last.to_image()
            images.extend(tail for _ in range(len(targets) - j))
            j = len(targets)
    return images


def _decode_serial(container: Any, stream: Any, targets: list[float]) -> list[PILImage]:
    """Single decode pass over the whole file; the frame-selection reference.

    Kept as the fallback for containers where seeking is unavailable or
    unreliable; ``_decode_seek`` must match its selection byte-for-byte.
    """
    images: list[PILImage] = []
    j = 0
    last_frame = None
    for frame in container.decode(stream):
        t = float(frame.time) if frame.time is not None else 0.0
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
