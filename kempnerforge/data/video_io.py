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

from typing import TYPE_CHECKING, Any

from kempnerforge.config.registry import registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PILImage

# AV_TIME_BASE: container.duration is expressed in microseconds.
_AV_TIME_BASE = 1_000_000.0


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
    ``"uniform"`` = ``sample_timestamps``) and read in a single decode pass: each
    target timestamp is mapped to the first decoded frame at or after it
    (timestamps past the last frame map to the last frame, so the final frame is
    always returned). The returned list has length equal to the number of sampled
    timestamps (``<= max_frames``), or is empty when the file has no decodable
    video stream.

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
    images: list[PILImage] = []
    with av.open(path) as container:
        if not container.streams.video:
            return images
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration_s = _video_duration_seconds(stream, container)
        targets = sample(duration_s, fps, min_frames, max_frames)

        j = 0
        eps = 1e-3
        last_frame = None
        for frame in container.decode(stream):
            t = float(frame.time) if frame.time is not None else 0.0
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
