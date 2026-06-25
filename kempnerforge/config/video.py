"""Video input configuration.

``VideoConfig`` is the ``[video]`` top-level section. When present, the job
trains on a video dataset through the VLM wrapper: a clip is decoded into an
ordered set of frames, each preprocessed like an image and fed to the vision
encoder. The section is a sibling of ``[vision_encoder]`` / ``[adapter]`` /
``[vlm]`` and requires ``[vlm]`` to be set.

Frame-sampling defaults follow the Molmo2 paper (sample at ``fps`` per second,
include the first and last frame, cap at ``max_frames``). ``max_frames`` is the
per-clip frame budget; the number of visual tokens it implies
(``max_frames * tokens_per_frame``) feeds the residual-stream / sequence-length
math once the model consumes video.
"""

from __future__ import annotations

from dataclasses import dataclass

_VIDEO_SPLITS = ("train", "validation")


@dataclass
class VideoConfig:
    """Video dataset location and frame-sampling knobs.

    Fields:
        data_root: Root directory of the on-disk video dataset.
        split: Which split to read (``"train"`` or ``"validation"``).
        max_samples: Cap the manifest to this many examples (``0`` = all).
            Useful for quick smoke runs over a 10M-scale corpus.
        max_frames: Maximum frames sampled per clip (the per-clip budget).
        min_frames: Minimum frames sampled per clip (short clips are padded
            with masked frames up to this many real samples where possible).
        fps: Target sampling rate in frames per second (Molmo2 uses 2).
        frame_size: Square pixel size each frame is resized to.
        prompt: Optional fixed instruction prepended to the target text and
            masked from the loss (empty = no prompt).
    """

    data_root: str = ""
    split: str = "train"
    max_samples: int = 0
    max_frames: int = 16
    min_frames: int = 4
    fps: float = 2.0
    frame_size: int = 224
    prompt: str = ""

    def __post_init__(self) -> None:
        if self.split not in _VIDEO_SPLITS:
            raise ValueError(f"video.split must be one of {_VIDEO_SPLITS} (got {self.split!r})")
        if self.max_samples < 0:
            raise ValueError(f"video.max_samples must be non-negative (got {self.max_samples})")
        if self.min_frames < 1:
            raise ValueError(f"video.min_frames must be >= 1 (got {self.min_frames})")
        if self.max_frames < 1:
            raise ValueError(f"video.max_frames must be >= 1 (got {self.max_frames})")
        if self.min_frames > self.max_frames:
            raise ValueError(
                f"video.min_frames ({self.min_frames}) must be <= video.max_frames "
                f"({self.max_frames})"
            )
        if self.fps <= 0:
            raise ValueError(f"video.fps must be positive (got {self.fps})")
        if self.frame_size <= 0:
            raise ValueError(f"video.frame_size must be positive (got {self.frame_size})")
