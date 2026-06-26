"""Video dataset and collator for the VLM video path (WebVid-style layout).

``WebVidVideoDataset`` reads a WebVid-style on-disk corpus — per-partition CSV
manifests (``videoid``, ``name`` = caption) plus ``.mp4`` files laid out under
``raw/videos/<split>/`` — and produces the video analogue of the single-image
``VLMSample``:

- ``pixel_values``: ``(F, 3, H, W)`` float tensor — ``F = max_frames`` frames,
  each resized/normalized exactly like the image path. Clips that yield fewer
  than ``F`` real frames are zero-padded.
- ``frame_mask``: ``(F,)`` bool — ``True`` for real frames, ``False`` for padding.
- ``input_ids`` / ``labels``: ``(T,)`` int64, right-padded to ``max_text_len``,
  with ``-100`` on pad/prompt positions. A clip that fails to decode contributes
  no loss (all labels ``-100``) so noisy data never crashes training.

``VideoCollator`` stacks samples into a fixed-shape batch
(``pixel_values: (B, F, 3, H, W)``, ``frame_mask: (B, F)``) so every DP rank
sees identical shapes under FSDP2.

Frame decoding lives in ``video_io.decode_video_frames`` and is imported at
module scope so tests can substitute a stub; ``av`` itself is imported lazily
inside the decoder.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from torch.utils.data import Dataset

from kempnerforge.config.registry import registry
from kempnerforge.data.video_io import decode_video_frames
from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    _pil_to_tensor,
    _tokenize_and_mask,
)

logger = logging.getLogger(__name__)

# WebVid layout: the metadata split directory ("val") differs from the video
# directory name ("validation"); "train" matches both.
_CSV_SUBDIR = {"train": "train", "validation": "val"}
_VIDEO_SUBDIR = {"train": "train", "validation": "validation"}


def _resolve_pad_id(tokenizer: Any) -> int:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    return int(pad_id)


class VideoDataset(Dataset):
    """Base for video-caption datasets feeding the VLM video path.

    A subclass is a map-style ``Dataset`` whose ``__getitem__`` returns the
    sample dict ``VideoCollator`` batches:

    - ``pixel_values``: ``(F, 3, H, W)`` float32 (``F = max_frames``, zero-padded).
    - ``frame_mask``: ``(F,)`` bool (``True`` for real frames).
    - ``input_ids`` / ``labels``: ``(T,)`` int64, padded to ``max_text_len`` with
      ``-100`` on pad/prompt positions.

    Register a new dataset style with ``@registry.register_video_dataset`` and
    select it via ``[video].dataset_type``; ``build_video_dataset`` dispatches
    through the registry. ``WebVidVideoDataset`` is the WebVid-style layout
    (per-partition CSV manifests + prefix-nested ``.mp4`` files); other styles
    (HuggingFace video sets, flat folders, alternate manifests) are follow-ups.
    """


class WebVidVideoDataset(VideoDataset):
    """Map-style WebVid-style video-caption dataset for VLM training.

    Args:
        data_root: Dataset root (contains ``raw/<dataset_name>/data`` and
            ``raw/videos``).
        split: ``"train"`` or ``"validation"``.
        tokenizer_path: HF tokenizer id or local path.
        max_text_len: Fixed-length text pad target.
        max_frames / min_frames / fps: Frame-sampling knobs (see ``video_io``).
        frame_size: Square pixel size per frame.
        max_samples: Cap the manifest (``0`` = all).
        prompt: Optional instruction prepended and masked from the loss.
        image_mean / image_std: Per-channel normalization (SigLIP defaults).
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        max_frames: int,
        min_frames: int,
        fps: float,
        frame_size: int = 224,
        max_samples: int = 0,
        prompt: str = "",
        dataset_name: str = "webvid-10M",
        sampling_policy: str = "uniform",
        image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN,
        image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD,
    ) -> None:
        from transformers import AutoTokenizer

        if split not in _VIDEO_SUBDIR:
            raise ValueError(f"split must be one of {tuple(_VIDEO_SUBDIR)} (got {split!r})")
        self._split = split
        self._video_dir = os.path.join(data_root, "raw", "videos", _VIDEO_SUBDIR[split])
        # ``dataset_name`` names the on-disk corpus (e.g. "webvid-10M"); the WebVid
        # *style* (CSV manifests + prefix-nested mp4s) is shared, so other
        # WebVid-style datasets differ only by this directory.
        csv_dir = os.path.join(
            data_root, "raw", dataset_name, "data", _CSV_SUBDIR[split], "partitions"
        )
        self._ids, self._caps = self._load_manifest(csv_dir, max_samples)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._pad_id = _resolve_pad_id(self._tokenizer)
        self._max_text_len = max_text_len
        self._max_frames = max_frames
        self._min_frames = min_frames
        self._fps = fps
        self._frame_size = frame_size
        self._prompt = prompt
        self._sampling_policy = sampling_policy
        self._image_mean = image_mean
        self._image_std = image_std
        logger.info(
            "WebVidVideoDataset: %s/%s [%s], %d clips, max_frames=%d, fps=%s, frame_size=%d",
            data_root,
            dataset_name,
            split,
            len(self._ids),
            max_frames,
            fps,
            frame_size,
        )

    @staticmethod
    def _load_manifest(csv_dir: str, max_samples: int) -> tuple[list[str], list[str]]:
        """Read partition CSVs into (videoid, caption) lists.

        Reads partitions in sorted order, stopping early once ``max_samples``
        rows are collected so a quick run does not scan the entire corpus.
        ``videoid`` is kept as a string to preserve the digits used by the
        on-disk path mapping.
        """
        import glob

        import pandas as pd

        files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No partition CSVs found under {csv_dir!r}")
        ids: list[str] = []
        caps: list[str] = []
        for path in files:
            df = pd.read_csv(path, usecols=["videoid", "name"], dtype={"videoid": str})
            ids.extend(df["videoid"].tolist())
            caps.extend(df["name"].astype(str).tolist())
            if max_samples and len(ids) >= max_samples:
                break
        if max_samples:
            ids = ids[:max_samples]
            caps = caps[:max_samples]
        return ids, caps

    def _video_path(self, videoid: str) -> str:
        """Map a videoid to its ``.mp4`` path.

        Train videos are nested by id prefixes (``id[:2]/id[:4]/id[:6]/id.mp4``);
        validation videos are flat (``id.mp4``).
        """
        s = str(videoid)
        if self._split == "train":
            return os.path.join(self._video_dir, s[:2], s[:4], s[:6], f"{s}.mp4")
        return os.path.join(self._video_dir, f"{s}.mp4")

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        videoid = self._ids[idx]
        caption = self._caps[idx]
        path = self._video_path(videoid)
        try:
            frames, frame_times_s = decode_video_frames(
                path,
                fps=self._fps,
                min_frames=self._min_frames,
                max_frames=self._max_frames,
                sampling_policy=self._sampling_policy,
            )
        except Exception as e:  # noqa: BLE001 - any decode failure -> skip-with-mask
            logger.debug("video decode failed for %s: %s", path, e)
            frames, frame_times_s = [], []

        f = self._max_frames
        size = self._frame_size
        pixel_values = torch.zeros(f, 3, size, size, dtype=torch.float32)
        frame_mask = torch.zeros(f, dtype=torch.bool)
        # Per-frame timestamp in seconds (0.0 for pad frames; the model reads
        # these only where frame_mask is True, so the pad value is inert).
        frame_times = torch.zeros(f, dtype=torch.float32)
        n_real = min(len(frames), f)
        for i in range(n_real):
            pixel_values[i] = _pil_to_tensor(frames[i], size, self._image_mean, self._image_std)
            frame_mask[i] = True
            frame_times[i] = frame_times_s[i]

        prompt = self._prompt or None
        input_ids, labels = _tokenize_and_mask(self._tokenizer, caption, self._max_text_len, prompt)
        if n_real == 0:
            # Undecodable clip: keep static shapes but contribute no loss.
            labels = torch.full_like(labels, -100)
        return {
            "pixel_values": pixel_values,
            "frame_mask": frame_mask,
            "frame_times": frame_times,
            "input_ids": input_ids,
            "labels": labels,
        }


class VideoCollator:
    """Stack video samples into a fixed-shape batch.

    Output keys:
      - ``pixel_values``: ``(B, F, 3, H, W)`` float32.
      - ``frame_mask``: ``(B, F)`` bool (``True`` = real frame).
      - ``frame_times``: ``(B, F)`` float32 (per-frame time in seconds).
      - ``input_ids``: ``(B, max_text_len)`` int64.
      - ``labels``: ``(B, max_text_len)`` int64 with ``-100`` on pad/prompt.

    Text is always padded to ``max_text_len`` (never batch-max) so DP ranks
    see identical shapes under FSDP2, matching ``VLMCollator``.
    """

    def __init__(self, pad_id: int, max_text_len: int) -> None:
        if max_text_len <= 0:
            raise ValueError("max_text_len must be positive")
        self.pad_id = pad_id
        self.max_text_len = max_text_len

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("VideoCollator received an empty batch")
        b = len(samples)
        pixel_values = torch.stack([s["pixel_values"] for s in samples], dim=0)
        frame_mask = torch.stack([s["frame_mask"] for s in samples], dim=0)
        frame_times = torch.stack([s["frame_times"] for s in samples], dim=0)
        input_ids = torch.full((b, self.max_text_len), self.pad_id, dtype=torch.long)
        labels = torch.full((b, self.max_text_len), -100, dtype=torch.long)
        for i, s in enumerate(samples):
            ids = s["input_ids"]
            lbl = s["labels"]
            n = min(ids.shape[0], self.max_text_len)
            input_ids[i, :n] = ids[:n]
            labels[i, :n] = lbl[:n]
        return {
            "pixel_values": pixel_values,
            "frame_mask": frame_mask,
            "frame_times": frame_times,
            "input_ids": input_ids,
            "labels": labels,
        }


@registry.register_video_dataset("webvid")
def _build_webvid(video_config: Any, tokenizer_path: str, max_text_len: int) -> WebVidVideoDataset:
    """Registry builder for the WebVid-style dataset (see ``WebVidVideoDataset``)."""
    return WebVidVideoDataset(
        data_root=video_config.data_root,
        split=video_config.split,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        max_frames=video_config.max_frames,
        min_frames=video_config.min_frames,
        fps=video_config.fps,
        frame_size=video_config.frame_size,
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        dataset_name=video_config.dataset_name,
        sampling_policy=video_config.sampling_policy,
    )


def build_video_dataset(video_config: Any, tokenizer_path: str, max_text_len: int) -> VideoDataset:
    """Build the video dataset selected by ``video_config.dataset_type``.

    Dispatches through the ``video_dataset`` registry, so a new dataset style is
    one ``@registry.register_video_dataset`` builder + a config string. The
    config is duck-typed to avoid a data->config import cycle.
    """
    builder = registry.get_video_dataset(video_config.dataset_type)
    return builder(video_config, tokenizer_path, max_text_len)
