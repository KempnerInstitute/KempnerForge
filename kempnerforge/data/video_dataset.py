"""Video datasets and collator for the VLM video path.

``VideoQADataset`` is the shared base: given a per-index ``VideoRecord``
(``video_path``, ``prompt``, ``target``) it decodes, pads, timestamps, and
tokenizes one clip. Corpus-specific layouts live in subclasses —
``WebVidVideoDataset`` here (per-partition CSV manifests plus prefix-nested
``.mp4`` files under ``raw/videos/<split>/``), the rest in
``video_qa_datasets``. Each produces the video analogue of the single-image
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
sees identical shapes under FSDP2. ``build_video_data`` resolves the configured
corpora, returning a weighted ``MixtureDataset`` when more than one is listed.

Frame decoding lives in ``video_io.decode_video_frames`` and is imported at
module scope so tests can substitute a stub; ``av`` itself is imported lazily
inside the decoder.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch.utils.data import Dataset

from kempnerforge.config.registry import registry
from kempnerforge.data.frame_selection import (
    ScorerUnavailableError,
    build_frame_selector,
    select_video_frames,
)
from kempnerforge.data.video_io import decode_video_frames
from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    _tokenize_and_mask,
    frames_to_clip_tensor,
    resolve_pad_id,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

# WebVid layout: the metadata split directory ("val") differs from the video
# directory name ("validation"); "train" matches both.
_CSV_SUBDIR = {"train": "train", "validation": "val"}
_VIDEO_SUBDIR = {"train": "train", "validation": "validation"}


class VideoDataset(Dataset):
    """Base for video-caption datasets feeding the VLM video path.

    A subclass is a map-style ``Dataset`` whose ``__getitem__`` returns the
    sample dict ``VideoCollator`` batches:

    - ``pixel_values``: ``(F, 3, H, W)`` float32 (``F = max_frames``, zero-padded).
    - ``frame_mask``: ``(F,)`` bool (``True`` for real frames).
    - ``frame_times``: ``(F,)`` float32 — per-frame time in seconds (``0.0`` on pad
      frames); required by ``VideoCollator``.
    - ``input_ids`` / ``labels``: ``(T,)`` int64, padded to ``max_text_len`` with
      ``-100`` on pad/prompt positions.

    Register a new dataset style with ``@registry.register_video_dataset`` and
    select it via ``[video].dataset_type``; ``build_video_dataset`` dispatches
    through the registry. Most styles subclass ``VideoQADataset``, which
    implements the sample contract once and asks only for per-index records.

    Query-aware frame selection is a base-class capability, so every dataset
    style gets it for free — ``VideoQADataset`` already routes its decode
    through ``_decode_clip``, so subclasses on that base need only thread the
    ``frame_selector`` object to ``super().__init__``. A dataset that bypasses
    ``VideoQADataset`` adopts selection in three steps:

    1. Take ``frame_selector`` in its builder and pass it to
       ``self._init_frame_selector(...)`` in ``__init__``.
    2. Decode via ``self._decode_clip(path, query, ...)`` instead of calling
       ``decode_video_frames`` directly, passing the sample's query text — the
       question/instruction prompt the model is conditioned on at inference.
    3. Nothing else — the returned frames/times flow through the existing
       ``frames_to_clip_tensor`` packing unchanged.

    A dataset without a decodable file path (pre-extracted frames / byte
    streams) can call ``self._frame_selector.select(frames, times, query, k,
    seed_key=...)`` directly; ``_decode_clip`` is a convenience, not the only door.

    The selector is built once at the ``build_video_data`` seam and injected, so a
    mixture shares a single scorer rather than loading one identical copy per
    corpus per worker.
    """

    # Class-level default so subclasses (and tests that bypass ``__init__``)
    # have it without calling ``_init_frame_selector``.
    _frame_selector: Any = None

    def _init_frame_selector(self, frame_selector: Any | None) -> None:
        """Store the prebuilt frame selector (or leave selection off when ``None``).

        The selector is constructed once at the ``build_video_data`` seam and
        injected here, so a mixture of N corpora shares a single ``FrameQueryScorer``
        instead of instantiating N identical copies (each of which would lazily load
        a full scorer in every dataloader worker). The scorer runs on CPU in fp32
        (weights load lazily on first sample).
        """
        if frame_selector is None:
            return
        self._frame_selector = frame_selector

    def _decode_clip(
        self,
        path: str,
        query: str | None,
        *,
        fps: float,
        min_frames: int,
        max_frames: int,
        sampling_policy: str,
        seed_key: str,
    ) -> tuple[list[PILImage], list[float]]:
        """Decode a clip to frames + times, applying query-aware selection when a
        selector is configured (else the plain uniform decode). Shared by every
        dataset style so the decode-vs-select branch lives in one place.
        """
        if self._frame_selector is not None:
            return select_video_frames(
                path,
                query,
                self._frame_selector,
                max_frames,
                sampling_policy=sampling_policy,
                seed_key=seed_key,
            )
        return decode_video_frames(
            path,
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
            sampling_policy=sampling_policy,
        )


class VideoRecord(NamedTuple):
    """One training example before decoding: a clip plus its text.

    ``prompt`` is prepended to ``target`` and masked out of the loss, so a
    caption corpus leaves it empty and a QA corpus puts the question (and any
    answer options) there.
    """

    video_path: str
    prompt: str
    target: str


class VideoQADataset(VideoDataset):
    """Base implementing the sample contract for any video-text corpus.

    Subclasses supply records — either eagerly (pass ``records``) or lazily
    (override ``_record``/``__len__``) — and inherit decode, frame padding,
    per-frame timestamps, prompt masking, and the skip-with-mask behavior that
    keeps a corrupt clip from crashing training.

    Args:
        records: Per-index records; omit when overriding ``_record``.
        tokenizer_path: HF tokenizer id or local path.
        max_text_len: Fixed-length text pad target.
        max_frames / min_frames / fps: Frame-sampling knobs (see ``video_io``).
        frame_size: Square pixel size per frame.
        sampling_policy: Registry key for the frame-sampling policy.
        image_mean / image_std: Per-channel normalization (SigLIP defaults).
        frame_selector: Optional prebuilt ``FrameSelector`` (from the
            ``[frame_selector]`` section, constructed once at the
            ``build_video_data`` seam) enabling query-aware frame selection
            (``None`` = uniform decode, unchanged).
    """

    def __init__(
        self,
        records: list[VideoRecord] | None = None,
        *,
        tokenizer_path: str,
        max_text_len: int,
        max_frames: int,
        min_frames: int,
        fps: float,
        frame_size: int = 224,
        sampling_policy: str = "uniform",
        image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN,
        image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD,
        frame_selector: Any | None = None,
    ) -> None:
        from transformers import AutoTokenizer

        self._records: list[VideoRecord] = records or []
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._pad_id = resolve_pad_id(self._tokenizer)
        self._max_text_len = max_text_len
        self._max_frames = max_frames
        self._min_frames = min_frames
        self._fps = fps
        self._frame_size = frame_size
        self._sampling_policy = sampling_policy
        self._image_mean = image_mean
        self._image_std = image_std
        # Optional query-aware frame selection (no-op when None); inherited from
        # VideoDataset so every corpus gets it by threading one kwarg.
        self._init_frame_selector(frame_selector)
        # Subclasses populate their records before calling super().__init__(),
        # so the probe can read them here and surface a silently-unsupervised
        # corpus at startup rather than after a flat-loss run.
        self.probe_supervision()

    def _record(self, idx: int) -> VideoRecord:
        """Record at ``idx``. Override to build records lazily."""
        return self._records[idx]

    def __len__(self) -> int:
        return len(self._records)

    def probe_supervision(self, n: int = 128) -> float:
        """Fraction of the first ``n`` samples that supervise at least one token.

        Three silent failure modes end with an all-``-100`` label row: a record
        whose text is missing on disk, a QA prompt so long that the answer is
        truncated past ``max_text_len``, and a record whose *video* is absent
        (``__getitem__`` decodes zero frames and masks the labels). All train on
        nothing while looking healthy, so subclasses call this at construction and
        warn when the rate is low. It stats each clip path (cheap) but never
        decodes video, so a corpus built with ``require_video_file=False`` over a
        partially-downloaded tree is reported honestly rather than at false-high.
        """
        probe = min(n, len(self))
        if probe == 0:
            return 0.0
        supervised = 0
        for idx in range(probe):
            record = self._record(idx)
            if not record.target.strip():
                continue
            # Mirror __getitem__: a missing clip decodes to zero frames -> all
            # labels masked, so an unsupervised sample regardless of its text.
            if not record.video_path or not os.path.exists(record.video_path):
                continue
            _, labels = _tokenize_and_mask(
                self._tokenizer, record.target, self._max_text_len, record.prompt or None
            )
            supervised += int(bool((labels != -100).any()))
        rate = supervised / probe
        log = logger.warning if rate < 0.5 else logger.info
        log(
            "%s: %.0f%% of the first %d samples supervise at least one token "
            "(max_text_len=%d); the rest contribute no loss.",
            type(self).__name__,
            100.0 * rate,
            probe,
            self._max_text_len,
        )
        return rate

    def _selection_query(self, record: VideoRecord) -> str | None:
        """Query text handed to a configured frame selector for this sample.

        Always the sample's question/instruction (the ``prompt`` the model is
        conditioned on at inference), never its ``target``: the caption/answer is
        not available at test time, so selecting frames on it would skew training
        against eval. A captioning corpus, whose prompt is a static instruction or
        empty, therefore has no meaningful per-sample query — the deliberate,
        falsifiable expectation that query-aware selection helps VQA but not
        captioning. A no-op when no selector is configured.
        """
        return record.prompt or None

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self._record(idx)
        query = self._selection_query(record)
        try:
            frames, frame_times_s = self._decode_clip(
                record.video_path,
                query,
                fps=self._fps,
                min_frames=self._min_frames,
                max_frames=self._max_frames,
                sampling_policy=self._sampling_policy,
                # Seed on (clip path, query): a QA corpus emits many questions per
                # clip, so path alone would draw identical Gumbel noise for every
                # question. Both parts are stable across ranks/resumes, keeping the
                # draw reproducible while independent per question. Mirrors the
                # answer-shuffle seed (`f"{path}|{question}"`).
                seed_key=f"{record.video_path}|{query or ''}",
            )
        except ScorerUnavailableError:
            # Systemic misconfiguration (e.g. an un-prefetched scorer on an
            # offline node): fail loudly rather than silently masking every clip.
            raise
        except Exception as e:  # noqa: BLE001 - any decode/selection failure -> skip-with-mask
            logger.debug("video decode/selection failed for %s: %s", record.video_path, e)
            frames, frame_times_s = [], []

        pixel_values, frame_mask = frames_to_clip_tensor(
            frames,
            max_frames=self._max_frames,
            frame_size=self._frame_size,
            image_mean=self._image_mean,
            image_std=self._image_std,
        )
        # Per-frame timestamp in seconds; 0.0 for pad frames. The time projection
        # runs over every frame and does not consult frame_mask, so pad frames
        # still receive a (time-0) embedding — harmless only while padded frames
        # are themselves unmasked from attention, and inert once they are masked.
        frame_times = torch.zeros(self._max_frames, dtype=torch.float32)
        n_real = min(len(frames), self._max_frames)
        frame_times[:n_real] = torch.tensor(frame_times_s[:n_real], dtype=torch.float32)

        input_ids, labels = _tokenize_and_mask(
            self._tokenizer, record.target, self._max_text_len, record.prompt or None
        )
        if not frames or not record.target.strip():
            # Undecodable clip, or a record whose text is missing on disk: keep
            # static shapes but contribute no loss.
            labels = torch.full_like(labels, -100)
        return {
            "pixel_values": pixel_values,
            "frame_mask": frame_mask,
            "frame_times": frame_times,
            "input_ids": input_ids,
            "labels": labels,
        }


class WebVidVideoDataset(VideoQADataset):
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
        frame_selector: Optional prebuilt ``FrameSelector`` enabling query-aware
            frame selection (``None`` = uniform decode, unchanged).
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
        frame_selector: Any | None = None,
    ) -> None:
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
        # Records stay lazy: the manifest reaches 10M rows, so ids/captions are
        # kept as parallel lists and a VideoRecord is built per __getitem__.
        self._ids, self._caps = self._load_manifest(csv_dir, max_samples)
        self._prompt = prompt
        super().__init__(
            tokenizer_path=tokenizer_path,
            max_text_len=max_text_len,
            max_frames=max_frames,
            min_frames=min_frames,
            fps=fps,
            frame_size=frame_size,
            sampling_policy=sampling_policy,
            image_mean=image_mean,
            image_std=image_std,
            frame_selector=frame_selector,
        )
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

    def _record(self, idx: int) -> VideoRecord:
        return VideoRecord(self._video_path(self._ids[idx]), self._prompt, self._caps[idx])


class VideoCollator:
    """Stack video samples into a fixed-shape batch.

    Output keys:
      - ``pixel_values``: ``(B, F, 3, H, W)`` float32.
      - ``frame_mask``: ``(B, F)`` bool (``True`` = real frame).
      - ``frame_times``: ``(B, F)`` float32 (per-frame time in seconds).
      - ``input_ids``: ``(B, max_text_len)`` int64.
      - ``labels``: ``(B, max_text_len)`` int64 with ``-100`` on pad/prompt.
      - ``dataset_idx``: ``(B,)`` int64 — only when mixing, so the loop can
        attribute loss to the source corpus.

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
        batch = {
            "pixel_values": pixel_values,
            "frame_mask": frame_mask,
            "frame_times": frame_times,
            "input_ids": input_ids,
            "labels": labels,
        }
        # MixtureDataset tags each sample with its source index; forward it so
        # per-dataset loss can be logged (absent for a single-corpus run).
        if "dataset_idx" in samples[0]:
            batch["dataset_idx"] = torch.tensor(
                [int(s["dataset_idx"]) for s in samples], dtype=torch.long
            )
        return batch


@registry.register_video_dataset("webvid")
def _build_webvid(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    frame_selector: Any | None = None,
) -> WebVidVideoDataset:
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
        frame_selector=frame_selector,
    )


def build_video_dataset(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    frame_selector: Any | None = None,
) -> VideoDataset:
    """Build the video dataset selected by ``video_config.dataset_type``.

    Dispatches through the ``video_dataset`` registry, so a new dataset style is
    one ``@registry.register_video_dataset`` builder + a config string. The
    configs are duck-typed to avoid a data->config import cycle. ``frame_selector``
    is an optional *prebuilt* ``FrameSelector`` (constructed once at the
    ``build_video_data`` seam); builders thread it to
    ``VideoDataset._init_frame_selector`` so query-aware selection works for every
    dataset style while a mixture shares one scorer.
    """
    builder = registry.get_video_dataset(video_config.dataset_type)
    return builder(video_config, tokenizer_path, max_text_len, frame_selector)


def build_video_data(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    frame_selector_config: Any | None = None,
) -> tuple[Dataset, Any, list[float]]:
    """Build the video dataset, or a weighted mixture of several corpora.

    Returns ``(dataset, mixture, weights)``. ``mixture`` is the
    ``MixtureDataset`` when ``[[video.datasets]]`` lists more than one corpus —
    its ``cumulative_sizes`` / ``dataset_names`` drive ``MixtureSampler`` and
    per-dataset metrics — and ``None`` for a single corpus, which keeps the
    plain ``DistributedSampler`` path unchanged.

    Every corpus is built from ``video_config.for_source(src)``, so frame
    geometry is shared and only the per-source fields differ. The optional
    ``frame_selector_config`` (``[frame_selector]``) is realized into a single
    ``FrameSelector`` here and injected into every corpus, so a mixture shares one
    scorer instead of loading N identical copies (per corpus, per worker), while
    query-aware selection still applies across the whole mixture.
    """
    selector = (
        build_frame_selector(frame_selector_config) if frame_selector_config is not None else None
    )
    sources = video_config.sources()
    datasets: list[Dataset] = [
        build_video_dataset(
            video_config.for_source(src),
            tokenizer_path,
            max_text_len,
            selector,
        )
        for src in sources
    ]
    if len(datasets) == 1:
        return datasets[0], None, [sources[0].weight]

    from kempnerforge.data.dataset import MixtureDataset

    mixture = MixtureDataset(datasets, [src.metrics_name for src in sources])
    return mixture, mixture, [src.weight for src in sources]
