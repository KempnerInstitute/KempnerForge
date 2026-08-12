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

With a ``[frame_selector]`` configured, datasets instead emit *candidate-pool*
samples (raw uint8 frames + query text; see ``VideoDataset``); the main
training process scores them on the accelerator and produces the contract
above via ``frame_selection.apply_frame_selection``.

Frame decoding lives in ``video_io.decode_video_frames`` and is imported at
module scope so tests can substitute a stub; ``av`` itself is imported lazily
inside the decoder.
"""

from __future__ import annotations

import logging
import os
from typing import Any, NamedTuple

import torch
from torch.utils.data import Dataset

from kempnerforge.config.registry import registry
from kempnerforge.data.frame_selection import (
    CandidatePoolSpec,
    _derive_seed,
    decode_candidate_pool,
    make_seed_key,
    uniform_stride_indices,
)
from kempnerforge.data.video_io import decode_video_frames
from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    _tokenize_and_mask,
    frames_to_clip_tensor,
    pil_to_uint8_tensor,
    resolve_pad_id,
)

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
    style gets it for free: when a ``[frame_selector]`` section is configured,
    ``build_video_data`` threads a ``CandidatePoolSpec`` to every corpus and
    ``VideoQADataset.__getitem__`` switches to *pool mode* — it decodes the
    candidate pool and emits it raw (``candidate_pixels`` uint8 +
    ``candidate_count``/``candidate_times``/``query``/``seed_key``) instead of
    a final clip. Scoring and selection run on the accelerator in the main
    training process (``frame_selection.apply_frame_selection``), which
    replaces the pool keys with the standard
    ``pixel_values``/``frame_mask``/``frame_times`` contract. Workers never
    hold a scorer. A dataset style that bypasses ``VideoQADataset`` adopts
    selection by threading ``candidate_spec`` to ``_init_candidate_spec`` and
    emitting the same pool-mode sample dict (see
    ``VideoQADataset.__getitem__`` for the reference implementation).
    """

    # Class-level default so subclasses (and tests that bypass ``__init__``)
    # have it without calling ``_init_candidate_spec``.
    _candidate_spec: CandidatePoolSpec | None = None

    def _init_candidate_spec(self, candidate_spec: CandidatePoolSpec | None) -> None:
        """Store the candidate-pool geometry (or leave selection off when ``None``).

        Derived from the ``[frame_selector]`` section at the ``build_video_data``
        seam. A non-``None`` spec switches ``__getitem__`` to pool mode; the
        dataset never holds a selector or scorer object.
        """
        if candidate_spec is None:
            return
        self._candidate_spec = candidate_spec


class VideoRecord(NamedTuple):
    """One training example before decoding: a clip plus its text.

    ``prompt`` is prepended to ``target`` and masked out of the loss, so a
    caption corpus leaves it empty and a QA corpus puts the question (and any
    answer options) there. ``query`` is the *bare* question text for
    query-aware frame selection: the rendered ``prompt`` buries the question
    in options and boilerplate (and typically overflows the scorer's ~64-token
    text context, forcing chunk+mean-pool), so QA builders thread the raw
    question here. Empty means "no bare question available" and selection
    falls back to ``prompt``.
    """

    video_path: str
    prompt: str
    target: str
    query: str = ""
    # Archive-backed corpora only: the container holding this clip and its
    # ``(offset, size)`` inside it. Unset means ``video_path`` is the file.
    archive: str = ""
    byte_range: tuple[int, int] | None = None


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
        candidate_spec: Optional ``CandidatePoolSpec`` (derived from the
            ``[frame_selector]`` section at the ``build_video_data`` seam)
            switching ``__getitem__`` to candidate-pool mode for main-process
            query-aware selection (``None`` = uniform decode, unchanged).
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
        candidate_spec: CandidatePoolSpec | None = None,
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
        # VideoDataset so every corpus gets it by threading one kwarg. Pool-mode
        # frames ship un-normalized; the MODEL-path normalization of the selected
        # frames happens downstream with the SigLIP defaults, which these
        # defaults pin — so a custom mean/std cannot be honored there. (The
        # scorer's own preprocessing is independent of these values: it is
        # derived from the scorer's processor config in frame_selection.)
        if candidate_spec is not None and (
            image_mean != DEFAULT_IMAGE_MEAN or image_std != DEFAULT_IMAGE_STD
        ):
            raise ValueError(
                "[frame_selector] pool mode normalizes frames in the main process "
                "with the default SigLIP image_mean/image_std; custom values are "
                "not supported alongside a frame selector."
            )
        self._init_candidate_spec(candidate_spec)
        # Subclasses populate their records before calling super().__init__(),
        # so the probe can read them here and surface a silently-unsupervised
        # corpus at startup rather than after a flat-loss run.
        self.probe_supervision()

    def _record(self, idx: int) -> VideoRecord:
        """Record at ``idx``. Override to build records lazily."""
        return self._records[idx]

    def set_epoch(self, epoch: int) -> None:
        """Note the epoch, so prompt-pool draws vary across passes.

        The dataloader forwards it from its checkpointed state, so a resume
        redraws the same prompts.
        """
        self._epoch = int(epoch)

    def _prompt_for(self, idx: int) -> str:
        """The instruction for ``idx``: one pool draw, or the fixed ``prompt``.

        Seeded from ``(idx, epoch)``, not a shared generator: forked workers
        would desynchronize, and every rank must draw the same prompt.
        """
        pool = getattr(self, "_prompt_pool", None)
        if not pool:
            return self._prompt
        seed = _derive_seed(getattr(self, "_epoch", 0), f"prompt|{idx}")
        return pool[seed % len(pool)]

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
            # Archive-backed: video_path is a locator, so stat the archive.
            clip = record.archive or record.video_path
            if not clip or not os.path.exists(clip):
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

        The bare question (``record.query``) when the corpus provides one —
        that is where the information about which frames matter lives, and it
        fits the scorer's text context in one forward instead of being
        chunk+mean-pooled with options and boilerplate. Falls back to the
        rendered ``prompt`` (a QA corpus without a threaded question, or a
        captioning corpus's static instruction). Never the ``target``: the
        caption/answer is not available at test time, so selecting frames on
        it would skew training against eval. A captioning corpus, whose prompt
        is a static instruction or empty, has no meaningful per-sample query —
        the deliberate, falsifiable expectation that query-aware selection
        helps VQA but not captioning. A no-op when no selector is configured.
        """
        return record.query or record.prompt or None

    def _candidate_pool_sample(self, record: VideoRecord, query: str | None) -> dict[str, Any]:
        """Pool-mode sample: raw resized candidates, selection deferred downstream.

        Emitted when a ``CandidatePoolSpec`` is configured. The pool ships
        *ragged* — ``candidate_pixels`` is ``(n_shipped, 3, S, S)`` uint8 (4x
        smaller over worker IPC than float32; no padding to
        ``candidate_frames``, which the collator applies batch-wide) — with
        ``candidate_count = n_shipped`` (0 = undecodable clip, which also
        masks the labels so the sample trains on nothing). Pools also come
        back smaller than ``candidate_frames`` when ``decode_candidate_pool``
        drops time-duplicate frames.

        A sample with no query (empty/None prompt) can never be scored: the
        selector's ``FrameSelector._fallback_indices`` uniform-strides it. That
        stride is applied *here in the worker* instead, so the full pool never
        crosses IPC nor reaches the GPU (see the fast path below).

        ``query``/``seed_key`` ride along for the downstream scoring stage;
        ``seed_key`` mirrors the non-pool seed: (clip path, query), stable
        across ranks/resumes and independent per question of the same clip.
        """
        spec = self._candidate_spec
        assert spec is not None
        try:
            frames, times = decode_candidate_pool(
                self._decode_source(record),
                candidate_frames=spec.candidate_frames,
                candidate_fps=spec.candidate_fps,
                sampling_policy=self._sampling_policy,
            )
        except Exception as e:  # noqa: BLE001 - any decode failure -> skip-with-mask
            logger.debug("candidate pool decode failed for %s: %s", record.video_path, e)
            frames, times = [], []

        n = min(len(frames), spec.candidate_frames)
        frames = list(frames[:n])
        times = [float(t) for t in times[:n]]
        if (query is None or not query.strip()) and n > self._max_frames:
            # Empty-query fast path (same strip semantics as
            # FrameSelector._fallback_indices, so a whitespace-only prompt
            # takes it too): pre-apply the exact uniform stride
            # _fallback_indices would take ([i*n//k]), so the
            # shipped k-frame pool round-trips through selection unchanged
            # (downstream, n_shipped <= k hits the take-all branch, which
            # _fallback_indices checks BEFORE its own empty-query stride).
            # This must stride the *candidate pool* rather than fall back to a
            # standard [video].fps decode: the pool's timestamps are
            # i*duration/(candidate_frames-1) from the candidate decode, so a
            # standard uniform decode would land on different frames/times and
            # silently change what empty-query samples train on.
            sel = uniform_stride_indices(n, self._max_frames)
            frames = [frames[i] for i in sel]
            times = [times[i] for i in sel]

        n_shipped = len(frames)
        candidate_pixels = torch.zeros(
            n_shipped, 3, self._frame_size, self._frame_size, dtype=torch.uint8
        )
        for i, frame in enumerate(frames):
            candidate_pixels[i] = pil_to_uint8_tensor(frame, self._frame_size)
        candidate_times = torch.tensor(times, dtype=torch.float32)

        input_ids, labels = _tokenize_and_mask(
            self._tokenizer, record.target, self._max_text_len, record.prompt or None
        )
        # Label masking keys off the pre-stride pool size: the stride only runs
        # when n > 0, so this is exactly the "undecodable clip" condition.
        if n == 0 or not record.target.strip():
            labels = torch.full_like(labels, -100)
        return {
            "candidate_pixels": candidate_pixels,
            "candidate_count": torch.tensor(n_shipped, dtype=torch.long),
            "candidate_times": candidate_times,
            "query": query or "",
            "seed_key": make_seed_key(record.video_path, query),
            "input_ids": input_ids,
            "labels": labels,
        }

    @staticmethod
    def _decode_source(record: VideoRecord) -> str | bytes:
        """What ``decode_video_frames`` should read for ``record``.

        A path for an on-disk corpus; the clip's bytes when the record points
        into an archive, read by one seek rather than by unpacking the container.
        """
        if not record.archive or record.byte_range is None:
            return record.video_path
        offset, size = record.byte_range
        with open(record.archive, "rb") as fh:
            fh.seek(offset)
            buf = fh.read(size)
        if len(buf) != size:
            raise OSError(f"{record.archive}: read {len(buf)} of {size} bytes at {offset}")
        return buf

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._record(idx)
        query = self._selection_query(record)
        if self._candidate_spec is not None:
            return self._candidate_pool_sample(record, query)
        try:
            frames, frame_times_s = decode_video_frames(
                self._decode_source(record),
                fps=self._fps,
                min_frames=self._min_frames,
                max_frames=self._max_frames,
                sampling_policy=self._sampling_policy,
            )
        except Exception as e:  # noqa: BLE001 - any decode failure -> skip-with-mask
            logger.debug("video decode failed for %s: %s", record.video_path, e)
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
        candidate_spec: Optional ``CandidatePoolSpec`` switching to
            candidate-pool mode for query-aware selection (``None`` = uniform
            decode, unchanged).
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
        prompt_pool: list[str] | None = None,
        dataset_name: str = "webvid-10M",
        sampling_policy: str = "uniform",
        image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN,
        image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD,
        candidate_spec: CandidatePoolSpec | None = None,
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
        self._prompt_pool = list(prompt_pool or ())
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
            candidate_spec=candidate_spec,
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
        return VideoRecord(
            self._video_path(self._ids[idx]), self._prompt_for(idx), self._caps[idx]
        )


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

    Pool-mode samples (a configured ``[frame_selector]``) swap the first three
    keys for the candidate-pool contract consumed by
    ``frame_selection.apply_frame_selection``. Samples arrive *ragged*
    (``(n_i, 3, H, W)``) and are zero-padded here to the batch maximum:
      - ``candidate_pixels``: ``(B, C, 3, H, W)`` uint8
        (``C = batch-max candidate count``, floored at 1 so an all-failed
        batch still has a valid shape).
      - ``candidate_count``: ``(B,)`` int64 (real-prefix length per sample).
      - ``candidate_times``: ``(B, C)`` float32.
      - ``query`` / ``seed_key``: ``list[str]`` of length ``B`` (strings pass
        through pin_memory untouched).

    Text is always padded to ``max_text_len`` (never batch-max) so DP ranks
    see identical shapes under FSDP2, matching ``VLMCollator``.
    """

    def __init__(self, pad_id: int, max_text_len: int) -> None:
        if max_text_len <= 0:
            raise ValueError("max_text_len must be positive")
        self.pad_id = pad_id
        self.max_text_len = max_text_len

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("VideoCollator received an empty batch")
        b = len(samples)
        batch: dict[str, Any]
        if "candidate_pixels" in samples[0]:
            # Ragged pools -> one zero-padded (b, cmax, ...) batch. cmax is the
            # batch-max candidate count, floored at 1 so a batch of all-failed
            # decodes (every count 0) still collates to a valid shape.
            counts = torch.stack([s["candidate_count"] for s in samples], dim=0)
            cmax = max(1, int(counts.max().item()))
            size = samples[0]["candidate_pixels"].shape[-1]
            pixels = torch.zeros(b, cmax, 3, size, size, dtype=torch.uint8)
            times = torch.zeros(b, cmax, dtype=torch.float32)
            for i, s in enumerate(samples):
                n = min(int(s["candidate_count"]), cmax)
                if n > 0:
                    pixels[i, :n] = s["candidate_pixels"][:n]
                    times[i, :n] = s["candidate_times"][:n]
            batch = {
                "candidate_pixels": pixels,
                "candidate_count": counts,
                "candidate_times": times,
                "query": [s["query"] for s in samples],
                "seed_key": [s["seed_key"] for s in samples],
            }
        else:
            batch = {
                "pixel_values": torch.stack([s["pixel_values"] for s in samples], dim=0),
                "frame_mask": torch.stack([s["frame_mask"] for s in samples], dim=0),
                "frame_times": torch.stack([s["frame_times"] for s in samples], dim=0),
            }
        input_ids = torch.full((b, self.max_text_len), self.pad_id, dtype=torch.long)
        labels = torch.full((b, self.max_text_len), -100, dtype=torch.long)
        for i, s in enumerate(samples):
            ids = s["input_ids"]
            lbl = s["labels"]
            n = min(ids.shape[0], self.max_text_len)
            input_ids[i, :n] = ids[:n]
            labels[i, :n] = lbl[:n]
        batch["input_ids"] = input_ids
        batch["labels"] = labels
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
    candidate_spec: CandidatePoolSpec | None = None,
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
        prompt_pool=video_config.prompt_pool,
        dataset_name=video_config.dataset_name,
        sampling_policy=video_config.sampling_policy,
        candidate_spec=candidate_spec,
    )


def build_video_dataset(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    candidate_spec: CandidatePoolSpec | None = None,
) -> VideoDataset:
    """Build the video dataset selected by ``video_config.dataset_type``.

    Dispatches through the ``video_dataset`` registry, so a new dataset style is
    one ``@registry.register_video_dataset`` builder + a config string. The
    configs are duck-typed to avoid a data->config import cycle.
    ``candidate_spec`` (derived from ``[frame_selector]`` at the
    ``build_video_data`` seam) is threaded to
    ``VideoDataset._init_candidate_spec`` so pool-mode query-aware selection
    works for every dataset style.
    """
    builder = registry.get_video_dataset(video_config.dataset_type)
    return builder(video_config, tokenizer_path, max_text_len, candidate_spec)


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
    ``frame_selector_config`` (``[frame_selector]``) is reduced to a
    ``CandidatePoolSpec`` injected into every corpus — datasets only decode
    candidate pools; the selector itself (and its scorer) lives in the main
    training process, which applies scoring/selection per batch via
    ``frame_selection.apply_frame_selection``.
    """
    spec = (
        CandidatePoolSpec.from_config(frame_selector_config)
        if frame_selector_config is not None
        else None
    )
    sources = video_config.sources()
    datasets: list[Dataset] = [
        build_video_dataset(
            video_config.for_source(src),
            tokenizer_path,
            max_text_len,
            spec,
        )
        for src in sources
    ]
    if len(datasets) == 1:
        return datasets[0], None, [sources[0].weight]

    from kempnerforge.data.dataset import MixtureDataset

    mixture = MixtureDataset(datasets, [src.metrics_name for src in sources])
    return mixture, mixture, [src.weight for src in sources]
