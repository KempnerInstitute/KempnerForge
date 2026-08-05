"""Corpus adapters for video caption / QA fine-tuning.

Each class here does one job: turn an on-disk corpus into
``VideoRecord(video_path, prompt, target)`` list. Everything downstream —
frame sampling, decode, padding, per-frame timestamps, prompt masking, and the
skip-with-mask behavior for undecodable clips — is inherited from
``VideoQADataset`` in ``video_dataset``.

Corpora, and the layout each expects under its ``data_root``:

- ``molmo2_videocapqa``: ``videos/<subset>/<id>/<id>.{mp4,mkv,webm}`` with
  optional yt-dlp sidecars (``<id>.json`` / ``<id>.info.json``) and a denoised
  ASR sidecar (``<id>.grover.json``).
- ``perception_test``: ``mc_question_<split>.json`` (``{video_id: {metadata,
  mc_question: [...]}}``) plus flat ``videos/<video_id>.mp4``.
- ``nextqa``: ``annotations/MC/<split>.csv`` (options in ``a0``..``a4``, answer
  is an index) or ``annotations/OE/<split>-*.parquet`` (free-text answer), plus
  flat ``videos/<video>.mp4``.
- ``cinepile``: ``v1``/``v2`` parquet shards plus flat ``videos/<ytid>.mp4``.

Multiple-choice questions are rendered by a registered ``qa_format`` policy
(``qa_format.py``) so the prompt/target shape is a config knob rather than a
per-corpus decision.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import random
import zlib
from collections.abc import Sequence
from typing import Any

from kempnerforge.config.registry import registry
from kempnerforge.data.qa_format import format_multiple_choice, format_open_ended
from kempnerforge.data.video_dataset import VideoQADataset, VideoRecord

logger = logging.getLogger(__name__)

# Containers PyAV decodes; Molmo2's yt-dlp output mixes all three.
_VIDEO_EXTS = (".mp4", ".mkv", ".webm")


def _geometry(video_config: Any, frame_selector: Any | None = None) -> dict[str, Any]:
    """Frame-geometry kwargs shared by every corpus (global on ``[video]``).

    The optional prebuilt ``frame_selector`` (from the ``[frame_selector]``
    section, constructed once at the ``build_video_data`` seam) rides along here
    so it reaches ``VideoQADataset.__init__`` via each dataset's ``**geometry``
    forwarding, enabling query-aware selection for every corpus.
    """
    return {
        "max_frames": video_config.max_frames,
        "min_frames": video_config.min_frames,
        "fps": video_config.fps,
        "frame_size": video_config.frame_size,
        "sampling_policy": video_config.sampling_policy,
        "frame_selector": frame_selector,
    }


def _cap[T](items: list[T], max_samples: int) -> list[T]:
    return items[:max_samples] if max_samples else items


def _log_built(name: str, root: str, split: str, records: Sequence[Any]) -> None:
    logger.info("%s: %s [%s], %d samples", name, root, split, len(records))


class Molmo2VideoCapQADataset(VideoQADataset):
    """Molmo2-VideoCapQA: per-video directories of YouTube clips.

    Videos live at ``videos/<subset>/<id>/<id>.<ext>`` (yt-dlp writes ``.mp4``,
    ``.mkv`` or ``.webm``), and the target text comes from a sidecar in the same
    directory. ``text_source`` selects it:

    - ``"title"`` (default): the video title from ``<id>.json`` /
      ``<id>.info.json``.
    - ``"description"``: the uploader's description, truncated to
      ``description_max_chars``.
    - ``"asr"``: the denoised ASR transcript from ``<id>.grover.json``.

    Sidecar coverage is uneven — only ``youtube-cc-temporal`` carries ASR, and
    ``youtube-cc-exist`` carries no sidecar at all — so text is read lazily per
    sample and a clip with no text contributes no loss. Reading it eagerly would
    mean parsing one ~140 KB JSON per video across the whole corpus at startup,
    on every rank.

    The manifest comes from ``videos/download_log.csv`` when present (the
    authoritative record of what actually downloaded); otherwise the subset
    directories are scanned.

    Args:
        data_root: Corpus root (contains ``videos/``).
        subset: Subset directory, or ``""`` for every subset under ``videos/``.
        tokenizer_path / max_text_len: Tokenization.
        max_samples: Cap the manifest (``0`` = all).
        prompt: Instruction prepended and masked from the loss.
        text_source: Which sidecar field supplies the caption.
        description_max_chars: Truncation for ``text_source="description"``.
    """

    _TEXT_SOURCES = ("title", "description", "asr")

    def __init__(
        self,
        data_root: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        subset: str = "",
        max_samples: int = 0,
        prompt: str = "",
        text_source: str = "title",
        description_max_chars: int = 512,
        **geometry: Any,
    ) -> None:
        text_source = text_source or "title"
        if text_source not in self._TEXT_SOURCES:
            raise ValueError(
                f"molmo2_videocapqa text_source must be one of {self._TEXT_SOURCES} "
                f"(got {text_source!r})"
            )
        videos_dir = os.path.join(data_root, "videos")
        if not os.path.isdir(videos_dir):
            raise FileNotFoundError(f"No videos/ directory under {data_root!r}")

        self._paths = _cap(self._load_manifest(videos_dir, subset), max_samples)
        if not self._paths:
            raise FileNotFoundError(f"No clips found under {videos_dir!r}")
        self._prompt = prompt
        self._text_source = text_source
        self._description_max_chars = description_max_chars

        super().__init__(tokenizer_path=tokenizer_path, max_text_len=max_text_len, **geometry)
        logger.info(
            "Molmo2VideoCapQADataset: %s [%s], %d clips, text_source=%s",
            data_root,
            subset or "all subsets",
            len(self._paths),
            text_source,
        )

    @classmethod
    def _load_manifest(cls, videos_dir: str, subset: str) -> list[str]:
        """Absolute clip paths, from the download log when available."""
        log_path = os.path.join(videos_dir, "download_log.csv")
        if os.path.exists(log_path):
            paths = cls._paths_from_log(log_path, videos_dir, subset)
            if paths:
                return paths
            logger.warning(
                "download_log.csv listed no clips for subset=%r; scanning %s instead",
                subset,
                videos_dir,
            )
        return cls._paths_from_scan(videos_dir, subset)

    @staticmethod
    def _paths_from_log(log_path: str, videos_dir: str, subset: str) -> list[str]:
        """Clip paths from the yt-dlp download log (CRLF-terminated CSV)."""
        import pandas as pd

        df = pd.read_csv(log_path, usecols=["rel_path", "status"], dtype=str)
        rel = df.loc[df["status"] == "ok", "rel_path"].dropna()
        rel = rel[rel.str.endswith(_VIDEO_EXTS)]
        if subset:
            rel = rel[rel.str.startswith(f"{subset}/")]
        return sorted(os.path.join(videos_dir, p) for p in rel)

    @staticmethod
    def _paths_from_scan(videos_dir: str, subset: str) -> list[str]:
        """Clip paths by walking ``videos/<subset>/<id>/``."""
        subsets = [subset] if subset else sorted(os.listdir(videos_dir))
        paths: list[str] = []
        for name in subsets:
            subset_dir = os.path.join(videos_dir, name)
            if not os.path.isdir(subset_dir):
                if subset:
                    raise FileNotFoundError(f"No subset directory {subset_dir!r}")
                continue
            for video_id in sorted(os.listdir(subset_dir)):
                entry_dir = os.path.join(subset_dir, video_id)
                if not os.path.isdir(entry_dir):
                    continue
                paths.extend(
                    os.path.join(entry_dir, f)
                    for f in sorted(os.listdir(entry_dir))
                    if f.endswith(_VIDEO_EXTS)
                )
        return paths

    def __len__(self) -> int:
        return len(self._paths)

    def _record(self, idx: int) -> VideoRecord:
        path = self._paths[idx]
        return VideoRecord(path, self._prompt, self._text_for(path))

    def sidecar_fields(self, video_path: str) -> dict[str, str]:
        """Every sidecar text variant for a clip, untruncated.

        Public so a data-inspection pass can compare title / description / ASR
        side by side before choosing one. Coverage differs by subset:
        ``youtube-cc-temporal`` carries all three, ``youtube-cc-kw`` has no ASR,
        and ``youtube-cc-exist`` has no sidecar at all. Missing fields come back
        as ``""``.
        """
        saved = self._text_source
        try:
            fields = {}
            for source in self._TEXT_SOURCES:
                self._text_source = source
                # Bypass the description truncation so the caller sees the real
                # length and can decide where to cut.
                limit, self._description_max_chars = self._description_max_chars, 10**9
                fields[source] = self._text_for(video_path)
                self._description_max_chars = limit
            return fields
        finally:
            self._text_source = saved

    def _text_for(self, video_path: str) -> str:
        """Target text from the sidecars beside ``video_path`` (``""`` if none)."""
        video_dir, filename = os.path.split(video_path)
        video_id = os.path.splitext(filename)[0]
        if self._text_source == "asr":
            payload = _read_json(os.path.join(video_dir, f"{video_id}.grover.json"))
            if not payload:
                return ""
            # grover.json: {"denoised": [{"noisyasr": ..., "cleanasr": ...}, ...]}.
            # ``cleanasr`` is the punctuated, denoised transcript and is what we
            # want; ``noisyasr`` is the raw lowercase ASR it was cleaned from.
            # Most videos have one entry, a few have several consecutive
            # segments, so entries are joined in order.
            chunks = [
                str(e.get("cleanasr") or e.get("noisyasr") or "")
                for e in payload.get("denoised") or []
                if isinstance(e, dict)
            ]
            return " ".join(c for c in chunks if c).strip()

        # yt-dlp writes either <id>.json or <id>.info.json depending on the pass.
        info = _read_json(os.path.join(video_dir, f"{video_id}.json")) or _read_json(
            os.path.join(video_dir, f"{video_id}.info.json")
        )
        if not info:
            return ""
        if self._text_source == "description":
            return str(info.get("description") or "").strip()[: self._description_max_chars]
        return str(info.get("title") or "").strip()


class SpatialVIDDataset(VideoQADataset):
    """SpatialVID: LLM-written descriptive captions over camera-motion-rich clips.

    ``metadata.csv`` at the corpus root is a flat manifest
    (``video_path``, ``caption``, ``height``, ``width``, ``fps``, ``n_frames``)
    whose ``video_path`` is relative to ``data_root``. Each clip also has
    ``annotations/<group>/<id>/caption.json`` carrying four differently-scoped
    captions; ``text_source`` picks which one supervises:

    - ``"shot_immersion"`` (default): cinematic narration blending scene and
      camera movement. This is exactly the manifest's ``caption`` column, so it
      is read straight from the CSV with no per-clip file access.
    - ``"scene_description"``: the longest, and the least camera-centric.
    - ``"scene_summary"``: a one-sentence summary.
    - ``"camera_motion"``: camera movement only.

    The non-default fields are read lazily from each clip's ``caption.json``,
    since eagerly parsing one JSON per clip across ~348k clips would dominate
    startup on every rank.

    Note the corpus ships several derivative manifests
    (``metadata_truncated.csv``, ``metadata_updated.csv``, ``metadata_weird*.csv``)
    left over from another group's filtering, so ``manifest`` is configurable
    rather than hardcoded.
    """

    _TEXT_SOURCES = {
        "shot_immersion": "ShotImmersion",
        "scene_summary": "SceneSummary",
        "scene_description": "SceneDescription",
        "camera_motion": "CameraMotion",
    }

    def __init__(
        self,
        data_root: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        manifest: str = "metadata.csv",
        text_source: str = "shot_immersion",
        max_samples: int = 0,
        prompt: str = "",
        require_video_file: bool = False,
        **geometry: Any,
    ) -> None:
        text_source = text_source or "shot_immersion"
        if text_source not in self._TEXT_SOURCES:
            raise ValueError(
                f"spatialvid text_source must be one of {tuple(self._TEXT_SOURCES)} "
                f"(got {text_source!r})"
            )
        rows = _read_table(os.path.join(data_root, manifest or "metadata.csv"))
        self._root = data_root
        self._text_source = text_source
        self._prompt = prompt
        # Only the default field lives in the manifest; the others are read per
        # sample, so there is nothing to keep in memory for them.
        keep_caption = text_source == "shot_immersion"

        self._rel_paths: list[str] = []
        self._captions: list[str] = []
        missing = 0
        for row in rows:
            rel = str(row["video_path"])
            if require_video_file and not os.path.exists(os.path.join(data_root, rel)):
                missing += 1
                continue
            self._rel_paths.append(rel)
            if keep_caption:
                self._captions.append(str(row.get("caption") or ""))
            if max_samples and len(self._rel_paths) >= max_samples:
                break

        _warn_missing("SpatialVIDDataset", missing)
        _log_built(f"SpatialVIDDataset[{text_source}]", data_root, manifest, self._rel_paths)
        super().__init__(tokenizer_path=tokenizer_path, max_text_len=max_text_len, **geometry)

    def __len__(self) -> int:
        return len(self._rel_paths)

    def _record(self, idx: int) -> VideoRecord:
        rel = self._rel_paths[idx]
        path = os.path.join(self._root, rel)
        if self._text_source == "shot_immersion":
            text = self._captions[idx]
        else:
            text = self.caption_fields(path).get(self._TEXT_SOURCES[self._text_source], "")
        return VideoRecord(path, self._prompt, text)

    def caption_fields(self, video_path: str) -> dict[str, str]:
        """Every caption variant for a clip, from its ``caption.json``.

        Public so a data-inspection pass can compare the fields side by side
        before committing to one; returns ``{}`` when the sidecar is absent.
        """
        # videos/<group>/<id>.mp4 -> annotations/<group>/<id>/caption.json
        group = os.path.basename(os.path.dirname(video_path))
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        payload = _read_json(
            os.path.join(self._root, "annotations", group, video_id, "caption.json")
        )
        if not payload:
            return {}
        return {
            key: str(payload.get(key) or "").strip().replace("\n", " ")
            for key in self._TEXT_SOURCES.values()
        }


class Molmo2CapQADataset(VideoQADataset):
    """Molmo2-VideoCapQA: the released multiple-choice QA over the same clips.

    Annotations are the ``allenai/Molmo2-VideoCapQA`` parquet files, expected at
    ``<data_root>/data/<subset>-*.parquet`` so the HF repo layout drops in next
    to ``videos/``. Two subsets:

    - ``"CapQA"``: one row per question (~5 per video), columns ``video_id``,
      ``Question``, ``Answer``, ``NegativeAnswers`` (3 distractors), ``Category``.
    - ``"LongCapQA"``: one row per video with a nested ``qa_list``, exploded here
      into one sample per question.

    Rows are joined to clips by ``video_id`` against the on-disk tree, so a
    partially-downloaded corpus simply yields fewer samples.

    The correct answer is shuffled into the option list with a seed derived from
    the row itself, so its position varies but is identical on every rank and
    every epoch. Leaving it first would let the model learn the position instead
    of the content.
    """

    _SUBSETS = ("CapQA", "LongCapQA")

    def __init__(
        self,
        data_root: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        subset: str = "CapQA",
        max_samples: int = 0,
        prompt: str = "",
        qa_format: str = "mcq_letter",
        require_video_file: bool = True,
        **geometry: Any,
    ) -> None:
        subset = subset or "CapQA"
        if subset not in self._SUBSETS:
            raise ValueError(f"molmo2_capqa subset must be one of {self._SUBSETS} (got {subset!r})")
        rows = _read_table(os.path.join(data_root, "data", f"{subset}-*.parquet"))
        index = self._video_index(data_root)

        self._items: list[tuple[str, str, str, list[str]]] = []
        missing = 0
        for row in rows:
            video_id = str(row["video_id"])
            path = index.get(video_id)
            if path is None:
                missing += 1
                if require_video_file:
                    continue
            # CapQA is one question per row; LongCapQA nests them in qa_list.
            questions = row["qa_list"] if subset == "LongCapQA" else [row]
            for qa in questions:
                self._items.append(
                    (
                        path or "",
                        str(qa["Question"]),
                        str(qa["Answer"]),
                        [str(x) for x in qa["NegativeAnswers"]],
                    )
                )
                if max_samples and len(self._items) >= max_samples:
                    break
            if max_samples and len(self._items) >= max_samples:
                break

        self._prompt = prompt
        self._qa_format = qa_format
        if missing:
            logger.info(
                "Molmo2CapQADataset: %d rows have no local clip (%d samples kept)",
                missing,
                len(self._items),
            )
        _log_built(f"Molmo2CapQADataset[{subset}]", data_root, subset, self._items)
        super().__init__(tokenizer_path=tokenizer_path, max_text_len=max_text_len, **geometry)

    @staticmethod
    def _video_index(data_root: str) -> dict[str, str]:
        """``video_id`` -> clip path, over every subset plus the ``youtube/`` tree."""
        videos_dir = os.path.join(data_root, "videos")
        index: dict[str, str] = {}
        if os.path.isdir(videos_dir):
            for path in Molmo2VideoCapQADataset._load_manifest(videos_dir, subset=""):
                index.setdefault(os.path.splitext(os.path.basename(path))[0], path)
        # LongCapQA clips land in a sibling tree that the download log omits.
        extra = os.path.join(data_root, "youtube")
        if os.path.isdir(extra):
            for entry in os.scandir(extra):
                if not entry.is_dir():
                    continue
                for name in sorted(os.listdir(entry.path)):
                    if name.endswith(_VIDEO_EXTS):
                        index.setdefault(entry.name, os.path.join(entry.path, name))
                        break
        return index

    def __len__(self) -> int:
        return len(self._items)

    def _record(self, idx: int) -> VideoRecord:
        path, question, answer, negatives = self._items[idx]
        options, answer_index = _shuffled_options(answer, negatives, f"{path}|{question}")
        text = format_multiple_choice(
            self._qa_format,
            question=question,
            options=options,
            answer_index=answer_index,
            instruction=self._prompt,
        )
        return VideoRecord(path, text.prompt, text.target)


class PerceptionTestDataset(VideoQADataset):
    """PerceptionTest multiple-choice QA over flat ``videos/<video_id>.mp4``.

    ``mc_question_<split>.json`` maps a video id to its metadata and a list of
    questions, each with three ``options`` and an ``answer_id`` index; one
    question becomes one sample, so a video recurs once per question.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        max_samples: int = 0,
        prompt: str = "",
        qa_format: str = "mcq_letter",
        require_video_file: bool = True,
        **geometry: Any,
    ) -> None:
        manifest = os.path.join(data_root, f"mc_question_{split}.json")
        if not os.path.exists(manifest):
            raise FileNotFoundError(f"No PerceptionTest manifest at {manifest!r} (split={split!r})")
        with open(manifest) as f:
            annotations = json.load(f)
        videos_dir = os.path.join(data_root, "videos")

        records: list[VideoRecord] = []
        missing = 0
        for video_id in sorted(annotations):
            path = os.path.join(videos_dir, f"{video_id}.mp4")
            if require_video_file and not os.path.exists(path):
                missing += 1
                continue
            for question in annotations[video_id].get("mc_question") or []:
                text = format_multiple_choice(
                    qa_format,
                    question=str(question["question"]),
                    options=[str(o) for o in question["options"]],
                    answer_index=int(question["answer_id"]),
                    instruction=prompt,
                )
                records.append(VideoRecord(path, text.prompt, text.target))
                if max_samples and len(records) >= max_samples:
                    break
            if max_samples and len(records) >= max_samples:
                break

        _warn_missing("PerceptionTestDataset", missing)
        _log_built("PerceptionTestDataset", data_root, split, records)
        super().__init__(
            records, tokenizer_path=tokenizer_path, max_text_len=max_text_len, **geometry
        )


class NExTQADataset(VideoQADataset):
    """NExT-QA over flat ``videos/<video>.mp4``.

    ``subset="MC"`` reads ``annotations/MC/<split>.csv`` — options live in the
    ``a0``..``a4`` columns and ``answer`` is an index into them. ``subset="OE"``
    reads ``annotations/OE/<split>-*.parquet``, whose ``answer`` is free text
    with no options.
    """

    # The two subsets spell the validation split differently on disk.
    _MC_SPLITS = {"train": "train", "validation": "val", "test": "test"}
    _OE_SPLITS = {"train": "train", "validation": "validation", "test": "test"}
    _OPTION_COLS = ("a0", "a1", "a2", "a3", "a4")

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        subset: str = "MC",
        max_samples: int = 0,
        prompt: str = "",
        qa_format: str = "mcq_letter",
        require_video_file: bool = True,
        **geometry: Any,
    ) -> None:
        subset = (subset or "MC").upper()
        if subset not in ("MC", "OE"):
            raise ValueError(f"nextqa subset must be 'MC' or 'OE' (got {subset!r})")
        splits = self._MC_SPLITS if subset == "MC" else self._OE_SPLITS
        if split not in splits:
            raise ValueError(f"nextqa split must be one of {tuple(splits)} (got {split!r})")

        pattern = (
            os.path.join(data_root, "annotations", "MC", f"{splits[split]}.csv")
            if subset == "MC"
            else os.path.join(data_root, "annotations", "OE", f"{splits[split]}-*.parquet")
        )
        rows = _read_table(pattern)
        videos_dir = os.path.join(data_root, "videos")

        records: list[VideoRecord] = []
        missing = 0
        for row in rows:
            path = os.path.join(videos_dir, f"{row['video']}.mp4")
            if require_video_file and not os.path.exists(path):
                missing += 1
                continue
            question = str(row["question"])
            if subset == "MC":
                text = format_multiple_choice(
                    qa_format,
                    question=question,
                    options=[str(row[c]) for c in self._OPTION_COLS],
                    answer_index=int(row["answer"]),
                    instruction=prompt,
                )
            else:
                text = format_open_ended(
                    question=question, answer=str(row["answer"]), instruction=prompt
                )
            records.append(VideoRecord(path, text.prompt, text.target))
            if max_samples and len(records) >= max_samples:
                break

        _warn_missing("NExTQADataset", missing)
        _log_built(f"NExTQADataset[{subset}]", data_root, split, records)
        super().__init__(
            records, tokenizer_path=tokenizer_path, max_text_len=max_text_len, **geometry
        )


class CinePileDataset(VideoQADataset):
    """CinePile movie-clip QA over flat ``videos/<youtube_id>.mp4``.

    Annotations are parquet shards under ``v1``/``v2`` (``dataset_name``).
    ``choices`` is a list column and ``answer_key`` is the answer text; ``v2``
    carries an explicit ``videoID`` while ``v1`` needs it parsed out of
    ``yt_clip_link``.

    Only a fraction of the YouTube clips may be downloaded, so
    ``require_video_file`` defaults to on here — otherwise most samples would
    decode to a blank clip and contribute no gradient.
    """

    _SPLITS = {"train": "train", "test": "test"}

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        dataset_name: str = "v2",
        max_samples: int = 0,
        prompt: str = "",
        qa_format: str = "mcq_letter",
        require_video_file: bool = True,
        **geometry: Any,
    ) -> None:
        if split not in self._SPLITS:
            raise ValueError(
                f"cinepile split must be one of {tuple(self._SPLITS)} (got {split!r}); "
                "the corpus ships no validation split."
            )
        rows = _read_table(os.path.join(data_root, dataset_name, f"{split}-*.parquet"))
        videos_dir = os.path.join(data_root, "videos")

        records: list[VideoRecord] = []
        missing = 0
        for row in rows:
            video_id = row.get("videoID") or _youtube_id(str(row.get("yt_clip_link") or ""))
            if not video_id:
                continue
            path = os.path.join(videos_dir, f"{video_id}.mp4")
            if require_video_file and not os.path.exists(path):
                missing += 1
                continue
            options = [str(c) for c in row["choices"]]
            text = format_multiple_choice(
                qa_format,
                question=str(row["question"]),
                options=options,
                answer_index=int(row["answer_key_position"]),
                instruction=prompt,
            )
            records.append(VideoRecord(path, text.prompt, text.target))
            if max_samples and len(records) >= max_samples:
                break

        if missing:
            logger.info(
                "CinePileDataset: dropped %d rows whose clip is not downloaded (%d kept)",
                missing,
                len(records),
            )
        _log_built(f"CinePileDataset[{dataset_name}]", data_root, split, records)
        super().__init__(
            records, tokenizer_path=tokenizer_path, max_text_len=max_text_len, **geometry
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _read_json(path: str) -> dict[str, Any] | None:
    """Parse a JSON sidecar, returning ``None`` if it is absent or malformed."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("unreadable sidecar %s: %s", path, e)
        return None
    return payload if isinstance(payload, dict) else None


def _read_table(pattern: str) -> list[dict[str, Any]]:
    """Read CSV or parquet file(s) matching ``pattern`` into row dicts.

    ``pattern`` may name a single file or glob several shards; shards are read
    in sorted order so the row order is deterministic across ranks.
    """
    import pandas as pd

    # Stat the literal path first: a real single-file manifest whose path happens
    # to contain a glob metacharacter (``data_root`` like ``/data/set[v2]/...``)
    # must not be reinterpreted as a pattern. Only fall back to globbing when the
    # literal does not exist, which is exactly the shard-pattern case.
    files = [pattern] if os.path.exists(pattern) else sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No annotation files matching {pattern!r}")
    frames = [
        pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f, dtype={"video": str})
        for f in files
    ]
    return pd.concat(frames, ignore_index=True).to_dict("records")


def _shuffled_options(answer: str, negatives: list[str], seed_text: str) -> tuple[list[str], int]:
    """Interleave ``answer`` among ``negatives`` at a stable pseudo-random position.

    Datasets that list the correct answer first would otherwise teach the model
    the position rather than the content. The seed comes from the row's own text
    via CRC32 (Python's ``hash`` is salted per process, so it would disagree
    across DP ranks), making the layout identical on every rank and every epoch.
    """
    options = [answer, *negatives]
    rng = random.Random(zlib.crc32(seed_text.encode("utf-8")))
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(0)


def _youtube_id(url: str) -> str:
    """YouTube video id from any common URL shape; ``""`` if not present.

    Handles watch URLs (``...watch?v=<id>``, parsed so an earlier query param
    like ``pv=1`` cannot be mistaken for the id), short links (``youtu.be/<id>``),
    and ``/embed/<id>`` / ``/shorts/<id>`` / ``/v/<id>`` path forms. Scheme-less
    inputs (``youtu.be/<id>``) are handled too. CinePile v1 rows carry only
    ``yt_clip_link``, so a wrong parse silently points at the wrong ``videos/<id>.mp4``.
    """
    from urllib.parse import parse_qs, urlparse

    raw = url.strip()
    if not raw:
        return ""
    # Give a scheme-less URL a netloc so urlparse populates host/path correctly.
    if "://" not in raw and not raw.startswith("//"):
        raw = "//" + raw
    parsed = urlparse(raw)
    if parsed.netloc.lower().endswith("youtu.be"):
        return parsed.path.lstrip("/").split("/", 1)[0]
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in ("embed", "shorts", "v"):
        return parts[1]
    values = parse_qs(parsed.query).get("v")
    return values[0] if values else ""


def _warn_missing(name: str, missing: int) -> None:
    if missing:
        logger.warning("%s: skipped %d rows whose video file is missing", name, missing)


# ---------------------------------------------------------------------------
# Registry builders — selected by [video].dataset_type / [[video.datasets]].type
# ---------------------------------------------------------------------------


def _require_video_file(video_config: Any, default: bool) -> bool:
    """Resolve the tri-state ``require_video_file`` knob against a corpus default."""
    configured = getattr(video_config, "require_video_file", None)
    return default if configured is None else bool(configured)


@registry.register_video_dataset("molmo2_videocapqa")
def _build_molmo2_videocapqa(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    frame_selector: Any | None = None,
) -> Molmo2VideoCapQADataset:
    """Registry builder for Molmo2-VideoCapQA (see ``Molmo2VideoCapQADataset``)."""
    return Molmo2VideoCapQADataset(
        data_root=video_config.data_root,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        subset=video_config.subset,
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        text_source=video_config.text_source or "title",
        **_geometry(video_config, frame_selector),
    )


@registry.register_video_dataset("spatialvid")
def _build_spatialvid(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    frame_selector: Any | None = None,
) -> SpatialVIDDataset:
    """Registry builder for SpatialVID (see ``SpatialVIDDataset``)."""
    # dataset_name selects the manifest CSV; the [video] default names the
    # WebVid corpus, so fall back to SpatialVID's own.
    manifest = video_config.dataset_name
    return SpatialVIDDataset(
        data_root=video_config.data_root,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        manifest=manifest if manifest.endswith(".csv") else "metadata.csv",
        text_source=video_config.text_source or "shot_immersion",
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        require_video_file=_require_video_file(video_config, default=False),
        **_geometry(video_config, frame_selector),
    )


@registry.register_video_dataset("molmo2_capqa")
def _build_molmo2_capqa(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    frame_selector: Any | None = None,
) -> Molmo2CapQADataset:
    """Registry builder for the released Molmo2 QA (see ``Molmo2CapQADataset``)."""
    return Molmo2CapQADataset(
        data_root=video_config.data_root,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        subset=video_config.subset or "CapQA",
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        qa_format=video_config.qa_format,
        require_video_file=_require_video_file(video_config, default=True),
        **_geometry(video_config, frame_selector),
    )


@registry.register_video_dataset("perception_test")
def _build_perception_test(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    frame_selector: Any | None = None,
) -> PerceptionTestDataset:
    """Registry builder for PerceptionTest (see ``PerceptionTestDataset``)."""
    return PerceptionTestDataset(
        data_root=video_config.data_root,
        split=video_config.split,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        qa_format=video_config.qa_format,
        require_video_file=_require_video_file(video_config, default=True),
        **_geometry(video_config, frame_selector),
    )


@registry.register_video_dataset("nextqa")
def _build_nextqa(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    frame_selector: Any | None = None,
) -> NExTQADataset:
    """Registry builder for NExT-QA (see ``NExTQADataset``)."""
    return NExTQADataset(
        data_root=video_config.data_root,
        split=video_config.split,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        subset=video_config.subset or "MC",
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        qa_format=video_config.qa_format,
        require_video_file=_require_video_file(video_config, default=True),
        **_geometry(video_config, frame_selector),
    )


@registry.register_video_dataset("cinepile")
def _build_cinepile(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    frame_selector: Any | None = None,
) -> CinePileDataset:
    """Registry builder for CinePile (see ``CinePileDataset``)."""
    # dataset_name selects the annotation revision; the [video] default names
    # the WebVid corpus, so fall back to the newest CinePile release.
    revision = video_config.dataset_name
    return CinePileDataset(
        data_root=video_config.data_root,
        split=video_config.split,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        dataset_name=revision if revision in ("v1", "v2") else "v2",
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        qa_format=video_config.qa_format,
        require_video_file=_require_video_file(video_config, default=True),
        **_geometry(video_config, frame_selector),
    )
