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
- ``llava_video``: the corpus's own ``manifest.jsonl`` plus ``videos/<rel>``.
- ``tempcompass`` / ``mlvu`` / ``perception_test_val``: HuggingFace parquet
  snapshots under ``hub/`` plus a flat per-benchmark video directory.
- ``onevision2``: no per-clip files at all -- clips are members of
  WebDataset ``.tar`` shards, addressed through a prebuilt byte-offset index
  under ``index/`` (see ``OneVision2Dataset``).

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
from kempnerforge.data.frame_selection import CandidatePoolSpec
from kempnerforge.data.qa_format import format_multiple_choice, format_open_ended
from kempnerforge.data.video_dataset import VideoQADataset, VideoRecord

logger = logging.getLogger(__name__)

# Containers PyAV decodes; Molmo2's yt-dlp output mixes all three.
_VIDEO_EXTS = (".mp4", ".mkv", ".webm")


def _geometry(
    video_config: Any, candidate_spec: CandidatePoolSpec | None = None
) -> dict[str, Any]:
    """Frame-geometry kwargs shared by every corpus (global on ``[video]``).

    The optional ``candidate_spec`` (derived from the ``[frame_selector]``
    section at the ``build_video_data`` seam) rides along here so it reaches
    ``VideoQADataset.__init__`` via each dataset's ``**geometry`` forwarding,
    enabling pool-mode query-aware selection for every corpus.
    """
    return {
        "max_frames": video_config.max_frames,
        "min_frames": video_config.min_frames,
        "fps": video_config.fps,
        "frame_size": video_config.frame_size,
        "sampling_policy": video_config.sampling_policy,
        "candidate_spec": candidate_spec,
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
            seed=_render_seed(f"{path}|{question}"),
        )
        return VideoRecord(path, text.prompt, text.target, query=question)


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
                    seed=_render_seed(f'{path}|{question["question"]}'),
                )
                records.append(
                    VideoRecord(path, text.prompt, text.target, query=str(question["question"]))
                )
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
        strata: list[str] = []
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
                    seed=_render_seed(f"{path}|{question}"),
                )
            else:
                text = format_open_ended(
                    question=question, answer=str(row["answer"]), instruction=prompt
                )
            records.append(VideoRecord(path, text.prompt, text.target, query=question))
            # causal (CW/CH), temporal (TN/TC/TP) or descriptive (D*) --
            # the temporal ones are where a time embedding should show up
            strata.append(str(row.get("type") or ""))
            if max_samples and len(records) >= max_samples:
                break

        _warn_missing("NExTQADataset", missing)
        self.strata = strata
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
                seed=_render_seed(f'{path}|{row["question"]}'),
            )
            records.append(
                VideoRecord(path, text.prompt, text.target, query=str(row["question"]))
            )
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



_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class TempCompassDataset(VideoQADataset):
    """TempCompass, the temporal-understanding benchmark, over ``videos/<id>.mp4``.

    Four subsets share one video pool and differ only in how the question is
    posed: ``multi-choice`` and ``yes_no`` carry the options inside ``question``
    already, ``caption_matching`` asks which of two captions fits, and
    ``captioning`` is open-ended with a paired MCQ form in ``mc_question``.
    Because the options are pre-rendered by the benchmark, this loader passes the
    question through verbatim rather than re-formatting it through ``qa_format``
    -- rewriting it would stop the score being comparable to published numbers.

    ``dim`` (direction / order / action / speed / attribute_change) is kept on
    ``self.strata`` so a run can be stratified or analysed per temporal axis.
    """

    _SUBSETS = ("multi-choice", "yes_no", "caption_matching", "captioning")
    # Copied verbatim from lmms_eval/tasks/tempcompass/_default_template_yaml
    # (lmms_eval_specific_kwargs.default.post_prompt). The benchmark's published
    # scores are produced with these exact instructions; substituting a generic
    # "Answer:" changes the model's input distribution and makes our numbers
    # incomparable to anything reported elsewhere.
    _POST_PROMPT = {
        "multi-choice": "\nPlease directly give the best option:",
        "yes_no": "\nPlease answer yes or no:",
        "caption_matching": "\nPlease directly give the best option:",
        "captioning": "",
    }

    def __init__(
        self,
        data_root: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        subset: str = "multi-choice",
        max_samples: int = 0,
        prompt: str = "",
        require_video_file: bool = True,
        **geometry: Any,
    ) -> None:
        subset = subset or "multi-choice"
        if subset not in self._SUBSETS:
            raise ValueError(
                f"tempcompass subset must be one of {self._SUBSETS} (got {subset!r})"
            )
        rows = _read_table(
            os.path.join(data_root, "hub", "datasets--lmms-eval--TempCompass",
                         "snapshots", "*", subset, "*.parquet")
        )
        videos_dir = os.path.join(data_root, "tempcompass", "videos")

        records: list[VideoRecord] = []
        strata: list[str] = []
        missing = 0
        for row in rows:
            path = os.path.join(videos_dir, f"{row['video_id']}.mp4")
            if require_video_file and not os.path.exists(path):
                missing += 1
                continue
            question = str(row["question"])
            post = prompt if prompt else self._POST_PROMPT[subset]
            # `answer` for every subset, matching tempcompass_doc_to_answer.
            # `mc_answer` letters index `mc_question`, which the captioning
            # prompt never shows -- it belongs to the upstream ChatGPT path that
            # re-asks that question about the generated caption. Against the
            # rendered prompt (Information A/B/C, keyed to `answer`) it is a
            # label error: the two disagree on 3.0% of letters.
            answer = str(row["answer"]).strip()
            records.append(VideoRecord(path, f"{question}{post}", f" {answer}"))
            strata.append(str(row.get("dim") or ""))
            if max_samples and len(records) >= max_samples:
                break

        self.strata = strata
        _warn_missing("TempCompassDataset", missing)
        _log_built(f"TempCompassDataset[{subset}]", data_root, subset, records)
        super().__init__(
            records, tokenizer_path=tokenizer_path, max_text_len=max_text_len, **geometry
        )


class MLVUDataset(VideoQADataset):
    """MLVU long-video understanding; videos sit flat as ``mlvu/<video_name>``.

    ``question`` already embeds its options as ``(A) ...`` lines and ``answer``
    is the bare letter, so like TempCompass the text is passed through unchanged.
    Videos here are long (median ~10 min), which matters: at a fixed frame budget
    the sampled frames are minutes apart, so a low score reflects frame coverage
    as much as model quality.

    ``task_type`` (plotQA / needle / ego / order / count / ...) is kept on
    ``self.strata``.
    """

    _POST_PROMPT = "\nOnly give the best option.\nBest option: ("

    def __init__(
        self,
        data_root: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        max_samples: int = 0,
        prompt: str = "",
        require_video_file: bool = True,
        **geometry: Any,
    ) -> None:
        rows = _read_table(
            os.path.join(data_root, "hub", "datasets--sy1998--MLVU_dev",
                         "snapshots", "*", "mlvu", "*.parquet")
        )
        videos_dir = os.path.join(data_root, "mlvu")

        records: list[VideoRecord] = []
        strata: list[str] = []
        missing = 0
        for row in rows:
            name = str(row["video_name"])
            path = os.path.join(videos_dir, name if name.endswith(".mp4") else f"{name}.mp4")
            if require_video_file and not os.path.exists(path):
                missing += 1
                continue
            question = str(row["question"])
            # lmms_eval/tasks/mlvu/mlvu_dev.yaml post_prompt. The trailing open
            # paren is load-bearing: options render as "(A)", so the primer
            # constrains the next token to a bare letter. Replacing it with
            # "Answer:" removes the very constraint that makes MLVU's published
            # numbers reachable by a small model.
            post = prompt if prompt else self._POST_PROMPT
            answer = str(row["answer"]).strip()
            records.append(VideoRecord(path, f"{question}{post}", f" {answer}"))
            strata.append(str(row.get("task_type") or ""))
            if max_samples and len(records) >= max_samples:
                break

        self.strata = strata
        _warn_missing("MLVUDataset", missing)
        _log_built("MLVUDataset", data_root, "dev", records)
        super().__init__(
            records, tokenizer_path=tokenizer_path, max_text_len=max_text_len, **geometry
        )


class PerceptionTestValDataset(VideoQADataset):
    """PerceptionTest validation split in its HuggingFace parquet packaging.

    Distinct from ``PerceptionTestDataset``, which reads the original
    ``mc_question_<split>.json`` layout; this one reads the redistributed parquet
    (``options`` list + ``answer_id`` index) shipped for evaluation. Options are
    raw here, so they ARE rendered through ``qa_format`` -- unlike TempCompass
    and MLVU, whose questions arrive pre-rendered.

    ``area`` and ``reasoning`` are joined into ``self.strata``.
    """

    # lmms_eval/tasks/perceptiontest/val/utils.py. Defining it as an attribute
    # (as TempCompass and MLVU do) is what marks this loader as reproducing the
    # published rendering rather than a training one.
    _POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."

    def __init__(
        self,
        data_root: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        max_samples: int = 0,
        prompt: str = "",
        qa_format: str = "mcq_letter",
        require_video_file: bool = True,
        **geometry: Any,
    ) -> None:
        rows = _read_table(
            os.path.join(data_root, "hub", "datasets--lmms-eval--PerceptionTest_Val",
                         "snapshots", "*", "mc_question_val", "*.parquet")
        )
        videos_dir = os.path.join(data_root, "perceptiontest_val", "videos")

        records: list[VideoRecord] = []
        strata: list[str] = []
        missing = 0
        for row in rows:
            name = str(row["video_name"])
            path = os.path.join(videos_dir, name if name.endswith(".mp4") else f"{name}.mp4")
            if require_video_file and not os.path.exists(path):
                missing += 1
                continue
            # Rendered to match lmms_eval/tasks/perceptiontest/val/utils.py:43-56
            # exactly: no "Question:" prefix, options as "A. text", and the
            # benchmark's own trailing instruction. Routing this through
            # qa_format added a prefix, a different hint AND a second answer cue.
            opts = [str(o) for o in row["options"]]
            question = str(row["question"])
            for i, op in enumerate(opts):
                question += "\n" + f"{_LETTERS[i]}. " + op
            post = prompt if prompt else self._POST_PROMPT
            answer = _LETTERS[int(row["answer_id"])]
            records.append(VideoRecord(path, f"{question}{post}", f" {answer}"))
            strata.append(f"{row.get('area', '')}/{row.get('reasoning', '')}")
            if max_samples and len(records) >= max_samples:
                break

        self.strata = strata
        _warn_missing("PerceptionTestValDataset", missing)
        _log_built("PerceptionTestValDataset", data_root, "validation", records)
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


def _render_seed(seed_text: str) -> int:
    """Stable per-sample seed for ``mcq_varied`` rendering, from the row's text.

    CRC32 of the row's own text, for the same reason ``_shuffled_options`` uses
    it: Python's ``hash`` is salted per process, so it would disagree across DP
    ranks and dataloader workers. Keyed on content rather than position, so the
    rendering survives a reordering of the manifest.

    A sample therefore renders identically in every epoch. Varying it per epoch
    would need rendering moved out of construction and into ``_record``, which
    only ``Molmo2CapQADataset`` does today.
    """
    return zlib.crc32(f"render|{seed_text}".encode("utf-8"))


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
    candidate_spec: CandidatePoolSpec | None = None,
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
        **_geometry(video_config, candidate_spec),
    )


@registry.register_video_dataset("spatialvid")
def _build_spatialvid(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    candidate_spec: CandidatePoolSpec | None = None,
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
        **_geometry(video_config, candidate_spec),
    )


@registry.register_video_dataset("molmo2_capqa")
def _build_molmo2_capqa(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    candidate_spec: CandidatePoolSpec | None = None,
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
        **_geometry(video_config, candidate_spec),
    )


@registry.register_video_dataset("perception_test")
def _build_perception_test(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    candidate_spec: CandidatePoolSpec | None = None,
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
        **_geometry(video_config, candidate_spec),
    )


@registry.register_video_dataset("nextqa")
def _build_nextqa(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    candidate_spec: CandidatePoolSpec | None = None,
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
        **_geometry(video_config, candidate_spec),
    )


@registry.register_video_dataset("cinepile")
def _build_cinepile(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    candidate_spec: CandidatePoolSpec | None = None,
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
        **_geometry(video_config, candidate_spec),
    )


@registry.register_video_dataset("tempcompass")
def _build_tempcompass(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    candidate_spec: CandidatePoolSpec | None = None,
):
    """Registry builder for TempCompass (see ``TempCompassDataset``)."""
    return TempCompassDataset(
        data_root=video_config.data_root,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        subset=video_config.subset or "multi-choice",
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        require_video_file=_require_video_file(video_config, default=True),
        **_geometry(video_config, candidate_spec),
    )


@registry.register_video_dataset("mlvu")
def _build_mlvu(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    candidate_spec: CandidatePoolSpec | None = None,
):
    """Registry builder for MLVU (see ``MLVUDataset``)."""
    return MLVUDataset(
        data_root=video_config.data_root,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        require_video_file=_require_video_file(video_config, default=True),
        **_geometry(video_config, candidate_spec),
    )


@registry.register_video_dataset("perception_test_val")
def _build_perception_test_val(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    candidate_spec: CandidatePoolSpec | None = None,
):
    """Registry builder for the PerceptionTest val parquet (see
    ``PerceptionTestValDataset``)."""
    return PerceptionTestValDataset(
        data_root=video_config.data_root,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        qa_format=video_config.qa_format,
        require_video_file=_require_video_file(video_config, default=True),
        **_geometry(video_config, candidate_spec),
    )


class LLaVAVideoDataset(VideoQADataset):
    """LLaVA-Video-178K captions, read through the corpus's own ``manifest.jsonl``.

    178,508 rows of ``{video, subset, words, caption}`` where ``video`` is a path
    relative to ``videos/``. Captions are long -- 194 to 1,043 median words by
    subset, 470 on average -- so this corpus is worth roughly 40x WebVid's
    supervision per clip at identical vision cost.

    Rows are indexed by byte offset and parsed on access. Holding them in memory
    costs ~500 MB of caption text per process, which is multiplied by every
    dataloader worker on the node.

    Only 95.4% of the videos are ``.mp4`` (3.6% carry no extension, 1.1% are
    ``.mkv``), so paths come from the manifest and are never globbed. The manifest
    is already filtered to files present on disk, hence ``require_video_file``
    defaults to False.

    ``subset`` is exposed on ``self.strata`` -- it encodes video length
    (``0_30_s`` .. ``2_3_m``), which drives how much the frame budget subsamples.
    """

    def __init__(
        self,
        data_root: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        subset: str = "",
        max_samples: int = 0,
        prompt: str = "",
        prompt_pool: list[str] | None = None,
        require_video_file: bool = False,
        **geometry: Any,
    ) -> None:
        self._manifest = os.path.join(data_root, "manifest.jsonl")
        if not os.path.exists(self._manifest):
            raise FileNotFoundError(f"No manifest.jsonl under {data_root!r}")
        self._videos_dir = os.path.join(data_root, "videos")
        self._prompt = prompt
        self._prompt_pool = list(prompt_pool or ())
        self._fh: Any = None

        offsets, strata, missing = [], [], 0
        with open(self._manifest, "rb") as fh:
            pos = fh.tell()
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    keep = (not subset) or row.get("subset", "") == subset
                    if keep and require_video_file and not os.path.exists(
                        os.path.join(self._videos_dir, row["video"])
                    ):
                        missing += 1
                    elif keep:
                        offsets.append(pos)
                        strata.append(str(row.get("subset") or ""))
                pos = fh.tell()
        if not offsets:
            raise FileNotFoundError(
                f"manifest.jsonl yielded no rows for subset={subset!r} under {data_root!r}"
            )
        self._offsets = _cap(offsets, max_samples)
        self.strata = _cap(strata, max_samples)

        super().__init__(tokenizer_path=tokenizer_path, max_text_len=max_text_len, **geometry)
        _warn_missing("LLaVAVideoDataset", missing)
        logger.info(
            "LLaVAVideoDataset: %s [%s], %d clips", data_root, subset or "all subsets",
            len(self._offsets),
        )

    def __len__(self) -> int:
        return len(self._offsets)

    def _row(self, idx: int) -> dict[str, Any]:
        # Reopened per process: a handle inherited across fork shares one file
        # offset, so concurrent workers would read each other's seeks.
        if self._fh is None or getattr(self, "_fh_pid", None) != os.getpid():
            self._fh = open(self._manifest, "rb")
            self._fh_pid = os.getpid()
        self._fh.seek(self._offsets[idx])
        return json.loads(self._fh.readline())

    def _record(self, idx: int) -> VideoRecord:
        row = self._row(idx)
        return VideoRecord(
            os.path.join(self._videos_dir, row["video"]), self._prompt_for(idx),
            f" {str(row['caption']).strip()}",
        )


@registry.register_video_dataset("llava_video")
def _build_llava_video(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    candidate_spec: CandidatePoolSpec | None = None,
):
    """Registry builder for LLaVA-Video-178K (see ``LLaVAVideoDataset``)."""
    return LLaVAVideoDataset(
        data_root=video_config.data_root,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        subset=video_config.subset,
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        prompt_pool=video_config.prompt_pool,
        require_video_file=_require_video_file(video_config, default=False),
        **_geometry(video_config, candidate_spec),
    )


class OneVision2Dataset(VideoQADataset):
    """LLaVA-OneVision-2 captions over WebDataset tar shards, read by byte offset.

    The corpus ships as 20,516 ``.tar`` shards (66 TB) with the clips *inside* the
    archives, so there is no per-clip path to decode. A prebuilt index gives each
    clip's ``(shard, offset, size)``, and the record carries that byte range so
    ``VideoQADataset._decode_source`` reads exactly the one member -- no
    unpacking, which would turn 6.9M clips into 6.9M small files on Lustre.

    Captions live in two 26 GB JSONL files and are likewise addressed by byte
    offset and parsed per access, so no caption text is held in memory.

    ``data_root`` is the *index* root, not the corpus root: it holds
    ``index/train_index.npz`` plus ``index/sources.json``, which records where
    the shards and caption files live. Keeping those paths in the index rather
    than in TOML means a config cannot pair an index with the wrong corpus.

    ``text_source`` picks the rendering: ``"chat"`` (default) keeps the caption's
    own authored instruction -- the corpus carries 76 paraphrases of the same
    six-heading request, and normalizing them away would discard that diversity.
    ``"caption"`` substitutes ``prompt`` and supervises the caption body alone,
    which is what fits a small ``max_text_len``: captions run ~700 tokens (p99
    1027), so anything under ~1024 truncates most of them.
    """

    def __init__(
        self,
        data_root: str,
        tokenizer_path: str,
        max_text_len: int,
        *,
        max_samples: int = 0,
        prompt: str = "",
        text_source: str = "chat",
        **geometry: Any,
    ) -> None:
        import numpy as np  # noqa: PLC0415 - only needed by this corpus

        if text_source not in ("chat", "caption"):
            raise ValueError(
                f"onevision2 text_source must be 'chat' or 'caption' (got {text_source!r})"
            )
        self._text_source = text_source
        self._prompt = prompt
        index_dir = os.path.join(data_root, "index")
        sources_path = os.path.join(index_dir, "sources.json")
        if not os.path.exists(sources_path):
            raise FileNotFoundError(f"No index/sources.json under {data_root!r}")
        with open(sources_path) as fh:
            sources = json.load(fh)
        self._caption_dir = sources["caption_dir"]
        self._cap_files = tuple(sources["caption_files"])

        with open(sources["shard_table"]) as fh:
            self._shards = [row["path"] for row in json.load(fh)]

        npz = os.path.join(index_dir, "train_index.npz")
        if not os.path.exists(npz):
            raise FileNotFoundError(f"No train_index.npz under {index_dir!r}")
        # mmap_mode needs the arrays unpacked; np.load on an npz is lazy per key
        # already, and the whole index is ~200 MB, so read it once per process.
        with np.load(npz) as z:
            keys = ("shard_row", "offset", "size", "cap_file", "cap_offset", "cap_len")
            self._idx = {k: _cap(z[k], max_samples) for k in keys}
        self._fh: dict[int, Any] = {}
        self._fh_pid: int | None = None

        n = len(self._idx["shard_row"])
        if not n:
            raise FileNotFoundError(f"train_index.npz under {index_dir!r} is empty")
        logger.info(
            "OneVision2Dataset: %s, %d clips across %d shards (text_source=%s)",
            index_dir, n, len(self._shards), text_source,
        )
        super().__init__(tokenizer_path=tokenizer_path, max_text_len=max_text_len, **geometry)

    def __len__(self) -> int:
        return len(self._idx["shard_row"])

    def _caption(self, idx: int) -> tuple[str, str]:
        """(instruction, caption) for ``idx``, parsed from one seek+read."""
        # Reopened per process: a handle inherited across fork shares one file
        # offset, so concurrent workers would read each other's seeks.
        if self._fh_pid != os.getpid():
            self._fh = {}
            self._fh_pid = os.getpid()
        file_id = int(self._idx["cap_file"][idx])
        fh = self._fh.get(file_id)
        if fh is None:
            # Deliberately long-lived: one handle per caption file per process,
            # reused across seeks. A context manager would reopen a 15 GB file
            # on every __getitem__.
            fh = open(  # noqa: SIM115
                os.path.join(self._caption_dir, self._cap_files[file_id]), "rb"
            )
            self._fh[file_id] = fh
        fh.seek(int(self._idx["cap_offset"][idx]))
        row = json.loads(fh.read(int(self._idx["cap_len"][idx])))
        messages = row["messages"]
        return str(messages[0]["content"]), str(messages[1]["content"])

    def _record(self, idx: int) -> VideoRecord:
        instruction, caption = self._caption(idx)
        shard = self._shards[int(self._idx["shard_row"][idx])]
        offset, size = int(self._idx["offset"][idx]), int(self._idx["size"][idx])
        prompt = instruction if self._text_source == "chat" else self._prompt
        return VideoRecord(
            # Identity, not a decodable path: the clip is a member of `shard`, so
            # this only has to be unique per clip (it seeds frame selection).
            f"{shard}#{offset}",
            prompt,
            f" {caption.strip()}",
            archive=shard,
            byte_range=(offset, size),
        )


@registry.register_video_dataset("onevision2")
def _build_onevision2(
    video_config: Any,
    tokenizer_path: str,
    max_text_len: int,
    candidate_spec: CandidatePoolSpec | None = None,
):
    """Registry builder for LLaVA-OneVision-2 (see ``OneVision2Dataset``)."""
    return OneVision2Dataset(
        data_root=video_config.data_root,
        tokenizer_path=tokenizer_path,
        max_text_len=max_text_len,
        max_samples=video_config.max_samples,
        prompt=video_config.prompt,
        text_source=video_config.text_source or "chat",
        **_geometry(video_config, candidate_spec),
    )
