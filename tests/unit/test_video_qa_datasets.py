"""Unit tests for the video caption / QA corpus adapters.

Each corpus gets a synthetic on-disk layout in ``tmp_path`` matching the real
one, so manifest parsing, video-path mapping, and prompt/target rendering are
exercised without any real corpus. Frame decoding is stubbed at
``video_dataset.decode_video_frames`` (the module-scope patch point the existing
video tests use), so no ``av`` and no real ``.mp4`` are needed.

The gpt2 tokenizer matches the existing video/VLM dataset tests.
"""

from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from kempnerforge.data import video_dataset as vd
from kempnerforge.data.video_qa_datasets import (
    CinePileDataset,
    Molmo2VideoCapQADataset,
    NExTQADataset,
    PerceptionTestDataset,
)

GEOMETRY = {"max_frames": 4, "min_frames": 2, "fps": 2.0, "frame_size": 16}
TOKENIZER = "gpt2"


@pytest.fixture
def stub_decoder(monkeypatch):
    """Decode to 4 tiny frames at 0/1/2/3 s, no real video needed."""

    def _decode(*args, **kwargs):
        frames = [Image.new("RGB", (16, 16), color=(i * 40 % 255, 0, 0)) for i in range(4)]
        return frames, [0.0, 1.0, 2.0, 3.0]

    monkeypatch.setattr(vd, "decode_video_frames", _decode)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


# ---------------------------------------------------------------------------
# PerceptionTest — nested JSON, one sample per question
# ---------------------------------------------------------------------------


def _perception_root(tmp_path, n_videos=2, n_questions=3, with_videos=True):
    annotations = {}
    for v in range(n_videos):
        vid = f"video_{v}"
        annotations[vid] = {
            "metadata": {"video_id": vid, "split": "train"},
            "mc_question": [
                {
                    "id": q,
                    "question": f"question {q} about {vid}?",
                    "options": ["opt a", "opt b", "opt c"],
                    "answer_id": q % 3,
                    "area": "physics",
                    "reasoning": "descriptive",
                    "tag": ["motion"],
                }
                for q in range(n_questions)
            ],
        }
        if with_videos:
            _touch(tmp_path / "videos" / f"{vid}.mp4")
    (tmp_path / "mc_question_train.json").write_text(json.dumps(annotations))
    return tmp_path


class TestPerceptionTest:
    def test_one_sample_per_question(self, tmp_path, stub_decoder):
        ds = PerceptionTestDataset(
            str(_perception_root(tmp_path, n_videos=2, n_questions=3)),
            "train",
            TOKENIZER,
            64,
            **GEOMETRY,
        )
        assert len(ds) == 6  # 2 videos x 3 questions

    def test_record_renders_question_and_letter(self, tmp_path, stub_decoder):
        ds = PerceptionTestDataset(
            str(_perception_root(tmp_path, n_videos=1, n_questions=3)),
            "train",
            TOKENIZER,
            64,
            **GEOMETRY,
        )
        # answer_id cycles 0,1,2 -> letters A,B,C
        assert [ds._record(i).target for i in range(3)] == [" A", " B", " C"]
        assert "question 0 about video_0?" in ds._record(0).prompt
        assert ds._record(0).video_path.endswith("videos/video_0.mp4")

    def test_missing_videos_are_dropped(self, tmp_path, stub_decoder):
        root = _perception_root(tmp_path, n_videos=2, n_questions=2, with_videos=False)
        ds = PerceptionTestDataset(str(root), "train", TOKENIZER, 64, **GEOMETRY)
        assert len(ds) == 0

    def test_require_video_file_off_keeps_rows(self, tmp_path, stub_decoder):
        root = _perception_root(tmp_path, n_videos=2, n_questions=2, with_videos=False)
        ds = PerceptionTestDataset(
            str(root), "train", TOKENIZER, 64, require_video_file=False, **GEOMETRY
        )
        assert len(ds) == 4

    def test_max_samples_caps(self, tmp_path, stub_decoder):
        ds = PerceptionTestDataset(
            str(_perception_root(tmp_path, n_videos=3, n_questions=4)),
            "train",
            TOKENIZER,
            64,
            max_samples=5,
            **GEOMETRY,
        )
        assert len(ds) == 5

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="PerceptionTest manifest"):
            PerceptionTestDataset(str(tmp_path), "train", TOKENIZER, 64, **GEOMETRY)

    def test_sample_shape(self, tmp_path, stub_decoder):
        ds = PerceptionTestDataset(
            str(_perception_root(tmp_path, n_videos=1, n_questions=1)),
            "train",
            TOKENIZER,
            64,
            **GEOMETRY,
        )
        item = ds[0]
        assert item["pixel_values"].shape == (4, 3, 16, 16)
        assert item["frame_mask"].tolist() == [True] * 4
        assert item["frame_times"].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert item["input_ids"].shape == (64,)
        assert (item["labels"] != -100).any()  # the answer letter is supervised


# ---------------------------------------------------------------------------
# NExT-QA — MC csv (5 option columns, index answer) and OE parquet (text answer)
# ---------------------------------------------------------------------------


def _nextqa_root(tmp_path, subset="MC", split="train", n=3, with_videos=True):
    import pandas as pd

    rows = []
    for i in range(n):
        base = {
            "video": f"{1000 + i}",
            "frame_count": 100,
            "width": 320,
            "height": 240,
            "question": f"what happens in clip {i}",
            "qid": i,
            "type": "DC",
        }
        if subset == "MC":
            base |= {"answer": i % 5, **{f"a{j}": f"choice {j}" for j in range(5)}}
        else:
            base |= {"answer": f"free answer {i}", "additional_ref_answer": None}
        rows.append(base)
        if with_videos:
            _touch(tmp_path / "videos" / f"{1000 + i}.mp4")
    out = tmp_path / "annotations" / subset
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if subset == "MC":
        df.to_csv(out / f"{split}.csv", index=False)
    else:
        df.to_parquet(out / f"{split}-00000-of-00001.parquet", index=False)
    return tmp_path


class TestNExTQA:
    def test_mc_reads_option_columns_and_index_answer(self, tmp_path, stub_decoder):
        ds = NExTQADataset(
            str(_nextqa_root(tmp_path, "MC", "train", n=3)),
            "train",
            TOKENIZER,
            96,
            subset="MC",
            **GEOMETRY,
        )
        assert len(ds) == 3
        assert [ds._record(i).target for i in range(3)] == [" A", " B", " C"]
        prompt = ds._record(0).prompt
        for letter, j in zip("ABCDE", range(5), strict=True):
            assert f"{letter}. choice {j}" in prompt

    def test_oe_reads_text_answer_and_has_no_options(self, tmp_path, stub_decoder):
        ds = NExTQADataset(
            str(_nextqa_root(tmp_path, "OE", "train", n=2)),
            "train",
            TOKENIZER,
            96,
            subset="OE",
            **GEOMETRY,
        )
        assert ds._record(0).target == " free answer 0"
        assert "A." not in ds._record(0).prompt

    def test_validation_maps_to_val_csv_for_mc(self, tmp_path, stub_decoder):
        # MC spells the split "val" on disk; OE spells it "validation".
        ds = NExTQADataset(
            str(_nextqa_root(tmp_path, "MC", "val", n=2)),
            "validation",
            TOKENIZER,
            96,
            subset="MC",
            **GEOMETRY,
        )
        assert len(ds) == 2

    def test_validation_maps_to_validation_parquet_for_oe(self, tmp_path, stub_decoder):
        ds = NExTQADataset(
            str(_nextqa_root(tmp_path, "OE", "validation", n=2)),
            "validation",
            TOKENIZER,
            96,
            subset="OE",
            **GEOMETRY,
        )
        assert len(ds) == 2

    def test_video_path_is_flat(self, tmp_path, stub_decoder):
        ds = NExTQADataset(
            str(_nextqa_root(tmp_path, "MC", "train", n=1)),
            "train",
            TOKENIZER,
            96,
            subset="MC",
            **GEOMETRY,
        )
        assert ds._record(0).video_path.endswith("videos/1000.mp4")

    def test_bad_subset_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="subset must be"):
            NExTQADataset(str(tmp_path), "train", TOKENIZER, 96, subset="XX", **GEOMETRY)

    def test_bad_split_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="split must be"):
            NExTQADataset(str(tmp_path), "nope", TOKENIZER, 96, subset="MC", **GEOMETRY)

    def test_missing_annotations_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No annotation files"):
            NExTQADataset(str(tmp_path), "train", TOKENIZER, 96, subset="MC", **GEOMETRY)


# ---------------------------------------------------------------------------
# CinePile — sharded parquet, list-valued choices, v1/v2 id handling
# ---------------------------------------------------------------------------


def _cinepile_root(tmp_path, revision="v2", n=4, n_shards=2, present=None):
    import pandas as pd

    present = range(n) if present is None else present
    rows = []
    for i in range(n):
        vid = f"ytid{i:03d}"
        row = {
            "movie_name": f"Movie {i}",
            "year": 2000 + i,
            "yt_clip_link": f"https://youtube.com/watch?v={vid}&t=1",
            "subtitles": "<subtitle> hi",
            "question": f"what happens in scene {i}?",
            "choices": [f"choice {j}" for j in range(5)],
            "answer_key": f"choice {i % 5}",
            "answer_key_position": i % 5,
            "question_category": "Plot",
        }
        if revision == "v2":
            row["videoID"] = vid
        rows.append(row)
        if i in present:
            _touch(tmp_path / "videos" / f"{vid}.mp4")
    out = tmp_path / revision
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    per = max(1, len(df) // n_shards)
    for s in range(n_shards):
        chunk = df.iloc[s * per : (s + 1) * per if s < n_shards - 1 else len(df)]
        chunk.to_parquet(out / f"train-{s:05d}-of-{n_shards:05d}.parquet", index=False)
    return tmp_path


class TestCinePile:
    def test_reads_sharded_parquet(self, tmp_path, stub_decoder):
        ds = CinePileDataset(
            str(_cinepile_root(tmp_path, n=4, n_shards=2)),
            "train",
            TOKENIZER,
            192,
            **GEOMETRY,
        )
        assert len(ds) == 4

    def test_list_choices_render_as_letters(self, tmp_path, stub_decoder):
        ds = CinePileDataset(
            str(_cinepile_root(tmp_path, n=1)), "train", TOKENIZER, 192, **GEOMETRY
        )
        assert ds._record(0).target == " A"  # answer_key_position 0
        for letter, j in zip("ABCDE", range(5), strict=True):
            assert f"{letter}. choice {j}" in ds._record(0).prompt

    def test_v1_derives_video_id_from_the_link(self, tmp_path, stub_decoder):
        # v1 has no videoID column, so the id comes out of yt_clip_link's ?v=.
        ds = CinePileDataset(
            str(_cinepile_root(tmp_path, revision="v1", n=2)),
            "train",
            TOKENIZER,
            192,
            dataset_name="v1",
            **GEOMETRY,
        )
        assert ds._record(0).video_path.endswith("videos/ytid000.mp4")

    def test_undownloaded_clips_are_dropped_by_default(self, tmp_path, stub_decoder):
        # Only clips 0 and 2 exist on disk; the corpus is partially downloaded.
        ds = CinePileDataset(
            str(_cinepile_root(tmp_path, n=4, present={0, 2})),
            "train",
            TOKENIZER,
            192,
            **GEOMETRY,
        )
        assert len(ds) == 2
        assert {r.video_path.split("/")[-1] for r in [ds._record(0), ds._record(1)]} == {
            "ytid000.mp4",
            "ytid002.mp4",
        }

    def test_require_video_file_off_keeps_everything(self, tmp_path, stub_decoder):
        ds = CinePileDataset(
            str(_cinepile_root(tmp_path, n=4, present={0})),
            "train",
            TOKENIZER,
            192,
            require_video_file=False,
            **GEOMETRY,
        )
        assert len(ds) == 4

    def test_validation_split_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="ships no validation split"):
            CinePileDataset(str(tmp_path), "validation", TOKENIZER, 192, **GEOMETRY)


# ---------------------------------------------------------------------------
# Molmo2-VideoCapQA — per-video directories, sidecar text, mixed containers
# ---------------------------------------------------------------------------


def _molmo2_root(tmp_path, n=3, subset="youtube-cc-kw", write_log=True, ext=".mp4", sidecar="json"):
    rows = []
    for i in range(n):
        vid = f"vid{i:03d}"
        d = tmp_path / "videos" / subset / vid
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{vid}{ext}").write_bytes(b"")
        if sidecar == "json":
            (d / f"{vid}.json").write_text(
                json.dumps({"title": f"title {i}", "description": f"description {i} " + "x" * 999})
            )
        elif sidecar == "info":
            (d / f"{vid}.info.json").write_text(json.dumps({"title": f"info title {i}"}))
        elif sidecar == "grover":
            # Real key names, verified against the corpus: the denoised text is
            # `cleanasr`, the raw ASR it came from is `noisyasr`.
            (d / f"{vid}.grover.json").write_text(
                json.dumps({"denoised": [{"noisyasr": f"noisy {i}", "cleanasr": f"clean {i}"}]})
            )
        elif sidecar == "grover_noisy_only":
            (d / f"{vid}.grover.json").write_text(
                json.dumps({"denoised": [{"noisyasr": f"noisy {i}"}]})
            )
        elif sidecar == "grover_segments":
            (d / f"{vid}.grover.json").write_text(
                json.dumps({"denoised": [{"cleanasr": f"part {i}a"}, {"cleanasr": f"part {i}b"}]})
            )
        rows.append(f"2026-08-01T00:00:0{i},{vid},{subset}/{vid}/{vid}{ext},100,ok")
    if write_log:
        (tmp_path / "videos" / "download_log.csv").write_text(
            "timestamp,video_id,rel_path,bytes,status\r\n" + "\r\n".join(rows) + "\r\n"
        )
    return tmp_path


class TestMolmo2VideoCapQA:
    def test_manifest_from_download_log(self, tmp_path, stub_decoder):
        ds = Molmo2VideoCapQADataset(str(_molmo2_root(tmp_path, n=3)), TOKENIZER, 64, **GEOMETRY)
        assert len(ds) == 3
        assert ds._record(0).target == "title 0"

    def test_falls_back_to_directory_scan_without_a_log(self, tmp_path, stub_decoder):
        ds = Molmo2VideoCapQADataset(
            str(_molmo2_root(tmp_path, n=2, write_log=False)), TOKENIZER, 64, **GEOMETRY
        )
        assert len(ds) == 2

    def test_log_rows_for_other_subsets_are_filtered(self, tmp_path, stub_decoder):
        root = _molmo2_root(tmp_path, n=2, subset="youtube-cc-kw")
        _molmo2_root(root, n=3, subset="youtube-cc-temporal", write_log=False)
        # The log only lists the kw rows, and subset= narrows further.
        assert (
            len(
                Molmo2VideoCapQADataset(
                    str(root), TOKENIZER, 64, subset="youtube-cc-kw", **GEOMETRY
                )
            )
            == 2
        )

    @pytest.mark.parametrize("ext", [".mp4", ".mkv", ".webm"])
    def test_mixed_containers_are_found(self, tmp_path, stub_decoder, ext):
        ds = Molmo2VideoCapQADataset(
            str(_molmo2_root(tmp_path, n=1, ext=ext)), TOKENIZER, 64, **GEOMETRY
        )
        assert ds._record(0).video_path.endswith(ext)

    def test_text_source_description_is_truncated(self, tmp_path, stub_decoder):
        ds = Molmo2VideoCapQADataset(
            str(_molmo2_root(tmp_path, n=1)),
            TOKENIZER,
            64,
            text_source="description",
            description_max_chars=20,
            **GEOMETRY,
        )
        assert ds._record(0).target == "description 0 xxxxxx"

    def test_text_source_asr_prefers_the_denoised_transcript(self, tmp_path, stub_decoder):
        # `cleanasr` (punctuated, denoised) wins over the raw `noisyasr`.
        ds = Molmo2VideoCapQADataset(
            str(_molmo2_root(tmp_path, n=1, sidecar="grover")),
            TOKENIZER,
            64,
            text_source="asr",
            **GEOMETRY,
        )
        assert ds._record(0).target == "clean 0"

    def test_text_source_asr_falls_back_to_noisy(self, tmp_path, stub_decoder):
        ds = Molmo2VideoCapQADataset(
            str(_molmo2_root(tmp_path, n=1, sidecar="grover_noisy_only")),
            TOKENIZER,
            64,
            text_source="asr",
            **GEOMETRY,
        )
        assert ds._record(0).target == "noisy 0"

    def test_text_source_asr_joins_consecutive_segments(self, tmp_path, stub_decoder):
        ds = Molmo2VideoCapQADataset(
            str(_molmo2_root(tmp_path, n=1, sidecar="grover_segments")),
            TOKENIZER,
            64,
            text_source="asr",
            **GEOMETRY,
        )
        assert ds._record(0).target == "part 0a part 0b"

    def test_info_json_sidecar_is_accepted(self, tmp_path, stub_decoder):
        ds = Molmo2VideoCapQADataset(
            str(_molmo2_root(tmp_path, n=1, sidecar="info")), TOKENIZER, 64, **GEOMETRY
        )
        assert ds._record(0).target == "info title 0"

    def test_missing_sidecar_yields_no_supervision(self, tmp_path, stub_decoder):
        ds = Molmo2VideoCapQADataset(
            str(_molmo2_root(tmp_path, n=2, sidecar=None)), TOKENIZER, 64, **GEOMETRY
        )
        assert ds._record(0).target == ""
        # Static shapes are preserved, but the sample contributes no loss.
        item = ds[0]
        assert item["pixel_values"].shape == (4, 3, 16, 16)
        assert (item["labels"] == -100).all()
        assert ds.probe_supervision() == 0.0

    def test_bad_text_source_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="text_source must be"):
            Molmo2VideoCapQADataset(
                str(_molmo2_root(tmp_path, n=1)),
                TOKENIZER,
                64,
                text_source="captions",
                **GEOMETRY,
            )

    def test_missing_videos_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No videos/ directory"):
            Molmo2VideoCapQADataset(str(tmp_path), TOKENIZER, 64, **GEOMETRY)

    def test_prompt_is_masked_from_the_loss(self, tmp_path, stub_decoder):
        ds = Molmo2VideoCapQADataset(
            str(_molmo2_root(tmp_path, n=1)),
            TOKENIZER,
            64,
            prompt="Describe the video:",
            **GEOMETRY,
        )
        item = ds[0]
        supervised = item["labels"] != -100
        text = ds._tokenizer.decode(item["labels"][supervised])
        assert "Describe" not in text
        assert "title 0" in text


# ---------------------------------------------------------------------------
# SpatialVID — flat CSV manifest + per-clip caption.json with four variants
# ---------------------------------------------------------------------------


def _spatialvid_root(tmp_path, n=3, with_videos=True, with_json=True):
    import pandas as pd

    rows = []
    for i in range(n):
        group, vid = f"group_{i // 2:04d}", f"uuid-{i:03d}"
        rel = f"videos/{group}/{vid}.mp4"
        if with_videos:
            _touch(tmp_path / rel)
        rows.append(
            {
                "video_path": rel,
                "caption": f"immersion {i}",  # mirrors ShotImmersion, as upstream does
                "height": 720,
                "width": 1280,
                "fps": 30.0,
                "n_frames": 300,
            }
        )
        if with_json:
            d = tmp_path / "annotations" / group / vid
            d.mkdir(parents=True, exist_ok=True)
            (d / "caption.json").write_text(
                json.dumps(
                    {
                        "ShotImmersion": f"immersion {i}",
                        "SceneSummary": f"summary {i}",
                        "SceneDescription": f"description {i}",
                        "CameraMotion": f"motion {i}",
                        "CategoryTags": {"brightness": "Bright"},
                    }
                )
            )
    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)
    return tmp_path


class TestSpatialVID:
    def test_reads_the_flat_manifest(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import SpatialVIDDataset

        ds = SpatialVIDDataset(str(_spatialvid_root(tmp_path, n=3)), TOKENIZER, 256, **GEOMETRY)
        assert len(ds) == 3

    def test_default_source_comes_from_the_csv_column(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import SpatialVIDDataset

        # No caption.json at all: the default field must still resolve, since it
        # is read from the manifest rather than the per-clip sidecar.
        ds = SpatialVIDDataset(
            str(_spatialvid_root(tmp_path, n=2, with_json=False)), TOKENIZER, 256, **GEOMETRY
        )
        assert ds._record(0).target == "immersion 0"

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("shot_immersion", "immersion 0"),
            ("scene_summary", "summary 0"),
            ("scene_description", "description 0"),
            ("camera_motion", "motion 0"),
        ],
    )
    def test_every_text_source_resolves(self, tmp_path, stub_decoder, source, expected):
        from kempnerforge.data.video_qa_datasets import SpatialVIDDataset

        ds = SpatialVIDDataset(
            str(_spatialvid_root(tmp_path, n=2)),
            TOKENIZER,
            256,
            text_source=source,
            **GEOMETRY,
        )
        assert ds._record(0).target == expected

    def test_video_path_is_manifest_relative(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import SpatialVIDDataset

        ds = SpatialVIDDataset(str(_spatialvid_root(tmp_path, n=1)), TOKENIZER, 256, **GEOMETRY)
        assert ds._record(0).video_path == str(tmp_path / "videos" / "group_0000" / "uuid-000.mp4")

    def test_caption_fields_exposes_every_variant(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import SpatialVIDDataset

        ds = SpatialVIDDataset(str(_spatialvid_root(tmp_path, n=1)), TOKENIZER, 256, **GEOMETRY)
        fields = ds.caption_fields(ds._record(0).video_path)
        assert fields == {
            "ShotImmersion": "immersion 0",
            "SceneSummary": "summary 0",
            "SceneDescription": "description 0",
            "CameraMotion": "motion 0",
        }

    def test_missing_sidecar_yields_no_supervision(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import SpatialVIDDataset

        ds = SpatialVIDDataset(
            str(_spatialvid_root(tmp_path, n=2, with_json=False)),
            TOKENIZER,
            256,
            text_source="scene_description",
            **GEOMETRY,
        )
        assert ds._record(0).target == ""
        assert (ds[0]["labels"] == -100).all()

    def test_require_video_file_drops_absent_clips(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import SpatialVIDDataset

        ds = SpatialVIDDataset(
            str(_spatialvid_root(tmp_path, n=3, with_videos=False)),
            TOKENIZER,
            256,
            require_video_file=True,
            **GEOMETRY,
        )
        assert len(ds) == 0

    def test_bad_text_source_rejected(self, tmp_path):
        from kempnerforge.data.video_qa_datasets import SpatialVIDDataset

        with pytest.raises(ValueError, match="text_source must be one of"):
            SpatialVIDDataset(
                str(_spatialvid_root(tmp_path, n=1)),
                TOKENIZER,
                256,
                text_source="tags",
                **GEOMETRY,
            )

    def test_missing_manifest_raises(self, tmp_path):
        from kempnerforge.data.video_qa_datasets import SpatialVIDDataset

        with pytest.raises(FileNotFoundError, match="No annotation files"):
            SpatialVIDDataset(str(tmp_path), TOKENIZER, 256, **GEOMETRY)


# ---------------------------------------------------------------------------
# Molmo2-VideoCapQA — the released multiple-choice annotations
# ---------------------------------------------------------------------------


def _capqa_root(tmp_path, subset="CapQA", n_videos=3, n_qa=5, present=None):
    """HF-repo layout: data/<subset>-*.parquet beside the videos/ tree."""
    import pandas as pd

    present = range(n_videos) if present is None else present
    _molmo2_root(tmp_path, n=0, write_log=False)  # ensure videos/ exists
    (tmp_path / "videos").mkdir(parents=True, exist_ok=True)
    rows, log = [], []
    for v in range(n_videos):
        vid = f"vid{v:03d}"
        if v in present:
            d = tmp_path / "videos" / "youtube-cc-kw" / vid
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{vid}.mp4").write_bytes(b"")
            log.append(f"2026-08-01T00:00:00,{vid},youtube-cc-kw/{vid}/{vid}.mp4,100,ok")
        qas = [
            {
                "Question": f"q{q} about {vid}?",
                "Answer": f"correct {v}-{q}",
                "NegativeAnswers": [f"wrong {v}-{q}-{k}" for k in range(3)],
                "Category": "object presence",
            }
            for q in range(n_qa)
        ]
        if subset == "CapQA":
            rows.extend({"video_id": vid, **qa} for qa in qas)
        else:
            rows.append({"video_id": vid, "qa_list": qas})
    (tmp_path / "videos" / "download_log.csv").write_text(
        "timestamp,video_id,rel_path,bytes,status\r\n" + "\r\n".join(log) + "\r\n"
    )
    out = tmp_path / "data"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / f"{subset}-00000-of-00001.parquet", index=False)
    return tmp_path


class TestMolmo2CapQA:
    def test_capqa_is_one_sample_per_question(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import Molmo2CapQADataset

        ds = Molmo2CapQADataset(
            str(_capqa_root(tmp_path, n_videos=3, n_qa=5)), TOKENIZER, 192, **GEOMETRY
        )
        assert len(ds) == 15

    def test_longcapqa_qa_list_is_exploded(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import Molmo2CapQADataset

        ds = Molmo2CapQADataset(
            str(_capqa_root(tmp_path, subset="LongCapQA", n_videos=2, n_qa=4)),
            TOKENIZER,
            192,
            subset="LongCapQA",
            **GEOMETRY,
        )
        assert len(ds) == 8

    def test_all_four_options_render_and_target_marks_the_right_one(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import Molmo2CapQADataset

        ds = Molmo2CapQADataset(
            str(_capqa_root(tmp_path, n_videos=1, n_qa=1)), TOKENIZER, 192, **GEOMETRY
        )
        record = ds._record(0)
        # 1 correct + 3 negatives, all present exactly once.
        for opt in ["correct 0-0", *[f"wrong 0-0-{k}" for k in range(3)]]:
            assert record.prompt.count(opt) == 1
        # The target letter indexes the line holding the correct answer.
        letter = record.target.strip()
        line = next(ln for ln in record.prompt.splitlines() if ln.startswith(f"{letter}. "))
        assert line == f"{letter}. correct 0-0"

    def test_answer_position_varies_across_rows(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import Molmo2CapQADataset

        ds = Molmo2CapQADataset(
            str(_capqa_root(tmp_path, n_videos=12, n_qa=5)), TOKENIZER, 192, **GEOMETRY
        )
        letters = {ds._record(i).target.strip() for i in range(len(ds))}
        # Not pinned to "A" — otherwise the model learns position, not content.
        assert len(letters) > 1

    def test_shuffle_is_stable_across_instances(self, tmp_path, stub_decoder):
        # Two builds must agree, or DP ranks would disagree on the same index.
        from kempnerforge.data.video_qa_datasets import Molmo2CapQADataset

        root = str(_capqa_root(tmp_path, n_videos=4, n_qa=5))
        a = Molmo2CapQADataset(root, TOKENIZER, 192, **GEOMETRY)
        b = Molmo2CapQADataset(root, TOKENIZER, 192, **GEOMETRY)
        assert [a._record(i) for i in range(len(a))] == [b._record(i) for i in range(len(b))]

    def test_rows_without_a_local_clip_are_dropped(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import Molmo2CapQADataset

        ds = Molmo2CapQADataset(
            str(_capqa_root(tmp_path, n_videos=4, n_qa=5, present={0, 2})),
            TOKENIZER,
            192,
            **GEOMETRY,
        )
        assert len(ds) == 10

    def test_max_samples_caps(self, tmp_path, stub_decoder):
        from kempnerforge.data.video_qa_datasets import Molmo2CapQADataset

        ds = Molmo2CapQADataset(
            str(_capqa_root(tmp_path, n_videos=4, n_qa=5)),
            TOKENIZER,
            192,
            max_samples=7,
            **GEOMETRY,
        )
        assert len(ds) == 7

    def test_bad_subset_rejected(self, tmp_path):
        from kempnerforge.data.video_qa_datasets import Molmo2CapQADataset

        with pytest.raises(ValueError, match="subset must be one of"):
            Molmo2CapQADataset(str(tmp_path), TOKENIZER, 192, subset="Nope", **GEOMETRY)

    def test_missing_annotations_raise(self, tmp_path):
        from kempnerforge.data.video_qa_datasets import Molmo2CapQADataset

        with pytest.raises(FileNotFoundError, match="No annotation files"):
            Molmo2CapQADataset(str(tmp_path), TOKENIZER, 192, **GEOMETRY)


# ---------------------------------------------------------------------------
# Supervision probe — the shared guard against silently unsupervised corpora
# ---------------------------------------------------------------------------


class TestProbeSupervision:
    """``kempnerforge.metrics.logger._configure_root`` sets ``propagate = False``
    on the ``kempnerforge`` logger, which blocks pytest's caplog (attached at the
    python root). Re-enable propagation around the warning test so it observes
    the record regardless of whether an earlier test configured logging."""

    def setup_method(self):
        import logging

        self._kf_logger = logging.getLogger("kempnerforge")
        self._old_propagate = self._kf_logger.propagate
        self._kf_logger.propagate = True

    def teardown_method(self):
        self._kf_logger.propagate = self._old_propagate

    def test_full_supervision_reports_one(self, tmp_path, stub_decoder):
        ds = Molmo2VideoCapQADataset(str(_molmo2_root(tmp_path, n=4)), TOKENIZER, 64, **GEOMETRY)
        assert ds.probe_supervision() == 1.0

    def test_prompt_longer_than_max_text_len_loses_supervision(self, tmp_path, stub_decoder):
        # A CinePile-shaped prompt that cannot fit leaves no room for the answer.
        ds = CinePileDataset(str(_cinepile_root(tmp_path, n=2)), "train", TOKENIZER, 8, **GEOMETRY)
        assert ds.probe_supervision() == 0.0
        assert (ds[0]["labels"] == -100).all()

    def test_warns_when_mostly_unsupervised(self, tmp_path, stub_decoder, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            Molmo2VideoCapQADataset(
                str(_molmo2_root(tmp_path, n=2, sidecar=None)), TOKENIZER, 64, **GEOMETRY
            )
        assert any("supervise at least one token" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_every_corpus_is_registered(self):
        from kempnerforge.config.registry import registry
        from kempnerforge.config.video import VideoConfig

        VideoConfig()  # populates the registries via its late imports
        assert set(registry.list_video_datasets()) == {
            "webvid",
            "molmo2_videocapqa",
            "molmo2_capqa",
            "spatialvid",
            "perception_test",
            "nextqa",
            "cinepile",
        }

    def test_builders_go_through_build_video_dataset(self, tmp_path, stub_decoder):
        from kempnerforge.config.video import VideoConfig
        from kempnerforge.data.video_dataset import VideoDataset, build_video_dataset

        cfg = VideoConfig(
            data_root=str(_perception_root(tmp_path, n_videos=1, n_questions=2)),
            dataset_type="perception_test",
            split="train",
            **GEOMETRY,
        )
        ds = build_video_dataset(cfg, TOKENIZER, max_text_len=64)
        assert isinstance(ds, VideoDataset)
        assert len(ds) == 2

    def test_collator_forwards_dataset_idx(self):
        from kempnerforge.data.video_dataset import VideoCollator

        def sample(idx):
            return {
                "pixel_values": torch.zeros(2, 3, 4, 4),
                "frame_mask": torch.ones(2, dtype=torch.bool),
                "frame_times": torch.zeros(2),
                "input_ids": torch.zeros(8, dtype=torch.long),
                "labels": torch.full((8,), -100, dtype=torch.long),
                "dataset_idx": idx,
            }

        batch = VideoCollator(pad_id=0, max_text_len=8)([sample(0), sample(2), sample(1)])
        assert batch["dataset_idx"].tolist() == [0, 2, 1]
        assert batch["dataset_idx"].dtype == torch.long

    def test_collator_omits_dataset_idx_without_mixing(self):
        from kempnerforge.data.video_dataset import VideoCollator

        batch = VideoCollator(pad_id=0, max_text_len=8)(
            [
                {
                    "pixel_values": torch.zeros(2, 3, 4, 4),
                    "frame_mask": torch.ones(2, dtype=torch.bool),
                    "frame_times": torch.zeros(2),
                    "input_ids": torch.zeros(8, dtype=torch.long),
                    "labels": torch.zeros(8, dtype=torch.long),
                }
            ]
        )
        assert "dataset_idx" not in batch
