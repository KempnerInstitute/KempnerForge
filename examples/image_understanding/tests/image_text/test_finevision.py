"""Tests for the FineVision Arrow converter and reader."""

from __future__ import annotations

import importlib.util
import io
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_text.finevision import FineVisionArrow
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "convert_finevision_to_arrow.py"

SCHEMA = pa.schema(
    [
        ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
        ("texts", pa.list_(pa.struct([("user", pa.string()), ("assistant", pa.string())]))),
        ("source", pa.string()),
        ("relevance_min", pa.int64()),
    ]
)


def _png(width: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, 3), (9, 9, 9)).save(buffer, format="PNG")
    return buffer.getvalue()


def _row(n_images: int, turns: list[tuple[str, str]], rating: int | None, width: int = 4) -> dict:
    return {
        "images": [{"bytes": _png(width + i), "path": None} for i in range(n_images)],
        "texts": [{"user": u, "assistant": a} for u, a in turns],
        "source": "test",
        "relevance_min": rating,
    }


def _write_parquet(path: Path, groups: list[list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, SCHEMA) as writer:
        for rows in groups:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))


def _run(*args: str) -> None:
    subprocess.run([sys.executable, str(SCRIPT), *args], check=True, capture_output=True)


def _run_failing(*args: str) -> str:
    done = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    assert done.returncode != 0
    return done.stderr


def _converter() -> Any:
    spec = importlib.util.spec_from_file_location("convert_finevision_to_arrow", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def subset(tmp_path: Path) -> Path:
    src, dst = tmp_path / "parquet", tmp_path / "arrow"
    _write_parquet(
        src / "demo" / "train-00000-of-00002.parquet",
        [
            [_row(1, [("Q0", "A0")], 5), _row(2, [("Q1", "A1")], 5)],
            [_row(1, [("Q2", ""), ("Q2b", " ")], 5)],
        ],
    )
    _write_parquet(
        src / "demo" / "train-00001-of-00002.parquet",
        [
            [
                _row(1, [("Q3", "A3"), ("Q3b", "A3b"), ("Q3c", "")], None, width=7),
                _row(1, [("Q4", "A4")], 2),
            ]
        ],
    )
    _run("convert", "--src", str(src), "--dst", str(dst), "--procs", "1")
    _run("index", "--src", str(src), "--dst", str(dst), "--procs", "1")
    return dst / "demo"


class TestConversion:
    def test_index_records_layout_and_row_flags(self, subset: Path) -> None:
        index = np.load(subset / "index.npz")
        assert list(index["files"]) == ["train-00000-of-00002.arrow", "train-00001-of-00002.arrow"]
        assert index["file_batches"].tolist() == [2, 1]
        assert index["batch_rows"].tolist() == [2, 1, 2]
        assert index["keep"].tolist() == [True, True, False, True, True]
        assert index["num_images"].tolist() == [1, 2, 1, 1, 1]
        assert index["relevance_min"].tolist() == [5, 5, 5, -1, 2]

    def test_index_fills_image_counts_for_old_sidecars(self, subset: Path) -> None:
        for meta in subset.glob("*.meta.npz"):
            arrays = dict(np.load(meta))
            del arrays["num_images"]
            np.savez(meta, **arrays)
        (subset / "index.npz").unlink()
        root = subset.parents[1]
        _run("index", "--src", str(root / "parquet"), "--dst", str(root / "arrow"))
        assert np.load(subset / "index.npz")["num_images"].tolist() == [1, 2, 1, 1, 1]

    def test_an_empty_row_group_keeps_its_batch(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        _write_parquet(
            src / "demo" / "train-00000-of-00001.parquet",
            [[_row(1, [("Q0", "A0")], 5)], [], [_row(1, [("Q1", "A1")], 5)]],
        )
        assert pq.ParquetFile(src / "demo" / "train-00000-of-00001.parquet").num_row_groups == 3
        _run("convert", "--src", str(src), "--dst", str(dst))
        _run("index", "--src", str(src), "--dst", str(dst))
        index = np.load(dst / "demo" / "index.npz")
        assert index["file_batches"].tolist() == [3]
        assert index["batch_rows"].tolist() == [1, 0, 1]
        reader = FineVisionArrow(dst / "demo")
        assert [reader[i].turns[0].user for i in range(len(reader))] == ["Q0", "Q1"]

    def test_a_stale_claim_is_taken_over_and_a_live_one_respected(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        parquet = src / "demo" / "train-00000-of-00001.parquet"
        _write_parquet(parquet, [[_row(1, [("Q0", "A0")], 5)]])
        (dst / "demo").mkdir(parents=True)
        arrow = dst / "demo" / "train-00000-of-00001.arrow"
        claim = arrow.with_name(arrow.name + ".claim")
        claim.write_text("otherhost-1 0\n")
        converter = _converter()
        assert converter._convert_job((parquet, arrow, 3600.0)) == 0
        assert not arrow.exists() and claim.read_text() == "otherhost-1 0\n"
        os.utime(claim, (0, 0))
        assert converter._convert_job((parquet, arrow, 3600.0)) > 0
        assert arrow.exists() and not claim.exists()
        assert not list((dst / "demo").glob("*.tmp*"))

    def test_keep_flags_need_one_complete_turn(self) -> None:
        texts = pa.array(
            [
                [{"user": "Q", "assistant": "A"}],
                [{"user": "Q", "assistant": ""}, {"user": " ", "assistant": "A"}],
                [{"user": "Q", "assistant": None}, {"user": "Q2", "assistant": "A2"}],
                [],
                None,
            ],
            type=SCHEMA.field("texts").type,
        )
        assert _converter().keep_flags(texts).tolist() == [True, False, True, False, False]
        assert _converter().keep_flags(texts.slice(2, 2)).tolist() == [True, False]

    def test_image_counts_without_an_images_column(self) -> None:
        converter = _converter()
        batch = pa.RecordBatch.from_pydict({"texts": [[], []]})
        assert converter.image_counts(batch).tolist() == [0, 0]
        nulls = pa.RecordBatch.from_pydict({"images": pa.array([None, None], type=pa.null())})
        assert converter.image_counts(nulls).tolist() == [0, 0]

    def test_a_failed_file_is_reported_and_leaves_nothing_behind(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        _write_parquet(
            src / "demo" / "train-00000-of-00002.parquet", [[_row(1, [("Q0", "A0")], 5)]]
        )
        (src / "demo" / "train-00001-of-00002.parquet").write_bytes(b"not parquet")
        stderr = _run_failing("convert", "--src", str(src), "--dst", str(dst))
        assert "train-00001-of-00002.parquet" in stderr and "1 files failed" in stderr
        assert (dst / "demo" / "train-00000-of-00002.meta.npz").exists()
        assert sorted(p.name for p in (dst / "demo").iterdir()) == [
            "train-00000-of-00002.arrow",
            "train-00000-of-00002.meta.npz",
        ]

    def test_the_claim_is_touched_after_every_row_group(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        parquet = src / "demo" / "train-00000-of-00001.parquet"
        _write_parquet(parquet, [[_row(1, [("Q0", "A0")], 5)], [_row(1, [("Q1", "A1")], 5)]])
        (dst / "demo").mkdir(parents=True)
        beats: list[int] = []
        arrow = dst / "demo" / "train-00000-of-00001.arrow"
        _converter().convert_file(parquet, arrow, "t", heartbeat=lambda: beats.append(1))
        assert len(beats) == 2

    def test_index_refuses_a_source_that_changed(self, subset: Path) -> None:
        root = subset.parents[1]
        with open(root / "parquet" / "demo" / "train-00001-of-00002.parquet", "ab") as f:
            f.write(b"\0")
        stderr = _run_failing("index", "--src", str(root / "parquet"), "--dst", str(root / "arrow"))
        assert "not fully converted" in stderr

    def test_only_the_holder_releases_a_claim(self, tmp_path: Path) -> None:
        converter = _converter()
        claim = tmp_path / "x.arrow.claim"
        claim.write_text("otherhost-1 0\n")
        converter._release_claim(claim, "thishost-2")
        assert claim.exists()
        converter._release_claim(claim, "otherhost-1")
        assert not claim.exists()
        converter._release_claim(claim, "otherhost-1")


class TestFineVisionArrow:
    def test_reads_kept_rows_across_files_and_batches(self, subset: Path) -> None:
        reader = FineVisionArrow(subset, max_images=None)
        assert len(reader) == 4
        assert [reader[i].turns[0].user for i in range(4)] == ["Q0", "Q1", "Q3", "Q4"]

    def test_single_image_rows_by_default(self, subset: Path) -> None:
        reader = FineVisionArrow(subset)
        assert [reader[i].turns[0].user for i in range(len(reader))] == ["Q0", "Q3", "Q4"]
        assert all(len(reader[i].images) == 1 for i in range(len(reader)))

    def test_documents_are_conversations(self, subset: Path) -> None:
        doc = FineVisionArrow(subset)[1]
        assert doc.kind == "conversation"
        assert [(t.user, t.answers) for t in doc.turns] == [("Q3", ("A3",)), ("Q3b", ("A3b",))]
        assert doc.images[0].size == (7, 3)

    def test_max_images_keeps_rows_within_the_limit(self, subset: Path) -> None:
        reader = FineVisionArrow(subset, max_images=1)
        assert [reader[i].turns[0].user for i in range(len(reader))] == ["Q0", "Q3", "Q4"]

    def test_num_images_overrides_the_index(self, subset: Path) -> None:
        counts = np.array([2, 1, 1, 1, 1], dtype=np.int16)
        reader = FineVisionArrow(subset, max_images=1, num_images=counts)
        assert [reader[i].turns[0].user for i in range(len(reader))] == ["Q1", "Q3", "Q4"]

    def test_max_images_needs_image_counts(self, subset: Path) -> None:
        arrays = dict(np.load(subset / "index.npz"))
        del arrays["num_images"]
        np.savez(subset / "index.npz", **arrays)
        with pytest.raises(ValueError, match="no num_images"):
            FineVisionArrow(subset, max_images=1)

    def test_ratings_filter_drops_missing_and_low_scores(self, subset: Path) -> None:
        reader = FineVisionArrow(subset, max_images=None, min_ratings={"relevance_min": 3})
        assert [reader[i].turns[0].user for i in range(len(reader))] == ["Q0", "Q1"]

    def test_unknown_rating_is_refused(self, subset: Path) -> None:
        with pytest.raises(ValueError, match="unknown rating"):
            FineVisionArrow(subset, min_ratings={"beauty_min": 3})

    def test_text_only_reads_no_images_and_takes_every_row(self, subset: Path) -> None:
        reader = FineVisionArrow(subset, text_only=True)
        assert len(reader) == 4 and reader[1].images == ()

    def test_survives_pickling(self, subset: Path) -> None:
        reader = FineVisionArrow(subset, max_images=None)
        reader[0]
        clone = pickle.loads(pickle.dumps(reader))
        assert not clone._readers
        assert clone[3].turns[0].user == "Q4"

    def test_open_files_are_capped(self, subset: Path) -> None:
        reader = FineVisionArrow(subset, max_images=None, max_open_files=1)
        reader[0]
        reader[3]
        assert list(reader._readers) == [1]

    @pytest.mark.parametrize("option", ["max_attempts", "max_open_files"])
    def test_limits_must_be_positive(self, subset: Path, option: str) -> None:
        with pytest.raises(ValueError, match="positive"):
            FineVisionArrow(subset, **{option: 0})

    def test_a_row_without_a_complete_turn_is_passed_over(self, subset: Path) -> None:
        """An index that kept such a row (an older keep rule) still reads past it."""
        arrays = dict(np.load(subset / "index.npz"))
        arrays["keep"][:] = True
        np.savez(subset / "index.npz", **arrays)
        reader = FineVisionArrow(subset, max_images=None)
        assert len(reader) == 5
        assert reader[2].turns[0].user == "Q3"
        assert reader.skipped == 1

    def test_an_undecodable_image_steps_to_the_next_row(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        broken = _row(1, [("Bad", "x")], 5)
        broken["images"][0]["bytes"] = b"junk"
        _write_parquet(
            src / "demo" / "train-00000-of-00001.parquet", [[broken, _row(1, [("Good", "y")], 5)]]
        )
        _run("convert", "--src", str(src), "--dst", str(dst))
        _run("index", "--src", str(src), "--dst", str(dst))
        reader = FineVisionArrow(dst / "demo")
        assert reader[0].turns[0].user == "Good"
        assert reader.skipped == 1
