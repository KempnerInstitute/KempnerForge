"""Tests for the FineVision Arrow converter and reader."""

from __future__ import annotations

import io
import pickle
import subprocess
import sys
from pathlib import Path

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


@pytest.fixture
def subset(tmp_path: Path) -> Path:
    src, dst = tmp_path / "parquet", tmp_path / "arrow"
    _write_parquet(
        src / "demo" / "train-00000-of-00002.parquet",
        [
            [_row(1, [("Q0", "A0")], 5), _row(2, [("Q1", "A1")], 5)],
            [_row(1, [("Q2", "A2"), ("Q2b", "")], 5)],
        ],
    )
    _write_parquet(
        src / "demo" / "train-00001-of-00002.parquet",
        [[_row(1, [("Q3", "A3"), ("Q3b", "A3b")], None, width=7), _row(1, [("Q4", "A4")], 2)]],
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


class TestFineVisionArrow:
    def test_reads_kept_rows_across_files_and_batches(self, subset: Path) -> None:
        reader = FineVisionArrow(subset)
        assert len(reader) == 4
        assert [reader[i].turns[0].user for i in range(4)] == ["Q0", "Q1", "Q3", "Q4"]

    def test_documents_are_conversations(self, subset: Path) -> None:
        doc = FineVisionArrow(subset)[2]
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
        reader = FineVisionArrow(subset, min_ratings={"relevance_min": 3})
        assert [reader[i].turns[0].user for i in range(len(reader))] == ["Q0", "Q1"]

    def test_unknown_rating_is_refused(self, subset: Path) -> None:
        with pytest.raises(ValueError, match="unknown rating"):
            FineVisionArrow(subset, min_ratings={"beauty_min": 3})

    def test_text_only_reads_no_images(self, subset: Path) -> None:
        assert FineVisionArrow(subset, text_only=True)[0].images == ()

    def test_survives_pickling(self, subset: Path) -> None:
        reader = FineVisionArrow(subset)
        reader[0]
        clone = pickle.loads(pickle.dumps(reader))
        assert not clone._readers
        assert clone[3].turns[0].user == "Q4"

    def test_open_files_are_capped(self, subset: Path) -> None:
        reader = FineVisionArrow(subset, max_open_files=1)
        reader[0]
        reader[3]
        assert list(reader._readers) == [1]

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
