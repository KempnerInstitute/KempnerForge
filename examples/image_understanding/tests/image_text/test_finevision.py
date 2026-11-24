"""Tests for the FineVision Arrow converter and reader."""

from __future__ import annotations

import errno
import importlib.util
import io
import os
import pickle
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_text.finevision import FineVisionArrow
from image_text.render import ChatRenderer, RenderedDataset
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


def _write_parquet(path: Path, groups: list[list[dict]], schema: pa.Schema = SCHEMA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema) as writer:
        for rows in groups:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def _run(*args: str) -> None:
    done = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def _run_failing(*args: str) -> str:
    done = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    assert done.returncode != 0
    return done.stderr


def _converter() -> Any:
    spec = importlib.util.spec_from_file_location("convert_finevision_to_arrow", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _listing(directory: Path) -> list[str]:
    """The sorted names in ``directory``, without the ``.nfs*`` entries an NFS client can keep
    for a removed file for a moment after the removal."""
    return sorted(p.name for p in directory.iterdir() if not p.name.startswith(".nfs"))


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

    @pytest.mark.parametrize("nfs_held", [False, True])
    def test_a_stale_claim_is_taken_over_and_a_live_one_respected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nfs_held: bool
    ) -> None:
        """With ``nfs_held`` each removed claim stays under a ``.nfs`` name, as an NFS client
        can keep a removed file for a moment; the listing leaves those entries out."""
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
        held: list[Path] = []
        if nfs_held:
            unlink = Path.unlink

            def keep_under_nfs_name(path: Path, missing_ok: bool = False) -> None:
                if path.exists():
                    held.append(path.with_name(f".nfs{len(held) + 1:024x}"))
                    os.link(path, held[-1])
                unlink(path, missing_ok=missing_ok)

            monkeypatch.setattr(Path, "unlink", keep_under_nfs_name)
        assert converter._convert_job((parquet, arrow, 3600.0)) > 0
        tag = f"{socket.gethostname()}-{os.getpid()}"
        assert [p.read_text().split()[0] for p in held] == (
            ["otherhost-1", tag] if nfs_held else []
        )
        assert _listing(dst / "demo") == [
            "train-00000-of-00001.arrow",
            "train-00000-of-00001.meta.npz",
        ]

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
        assert _converter().keep_flags(texts).tolist() == [True, False, False, False, False]
        assert _converter().keep_flags(texts.slice(0, 2)).tolist() == [True, False]
        assert _converter().keep_flags(texts.slice(3, 2)).tolist() == [False, False]

    def test_keep_flags_skip_the_values_under_a_null_slot(self) -> None:
        values = pa.array(
            [{"user": "Q", "assistant": "A"}] * 3, type=SCHEMA.field("texts").type.value_type
        )
        texts = pa.ListArray.from_arrays(
            pa.array([0, 1, 3, 3], type=pa.int32()), values, mask=pa.array([False, True, False])
        )
        assert _converter().keep_flags(texts).tolist() == [True, False, False]

    def test_image_counts_without_an_images_column(self) -> None:
        converter = _converter()
        batch = pa.RecordBatch.from_pydict({"texts": [[], []]})
        assert converter.image_counts(batch).tolist() == [0, 0]
        nulls = pa.RecordBatch.from_pydict({"images": pa.array([None, None], type=pa.null())})
        assert converter.image_counts(nulls).tolist() == [0, 0]

    def test_image_counts_leave_out_entries_without_bytes(self) -> None:
        """A blank, null or path-only entry is no image, in a whole batch or a slice."""
        converter = _converter()
        struct = SCHEMA.field("images").type.value_type
        real, blank = {"bytes": b"x", "path": None}, {"bytes": None, "path": None}
        by_path = {"bytes": None, "path": "a.png"}
        lists = [[real], [blank, real], [blank], [None, real], [], None, [by_path]]
        batch = pa.RecordBatch.from_pydict({"images": pa.array(lists, pa.list_(struct))})
        assert converter.image_counts(batch).tolist() == [1, 1, 0, 1, 0, 0, 0]
        assert converter.image_counts(batch.slice(1, 6)).tolist() == [1, 0, 1, 0, 0, 0]

    def test_image_counts_skip_the_entries_under_a_null_row(self) -> None:
        """A null row whose offsets still span entries, as Arrow allows, holds no image."""
        converter = _converter()
        struct = SCHEMA.field("images").type.value_type
        values = pa.array([{"bytes": b"x", "path": None}] * 4, struct)
        offsets = pa.array([0, 1, 3, 4], pa.int32())
        images = pa.ListArray.from_arrays(offsets, values, mask=pa.array([False, True, False]))
        batch = pa.RecordBatch.from_pydict({"images": images})
        assert converter.image_counts(batch).tolist() == [1, 0, 1]

    def test_index_recounts_images_on_request(self, subset: Path) -> None:
        """Counts a sidecar already holds are kept unless index is asked to recount."""
        for meta in subset.glob("*.meta.npz"):
            arrays = dict(np.load(meta))
            arrays["num_images"] = arrays["num_images"] + 1
            np.savez(meta, **arrays)
        root = subset.parents[1]
        args = ("index", "--src", str(root / "parquet"), "--dst", str(root / "arrow"))
        _run(*args)
        assert np.load(subset / "index.npz")["num_images"].tolist() == [2, 3, 2, 2, 2]
        _run(*args, "--recount-images")
        assert np.load(subset / "index.npz")["num_images"].tolist() == [1, 2, 1, 1, 1]
        _run(*args)
        assert np.load(subset / "index.npz")["num_images"].tolist() == [1, 2, 1, 1, 1]
        assert sorted(p.name for p in subset.glob("*.tmp.npz")) == []

    def test_each_sidecar_rewrite_has_a_temp_file_of_its_own(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two writers of one sidecar never share a temp file, and a failed write keeps the
        sidecar and removes only its own temp file."""
        converter = _converter()
        sidecar = tmp_path / "train.meta.npz"
        np.savez(sidecar, num_images=np.array([1, 2]))
        temps: list[str] = []
        savez = np.savez

        def recording(path, **arrays):
            temps.append(Path(path).name)
            savez(path, **arrays)

        monkeypatch.setattr(converter.np, "savez", recording)
        converter._write_npz(sidecar, {"num_images": np.array([1, 1])})
        converter._write_npz(sidecar, {"num_images": np.array([1, 0])})
        assert len(set(temps)) == 2 and all(t.endswith(".tmp.npz") for t in temps)
        assert np.load(sidecar)["num_images"].tolist() == [1, 0]

        def failing(path, **arrays):
            savez(path, **arrays)
            raise OSError("disk full")

        other = tmp_path / "train.meta.another-writer.tmp.npz"
        other.write_bytes(b"in progress")
        monkeypatch.setattr(converter.np, "savez", failing)
        with pytest.raises(OSError, match="disk full"):
            converter._write_npz(sidecar, {"num_images": np.array([0, 0])})
        assert np.load(sidecar)["num_images"].tolist() == [1, 0]
        assert sorted(p.name for p in tmp_path.glob("train.meta*")) == [other.name, sidecar.name]

    @pytest.mark.parametrize("step", ["savez", "replace"])
    def test_a_failed_index_write_keeps_the_index_and_removes_its_temp_file(
        self, subset: Path, monkeypatch: pytest.MonkeyPatch, step: str
    ) -> None:
        """A failed write of index.npz keeps the old index and the sidecars, and removes only
        the temp file of the run that failed."""
        converter = _converter()
        before = {p.name: p.read_bytes() for p in subset.glob("*.npz")}
        other = subset / "index.another-writer.tmp.npz"
        other.write_bytes(b"in progress")
        calls: list[str] = []
        savez = np.savez

        def failing_savez(path, **arrays):
            calls.append(Path(path).name)
            savez(path, **arrays)
            raise OSError("disk full")

        def failing_replace(source, target):
            calls.append(Path(source).name)
            raise OSError("disk full")

        if step == "savez":
            monkeypatch.setattr(converter.np, "savez", failing_savez)
        else:
            monkeypatch.setattr(converter.os, "replace", failing_replace)
        root = subset.parents[1]
        with pytest.raises(OSError, match="disk full"):
            converter._subset_index((root / "parquet", root / "arrow", subset.name, False))
        assert len(calls) == 1 and calls[0].startswith("index.")
        assert {p.name: p.read_bytes() for p in subset.glob("*.npz") if p != other} == before
        assert sorted(p.name for p in subset.glob("*.tmp.npz")) == [other.name]

    def test_a_failed_file_is_reported_and_leaves_nothing_behind(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        _write_parquet(
            src / "demo" / "train-00000-of-00002.parquet", [[_row(1, [("Q0", "A0")], 5)]]
        )
        (src / "demo" / "train-00001-of-00002.parquet").write_bytes(b"not parquet")
        stderr = _run_failing("convert", "--src", str(src), "--dst", str(dst))
        assert "train-00001-of-00002.parquet" in stderr and "1 files failed" in stderr
        assert (dst / "demo" / "train-00000-of-00002.meta.npz").exists()
        assert _listing(dst / "demo") == [
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
        assert "changed size" in stderr

    def test_a_missing_arrow_file_is_converted_again(self, subset: Path) -> None:
        root = subset.parents[1]
        src, dst = str(root / "parquet"), str(root / "arrow")
        (subset / "train-00001-of-00002.arrow").unlink()
        assert "not converted yet" in _run_failing("index", "--src", src, "--dst", dst)
        _run("convert", "--src", src, "--dst", dst, "--procs", "1")
        assert (subset / "train-00001-of-00002.arrow").exists()
        _run("index", "--src", src, "--dst", dst, "--procs", "1")

    @pytest.mark.parametrize("command", ["convert", "index"])
    def test_a_subset_without_parquet_files_is_reported(self, tmp_path: Path, command: str) -> None:
        (tmp_path / "parquet" / "typo").mkdir(parents=True)
        src, dst = str(tmp_path / "parquet"), str(tmp_path / "arrow")
        stderr = _run_failing(command, "--src", src, "--dst", dst, "--subsets", "typo")
        assert "typo: no train-*.parquet files" in stderr

    def test_a_claim_taken_over_stops_its_holder(self, tmp_path: Path) -> None:
        converter = _converter()
        claim = tmp_path / "x.arrow.claim"
        with pytest.raises(converter.ClaimLostError, match="taken over"):
            converter._touch_claim(claim, "host-1")
        claim.write_text("host-2 0\n")
        with pytest.raises(converter.ClaimLostError, match="another"):
            converter._touch_claim(claim, "host-1")
        converter._touch_claim(claim, "host-2")
        assert claim.stat().st_mtime > 0

    def test_only_the_holder_releases_a_claim(self, tmp_path: Path) -> None:
        converter = _converter()
        claim = tmp_path / "x.arrow.claim"
        claim.write_text("otherhost-1 0\n")
        converter._release_claim(claim, "thishost-2")
        assert claim.exists()
        converter._release_claim(claim, "otherhost-1")
        assert not claim.exists()
        converter._release_claim(claim, "otherhost-1")

    def test_a_release_by_another_process_never_moves_the_claim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The holder's heartbeat finds its claim however another's release interleaves."""
        converter = _converter()
        claim = tmp_path / "x.arrow.claim"
        claim.write_text("holder-1 0\n")
        moves: list[Any] = []
        monkeypatch.setattr(converter.os, "rename", lambda *args: moves.append(args))
        monkeypatch.setattr(Path, "unlink", lambda *args, **kwargs: moves.append(args))
        converter._release_claim(claim, "other-2")
        assert moves == []

    def test_a_lost_claim_is_left_to_its_taker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        parquet = src / "demo" / "train-00000-of-00001.parquet"
        _write_parquet(parquet, [[_row(1, [("Q0", "A0")], 5)]])
        (dst / "demo").mkdir(parents=True)
        arrow = dst / "demo" / "train-00000-of-00001.arrow"
        claim = arrow.with_name(arrow.name + ".claim")
        converter = _converter()

        def taken_over(*args: Any, **kwargs: Any) -> int:
            claim.write_text("taker-9 0\n")
            raise converter.ClaimLostError("taken over")

        monkeypatch.setattr(converter, "convert_file", taken_over)
        assert converter._convert_job((parquet, arrow, 3600.0)) == 0
        assert claim.read_text() == "taker-9 0\n"

    def test_an_unwritable_subset_fails_alone(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        for name in ("demo", "locked"):
            _write_parquet(
                src / name / "train-00000-of-00001.parquet", [[_row(1, [("Q0", "A0")], 5)]]
            )
        (dst / "locked").mkdir(parents=True)
        (dst / "locked").chmod(0o555)
        try:
            stderr = _run_failing("convert", "--src", str(src), "--dst", str(dst), "--procs", "1")
        finally:
            (dst / "locked").chmod(0o755)
        assert "PermissionError" in stderr and "1 files failed" in stderr
        assert _listing(dst / "demo") == [
            "train-00000-of-00001.arrow",
            "train-00000-of-00001.meta.npz",
        ]
        assert _listing(dst / "locked") == []

    def test_a_failed_claim_write_leaves_no_claim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full disk while writing the claim fails the file, and a rerun converts it."""
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        parquet = src / "demo" / "train-00000-of-00001.parquet"
        _write_parquet(parquet, [[_row(1, [("Q0", "A0")], 5)]])
        (dst / "demo").mkdir(parents=True)
        arrow = dst / "demo" / "train-00000-of-00001.arrow"
        converter = _converter()
        write = converter.os.write

        def full_once(fd: int, data: bytes) -> int:
            monkeypatch.setattr(converter.os, "write", write)
            raise OSError(errno.EDQUOT, "Disk quota exceeded")

        monkeypatch.setattr(converter.os, "write", full_once)
        assert "Disk quota exceeded" in converter._convert_job((parquet, arrow, 3600.0))
        assert _listing(dst / "demo") == []
        assert converter._convert_job((parquet, arrow, 3600.0)) > 0

    def test_the_recorded_source_size_is_the_converted_file(self, tmp_path: Path) -> None:
        src = tmp_path / "parquet" / "demo" / "train-00000-of-00001.parquet"
        _write_parquet(src, [[_row(1, [("Q0", "A0")], 5)]])
        size = src.stat().st_size
        replacement = tmp_path / "replacement.parquet"
        _write_parquet(replacement, [[_row(1, [(f"Q{i}", "A")], 5) for i in range(4)]])
        assert replacement.stat().st_size != size
        arrow = tmp_path / "arrow" / "demo" / "train-00000-of-00001.arrow"
        arrow.parent.mkdir(parents=True)

        def replace_source() -> None:
            if replacement.exists():
                os.replace(replacement, src)

        _converter().convert_file(src, arrow, "t", heartbeat=replace_source)
        meta = np.load(arrow.with_name(arrow.stem + ".meta.npz"))
        assert int(meta["source_size"]) == size and meta["batch_rows"].tolist() == [1]


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

    def test_text_only_reads_a_subset_without_an_images_column(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        schema = pa.schema([f for f in SCHEMA if f.name != "images"])
        rows = [_row(0, [("Q0", "A0")], 5), _row(0, [("Q1", "A1")], 5)]
        for row in rows:
            del row["images"]
        _write_parquet(src / "text_demo" / "train-00000-of-00001.parquet", [rows], schema)
        _run("convert", "--src", str(src), "--dst", str(dst))
        _run("index", "--src", str(src), "--dst", str(dst))
        reader = FineVisionArrow(dst / "text_demo", text_only=True)
        assert len(reader) == 2 and reader[1].images == () and reader[1].turns[0].user == "Q1"
        with pytest.raises(ValueError, match="needs text_only"):
            FineVisionArrow(dst / "text_demo")

    def test_text_only_keeps_only_the_rows_without_images(self, subset: Path) -> None:
        with pytest.raises(ValueError, match="no row"):
            FineVisionArrow(subset, text_only=True)

    def test_a_rating_missing_from_the_index_is_refused(self, subset: Path) -> None:
        arrays = dict(np.load(subset / "index.npz"))
        del arrays["formatting_min"]
        np.savez(subset / "index.npz", **arrays)
        with pytest.raises(ValueError, match="has no formatting_min"):
            FineVisionArrow(subset, min_ratings={"formatting_min": 1})

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

    @pytest.mark.parametrize("option", ["max_attempts", "max_open_files", "max_images"])
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

    def test_null_images_and_turns_are_tolerated(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        no_images = _row(0, [("Q0", "A0")], 5)
        no_images["images"] = None
        no_bytes = _row(1, [("Q1", "A1")], 5)
        no_bytes["images"][0]["bytes"] = None
        null_turn = _row(1, [("Q2", "A2")], 5)
        null_turn["texts"] = [*null_turn["texts"], None]
        _write_parquet(
            src / "demo" / "train-00000-of-00001.parquet", [[no_images, no_bytes, null_turn]]
        )
        _run("convert", "--src", str(src), "--dst", str(dst))
        _run("index", "--src", str(src), "--dst", str(dst))
        reader = FineVisionArrow(dst / "demo", max_images=None)
        assert reader.rows.tolist() == [2]
        assert reader[0].turns[0].user == "Q2" and reader.skipped == 0
        text = FineVisionArrow(dst / "demo", text_only=True)
        assert text.rows.tolist() == [0, 1]
        assert text[0].images == () and text[1].images == ()

    def test_a_blank_or_null_entry_is_not_an_image(self, tmp_path: Path) -> None:
        """Each row with one image is read with it, and a row whose only entry is blank is a
        text-only document."""
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        rows = [_row(1, [(f"Q{i}", f"A{i}")], 5) for i in range(5)]
        blank = {"bytes": None, "path": None}
        rows[1]["images"] = [blank, *rows[1]["images"]]
        rows[2]["images"] = [blank]
        rows[3]["images"] = [None, *rows[3]["images"]]
        rows[4]["images"] = []
        _write_parquet(src / "demo" / "train-00000-of-00001.parquet", [rows])
        _run("convert", "--src", str(src), "--dst", str(dst))
        _run("index", "--src", str(src), "--dst", str(dst))
        assert np.load(dst / "demo" / "index.npz")["num_images"].tolist() == [1, 1, 0, 1, 0]
        reader = FineVisionArrow(dst / "demo")
        assert reader.rows.tolist() == [0, 1, 3]
        documents = [reader[i] for i in range(len(reader))]
        assert [d.turns[0].user for d in documents] == ["Q0", "Q1", "Q3"]
        assert [len(d.images) for d in documents] == [1, 1, 1] and reader.skipped == 0
        text = FineVisionArrow(dst / "demo", text_only=True)
        assert text.rows.tolist() == [2, 4]
        assert [text[i].images for i in range(2)] == [(), ()]

    def test_a_row_counted_before_blank_entries_stopped_counting_is_passed_over(
        self, tmp_path: Path
    ) -> None:
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        blank_only, good = _row(1, [("Blank", "x")], 5), _row(1, [("Good", "y")], 5)
        blank_only["images"] = [{"bytes": None, "path": None}]
        _write_parquet(src / "demo" / "train-00000-of-00001.parquet", [[blank_only, good]])
        _run("convert", "--src", str(src), "--dst", str(dst))
        _run("index", "--src", str(src), "--dst", str(dst))
        reader = FineVisionArrow(dst / "demo", num_images=np.array([1, 1]))
        assert reader.rows.tolist() == [0, 1]
        assert reader[0].turns[0].user == "Good" and reader.skipped == 1

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

    def test_rendering_continues_after_the_row_the_reader_reached(
        self, tmp_path: Path, tokenizer: Any
    ) -> None:
        """A refused row after a run of undecodable ones steps on from where the reader got to."""
        src, dst = tmp_path / "parquet", tmp_path / "arrow"
        broken = _row(1, [("Bad", "x")], 5)
        broken["images"][0]["bytes"] = b"junk"
        rows = [broken] * 15 + [_row(1, [("long " * 300, "x")], 5), _row(1, [("Good", "y")], 5)]
        _write_parquet(src / "demo" / "train-00000-of-00001.parquet", [rows])
        _run("convert", "--src", str(src), "--dst", str(dst))
        _run("index", "--src", str(src), "--dst", str(dst))
        renderer = ChatRenderer(tokenizer, max_text_len=256, image_size=8, answer_only=True)
        dataset = RenderedDataset(FineVisionArrow(dst / "demo"), renderer)
        labels = dataset[0]["labels"]
        assert tokenizer.decode(labels[labels != -100]) == "y<|im_end|>"
        assert dataset.skipped == 16
