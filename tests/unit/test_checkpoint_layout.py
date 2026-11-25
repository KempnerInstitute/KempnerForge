"""How a checkpoint lays out its DCP shards: flat, or one index per pipeline stage."""

from __future__ import annotations

import json
import logging

import pytest

from kempnerforge.checkpoint.layout import (
    SavedLayout,
    recorded_pp_degree,
    saved_layout,
    stage_dir,
    stage_dirs,
)


def _index(directory) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".metadata").write_text("")


def _record(ckpt, **fields) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "metadata.json").write_text(json.dumps({"step": 1, "tokens_seen": 0, **fields}))


@pytest.fixture
def warnings_seen(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="kempnerforge"):
        yield caplog


class TestSavedLayout:
    def test_an_index_at_the_top_is_the_flat_layout(self, tmp_path):
        _index(tmp_path)
        _record(tmp_path)
        assert saved_layout(tmp_path) == SavedLayout(0)
        assert saved_layout(tmp_path).flat

    def test_a_recorded_degree_gives_the_stage_layout(self, tmp_path):
        _record(tmp_path, pp=3)
        layout = saved_layout(tmp_path)
        assert layout == SavedLayout(3, recorded=True)
        assert not layout.flat
        assert stage_dirs(tmp_path, layout) == [tmp_path / f"pp{k}" for k in range(3)]

    def test_the_record_wins_over_an_index_left_at_the_top(self, tmp_path):
        """A save under pipeline parallelism into a step that a flat save wrote before."""
        _index(tmp_path)
        _record(tmp_path, pp=2)
        assert saved_layout(tmp_path) == SavedLayout(2, recorded=True)

    def test_without_a_record_the_stage_directories_give_the_degree(self, tmp_path):
        for name in ("pp10", "pp0", "pp1"):
            _index(tmp_path / name)
        (tmp_path / "pp_old").mkdir()
        (tmp_path / "ppx").mkdir()
        (tmp_path / "pp3").write_text("not a directory")
        _record(tmp_path)
        layout = saved_layout(tmp_path)
        assert layout == SavedLayout(3)
        assert stage_dirs(tmp_path, layout) == [tmp_path / name for name in ("pp0", "pp1", "pp10")]

    def test_a_stage_directory_without_its_index_still_counts(self, tmp_path):
        _index(tmp_path / "pp0")
        (tmp_path / "pp1").mkdir()
        assert saved_layout(tmp_path) == SavedLayout(2)

    @pytest.mark.parametrize("make", ["empty", "missing", "file"])
    def test_a_checkpoint_without_an_index_has_no_layout(self, tmp_path, make):
        ckpt = tmp_path / "step_1"
        if make == "empty":
            ckpt.mkdir()
        elif make == "file":
            ckpt.write_text("not a directory")
        assert saved_layout(ckpt) is None
        assert recorded_pp_degree(ckpt) is None

    @pytest.mark.parametrize("value", ["two", 0, -1, 1.5, True, [2]])
    def test_a_record_that_is_not_a_degree_counts_as_none(self, tmp_path, value, warnings_seen):
        _index(tmp_path)
        _record(tmp_path, pp=value)
        assert recorded_pp_degree(tmp_path) is None
        assert saved_layout(tmp_path) == SavedLayout(0)
        assert "which is not a pipeline degree" in warnings_seen.text

    @pytest.mark.parametrize("text", ["{not json", "[1, 2]"])
    def test_an_unreadable_record_falls_back_to_the_directories(self, tmp_path, text):
        for k in range(2):
            _index(tmp_path / f"pp{k}")
        (tmp_path / "metadata.json").write_text(text)
        assert saved_layout(tmp_path) == SavedLayout(2)

    def test_an_unreadable_record_is_logged(self, tmp_path, warnings_seen):
        (tmp_path / "metadata.json").write_text("{not json")
        assert recorded_pp_degree(tmp_path) is None
        assert "Could not read the pipeline degree" in warnings_seen.text


class TestDescribe:
    def test_each_layout_names_itself(self):
        assert "without pipeline parallelism" in SavedLayout(0).describe()
        assert "pp=4" in SavedLayout(4, recorded=True).describe()
        assert "pp0/ to pp3/" in SavedLayout(4, recorded=True).describe()
        assert "(pp0/)" in SavedLayout(1).describe()
        unrecorded = SavedLayout(2).describe()
        assert "before the degree was recorded" in unrecorded
        assert "2 stage directories (pp0/ to pp1/)" in unrecorded

    def test_stage_dir_names_the_stage(self, tmp_path):
        assert stage_dir(tmp_path, 7) == tmp_path / "pp7"
