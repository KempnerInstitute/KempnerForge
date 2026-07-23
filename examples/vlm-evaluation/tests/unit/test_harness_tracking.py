"""Unit tests for the harness's experiment-tracking helpers.
``load_train_state_extras`` is monkeypatched at its source module — the
harness re-imports it per call."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import vlm_eval_harness

import kempnerforge.checkpoint
from kempnerforge.config.schema import JobConfig

# --------------------------------------------------------------------------- #
# _checkpoint_step
# --------------------------------------------------------------------------- #


def test_checkpoint_step_from_metadata(tmp_path: Path):
    ckpt = tmp_path / "step_500"
    ckpt.mkdir()
    (ckpt / "metadata.json").write_text(json.dumps({"step": 10_000, "tokens_seen": 1}))

    assert vlm_eval_harness._checkpoint_step(ckpt) == 10_000


def test_checkpoint_step_falls_back_to_dir_name(tmp_path: Path):
    ckpt = tmp_path / "step_500"
    ckpt.mkdir()

    assert vlm_eval_harness._checkpoint_step(ckpt) == 500


def test_checkpoint_step_unknown_warns_and_returns_zero(tmp_path: Path, caplog):
    ckpt = tmp_path / "final"
    ckpt.mkdir()

    with caplog.at_level(logging.WARNING, logger="vlm_eval_harness"):
        step = vlm_eval_harness._checkpoint_step(ckpt)

    assert step == 0
    assert any("training step" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# _resolve_run_id
# --------------------------------------------------------------------------- #


def _config(**metrics_overrides) -> JobConfig:
    config = JobConfig()
    for key, value in metrics_overrides.items():
        setattr(config.metrics, key, value)
    return config


def test_explicit_run_id_wins_without_reading_checkpoint(tmp_path: Path, monkeypatch):
    def _boom(_ckpt_dir):
        raise AssertionError("checkpoint must not be consulted when an id is explicit")

    monkeypatch.setattr(kempnerforge.checkpoint, "load_train_state_extras", _boom)
    config = _config(wandb_run_id="explicit-id")

    vlm_eval_harness._resolve_run_id(config, tmp_path / "step_1")

    assert config.metrics.wandb_run_id == "explicit-id"


def test_checkpoint_run_id_adopted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        kempnerforge.checkpoint,
        "load_train_state_extras",
        lambda _ckpt_dir: {"wandb_run_id": "ckpt-run"},
    )
    config = _config()

    vlm_eval_harness._resolve_run_id(config, tmp_path / "step_1")

    assert config.metrics.wandb_run_id == "ckpt-run"
    assert config.metrics.wandb_run_name is None  # attach path never renames the run


def test_no_run_id_warns_and_derives_fresh_run_name(tmp_path: Path, monkeypatch, caplog):
    monkeypatch.setattr(kempnerforge.checkpoint, "load_train_state_extras", lambda _d: {})
    config = _config()
    ckpt_dir = tmp_path / "vlm_run" / "step_1000"

    with caplog.at_level(logging.WARNING, logger="vlm_eval_harness"):
        vlm_eval_harness._resolve_run_id(config, ckpt_dir)

    assert config.metrics.wandb_run_id == ""
    assert config.metrics.wandb_run_name == "vlm_run-step_1000"
    assert any("no saved wandb_run_id" in r.getMessage() for r in caplog.records)


def test_no_run_id_respects_explicit_run_name(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(kempnerforge.checkpoint, "load_train_state_extras", lambda _d: {})
    config = _config(wandb_run_name="my-eval-run")

    vlm_eval_harness._resolve_run_id(config, tmp_path / "run" / "step_1")

    assert config.metrics.wandb_run_name == "my-eval-run"


def test_unreadable_train_state_degrades_to_fresh_run(tmp_path: Path, monkeypatch, caplog):
    def _foreign(_ckpt_dir):
        raise PermissionError("Refusing to load: owned by uid=999")

    monkeypatch.setattr(kempnerforge.checkpoint, "load_train_state_extras", _foreign)
    config = _config()
    ckpt_dir = tmp_path / "run" / "step_2000"

    with caplog.at_level(logging.WARNING, logger="vlm_eval_harness"):
        vlm_eval_harness._resolve_run_id(config, ckpt_dir)

    assert config.metrics.wandb_run_id == ""
    assert config.metrics.wandb_run_name == "run-step_2000"
    messages = [r.getMessage() for r in caplog.records]
    assert any("Could not read" in m for m in messages)
    assert any("no saved wandb_run_id" in m for m in messages)


# --------------------------------------------------------------------------- #
# _track_eval glue
# --------------------------------------------------------------------------- #


def test_track_eval_failure_never_raises(tmp_path: Path, monkeypatch, caplog):
    """Any tracking-side exception is downgraded to a warning."""

    def _boom(_ckpt_dir):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(kempnerforge.checkpoint, "load_train_state_extras", _boom)
    monkeypatch.setattr(vlm_eval_harness, "_resolve_run_id", None)  # force a TypeError inside

    with caplog.at_level(logging.WARNING, logger="vlm_eval_harness"):
        vlm_eval_harness._track_eval(_config(), {"results": {}}, ["t"], tmp_path)

    assert any("Experiment tracking failed" in r.getMessage() for r in caplog.records)
