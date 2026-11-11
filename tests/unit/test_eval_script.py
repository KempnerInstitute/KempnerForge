"""Unit tests for the checkpoint that ``scripts/eval.py`` loads."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval.py"


def _eval_script():
    """``scripts/eval.py`` as a module, so ``main`` can run with its heavy steps replaced."""
    spec = importlib.util.spec_from_file_location("kempnerforge_eval_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _saved(step_dir: Path, *, train_state: bool) -> None:
    """A ``step_N`` with model and optimizer shards, with or without ``train_state.pt``."""
    state = {"model": {"w": torch.zeros(2)}, "optimizer": {"state": {"w": {"m": torch.zeros(2)}}}}
    dcp.save(state, checkpoint_id=str(step_dir))
    if train_state:
        torch.save({"step": 0, "tokens_seen": 0}, step_dir / "train_state.pt")


class _Picked(Exception):
    """Stops ``main`` once it has chosen the checkpoint to load."""


class TestEvalScriptCheckpoint:
    def test_eval_loads_the_newest_save_even_without_its_train_state(self, tmp_path, monkeypatch):
        """Eval loads the weights alone, so a save that lacks only ``train_state.pt`` serves."""
        script = _eval_script()
        run = tmp_path / "run"
        _saved(run / "step_10", train_state=True)
        _saved(run / "step_20", train_state=False)
        config_path = tmp_path / "eval.toml"
        config_path.write_text("")
        loads: list[dict] = []

        class RecordingManager:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def load(self, **kwargs):
                loads.append(kwargs)
                raise _Picked

        monkeypatch.setattr(script, "get_world_info", lambda: (0, 0, 1))
        monkeypatch.setattr(script, "init_distributed", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            script, "build_parallel_model", lambda *args, **kwargs: torch.nn.Linear(2, 2)
        )
        monkeypatch.setattr(script, "CheckpointManager", RecordingManager)
        argv = [str(config_path), f"--checkpoint.dir={run}", f"--eval.dataset_path={tmp_path}"]
        monkeypatch.setattr(sys, "argv", ["eval.py", *argv])

        with pytest.raises(_Picked):
            script.main()

        assert loads == [{"path": str(run / "step_20"), "exclude_keys": ["optimizer"]}]
