"""Tests for the example's entry point."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def _load_entry_point() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_image_understanding_example_train", EXAMPLE_ROOT / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entry_point_passes_the_loaded_config_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text("[train]\nmax_steps = 1\nseq_len = 64\n")
    module = _load_entry_point()
    seen: dict[str, Any] = {}
    monkeypatch.setattr(module, "run_training", lambda config: seen.update(config=config))
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path), "--train.max_steps=3"])
    module.main()
    assert seen["config"].train.max_steps == 3
    assert seen["config"].train.seq_len == 64


def test_entry_point_requires_a_config(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_entry_point()
    monkeypatch.setattr(sys, "argv", ["train.py"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1
