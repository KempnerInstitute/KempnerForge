"""Tests for the example's entry point."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))


def _load_entry_point() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_image_understanding_example_train", EXAMPLE_ROOT / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(seen: dict[str, Any]) -> Any:
    def run_training(config: Any, *, data_fn: Any = None) -> None:
        seen.update(config=config, data_fn=data_fn)

    return run_training


def test_entry_point_passes_the_loaded_config_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text("[train]\nmax_steps = 1\nseq_len = 64\n")
    module = _load_entry_point()
    seen: dict[str, Any] = {}
    monkeypatch.setattr(module, "run_training", _record(seen))
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path), "--train.max_steps=3"])
    module.main()
    assert seen["config"].train.max_steps == 3
    assert seen["config"].train.seq_len == 64
    assert seen["data_fn"] is None


def test_a_mixture_supplies_the_data_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text("[train]\nmax_steps = 1\nseq_len = 64\n")
    module = _load_entry_point()
    seen: dict[str, Any] = {}
    monkeypatch.setattr(module, "run_training", _record(seen))
    monkeypatch.setattr(module, "mixture_data_fn", lambda path: ("data_fn", path))
    monkeypatch.setattr(module, "check_run", lambda config: seen.update(checked=config))
    argv = ["train.py", str(config_path), "--mixture=mix.toml", "--train.max_steps=2"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert seen["data_fn"] == ("data_fn", "mix.toml")
    assert seen["config"].train.max_steps == 2
    assert seen["checked"] is seen["config"]


@pytest.mark.parametrize(
    ("run", "mixture", "message"),
    [
        (
            "[vlm]\narch = 'joint_decoder'\n[data]\npack_sequences = true\n"
            "[train]\nbatch_size = 2\n",
            "[[source]]\nreader = 'vqav2'\nannotation_file = 'a'\nimage_root = 'b'\n",
            "batch_size = 1",
        ),
        (
            "[vlm]\narch = 'joint_decoder'\n[data]\npack_sequences = true\n"
            "[train]\nbatch_size = 1\n",
            "[packng]\nview_groups = true\n[[source]]\nreader = 'vqav2'\n",
            "unknown mixture table",
        ),
    ],
)
def test_a_mixture_run_is_checked_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run: str, mixture: str, message: str
) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text("[vision_encoder]\ntype = 'random'\n" + run)
    mixture_path = tmp_path / "mix.toml"
    mixture_path.write_text(mixture)
    module = _load_entry_point()
    monkeypatch.setattr(module, "run_training", lambda *a, **k: pytest.fail("training started"))
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path), f"--mixture={mixture_path}"])
    with pytest.raises(ValueError, match=message):
        module.main()


@pytest.mark.parametrize(
    "argv", [["train.py"], ["train.py", "run.toml", "--mixture=a", "--mixture=b"]]
)
def test_entry_point_requires_a_config_and_at_most_one_mixture(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    module = _load_entry_point()
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1
