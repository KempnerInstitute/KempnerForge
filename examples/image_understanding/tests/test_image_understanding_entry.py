"""Tests for the example's entry point."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

VLM_CONFIG = """\
[train]
max_steps = 1
seq_len = 64

[vision_encoder]
type = "random"
feature_dim = 24
num_tokens = 4

[vlm]
arch = "joint_decoder"
"""


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


@pytest.fixture
def evaluation_flags(tmp_path: Path) -> list[str]:
    command = f"{sys.executable} evaluate.py {{config}} {{checkpoint}} {{output}} {{overrides}}"
    return [
        "--eval-tasks",
        str(EXAMPLE_ROOT / "configs" / "eval_tasks.toml"),
        "--eval-command",
        command,
        "--eval-output",
        str(tmp_path / "results"),
    ]


def _run_main(module: Any, monkeypatch: pytest.MonkeyPatch, *argv: str) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        module, "run_training", lambda config, **options: seen.update(config=config, **options)
    )
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    module.main()
    return seen


def test_training_runs_without_hooks_when_periodic_evaluation_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(VLM_CONFIG)
    seen = _run_main(_load_entry_point(), monkeypatch, str(config_path), "--train.tf32")
    assert set(seen) == {"config"}
    assert seen["config"].train.tf32 is True


def test_the_flags_turn_periodic_evaluation_on_and_leave_the_overrides_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evaluation_flags: list[str]
) -> None:
    from evaluation.periodic import PeriodicEvaluationHook

    config_path = tmp_path / "run.toml"
    config_path.write_text(VLM_CONFIG)
    seen = _run_main(
        _load_entry_point(),
        monkeypatch,
        str(config_path),
        "--train.max_steps=3",
        *evaluation_flags,
        "--train.tf32",
    )
    assert seen["config"].train.max_steps == 3
    assert seen["config"].train.tf32 is True
    (hook,) = seen["hooks"].hooks
    assert isinstance(hook, PeriodicEvaluationHook)
    assert hook.settings.config == str(config_path)
    assert hook.settings.overrides == ("--train.max_steps=3", "--train.tf32")
    assert hook.settings.output_root == str(tmp_path / "results")


@pytest.mark.parametrize("missing", ["--eval-tasks", "--eval-command", "--eval-output"])
def test_every_evaluation_flag_is_needed_with_the_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_flags: list[str],
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(VLM_CONFIG)
    index = evaluation_flags.index(missing)
    del evaluation_flags[index : index + 2]
    with pytest.raises(SystemExit) as exc:
        _run_main(_load_entry_point(), monkeypatch, str(config_path), *evaluation_flags)
    assert exc.value.code == 2
    assert f"also needs {missing}" in capsys.readouterr().err


def test_a_bad_command_stops_the_run_before_training_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_flags: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(VLM_CONFIG)
    evaluation_flags[3] = f"{sys.executable} {{checkpoint}} {{output}} {{epoch}}"
    module = _load_entry_point()
    ran: list[bool] = []
    monkeypatch.setattr(module, "run_training", lambda *args, **options: ran.append(True))
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path), *evaluation_flags])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "unknown placeholder {epoch}" in capsys.readouterr().err
    assert ran == []


def test_a_config_the_evaluator_cannot_decode_stops_the_run_before_training_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_flags: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text("[train]\nmax_steps = 1\n")
    with pytest.raises(SystemExit) as exc:
        _run_main(_load_entry_point(), monkeypatch, str(config_path), *evaluation_flags)
    assert exc.value.code == 2
    assert "no [vlm] section" in capsys.readouterr().err
