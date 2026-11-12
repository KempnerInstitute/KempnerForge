"""Tests for the evaluation entry point."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from evaluation import evaluator

from kempnerforge.config.loader import load_config

EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINT = EXAMPLE_ROOT / "evaluate_checkpoint.py"
SAMPLE_TASKS = EXAMPLE_ROOT / "configs" / "eval_tasks.toml"


def _load_entry_point() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_image_understanding_example_evaluate_checkpoint", ENTRY_POINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "run.toml"
    path.write_text("[train]\nmax_steps = 1\nseq_len = 64\n")
    return path


def _handed_off(monkeypatch: pytest.MonkeyPatch, *argv: str) -> dict[str, Any]:
    """What ``main`` passes to ``evaluate_checkpoint`` for the command line ``argv``."""
    module = _load_entry_point()
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        module,
        "evaluate_checkpoint",
        lambda config, tasks, **options: seen.update(config=config, tasks=tasks, **options),
    )
    monkeypatch.setattr(sys, "argv", ["evaluate_checkpoint.py", *argv])
    module.main()
    return seen


def test_the_loaded_config_and_tasks_are_handed_off(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _handed_off(
        monkeypatch,
        str(config_path),
        "--checkpoint=runs/a",
        "--tasks",
        str(SAMPLE_TASKS),
        "--output",
        "results/a",
    )
    assert seen["config"].train.max_steps == 1
    assert [task.name for task in seen["tasks"]] == [
        "vqav2",
        "textvqa",
        "docvqa",
        "chartqa-human",
        "chartqa-machine",
    ]
    assert {key: seen[key] for key in ("checkpoint", "output", "batch_size", "publish")} == {
        "checkpoint": "runs/a",
        "output": "results/a",
        "batch_size": 32,
        "publish": True,
    }


def test_options_and_config_overrides_reach_the_evaluation(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _handed_off(
        monkeypatch,
        "--train.max_steps=3",
        "--checkpoint",
        "runs/a",
        str(config_path),
        "--no-publish",
        "--tasks",
        str(SAMPLE_TASKS),
        "--batch-size",
        "8",
        "--output",
        "results/a",
        "--train.tf32",
    )
    assert (seen["batch_size"], seen["publish"]) == (8, False)
    assert seen["config"].train.max_steps == 3
    assert seen["config"].train.seq_len == 64
    assert seen["config"].train.tf32 is True


@pytest.mark.parametrize("missing", ["config", "--checkpoint", "--tasks", "--output"])
def test_every_input_is_required(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    arguments = {
        "config": [str(config_path)],
        "--checkpoint": ["--checkpoint", "runs/a"],
        "--tasks": ["--tasks", str(SAMPLE_TASKS)],
        "--output": ["--output", "results/a"],
    }
    del arguments[missing]
    with pytest.raises(SystemExit) as exc:
        _handed_off(monkeypatch, *(arg for group in arguments.values() for arg in group))
    assert exc.value.code == 2


@pytest.mark.parametrize("stray", [["extra"], ["--batch_size", "8"], ["--checkpoin", "runs/b"]])
def test_an_argument_that_is_neither_an_option_nor_an_override_is_refused(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, stray: list[str]
) -> None:
    argv = [str(config_path), "--checkpoint", "runs/a", "--tasks", str(SAMPLE_TASKS)]
    with pytest.raises(SystemExit) as exc:
        _handed_off(monkeypatch, *argv, "--output", "results/a", *stray)
    assert exc.value.code == 2


def test_a_misspelled_option_is_refused_as_an_unknown_config_key(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = [str(config_path), "--checkpoint", "runs/a", "--tasks", str(SAMPLE_TASKS)]
    with pytest.raises(ValueError, match="Unknown config keys in JobConfig: \\['no-publsh'\\]"):
        _handed_off(monkeypatch, *argv, "--output", "results/a", "--no-publsh")


def _stale_results(tmp_path: Path) -> Path:
    output = tmp_path / "results"
    output.mkdir()
    (output / "results.json").write_text("{}")
    return output


@pytest.mark.parametrize(
    ("broken", "error"), [("config override", ValueError), ("tasks file", FileNotFoundError)]
)
def test_a_stale_results_json_does_not_outlive_a_config_or_tasks_that_do_not_load(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, broken: str, error: type
) -> None:
    output = _stale_results(tmp_path)
    tasks = SAMPLE_TASKS if broken == "config override" else tmp_path / "missing.toml"
    argv = [str(config_path), "--checkpoint", "runs/a", "--tasks", str(tasks)]
    overrides = ["--no_such_section.key=1"] if broken == "config override" else []
    with pytest.raises(error):
        _handed_off(monkeypatch, *argv, "--output", str(output), *overrides)
    assert not (output / "results.json").exists()


def test_a_marker_that_cannot_be_removed_is_handed_to_the_evaluation(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "results"
    (output / "results.json").mkdir(parents=True)
    seen = _handed_off(
        monkeypatch,
        str(config_path),
        "--checkpoint",
        "runs/a",
        "--tasks",
        str(SAMPLE_TASKS),
        "--output",
        str(output),
    )
    assert isinstance(seen["cleared"].failure, OSError)


def test_a_marker_that_is_removed_leaves_no_failure(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _stale_results(tmp_path)
    seen = _handed_off(
        monkeypatch,
        str(config_path),
        "--checkpoint",
        "runs/a",
        "--tasks",
        str(SAMPLE_TASKS),
        "--output",
        str(output),
    )
    assert seen["cleared"].failure is None
    assert not (output / "results.json").exists()


def test_the_script_evaluates_a_checkpoint(
    write_config: Callable[..., Path],
    save_checkpoint: Callable[..., Path],
    write_task: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Run as a script, the way it is launched: the example root is the script's directory."""
    config = write_config()
    step_dir = save_checkpoint(load_config(config, cli_args=[]))
    output = tmp_path / "results"
    command = [
        sys.executable,
        str(ENTRY_POINT),
        str(config),
        "--checkpoint",
        str(step_dir.parent),
        "--tasks",
        str(write_task(rows=3)),
        "--output",
        str(output),
        "--batch-size",
        "2",
        "--no-publish",
        "--train.seed=1",
    ]
    finished = subprocess.run(command, capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert finished.returncode == 0, finished.stderr
    results = json.loads((output / "results.json").read_text())
    assert (results["checkpoint"], results["step"]) == (str(step_dir.resolve()), 3)
    assert results["tasks"]["docvqa"]["samples"] == 3
    assert "not published" not in finished.stdout + finished.stderr


def test_the_command_line_clears_the_output_once(
    write_config: Callable[..., Path],
    save_checkpoint: Callable[..., Path],
    write_task: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = write_config()
    step_dir = save_checkpoint(load_config(config, cli_args=[]))
    calls: list[Path] = []
    clear = evaluator.clear_results

    def counted(output: str | Path) -> Any:
        calls.append(Path(output))
        return clear(output)

    monkeypatch.setattr(evaluator, "clear_results", counted)
    module = _load_entry_point()
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_checkpoint.py",
            str(config),
            "--checkpoint",
            str(step_dir),
            "--tasks",
            str(write_task(rows=2)),
            "--output",
            str(output),
            "--no-publish",
        ],
    )
    module.main()
    assert calls == [output]
    assert (output / "results.json").exists()
