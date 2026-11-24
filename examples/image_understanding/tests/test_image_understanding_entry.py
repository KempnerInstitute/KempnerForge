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


def test_the_mixture_reaches_training_and_each_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evaluation_flags: list[str]
) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(VLM_CONFIG)
    module = _load_entry_point()
    monkeypatch.setattr(module, "mixture_data_fn", lambda path: ("data_fn", path))
    monkeypatch.setattr(module, "check_run", lambda config: None)
    monkeypatch.chdir(tmp_path)
    seen = _run_main(
        module,
        monkeypatch,
        str(config_path),
        "--mixture=mix.toml",
        "--train.tf32",
        *evaluation_flags,
    )
    assert seen["data_fn"] == ("data_fn", "mix.toml")
    (hook,) = seen["hooks"].hooks
    assert hook.settings.overrides == (f"--mixture={tmp_path / 'mix.toml'}", "--train.tf32")


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


@pytest.mark.parametrize(
    ("arguments", "stray"),
    [
        (["--train.max_steps", "3"], "3"),
        (["extra.toml"], "extra.toml"),
        (["-x", "--train.tf32"], "-x"),
    ],
)
def test_an_argument_that_is_not_an_override_stops_a_run_with_periodic_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_flags: list[str],
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    stray: str,
) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(VLM_CONFIG)
    module = _load_entry_point()
    ran: list[bool] = []
    monkeypatch.setattr(module, "run_training", lambda *args, **options: ran.append(True))
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path), *arguments, *evaluation_flags])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert f"unrecognized arguments: {stray};" in error
    assert "--section.key=value" in error
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


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--train.max_steps=3"],
        ["--train.tf32"],
        ["-h"],
        ["--help"],
        ["--", "--train.max_steps=3"],
        ["--train.max_steps", "3"],
        ["--eval.interval=5"],
        ["-x", "--eval", "y"],
        ["--train.lr=-1"],
    ],
)
def test_without_an_evaluation_flag_every_argument_list_reaches_the_loader_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    module = _load_entry_point()
    received: list[Any] = []
    sentinel = object()

    def load(path: str, cli_args: list[str]) -> object:
        received.append((path, cli_args))
        return sentinel

    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(module, "load_config", load)
    monkeypatch.setattr(module, "run_training", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(sys, "argv", ["train.py", "run.toml", *arguments])
    module.main()
    assert received == [("run.toml", arguments)]
    assert calls == [((sentinel,), {})]
