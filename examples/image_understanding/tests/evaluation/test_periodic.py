"""Tests for evaluating each checkpoint a training run saves."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from evaluation import periodic
from evaluation.periodic import (
    LAUNCH_LOG_FILE,
    LAUNCHED_FILE,
    PeriodicEvaluation,
    PeriodicEvaluationHook,
    step_dir_complete,
)

from kempnerforge.config.job import JobConfig
from kempnerforge.training.hooks import HookRunner

EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_TASKS = EXAMPLE_ROOT / "configs" / "eval_tasks.toml"
COMMAND = (
    f"{sys.executable} evaluate.py {{config}} {{checkpoint}} {{tasks}} {{output}} {{overrides}}"
)


class FakeChild:
    def __init__(self, status: int | None = 0) -> None:
        self.status = status

    def poll(self) -> int | None:
        return self.status


class Launcher:
    """Stands in for ``subprocess.Popen`` and records every launch."""

    def __init__(self, status: int | None = 0, error: OSError | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.status = status
        self.error = error
        self.children: list[FakeChild] = []

    def __call__(self, argv: list[str], **options: Any) -> FakeChild:
        if self.error is not None:
            raise self.error
        self.calls.append((argv, options))
        child = FakeChild(self.status)
        self.children.append(child)
        return child


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_settings(tmp_path: Path, command: str = COMMAND, **options: Any) -> PeriodicEvaluation:
    return PeriodicEvaluation(
        config=options.pop("config", "run.toml"),
        tasks=str(SAMPLE_TASKS),
        command=command,
        output_root=str(tmp_path / "results"),
        **options,
    )


def save_step(run: Path, step: int, *, complete: bool = True) -> Path:
    step_dir = run / f"step_{step}"
    step_dir.mkdir(parents=True)
    names = (".metadata", "train_state.pt", "metadata.json") if complete else ("train_state.pt",)
    for name in names:
        (step_dir / name).write_text("")
    return step_dir


def finish_flush(step_dir: Path) -> None:
    (step_dir / ".metadata").write_text("")
    (step_dir / "metadata.json").write_text("")


class Messages(list):
    def warning(self, message: str) -> None:
        self.append(message)

    info = warning


def text(messages: list[str]) -> str:
    return "\n".join(messages)


@pytest.fixture
def messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    recorded = Messages()
    monkeypatch.setattr(periodic, "logger", recorded)
    return recorded


@pytest.fixture
def run(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path


@pytest.fixture
def rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(periodic.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(periodic.dist, "get_rank", lambda: 0)


def make_hook(
    tmp_path: Path, launcher: Launcher, clock: Clock | None = None, **options: Any
) -> PeriodicEvaluationHook:
    return PeriodicEvaluationHook(
        make_settings(tmp_path, **options.pop("settings", {})),
        clock=clock or Clock(),
        launch=launcher,
        **options,
    )


def test_a_valid_command_is_accepted_and_creates_the_output_root(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert Path(settings.output_root).is_dir()


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("", "the evaluation command is empty"),
        ("   ", "the evaluation command is empty"),
        ("'unterminated {checkpoint} {output}", "cannot be split"),
        (f"{sys.executable} {{checkpoint}} {{output}} {{epoch}}", r"unknown placeholder \{epoch\}"),
        (f"{sys.executable} {{checkpoint}} {{output.x}}", r"unknown placeholder \{output.x\}"),
        (f"{sys.executable} {{checkpoint}} {{output}} {{", "malformed"),
        (f"{sys.executable} {{checkpoint}}", r"missing \['\{output\}'\]"),
        (f"{sys.executable} {{output}} {{step}}", r"missing \['\{checkpoint\}'\]"),
        ("no-such-program-here {checkpoint} {output}", "was not found"),
    ],
)
def test_a_bad_command_is_refused_naming_the_valid_options(
    tmp_path: Path, command: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_settings(tmp_path, command=command)


def test_an_unknown_placeholder_lists_the_valid_ones(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as exc:
        make_settings(tmp_path, command=f"{sys.executable} {{checkpoint}} {{output}} {{epoch}}")
    for name in ("config", "tasks", "checkpoint", "step", "output", "overrides"):
        assert f"{{{name}}}" in str(exc.value)


def test_a_bad_task_file_is_refused_before_anything_runs(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.toml"
    tasks.write_text('[[tasks]]\nname = "x"\n')
    with pytest.raises(ValueError, match="tasks.toml"):
        PeriodicEvaluation(
            config="run.toml", tasks=str(tasks), command=COMMAND, output_root=str(tmp_path / "out")
        )


def test_an_output_root_that_is_a_file_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "file"
    root.write_text("")
    with pytest.raises(ValueError, match="is not a directory"):
        PeriodicEvaluation(
            config="run.toml", tasks=str(SAMPLE_TASKS), command=COMMAND, output_root=str(root)
        )


def test_the_command_is_split_before_the_placeholders_are_filled(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        command=f"{sys.executable} --name=eval-{{step}} {{config}} {{checkpoint}} {{output}} "
        "{overrides} {tasks}",
        config="my run.toml",
        overrides=("--train.max_steps=3", "--train.tf32"),
    )
    step_dir = tmp_path / "a dir" / "step_7"
    argv = settings.argv(step_dir, 7)
    assert argv[1:] == [
        "--name=eval-7",
        "my run.toml",
        str(step_dir),
        str(settings.step_output(7)),
        "--train.max_steps=3",
        "--train.tf32",
        str(SAMPLE_TASKS),
    ]
    assert settings.step_output(7) == (tmp_path / "results" / "step_7").resolve()


def test_an_empty_overrides_placeholder_adds_no_argument(tmp_path: Path) -> None:
    argv = make_settings(tmp_path).argv(tmp_path / "step_1", 1)
    assert argv[-1] == str(make_settings(tmp_path).step_output(1))


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        ((".metadata", "train_state.pt", "metadata.json"), True),
        ((".metadata", "train_state.pt"), False),
        (("train_state.pt", "metadata.json"), False),
        ((".metadata", "metadata.json"), False),
        ((), False),
    ],
)
def test_a_step_directory_is_complete_only_with_every_file(
    tmp_path: Path, present: tuple[str, ...], expected: bool
) -> None:
    step_dir = tmp_path / "step_1"
    step_dir.mkdir()
    for name in present:
        (step_dir / name).write_text("")
    assert step_dir_complete(step_dir) is expected
    assert step_dir_complete(tmp_path / "missing") is False


def test_a_complete_save_is_launched_at_once_detached(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    launcher = Launcher()
    hook = make_hook(tmp_path, launcher)
    step_dir = save_step(run, 5)
    hook.on_checkpoint_save(5, str(run))
    assert len(launcher.calls) == 1
    argv, options = launcher.calls[0]
    assert argv[2:] == ["run.toml", str(step_dir.resolve()), str(SAMPLE_TASKS)] + [
        str(hook.settings.step_output(5))
    ]
    assert options["start_new_session"] is True
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stderr"] == subprocess.STDOUT
    record = json.loads((hook.settings.step_output(5) / LAUNCHED_FILE).read_text())
    assert record == {"argv": argv}
    assert (hook.settings.step_output(5) / LAUNCH_LOG_FILE).exists()


def test_a_step_is_launched_once_however_often_it_is_reported(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    launcher = Launcher()
    hook = make_hook(tmp_path, launcher)
    save_step(run, 5)
    hook.on_checkpoint_save(5, str(run))
    hook.on_checkpoint_save(5, str(run))
    hook.on_train_end(5, 0)
    assert len(launcher.calls) == 1


def test_a_save_that_is_still_flushing_waits_for_a_later_check(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    launcher = Launcher()
    clock = Clock()
    hook = make_hook(tmp_path, launcher, clock, poll_interval=10.0)
    step_dir = save_step(run, 5, complete=False)
    hook.on_checkpoint_save(5, str(run))
    assert launcher.calls == []
    finish_flush(step_dir)
    clock.now = 4.0
    hook.on_step_end(None)
    assert launcher.calls == []
    clock.now = 11.0
    hook.on_step_end(None)
    assert len(launcher.calls) == 1
    hook.on_step_end(None)
    assert len(launcher.calls) == 1


def test_a_flush_that_ends_with_training_is_launched_at_the_end(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    launcher = Launcher()
    hook = make_hook(tmp_path, launcher)
    step_dir = save_step(run, 5, complete=False)
    hook.on_checkpoint_save(5, str(run))
    finish_flush(step_dir)
    hook.on_train_end(5, 0)
    assert len(launcher.calls) == 1


def test_a_step_that_never_completed_is_skipped_with_a_log_line(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str]
) -> None:
    launcher = Launcher()
    hook = make_hook(tmp_path, launcher)
    save_step(run, 5, complete=False)
    hook.on_checkpoint_save(5, str(run))
    hook.on_train_end(5, 0)
    assert launcher.calls == []
    out = text(messages)
    assert "skips step 5" in out and "never became complete" in out
    messages.clear()
    hook.on_train_end(5, 0)
    assert messages == []


def test_a_pruned_step_is_skipped_with_a_log_line(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str]
) -> None:
    launcher = Launcher()
    clock = Clock()
    hook = make_hook(tmp_path, launcher, clock)
    step_dir = save_step(run, 5, complete=False)
    hook.on_checkpoint_save(5, str(run))
    for child in step_dir.iterdir():
        child.unlink()
    step_dir.rmdir()
    clock.now = 100.0
    hook.on_step_end(None)
    assert launcher.calls == []
    out = text(messages)
    assert "skips step 5" in out and "pruned" in out
    assert not hook._pending


def test_the_other_ranks_launch_nothing(
    tmp_path: Path, run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(periodic.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(periodic.dist, "get_rank", lambda: 1)
    launcher = Launcher()
    hook = make_hook(tmp_path, launcher)
    save_step(run, 5)
    hook.on_train_begin(JobConfig())
    hook.on_checkpoint_save(5, str(run))
    hook.on_step_end(None)
    hook.on_train_end(5, 0)
    assert launcher.calls == []
    assert not hook._pending


def test_a_resumed_run_launches_only_what_no_run_launched(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    first = Launcher()
    save_step(run, 5)
    hook = make_hook(tmp_path, first)
    hook.on_checkpoint_save(5, str(run))
    save_step(run, 10)
    resumed = Launcher()
    again = make_hook(tmp_path, resumed)
    config = JobConfig()
    config.checkpoint.dir = str(run)
    again.on_train_begin(config)
    again.on_checkpoint_save(10, str(run))
    again.on_checkpoint_save(5, str(run))
    assert len(first.calls) == 1
    assert [argv[3] for argv, _ in resumed.calls] == [str((run / "step_10").resolve())]


def test_a_step_with_results_is_not_launched_again(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    launcher = Launcher()
    hook = make_hook(tmp_path, launcher)
    save_step(run, 5)
    output = hook.settings.step_output(5)
    output.mkdir(parents=True)
    (output / "results.json").write_text("{}")
    hook.on_checkpoint_save(5, str(run))
    assert launcher.calls == []


def test_a_claimed_step_is_not_launched_by_another_run(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    launcher = Launcher()
    hook = make_hook(tmp_path, launcher)
    save_step(run, 5)
    claimed = hook.settings.step_output(5) / LAUNCHED_FILE
    claimed.parent.mkdir(parents=True)
    claimed.write_text("{}")
    hook.on_checkpoint_save(5, str(run))
    assert launcher.calls == []
    assert claimed.exists()


def test_the_scan_at_training_begin_takes_up_only_complete_steps(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    launcher = Launcher()
    hook = make_hook(tmp_path, launcher)
    save_step(run, 5)
    save_step(run, 10, complete=False)
    (run / "step_x").mkdir()
    (run / "step_3").write_text("")
    (run / "latest").symlink_to("step_5")
    config = JobConfig()
    config.checkpoint.dir = str(run)
    hook.on_train_begin(config)
    assert [argv[3] for argv, _ in launcher.calls] == [str((run / "step_5").resolve())]


def test_the_scan_of_a_missing_checkpoint_directory_does_nothing(
    tmp_path: Path, rank_zero: None
) -> None:
    launcher = Launcher()
    hook = make_hook(tmp_path, launcher)
    config = JobConfig()
    config.checkpoint.dir = str(tmp_path / "none")
    hook.on_train_begin(config)
    assert launcher.calls == []


def test_a_failed_launch_is_a_warning_and_can_be_retried_by_a_later_run(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str]
) -> None:
    launcher = Launcher(error=FileNotFoundError("no sbatch"))
    hook = make_hook(tmp_path, launcher)
    save_step(run, 5)
    hook.on_checkpoint_save(5, str(run))
    assert "could not launch step 5" in text(messages)
    assert not (hook.settings.step_output(5) / LAUNCHED_FILE).exists()
    hook.on_train_end(5, 0)
    assert launcher.calls == []


def test_a_command_that_exits_non_zero_is_a_warning_and_is_released(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str]
) -> None:
    launcher = Launcher(status=3)
    hook = make_hook(tmp_path, launcher)
    save_step(run, 5)
    hook.on_checkpoint_save(5, str(run))
    assert (hook.settings.step_output(5) / LAUNCHED_FILE).exists()
    hook.on_train_end(5, 0)
    assert "exited with status 3" in text(messages)
    assert not (hook.settings.step_output(5) / LAUNCHED_FILE).exists()


def test_a_running_command_is_never_waited_for(tmp_path: Path, run: Path, rank_zero: None) -> None:
    launcher = Launcher(status=None)
    clock = Clock()
    hook = make_hook(tmp_path, launcher, clock)
    save_step(run, 5)
    hook.on_checkpoint_save(5, str(run))
    hook.on_step_end(None)
    hook.on_train_end(5, 0)
    assert len(hook._children) == 1
    assert (hook.settings.step_output(5) / LAUNCHED_FILE).exists()


def test_an_error_in_the_hook_never_reaches_training(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str]
) -> None:
    class Broken(Launcher):
        def __call__(self, argv: list[str], **options: Any) -> FakeChild:
            raise RuntimeError("boom")

    hook = make_hook(tmp_path, Broken())
    save_step(run, 5)
    hook.on_checkpoint_save(5, str(run))
    assert "RuntimeError: boom" in text(messages)


def test_each_save_of_a_run_is_evaluated_once(tmp_path: Path, run: Path, rank_zero: None) -> None:
    launcher = Launcher()
    clock = Clock()
    hook = make_hook(tmp_path, launcher, clock, poll_interval=1.0)
    runner = hook.runner()
    assert isinstance(runner, HookRunner)
    runner.on_train_begin(JobConfig())
    for step in (2, 4, 6):
        save_step(run, step)
        runner.on_checkpoint_save(step, str(run))
        clock.now += 5.0
        runner.on_step_end(None)
    runner.on_train_end(6, 0)
    assert [Path(argv[3]).name for argv, _ in launcher.calls] == ["step_2", "step_4", "step_6"]


def test_the_default_launch_starts_the_command_and_logs_its_output(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    script = tmp_path / "evaluate.py"
    script.write_text("import sys\nprint('evaluating', sys.argv[1])\n")
    children: list[subprocess.Popen] = []

    def launch(argv: list[str], **options: Any) -> subprocess.Popen:
        child = subprocess.Popen(argv, **options)
        children.append(child)
        return child

    settings = make_settings(
        tmp_path, command=f"{sys.executable} {script} {{checkpoint}} {{output}}"
    )
    hook = PeriodicEvaluationHook(settings, launch=launch)
    save_step(run, 5)
    hook.on_checkpoint_save(5, str(run))
    assert children[0].wait(timeout=60) == 0
    log = (settings.step_output(5) / LAUNCH_LOG_FILE).read_text()
    assert log.strip() == f"evaluating {(run / 'step_5').resolve()}"


def test_a_claim_is_won_by_one_process_only(tmp_path: Path) -> None:
    output = tmp_path / "step_1"
    assert periodic._claim(output, ["a", "b"]) is True
    assert periodic._claim(output, ["c"]) is False
    assert json.loads((output / LAUNCHED_FILE).read_text()) == {"argv": ["a", "b"]}


def test_a_step_claimed_between_the_check_and_the_launch_is_not_launched(
    tmp_path: Path, run: Path, rank_zero: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = Launcher()
    hook = make_hook(tmp_path, launcher)
    save_step(run, 5)
    monkeypatch.setattr(periodic, "_claim", lambda output, argv: False)
    hook.on_checkpoint_save(5, str(run))
    assert launcher.calls == []
