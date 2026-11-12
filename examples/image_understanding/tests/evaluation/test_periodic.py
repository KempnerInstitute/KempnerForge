"""Tests for evaluating each checkpoint a training run saves."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import evaluation.periodic as periodic
import pytest
from evaluation.periodic import (
    LAUNCH_LOG_FILE,
    LAUNCHED_FILE,
    EvaluationLauncher,
    PeriodicEvaluation,
    PeriodicEvaluationHook,
    child_environment,
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
        self.timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.status

    def wait(self, timeout: float | None = None) -> int:
        self.timeouts.append(timeout)
        if self.status is None:
            raise subprocess.TimeoutExpired("evaluate", timeout or 0.0)
        return self.status


class ExitsWhenWaitedFor(FakeChild):
    """A command that is still running until it is waited for, then exits with status 0."""

    def __init__(self) -> None:
        super().__init__(None)

    def wait(self, timeout: float | None = None) -> int:
        self.timeouts.append(timeout)
        self.status = 0
        return 0


class Launcher:
    """Stands in for ``subprocess.Popen`` and records every launch."""

    def __init__(
        self,
        status: int | None = 0,
        error: Exception | None = None,
        child: Callable[[list[str]], FakeChild] | None = None,
    ) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.attempts = 0
        self.status = status
        self.error = error
        self.child = child
        self.children: list[FakeChild] = []

    def __call__(self, argv: list[str], **options: Any) -> Any:
        self.attempts += 1
        if self.error is not None:
            raise self.error
        self.calls.append((argv, options))
        child = FakeChild(self.status) if self.child is None else self.child(argv)
        self.children.append(child)
        return child


def launched_steps(popen: Launcher) -> list[str]:
    return [Path(argv[3]).name for argv, _ in popen.calls]


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


def write_record(output: Path, name: str, checkpoint: str | Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    record = output / name
    record.write_text(json.dumps({"checkpoint": str(checkpoint)}))
    return record


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


def make_launcher(tmp_path: Path, popen: Launcher, **settings: Any) -> EvaluationLauncher:
    return EvaluationLauncher(make_settings(tmp_path, **settings), launch=popen)


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
        (f"{sys.executable} {{checkpoint}} {{output}} {{step:04d}}", "takes no format"),
        (f"{sys.executable} {{checkpoint!r}} {{output}}", "takes no format"),
        (f"{sys.executable} {{checkpoint}} {{output:{{x}}}}", "takes no format"),
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
        str(Path("my run.toml").resolve()),
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


def test_relative_config_and_task_paths_reach_the_command_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    (work / "configs").mkdir(parents=True)
    (work / "configs" / "tasks.toml").write_text(SAMPLE_TASKS.read_text())
    monkeypatch.chdir(work)
    settings = PeriodicEvaluation(
        config="configs/run.toml",
        tasks="configs/tasks.toml",
        command=COMMAND,
        output_root=str(tmp_path / "results"),
    )
    argv = settings.argv(tmp_path / "run" / "step_1", 1)
    config = str((work / "configs" / "run.toml").resolve())
    tasks = str((work / "configs" / "tasks.toml").resolve())
    assert (settings.config, settings.tasks) == (config, tasks)
    assert (argv[2], argv[4]) == (config, tasks)
    assert all(Path(argument).is_absolute() for argument in argv[2:])


def test_a_complete_save_is_launched_detached_without_the_distributed_environment(
    tmp_path: Path, run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "WORLD_SIZE": "4",
        "RANK": "1",
        "TORCHELASTIC_RUN_ID": "x",
        "KEEP": "1",
    }.items():
        monkeypatch.setenv(name, value)
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_dir = save_step(run, 5)
    launcher.saved(5, run)
    assert len(popen.calls) == 1
    argv, options = popen.calls[0]
    assert argv[2:] == [
        str(Path("run.toml").resolve()),
        str(step_dir.resolve()),
        str(SAMPLE_TASKS),
        str(launcher.settings.step_output(5)),
    ]
    assert options["start_new_session"] is True
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stderr"] == subprocess.STDOUT
    assert options["env"]["KEEP"] == "1"
    assert not {"WORLD_SIZE", "SLURM_NTASKS", "TORCHELASTIC_RUN_ID"} & set(options["env"])
    assert not (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()
    assert (launcher.settings.step_output(5) / LAUNCH_LOG_FILE).exists()


def test_the_child_environment_drops_every_rendezvous_variable() -> None:
    environment = {
        "MASTER_ADDR": "a",
        "MASTER_PORT": "1",
        "LOCAL_WORLD_SIZE": "4",
        "GROUP_RANK": "0",
        "TORCHELASTIC_RESTART_COUNT": "0",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "PATH": "/bin",
    }
    assert child_environment(environment) == {"CUDA_VISIBLE_DEVICES": "0,1", "PATH": "/bin"}


def test_a_step_is_launched_once_however_often_it_is_reported(tmp_path: Path, run: Path) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    launcher.saved(5, run)
    launcher.saved(5, run)
    launcher.finish(run)
    assert len(popen.calls) == 1


def test_a_save_that_is_still_flushing_waits_for_a_later_check(tmp_path: Path, run: Path) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_dir = save_step(run, 5, complete=False)
    launcher.saved(5, run)
    assert popen.calls == [] and launcher.waiting
    launcher.check()
    assert popen.calls == []
    finish_flush(step_dir)
    launcher.check()
    assert len(popen.calls) == 1
    launcher.check()
    assert len(popen.calls) == 1


def test_a_flush_that_ends_with_training_is_launched_at_the_end(tmp_path: Path, run: Path) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_dir = save_step(run, 5, complete=False)
    launcher.saved(5, run)
    finish_flush(step_dir)
    launcher.finish(None)
    assert len(popen.calls) == 1


def test_a_save_without_a_hook_call_is_found_at_the_end(tmp_path: Path, run: Path) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 7)
    launcher.finish(run)
    assert [Path(argv[3]).name for argv, _ in popen.calls] == ["step_7"]


def test_a_step_that_never_completed_is_skipped_with_a_log_line(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5, complete=False)
    launcher.saved(5, run)
    launcher.finish(None)
    assert popen.calls == []
    out = text(messages)
    assert "skips step 5" in out and "never became complete" in out
    messages.clear()
    launcher.finish(None)
    assert messages == []


def test_a_pruned_step_is_skipped_with_a_log_line(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_dir = save_step(run, 5, complete=False)
    launcher.saved(5, run)
    for child in step_dir.iterdir():
        child.unlink()
    step_dir.rmdir()
    launcher.check()
    assert popen.calls == []
    assert "skips step 5" in text(messages) and "pruned" in text(messages)
    assert not launcher.waiting


def test_a_resumed_run_launches_only_what_no_run_launched(tmp_path: Path, run: Path) -> None:
    first = Launcher()
    save_step(run, 5)
    earlier = make_launcher(tmp_path, first)
    earlier.saved(5, run)
    earlier.check()
    save_step(run, 10)
    resumed = Launcher()
    again = make_launcher(tmp_path, resumed)
    again.scan(run)
    again.saved(10, run)
    again.saved(5, run)
    assert len(first.calls) == 1
    assert [argv[3] for argv, _ in resumed.calls] == [str((run / "step_10").resolve())]


@pytest.mark.parametrize("name", ["results.json", LAUNCHED_FILE])
def test_a_step_with_a_record_of_its_checkpoint_is_not_launched_again(
    tmp_path: Path, run: Path, name: str
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_dir = save_step(run, 5)
    record = write_record(launcher.settings.step_output(5), name, step_dir.resolve())
    launcher.saved(5, run)
    launcher.finish(run)
    assert popen.calls == []
    assert record.exists()


@pytest.mark.parametrize("name", ["results.json", LAUNCHED_FILE])
def test_a_record_of_another_checkpoint_does_not_count(
    tmp_path: Path, run: Path, messages: list[str], name: str
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_dir = save_step(run, 5)
    other = save_step(tmp_path / "earlier run", 5)
    output = launcher.settings.step_output(5)
    record = write_record(output, name, other.resolve())
    launcher.saved(5, run)
    assert launched_steps(popen) == ["step_5"]
    assert f"{record} is a record of {other.resolve()}, not of {step_dir.resolve()}" in text(
        messages
    )
    launcher.check()
    assert json.loads((output / LAUNCHED_FILE).read_text())["checkpoint"] == str(step_dir.resolve())


def test_a_record_naming_the_step_directory_through_a_symlink_counts(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    link = tmp_path / "link"
    link.symlink_to(run, target_is_directory=True)
    write_record(launcher.settings.step_output(5), "results.json", link / "step_5")
    launcher.saved(5, run)
    assert popen.calls == []


def test_a_record_naming_a_missing_checkpoint_does_not_count(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    gone = tmp_path / "removed run" / "step_5"
    write_record(launcher.settings.step_output(5), "results.json", gone)
    launcher.saved(5, run)
    assert launched_steps(popen) == ["step_5"]
    assert f"is a record of {gone}" in text(messages)


def test_a_record_of_another_checkpoint_beside_one_of_this_checkpoint_counts(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_dir = save_step(run, 5)
    output = launcher.settings.step_output(5)
    write_record(output, "results.json", tmp_path / "earlier run" / "step_5")
    write_record(output, LAUNCHED_FILE, step_dir.resolve())
    launcher.saved(5, run)
    assert popen.calls == []


@pytest.mark.parametrize(
    "content",
    ["{", "[1]", '{"argv": ["sbatch", "job.sh"]}', '{"checkpoint": 5}', None],
    ids=["truncated", "not-an-object", "no-checkpoint", "not-a-path", "a-directory"],
)
def test_a_record_that_cannot_be_read_counts_and_names_the_file_to_delete(
    tmp_path: Path, run: Path, messages: list[str], content: str | None
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    record = launcher.settings.step_output(5) / LAUNCHED_FILE
    if content is None:
        record.mkdir(parents=True)
    else:
        record.parent.mkdir(parents=True)
        record.write_text(content)
    launcher.saved(5, run)
    launcher.finish(run)
    assert popen.calls == []
    assert f"{record} cannot be read" in text(messages)
    assert "delete it to evaluate the step again" in text(messages)


def test_the_scan_takes_up_only_complete_steps(tmp_path: Path, run: Path) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    save_step(run, 10, complete=False)
    (run / "step_x").mkdir()
    (run / "step_3").write_text("")
    (run / "latest").symlink_to("step_5")
    launcher.scan(run)
    assert [argv[3] for argv, _ in popen.calls] == [str((run / "step_5").resolve())]


def test_the_scan_of_a_missing_checkpoint_directory_does_nothing(tmp_path: Path) -> None:
    popen = Launcher()
    make_launcher(tmp_path, popen).scan(tmp_path / "none")
    assert popen.calls == []


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("no sbatch"),
        ValueError("embedded null byte"),
        subprocess.SubprocessError("cannot start"),
        RuntimeError("anything else"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_a_command_that_cannot_start_leaves_no_record_and_is_tried_at_the_next_save(
    tmp_path: Path, run: Path, messages: list[str], error: Exception
) -> None:
    popen = Launcher(error=error)
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    launcher.saved(5, run)
    launcher.check()
    assert popen.attempts == 1
    assert f"could not launch step 5: {type(error).__name__}: {error}" in text(messages)
    assert "tried again at the next save" in text(messages)
    assert not (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()
    popen.error = None
    save_step(run, 6)
    launcher.saved(6, run)
    assert launched_steps(popen) == ["step_5", "step_6"]
    launcher.check()
    assert (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()


def test_a_command_that_exits_non_zero_is_tried_at_the_next_saves_at_most_three_times(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher(status=3)
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    launcher.saved(5, run)
    launcher.check()
    launcher.check()
    assert popen.attempts == 1
    log = launcher.settings.step_output(5) / LAUNCH_LOG_FILE
    assert f"exited with status 3; its output is in {log}" in text(messages)
    for step in (6, 7, 8):
        save_step(run, step, complete=False)
        launcher.saved(step, run)
        launcher.check()
    launcher.finish(None)
    assert popen.attempts == periodic.LAUNCH_ATTEMPTS == 3
    assert launched_steps(popen) == ["step_5"] * 3
    assert not (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()
    assert text(messages).count("step 5 is tried again at the next save") == 2
    assert "no attempt left in this run, step 5 is left to a later run" in text(messages)


def test_a_failed_step_is_tried_again_at_the_end_of_training(tmp_path: Path, run: Path) -> None:
    popen = Launcher(status=3)
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    launcher.saved(5, run)
    popen.status = 0
    launcher.finish(None)
    assert popen.attempts == 2
    assert (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()


def test_a_launch_that_fails_at_the_end_of_training_is_left_to_a_later_run(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher(error=OSError(errno.EAGAIN, "try again"))
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    launcher.finish(run)
    launcher.finish(run)
    assert popen.attempts == 1
    assert "tried again" not in text(messages)
    assert "no attempt left in this run, step 5 is left to a later run" in text(messages)


def test_a_launch_log_that_cannot_be_opened_is_tried_again_within_the_run(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    log = launcher.settings.step_output(5) / LAUNCH_LOG_FILE
    log.mkdir(parents=True)
    launcher.saved(5, run)
    assert popen.calls == [] and "could not launch step 5" in text(messages)
    log.rmdir()
    launcher.finish(None)
    assert launched_steps(popen) == ["step_5"]
    assert (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()


def test_a_command_that_exits_zero_is_recorded_with_its_checkpoint_and_arguments(
    tmp_path: Path, run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaced: list[tuple[Path, Path]] = []
    original = os.replace

    def replace(source: str | Path, target: str | Path) -> None:
        json.loads(Path(source).read_text())
        replaced.append((Path(source), Path(target)))
        original(source, target)

    popen = Launcher(status=None)
    launcher = make_launcher(tmp_path, popen)
    step_dir = save_step(run, 5)
    monkeypatch.setattr(periodic.os, "replace", replace)
    launcher.saved(5, run)
    output = launcher.settings.step_output(5)
    assert not (output / LAUNCHED_FILE).exists()
    popen.children[0].status = 0
    launcher.check()
    argv, _ = popen.calls[0]
    assert json.loads((output / LAUNCHED_FILE).read_text()) == {
        "checkpoint": str(step_dir.resolve()),
        "argv": argv,
    }
    ((source, target),) = replaced
    assert source.parent == output and source.name != LAUNCHED_FILE
    assert target == output / LAUNCHED_FILE
    assert sorted(path.name for path in output.iterdir()) == [LAUNCH_LOG_FILE, LAUNCHED_FILE]


def test_a_launch_that_cannot_be_recorded_is_a_warning_and_a_later_run_launches_it_again(
    tmp_path: Path, run: Path, messages: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(source: str | Path, target: str | Path) -> None:
        raise OSError(errno.ENOSPC, "no space left on device")

    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    original = os.replace
    monkeypatch.setattr(periodic.os, "replace", refuse)
    launcher.saved(5, run)
    launcher.check()
    monkeypatch.setattr(periodic.os, "replace", original)
    record = launcher.settings.step_output(5) / LAUNCHED_FILE
    assert not record.exists()
    assert f"could not record the launch of step 5 in {record}" in text(messages)
    assert "a later run may launch the step again" in text(messages)
    launcher.saved(5, run)
    launcher.finish(run)
    assert len(popen.calls) == 1
    later = Launcher()
    make_launcher(tmp_path, later).scan(run)
    assert len(later.calls) == 1


def test_a_running_command_is_never_waited_for_while_training_goes_on(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher(status=None)
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    launcher.saved(5, run)
    launcher.check()
    launcher.finish(None)
    assert launcher.waiting
    assert popen.children[0].timeouts == []
    assert not (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()


def test_the_default_launch_starts_the_command_and_logs_its_output(
    tmp_path: Path, run: Path
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
    launcher = EvaluationLauncher(settings, launch=launch)
    save_step(run, 5)
    launcher.saved(5, run)
    assert children[0].wait(timeout=60) == 0
    log = (settings.step_output(5) / LAUNCH_LOG_FILE).read_text()
    assert log.strip() == f"evaluating {(run / 'step_5').resolve()}"


class RecordingLauncher:
    """Stands in for ``EvaluationLauncher`` and records what the worker thread asks of it.

    ``waited`` records each end-of-training wait: its deadline, the thread that waited and
    whether the worker's last look had returned by then.
    """

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.waited: list[tuple[float, int, bool]] = []
        self.waiting = False
        self.fail_on = fail_on
        self.finished = threading.Event()

    def _record(self, kind: str, payload: Any) -> None:
        self.calls.append((kind, payload))
        if kind == self.fail_on:
            raise RuntimeError("boom")

    def begin(
        self, directory: str, resumed_step: int | None, saves_resumed_step: bool = False
    ) -> None:
        self._record("begin", (directory, resumed_step, saves_resumed_step))

    def saved(self, step: int, directory: str) -> None:
        self._record("saved", (step, directory))

    def check(self) -> None:
        self._record("check", None)

    def finish(self, directory: str | None) -> None:
        self._record("finish", directory)
        self.finished.set()

    def wait(self, deadline: float) -> None:
        self.waited.append((deadline, threading.get_ident(), self.finished.is_set()))
        if self.fail_on == "wait":
            raise RuntimeError("boom")


def make_hook(
    tmp_path: Path, recorder: RecordingLauncher, **options: Any
) -> PeriodicEvaluationHook:
    return PeriodicEvaluationHook(
        make_settings(tmp_path), launcher=cast(EvaluationLauncher, recorder), **options
    )


def config_for(run: Path) -> JobConfig:
    config = JobConfig()
    config.checkpoint.dir = str(run)
    return config


def test_the_hook_hands_a_run_to_its_worker_in_order(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    recorder = RecordingLauncher()
    hook = make_hook(tmp_path, recorder)
    runner = hook.runner()
    assert isinstance(runner, HookRunner)
    runner.on_train_begin(config_for(run))
    for step in (2, 4):
        runner.on_checkpoint_save(step, str(run))
    before = time.monotonic()
    runner.on_train_end(4, 0)
    after = time.monotonic()
    assert recorder.calls == [
        ("begin", (str(run), None, False)),
        ("saved", (2, str(run))),
        ("saved", (4, str(run))),
        ("finish", str(run)),
    ]
    assert hook._worker is not None and not hook._worker.is_alive()
    ((deadline, thread, finished),) = recorder.waited
    timeout = periodic.FINISH_TIMEOUT_SECONDS
    assert before + timeout <= deadline <= after + timeout
    assert thread == threading.get_ident() and finished


def test_the_hook_does_not_touch_the_filesystem_on_the_training_thread(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    recorder = RecordingLauncher()
    hook = make_hook(tmp_path, recorder)
    hook.on_train_begin(config_for(run))
    caller = threading.get_ident()
    seen: list[int] = []
    original = recorder._record

    def record(kind: str, payload: Any) -> None:
        seen.append(threading.get_ident())
        original(kind, payload)

    recorder._record = record
    hook.on_checkpoint_save(2, str(run))
    hook.on_train_end(2, 0)
    assert seen and caller not in seen


def test_a_worker_error_is_a_warning_and_the_worker_goes_on(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str]
) -> None:
    recorder = RecordingLauncher(fail_on="saved")
    hook = make_hook(tmp_path, recorder)
    hook.on_train_begin(config_for(run))
    hook.on_checkpoint_save(2, str(run))
    hook.on_checkpoint_save(4, str(run))
    hook.on_train_end(4, 0)
    assert [kind for kind, _ in recorder.calls] == ["begin", "saved", "saved", "finish"]
    assert "RuntimeError: boom" in text(messages)


def test_a_failing_last_look_still_ends_the_worker(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str]
) -> None:
    recorder = RecordingLauncher(fail_on="finish")
    hook = make_hook(tmp_path, recorder)
    hook.on_train_begin(config_for(run))
    hook.on_train_end(1, 0)
    assert hook._worker is not None and not hook._worker.is_alive()
    assert "could not handle finish" in text(messages)


def test_a_pending_step_is_rechecked_while_training_goes_on(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    recorder = RecordingLauncher()
    recorder.waiting = True
    hook = make_hook(tmp_path, recorder, poll_interval=0.01)
    hook.on_train_begin(config_for(run))
    deadline = time.monotonic() + 30
    while not any(kind == "check" for kind, _ in recorder.calls) and time.monotonic() < deadline:
        time.sleep(0.01)
    hook.on_train_end(1, 0)
    assert any(kind == "check" for kind, _ in recorder.calls)


def test_a_worker_that_does_not_finish_is_reported_and_not_waited_for(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str]
) -> None:
    release = threading.Event()

    class Slow(RecordingLauncher):
        def finish(self, directory: str | None) -> None:
            release.wait(30)

    slow = Slow()
    hook = make_hook(tmp_path, slow, finish_timeout=0.05)
    hook.on_train_begin(config_for(run))
    hook.on_train_end(1, 0)
    assert "did not finish its last check" in text(messages)
    assert slow.waited == []
    release.set()
    assert hook._worker is not None
    hook._worker.join(30)


def test_an_error_reading_the_exit_statuses_at_the_end_is_a_warning(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str]
) -> None:
    recorder = RecordingLauncher(fail_on="wait")
    hook = make_hook(tmp_path, recorder)
    hook.on_train_begin(config_for(run))
    hook.on_train_end(1, 0)
    assert len(recorder.waited) == 1
    assert "could not read the exit status" in text(messages)
    assert "RuntimeError: boom" in text(messages)


def test_the_end_of_training_records_a_command_that_exits_in_time_and_warns_of_a_running_one(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str]
) -> None:
    children = {"step_5": ExitsWhenWaitedFor(), "step_6": FakeChild(None)}
    popen = Launcher(child=lambda argv: children[Path(argv[3]).name])
    launcher = make_launcher(tmp_path, popen)
    hook = PeriodicEvaluationHook(launcher.settings, launcher=launcher, finish_timeout=30)
    save_step(run, 5)
    save_step(run, 6)
    hook.on_train_begin(config_for(run))
    hook.on_train_end(6, 0)
    assert sorted(launched_steps(popen)) == ["step_5", "step_6"]
    assert (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()
    assert not (launcher.settings.step_output(6) / LAUNCHED_FILE).exists()
    results = launcher.settings.step_output(6) / "results.json"
    assert "command for step 6 is still running at the end of training" in text(messages)
    assert f"launches step 6 again unless {results} exists by then" in text(messages)
    timeouts = children["step_5"].timeouts + children["step_6"].timeouts
    assert len(timeouts) == 2
    assert all(timeout is not None and 0 <= timeout <= 30 for timeout in timeouts)


def test_the_end_of_training_waits_for_a_real_command_and_records_it(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    script = tmp_path / "submit.py"
    script.write_text("import sys\nprint('queued', sys.argv[1])\n")
    settings = make_settings(
        tmp_path, command=f"{sys.executable} {script} {{checkpoint}} {{output}}"
    )
    hook = PeriodicEvaluationHook(settings, finish_timeout=60)
    step_dir = save_step(run, 5)
    hook.on_train_begin(config_for(run))
    hook.on_train_end(5, 0)
    record = json.loads((settings.step_output(5) / LAUNCHED_FILE).read_text())
    assert record["checkpoint"] == str(step_dir.resolve())
    log = (settings.step_output(5) / LAUNCH_LOG_FILE).read_text()
    assert log.strip() == f"queued {step_dir.resolve()}"


def test_the_other_ranks_start_nothing(
    tmp_path: Path, run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(periodic.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(periodic.dist, "get_rank", lambda: 1)
    recorder = RecordingLauncher()
    hook = make_hook(tmp_path, recorder)
    hook.on_train_begin(config_for(run))
    hook.on_checkpoint_save(2, str(run))
    hook.on_train_end(2, 0)
    assert hook._worker is None and recorder.calls == []


def test_an_unopenable_launch_log_claims_nothing_and_spares_the_next_step(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    save_step(run, 6)
    output = launcher.settings.step_output(5)
    (output / LAUNCH_LOG_FILE).mkdir(parents=True)
    launcher.scan(run)
    assert [Path(argv[3]).name for argv, _ in popen.calls] == ["step_6"]
    assert not (output / LAUNCHED_FILE).exists()
    assert "could not launch step 5" in text(messages)


class ClosingFails:
    """A launch log whose close fails after the command was started."""

    def __init__(self, path: Path, mode: str) -> None:
        self.stream = open(path, mode)  # noqa: SIM115

    def fileno(self) -> int:
        return self.stream.fileno()

    def __enter__(self) -> ClosingFails:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stream.close()
        raise OSError(5, "input/output error")


def test_a_log_that_fails_to_close_after_the_launch_leaves_the_step_running(
    tmp_path: Path, run: Path, messages: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    popen = Launcher(status=None)
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    monkeypatch.setattr(periodic, "open", ClosingFails, raising=False)
    launcher.saved(5, run)
    monkeypatch.delattr(periodic, "open")
    assert len(popen.calls) == 1
    assert "could not close the launch log of step 5" in text(messages)
    assert launcher.waiting
    launcher.saved(5, run)
    launcher.check()
    assert len(popen.calls) == 1
    popen.children[0].status = 0
    launcher.check()
    assert (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()


def test_the_scan_logs_each_incomplete_step_it_drops(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 3, complete=False)
    save_step(run, 4)
    launcher.scan(run)
    assert [Path(argv[3]).name for argv, _ in popen.calls] == ["step_4"]
    assert "skips step 3" in text(messages) and "not complete" in text(messages)
    assert "skips step 4" not in text(messages)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_an_output_root_that_cannot_be_written_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "locked"
    root.mkdir()
    root.chmod(0o500)
    try:
        with pytest.raises(ValueError, match="not writable"):
            PeriodicEvaluation(
                config="run.toml", tasks=str(SAMPLE_TASKS), command=COMMAND, output_root=str(root)
            )
    finally:
        root.chmod(0o700)


def test_a_thread_that_cannot_start_turns_the_hook_off_and_training_goes_on(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    recorder = RecordingLauncher()
    hook = make_hook(tmp_path, recorder)
    monkeypatch.setattr(threading.Thread, "start", refuse)
    hook.on_train_begin(config_for(run))
    monkeypatch.undo()
    hook.on_checkpoint_save(2, str(run))
    hook.on_train_end(2, 0)
    assert hook._worker is None and recorder.calls == []
    assert "periodic evaluation is off" in text(messages)


def alternate_metadata_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.stat
    calls = {"count": 0}

    def stat(self: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if self.name == ".metadata":
            calls["count"] += 1
            if calls["count"] % 2 == 0:
                raise OSError(errno.ENOENT, "gone for a moment")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)


def test_a_resumed_step_the_run_saves_again_is_not_launched_while_its_metadata_is_missing(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_dir = save_step(run, 0)
    launcher.begin(run, 0, saves_resumed_step=True)
    launcher.saved(0, run)
    (step_dir / ".metadata").unlink()
    launcher.check()
    (step_dir / ".metadata").write_text("new")
    launcher.check()
    assert popen.calls == []


def test_a_resumed_step_the_run_saves_again_is_not_launched_when_its_metadata_cannot_be_read(
    tmp_path: Path, run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 0)
    launcher.begin(run, 0, saves_resumed_step=True)
    alternate_metadata_stat(monkeypatch)
    launcher.saved(0, run)
    launcher.check()
    launcher.check()
    assert popen.calls == []


def test_a_deferred_step_is_not_launched_by_the_first_look_a_check_or_its_own_save(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 0)
    launcher.begin(run, 0, saves_resumed_step=True)
    launcher.check()
    launcher.saved(0, run)
    launcher.check()
    assert popen.calls == []
    assert not launcher.waiting
    assert "skips step 0" not in text(messages)


def test_an_incomplete_deferred_step_is_not_logged_as_incomplete(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 0, complete=False)
    launcher.begin(run, 0, saves_resumed_step=True)
    assert popen.calls == [] and "skips step 0" not in text(messages)


def test_a_deferred_step_is_launched_once_when_a_later_step_is_saved(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 0)
    launcher.begin(run, 0, saves_resumed_step=True)
    save_step(run, 2)
    launcher.saved(2, run)
    assert sorted(Path(argv[3]).name for argv, _ in popen.calls) == ["step_0", "step_2"]
    launcher.saved(2, run)
    launcher.finish(run)
    assert len(popen.calls) == 2


def test_a_deferred_step_is_launched_once_by_the_last_look_when_no_later_step_is_saved(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 0)
    launcher.begin(run, 0, saves_resumed_step=True)
    launcher.saved(0, run)
    launcher.finish(run)
    assert [Path(argv[3]).name for argv, _ in popen.calls] == ["step_0"]
    launcher.finish(run)
    assert len(popen.calls) == 1


def test_a_deferred_step_is_released_by_the_last_look_without_a_checkpoint_directory(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 0)
    launcher.begin(run, 0, saves_resumed_step=True)
    launcher.finish(None)
    assert [Path(argv[3]).name for argv, _ in popen.calls] == ["step_0"]


def test_a_step_zero_that_did_not_exist_at_the_start_is_launched_as_soon_as_it_is_complete(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    launcher.begin(run, None)
    save_step(run, 0)
    launcher.saved(0, run)
    assert [Path(argv[3]).name for argv, _ in popen.calls] == ["step_0"]


def dynamic_config(run: Path) -> JobConfig:
    from kempnerforge.config.checkpoint import DynamicCheckpointWindow

    config = config_for(run)
    config.checkpoint.dyn_ckpt_window = DynamicCheckpointWindow(start=0, stop=8)
    return config


def first_look(tmp_path: Path, config: JobConfig) -> tuple[str, Any]:
    recorder = RecordingLauncher()
    hook = make_hook(tmp_path, recorder)
    hook.on_train_begin(config)
    hook.on_train_end(1, 0)
    return recorder.calls[0]


def test_the_hook_hands_the_worker_the_step_the_run_resumed_from(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    for step in (50, 100, 200):
        save_step(run, step)
    (run / "latest").symlink_to("step_100")
    assert first_look(tmp_path, dynamic_config(run)) == ("begin", (str(run), 100, False))


def test_the_hook_finds_the_resumed_step_without_a_latest_link(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    save_step(run, 100)
    save_step(run, 200, complete=False)
    assert first_look(tmp_path, config_for(run)) == ("begin", (str(run), 100, False))


def test_the_hook_says_the_run_saves_step_zero_again_when_it_resumes_from_a_dynamic_milestone(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    save_step(run, 0)
    assert first_look(tmp_path, dynamic_config(run)) == ("begin", (str(run), 0, True))


def test_the_hook_says_step_zero_is_not_saved_again_when_it_is_not_a_dynamic_milestone(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    save_step(run, 0)
    assert first_look(tmp_path, config_for(run)) == ("begin", (str(run), 0, False))


def test_a_fresh_start_has_no_resumed_step(tmp_path: Path, run: Path, rank_zero: None) -> None:
    save_step(run, 0, complete=False)
    assert first_look(tmp_path, dynamic_config(run)) == ("begin", (str(run), None, False))
    assert first_look(tmp_path, config_for(tmp_path / "none")) == (
        "begin",
        (str(tmp_path / "none"), None, False),
    )


def test_a_resumed_checkpoint_not_named_by_its_step_has_no_resumed_step(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    save_step(run, 100)
    (tmp_path / "imported").mkdir()
    (run / "latest").symlink_to(tmp_path / "imported")
    assert first_look(tmp_path, config_for(run)) == ("begin", (str(run), None, False))


def test_an_error_while_looking_for_the_resumed_step_counts_as_a_fresh_start_and_never_raises(
    tmp_path: Path, run: Path, rank_zero: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: Path) -> bool:
        raise OSError(5, "input/output error")

    save_step(run, 0)
    recorder = RecordingLauncher()
    hook = make_hook(tmp_path, recorder)
    monkeypatch.setattr(Path, "exists", refuse)
    hook.on_train_begin(dynamic_config(run))
    monkeypatch.undo()
    hook.on_train_end(1, 0)
    assert recorder.calls[0] == ("begin", (str(run), None, False))


def test_a_resume_defers_the_steps_above_it_and_evaluates_again_a_step_it_saves_again(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 100)
    step_200 = save_step(run, 200)
    output = launcher.settings.step_output(200)
    results = write_record(output, "results.json", step_200.resolve())
    write_record(output, LAUNCHED_FILE, step_200.resolve())
    launcher.begin(run, 100)
    assert launched_steps(popen) == ["step_100"]
    launcher.saved(200, run)
    launcher.check()
    assert launched_steps(popen) == ["step_100"]
    save_step(run, 300)
    launcher.saved(300, run)
    assert launched_steps(popen) == ["step_100", "step_200", "step_300"]
    assert not results.exists()
    assert "evaluates step 200 again since the run saved it again" in text(messages)
    launcher.check()
    record = json.loads((output / LAUNCHED_FILE).read_text())
    assert record == {"checkpoint": str(step_200.resolve()), "argv": popen.calls[1][0]}
    launcher.finish(run)
    assert launched_steps(popen) == ["step_100", "step_200", "step_300"]


def test_a_save_releases_only_the_deferred_steps_below_it(tmp_path: Path, run: Path) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    for step in (100, 200, 300):
        save_step(run, step)
    launcher.begin(run, 100)
    save_step(run, 150)
    launcher.saved(150, run)
    assert launched_steps(popen) == ["step_100", "step_150"]
    launcher.saved(200, run)
    assert launched_steps(popen) == ["step_100", "step_150"]
    launcher.saved(300, run)
    assert launched_steps(popen) == ["step_100", "step_150", "step_200"]
    launcher.finish(run)
    assert launched_steps(popen) == ["step_100", "step_150", "step_200", "step_300"]


def test_a_deferred_step_the_run_does_not_save_again_keeps_its_record(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 100)
    step_200 = save_step(run, 200)
    results = write_record(launcher.settings.step_output(200), "results.json", step_200.resolve())
    launcher.begin(run, 100)
    save_step(run, 250)
    launcher.saved(250, run)
    launcher.finish(run)
    assert launched_steps(popen) == ["step_100", "step_250"]
    assert results.exists()


def test_a_resume_from_step_zero_that_saves_it_again_keeps_its_earlier_record(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_0 = save_step(run, 0)
    results = write_record(launcher.settings.step_output(0), "results.json", step_0.resolve())
    launcher.begin(run, 0, saves_resumed_step=True)
    launcher.saved(0, run)
    assert popen.calls == []
    save_step(run, 2)
    launcher.saved(2, run)
    launcher.finish(run)
    assert launched_steps(popen) == ["step_2"]
    assert results.exists()


def test_a_fresh_start_defers_every_step_directory_on_disk(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_5 = save_step(run, 5)
    step_10 = save_step(run, 10, complete=False)
    results = write_record(launcher.settings.step_output(5), "results.json", step_5.resolve())
    launcher.begin(run, None)
    launcher.check()
    assert popen.calls == [] and "skips step" not in text(messages)
    launcher.saved(5, run)
    assert popen.calls == []
    finish_flush(step_10)
    launcher.saved(10, run)
    assert launched_steps(popen) == ["step_5"]
    assert not results.exists()
    launcher.finish(run)
    assert launched_steps(popen) == ["step_5", "step_10"]
    assert text(messages).count("since the run saved it again") == 1


def test_a_record_of_a_step_saved_again_that_cannot_be_removed_is_a_failed_launch(
    tmp_path: Path, run: Path, messages: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_5 = save_step(run, 5)
    results = write_record(launcher.settings.step_output(5), "results.json", step_5.resolve())
    launcher.begin(run, None)
    launcher.saved(5, run)
    original = Path.unlink

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(errno.EACCES, "permission denied", str(self))

    monkeypatch.setattr(Path, "unlink", refuse)
    save_step(run, 6)
    launcher.saved(6, run)
    monkeypatch.setattr(Path, "unlink", original)
    assert launched_steps(popen) == ["step_6"]
    assert "could not launch step 5: PermissionError" in text(messages)
    launcher.finish(run)
    assert launched_steps(popen) == ["step_6", "step_5"]
    assert not results.exists()
