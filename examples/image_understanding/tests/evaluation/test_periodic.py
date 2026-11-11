"""Tests for evaluating each checkpoint a training run saves."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from evaluation import periodic
from evaluation.periodic import (
    LAUNCH_LOG_FILE,
    LAUNCHED_FILE,
    EvaluationLauncher,
    PeriodicEvaluation,
    PeriodicEvaluationHook,
    child_environment,
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
    assert argv[2:] == ["run.toml", str(step_dir.resolve()), str(SAMPLE_TASKS)] + [
        str(launcher.settings.step_output(5))
    ]
    assert options["start_new_session"] is True
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stderr"] == subprocess.STDOUT
    assert options["env"]["KEEP"] == "1"
    assert not {"WORLD_SIZE", "SLURM_NTASKS", "TORCHELASTIC_RUN_ID"} & set(options["env"])
    record = json.loads((launcher.settings.step_output(5) / LAUNCHED_FILE).read_text())
    assert record == {"argv": argv}
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
    make_launcher(tmp_path, first).saved(5, run)
    save_step(run, 10)
    resumed = Launcher()
    again = make_launcher(tmp_path, resumed)
    again.scan(run)
    again.saved(10, run)
    again.saved(5, run)
    assert len(first.calls) == 1
    assert [argv[3] for argv, _ in resumed.calls] == [str((run / "step_10").resolve())]


def test_a_step_with_results_is_not_launched_again(tmp_path: Path, run: Path) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    output = launcher.settings.step_output(5)
    output.mkdir(parents=True)
    (output / "results.json").write_text("{}")
    launcher.saved(5, run)
    assert popen.calls == []


def test_a_claimed_step_is_not_launched_by_another_run(tmp_path: Path, run: Path) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    claimed = launcher.settings.step_output(5) / LAUNCHED_FILE
    claimed.parent.mkdir(parents=True)
    claimed.write_text("{}")
    launcher.saved(5, run)
    assert popen.calls == []
    assert claimed.exists()


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


def test_a_failed_launch_is_a_warning_and_is_released(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher(error=FileNotFoundError("no sbatch"))
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    launcher.saved(5, run)
    assert "could not launch step 5" in text(messages)
    assert not (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()
    launcher.finish(None)
    assert popen.calls == []


def test_a_command_that_exits_non_zero_is_a_warning_and_is_not_started_again(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher(status=3)
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    launcher.saved(5, run)
    launcher.finish(None)
    assert "exited with status 3" in text(messages)
    assert (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()
    again = Launcher()
    make_launcher(tmp_path, again).scan(run)
    assert again.calls == []


def test_a_running_command_is_never_waited_for(tmp_path: Path, run: Path) -> None:
    popen = Launcher(status=None)
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    launcher.saved(5, run)
    launcher.check()
    launcher.finish(None)
    assert launcher.waiting
    assert (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()


def test_a_claim_is_won_by_one_process_only(tmp_path: Path) -> None:
    output = tmp_path / "step_1"
    assert periodic._claim(output, ["a", "b"]) is True
    assert periodic._claim(output, ["c"]) is False
    assert json.loads((output / LAUNCHED_FILE).read_text()) == {"argv": ["a", "b"]}


def test_a_step_claimed_between_the_check_and_the_launch_is_not_launched(
    tmp_path: Path, run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    monkeypatch.setattr(periodic, "_claim", lambda output, argv: False)
    launcher.saved(5, run)
    assert popen.calls == []


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
    """Stands in for ``EvaluationLauncher`` and records what the worker thread asks of it."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.waiting = False
        self.fail_on = fail_on
        self.finished = threading.Event()

    def _record(self, kind: str, payload: Any) -> None:
        self.calls.append((kind, payload))
        if kind == self.fail_on:
            raise RuntimeError("boom")

    def scan(self, directory: str, rewritten: frozenset[int] = frozenset()) -> None:
        self._record("scan", (directory, rewritten))

    def saved(self, step: int, directory: str) -> None:
        self._record("saved", (step, directory))

    def check(self) -> None:
        self._record("check", None)

    def finish(self, directory: str | None) -> None:
        self._record("finish", directory)
        self.finished.set()


def make_hook(
    tmp_path: Path, recorder: RecordingLauncher, **options: Any
) -> PeriodicEvaluationHook:
    return PeriodicEvaluationHook(make_settings(tmp_path), launcher=recorder, **options)


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
    runner.on_train_end(4, 0)
    assert recorder.calls == [
        ("scan", (str(run), frozenset())),
        ("saved", (2, str(run))),
        ("saved", (4, str(run))),
        ("finish", str(run)),
    ]
    assert not hook._worker.is_alive()


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
    assert [kind for kind, _ in recorder.calls] == ["scan", "saved", "saved", "finish"]
    assert "RuntimeError: boom" in text(messages)


def test_a_failing_last_look_still_ends_the_worker(
    tmp_path: Path, run: Path, rank_zero: None, messages: list[str]
) -> None:
    recorder = RecordingLauncher(fail_on="finish")
    hook = make_hook(tmp_path, recorder)
    hook.on_train_begin(config_for(run))
    hook.on_train_end(1, 0)
    assert not hook._worker.is_alive()
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

    hook = make_hook(tmp_path, Slow(), finish_timeout=0.05)
    hook.on_train_begin(config_for(run))
    hook.on_train_end(1, 0)
    assert "did not finish its last check" in text(messages)
    release.set()
    hook._worker.join(30)


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


def test_a_log_that_fails_to_close_after_the_launch_keeps_the_step_launched(
    tmp_path: Path, run: Path, messages: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    popen = Launcher(status=None)
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    monkeypatch.setattr(periodic, "open", ClosingFails, raising=False)
    launcher.saved(5, run)
    assert len(popen.calls) == 1
    assert (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()
    assert launcher.waiting
    monkeypatch.undo()
    again = Launcher()
    make_launcher(tmp_path, again).scan(run)
    assert again.calls == []


def test_a_record_that_cannot_be_written_leaves_the_step_launchable(
    tmp_path: Path, run: Path, messages: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "no space left on device")

    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 5)
    monkeypatch.setattr(periodic.json, "dump", refuse)
    launcher.saved(5, run)
    monkeypatch.undo()
    assert popen.calls == []
    assert not (launcher.settings.step_output(5) / LAUNCHED_FILE).exists()
    assert "could not launch step 5" in text(messages)
    later = Launcher()
    make_launcher(tmp_path, later).scan(run)
    assert len(later.calls) == 1


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


def test_a_step_handed_to_the_scan_as_rewritten_is_not_launched_while_its_metadata_is_missing(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    step_dir = save_step(run, 0)
    launcher.scan(run, frozenset({0}))
    launcher.saved(0, run)
    (step_dir / ".metadata").unlink()
    launcher.check()
    (step_dir / ".metadata").write_text("new")
    launcher.check()
    assert popen.calls == []


def test_a_step_handed_to_the_scan_as_rewritten_is_not_launched_when_its_metadata_cannot_be_read(
    tmp_path: Path, run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 0)
    launcher.scan(run, frozenset({0}))
    alternate_metadata_stat(monkeypatch)
    launcher.saved(0, run)
    launcher.check()
    launcher.check()
    assert popen.calls == []


def test_a_deferred_step_is_not_launched_by_the_scan_a_check_or_its_own_save(
    tmp_path: Path, run: Path, messages: list[str]
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 0)
    launcher.scan(run, frozenset({0}))
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
    launcher.scan(run, frozenset({0}))
    assert popen.calls == [] and "skips step 0" not in text(messages)


def test_a_deferred_step_is_launched_once_when_a_later_step_is_saved(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 0)
    launcher.scan(run, frozenset({0}))
    save_step(run, 2)
    launcher.saved(2, run)
    launcher.saved(2, run)
    launcher.finish(run)
    assert sorted(Path(argv[3]).name for argv, _ in popen.calls) == ["step_0", "step_2"]


def test_a_deferred_step_is_launched_once_by_the_last_look_when_no_later_step_is_saved(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    save_step(run, 0)
    launcher.scan(run, frozenset({0}))
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
    launcher.scan(run, frozenset({0}))
    launcher.finish(None)
    assert [Path(argv[3]).name for argv, _ in popen.calls] == ["step_0"]


def test_a_step_zero_that_did_not_exist_at_the_start_is_launched_as_soon_as_it_is_complete(
    tmp_path: Path, run: Path
) -> None:
    popen = Launcher()
    launcher = make_launcher(tmp_path, popen)
    launcher.scan(run, frozenset())
    save_step(run, 0)
    launcher.saved(0, run)
    assert [Path(argv[3]).name for argv, _ in popen.calls] == ["step_0"]


def dynamic_config(run: Path) -> JobConfig:
    from kempnerforge.config.checkpoint import DynamicCheckpointWindow

    config = config_for(run)
    config.checkpoint.dyn_ckpt_window = DynamicCheckpointWindow(start=0, stop=8)
    return config


def test_the_hook_defers_step_zero_when_it_is_a_dynamic_milestone_and_its_directory_exists(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    save_step(run, 0)
    recorder = RecordingLauncher()
    hook = make_hook(tmp_path, recorder)
    hook.on_train_begin(dynamic_config(run))
    hook.on_train_end(1, 0)
    assert recorder.calls[0] == ("scan", (str(run), frozenset({0})))


def test_the_hook_defers_nothing_when_the_step_zero_directory_does_not_exist(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    recorder = RecordingLauncher()
    hook = make_hook(tmp_path, recorder)
    hook.on_train_begin(dynamic_config(run))
    hook.on_train_end(1, 0)
    assert recorder.calls[0] == ("scan", (str(run), frozenset()))


def test_the_hook_defers_nothing_when_step_zero_is_not_a_dynamic_milestone(
    tmp_path: Path, run: Path, rank_zero: None
) -> None:
    save_step(run, 0)
    recorder = RecordingLauncher()
    hook = make_hook(tmp_path, recorder)
    hook.on_train_begin(config_for(run))
    hook.on_train_end(1, 0)
    assert recorder.calls[0] == ("scan", (str(run), frozenset()))


def test_an_error_while_looking_for_step_zero_counts_as_existing_and_never_raises(
    tmp_path: Path, run: Path, rank_zero: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: Path) -> bool:
        raise OSError(5, "input/output error")

    recorder = RecordingLauncher()
    hook = make_hook(tmp_path, recorder)
    monkeypatch.setattr(Path, "exists", refuse)
    hook.on_train_begin(dynamic_config(run))
    monkeypatch.undo()
    hook.on_train_end(1, 0)
    assert recorder.calls[0] == ("scan", (str(run), frozenset({0})))
