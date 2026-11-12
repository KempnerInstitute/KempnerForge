"""Evaluation of each checkpoint a training run saves, launched without waiting for it.

``PeriodicEvaluation`` holds the settings the training entry point reads from its
command line and checks them before anything is built. ``PeriodicEvaluationHook``
is a ``TrainingHook`` that, on rank 0 only, starts the user's command for every
saved step whose checkpoint is complete on disk, and never waits for it while the
run trains: the command is started detached, so a scheduler's submit command returns
at once and a local script outlives the training process. When training ends, the
hook waits a bounded time for the commands still running, to record the ones that
succeeded.

The command is a template whose placeholders are ``{config}``, ``{tasks}``,
``{checkpoint}`` (the step directory), ``{step}`` and ``{output}`` (that step's
output directory), plus ``{overrides}``, which stands for the run's config
overrides as separate arguments. It is split into arguments first and the
placeholders are filled in afterwards, so a path with a space stays one argument;
every path is filled in absolute.

The command should submit the evaluation and return, as ``sbatch`` does. Nothing
limits how many evaluations start at once: every step that is due at the same
moment starts then, such as each complete step a resumed run finds. A local script
runs on the training job's node and GPUs, so it should queue its evaluations itself.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import shlex
import shutil
import string
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.distributed as dist

from evaluation.checkpoint import step_dir_complete
from evaluation.evaluator import RESULTS_FILE
from evaluation.tasks import load_tasks
from kempnerforge.config.job import JobConfig
from kempnerforge.metrics.logger import get_logger
from kempnerforge.resilience.elastic import resolve_resume_path
from kempnerforge.training.hooks import HookRunner, TrainingHook

logger = get_logger(__name__)

PLACEHOLDERS = ("config", "tasks", "checkpoint", "step", "output")
OVERRIDES_PLACEHOLDER = "{overrides}"
REQUIRED_PLACEHOLDERS = ("checkpoint", "output")
LAUNCHED_FILE = "launched.json"
LAUNCH_LOG_FILE = "launch.log"
LAUNCH_ATTEMPTS = 3
POLL_INTERVAL_SECONDS = 10.0
FINISH_TIMEOUT_SECONDS = 60.0

_DISTRIBUTED_VARIABLES = frozenset(
    {
        "RANK",
        "LOCAL_RANK",
        "GROUP_RANK",
        "ROLE_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "ROLE_NAME",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_NPROCS",
    }
)
_DISTRIBUTED_PREFIXES = ("TORCHELASTIC_",)
_STEP_PREFIX = "step_"
_RECORD_FILES = (RESULTS_FILE, LAUNCHED_FILE)


@dataclass(frozen=True)
class PeriodicEvaluation:
    """What evaluating each saved checkpoint needs, checked at construction.

    Attributes:
        config: path of the TOML config the run trains with.
        tasks: path of the task file, read once here so a mistake in it stops the run
            before it builds anything.
        command: template of the command that evaluates one step.
        output_root: directory that gets one ``step_<N>`` directory of results per step.
        overrides: the run's ``--section.key=value`` overrides, for ``{overrides}``.

    ``config`` and ``tasks`` are made absolute here, as ``{checkpoint}`` and ``{output}``
    are, so the command finds them from whatever directory it runs in.
    """

    config: str
    tasks: str
    command: str
    output_root: str
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", str(Path(self.config).resolve()))
        object.__setattr__(self, "tasks", str(Path(self.tasks).resolve()))
        _check_command(self.command)
        load_tasks(self.tasks)
        root = Path(self.output_root)
        if root.exists() and not root.is_dir():
            raise ValueError(f"the evaluation output root {self.output_root!r} is not a directory")
        root.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryFile(dir=root):
                pass
        except OSError as exc:
            raise ValueError(
                f"the evaluation output root {self.output_root!r} is not writable: {exc}"
            ) from exc

    def argv(self, step_dir: Path, step: int) -> list[str]:
        """The command's arguments for the checkpoint ``step_dir`` of step ``step``."""
        values = {
            "config": self.config,
            "tasks": self.tasks,
            "checkpoint": str(step_dir),
            "step": str(step),
            "output": str(self.step_output(step)),
        }
        argv: list[str] = []
        for token in shlex.split(self.command):
            if token == OVERRIDES_PLACEHOLDER:
                argv.extend(self.overrides)
            else:
                argv.append(token.format(**values))
        return argv

    def step_output(self, step: int) -> Path:
        """The directory the evaluation of step ``step`` writes to."""
        return Path(self.output_root).resolve() / f"{_STEP_PREFIX}{step}"


def _check_command(command: str) -> None:
    """Refuse a command template that is empty, names no known placeholder set or cannot run.

    Raises:
        ValueError: the template is empty, has a placeholder outside ``PLACEHOLDERS`` and
            ``{overrides}``, lacks ``{checkpoint}`` or ``{output}``, or its program is not found.
    """
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"the evaluation command {command!r} cannot be split: {exc}") from exc
    if not tokens:
        raise ValueError("the evaluation command is empty")
    used: set[str] = set()
    for token in tokens:
        if token == OVERRIDES_PLACEHOLDER:
            continue
        try:
            parsed = [
                (field, spec, conversion)
                for _, field, spec, conversion in string.Formatter().parse(token)
                if field is not None
            ]
        except ValueError as exc:
            raise ValueError(
                f"the evaluation command argument {token!r} is malformed: {exc}"
            ) from exc
        for name, spec, conversion in parsed:
            if spec or conversion:
                raise ValueError(
                    f"the placeholder {{{name}}} in the evaluation command takes no format "
                    f"specification or conversion (got {token!r})"
                )
            if name not in PLACEHOLDERS:
                raise ValueError(
                    f"unknown placeholder {{{name}}} in the evaluation command; valid "
                    f"placeholders are {[f'{{{p}}}' for p in PLACEHOLDERS]} and "
                    f"{OVERRIDES_PLACEHOLDER} as an argument of its own"
                )
            used.add(name)
    missing = [name for name in REQUIRED_PLACEHOLDERS if name not in used]
    if missing:
        raise ValueError(
            f"the evaluation command must contain {[f'{{{m}}}' for m in REQUIRED_PLACEHOLDERS]}; "
            f"missing {[f'{{{m}}}' for m in missing]}"
        )
    if shutil.which(tokens[0]) is None:
        raise ValueError(f"the evaluation command's program {tokens[0]!r} was not found")


def _step_of(name: str) -> int | None:
    """The ``N`` of a directory named ``step_N``, or None for any other name."""
    suffix = name[len(_STEP_PREFIX) :]
    return int(suffix) if name.startswith(_STEP_PREFIX) and suffix.isdigit() else None


def _step_directories(checkpoint_dir: str | Path) -> list[tuple[int, Path]]:
    """Every ``step_N`` directory under ``checkpoint_dir`` with its ``N``; none without one."""
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        return []
    found = []
    for entry in directory.iterdir():
        step = _step_of(entry.name)
        if step is not None and entry.is_dir():
            found.append((step, entry))
    return found


def _resume_of(config: JobConfig) -> tuple[int | None, bool]:
    """The step the run resumed from, and whether the run saves that step again.

    The step is found as the run found it, with ``resolve_resume_path`` on
    ``checkpoint.dir``. It is None for a fresh start, and when it cannot be told: the
    look raised, or found a directory not named ``step_N``. The run saves it again when it
    is step 0 and a dynamic checkpoint milestone, as the loop then saves step 0 before its
    first step. Never raises.
    """
    try:
        path = resolve_resume_path(config.checkpoint.dir)
        step = None if path is None else _step_of(path.name)
        return step, step == 0 and config.checkpoint.is_dynamic_milestone(0)
    except Exception:
        return None, False


def _recorded_checkpoint(record: Path) -> str | None:
    """The checkpoint the record file ``record`` names; None when it cannot be read or names none.

    Raises:
        FileNotFoundError: there is no ``record``.
    """
    try:
        checkpoint = json.loads(record.read_text(encoding="utf-8")).get("checkpoint")
    except FileNotFoundError:
        raise
    except (OSError, ValueError, AttributeError):
        return None
    return checkpoint if isinstance(checkpoint, str) else None


def _same_directory(path: str, step_dir: Path) -> bool:
    """Whether ``path`` is ``step_dir``, through a symlink too; False when either is missing."""
    try:
        return os.path.samefile(path, step_dir)
    except (OSError, ValueError):
        return False


def _counted_record(step: int, step_dir: Path, output: Path) -> bool:
    """Whether a record in ``output`` counts for the checkpoint ``step_dir`` of step ``step``.

    A ``results.json`` or a ``launched.json`` counts when it names ``step_dir`` or cannot be
    read; one of another checkpoint does not, so that step is evaluated again. A skip is
    logged, and a record that cannot be read or is of another checkpoint is a warning.
    """
    others: list[tuple[Path, str]] = []
    for record in (output / name for name in _RECORD_FILES):
        try:
            checkpoint = _recorded_checkpoint(record)
        except FileNotFoundError:
            continue
        if checkpoint is None:
            logger.warning(
                f"Periodic evaluation skips step {step}: {record} cannot be read, so it counts "
                f"as a launch of {step_dir}; delete it to evaluate the step again"
            )
            return True
        if _same_directory(checkpoint, step_dir):
            logger.info(f"Periodic evaluation skips step {step}: already started or finished")
            return True
        others.append((record, checkpoint))
    for record, checkpoint in others:
        logger.warning(
            f"Periodic evaluation evaluates step {step} again: {record} is a record of "
            f"{checkpoint}, not of {step_dir}"
        )
    return False


def _remove_records(step: int, output: Path) -> None:
    """Remove the records in ``output`` of what step ``step`` held before the run saved it again.

    Raises:
        OSError: a record could not be looked at or removed.
    """
    stale = [output / name for name in _RECORD_FILES if (output / name).exists()]
    for record in stale:
        record.unlink(missing_ok=True)
    if stale:
        logger.warning(
            f"Periodic evaluation evaluates step {step} again since the run saved it again; "
            f"removed {', '.join(str(record) for record in stale)}"
        )


def _is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def child_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` without what makes a launched process join the training run's process group.

    A launched evaluation is its own job, so it must not inherit the rendezvous variables
    of the ``torchrun`` that started training, whether it is a local script or an
    ``sbatch`` job that exports the environment.
    """
    return {
        key: value
        for key, value in environ.items()
        if key not in _DISTRIBUTED_VARIABLES and not key.startswith(_DISTRIBUTED_PREFIXES)
    }


@dataclass(frozen=True)
class _Launch:
    """A command started for the checkpoint ``step_dir`` that has not been seen to exit."""

    child: subprocess.Popen
    step_dir: Path
    argv: list[str]


class EvaluationLauncher:
    """Decides which saved steps to start and starts them; every call is synchronous.

    A saved step is pending until its directory is complete, then it is started unless a
    record of it counts: a ``results.json`` or a ``launched.json`` in its output directory
    that names its step directory, or that cannot be read. A record of another checkpoint,
    as after a rollback to a new ``checkpoint.dir`` with the same output root, does not
    count. So a resumed run starts only what no run recorded. A step whose directory is
    gone (pruned) or never completed is skipped with a log line. Which steps are running
    and which are done is kept in memory, so a run never starts a step twice at once.

    A launched command's exit status is read when the launcher checks, without waiting,
    and by ``wait`` at the end of training. Status 0 writes ``launched.json`` (for
    ``sbatch``, the job was queued). A command that cannot be started or that exits
    non-zero is a warning that leaves no record, and its step is tried again at the next
    save and at ``finish``, at most ``LAUNCH_ATTEMPTS`` times in one run; after the last
    try it is left to a later run, which tries it again since nothing records it. Nothing
    here raises.

    A step the run may save again over a directory that already exists is deferred (see
    ``begin``): its directory may be half rewritten, so it is not started until a later
    step is saved, since the checkpoint manager finishes one save before it starts the
    next, or until ``finish``. If the run saves it again, it holds new weights, unless it
    is the step the run resumed from: its records from before the run are then removed
    when it is started, and it is evaluated again.
    """

    def __init__(
        self,
        settings: PeriodicEvaluation,
        *,
        launch: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.settings = settings
        self._launch = launch
        self._pending: dict[int, Path] = {}
        self._deferred: dict[int, Path] = {}
        self._failed: dict[int, Path] = {}
        self._attempts: dict[int, int] = {}
        self._running: dict[int, _Launch] = {}
        self._done: set[int] = set()
        self._finished = False
        self._resumed_step: int | None = None
        self._saved_again: set[int] = set()

    @property
    def waiting(self) -> bool:
        """Whether a step is pending or a launched command has not been seen to exit.

        A deferred step does not count, nor does a failed one: a later save or ``finish``
        takes them up, not polling.
        """
        return bool(self._pending or self._running)

    def begin(
        self,
        checkpoint_dir: str | Path,
        resumed_step: int | None,
        saves_resumed_step: bool = False,
    ) -> None:
        """Take up the step directories already under ``checkpoint_dir`` when training begins.

        ``resumed_step`` is the step the run resumed from, None for a fresh start or when it
        cannot be told. Every step directory above it, every one when it is None, is
        deferred, since the run may save that step again over it; so is ``resumed_step``
        itself when ``saves_resumed_step``, though a save of it holds the weights the run
        loaded, so its records still count. A deferred step is neither pending nor logged
        as incomplete. The other steps are taken up as ``scan`` takes them up.
        """
        self._resumed_step = resumed_step
        for step, entry in _step_directories(checkpoint_dir):
            above = resumed_step is None or step > resumed_step
            if above or (saves_resumed_step and step == resumed_step):
                self._deferred[step] = entry.resolve()
            else:
                self._take_up(step, entry)
        self.check()

    def saved(self, step: int, checkpoint_dir: str | Path) -> None:
        """Note that step ``step`` was saved under ``checkpoint_dir`` and start it if complete.

        The steps whose launch failed are tried again, and every deferred step below
        ``step`` is released: the checkpoint manager finishes the previous save and writes
        ``train_state.pt`` and ``metadata.json`` inside ``save()`` before the next one
        starts, so once a later step is announced every earlier directory is whole. A
        deferred step stays deferred when its own save is announced, since that save may
        still be writing, and is marked as saved again unless it is the step the run
        resumed from.
        """
        self._retry()
        for earlier in [deferred for deferred in self._deferred if deferred < step]:
            self._pending[earlier] = self._deferred.pop(earlier)
        if step not in self._deferred:
            self._pending[step] = Path(checkpoint_dir).resolve() / f"{_STEP_PREFIX}{step}"
        elif step != self._resumed_step:
            self._saved_again.add(step)
        self.check()

    def scan(self, checkpoint_dir: str | Path) -> None:
        """Take up every complete step directory already under ``checkpoint_dir``.

        An incomplete one is skipped with a log line.
        """
        for step, entry in _step_directories(checkpoint_dir):
            self._take_up(step, entry)
        self.check()

    def check(self) -> None:
        """Start the pending steps that are complete and skip the ones that are gone.

        The exit status of every launched command is read first, without waiting. A step
        whose command failed is not tried again here, only at a save or at ``finish``.
        """
        self._reap()
        for step in sorted(self._pending):
            step_dir = self._pending[step]
            if not step_dir.exists():
                logger.warning(
                    f"Periodic evaluation skips step {step}: {step_dir} was removed before "
                    "it was started, so it was pruned; raise checkpoint.keep_last_n or "
                    "checkpoint.interval to keep more of them"
                )
                del self._pending[step]
            elif step_dir_complete(step_dir):
                del self._pending[step]
                self._start(step, step_dir)

    def finish(self, checkpoint_dir: str | Path | None) -> None:
        """Take a last look when training ends: a save that has drained, an emergency save.

        The steps whose launch failed are tried again and every deferred step is released
        first, so the look takes them up. What is still incomplete after it never will be,
        and is skipped with a log line. A launch that fails from here on is left to a later
        run.
        """
        self._retry()
        self._finished = True
        self._pending.update(self._deferred)
        self._deferred.clear()
        if checkpoint_dir is not None:
            self.scan(checkpoint_dir)
        else:
            self.check()
        for step, step_dir in sorted(self._pending.items()):
            logger.warning(
                f"Periodic evaluation skips step {step}: {step_dir} never became complete"
            )
        self._pending.clear()
        self._reap()

    def wait(self, deadline: float) -> None:
        """Wait until ``deadline``, a ``time.monotonic()`` value, for the commands still running.

        Meant for the end of training, after ``finish``. The exit statuses are then read as
        ``check`` reads them. A command still running is a warning that its launch is not
        recorded, so a later run launches its step again unless a ``results.json`` exists by
        then.
        """
        for launch in self._running.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                launch.child.wait(timeout=max(0.0, deadline - time.monotonic()))
        self._reap()
        for step in sorted(self._running):
            results = self.settings.step_output(step) / RESULTS_FILE
            logger.warning(
                f"Periodic evaluation command for step {step} is still running at the end of "
                f"training, so its launch is not recorded: a later run launches step {step} "
                f"again unless {results} exists by then"
            )

    def _take_up(self, step: int, entry: Path) -> None:
        if step_dir_complete(entry):
            self._pending[step] = entry.resolve()
        else:
            logger.info(f"Periodic evaluation skips step {step}: {entry} is not complete")

    def _retry(self) -> None:
        self._reap()
        self._pending.update(self._failed)
        self._failed.clear()

    def _start(self, step: int, step_dir: Path) -> None:
        if step in self._running or step in self._done:
            return
        output = self.settings.step_output(step)
        if step not in self._saved_again and _counted_record(step, step_dir, output):
            self._done.add(step)
            return
        argv = self.settings.argv(step_dir, step)
        self._attempts[step] = self._attempts.get(step, 0) + 1
        child = self._spawn(step, step_dir, argv, output)
        if child is not None:
            self._running[step] = _Launch(child, step_dir, argv)
            logger.info(f"Periodic evaluation launched step {step}: {shlex.join(argv)}")

    def _spawn(
        self, step: int, step_dir: Path, argv: list[str], output: Path
    ) -> subprocess.Popen | None:
        """Start the command of step ``step``; None, once the failure is handled, if it did not.

        The records of a step the run saved again are removed first. Once the command has
        started, an error such as one on closing the launch log is a warning only: the step
        is running.
        """
        child = None
        try:
            if step in self._saved_again:
                _remove_records(step, output)
            output.mkdir(parents=True, exist_ok=True)
            with open(output / LAUNCH_LOG_FILE, "ab") as log:
                child = self._launch(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=child_environment(os.environ),
                    start_new_session=True,
                )
        except Exception as exc:
            if child is None:
                self._fail(
                    step,
                    step_dir,
                    f"Periodic evaluation could not launch step {step}: "
                    f"{type(exc).__name__}: {exc}",
                )
            else:
                logger.warning(
                    f"Periodic evaluation could not close the launch log of step {step}: {exc}"
                )
        return child

    def _reap(self) -> None:
        for step, launch in list(self._running.items()):
            status = launch.child.poll()
            if status is None:
                continue
            del self._running[step]
            if status == 0:
                self._done.add(step)
                self._write_record(step, launch)
            else:
                log_path = self.settings.step_output(step) / LAUNCH_LOG_FILE
                self._fail(
                    step,
                    launch.step_dir,
                    f"Periodic evaluation command for step {step} exited with status "
                    f"{status}; its output is in {log_path}",
                )

    def _fail(self, step: int, step_dir: Path, reason: str) -> None:
        if self._attempts[step] < LAUNCH_ATTEMPTS and not self._finished:
            self._failed[step] = step_dir
            logger.warning(
                f"{reason}; step {step} is tried again at the next save or at the end of training"
            )
            return
        self._done.add(step)
        logger.warning(
            f"{reason}; with no attempt left in this run, step {step} is left to a later run, "
            "which tries it again since nothing records its launch"
        )

    def _write_record(self, step: int, launch: _Launch) -> None:
        record = self.settings.step_output(step) / LAUNCHED_FILE
        partial = record.with_name(f"{LAUNCHED_FILE}.partial")
        content = {"checkpoint": str(launch.step_dir), "argv": launch.argv}
        try:
            partial.write_text(json.dumps(content), encoding="utf-8")
            os.replace(partial, record)
        except OSError as exc:
            logger.warning(
                f"Periodic evaluation could not record the launch of step {step} in {record}: "
                f"{exc}; a later run may launch the step again"
            )


class PeriodicEvaluationHook(TrainingHook):
    """Hands each saved step to an ``EvaluationLauncher`` on a worker thread, on rank 0 only.

    The training thread only puts a message on a queue, so nothing it does on the step path
    touches the filesystem or a scheduler: the worker checks whether a step is complete (every
    ``poll_interval`` seconds while one is pending or a launched command has not been seen
    to exit, and when one is saved), launches it, and reads the exit status of what it
    launched. Training ending asks the worker for a last look, after the loop has drained a
    pending asynchronous save, and waits for it at most ``finish_timeout`` seconds. Once the
    worker has returned, the training thread waits for the commands still running until
    that same deadline, ``finish_timeout`` seconds after training ended, and reads their
    exit statuses; it never touches the launcher while the worker is alive. A step saved
    outside the loop's hook, as by the shutdown handler, is found by that last look and by
    the first look of the next run. An error on the worker or in that wait is a warning,
    never raised into training.

    At train begin the hook looks once, on the training thread and before the loop can
    save, for the step the run resumed from, and whether the run saves that step again
    (step 0 of a dynamic checkpoint window); the launcher defers the steps the run may save
    again, so none is evaluated while it may be half rewritten. That look never raises.
    """

    def __init__(
        self,
        settings: PeriodicEvaluation,
        *,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        finish_timeout: float = FINISH_TIMEOUT_SECONDS,
        launcher: EvaluationLauncher | None = None,
    ) -> None:
        self.settings = settings
        self._launcher = launcher or EvaluationLauncher(settings)
        self._poll_interval = poll_interval
        self._finish_timeout = finish_timeout
        self._messages: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._worker: threading.Thread | None = None
        self._checkpoint_dir: str | None = None

    def runner(self) -> HookRunner:
        """A ``HookRunner`` holding this hook, as ``run_training`` takes it."""
        return HookRunner([self])

    def on_train_begin(self, config: JobConfig) -> None:
        if not _is_rank_zero() or self._worker is not None:
            return
        self._checkpoint_dir = config.checkpoint.dir
        resumed_step, saves_resumed_step = _resume_of(config)
        worker = threading.Thread(target=self._work, name="periodic-evaluation", daemon=True)
        try:
            worker.start()
        except Exception as exc:
            logger.warning(
                f"Periodic evaluation could not start its worker thread ({type(exc).__name__}: "
                f"{exc}); periodic evaluation is off for this run"
            )
            return
        self._worker = worker
        self._messages.put(("begin", (config.checkpoint.dir, resumed_step, saves_resumed_step)))

    def on_checkpoint_save(self, step: int, path: str) -> None:
        if self._worker is not None:
            self._messages.put(("saved", (step, path)))

    def on_train_end(self, step: int, tokens_seen: int) -> None:
        if self._worker is None:
            return
        deadline = time.monotonic() + self._finish_timeout
        self._messages.put(("finish", self._checkpoint_dir))
        self._worker.join(self._finish_timeout)
        if self._worker.is_alive():
            logger.warning(
                f"Periodic evaluation did not finish its last check within "
                f"{self._finish_timeout:g} s; steps still pending, and launches whose exit "
                "was not read, are left to the next run"
            )
            return
        try:
            self._launcher.wait(deadline)
        except Exception as exc:
            logger.warning(
                f"Periodic evaluation could not read the exit status of what it launched: "
                f"{type(exc).__name__}: {exc}"
            )

    def _work(self) -> None:
        payload: Any
        while True:
            try:
                kind, payload = self._messages.get(
                    timeout=self._poll_interval if self._launcher.waiting else None
                )
            except queue.Empty:
                kind, payload = "check", None
            try:
                if kind == "begin":
                    self._launcher.begin(*payload)
                elif kind == "saved":
                    self._launcher.saved(*payload)
                elif kind == "check":
                    self._launcher.check()
                else:
                    self._launcher.finish(payload)
                    return
            except Exception as exc:
                logger.warning(
                    f"Periodic evaluation could not handle {kind}: {type(exc).__name__}: {exc}"
                )
                if kind == "finish":
                    return
