"""Evaluation of each checkpoint a training run saves, launched without waiting for it.

``PeriodicEvaluation`` holds the settings the training entry point reads from its
command line and checks them before anything is built. ``PeriodicEvaluationHook``
is a ``TrainingHook`` that, on rank 0 only, starts the user's command for every
saved step whose checkpoint is complete on disk, and never waits for it: the
command is started detached, so a scheduler's submit command returns at once and a
local script outlives the training process.

The command is a template whose placeholders are ``{config}``, ``{tasks}``,
``{checkpoint}`` (the step directory), ``{step}`` and ``{output}`` (that step's
output directory), plus ``{overrides}``, which stands for the run's config
overrides as separate arguments. It is split into arguments first and the
placeholders are filled in afterwards, so a path with a space stays one argument.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import string
import subprocess
import tempfile
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import torch.distributed as dist

from evaluation.checkpoint import step_dir_complete
from evaluation.evaluator import RESULTS_FILE
from evaluation.tasks import load_tasks
from kempnerforge.config.job import JobConfig
from kempnerforge.metrics.logger import get_logger
from kempnerforge.training.hooks import HookRunner, TrainingHook

logger = get_logger(__name__)

PLACEHOLDERS = ("config", "tasks", "checkpoint", "step", "output")
OVERRIDES_PLACEHOLDER = "{overrides}"
REQUIRED_PLACEHOLDERS = ("checkpoint", "output")
LAUNCHED_FILE = "launched.json"
LAUNCH_LOG_FILE = "launch.log"
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


def _claim(output: Path, argv: Sequence[str]) -> bool:
    """Record that a launch of ``argv`` is under way; False when one was recorded before.

    The record is created exclusively, so of several processes claiming a step, one wins.
    A record that cannot be written is removed again, so the step stays launchable.

    Raises:
        OSError: the record could not be created or written.
    """
    output.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(output / LAUNCHED_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"argv": list(argv)}, f)
    except Exception:
        (output / LAUNCHED_FILE).unlink(missing_ok=True)
        raise
    return True


def _steps_the_run_may_save_again(config: JobConfig) -> frozenset[int]:
    """The steps the run may save over a directory that already exists: step 0 or none.

    An error while looking counts as the directory existing. Never raises.
    """
    if not config.checkpoint.is_dynamic_milestone(0):
        return frozenset()
    try:
        exists = (Path(config.checkpoint.dir) / f"{_STEP_PREFIX}0").exists()
    except OSError:
        exists = True
    return frozenset({0}) if exists else frozenset()


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


class EvaluationLauncher:
    """Decides which saved steps to start and starts them; every call is synchronous.

    A saved step is pending until its directory is complete, then it is launched. A step
    whose directory is gone (pruned) or never completed is skipped with a log line. A
    step with a ``results.json`` or a ``launched.json`` in its output directory is not
    launched again, so a resumed run starts only what no run started before. A launch
    that fails is a warning and releases its ``launched.json``; a command that exits with
    a non-zero status is a warning that keeps it, so a step is started at most once and
    retrying one is deleting its ``launched.json``. Nothing here raises.

    A step the run may save again over a directory that already exists (step 0 of a
    dynamic checkpoint window, on a run that resumes from it) is deferred: its directory
    may be half rewritten, so it is not started until a later step is saved, since the
    checkpoint manager finishes one save before it starts the next, or until ``finish``.
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
        self._handled: set[int] = set()
        self._children: list[tuple[int, subprocess.Popen, Path]] = []
        self._deferred: dict[int, Path] = {}

    @property
    def waiting(self) -> bool:
        """Whether a step is pending or a launched command has not been seen to exit.

        A deferred step does not count: a later save or ``finish`` releases it, not polling.
        """
        return bool(self._pending or self._children)

    def saved(self, step: int, checkpoint_dir: str | Path) -> None:
        """Note that step ``step`` was saved under ``checkpoint_dir`` and start it if complete.

        A deferred step is left as it is, since its save may still be writing. Any other
        step first releases every deferred one: the checkpoint manager finishes the previous
        save and writes ``train_state.pt`` and ``metadata.json`` inside ``save()`` before the
        next one starts, so once a later step is announced every earlier directory is whole.
        """
        if step in self._deferred:
            return
        self._pending.update(self._deferred)
        self._deferred.clear()
        self._pending[step] = Path(checkpoint_dir).resolve() / f"{_STEP_PREFIX}{step}"
        self.check()

    def scan(self, checkpoint_dir: str | Path, rewritten: Collection[int] = ()) -> None:
        """Take up every complete step directory already under ``checkpoint_dir``.

        A step in ``rewritten`` is one the run may save again over its existing directory.
        Complete or not, it is deferred: recorded, neither pending nor logged as incomplete,
        and released by the save of a later step or by ``finish``.
        """
        directory = Path(checkpoint_dir)
        if directory.is_dir():
            for entry in directory.iterdir():
                suffix = entry.name[len(_STEP_PREFIX) :]
                is_step = entry.name.startswith(_STEP_PREFIX) and suffix.isdigit()
                if not is_step or not entry.is_dir():
                    continue
                step = int(suffix)
                if step in rewritten:
                    self._deferred[step] = entry.resolve()
                elif not step_dir_complete(entry):
                    logger.info(f"Periodic evaluation skips step {step}: {entry} is not complete")
                else:
                    self._pending[step] = entry.resolve()
        self.check()

    def check(self) -> None:
        """Start the pending steps that are complete and skip the ones that are gone."""
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

        Every deferred step is released first, so the look takes it up. What is still
        incomplete after it never will be, and is skipped with a log line.
        """
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

    def _start(self, step: int, step_dir: Path) -> None:
        if step in self._handled:
            return
        self._handled.add(step)
        output = self.settings.step_output(step)
        if (output / RESULTS_FILE).exists() or (output / LAUNCHED_FILE).exists():
            logger.info(f"Periodic evaluation skips step {step}: already started or finished")
            return
        argv = self.settings.argv(step_dir, step)
        log_path = output / LAUNCH_LOG_FILE
        child = self._spawn(step, argv, output, log_path)
        if child is None:
            return
        self._children.append((step, child, log_path))
        logger.info(f"Periodic evaluation launched step {step}: {shlex.join(argv)}")

    def _spawn(
        self, step: int, argv: list[str], output: Path, log_path: Path
    ) -> subprocess.Popen | None:
        """Claim the step and start its command; None when nothing was started.

        Once the command has started the step stays claimed whatever happens after, such as
        an error on closing the launch log.
        """
        child = None
        try:
            output.mkdir(parents=True, exist_ok=True)
            with open(log_path, "ab") as log:
                child = self._run(step, argv, output, log)
        except OSError as exc:
            if child is None:
                logger.warning(f"Periodic evaluation could not launch step {step}: {exc}")
            else:
                logger.warning(
                    f"Periodic evaluation could not close the launch log of step {step}: {exc}"
                )
        return child

    def _run(
        self, step: int, argv: list[str], output: Path, log: BinaryIO
    ) -> subprocess.Popen | None:
        claimed = False
        try:
            claimed = _claim(output, argv)
            if not claimed:
                logger.info(f"Periodic evaluation skips step {step}: already started")
                return None
            return self._launch(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_environment(os.environ),
                start_new_session=True,
            )
        except OSError as exc:
            if claimed:
                (output / LAUNCHED_FILE).unlink(missing_ok=True)
            logger.warning(f"Periodic evaluation could not launch step {step}: {exc}")
            return None

    def _reap(self) -> None:
        running = []
        for step, child, log_path in self._children:
            status = child.poll()
            if status is None:
                running.append((step, child, log_path))
            elif status != 0:
                logger.warning(
                    f"Periodic evaluation command for step {step} exited with status "
                    f"{status}; its output is in {log_path}. The step is not started again "
                    f"unless {self.settings.step_output(step) / LAUNCHED_FILE} is deleted"
                )
        self._children = running


class PeriodicEvaluationHook(TrainingHook):
    """Hands each saved step to an ``EvaluationLauncher`` on a worker thread, on rank 0 only.

    The training thread only puts a message on a queue, so nothing it does on the step path
    touches the filesystem or a scheduler: the worker checks whether a step is complete (every
    ``poll_interval`` seconds while one is pending, and when one is saved), claims and
    launches it, and reads the exit status of what it launched. Training ending asks the
    worker for a last look, after the loop has drained a pending asynchronous save, and
    waits for it at most ``finish_timeout`` seconds. A step saved outside the loop's
    hook, as by the shutdown handler, is found by that look and by the scan at the
    start of the next run. An error on the worker is a warning, never raised into training.

    At train begin the hook looks once, on the training thread and before the loop can
    save, whether step 0 is a dynamic checkpoint milestone whose directory already exists;
    if so the launcher defers it, so a step the run saves again is never evaluated while it
    may be half rewritten. That look never raises.
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
        rewritten = _steps_the_run_may_save_again(config)
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
        self._messages.put(("scan", (config.checkpoint.dir, rewritten)))

    def on_checkpoint_save(self, step: int, path: str) -> None:
        if self._worker is not None:
            self._messages.put(("saved", (step, path)))

    def on_train_end(self, step: int, tokens_seen: int) -> None:
        if self._worker is None:
            return
        self._messages.put(("finish", self._checkpoint_dir))
        self._worker.join(self._finish_timeout)
        if self._worker.is_alive():
            logger.warning(
                f"Periodic evaluation did not finish its last check within "
                f"{self._finish_timeout:g} s; steps still pending are left to the next run"
            )

    def _work(self) -> None:
        while True:
            try:
                kind, payload = self._messages.get(
                    timeout=self._poll_interval if self._launcher.waiting else None
                )
            except queue.Empty:
                kind, payload = "check", None
            try:
                if kind == "scan":
                    self._launcher.scan(*payload)
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
