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
import shlex
import shutil
import string
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch.distributed as dist

from evaluation.evaluator import RESULTS_FILE
from evaluation.tasks import load_tasks
from kempnerforge.config.job import JobConfig
from kempnerforge.metrics.logger import get_logger
from kempnerforge.training.hooks import HookRunner, StepContext, TrainingHook

logger = get_logger(__name__)

PLACEHOLDERS = ("config", "tasks", "checkpoint", "step", "output")
OVERRIDES_PLACEHOLDER = "{overrides}"
REQUIRED_PLACEHOLDERS = ("checkpoint", "output")
LAUNCHED_FILE = "launched.json"
LAUNCH_LOG_FILE = "launch.log"
POLL_INTERVAL_SECONDS = 10.0

_STEP_PREFIX = "step_"
_DCP_METADATA_FILE = ".metadata"
_TRAIN_STATE_FILE = "train_state.pt"
_SIDECAR_FILE = "metadata.json"
_COMPLETE_FILES = (_DCP_METADATA_FILE, _TRAIN_STATE_FILE, _SIDECAR_FILE)


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
    """

    config: str
    tasks: str
    command: str
    output_root: str
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_command(self.command)
        load_tasks(self.tasks)
        root = Path(self.output_root)
        if root.exists() and not root.is_dir():
            raise ValueError(f"the evaluation output root {self.output_root!r} is not a directory")
        root.mkdir(parents=True, exist_ok=True)

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
            fields = [
                field for _, field, _, _ in string.Formatter().parse(token) if field is not None
            ]
        except ValueError as exc:
            raise ValueError(
                f"the evaluation command argument {token!r} is malformed: {exc}"
            ) from exc
        for name in fields:
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


def step_dir_complete(step_dir: Path) -> bool:
    """Whether the checkpoint in ``step_dir`` is complete and its evaluation can start.

    The DCP ``.metadata`` is written once every shard is durable, after which the
    checkpoint is readable whether or not the manager has pointed ``latest`` at it yet;
    ``train_state.pt`` and ``metadata.json`` are rank 0's other files of a save.
    Three ``stat`` calls, so cheap enough for the training thread.
    """
    return all((step_dir / name).exists() for name in _COMPLETE_FILES)


def _claim(output: Path, argv: Sequence[str]) -> bool:
    """Record that a launch of ``argv`` is under way; False when one was recorded before.

    The record is created exclusively, so of several processes claiming a step, one wins.
    """
    output.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(output / LAUNCHED_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"argv": list(argv)}, f)
    return True


def _is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


class PeriodicEvaluationHook(TrainingHook):
    """Starts the evaluation command for each complete checkpoint, on rank 0, without waiting.

    A saved step is first pending. It is launched as soon as its directory is complete,
    which is checked when it is saved and then at most every ``poll_interval`` seconds
    while a step is pending, and once more when training ends, after the loop has drained
    a pending asynchronous save. A step whose directory is gone (pruned) or never
    completed is skipped with a log line. A step with a ``results.json`` or a
    ``launched.json`` in its output directory is not launched again, so a resumed run
    starts only what no earlier run started; steps that are on disk and complete when
    training begins are taken up the same way.

    A launch that fails, or a command that exits with a non-zero status, is a warning, and
    its ``launched.json`` is removed so a later run starts that step again. Nothing here
    raises into the training loop.
    """

    def __init__(
        self,
        settings: PeriodicEvaluation,
        *,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        launch: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.settings = settings
        self._poll_interval = poll_interval
        self._clock = clock
        self._launch = launch
        self._pending: dict[int, Path] = {}
        self._handled: set[int] = set()
        self._children: list[tuple[int, subprocess.Popen, Path]] = []
        self._last_check = float("-inf")

    def runner(self) -> HookRunner:
        """A ``HookRunner`` holding this hook, as ``run_training`` takes it."""
        return HookRunner([self])

    def on_train_begin(self, config: JobConfig) -> None:
        self._guarded("scan the checkpoint directory", self._scan, Path(config.checkpoint.dir))

    def on_step_end(self, ctx: StepContext) -> None:
        if self._pending or self._children:
            self._guarded("check the pending checkpoints", self._tick)

    def on_checkpoint_save(self, step: int, path: str) -> None:
        if not _is_rank_zero():
            return
        self._pending[step] = Path(path).resolve() / f"{_STEP_PREFIX}{step}"
        self._guarded("check the pending checkpoints", self._check)

    def on_train_end(self, step: int, tokens_seen: int) -> None:
        self._guarded("finish the pending checkpoints", self._finish)

    def _guarded(self, what: str, action: Callable[..., None], *args: object) -> None:
        if not _is_rank_zero():
            return
        try:
            action(*args)
        except Exception as exc:
            logger.warning(f"Periodic evaluation could not {what}: {type(exc).__name__}: {exc}")

    def _scan(self, checkpoint_dir: Path) -> None:
        if not checkpoint_dir.is_dir():
            return
        for entry in checkpoint_dir.iterdir():
            suffix = entry.name[len(_STEP_PREFIX) :]
            is_step = entry.name.startswith(_STEP_PREFIX) and suffix.isdigit()
            if is_step and entry.is_dir() and step_dir_complete(entry):
                self._pending[int(suffix)] = entry.resolve()
        self._check()

    def _tick(self) -> None:
        self._reap()
        if self._pending and self._clock() - self._last_check >= self._poll_interval:
            self._check()

    def _check(self) -> None:
        self._reap()
        self._last_check = self._clock()
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

    def _finish(self) -> None:
        self._check()
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
        if not _claim(output, argv):
            logger.info(f"Periodic evaluation skips step {step}: already started")
            return
        log_path = output / LAUNCH_LOG_FILE
        try:
            with open(log_path, "ab") as log:
                child = self._launch(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            (output / LAUNCHED_FILE).unlink(missing_ok=True)
            logger.warning(f"Periodic evaluation could not launch step {step}: {exc}")
            return
        self._children.append((step, child, log_path))
        logger.info(f"Periodic evaluation launched step {step}: {shlex.join(argv)}")

    def _reap(self) -> None:
        running = []
        for step, child, log_path in self._children:
            status = child.poll()
            if status is None:
                running.append((step, child, log_path))
            elif status != 0:
                (self.settings.step_output(step) / LAUNCHED_FILE).unlink(missing_ok=True)
                logger.warning(
                    f"Periodic evaluation command for step {step} exited with status "
                    f"{status}; its output is in {log_path}"
                )
        self._children = running
