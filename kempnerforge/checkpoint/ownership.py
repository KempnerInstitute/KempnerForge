"""One training run per checkpoint directory.

Before a training run resumes or saves, rank 0 records the run in ``<checkpoint.dir>/.owner``,
and it marks the record released when the run ends cleanly. Two live runs in one directory
would overwrite each other's checkpoints, prune them and resume from them, so a run is
refused a directory that another run still owns.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OWNER_FILE = ".owner"

_ENDED_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)

_CLAIM_ID = re.compile(r"[0-9a-f]{32}")
_PID_NAMESPACE = "/proc/self/ns/pid"
_ATTEMPTS = 3
_RETRY_SEC = 1.0
_LOCK = threading.Lock()


class CheckpointDirInUseError(RuntimeError):
    """Another run that is alive, or not known to have ended, owns the checkpoint directory."""


def _after_fork_in_child() -> None:
    """Give a child forked from this process a random id of its own and a free lock.

    A thread of the parent may hold the lock when the process forks, and that thread does
    not run in the child, which would keep the lock held.
    """
    global _process_id, _LOCK
    _process_id = uuid.uuid4().hex
    _LOCK = threading.Lock()


_process_id = uuid.uuid4().hex
os.register_at_fork(after_in_child=_after_fork_in_child)


def _pid_namespace() -> int | None:
    """The inode of this process's pid namespace, or ``None`` where ``/proc`` does not show it."""
    try:
        return os.stat(_PID_NAMESPACE).st_ino
    except OSError:
        return None


def this_run() -> dict[str, Any]:
    """A new owner record for this process.

    ``claim`` is new for every record. ``process`` is a random id of this process, by which
    it knows its own records. ``pid_ns`` is the pid namespace ``pid`` is numbered in, since
    containers that share the host name but not that namespace reuse pids. ``slurm_job`` is
    the SLURM job, or job array element (``<array job>_<task>``), that runs the process, and
    ``slurm_restart`` counts the times SLURM requeued it.
    """
    job = os.environ.get("SLURM_JOB_ID") or None
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job and array_job and array_task:
        job = f"{array_job}_{array_task}"
    return {
        "claim": uuid.uuid4().hex,
        "slurm_cluster": (os.environ.get("SLURM_CLUSTER_NAME") or None) if job else None,
        "slurm_job": job,
        "slurm_restart": int(os.environ.get("SLURM_RESTART_COUNT") or 0) if job else None,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "process": _process_id,
        "pid_ns": _pid_namespace(),
        "since": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def _same_job(owner: dict[str, Any], me: dict[str, Any]) -> bool:
    """Whether ``owner`` ran in the SLURM job, of the same cluster, that runs this process."""
    job = (owner.get("slurm_job"), owner.get("slurm_cluster"))
    return bool(me["slurm_job"]) and job == (me["slurm_job"], me["slurm_cluster"])


def _this_process(owner: dict[str, Any], me: dict[str, Any]) -> bool:
    """Whether this process wrote ``owner`` in an earlier claim that it did not release."""
    keys = ("host", "pid", "process", "slurm_job", "slurm_restart")
    return all(owner.get(key) == me[key] for key in keys)


def _pid_alive(owner: dict[str, Any], me: dict[str, Any]) -> bool | None:
    """Whether ``owner``'s pid still runs, or ``None`` when this process cannot tell.

    Only a process on ``owner``'s host and in its pid namespace can tell, so a namespace that
    either record does not know cannot be checked.
    """
    pid = owner.get("pid")
    if owner.get("host") != me["host"] or me["pid_ns"] is None:
        return None
    if owner.get("pid_ns") != me["pid_ns"]:
        return None
    if type(pid) is not int or not 0 < pid < 2**31:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _job_alive(owner: dict[str, Any], me: dict[str, Any]) -> bool | None:
    """Whether ``owner``'s SLURM job has not ended, or ``None`` when this host cannot tell.

    ``squeue`` runs without the ``SQUEUE_*`` and ``SLURM_CLUSTERS`` settings that narrow what
    it lists, so a job it no longer lists has ended. It asks the owner's cluster by name when
    this process runs outside SLURM. A job of another cluster than this job's cannot be
    checked here.
    """
    cluster = owner.get("slurm_cluster")
    if cluster and me["slurm_cluster"] and cluster != me["slurm_cluster"]:
        return None
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SQUEUE_") and key != "SLURM_CLUSTERS"
    }
    command = [
        "squeue",
        "--noheader",
        "--states=all",
        f"--jobs={owner['slurm_job']}",
        "--format=%T",
    ]
    if cluster and not me["slurm_cluster"]:
        command.append(f"--clusters={cluster}")
    try:
        listed = subprocess.run(command, capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if listed.returncode == 0:
        states = {line.strip() for line in listed.stdout.splitlines()}
        return bool(states - _ENDED_STATES - {"", f"CLUSTER: {cluster}"})
    if "Invalid job id" in listed.stderr:
        return False
    return None


def _alive(owner: dict[str, Any], me: dict[str, Any]) -> bool | None:
    """Whether the run ``owner`` names is still alive, or ``None`` when this host cannot tell.

    A released record's run has ended. So has an earlier attempt of this process's SLURM job,
    since a requeued job goes back to pending before it runs again. Another run in this job,
    like a run outside SLURM, is alive while its pid runs, which only its own host can check.
    Any other SLURM job is alive until ``squeue`` lists it in a final state or no longer lists
    it. A pid is checked only on its host and in its pid namespace.
    """
    if owner.get("released"):
        return False
    if not owner.get("slurm_job"):
        return _pid_alive(owner, me)
    if not _same_job(owner, me):
        return _job_alive(owner, me)
    if owner.get("slurm_restart") != me["slurm_restart"]:
        return False
    return _pid_alive(owner, me)


def _read_owner(path: Path) -> dict[str, Any] | None:
    """The owner record at ``path``, or ``None`` when it is missing or unreadable."""
    try:
        owner = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(owner, dict) or not _CLAIM_ID.fullmatch(str(owner.get("claim"))):
        return None
    return owner


def _describe(owner: dict[str, Any] | None) -> str:
    """``owner`` in words, for the messages."""
    if owner is None:
        return "an owner record that cannot be read"
    where = f"host {owner.get('host')} (pid {owner.get('pid')}), since {owner.get('since')}"
    job = owner.get("slurm_job")
    return f"SLURM job {job} on {where}" if job else f"a run on {where}"


def _holds(path: Path, claim: str) -> bool:
    """Whether the record at ``path`` is the record of ``claim``."""
    record = _read_owner(path)
    return record is not None and record["claim"] == claim


def _link(source: Path, target: Path, claim: str) -> bool:
    """Link ``source`` to ``target`` unless ``target`` exists; a ``target`` of ``claim`` counts.

    NFS can answer a retried ``link`` that went through with ``EEXIST``, and the claim id in
    the record tells that case apart from a link that another claimer made.
    """
    try:
        os.link(source, target)
    except FileExistsError:
        return _holds(target, claim)
    return True


def _take_over(
    path: Path, held: dict[str, Any], tmp: Path, me: dict[str, Any]
) -> tuple[bool, Path | None]:
    """Replace ``path``, which holds ``held``, the record of an ended run, with this run's ``tmp``.

    Only the claimer that links ``tmp`` to ``.owner.<held's claim>.takeover`` may replace it:
    that claimer checks that ``path`` still holds ``held``, since an earlier claimer may have
    replaced it already, and renames a link of ``tmp`` over it. A token whose claimer has
    ended too is taken over the same way, level by level, so claimers that die holding
    tokens do not block the directory, and a token this process left in an earlier claim is
    taken over the same way. Each rename moves a name that only this claim uses, so a rename
    that NFS replays finds nothing to move. The tokens are removed afterward, but kept when a
    rename fails, since the record they guard may still be in place.

    Returns:
        Whether ``path`` was replaced, and the token of a claimer that is alive, or not known
        to have ended, when one holds it.
    """
    levels: list[tuple[Path, dict[str, Any], Path]] = []
    seen = {held["claim"]}
    while True:
        token = path.with_name(f"{OWNER_FILE}.{held['claim']}.takeover")
        levels.append((path, held, token))
        if _link(tmp, token, me["claim"]):
            break
        holder = _read_owner(token)
        if holder is None or holder["claim"] in seen:
            return False, token
        if not _this_process(holder, me) and _alive(holder, me) is not False:
            return False, token
        seen.add(holder["claim"])
        path, held = token, holder
    owned = [levels[-1][2]]
    try:
        for n, (path, held, _) in enumerate(reversed(levels)):
            if not _holds(path, held["claim"]):
                return False, None
            source = tmp.with_name(f"{OWNER_FILE}.{me['claim']}.{n}.tmp")
            try:
                if not _link(tmp, source, me["claim"]):
                    raise FileExistsError(source)
                os.rename(source, path)
            finally:
                source.unlink(missing_ok=True)
            if path is not levels[0][0]:
                owned.append(path)
        return True, None
    except BaseException:
        owned.clear()
        raise
    finally:
        for token in owned:
            token.unlink(missing_ok=True)


def _refusal(
    directory: Path, owner: dict[str, Any] | None, alive: bool | None
) -> CheckpointDirInUseError:
    """The error for ``directory`` owned by ``owner``, which is alive or not known to have ended."""
    record = directory / OWNER_FILE
    reason = (
        f"which is still running (its record is {record})"
        if alive
        else f"and this process cannot tell whether it is still running; remove {record} if "
        "it is not"
    )
    return CheckpointDirInUseError(
        f"Checkpoint directory {directory} belongs to {_describe(owner)}, {reason}. Two runs in "
        "one checkpoint directory overwrite, prune and resume from each other's checkpoints: "
        "point checkpoint.dir at a directory of this run's own."
    )


def claim_checkpoint_dir(directory: Path) -> str:
    """Record this run as the owner of ``directory`` unless another run owns it.

    The record is written to a temp file of its own and linked into place, so it appears
    whole and only one claimer creates it. A record this process wrote is replaced, unless it
    released it, since another claimer may be taking a released record over. A record
    of a run that has ended is replaced by one claimer alone (see ``_take_over``), which reads
    the record back before it counts the claim; the replacement is logged as a warning unless
    the record was released. Claims and releases in one process run one at a time.

    Returns:
        The new record's claim id, for ``release_checkpoint_dir``.

    Raises:
        CheckpointDirInUseError: A run that is alive, or whose state this host cannot
            check, owns ``directory``, or another claimer is taking it over.
    """
    with _LOCK:
        return _claim(directory)


def _claim(directory: Path) -> str:
    """``claim_checkpoint_dir``, with this process's lock held."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OWNER_FILE
    me = this_run()
    claim = me["claim"]
    tmp = directory / f"{OWNER_FILE}.{claim}.tmp"
    tmp.write_text(json.dumps(me))
    blocked_by: Path | None = None
    try:
        for attempt in range(_ATTEMPTS):
            if attempt:
                time.sleep(_RETRY_SEC)
            if _link(tmp, path, claim):
                return claim
            owner = _read_owner(path)
            if owner is None:
                if path.exists():
                    raise _refusal(directory, None, None)
                continue
            if _this_process(owner, me) and not owner.get("released"):
                os.replace(tmp, path)
                return claim
            alive = _alive(owner, me)
            if alive is not False:
                raise _refusal(directory, owner, alive)
            replaced, blocked_by = _take_over(path, owner, tmp, me)
            if replaced and _holds(path, claim):
                if not owner.get("released"):
                    logger.warning(
                        f"Took over checkpoint directory {directory} from {_describe(owner)}, "
                        "which has ended"
                    )
                return claim
    finally:
        tmp.unlink(missing_ok=True)
    if blocked_by is not None:
        raise CheckpointDirInUseError(
            f"Checkpoint directory {directory} is being taken over by "
            f"{_describe(_read_owner(blocked_by))}. If that run is not starting, remove "
            f"{blocked_by} and start this run again."
        )
    raise CheckpointDirInUseError(
        f"Checkpoint directory {directory} changed owner while this run claimed it; "
        "start the run again once the other run has its directory."
    )


def release_checkpoint_dir(directory: Path, claim: str) -> None:
    """Mark ``directory``'s owner record released when it is the record of ``claim``.

    The record is replaced, through the same token as a takeover (see ``_take_over``), by a
    released record with a claim id of its own; nothing removes ``.owner``, so a removal that
    NFS replays cannot remove a later run's record. An error marking it is logged rather than
    raised, so the rest of the run's teardown still runs.
    """
    with _LOCK:
        _release(directory, claim)


def _release(directory: Path, claim: str) -> None:
    """``release_checkpoint_dir``, with this process's lock held."""
    path = directory / OWNER_FILE
    owner = _read_owner(path)
    if owner is None or owner["claim"] != claim:
        return
    released = {**this_run(), "released": True}
    tmp = directory / f"{OWNER_FILE}.{released['claim']}.tmp"
    try:
        tmp.write_text(json.dumps(released))
        try:
            _take_over(path, owner, tmp, released)
        finally:
            tmp.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(
            f"Could not mark the checkpoint directory's owner record {path} released ({e})"
        )
