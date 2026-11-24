"""One training run per checkpoint directory, and the move of ``latest``."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

import kempnerforge.checkpoint.ownership as ownership
from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.checkpoint.ownership import (
    OWNER_FILE,
    CheckpointDirInUseError,
    claim_checkpoint_dir,
    release_checkpoint_dir,
    this_run,
)
from kempnerforge.config.schema import CheckpointConfig

_SLURM_VARS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_RESTART_COUNT",
    "SLURM_CLUSTER_NAME",
)


@pytest.fixture(autouse=True)
def _outside_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SLURM_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ownership, "_RETRY_SEC", 0.0)


@pytest.fixture
def warnings_logged() -> list[str]:
    """Warnings the ownership module logs, captured on its own logger."""
    messages: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = Collect(level=logging.WARNING)
    log = logging.getLogger(ownership.__name__)
    log.addHandler(handler)
    yield messages
    log.removeHandler(handler)


def _in_job(monkeypatch: pytest.MonkeyPatch, job: str = "123", restart: int | None = None) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "odyssey")
    if restart is not None:
        monkeypatch.setenv("SLURM_RESTART_COUNT", str(restart))


def _write_owner(directory: Path, **record: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OWNER_FILE
    defaults = {
        "claim": uuid.uuid4().hex,
        "slurm_cluster": None,
        "slurm_job": None,
        "slurm_restart": None,
        "host": "elsewhere",
        "pid": 1,
        "pid_ns": ownership._pid_namespace(),
        "since": "then",
    }
    path.write_text(json.dumps({**defaults, **record}))
    return path


def _in_this_job(**record: Any) -> dict[str, Any]:
    return {"slurm_cluster": "odyssey", "slurm_job": "123", "slurm_restart": 0, **record}


def _owner(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _squeue(monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str = "", stderr: str = ""):
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(ownership.subprocess, "run", run)
    return calls


def _query(job: str) -> list[str]:
    return ["squeue", "--noheader", "--states=all", f"--jobs={job}", "--format=%T"]


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.endswith((".tmp", ".takeover")))


def _exited_pid() -> int:
    child = subprocess.Popen(["true"])
    child.wait()
    return child.pid


def _in_forked_child(work: Callable[[], str], timeout: float = 60.0) -> tuple[int, str]:
    """Run ``work`` in a forked child; return its exit code (-1 if killed) and its text.

    The child exits whatever ``work`` does, and one still running after ``timeout``
    seconds, or when the parent is interrupted, is killed and reaped.
    """
    read, write = os.pipe()
    child = os.fork()
    if child == 0:
        code = 1
        try:
            os.write(write, work().encode())
            code = 0
        finally:
            os._exit(code)
    os.close(write)
    pipe = os.fdopen(read)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            done, status = os.waitpid(child, os.WNOHANG)
            if done:
                child = 0
                return os.waitstatus_to_exitcode(status), pipe.read()
            time.sleep(0.05)
        return -1, ""
    finally:
        if child:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
        pipe.close()


class TestThisRun:
    def test_a_run_outside_slurm(self) -> None:
        record = this_run()
        assert [record[k] for k in ("slurm_cluster", "slurm_job", "slurm_restart")] == [None] * 3
        assert (record["host"], record["pid"]) == (socket.gethostname(), os.getpid())
        assert record["claim"] != this_run()["claim"]

    def test_a_slurm_job_and_its_requeues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _in_job(monkeypatch)
        record = this_run()
        assert [record[k] for k in ("slurm_cluster", "slurm_job", "slurm_restart")] == [
            "odyssey",
            "123",
            0,
        ]
        monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
        assert this_run()["slurm_restart"] == 2

    def test_a_job_array_element_is_named_by_its_array_job_and_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _in_job(monkeypatch, job="1005")
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1000")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "5")
        assert this_run()["slurm_job"] == "1000_5"

    def test_the_record_names_the_pid_namespace(self) -> None:
        assert this_run()["pid_ns"] == os.stat("/proc/self/ns/pid").st_ino

    def test_without_proc_the_pid_namespace_is_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ownership, "_PID_NAMESPACE", str(tmp_path / "missing"))
        assert this_run()["pid_ns"] is None

    def test_a_process_keeps_its_id_and_a_forked_child_gets_its_own(self) -> None:
        mine = this_run()["process"]
        assert re.fullmatch(r"[0-9a-f]{32}", mine) and this_run()["process"] == mine
        code, child = _in_forked_child(lambda: this_run()["process"])
        assert code == 0 and child not in ("", mine)


class TestClaim:
    def test_a_free_directory_is_claimed(self, tmp_path: Path) -> None:
        directory = tmp_path / "ckpt"
        claim = claim_checkpoint_dir(directory)
        record = _owner(directory / OWNER_FILE)
        assert record["claim"] == claim
        assert (record["host"], record["pid"]) == (socket.gethostname(), os.getpid())
        assert record["slurm_job"] is None
        assert _leftovers(directory) == []

    def test_this_processs_earlier_claim_is_replaced(
        self, tmp_path: Path, warnings_logged: list[str]
    ) -> None:
        """A run that raised keeps its claim, and a later run in the same process replaces it."""
        first = claim_checkpoint_dir(tmp_path)
        second = claim_checkpoint_dir(tmp_path)
        assert second != first
        assert _owner(tmp_path / OWNER_FILE)["claim"] == second
        assert warnings_logged == []
        assert _leftovers(tmp_path) == []

    def test_a_requeued_job_takes_over_its_earlier_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warnings_logged: list[str]
    ) -> None:
        """The earlier attempt ran on another node with another pid, and SLURM ended it."""
        _in_job(monkeypatch, restart=1)
        calls = _squeue(monkeypatch, 0, "RUNNING\n")
        path = _write_owner(tmp_path, **_in_this_job())
        claim = claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == claim
        assert calls == []
        assert any("Took over" in m and "SLURM job 123" in m for m in warnings_logged)
        assert _leftovers(tmp_path) == []

    def test_another_live_run_in_this_job_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two trainings in one allocation share the job id, so their pids tell them apart."""
        _in_job(monkeypatch)
        calls = _squeue(monkeypatch, 0, "RUNNING\n")
        path = _write_owner(tmp_path, **_in_this_job(host=socket.gethostname(), pid=os.getppid()))
        before = path.read_text()
        with pytest.raises(CheckpointDirInUseError, match="still running"):
            claim_checkpoint_dir(tmp_path)
        assert path.read_text() == before
        assert calls == []
        assert _leftovers(tmp_path) == []

    def test_an_ended_run_in_this_job_is_taken_over(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _in_job(monkeypatch)
        path = _write_owner(tmp_path, **_in_this_job(host=socket.gethostname(), pid=_exited_pid()))
        claim = claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == claim

    def test_another_run_in_this_job_on_another_host_cannot_be_checked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _in_job(monkeypatch)
        _write_owner(tmp_path, **_in_this_job())
        with pytest.raises(CheckpointDirInUseError, match="cannot tell"):
            claim_checkpoint_dir(tmp_path)

    def test_a_job_of_another_cluster_cannot_be_checked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It may share this job's id, and this cluster's squeue does not list it."""
        _in_job(monkeypatch, restart=1)
        calls = _squeue(monkeypatch, 1, "", "slurm_load_jobs error: Invalid job id specified\n")
        _write_owner(tmp_path, **_in_this_job(slurm_cluster="other"))
        with pytest.raises(CheckpointDirInUseError, match="cannot tell"):
            claim_checkpoint_dir(tmp_path)
        assert calls == []

    @pytest.mark.parametrize("state", ["RUNNING", "PENDING", "COMPLETING", "SUSPENDED"])
    def test_a_live_slurm_job_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
    ) -> None:
        calls = _squeue(monkeypatch, 0, f"{state}\n")
        path = _write_owner(tmp_path, slurm_job="456", slurm_restart=0)
        before = path.read_text()
        with pytest.raises(CheckpointDirInUseError, match="SLURM job 456 on host elsewhere"):
            claim_checkpoint_dir(tmp_path)
        assert path.read_text() == before
        assert [command for command, _ in calls] == [_query("456")]
        assert _leftovers(tmp_path) == []

    def test_squeue_lists_every_state_whatever_the_environment_selects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SQUEUE_STATES=PENDING alone would hide a running owner from the listing."""
        narrowing = {
            "SQUEUE_STATES": "PENDING",
            "SQUEUE_USERS": "root",
            "SQUEUE_PARTITION": "test",
            "SLURM_CLUSTERS": "other",
        }
        for name, value in narrowing.items():
            monkeypatch.setenv(name, value)
        calls = _squeue(monkeypatch, 0, "RUNNING\n")
        _write_owner(tmp_path, slurm_job="456", slurm_restart=0)
        with pytest.raises(CheckpointDirInUseError, match="still running"):
            claim_checkpoint_dir(tmp_path)
        [(command, env)] = calls
        assert "--states=all" in command
        assert not set(narrowing) & set(env)
        assert env["PATH"] == os.environ["PATH"]

    def test_a_job_array_element_is_checked_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The array job's own id would list its other elements too."""
        calls = _squeue(monkeypatch, 0, "COMPLETED\n")
        _write_owner(tmp_path, slurm_job="1000_5", slurm_restart=0)
        claim_checkpoint_dir(tmp_path)
        assert [command for command, _ in calls] == [_query("1000_5")]

    def test_a_run_outside_slurm_asks_the_owners_cluster_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This host's squeue may answer for another cluster than the owner's."""
        calls = _squeue(monkeypatch, 0, "RUNNING\n")
        _write_owner(tmp_path, slurm_cluster="other", slurm_job="456", slurm_restart=0)
        with pytest.raises(CheckpointDirInUseError, match="still running"):
            claim_checkpoint_dir(tmp_path)
        assert [command for command, _ in calls] == [[*_query("456"), "--clusters=other"]]

    @pytest.mark.parametrize(
        ("returncode", "stdout", "stderr"),
        [
            (1, "", "slurm_load_jobs error: Invalid job id specified\n"),
            (0, "", ""),
            (0, "COMPLETED\n", ""),
            (0, "CANCELLED\n", ""),
            (0, "TIMEOUT\n", ""),
        ],
        ids=["invalid-job-id", "not-listed", "completed", "cancelled", "timeout"],
    )
    def test_a_slurm_job_that_has_ended_is_taken_over(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        warnings_logged: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        _squeue(monkeypatch, returncode, stdout, stderr)
        path = _write_owner(tmp_path, slurm_job="456", slurm_restart=0)
        claim = claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == claim
        assert any("Took over" in m and "SLURM job 456" in m for m in warnings_logged)
        assert _leftovers(tmp_path) == []

    def test_a_scheduler_that_cannot_answer_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _squeue(monkeypatch, 1, "", "slurm_load_jobs error: Unable to contact slurm controller\n")
        path = _write_owner(tmp_path, slurm_job="456", slurm_restart=0)
        with pytest.raises(CheckpointDirInUseError, match="cannot tell") as refused:
            claim_checkpoint_dir(tmp_path)
        assert str(path) in str(refused.value)

    def test_a_host_without_squeue_cannot_tell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(command, **kwargs):
            raise FileNotFoundError("squeue")

        monkeypatch.setattr(ownership.subprocess, "run", missing)
        _write_owner(tmp_path, slurm_job="456", slurm_restart=0)
        with pytest.raises(CheckpointDirInUseError, match="cannot tell"):
            claim_checkpoint_dir(tmp_path)

    def test_a_run_outside_slurm_is_alive_while_its_pid_runs(self, tmp_path: Path) -> None:
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=os.getppid())
        refusal = rf"still running \(its record is {re.escape(str(path))}\)"
        with pytest.raises(CheckpointDirInUseError, match=refusal):
            claim_checkpoint_dir(tmp_path)

    @pytest.mark.parametrize("in_job", [False, True])
    def test_a_pid_of_another_pid_namespace_is_neither_this_processs_nor_checked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_job: bool
    ) -> None:
        """Containers that share the host name but not the pid namespace reuse pids, so a
        record from another namespace is not replaced as this process's own, and its pid,
        this process's or a running one, says nothing."""
        record: dict[str, Any] = {"host": socket.gethostname(), "pid_ns": -1}
        if in_job:
            _in_job(monkeypatch)
            record = _in_this_job(**record)
        for pid in (os.getpid(), os.getppid()):
            path = _write_owner(tmp_path / str(pid), pid=pid, **record)
            before = path.read_text()
            refusal = f"cannot tell whether it is still running; remove {re.escape(str(path))}"
            with pytest.raises(CheckpointDirInUseError, match=refusal):
                claim_checkpoint_dir(tmp_path / str(pid))
            assert path.read_text() == before

    def test_without_proc_a_pid_cannot_be_checked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Where /proc does not show the pid namespace, a pid could be another container's,
        even this process's own pid, so neither record's run can be checked by it."""
        monkeypatch.setattr(ownership, "_PID_NAMESPACE", str(tmp_path / "missing"))
        for pid in (os.getpid(), os.getppid()):
            directory = tmp_path / str(pid)
            path = _write_owner(directory, host=socket.gethostname(), pid=pid, pid_ns=None)
            before = path.read_text()
            with pytest.raises(CheckpointDirInUseError, match="cannot tell"):
                claim_checkpoint_dir(directory)
            assert path.read_text() == before

    def test_this_processs_own_record_is_known_by_its_process_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without /proc too, this process replaces its own earlier claim, which holds its
        process id, and not a record of its host and pid that another process wrote."""
        monkeypatch.setattr(ownership, "_PID_NAMESPACE", str(tmp_path / "missing"))
        first = claim_checkpoint_dir(tmp_path / "own")
        second = claim_checkpoint_dir(tmp_path / "own")
        assert second != first and _owner(tmp_path / "own" / OWNER_FILE)["claim"] == second
        other = {**this_run(), "process": uuid.uuid4().hex}
        path = _write_owner(tmp_path / "other", **other)
        with pytest.raises(CheckpointDirInUseError, match="cannot tell"):
            claim_checkpoint_dir(tmp_path / "other")
        assert _owner(path)["claim"] == other["claim"]

    def test_a_run_outside_slurm_whose_pid_exited_is_taken_over(
        self, tmp_path: Path, warnings_logged: list[str]
    ) -> None:
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=_exited_pid())
        claim = claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == claim
        assert any("Took over" in m for m in warnings_logged)

    def test_a_run_outside_slurm_on_another_host_cannot_be_checked(self, tmp_path: Path) -> None:
        _write_owner(tmp_path, host="elsewhere", pid=os.getpid())
        with pytest.raises(CheckpointDirInUseError, match="a run on host elsewhere"):
            claim_checkpoint_dir(tmp_path)

    def test_a_job_does_not_claim_a_directory_a_run_outside_slurm_holds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _in_job(monkeypatch)
        _write_owner(tmp_path, host=socket.gethostname(), pid=os.getppid())
        with pytest.raises(CheckpointDirInUseError, match="still running"):
            claim_checkpoint_dir(tmp_path)

    @pytest.mark.parametrize("pid", [0, -5, True, 2**128, "12", None])
    def test_a_pid_that_names_no_process_cannot_be_checked(self, tmp_path: Path, pid) -> None:
        _write_owner(tmp_path, host=socket.gethostname(), pid=pid)
        with pytest.raises(CheckpointDirInUseError, match="cannot tell"):
            claim_checkpoint_dir(tmp_path)

    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            "[]",
            json.dumps({"host": "elsewhere", "pid": 1}),
            json.dumps({"claim": "../../escape", "host": "elsewhere", "pid": 1}),
        ],
        ids=["not-json", "not-a-record", "no-claim", "claim-not-an-id"],
    )
    def test_an_unreadable_record_is_refused(self, tmp_path: Path, text: str) -> None:
        path = tmp_path / OWNER_FILE
        path.write_text(text)
        with pytest.raises(CheckpointDirInUseError, match="cannot be read") as refused:
            claim_checkpoint_dir(tmp_path)
        assert str(path) in str(refused.value)
        assert path.read_text() == text
        assert _leftovers(tmp_path) == []

    def test_a_claim_another_run_makes_first_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only one of two claims at the same moment creates the record."""
        _squeue(monkeypatch, 0, "RUNNING\n")
        link = os.link

        def other_first(source, target):
            _write_owner(tmp_path, slurm_job="789", slurm_restart=0)
            monkeypatch.setattr(ownership.os, "link", link)
            return link(source, target)

        monkeypatch.setattr(ownership.os, "link", other_first)
        with pytest.raises(CheckpointDirInUseError, match="SLURM job 789"):
            claim_checkpoint_dir(tmp_path)
        assert _leftovers(tmp_path) == []

    def test_a_record_removed_while_claiming_is_claimed_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_owner(tmp_path, slurm_job="456", slurm_restart=0)
        read = ownership._read_owner

        def released_meanwhile(owner_path):
            owner_path.unlink()
            monkeypatch.setattr(ownership, "_read_owner", read)
            return None

        monkeypatch.setattr(ownership, "_read_owner", released_meanwhile)
        claim = claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == claim

    @pytest.mark.parametrize("target", ["record", "takeover", "private"])
    def test_a_link_that_went_through_counts_when_nfs_reports_it_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
    ) -> None:
        """NFS can answer a retried link that went through with EEXIST."""
        if target != "record":
            _write_owner(tmp_path, host=socket.gethostname(), pid=_exited_pid())
        link = os.link
        names = {"takeover": r".*\.takeover", "private": r".*\.[0-9a-f]{32}\.\d+\.tmp"}

        def through_then_existing(source, dest):
            link(source, dest)
            if target == "record" or re.fullmatch(names[target], Path(dest).name):
                raise FileExistsError(dest)

        monkeypatch.setattr(ownership.os, "link", through_then_existing)
        claim = claim_checkpoint_dir(tmp_path)
        assert _owner(tmp_path / OWNER_FILE)["claim"] == claim
        assert _leftovers(tmp_path) == []

    def test_two_claimers_never_both_take_over_one_ended_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warnings_logged: list[str]
    ) -> None:
        """Another process takes the record over between this claimer's check and its token.

        That process has a lock of its own, so it runs the claim without this one's.
        """
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=_exited_pid())
        ended = _owner(path)["claim"]
        other = {**this_run(), "pid": os.getppid()}
        alive = ownership._alive
        raced: list[str] = []

        def check(owner, me):
            if owner["claim"] == ended and me is not other and not raced:
                with monkeypatch.context() as patch:
                    patch.setattr(ownership, "this_run", lambda: other)
                    raced.append(ownership._claim(tmp_path))
            return alive(owner, me)

        monkeypatch.setattr(ownership, "_alive", check)
        with pytest.raises(CheckpointDirInUseError, match="still running"):
            claim_checkpoint_dir(tmp_path)
        assert raced == [other["claim"]]
        assert _owner(path)["claim"] == other["claim"]
        assert sum("Took over" in m for m in warnings_logged) == 1
        assert _leftovers(tmp_path) == []

    def test_a_claimer_waits_for_a_takeover_in_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another claimer holds the token, and then renames it into place."""
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=_exited_pid())
        other = {**this_run(), "pid": os.getppid()}
        token = tmp_path / f"{OWNER_FILE}.{_owner(path)['claim']}.takeover"
        token.write_text(json.dumps(other))
        waits: list[float] = []

        def other_finishes(seconds: float) -> None:
            waits.append(seconds)
            os.rename(token, path)

        monkeypatch.setattr(ownership.time, "sleep", other_finishes)
        with pytest.raises(CheckpointDirInUseError, match="still running"):
            claim_checkpoint_dir(tmp_path)
        assert len(waits) == 1
        assert _owner(path)["claim"] == other["claim"]
        assert _leftovers(tmp_path) == []

    def test_a_takeover_that_never_finishes_names_its_token(self, tmp_path: Path) -> None:
        """A live claimer holds the token on every attempt, and the error says to remove it."""
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=_exited_pid())
        token = tmp_path / f"{OWNER_FILE}.{_owner(path)['claim']}.takeover"
        token.write_text(json.dumps({**this_run(), "pid": os.getppid()}))
        with pytest.raises(
            CheckpointDirInUseError, match="is being taken over by a run"
        ) as refused:
            claim_checkpoint_dir(tmp_path)
        assert str(token) in str(refused.value)
        token.unlink()
        claim = claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == claim
        assert _leftovers(tmp_path) == []

    @pytest.mark.parametrize("depth", [1, 2])
    def test_tokens_left_by_claimers_that_died_are_taken_over(
        self, tmp_path: Path, warnings_logged: list[str], depth: int
    ) -> None:
        """A claimer died holding the token, and at depth 2 so did one taking that token over."""
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=_exited_pid())
        held = _owner(path)
        for _ in range(depth):
            holder = {**this_run(), "pid": _exited_pid()}
            (tmp_path / f"{OWNER_FILE}.{held['claim']}.takeover").write_text(json.dumps(holder))
            held = holder
        claim = claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == claim
        assert sum("Took over" in m for m in warnings_logged) == 1
        assert _leftovers(tmp_path) == []

    def test_a_long_chain_of_tokens_left_by_claimers_that_died_is_taken_over(
        self, tmp_path: Path
    ) -> None:
        """More levels than Python's recursion limit allows."""
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=_exited_pid())
        held, dead = _owner(path), _exited_pid()
        for _ in range(sys.getrecursionlimit() + 100):
            holder = {**this_run(), "pid": dead}
            (tmp_path / f"{OWNER_FILE}.{held['claim']}.takeover").write_text(json.dumps(holder))
            held = holder
        claim = claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == claim
        assert _leftovers(tmp_path) == []

    def test_a_cycle_of_tokens_refuses_instead_of_looping(self, tmp_path: Path) -> None:
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=_exited_pid())
        record = _owner(path)
        token = tmp_path / f"{OWNER_FILE}.{record['claim']}.takeover"
        token.write_text(json.dumps(record))
        with pytest.raises(CheckpointDirInUseError, match="is being taken over") as refused:
            claim_checkpoint_dir(tmp_path)
        assert str(token) in str(refused.value)

    def test_a_replayed_rename_finds_nothing_to_move(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every rename moves a name of this claim's own, which no other claimer recreates."""
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=_exited_pid())
        replaced = _owner(path)["claim"]
        rename, renames = os.rename, []

        def recorded(source, target):
            renames.append((Path(source), Path(target)))
            rename(source, target)

        monkeypatch.setattr(ownership.os, "rename", recorded)
        claim = claim_checkpoint_dir(tmp_path)
        assert [source.name for source, _ in renames] == [f"{OWNER_FILE}.{claim}.0.tmp"]
        stale = {**this_run(), "pid": os.getppid()}
        (tmp_path / f"{OWNER_FILE}.{replaced}.takeover").write_text(json.dumps(stale))
        for source, target in renames:
            with pytest.raises(FileNotFoundError):
                rename(source, target)
        assert _owner(path)["claim"] == claim

    def test_this_processs_released_record_goes_through_the_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another process holds the token for it, so this one waits for it and then refuses."""
        path = tmp_path / OWNER_FILE
        release_checkpoint_dir(tmp_path, claim_checkpoint_dir(tmp_path))
        other = {**this_run(), "pid": os.getppid()}
        token = tmp_path / f"{OWNER_FILE}.{_owner(path)['claim']}.takeover"
        token.write_text(json.dumps(other))

        def other_finishes(seconds: float) -> None:
            path.write_text(json.dumps(other))
            token.unlink()

        monkeypatch.setattr(ownership.time, "sleep", other_finishes)
        with pytest.raises(CheckpointDirInUseError, match="still running"):
            claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == other["claim"]

    def test_a_record_replaced_right_after_the_rename_is_not_counted_as_this_claims(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The claim reads the record back before it counts it."""
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=_exited_pid())
        other = {**this_run(), "pid": os.getppid()}
        rename = os.rename

        def replayed(source, target):
            rename(source, target)
            Path(source).write_text(json.dumps(other))
            rename(source, target)

        monkeypatch.setattr(ownership.os, "rename", replayed)
        with pytest.raises(CheckpointDirInUseError, match="still running"):
            claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == other["claim"]
        assert _leftovers(tmp_path) == []

    def test_a_takeover_whose_rename_fails_keeps_its_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The record it guards is still in place; this process's next claim takes it over."""
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=_exited_pid())
        before = path.read_text()
        token = f"{OWNER_FILE}.{_owner(path)['claim']}.takeover"
        rename = os.rename

        def refuse(source, target):
            raise OSError("rename refused")

        monkeypatch.setattr(ownership.os, "rename", refuse)
        with pytest.raises(OSError, match="rename refused"):
            claim_checkpoint_dir(tmp_path)
        assert path.read_text() == before
        assert _leftovers(tmp_path) == [token]
        monkeypatch.setattr(ownership.os, "rename", rename)
        claim = claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == claim
        assert _leftovers(tmp_path) == []


_CLAIMER = """
import importlib.util, json, os, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location("ownership_under_test", sys.argv[1])
ownership = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ownership)
ownership._RETRY_SEC = 0.05
directory, go, done, out = (Path(arg) for arg in sys.argv[2:6])
while not go.exists():
    time.sleep(0.001)
try:
    result = {"claim": ownership.claim_checkpoint_dir(directory)}
except ownership.CheckpointDirInUseError as e:
    result = {"refused": str(e)}
out.with_suffix(".part").write_text(json.dumps(result))
os.replace(out.with_suffix(".part"), out)
while not done.exists():
    time.sleep(0.01)
"""


class TestOneProcess:
    def test_a_release_and_a_claim_in_two_threads_run_one_at_a_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The release has read its own record when the claim starts, and removes only that."""
        first = claim_checkpoint_dir(tmp_path)
        read, release_read, go_on = ownership._read_owner, threading.Event(), threading.Event()

        def pause_the_release(path):
            record = read(path)
            if threading.current_thread().name == "release":
                release_read.set()
                assert go_on.wait(10)
            return record

        monkeypatch.setattr(ownership, "_read_owner", pause_the_release)
        release = threading.Thread(
            target=release_checkpoint_dir, args=(tmp_path, first), name="release"
        )
        claims: list[str] = []
        claim = threading.Thread(target=lambda: claims.append(claim_checkpoint_dir(tmp_path)))
        release.start()
        assert release_read.wait(10)
        claim.start()
        claim.join(0.5)
        assert claim.is_alive()
        go_on.set()
        release.join(10)
        claim.join(10)
        assert _owner(tmp_path / OWNER_FILE)["claim"] == claims[0] != first

    def test_a_child_forked_while_a_thread_claims_gets_a_free_lock(self, tmp_path: Path) -> None:
        """The parent's thread that holds the lock when the process forks does not run in
        the child, so the child claims with a lock of its own."""
        held, done = threading.Event(), threading.Event()

        def hold_the_lock() -> None:
            with ownership._LOCK:
                held.set()
                done.wait(60)

        holder = threading.Thread(target=hold_the_lock)
        holder.start()
        try:
            assert held.wait(10)
            directory = tmp_path / "child"
            code, claim = _in_forked_child(lambda: claim_checkpoint_dir(directory), timeout=30)
        finally:
            done.set()
            holder.join(10)
        assert code == 0 and _owner(directory / OWNER_FILE)["claim"] == claim


class TestClaimersInProcesses:
    def test_one_of_several_processes_takes_over_an_ended_record(self, tmp_path: Path) -> None:
        """Every claimer stays alive until all have claimed, so exactly one owns the directory."""
        for trial in range(3):
            directory = tmp_path / f"ckpt{trial}"
            path = _write_owner(directory, host=socket.gethostname(), pid=_exited_pid())
            go, done = tmp_path / f"go{trial}", tmp_path / f"done{trial}"
            outs = [tmp_path / f"claimer{trial}_{i}.json" for i in range(6)]
            args = [ownership.__file__, str(directory), str(go), str(done)]
            claimers = [
                subprocess.Popen([sys.executable, "-c", _CLAIMER, *args, str(out)]) for out in outs
            ]
            try:
                go.touch()
                deadline = time.monotonic() + 120
                while not all(out.exists() for out in outs) and time.monotonic() < deadline:
                    time.sleep(0.05)
                results = [json.loads(out.read_text()) for out in outs]
            finally:
                done.touch()
                for claimer in claimers:
                    claimer.wait(timeout=60)
            winners = [result["claim"] for result in results if "claim" in result]
            assert len(winners) == 1, results
            assert _owner(path)["claim"] == winners[0]
            assert _leftovers(directory) == []


class TestRelease:
    def test_a_run_marks_its_own_record_released(self, tmp_path: Path) -> None:
        claim = claim_checkpoint_dir(tmp_path)
        release_checkpoint_dir(tmp_path, claim)
        record = _owner(tmp_path / OWNER_FILE)
        assert record["released"] is True
        assert record["claim"] != claim
        assert (record["host"], record["pid"]) == (socket.gethostname(), os.getpid())
        assert _leftovers(tmp_path) == []

    def test_a_release_removes_no_owner_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A removal NFS replays after another run claimed would remove that run's record."""
        claim = claim_checkpoint_dir(tmp_path)
        unlink, remove = Path.unlink, os.remove

        def guarded_unlink(self, missing_ok=False):
            assert self.name != OWNER_FILE, "release removed the owner record"
            return unlink(self, missing_ok=missing_ok)

        def guarded_remove(path, *args, **kwargs):
            assert Path(path).name != OWNER_FILE, "release removed the owner record"
            return remove(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", guarded_unlink)
        monkeypatch.setattr(ownership.os, "remove", guarded_remove)
        monkeypatch.setattr(ownership.os, "unlink", guarded_remove)
        release_checkpoint_dir(tmp_path, claim)
        assert _owner(tmp_path / OWNER_FILE)["released"] is True

    def test_another_run_takes_over_a_released_record_without_a_warning(
        self, tmp_path: Path, warnings_logged: list[str]
    ) -> None:
        """The record's run is still alive, but it released the directory."""
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=os.getppid(), released=True)
        claim = claim_checkpoint_dir(tmp_path)
        assert _owner(path)["claim"] == claim
        assert warnings_logged == []
        assert _leftovers(tmp_path) == []

    def test_another_claims_record_stays_even_in_this_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Release goes by the claim id, not the job, host or pid another claim may share."""
        _in_job(monkeypatch)
        path = _write_owner(tmp_path, **_in_this_job(host=socket.gethostname(), pid=os.getpid()))
        before = path.read_text()
        release_checkpoint_dir(tmp_path, uuid.uuid4().hex)
        assert path.read_text() == before

    def test_a_directory_without_a_record_is_left_alone(self, tmp_path: Path) -> None:
        release_checkpoint_dir(tmp_path, uuid.uuid4().hex)
        assert list(tmp_path.iterdir()) == []

    def test_an_error_marking_the_record_is_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warnings_logged: list[str]
    ) -> None:
        claim = claim_checkpoint_dir(tmp_path)

        def refuse(source, target):
            raise PermissionError("read-only file system")

        monkeypatch.setattr(ownership.os, "rename", refuse)
        release_checkpoint_dir(tmp_path, claim)
        assert _owner(tmp_path / OWNER_FILE)["claim"] == claim
        assert any("read-only file system" in m for m in warnings_logged)
        assert _leftovers(tmp_path) == [f"{OWNER_FILE}.{claim}.takeover"]


def _manager(directory: Path) -> CheckpointManager:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return CheckpointManager(CheckpointConfig(dir=str(directory)), model, optimizer)


class TestManager:
    def test_rank_zero_claims_and_releases(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path / "ckpt")
        manager.claim_dir()
        path = tmp_path / "ckpt" / OWNER_FILE
        assert _owner(path)["pid"] == os.getpid()
        manager.release_dir()
        assert _owner(path)["released"] is True

    def test_release_leaves_a_record_written_since_the_claim(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        manager.claim_dir()
        path = _write_owner(tmp_path, host=socket.gethostname(), pid=os.getpid())
        before = path.read_text()
        manager.release_dir()
        assert path.read_text() == before

    def test_a_refused_claim_raises(self, tmp_path: Path) -> None:
        _write_owner(tmp_path, host=socket.gethostname(), pid=os.getppid())
        with pytest.raises(CheckpointDirInUseError):
            _manager(tmp_path).claim_dir()

    def test_another_rank_only_agrees(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "ckpt")
        manager._rank = 1
        agreed: list[tuple[BaseException | None, str]] = []
        monkeypatch.setattr(manager, "agree", lambda error, what: agreed.append((error, what)))
        manager.claim_dir()
        manager.release_dir()
        assert agreed == [(None, "Checkpoint directory claim")]
        assert not (tmp_path / "ckpt").exists()

    def test_rank_zero_reports_its_refusal_to_the_other_ranks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_owner(tmp_path, host=socket.gethostname(), pid=os.getppid())
        manager = _manager(tmp_path)
        agreed: list[tuple[BaseException | None, str]] = []
        monkeypatch.setattr(manager, "agree", lambda error, what: agreed.append((error, what)))
        manager.claim_dir()
        [(error, what)] = agreed
        assert isinstance(error, CheckpointDirInUseError)
        assert what == "Checkpoint directory claim"


class TestLatestLink:
    def test_two_writers_moving_latest_at_once_do_not_collide(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another writer's move lands between this one's link and rename."""
        first, second = _manager(tmp_path), _manager(tmp_path)
        for manager in (first, second):
            monkeypatch.setattr(manager, "_cleanup", lambda: None)
        for step in (1, 2):
            (tmp_path / f"step_{step}").mkdir()
        rename = Path.rename
        interleaved: list[str] = []

        def other_writer_first(self, target):
            if not interleaved:
                interleaved.append(self.name)
                second._commit_latest(2, tmp_path / "step_2")
            return rename(self, target)

        monkeypatch.setattr(Path, "rename", other_writer_first)
        first._commit_latest(1, tmp_path / "step_1")
        assert interleaved and interleaved[0].startswith("latest.")
        assert os.readlink(tmp_path / "latest") == "step_1"
        assert _leftovers(tmp_path) == []

    def test_a_failed_rename_leaves_no_temp_link(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path)
        (tmp_path / "step_1").mkdir()
        (tmp_path / "latest").symlink_to("step_1")
        (tmp_path / "step_2").mkdir()

        def refuse(self, target):
            raise OSError("rename refused")

        monkeypatch.setattr(Path, "rename", refuse)
        with pytest.raises(OSError, match="rename refused"):
            manager._commit_latest(2, tmp_path / "step_2")
        assert os.readlink(tmp_path / "latest") == "step_1"
        assert _leftovers(tmp_path) == []
