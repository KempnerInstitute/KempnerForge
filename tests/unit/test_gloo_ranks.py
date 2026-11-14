"""The spawned-rank helpers in ``tests/gloo_ranks.py``."""

from __future__ import annotations

import errno
import signal
import socket
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tests import gloo_ranks
from tests.gloo_ranks import (
    HOST,
    gloo_group,
    held_store,
    mark,
    run_to_outcomes,
    spawn,
    write_outcome,
)


class _Interrupt(BaseException):
    """Stands in for a test timeout that interrupts the wait for the ranks."""


def _reduce(rank: int, world: int, port: int, joins: int) -> None:
    for _ in range(joins):
        with gloo_group(rank, world, port):
            total = torch.tensor([rank + 1.0])
            dist.all_reduce(total)
            assert total.item() == world * (world + 1) / 2


def _late_rank(rank: int, world: int, port: int, out_dir: str) -> None:
    leave = dist.destroy_process_group

    def recording_leave(*args, **kwargs):
        Path(out_dir, f"left{rank}").write_text(repr(time.time()))
        return leave(*args, **kwargs)

    dist.destroy_process_group = recording_leave
    with gloo_group(rank, world, port):
        if rank == 1:
            time.sleep(2.0)
        Path(out_dir, f"done{rank}").write_text(repr(time.time()))


def _reducing_in_a_thread(rank: int, world: int, port: int) -> None:
    """Wait to leave while this rank's own thread still reduces on the group.

    Rank 0's thread starts its reduce before rank 0 waits to leave, and rank 1's starts a second
    after rank 1 began to wait, so a collective wait would meet the reduces in a different order
    on each rank. The real teardown would end a reduce still in flight, so each rank joins its
    thread before the group is destroyed.
    """
    started = threading.Event()
    totals: list[float] = []

    def reduce() -> None:
        if rank == 1:
            time.sleep(1.0)
        total = torch.tensor([1.0])
        work = dist.all_reduce(total, async_op=True)
        started.set()
        work.wait()
        totals.append(total.item())

    thread = threading.Thread(target=reduce)
    leave = dist.destroy_process_group

    def leave_once_the_thread_ends(*args, **kwargs):
        thread.join()
        return leave(*args, **kwargs)

    dist.destroy_process_group = leave_once_the_thread_ends
    with gloo_group(rank, world, port, timeout=timedelta(seconds=30)):
        thread.start()
        if rank == 0:
            started.wait()
    assert totals == [float(world)]


def _wait_until_finished(port: int, rank: int) -> None:
    """Wait until the held store at ``port`` records that ``rank`` finished its block.

    Gives up after ``LEAVE_TIMEOUT``, as a rank waiting to leave does.
    """
    client = dist.TCPStore(HOST, port, is_master=False, timeout=gloo_ranks.STORE_TIMEOUT)
    join = client.get(f"joined/{rank}").decode()
    client.wait([f"{join}/finished/{rank}"], gloo_ranks.LEAVE_TIMEOUT)


def _raise_on_rank_1(rank: int, world: int, port: int) -> None:
    """Rank 1 raises once rank 0 has finished its block and waits to leave.

    Rank 0 records its block finished only after its group is set up, so rank 1's teardown
    cannot fail rank 0's setup, and rank 1 is the only rank the note can name.
    """
    with gloo_group(rank, world, port):
        if rank == 1:
            _wait_until_finished(port, 0)
            raise ValueError("rank 1 failed")


def _slow_block_on_rank_1(rank: int, world: int, port: int) -> None:
    """Rank 1 stays in its block well past the 3 s rank 0 waits to leave."""
    with gloo_group(rank, world, port, timeout=timedelta(seconds=3)):
        if rank == 1:
            time.sleep(60)


def _skip_the_reduce_on_rank_0(rank: int, world: int, port: int) -> None:
    """Rank 1 reduces, while rank 0 returns without reducing and waits 3 s to leave."""
    with gloo_group(rank, world, port, timeout=timedelta(seconds=3) if rank == 0 else None):
        if rank == 1:
            dist.all_reduce(torch.ones(1))


def _timed(rank: int, world: int, port: int, plan: dict, out_dir: str) -> None:
    """Sleep before the timed part, inside it and after it, as ``plan`` gives for ``rank``."""
    before, inside, after = plan.get(rank, (0.0, 0.0, 0.0))
    time.sleep(before)
    mark(out_dir, rank)
    time.sleep(inside)
    write_outcome(out_dir, rank, "returned")
    time.sleep(after)


def _stuck(rank: int, world: int, port: int, out_dir: str) -> None:
    Path(out_dir, f"started{rank}").touch()
    time.sleep(600)


def _stuck_ignoring_terminate(rank: int, world: int, port: int, out_dir: str) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _stuck(rank, world, port, out_dir)


def _interrupt_a_timed_join(join):
    """Wrap a process's ``join`` so that the grace period's join, the one with a timeout, raises."""

    def join_unless_timed(timeout=None):
        if timeout is not None:
            raise _Interrupt
        return join()

    return join_unless_timed


def _interrupt_once_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during_grace: bool = False
) -> list:
    """Make the next spawn's wait raise ``_Interrupt`` once both ranks have started.

    With ``during_grace``, the stop's wait for the grace period is interrupted as well.
    Returns the list the spawned processes are added to.
    """
    processes: list = []
    start_ranks = gloo_ranks.mp.spawn

    def spawn_then_interrupt(*args, **kwargs):
        context = start_ranks(*args, **kwargs)
        processes.extend(context.processes)

        def interrupted_join(timeout=None):
            deadline = time.monotonic() + 120
            while len(list(tmp_path.rglob("started*"))) < 2 and time.monotonic() < deadline:
                time.sleep(0.1)
            raise _Interrupt

        context.join = interrupted_join
        if during_grace:
            for process in context.processes:
                process.join = _interrupt_a_timed_join(process.join)
        return context

    monkeypatch.setattr(gloo_ranks.mp, "spawn", spawn_then_interrupt)
    return processes


class TestHeldStore:
    def test_no_other_socket_can_take_the_port_while_it_is_held(self):
        with held_store() as port:
            squatter = socket.socket()
            with pytest.raises(OSError) as info:
                squatter.bind((HOST, port))
            squatter.close()
        assert info.value.errno == errno.EADDRINUSE


class TestSpawn:
    @pytest.mark.parametrize(
        ("fn", "signum", "during_grace"),
        [
            (_stuck, signal.SIGTERM, False),
            (_stuck_ignoring_terminate, signal.SIGKILL, False),
            (_stuck_ignoring_terminate, signal.SIGKILL, True),
        ],
        ids=["terminated", "killed", "killed-after-a-second-interrupt"],
    )
    def test_an_interrupted_wait_stops_every_rank(
        self, tmp_path, monkeypatch, fn, signum, during_grace
    ):
        """A timeout that interrupts the wait leaves no rank running after the test.

        A rank that ignores the terminate signal is killed once the grace period ends, or at
        once when a second interrupt cuts the grace period short.
        """
        processes = _interrupt_once_started(tmp_path, monkeypatch, during_grace)
        with pytest.raises(_Interrupt):
            spawn(fn, 2, str(tmp_path))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["started0", "started1"]
        assert [p.exitcode for p in processes] == [-signum, -signum]

    def test_a_rank_that_raises_fails_the_test(self):
        """Rank 1's error reaches the test, and rank 0, waiting to leave, is stopped."""
        with pytest.raises(gloo_ranks.mp.ProcessRaisedException, match="rank 1 failed") as info:
            spawn(_raise_on_rank_1, 2)
        assert info.value.__notes__ == ["ranks [1] had not finished their gloo_group block"]


class TestGlooGroup:
    @pytest.mark.parametrize("joins", [1, 2])
    def test_ranks_join_through_the_held_store_and_reduce(self, joins):
        spawn(_reduce, 2, joins)

    def test_no_rank_leaves_before_every_rank_finished(self, tmp_path):
        """Rank 0 finishes first and still leaves only after rank 1's block ended."""
        spawn(_late_rank, 2, str(tmp_path))
        done_1 = float((tmp_path / "done1").read_text())
        left_0 = float((tmp_path / "left0").read_text())
        assert left_0 >= done_1

    def test_the_wait_to_leave_runs_no_collective(self):
        """A reduce still in flight on a rank's own thread pairs with its peer's, not with the
        wait to leave. Leaving would end that reduce, so a block must finish its background
        collectives before it returns; here each rank joins its thread before the teardown."""
        spawn(_reducing_in_a_thread, 2)

    def test_a_rank_that_waits_too_long_to_leave_names_the_ranks_inside(self):
        """Rank 1 stays in its block, and rank 0 gives up after the group's 3 s timeout."""
        message = r"rank 0 waited 3s to leave the group, but ranks \[1\] had not finished"
        with pytest.raises(gloo_ranks.mp.ProcessRaisedException, match=message):
            spawn(_slow_block_on_rank_1, 2)

    def test_a_collective_one_rank_skips_fails_fast_naming_the_rank(self):
        """Rank 0 skips a reduce that rank 1 runs: the test fails within seconds, not after the
        store's timeout, and names rank 1, whichever rank's error comes first."""
        start = time.monotonic()
        with pytest.raises(
            gloo_ranks.mp.ProcessRaisedException, match=r"ranks \[1\] had not finished their"
        ):
            spawn(_skip_the_reduce_on_rank_0, 2)
        assert time.monotonic() - start < gloo_ranks.LEAVE_TIMEOUT.total_seconds()


class TestRunToOutcomes:
    def test_start_up_does_not_count_against_the_part(self, tmp_path):
        """A rank slow to reach the part but quick inside it passes a deadline it would miss
        if the deadline started at the spawn."""
        plan = {1: (6.0, 1.0, 0.0)}
        outcomes = run_to_outcomes(_timed, 2, (plan,), tmp_path, part="save()", part_s=4.0)
        assert outcomes == {0: "returned", 1: "returned"}

    @pytest.mark.parametrize(
        ("plan", "limits", "message"),
        [
            (
                {1: (0.0, 30.0, 0.0)},
                {},
                r"ranks \[1\] still in save\(\) 3s after every rank entered it",
            ),
            (
                {0: (0.0, 0.0, 30.0)},
                {"exit_s": 10.0},
                r"ranks \[0\] wrote their outcome but did not exit within 10s",
            ),
        ],
        ids=["part", "exit"],
    )
    def test_each_phase_fails_with_its_own_message(self, tmp_path, plan, limits, message):
        with pytest.raises(pytest.fail.Exception, match=message):
            run_to_outcomes(_timed, 2, (plan,), tmp_path, part="save()", part_s=3.0, **limits)

    def test_ranks_that_never_start_fail_as_not_reaching_the_part(self, tmp_path, monkeypatch):
        """No rank ever marks: the start limit fails naming every rank, and stops them."""
        terminated: list[int] = []

        class Stuck:
            def __init__(self, rank: int) -> None:
                self.rank, self.alive = rank, True

            def is_alive(self) -> bool:
                return self.alive

            def terminate(self) -> None:
                terminated.append(self.rank)
                self.alive = False

            def join(self, timeout=None) -> None:
                return None

        class Context:
            processes = [Stuck(0), Stuck(1)]

            def join(self, timeout=None) -> bool:
                time.sleep(timeout or 0)
                return False

        monkeypatch.setattr(gloo_ranks.mp, "spawn", lambda *args, **kwargs: Context())
        late = r"ranks \[0, 1\] did not reach save\(\) within 1s of the spawn"
        with pytest.raises(pytest.fail.Exception, match=late):
            run_to_outcomes(_timed, 2, ({},), tmp_path, part="save()", part_s=1.0, start_s=1.0)
        assert terminated == [0, 1]

    def test_an_interrupted_wait_stops_every_rank(self, tmp_path, monkeypatch):
        """A timeout that interrupts the wait leaves no rank running after the test."""
        processes = _interrupt_once_started(tmp_path, monkeypatch)
        with pytest.raises(_Interrupt):
            run_to_outcomes(_stuck, 2, (), tmp_path / "ranks", part="save()", part_s=1.0)
        assert sorted(p.name for p in tmp_path.rglob("started*")) == ["started0", "started1"]
        assert [p.exitcode for p in processes] == [-signal.SIGTERM, -signal.SIGTERM]
