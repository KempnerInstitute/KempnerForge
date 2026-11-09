"""Spawn CPU ranks that join one gloo group, for tests that run real collectives.

The test process holds the rendezvous store open on a port the kernel picked, from before the
spawn until the ranks are joined, so no other process can take that port while the ranks start.
A rank leaves the group only once every rank has finished its part: gloo's connection setup is
asymmetric, so a rank that left early could close a connection another rank has not yet picked
up (pytorch/pytorch#199026). The ranks wait for each other on the store, not with a collective,
so the wait cannot interleave with collectives a rank's own threads still run on the group, such
as an async checkpoint save's. ``run_to_outcomes`` times one part of the ranks' work from the
moment every rank has entered it, so start-up and exit never count against that part.
"""

from __future__ import annotations

import contextlib
import itertools
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import timedelta
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

HOST = "127.0.0.1"
STORE_TIMEOUT = timedelta(seconds=300)

_joins = itertools.count()


@contextlib.contextmanager
def held_store() -> Iterator[int]:
    """Hold a rendezvous store on a port the kernel picks, for the block; yield the port."""
    store = dist.TCPStore(HOST, 0, is_master=True, wait_for_workers=False, timeout=STORE_TIMEOUT)
    try:
        yield store.port
    finally:
        del store


def _stop(processes: Sequence[BaseProcess], grace_s: float = 5.0) -> None:
    """Terminate the ranks still running, and kill any still alive ``grace_s`` later."""
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + grace_s
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            process.kill()
            process.join()


def spawn(fn: Callable[..., Any], world: int, *args: Any) -> None:
    """Run ``fn(rank, world, port, *args)`` on ``world`` spawned ranks; raise what a rank raised.

    When the wait is interrupted, for example by a test timeout, the ranks are stopped first.
    """
    with held_store() as port:
        context = mp.spawn(fn, args=(world, port, *args), nprocs=world, join=False)
        try:
            while not context.join():
                pass
        except BaseException:
            _stop(context.processes)
            raise


@contextlib.contextmanager
def gloo_group(
    rank: int, world: int, port: int, timeout: timedelta | None = None
) -> Iterator[None]:
    """Be ``rank`` of the ``world``-rank gloo group at ``port`` for the block.

    After a block that returns, the rank waits until every rank has finished its block, then
    leaves the group. A block that raises leaves at once. ``timeout`` bounds each collective,
    as in ``init_process_group``. Each join in a process uses keys of its own, so a rank
    function may join, leave and join again.
    """
    client = dist.TCPStore(HOST, port, is_master=False, timeout=STORE_TIMEOUT)
    store = dist.PrefixStore(f"join{next(_joins)}/", client)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world, timeout=timeout)
    try:
        yield
        store.set(f"finished/{rank}", "")
        store.wait([f"finished/{r}" for r in range(world)])
    finally:
        dist.destroy_process_group()


def mark(out_dir: str | Path, rank: int) -> None:
    """Record that ``rank`` has entered the part of its work that ``run_to_outcomes`` times."""
    Path(out_dir, "reached", f"rank{rank}").touch()


def write_outcome(out_dir: str | Path, rank: int, outcome: str) -> None:
    """Record how the timed part ended on ``rank``."""
    Path(out_dir, "outcome", f"rank{rank}").write_text(outcome)


def _ranks(directory: Path) -> set[int]:
    """The ranks that left a ``rank{N}`` file in ``directory``."""
    return {int(p.name[4:]) for p in directory.iterdir() if p.name.startswith("rank")}


def run_to_outcomes(
    fn: Callable[..., Any],
    world: int,
    args: tuple,
    out_dir: Path,
    *,
    part: str,
    part_s: float,
    start_s: float = 240.0,
    exit_s: float = 60.0,
) -> dict[int, str]:
    """Run ``fn(rank, world, port, *args, out_dir)`` on ``world`` ranks; return each outcome.

    Each rank calls ``mark`` when it enters the timed part and ``write_outcome`` when it leaves
    it. ``part_s`` starts once every rank has entered, so start-up never counts against it, and
    a rank still inside then fails the test as still in ``part``. The ranks get ``start_s`` from
    the spawn to enter, and ``exit_s`` to exit once every outcome is written, each failing with
    its own message. The ranks are stopped before the test fails, and when the wait is
    interrupted.
    """
    for sub in ("reached", "outcome"):
        (out_dir / sub).mkdir(parents=True)
    everyone = set(range(world))
    with held_store() as port:
        context = mp.spawn(fn, args=(world, port, *args, str(out_dir)), nprocs=world, join=False)
        try:
            phase, deadline = "start", time.monotonic() + start_s
            while not context.join(timeout=0.5):
                reached, done = _ranks(out_dir / "reached"), _ranks(out_dir / "outcome")
                if phase == "start" and reached == everyone:
                    phase, deadline = "part", time.monotonic() + part_s
                if phase == "part" and done == everyone:
                    phase, deadline = "exit", time.monotonic() + exit_s
                if time.monotonic() < deadline:
                    continue
                if phase == "start":
                    late = sorted(everyone - reached)
                    pytest.fail(
                        f"ranks {late} did not reach {part} within {start_s:.0f}s of the spawn"
                    )
                if phase == "part":
                    inside = sorted(everyone - done)
                    pytest.fail(
                        f"ranks {inside} still in {part} {part_s:.0f}s after every rank entered it"
                    )
                alive = [rank for rank, p in enumerate(context.processes) if p.is_alive()]
                pytest.fail(
                    f"ranks {alive} wrote their outcome but did not exit within {exit_s:.0f}s"
                )
        except BaseException:
            _stop(context.processes)
            raise
    done = _ranks(out_dir / "outcome")
    assert done == everyone, f"ranks {sorted(everyone - done)} exited without an outcome"
    return {rank: Path(out_dir, "outcome", f"rank{rank}").read_text() for rank in range(world)}
