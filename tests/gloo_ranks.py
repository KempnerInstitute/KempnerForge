"""Spawn CPU ranks that join one gloo group, for tests that run real collectives.

The test process holds the rendezvous store open on a port the kernel picked, from before the
spawn until the ranks are joined, so no other process can take that port while the ranks start.
A rank leaves the group only once every rank has finished its part: gloo's connection setup is
asymmetric, so a rank that left early could close a connection another rank has not yet picked
up (pytorch/pytorch#199026). The ranks wait for each other on the store, not with a collective,
so the wait cannot interleave with collectives a rank's own threads still run on the group, such
as an async checkpoint save's.
"""

from __future__ import annotations

import contextlib
import itertools
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import timedelta
from multiprocessing.process import BaseProcess
from typing import Any

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
