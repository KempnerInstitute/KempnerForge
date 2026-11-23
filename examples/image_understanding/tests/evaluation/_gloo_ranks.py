"""Join CPU ranks to one gloo group through a rendezvous port the test process holds.

The part of the repository's ``tests/gloo_ranks.py`` that the evaluator tests use, copied so
the example's tests stay self-contained. The test process holds the store on a port the kernel
picked until the ranks are done, so no other process can take the port while they start, and a
rank leaves the group only once every rank has finished its block.
"""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Iterator
from datetime import timedelta

import torch.distributed as dist

HOST = "127.0.0.1"
STORE_TIMEOUT = timedelta(seconds=300)
LEAVE_TIMEOUT = timedelta(seconds=45)

_joins = itertools.count()


@contextlib.contextmanager
def held_store() -> Iterator[int]:
    """Hold a rendezvous store on a port the kernel picks, for the block; yield the port."""
    store = dist.TCPStore(HOST, 0, is_master=True, wait_for_workers=False, timeout=STORE_TIMEOUT)
    try:
        yield store.port
    finally:
        del store


@contextlib.contextmanager
def gloo_group(
    rank: int, world: int, port: int, timeout: timedelta | None = None
) -> Iterator[None]:
    """Be ``rank`` of the ``world``-rank gloo group at ``port`` for the block.

    After a block that returns, the rank waits until every rank has finished its block, then
    leaves the group; a block that raises leaves at once. ``timeout`` bounds each collective
    and that wait, which otherwise gives up after ``LEAVE_TIMEOUT``.
    """
    client = dist.TCPStore(HOST, port, is_master=False, timeout=STORE_TIMEOUT)
    store = dist.PrefixStore(f"join{next(_joins)}", client)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world, timeout=timeout)
    try:
        yield
        store.set(f"finished/{rank}", "")
        _wait_to_leave(store, rank, world, LEAVE_TIMEOUT if timeout is None else timeout)
    finally:
        dist.destroy_process_group()


def _wait_to_leave(store: dist.Store, rank: int, world: int, limit: timedelta) -> None:
    """Wait for every rank to finish its block; past ``limit``, raise naming those that have not."""
    try:
        store.wait([f"finished/{r}" for r in range(world)], limit)
    except dist.DistStoreError:
        inside = [r for r in range(world) if not store.check([f"finished/{r}"])]
        raise TimeoutError(
            f"rank {rank} waited {limit.total_seconds():g}s to leave the group, but ranks "
            f"{inside} had not finished their block"
        ) from None
