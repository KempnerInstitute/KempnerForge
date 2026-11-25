"""A process that uses ``StatefulDataLoader`` exits when its code ends or raises.

The loader's workers ignore SIGTERM, so they survive a preemption signal long
enough for the emergency checkpoint. ``multiprocessing``'s exit hook stops
leftover workers with SIGTERM and then joins them, so a worker still alive when
it runs blocks the exit forever. The loader shuts its workers down through its
own sentinel path instead: from ``close()``, and at exit for every loader whose
workers are still alive.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import threading
import time
import weakref
from pathlib import Path

import pytest
import torch
import torch.utils.data._utils
from torch.utils.data import DataLoader, Dataset, TensorDataset

from kempnerforge.config.schema import DataConfig
from kempnerforge.data import dataloader
from kempnerforge.data.dataloader import StatefulDataLoader, shutdown_workers
from kempnerforge.data.sampler import DistributedSampler
from kempnerforge.resilience.signal_handler import ignore_shutdown_signals_in_worker
from tests.loader_workers import run_to_exit, worker_sockets_fit

LIVE_ITERATOR_AT_EXIT = """
import multiprocessing
import sys

import torch
from torch.utils.data import TensorDataset

from kempnerforge.config.schema import DataConfig
from kempnerforge.data.dataloader import StatefulDataLoader

if __name__ == "__main__":
    workers, ending = int(sys.argv[1]), sys.argv[2]
    loader = StatefulDataLoader(
        TensorDataset(torch.arange(64)),
        batch_size=4,
        config=DataConfig(num_workers=workers, pin_memory=False),
    )
    batches = iter(loader)
    next(batches)
    if ending == "takes the multiprocessing logger":
        multiprocessing.get_logger()
    print("served a batch", flush=True)
    if ending == "raises":
        raise RuntimeError("step failed")
"""

LIVE_ITERATOR_IN_A_CHILD = """
import multiprocessing
import sys

import torch
from torch.utils.data import TensorDataset

from kempnerforge.config.schema import DataConfig
from kempnerforge.data.dataloader import StatefulDataLoader

kept = []


def serve_a_batch(ending):
    loader = StatefulDataLoader(
        TensorDataset(torch.arange(64)),
        batch_size=4,
        config=DataConfig(num_workers=2, pin_memory=False),
    )
    next(iter(loader))
    if ending == "raises":
        raise RuntimeError("step failed")
    kept.append(loader)


if __name__ == "__main__":
    start_method, ending = sys.argv[1], sys.argv[2]
    context = multiprocessing.get_context(start_method)
    child = context.Process(target=serve_a_batch, args=(ending,))
    child.start()
    child.join()
    print(f"the child exited with {child.exitcode}", flush=True)
"""


def _skip_without_worker_sockets(workers: int) -> None:
    if workers and not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")


def _loader(workers: int, dataset: Dataset | None = None, shuffle: bool = True):
    """A loader over 64 numbers, or over ``dataset``, in batches of 4 on one rank."""
    dataset = TensorDataset(torch.arange(64)) if dataset is None else dataset
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=shuffle, seed=0)
    config = DataConfig(num_workers=workers, pin_memory=False)
    return StatefulDataLoader(dataset, batch_size=4, sampler=sampler, config=config)


def _workers(loader: StatefulDataLoader) -> list:
    """The worker processes the loader's live iterator started."""
    return list(loader._dataloader._iterator._workers)


class _StallsAfterFirstBatch(Dataset):
    """Serves items 0 to 3 at once; a later item touches ``stalled`` and blocks."""

    def __init__(self, stalled: Path) -> None:
        self.stalled = stalled

    def __len__(self) -> int:
        return 64

    def __getitem__(self, index: int) -> torch.Tensor:
        if index >= 4:
            self.stalled.touch()
            time.sleep(60)
        return torch.tensor([index])


class _Closes:
    """Stands in for a loader; counts the calls to ``close``."""

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _ResultQueue:
    """Records what the shutdown does to the queue the pin-memory thread reads."""

    def __init__(self) -> None:
        self.put_items = []
        self.join_cancelled = False
        self.closed = False

    def put(self, item) -> None:
        if self.closed:
            raise ValueError("Queue is closed")
        self.put_items.append(item)

    def cancel_join_thread(self) -> None:
        self.join_cancelled = True


class _PinMemoryIterator:
    """Holds a pin-memory thread as torch's iterator does, stuck pinning or not."""

    def __init__(self, stuck: bool) -> None:
        self._pin_memory_thread_done_event = threading.Event()
        self._worker_result_queue = _ResultQueue()
        self.unstick = threading.Event()
        waits_on = self.unstick if stuck else self._pin_memory_thread_done_event
        self._pin_memory_thread = threading.Thread(target=waits_on.wait, daemon=True)
        self._pin_memory_thread.start()


@pytest.mark.parametrize(
    ("workers", "ending"),
    [
        (0, "returns"),
        (0, "raises"),
        (2, "returns"),
        (2, "raises"),
        (2, "takes the multiprocessing logger"),
    ],
)
def test_a_process_with_a_live_iterator_exits(tmp_path, workers, ending):
    """The script's iterator is still alive at exit, held by a module global.

    ``multiprocessing.get_logger`` registers ``multiprocessing``'s exit hook
    again, so that hook then runs before every hook registered earlier.
    """
    _skip_without_worker_sockets(workers)
    script = tmp_path / "live_iterator.py"
    script.write_text(LIVE_ITERATOR_AT_EXIT)

    result = run_to_exit(script, str(workers), ending)

    assert result.stdout == "served a batch\n"
    if ending == "raises":
        assert result.returncode == 1
        assert "RuntimeError: step failed" in result.stderr
    else:
        assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
@pytest.mark.parametrize("ending", ["returns", "raises"])
def test_a_multiprocessing_child_with_a_live_iterator_exits(tmp_path, start_method, ending):
    """A ``multiprocessing`` child stops its own workers before it exits.

    Python 3.12 runs ``multiprocessing``'s exit function in the child as soon as
    its target returns or raises, before any ``atexit`` hook.
    """
    _skip_without_worker_sockets(2)
    script = tmp_path / "live_iterator_in_a_child.py"
    script.write_text(LIVE_ITERATOR_IN_A_CHILD)

    result = run_to_exit(script, start_method, ending)

    assert result.returncode == 0, result.stderr[-2000:]
    exit_code = 1 if ending == "raises" else 0
    assert result.stdout == f"the child exited with {exit_code}\n"


def test_close_stops_the_workers_through_their_sentinel():
    """Every worker leaves on its own, with exit code 0, rather than being killed."""
    _skip_without_worker_sockets(2)
    loader = _loader(workers=2)
    next(iter(loader))
    workers = _workers(loader)
    assert all(worker.is_alive() for worker in workers)

    loader.close()

    assert [worker.exitcode for worker in workers] == [0, 0]


def test_close_kills_a_worker_that_misses_the_sentinel(tmp_path, monkeypatch):
    """A worker still busy after the sentinel's wait ignores SIGTERM, so it is killed."""
    _skip_without_worker_sockets(1)
    stalled = tmp_path / "stalled"
    loader = _loader(workers=1, dataset=_StallsAfterFirstBatch(stalled), shuffle=False)
    next(iter(loader))
    (worker,) = _workers(loader)
    deadline = time.monotonic() + 30
    while not stalled.exists():
        assert time.monotonic() < deadline, "the worker never reached the stalling item"
        time.sleep(0.05)
    monkeypatch.setattr(torch.utils.data._utils, "MP_STATUS_CHECK_INTERVAL", 0.1)

    loader.close()

    assert worker.exitcode == -signal.SIGKILL


def test_close_keeps_the_position():
    """After ``close``, the next pass yields what an uninterrupted loader yields."""
    _skip_without_worker_sockets(2)
    closed, uninterrupted = _loader(workers=2), _loader(workers=2)
    try:
        batches = iter(closed)
        before = [next(batches) for _ in range(3)]
        state = closed.state_dict()

        closed.close()

        assert closed.state_dict() == state
        with pytest.raises(StopIteration):
            next(closed)
        assert closed.state_dict() == state
        after = [next(iter(closed)), next(closed), next(closed)]
        reference = [batch for _, batch in zip(range(6), uninterrupted, strict=False)]
        assert closed.state_dict() == uninterrupted.state_dict()
    finally:
        closed.close()
        uninterrupted.close()
    for got, want in zip(before + after, reference, strict=True):
        assert torch.equal(got[0], want[0])


@pytest.mark.parametrize("workers", [0, 2])
def test_close_is_idempotent_and_needs_no_live_iterator(workers):
    _skip_without_worker_sockets(workers)
    loader = _loader(workers)
    loader.close()
    next(iter(loader))

    loader.close()
    loader.close()

    assert loader._dataloader._iterator is None


def test_the_exit_hook_closes_a_loader_whose_workers_are_alive():
    _skip_without_worker_sockets(2)
    loader = _loader(workers=2)
    next(iter(loader))
    workers = _workers(loader)
    assert loader in dataloader._loaders_with_workers

    dataloader._close_open_loaders()

    assert [worker.exitcode for worker in workers] == [0, 0]
    assert loader not in dataloader._loaders_with_workers


def test_a_loader_without_workers_is_left_to_its_owner():
    loader = _loader(workers=0)
    next(iter(loader))

    assert loader not in dataloader._loaders_with_workers


def test_the_exit_hook_leaves_the_loaders_another_process_started(monkeypatch):
    """A forked child inherits its parent's loaders, but their workers are not its own."""
    mine, inherited = _Closes(), _Closes()
    started = weakref.WeakKeyDictionary({mine: os.getpid(), inherited: os.getpid() + 1})
    monkeypatch.setattr(dataloader, "_loaders_with_workers", started)

    dataloader._close_open_loaders()

    assert (mine.closed, inherited.closed) == (1, 0)


def test_the_exit_hooks_follow_every_start_of_workers(monkeypatch):
    """The ``atexit`` hook moves behind every hook registered before workers start.

    So it runs before them, torch's shutdown of pinned iterators included. The
    finalizer is created once per process: a ``multiprocessing`` child starts with
    none.
    """
    _skip_without_worker_sockets(2)
    calls, finalizers = [], []
    monkeypatch.setattr(dataloader.atexit, "register", lambda hook: calls.append(("add", hook)))
    monkeypatch.setattr(dataloader.atexit, "unregister", lambda hook: calls.append(("drop", hook)))

    def finalize(obj, callback, exitpriority):
        finalizers.append((obj, callback, exitpriority))

    monkeypatch.setattr(dataloader, "Finalize", finalize)
    dataloader._register_exit_finalizer.cache_clear()
    without_workers, with_workers = _loader(workers=0), _loader(workers=2)
    try:
        next(iter(without_workers))
        assert (calls, finalizers) == ([], [])
        next(iter(with_workers))
        next(iter(with_workers))
        dataloader._register_exit_finalizer(os.getpid() + 1)
    finally:
        with_workers.close()
        dataloader._register_exit_finalizer.cache_clear()
    hook = dataloader._close_open_loaders
    assert calls == [("drop", hook), ("add", hook)] * 2
    assert finalizers == [(None, hook, dataloader._EXIT_FINALIZER_PRIORITY)] * 2


def test_the_exit_finalizer_runs_before_multiprocessing_closes_its_queues():
    """The sentinel travels through ``multiprocessing`` queues, which close at exit too."""
    queue = multiprocessing.get_context().Queue()
    queue.put(0)
    try:
        queue_priority, _ = queue._close._key
        assert queue_priority < dataloader._EXIT_FINALIZER_PRIORITY
    finally:
        queue.get()
        queue.close()
        queue.join_thread()


def test_shutdown_workers_stops_a_plain_loaders_shielded_workers():
    _skip_without_worker_sockets(2)
    loader = DataLoader(
        TensorDataset(torch.arange(16)),
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
        worker_init_fn=ignore_shutdown_signals_in_worker,
    )
    next(iter(loader))
    workers = list(loader._iterator._workers)

    shutdown_workers(loader)

    assert [worker.exitcode for worker in workers] == [0, 0]
    assert loader._iterator is None


def test_the_pin_memory_thread_stops_on_its_done_event():
    iterator = _PinMemoryIterator(stuck=False)
    thread = iterator._pin_memory_thread

    dataloader._stop_pin_memory_thread(iterator)

    assert not thread.is_alive()
    assert iterator._worker_result_queue.put_items == [(None, None)]
    assert iterator._pin_memory_thread is thread
    assert not iterator._worker_result_queue.join_cancelled


def test_a_stuck_pin_memory_thread_is_left_behind(monkeypatch):
    """torch's shutdown would join it with no timeout, so it no longer sees it."""
    monkeypatch.setattr(dataloader, "_SHUTDOWN_WAIT_S", 0.1)
    iterator = _PinMemoryIterator(stuck=True)
    thread = iterator._pin_memory_thread
    try:
        dataloader._stop_pin_memory_thread(iterator)

        assert thread.is_alive()
        assert not hasattr(iterator, "_pin_memory_thread")
        assert iterator._worker_result_queue.join_cancelled
    finally:
        iterator.unstick.set()
        thread.join()


def test_a_pin_memory_thread_that_already_stopped_is_left_alone():
    """torch's own exit cleanup of a pinned iterator may stop it and close its queue first."""
    iterator = _PinMemoryIterator(stuck=False)
    iterator._pin_memory_thread_done_event.set()
    iterator._pin_memory_thread.join()
    iterator._worker_result_queue.closed = True

    dataloader._stop_pin_memory_thread(iterator)

    assert iterator._worker_result_queue.put_items == []
