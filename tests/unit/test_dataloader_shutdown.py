"""A process that uses ``StatefulDataLoader`` exits when its code ends or raises.

The loader's workers ignore SIGTERM, so they survive a preemption signal long
enough for the emergency checkpoint. ``multiprocessing``'s exit hook stops
leftover workers with SIGTERM and then joins them, so a worker still alive when
it runs blocks the exit forever. The loader shuts its workers down through its
own sentinel path instead: from ``close()``, and at exit for every loader whose
workers are still alive.
"""

from __future__ import annotations

import os
import signal
import time
import weakref
from pathlib import Path

import pytest
import torch
import torch.utils.data._utils
from torch.utils.data import Dataset, TensorDataset

from kempnerforge.config.schema import DataConfig
from kempnerforge.data import dataloader
from kempnerforge.data.dataloader import StatefulDataLoader
from kempnerforge.data.sampler import DistributedSampler
from tests.loader_workers import run_to_exit, worker_sockets_fit

LIVE_ITERATOR_AT_EXIT = """
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
    print("served a batch", flush=True)
    if ending == "raises":
        raise RuntimeError("step failed")
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


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("ending", ["returns", "raises"])
def test_a_process_with_a_live_iterator_exits(tmp_path, workers, ending):
    """The script's iterator is still alive at exit, held by a module global."""
    _skip_without_worker_sockets(workers)
    script = tmp_path / "live_iterator.py"
    script.write_text(LIVE_ITERATOR_AT_EXIT)

    result = run_to_exit(script, str(workers), ending)

    assert result.stdout == "served a batch\n"
    if ending == "returns":
        assert result.returncode == 0, result.stderr[-2000:]
    else:
        assert result.returncode == 1
        assert "RuntimeError: step failed" in result.stderr


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


def test_the_exit_hook_is_registered_once_workers_start(monkeypatch):
    """Registered after a worker starts, so after ``multiprocessing`` registered its own."""
    _skip_without_worker_sockets(2)
    registered = []
    monkeypatch.setattr(dataloader.atexit, "register", registered.append)
    dataloader._register_exit_hook.cache_clear()
    without_workers, with_workers = _loader(workers=0), _loader(workers=2)
    try:
        next(iter(without_workers))
        assert registered == []
        next(iter(with_workers))
        next(iter(with_workers))
        assert registered == [dataloader._close_open_loaders]
    finally:
        with_workers.close()
        dataloader._register_exit_hook.cache_clear()
