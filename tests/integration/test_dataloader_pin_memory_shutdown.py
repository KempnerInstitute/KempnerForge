"""A loader that pins its batches stops its pin-memory thread and its workers on close.

torch's worker shutdown joins the pin-memory thread with no timeout, so a thread stuck
pinning a batch would hold ``close()``, and the training loop's final save behind it.
"""

from __future__ import annotations

import threading

import pytest
import torch
import torch.utils.data._utils.pin_memory as pin_memory_module
from torch.utils.data import TensorDataset

from kempnerforge.config.schema import DataConfig
from kempnerforge.data import dataloader
from kempnerforge.data.dataloader import StatefulDataLoader
from kempnerforge.data.sampler import DistributedSampler
from tests.loader_workers import run_to_exit, worker_sockets_fit

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")

UNPINNED_THEN_PINNED_AT_EXIT = """
import time

import torch
from torch.utils.data import Dataset, TensorDataset

from kempnerforge.config.schema import DataConfig
from kempnerforge.data.dataloader import StatefulDataLoader
from kempnerforge.data.sampler import DistributedSampler


class StallsAfterFirstBatch(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, index):
        if index >= 4:
            time.sleep(600)
        return torch.tensor([index])


def loader(dataset, pin_memory):
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
    config = DataConfig(num_workers=1, pin_memory=pin_memory)
    return StatefulDataLoader(dataset, batch_size=4, sampler=sampler, config=config)


if __name__ == "__main__":
    unpinned = loader(TensorDataset(torch.arange(64)), pin_memory=False)
    pinned = loader(StallsAfterFirstBatch(), pin_memory=True)
    next(iter(unpinned))
    next(iter(pinned))
    print("served a batch from each", flush=True)
"""


def _pinning_loader() -> StatefulDataLoader:
    if not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    dataset = TensorDataset(torch.arange(64))
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
    config = DataConfig(num_workers=2, pin_memory=True)
    return StatefulDataLoader(dataset, batch_size=4, sampler=sampler, config=config)


def test_close_stops_the_pin_memory_thread_and_the_workers():
    loader = _pinning_loader()
    (batch,) = next(iter(loader))
    iterator = loader._dataloader._iterator
    thread, workers = iterator._pin_memory_thread, list(iterator._workers)
    assert batch.is_pinned()

    loader.close()

    assert not thread.is_alive()
    assert [worker.exitcode for worker in workers] == [0, 0]


def test_close_returns_while_the_pin_memory_thread_is_stuck(monkeypatch):
    pinned, stuck, unstick = threading.Event(), threading.Event(), threading.Event()
    pin = pin_memory_module.pin_memory

    def pin_once_then_stick(data, device=None):
        if pinned.is_set():
            stuck.set()
            unstick.wait()
        pinned.set()
        return pin(data, device)

    monkeypatch.setattr(pin_memory_module, "pin_memory", pin_once_then_stick)
    monkeypatch.setattr(dataloader, "_SHUTDOWN_WAIT_S", 0.5)
    loader = _pinning_loader()
    next(iter(loader))
    iterator = loader._dataloader._iterator
    thread, workers = iterator._pin_memory_thread, list(iterator._workers)
    closer = threading.Thread(target=loader.close, daemon=True)
    try:
        assert stuck.wait(timeout=60), "the pin-memory thread never took a second batch"
        closer.start()
        closer.join(timeout=60)
        assert not closer.is_alive(), "close() waited on the stuck pin-memory thread"
        assert [worker.exitcode for worker in workers] == [0, 0]
    finally:
        unstick.set()
        thread.join(timeout=30)


def test_a_process_with_a_stalled_pinned_loader_after_an_unpinned_one_exits(tmp_path):
    """torch registers its exit cleanup of pinned iterators when the pinned loader starts.

    That is after the first loader with workers started, and the cleanup leaves the
    stalled worker, which ignores SIGTERM, for the loader's own exit hook to kill.
    """
    if not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    script = tmp_path / "unpinned_then_pinned.py"
    script.write_text(UNPINNED_THEN_PINNED_AT_EXIT)

    result = run_to_exit(script)

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout == "served a batch from each\n"
