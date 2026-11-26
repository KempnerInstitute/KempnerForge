"""A resumed loader continues the CPU RNG and gives its workers the run's seeds.

torch draws the base seed of a ``DataLoader`` iterator from the CPU RNG whenever it
starts one, and worker ``k`` takes ``base_seed + k``. A loader without persistent
workers starts one at every pass; one with them starts one at its first pass and
after ``close()``. A resumed ``StatefulDataLoader`` draws its first base seed from
the CPU RNG state saved with its position instead, so it draws nothing from the CPU
RNG where the uninterrupted run drew nothing, and its workers get the seeds the
uninterrupted run's workers have. The loaders run through ``BatchStream``, as in
training: a new pass starts when one ends.
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from kempnerforge.config.schema import DataConfig
from kempnerforge.data.dataloader import StatefulDataLoader, shutdown_workers
from kempnerforge.data.sampler import DistributedSampler
from kempnerforge.resilience.signal_handler import ignore_shutdown_signals_in_worker
from kempnerforge.training.data_pipeline import DataPipeline
from kempnerforge.training.loop import BatchStream
from tests.loader_workers import worker_sockets_fit

BATCHES_PER_PASS = 16
KEY = "base_seed_rng_state"


class _Seeds(Dataset):
    """Item ``i`` is ``[i, worker id, worker seed]``; both are -1 in the main process."""

    def __len__(self) -> int:
        return 4 * BATCHES_PER_PASS

    def __getitem__(self, i: int) -> torch.Tensor:
        info = get_worker_info()
        if info is None:
            return torch.tensor([i, -1, -1])
        return torch.tensor([i, info.id, info.seed])


def _skip_without_worker_sockets(workers: int) -> None:
    if workers and not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")


def _loader(workers: int) -> StatefulDataLoader:
    sampler = DistributedSampler(_Seeds(), num_replicas=1, rank=0, shuffle=True, seed=0)
    config = DataConfig(num_workers=workers, pin_memory=False)
    return StatefulDataLoader(_Seeds(), batch_size=4, sampler=sampler, config=config)


def _stream(workers: int) -> BatchStream:
    return BatchStream(DataPipeline(dataloader=_loader(workers)))


def _take(stream: BatchStream, count: int) -> list[torch.Tensor]:
    return [stream.next_batch() for _ in range(count)]


def _order(batches: list[torch.Tensor]) -> list[list[int]]:
    return [batch[:, 0].tolist() for batch in batches]


def _seeds(batches: list[torch.Tensor]) -> dict[int, int]:
    """Worker id -> the seed of the worker that loaded each item."""
    return {worker: seed for batch in batches for _, worker, seed in batch.tolist()}


def _run_and_resume(
    workers: int,
    saved_after: int,
    then: int,
    *,
    close_before_save: bool = False,
    reset_before_save: bool = False,
) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """An uninterrupted run's batches ``saved_after`` to ``saved_after + then`` and a resume's.

    Both start from ``torch.manual_seed(0)``. The resume restores the CPU RNG and the
    loader state saved after ``saved_after`` batches. With ``reset_before_save`` both
    runs reset the stream there, as a data phase does, and with ``close_before_save``
    the saving run closes the loader first, as the training loop does before its
    final save. Returns each run's batches and its CPU RNG state after them.
    """
    torch.manual_seed(0)
    uninterrupted = _stream(workers)
    try:
        _take(uninterrupted, saved_after)
        if reset_before_save:
            uninterrupted.reset()
        expected = _take(uninterrupted, then)
        expected_rng = torch.random.get_rng_state()
    finally:
        uninterrupted.close()
    torch.manual_seed(0)
    saving = _stream(workers)
    try:
        _take(saving, saved_after)
        if reset_before_save:
            saving.reset()
        if close_before_save:
            saving.close()
        rng, state = torch.random.get_rng_state(), saving.dataloader.state_dict()
    finally:
        saving.close()
    torch.random.set_rng_state(rng)
    resumed = _stream(workers)
    try:
        resumed.dataloader.load_state_dict(state)
        got = _take(resumed, then)
        got_rng = torch.random.get_rng_state()
    finally:
        resumed.close()
    return expected, expected_rng, got, got_rng


def _assert_resumed(expected, expected_rng, got, got_rng, workers: int) -> None:
    assert _order(got) == _order(expected)
    assert torch.equal(got_rng, expected_rng), "the resume left the CPU RNG elsewhere"
    seeds = _seeds(got)
    assert seeds == _seeds(expected)
    assert len(seeds) == max(workers, 1)


@pytest.mark.parametrize("workers", [0, 2])
def test_a_resume_continues_the_cpu_rng_and_the_workers_seeds(workers):
    """Resumed after 3 batches, batches 3 to 5 and the CPU RNG after them match."""
    _skip_without_worker_sockets(workers)
    _assert_resumed(*_run_and_resume(workers, saved_after=3, then=3), workers)


@pytest.mark.parametrize("workers", [0, 2])
def test_a_resume_in_a_later_pass_keeps_the_seeds_of_that_pass(workers):
    """Resumed 3 batches into the second pass; persistent workers keep their first seeds."""
    _skip_without_worker_sockets(workers)
    _assert_resumed(*_run_and_resume(workers, saved_after=BATCHES_PER_PASS + 3, then=3), workers)


@pytest.mark.parametrize("workers", [0, 2])
def test_a_resume_after_the_last_batch_of_a_pass_starts_the_next_as_the_run_does(workers):
    """Saved after a pass's last batch, the resume starts the next pass as the run does.

    Without persistent workers that pass's iterator draws a new base seed from the
    CPU RNG in both runs; with them, the workers keep the seeds of the first pass.
    """
    _skip_without_worker_sockets(workers)
    _assert_resumed(*_run_and_resume(workers, saved_after=BATCHES_PER_PASS, then=3), workers)


@pytest.mark.parametrize("workers", [0, 2])
def test_a_checkpoint_saved_after_close_resumes_the_run_that_never_closed(workers):
    """``close()`` keeps the saved state, so the loop's final save resumes exactly."""
    _skip_without_worker_sockets(workers)
    _assert_resumed(*_run_and_resume(workers, 3, 3, close_before_save=True), workers)


@pytest.mark.parametrize("close_before_save", [False, True])
@pytest.mark.parametrize("workers", [0, 2])
def test_a_checkpoint_saved_after_a_reset_resumes_as_the_run_restarts_the_pass(
    workers, close_before_save
):
    """Saved after ``BatchStream.reset``, the resume starts the pass again as the run does.

    Without persistent workers the run's next iterator draws a new base seed from
    the CPU RNG, and so does the resume's; with them, both keep the workers' seeds.
    """
    _skip_without_worker_sockets(workers)
    resumed = _run_and_resume(
        workers, 3, 3, close_before_save=close_before_save, reset_before_save=True
    )
    _assert_resumed(*resumed, workers)


def test_a_load_into_a_loader_with_running_workers_starts_new_ones_from_the_loaded_state():
    """Workers started before ``load_state_dict`` stop; the next pass uses the loaded state."""
    _skip_without_worker_sockets(2)
    torch.manual_seed(0)
    saving = _stream(2)
    try:
        _take(saving, 3)
        rng, state = torch.random.get_rng_state(), saving.dataloader.state_dict()
        expected = _take(saving, 3)
        expected_rng = torch.random.get_rng_state()
    finally:
        saving.close()
    torch.manual_seed(1)
    running = _stream(2)
    try:
        _take(running, 5)
        torch.random.set_rng_state(rng)
        running.dataloader.load_state_dict(state)
        got = _take(running, 3)
        got_rng = torch.random.get_rng_state()
    finally:
        running.close()
    _assert_resumed(expected, expected_rng, got, got_rng, 2)


def _record_passes(loader, restart) -> list[tuple[torch.Tensor, dict[int, int]]]:
    """The CPU RNG and the worker seeds after each pass of a fixed sequence of passes.

    Two whole passes, a pass started again after 2 batches, and one more after
    ``restart(loader)`` stops the workers. Each pass calls ``iter()`` once, as
    training does: ``StatefulDataLoader.__iter__`` returns the loader, so
    ``list(iter(loader))`` would start a second iterator.
    """
    torch.manual_seed(0)
    records = []

    def record(batches):
        records.append((torch.random.get_rng_state(), _seeds(batches)))

    record(list(loader))
    record(list(loader))
    batches = iter(loader)
    partial = [next(batches), next(batches)]
    batches = iter(loader)
    record([*partial, next(batches), next(batches)])
    restart(loader)
    record([next(iter(loader))])
    restart(loader)
    return records


@pytest.mark.parametrize("workers", [0, 2])
def test_a_fresh_loader_draws_each_base_seed_from_the_cpu_rng_as_a_dataloader_does(workers):
    """Without a load, the loader draws from the CPU RNG exactly as torch's own loader."""
    _skip_without_worker_sockets(workers)
    sampler = DistributedSampler(_Seeds(), num_replicas=1, rank=0, shuffle=True, seed=0)
    plain = DataLoader(
        _Seeds(),
        batch_size=4,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        drop_last=True,
        worker_init_fn=ignore_shutdown_signals_in_worker,
    )
    expected = _record_passes(plain, shutdown_workers)
    got = _record_passes(_loader(workers), StatefulDataLoader.close)
    assert len(got) == len(expected) == 4
    for (got_rng, got_seeds), (rng, seeds) in zip(got, expected, strict=True):
        assert torch.equal(got_rng, rng)
        assert got_seeds == seeds


@pytest.mark.parametrize("workers", [0, 2])
def test_close_keeps_the_saved_state_and_the_next_pass_draws_a_new_base_seed(workers):
    """``close()`` leaves ``state_dict`` as it was; the next pass draws as after shutdown."""
    _skip_without_worker_sockets(workers)
    loader = _loader(workers)
    try:
        assert loader.state_dict()[KEY] is None
        before = torch.random.get_rng_state()
        next(iter(loader))
        saved = loader.state_dict()
        assert saved[KEY] == before.numpy().tobytes()
        loader.close()
        assert loader.state_dict() == saved
        again = torch.random.get_rng_state()
        next(iter(loader))
        assert loader.state_dict()[KEY] == again.numpy().tobytes()
        assert not torch.equal(torch.random.get_rng_state(), again)
    finally:
        loader.close()


@pytest.mark.parametrize("workers", [0, 2])
def test_the_saved_state_ends_with_a_pass_only_without_persistent_workers(workers):
    """A pass's end drops the state where the next pass's iterator draws a new base seed."""
    _skip_without_worker_sockets(workers)
    loader = _loader(workers)
    try:
        list(loader)
        assert (loader.state_dict()[KEY] is None) == (workers == 0)
    finally:
        loader.close()


@pytest.mark.parametrize(
    ("workers", "saved_after", "warns"),
    [(0, 3, True), (2, 3, True), (2, BATCHES_PER_PASS, True), (0, BATCHES_PER_PASS, False)],
)
def test_a_checkpoint_without_the_saved_state_resumes_as_before(
    workers, saved_after, warns, caplog, monkeypatch
):
    """A position saved without the state draws a new base seed from the CPU RNG.

    It warns where that changes the resume: not at a pass a loader without
    persistent workers starts anyway. The ``kempnerforge`` logger stops
    propagation once configured, so the test lets records reach ``caplog``.
    """
    _skip_without_worker_sockets(workers)
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    torch.manual_seed(0)
    saving = _stream(workers)
    try:
        _take(saving, saved_after)
        state = {k: v for k, v in saving.dataloader.state_dict().items() if k != KEY}
        expected = _take(saving, 2)
    finally:
        saving.close()
    resumed = _stream(workers)
    try:
        with caplog.at_level(logging.WARNING, logger="kempnerforge.data.dataloader"):
            resumed.dataloader.load_state_dict(state)
        before = torch.random.get_rng_state()
        resumed.ensure_started()
        drew = not torch.equal(torch.random.get_rng_state(), before)
        got = _take(resumed, 2)
    finally:
        resumed.close()
    assert drew
    assert _order(got) == _order(expected)
    warned = [r for r in caplog.records if "predates the saved base seed" in r.getMessage()]
    assert bool(warned) == warns
