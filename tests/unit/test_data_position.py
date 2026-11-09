"""The data position a resume restores, across data-parallel size and batch size changes.

Ranks are simulated with explicit ``num_replicas`` and ``rank``; the gloo tests run a
real process group through ``CheckpointManager``. Every check compares the samples
consumed before and after a resume against the epoch's global order: together they
must be one prefix of it, with no sample skipped and none read twice.
"""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kempnerforge.config.schema import DataConfig
from kempnerforge.data.dataloader import DataPositionError, StatefulDataLoader
from kempnerforge.data.sampler import DistributedSampler, MixtureSampler

SEED = 42
_CONFIG = DataConfig(num_workers=0, pin_memory=False)


class _Indices(torch.utils.data.Dataset):
    """Sample ``i`` is the integer ``i``."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


def _loader(dataset, replicas: int, rank: int, batch: int) -> StatefulDataLoader:
    sampler = DistributedSampler(dataset, num_replicas=replicas, rank=rank, seed=SEED)
    return StatefulDataLoader(dataset, batch_size=batch, sampler=sampler, config=_CONFIG)


def _order(n: int, epoch: int = 0) -> dict[int, int]:
    """Sample index -> its position in the epoch's global order."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
    return {sample: position for position, sample in enumerate(perm)}


def _run(dataset, replicas: int, batch: int, steps: int, state: dict | None = None):
    """Every rank takes ``steps`` batches, after loading ``state`` if given.

    Returns the samples each rank read and rank 0's state afterward, the copy a
    checkpoint holds.
    """
    loaders = [_loader(dataset, replicas, r, batch) for r in range(replicas)]
    if state is not None:
        for loader in loaders:
            loader.load_state_dict(state)
    iterators = [iter(loader) for loader in loaders]
    read = [[i for _ in range(steps) for i in next(it).tolist()] for it in iterators]
    return read, loaders[0].state_dict()


def _positions(read: list[list[int]], order: dict[int, int]) -> list[int]:
    return sorted(order[i] for rank in read for i in rank)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ((16, 4), (64, 4)),
        ((4, 4), (2, 4)),
        ((2, 4), (4, 4)),
        ((4, 4), (2, 8)),
        ((2, 8), (4, 4)),
        ((4, 4), (3, 5)),
        ((4, 4), (4, 8)),
    ],
    ids=["16to64", "4to2", "2to4", "4to2_batch_x2", "2to4_batch_half", "4to3_uneven", "batch_x2"],
)
def test_resume_continues_the_global_order(before, after):
    """(ranks, batch) before and after the resume; the issue's 16 to 64 example first.

    The two cases where ranks x batch is unchanged resumed exactly before the fix too;
    they guard against converting with the batch size of the resumed run.
    """
    dataset = _Indices(65_536)
    order = _order(len(dataset))
    first, state = _run(dataset, before[0], before[1], steps=100)
    second, _ = _run(dataset, after[0], after[1], steps=7, state=state)
    reached = before[0] * before[1] * 100
    assert _positions(first, order) == list(range(reached))
    assert _positions(second, order) == list(range(reached, reached + after[0] * after[1] * 7))


def test_two_resumes_on_three_sizes_stay_one_prefix():
    """An uneven resume saves a position the next resize carries on exactly."""
    dataset = _Indices(8_192)
    order = _order(len(dataset))
    first, state = _run(dataset, 4, 4, steps=50)
    second, state = _run(dataset, 3, 5, steps=11, state=state)
    third, _ = _run(dataset, 2, 4, steps=9, state=state)
    read = _positions(first, order) + _positions(second, order) + _positions(third, order)
    assert sorted(read) == list(range(800 + 165 + 72))


def test_resume_in_a_later_epoch_follows_that_epochs_order():
    dataset = _Indices(1_000)
    loaders = [_loader(dataset, 4, r, 4) for r in range(4)]
    for loader in loaders:
        assert sum(1 for _ in loader) == 62
    iterators = [iter(loader) for loader in loaders]
    first = [[i for _ in range(10) for i in next(it).tolist()] for it in iterators]
    state = loaders[0].state_dict()
    assert state["epoch"] == 1
    second, _ = _run(dataset, 2, 4, steps=5, state=state)
    order = _order(len(dataset), epoch=1)
    assert _positions(first, order) == list(range(160))
    assert _positions(second, order) == list(range(160, 200))


def test_uneven_resume_keeps_ranks_in_step_to_the_epoch_end():
    """Three ranks resuming at sample 1600 all end the epoch on the same step."""
    dataset = _Indices(2_000)
    _, state = _run(dataset, 4, 4, steps=100)
    loaders = [_loader(dataset, 3, r, 5) for r in range(3)]
    for loader in loaders:
        loader.load_state_dict(state)
    batches = [sum(1 for _ in loader) for loader in loaders]
    assert len(set(batches)) == 1, f"ranks end the epoch on different steps: {batches}"
    assert {loader.state_dict()["epoch"] for loader in loaders} == {1}


def test_position_without_a_full_batch_left_starts_the_next_epoch():
    """82 batches into a 3-rank epoch, the same position leaves 7 ranks one sample each."""
    dataset = _Indices(1_000)
    _, state = _run(dataset, 3, 4, steps=82)
    loader = _loader(dataset, 7, 0, 4)
    loader.load_state_dict(state)
    assert loader.state_dict()["epoch"] == 1
    assert loader.state_dict()["global_position"] == 0


def test_state_records_the_batch_size_and_the_global_position():
    dataset = _Indices(4_096)
    _, state = _run(dataset, 2, 4, steps=10)
    assert state["batch_size"] == 4
    assert state["global_position"] == 2 * 4 * 10
    assert state["batches_yielded"] == 10
    assert state["sampler"]["num_replicas"] == 2


def _mixture(replicas: int, rank: int, batch: int) -> StatefulDataLoader:
    sampler = MixtureSampler(
        [0, 600, 1000], [0.5, 0.5], num_replicas=replicas, rank=rank, seed=SEED
    )
    return StatefulDataLoader(_Indices(1_000), batch_size=batch, sampler=sampler, config=_CONFIG)


def test_mixture_on_another_data_parallel_size_is_refused():
    loader = _mixture(4, 0, 4)
    iterator = iter(loader)
    for _ in range(10):
        next(iterator)
    state = loader.state_dict()
    with pytest.raises(DataPositionError, match="data-parallel size 4 on data-parallel size 2"):
        _mixture(2, 0, 4).load_state_dict(state)


def test_mixture_continues_each_rank_at_another_batch_size():
    full = [i for batch in _mixture(2, 1, 4) for i in batch.tolist()]
    loader = _mixture(2, 1, 4)
    iterator = iter(loader)
    head = [i for _ in range(10) for i in next(iterator).tolist()]
    resumed = _mixture(2, 1, 8)
    resumed.load_state_dict(loader.state_dict())
    tail = [i for batch in resumed for i in batch.tolist()]
    assert head == full[:40]
    assert tail == full[40 : 40 + len(tail)]


def _legacy_state(replicas: int, batches: int) -> dict:
    """A loader state as written before the batch size was saved."""
    return {
        "epoch": 0,
        "batches_yielded": batches,
        "sampler": {"epoch": 0, "seed": SEED, "num_replicas": replicas, "rank": 0},
    }


def test_legacy_state_on_another_data_parallel_size_is_refused():
    loader = _loader(_Indices(65_536), 64, 0, 4)
    with pytest.raises(DataPositionError, match="predates the saved batch size"):
        loader.load_state_dict(_legacy_state(16, 100))


def test_legacy_state_on_the_same_data_parallel_size_resumes_as_before():
    dataset = _Indices(4_096)
    loader = _loader(dataset, 2, 1, 4)
    loader.load_state_dict(_legacy_state(2, 10))
    rank_order = list(_loader(dataset, 2, 1, 1).sampler)
    assert next(iter(loader)).tolist() == rank_order[40:44]


def _save_loader_state(directory: Path, loader_state: dict) -> None:
    """Write a step-3 checkpoint into ``directory`` whose train state holds ``loader_state``."""
    from kempnerforge.checkpoint.manager import CheckpointManager
    from kempnerforge.config.checkpoint import CheckpointConfig

    class _Saved:
        def state_dict(self) -> dict:
            return loader_state

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    CheckpointManager(CheckpointConfig(dir=str(directory)), model, optimizer).save(
        step=3, dataloader=_Saved()
    )


def _restore(tmp_path: Path, *, warm_start: bool):
    """Restore from ``tmp_path / "saved"`` the way the training entry does.

    An auto-resume finds the checkpoint in ``checkpoint.dir``; a warm start names
    it in ``checkpoint.load_path`` from a new, empty ``checkpoint.dir``.
    """
    from kempnerforge.checkpoint.manager import CheckpointManager
    from kempnerforge.config.checkpoint import CheckpointConfig
    from kempnerforge.config.job import JobConfig
    from kempnerforge.training.entry import restore_checkpoint

    saved = tmp_path / "saved"
    if warm_start:
        checkpoint = CheckpointConfig(
            dir=str(tmp_path / "new_run"),
            load_path=str(saved / "step_3"),
            exclude_from_loading=["optimizer"],
        )
    else:
        checkpoint = CheckpointConfig(dir=str(saved))
    config = JobConfig(checkpoint=checkpoint)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(config.checkpoint, model, optimizer)
    assert restore_checkpoint(config, model, None, manager)[0] == 3
    return manager


def _mixture_state(replicas: int) -> dict:
    loader = _mixture(replicas, 0, 4)
    iterator = iter(loader)
    for _ in range(10):
        next(iterator)
    return loader.state_dict()


@pytest.mark.parametrize(
    ("saved_state", "resumed_loader", "reason"),
    [
        (lambda: _legacy_state(1, 100), lambda: _loader(_Indices(4_096), 2, 0, 4), "predates"),
        (lambda: _mixture_state(4), lambda: _mixture(2, 0, 4), "draws each rank's samples"),
    ],
    ids=["legacy_checkpoint", "mixture"],
)
def test_warm_start_starts_the_data_fresh_when_the_position_cannot_carry_over(
    tmp_path, caplog, monkeypatch, saved_state, resumed_loader, reason
):
    """A ``load_path`` warm start on another data-parallel size warns instead of refusing.

    The ``kempnerforge`` logger stops propagation once configured, so the test lets
    records reach ``caplog``.
    """
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    _save_loader_state(tmp_path / "saved", saved_state())
    manager = _restore(tmp_path, warm_start=True)
    loader = resumed_loader()
    with caplog.at_level(logging.WARNING, logger="kempnerforge.checkpoint.manager"):
        manager.apply_dataloader_state(loader)
    assert "starts at its first sample" in caplog.text and reason in caplog.text
    fresh = resumed_loader()
    assert next(iter(loader)).tolist() == next(iter(fresh)).tolist()


def test_warm_start_carries_a_global_position_over(tmp_path):
    """A position that carries over to another data-parallel size does so on a warm start too."""
    dataset = _Indices(4_096)
    _, state = _run(dataset, 4, 4, steps=10)
    _save_loader_state(tmp_path / "saved", state)
    manager = _restore(tmp_path, warm_start=True)
    loader = _loader(dataset, 2, 1, 4)
    manager.apply_dataloader_state(loader)
    order = _order(len(dataset))
    assert [order[i] for i in next(iter(loader)).tolist()] == [161, 163, 165, 167]


def test_auto_resume_refuses_a_position_that_cannot_carry_over(tmp_path):
    _save_loader_state(tmp_path / "saved", _legacy_state(1, 100))
    manager = _restore(tmp_path, warm_start=False)
    with pytest.raises(DataPositionError, match="predates the saved batch size"):
        manager.apply_dataloader_state(_loader(_Indices(4_096), 2, 0, 4))


def _free_port() -> int:
    """An unused local TCP port for the gloo rendezvous."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _save_or_resume(
    rank: int, world: int, port: int, root: str, batch: int, steps: int, out: str, resume: bool
) -> None:
    """One rank of a gloo job: train ``steps`` batches, then save, or resume first."""
    from kempnerforge.checkpoint.manager import CheckpointManager
    from kempnerforge.config.checkpoint import CheckpointConfig

    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        manager = CheckpointManager(CheckpointConfig(dir=root), model, optimizer)
        loader = _loader(_Indices(4_096), world, rank, batch)
        if resume:
            manager.load(path=str(Path(root) / "step_1"))
            manager.apply_dataloader_state(loader)
        iterator = iter(loader)
        read = [i for _ in range(steps) for i in next(iterator).tolist()]
        if not resume:
            manager.save(step=1, dataloader=loader)
        gathered: list = [None] * world
        dist.all_gather_object(gathered, read)
        if rank == 0:
            Path(out).write_text(json.dumps(gathered))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("before", "after"), [((4, 4), (2, 4)), ((2, 4), (3, 4))], ids=["4to2", "2to3"]
)
def test_checkpoint_resume_on_another_world_size(tmp_path, before, after):
    """A checkpoint saved by one gloo job resumes in a job of another size."""
    root = str(tmp_path / "ckpt")
    saved, resumed = str(tmp_path / "saved.json"), str(tmp_path / "resumed.json")
    mp.spawn(
        _save_or_resume,
        args=(before[0], _free_port(), root, before[1], 50, saved, False),
        nprocs=before[0],
    )
    mp.spawn(
        _save_or_resume,
        args=(after[0], _free_port(), root, after[1], 5, resumed, True),
        nprocs=after[0],
    )
    order = _order(4_096)
    reached = before[0] * before[1] * 50
    assert _positions(json.loads(Path(saved).read_text()), order) == list(range(reached))
    assert _positions(json.loads(Path(resumed).read_text()), order) == list(
        range(reached, reached + after[0] * after[1] * 5)
    )
