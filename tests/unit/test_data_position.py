"""The data position a resume restores, across data-parallel size and batch size changes.

Ranks are simulated with explicit ``num_replicas`` and ``rank``; the gloo tests run a
real process group through ``CheckpointManager``. The resize tests compare the samples
consumed before and after a resume against the epoch's global order: together they
must be one prefix of it, with no sample skipped and none read twice. The mixture,
legacy, state and warm-start tests check refusals and what a resume restores. The
changed-topology tests auto-resume each rank's per-rank state, as the training entry
does, with and without ``checkpoint.ignore_rank_state_mismatch``.
"""

from __future__ import annotations

import json
import logging
import random
import socket
from datetime import timedelta
from pathlib import Path

import numpy as np
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


def _loader(
    dataset, replicas: int, rank: int, batch: int, cls: type = StatefulDataLoader
) -> StatefulDataLoader:
    sampler = DistributedSampler(dataset, num_replicas=replicas, rank=rank, seed=SEED)
    return cls(dataset, batch_size=batch, sampler=sampler, config=_CONFIG)


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
    """Three ranks resuming at sample 1600 all end the epoch on the same step.

    Rank 0 has 132 samples left after the position and ranks 1 and 2 have 133.
    With batch 7 that would be 18 batches against 19, so only the equal count
    ``samples_after`` gives every rank keeps them in step.
    """
    dataset = _Indices(2_000)
    _, state = _run(dataset, 4, 4, steps=100)
    loaders = [_loader(dataset, 3, r, 7) for r in range(3)]
    for loader in loaders:
        loader.load_state_dict(state)
    batches = [sum(1 for _ in loader) for loader in loaders]
    assert batches == [18, 18, 18], f"ranks end the epoch on different steps: {batches}"
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
    assert state["dataset_length"] == 4_096
    assert state["global_position"] == 2 * 4 * 10
    assert state["batches_yielded"] == 10
    assert state["sampler"]["num_replicas"] == 2


def _mixture(
    replicas: int, rank: int, batch: int, cls: type = StatefulDataLoader
) -> StatefulDataLoader:
    sampler = MixtureSampler(
        [0, 600, 1000], [0.5, 0.5], num_replicas=replicas, rank=rank, seed=SEED
    )
    return cls(_Indices(1_000), batch_size=batch, sampler=sampler, config=_CONFIG)


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


@pytest.mark.parametrize(
    ("saved", "remedy"),
    [(16, "Resume on 16 data-parallel ranks with"), (1, "Resume on 1 data-parallel rank with")],
)
def test_legacy_state_on_another_data_parallel_size_is_refused(saved, remedy):
    loader = _loader(_Indices(65_536), 64, 0, 4)
    with pytest.raises(DataPositionError, match="predates the saved batch size") as refused:
        loader.load_state_dict(_legacy_state(saved, 100))
    assert remedy in str(refused.value)


def test_a_mixture_position_in_a_single_dataset_run_names_the_mixture():
    """A mixture's per-rank count, loaded into a run without a mixture, says what it is."""
    with pytest.raises(DataPositionError, match="one rank's samples of a mixture"):
        _loader(_Indices(1_000), 2, 0, 4).load_state_dict(_mixture_state(4))


def test_check_state_refuses_what_load_state_dict_refuses_and_moves_nothing():
    """``CheckpointManager`` checks a position on every rank before any rank applies it.

    So ``check_state`` gives ``load_state_dict``'s verdict without applying the position.
    """
    _, state = _run(_Indices(4_096), 4, 4, steps=10)
    loader = _loader(_Indices(4_096), 2, 0, 4)
    loader.check_state(state)
    assert loader.state_dict()["global_position"] == 0
    with pytest.raises(DataPositionError, match="draws each rank's samples"):
        _mixture(2, 0, 4).check_state(_mixture_state(4))
    with pytest.raises(DataPositionError, match="streaming data position"):
        loader.check_state({"stream": {}, "epoch": 0, "batches_yielded": 3})


def _warnings(caplog, monkeypatch, action) -> str:
    """Run ``action`` and return the warnings it logged.

    The ``kempnerforge`` logger stops propagation once configured, so this lets
    records reach ``caplog``.
    """
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="kempnerforge.data.dataloader"):
        action()
    return caplog.text


def test_a_changed_seed_or_dataset_length_warns(caplog, monkeypatch):
    """The position counts samples of the saved order; another seed or dataset reorders it."""
    _, state = _run(_Indices(4_096), 2, 4, steps=10)
    same = _warnings(
        caplog, monkeypatch, lambda: _loader(_Indices(4_096), 2, 0, 4).load_state_dict(state)
    )
    assert "data seed" not in same and "samples but the checkpoint" not in same

    other_seed = DistributedSampler(_Indices(4_096), num_replicas=2, rank=0, seed=SEED + 1)
    loader = StatefulDataLoader(_Indices(4_096), batch_size=4, sampler=other_seed, config=_CONFIG)
    assert "The data seed is 43 but the checkpoint was saved with 42" in _warnings(
        caplog, monkeypatch, lambda: loader.load_state_dict(state)
    )
    assert "holds 5000 samples but the checkpoint was saved with 4096" in _warnings(
        caplog, monkeypatch, lambda: _loader(_Indices(5_000), 2, 0, 4).load_state_dict(state)
    )


def test_a_pre_change_position_assumes_the_batch_size_is_unchanged(caplog, monkeypatch):
    """An old checkpoint records no batch size, so its count is read at this run's size."""
    logged = _warnings(
        caplog,
        monkeypatch,
        lambda: _loader(_Indices(4_096), 2, 1, 8).load_state_dict(_legacy_state(2, 10)),
    )
    assert "assumes the batch size is unchanged (8)" in logged


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

    The ``kempnerforge`` logger stops propagation once configured, which importing the
    training entry in ``_restore`` may do first, so the test lets records reach
    ``caplog`` after the restore.
    """
    _save_loader_state(tmp_path / "saved", saved_state())
    manager = _restore(tmp_path, warm_start=True)
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
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


@pytest.mark.parametrize("warm_start", [True, False], ids=["warm_start", "resume"])
def test_a_loader_passed_to_load_follows_the_same_rule(tmp_path, caplog, monkeypatch, warm_start):
    """``load(dataloader=...)`` restores the position at once, under the same rule.

    A position that cannot carry over starts the data fresh with a warning on a warm
    start and raises on a resume, as it does through ``apply_dataloader_state``.
    """
    from kempnerforge.checkpoint.manager import CheckpointManager
    from kempnerforge.config.checkpoint import CheckpointConfig

    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    _save_loader_state(tmp_path / "saved", _legacy_state(1, 100))
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(CheckpointConfig(dir=str(tmp_path / "new_run")), model, optimizer)
    path = str(tmp_path / "saved" / "step_3")
    loader = _loader(_Indices(4_096), 2, 0, 4)
    if not warm_start:
        with pytest.raises(DataPositionError, match="predates the saved batch size"):
            manager.load(path=path, dataloader=loader)
        return
    with caplog.at_level(logging.WARNING, logger="kempnerforge.checkpoint.manager"):
        assert manager.load(path=path, dataloader=loader, warm_start=True)[0] == 3
    assert "starts at its first sample" in caplog.text
    fresh = _loader(_Indices(4_096), 2, 0, 4)
    assert next(iter(loader)).tolist() == next(iter(fresh)).tolist()


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


class _Recording(StatefulDataLoader):
    """A ``StatefulDataLoader`` that keeps the sampler rank of every state it loads."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loaded_from: list[int] = []

    def load_state_dict(self, state: dict) -> None:
        self.loaded_from.append(state["sampler"]["rank"])
        super().load_state_dict(state)


class _Collect(logging.Handler):
    """Keeps the message of every warning it handles."""

    def __init__(self, messages: list[str]) -> None:
        super().__init__(logging.WARNING)
        self.messages = messages

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _draw() -> list[float]:
    """One draw from each generator a checkpoint captures on CPU."""
    return [random.random(), float(np.random.random()), float(torch.rand(()))]


def _kind_loader(kind: str, world: int, rank: int) -> _Recording:
    if kind == "mixture":
        return _mixture(world, rank, 4, cls=_Recording)
    return _loader(_Indices(4_096), world, rank, 4, cls=_Recording)


def _init_gloo(rank: int, world: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=120),
    )


def _manager(root: str, kind: str, rank: int, world: int, lenient: bool = False):
    """A manager that records this rank's data-parallel coordinates, as the training entry's."""
    from kempnerforge.checkpoint.manager import CheckpointManager
    from kempnerforge.config.checkpoint import CheckpointConfig

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = CheckpointConfig(dir=str(Path(root) / kind), ignore_rank_state_mismatch=lenient)
    return CheckpointManager(config, model, optimizer, dp_info=(rank, world))


def _save_both_kinds(rank: int, world: int, port: int, root: str, out: str) -> None:
    """One rank of a gloo job that saves a single-dataset and a mixture checkpoint.

    Each rank seeds its generators with ``1000 + rank`` and takes 50 batches before
    each save. Rank 0 writes every rank's samples and its next random draw after
    each save, the draw an exact resume of that rank reproduces.
    """
    _init_gloo(rank, world, port)
    try:
        report = {}
        for kind in ("single", "mixture"):
            manager = _manager(root, kind, rank, world)
            loader = _kind_loader(kind, world, rank)
            _seed_all(1_000 + rank)
            iterator = iter(loader)
            read = [i for _ in range(50) for i in next(iterator).tolist()]
            manager.save(step=1, dataloader=loader)
            report[kind] = {"read": read, "draw": _draw()}
        gathered: list = [None] * world
        dist.all_gather_object(gathered, report)
        if rank == 0:
            Path(out).write_text(json.dumps(gathered))
    finally:
        dist.destroy_process_group()


def _auto_resume(
    rank: int, world: int, port: int, root: str, out: str, kind: str, lenient: bool, direct: bool
) -> None:
    """One rank of a gloo job that auto-resumes ``kind``'s checkpoint.

    It loads with ``exact_resume=True``, as the training entry's auto-resume does, and
    then applies the data position with ``apply_dataloader_state``, or hands the loader
    to ``load`` when ``direct``; ``lenient`` sets
    ``checkpoint.ignore_rank_state_mismatch``. The rank seeds its generators with
    ``555 + rank`` just before the load. Rank 0 writes every rank's report: the error
    the load raised or the 5 batches read after it, the next random draw and the one
    the fresh seeds give, the sampler ranks of the states the loader loaded, and the
    warnings logged.
    """
    _init_gloo(rank, world, port)
    warnings: list[str] = []
    try:
        manager = _manager(root, kind, rank, world, lenient)
        loader = _kind_loader(kind, world, rank)
        logging.getLogger("kempnerforge").addHandler(_Collect(warnings))
        _seed_all(555 + rank)
        report: dict = {"fresh": _draw(), "error": None, "read": []}
        _seed_all(555 + rank)
        try:
            if direct:
                manager.load(exact_resume=True, dataloader=loader)
            else:
                manager.load(exact_resume=True)
                manager.apply_dataloader_state(loader)
        except ValueError as e:
            report["error"] = f"{type(e).__name__}: {e}"
        report["draw"] = _draw()
        if report["error"] is None:
            iterator = iter(loader)
            report["read"] = [i for _ in range(5) for i in next(iterator).tolist()]
        report["loaded_from"] = loader.loaded_from
        report["warnings"] = warnings
        gathered: list = [None] * world
        dist.all_gather_object(gathered, report)
        if rank == 0:
            Path(out).write_text(json.dumps(gathered))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def four_rank_checkpoints(tmp_path_factory) -> tuple[str, list[dict]]:
    """A single-dataset and a mixture checkpoint saved by a 4-rank gloo job, and its reports."""
    root = tmp_path_factory.mktemp("topology")
    out = str(root / "saved.json")
    mp.spawn(_save_both_kinds, args=(4, _free_port(), str(root), out), nprocs=4)
    return str(root), json.loads(Path(out).read_text())


def _resume_job(root: str, world: int, kind: str, lenient: bool, direct: bool = False):
    out = str(Path(root) / f"resumed_{world}_{kind}_{lenient}_{direct}.json")
    mp.spawn(
        _auto_resume,
        args=(world, _free_port(), root, out, kind, lenient, direct),
        nprocs=world,
    )
    return json.loads(Path(out).read_text())


def test_a_changed_topology_with_the_flag_keeps_the_exact_data_position(four_rank_checkpoints):
    """Saved on 4 ranks, auto-resumed on 2 with ``ignore_rank_state_mismatch``.

    Every rank takes the saved rank 0's position, the samples all 4 ranks consumed,
    so the 2 ranks continue the global order at sample 800; the RNG streams restart
    from this run's seeds; and every rank warns what was kept and what was dropped.
    """
    root, saved = four_rank_checkpoints
    resumed = _resume_job(root, 2, "single", lenient=True)
    order = _order(4_096)
    assert _positions([r["single"]["read"] for r in saved], order) == list(range(800))
    assert [r["error"] for r in resumed] == [None, None]
    assert _positions([r["read"] for r in resumed], order) == list(range(800, 840))
    for rank, report in enumerate(resumed):
        assert report["loaded_from"] == [0], f"rank {rank} did not take the saved position"
        assert report["draw"] == report["fresh"], f"rank {rank} restored a saved RNG stream"
        kept = [w for w in report["warnings"] if "Kept: the saved data position" in w]
        assert kept and "Dropped: every rank's saved RNG state" in kept[0], report["warnings"]
        assert "world_size=4" in kept[0] and "world_size=2" in kept[0]


def test_a_changed_topology_without_the_flag_refuses_on_every_rank(four_rank_checkpoints):
    """The same resume without the flag refuses on every rank before any rank loads.

    The loader goes to ``load`` itself, so a rank that restored its position before
    refusing would show it; no rank does, and no rank restores a random stream.
    """
    root, _ = four_rank_checkpoints
    resumed = _resume_job(root, 2, "single", lenient=False, direct=True)
    for rank, report in enumerate(resumed):
        assert report["error"] is not None, f"rank {rank} resumed onto a changed topology"
        assert report["error"].startswith("RankStateMismatchError: ")
        assert "world_size=4" in report["error"] and "world_size=2" in report["error"]
        assert report["loaded_from"] == [], f"rank {rank} loaded a data position"
        assert report["draw"] == report["fresh"], f"rank {rank} restored a saved RNG stream"
    assert len({r["error"] for r in resumed}) == 1, "the ranks refused for different reasons"


def test_a_mixture_on_a_changed_topology_with_the_flag_restarts_the_data(four_rank_checkpoints):
    """A mixture's position has no counterpart on 2 ranks, so the data restarts.

    The flag opts in, so the resume does not raise: every rank warns and reads its
    first batches, as a fresh loader does.
    """
    root, _ = four_rank_checkpoints
    resumed = _resume_job(root, 2, "mixture", lenient=True)
    for rank, report in enumerate(resumed):
        assert report["error"] is None, report["error"]
        fresh = _mixture(2, rank, 4)
        iterator = iter(fresh)
        assert report["read"] == [i for _ in range(5) for i in next(iterator).tolist()]
        assert report["draw"] == report["fresh"], f"rank {rank} restored a saved RNG stream"
        restarted = [w for w in report["warnings"] if "does not carry over" in w]
        assert restarted and "starts at its first sample" in restarted[0], report["warnings"]
        assert "draws each rank's samples" in restarted[0]


@pytest.mark.parametrize("lenient", [True, False], ids=["flag", "no_flag"])
def test_the_saved_layout_restores_each_ranks_own_state(four_rank_checkpoints, lenient):
    """On the 4 ranks that saved it, each rank restores its own entry, the flag set or not.

    Each rank loads the state its own rank saved and continues its own RNG streams,
    together the ranks continue the global order at sample 800, and nothing is logged
    about a mismatch or a moved data position.
    """
    root, saved = four_rank_checkpoints
    resumed = _resume_job(root, 4, "single", lenient=lenient)
    order = _order(4_096)
    assert _positions([r["read"] for r in resumed], order) == list(range(800, 880))
    for rank, report in enumerate(resumed):
        assert report["error"] is None, report["error"]
        assert report["loaded_from"] == [rank], f"rank {rank} took another rank's state"
        assert report["draw"] == saved[rank]["single"]["draw"], f"rank {rank} lost its RNG"
        about_the_position = [
            w
            for w in report["warnings"]
            if "ignore_rank_state_mismatch" in w or "data position" in w.lower()
        ]
        assert about_the_position == [], about_the_position
