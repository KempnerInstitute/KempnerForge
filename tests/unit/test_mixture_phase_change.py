"""A data phase that changes the mixture weights inside an epoch.

From the phase's ``start_step`` on, the rest of the epoch draws with the new
weights from the samples the epoch has not drawn yet, so a dataset that no
weighting of the epoch oversamples repeats no sample in the epoch. A resume at
any step draws what the uninterrupted run draws, and a run whose weights never
change inside an epoch draws what the sampler always drew.
"""

from __future__ import annotations

import io
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from kempnerforge.config.data import DataConfig, DatasetSource, TrainingPhase
from kempnerforge.config.job import JobConfig
from kempnerforge.config.training import TrainConfig
from kempnerforge.data.sampler import MixtureSampler
from kempnerforge.training.data_pipeline import (
    advance_phases,
    build_data_pipeline,
    build_phase_state,
)
from kempnerforge.training.loop import BatchStream
from kempnerforge.training.runtime import RuntimeContext

SAMPLES = 1000
BATCH = 10
CPU = RuntimeContext(
    rank=0, local_rank=0, world_size=1, device=torch.device("cpu"), device_mesh=None
)
BASE_KEYS = {"epoch", "seed", "num_replicas", "rank"}


@pytest.fixture
def sources(tmp_path: Path) -> Path:
    """Datasets ``a`` and ``b`` of 1,000 samples each, every sample's tokens its id."""
    for i, name in enumerate("ab"):
        (tmp_path / name).mkdir()
        ids = np.arange(i * SAMPLES, (i + 1) * SAMPLES, dtype=np.int32)
        np.save(tmp_path / name / "tokens.npy", np.repeat(ids, 4))
    return tmp_path


def _roundtrip(state: dict) -> dict:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def _run(
    root: Path,
    phases: list[TrainingPhase],
    start: int,
    stop: int,
    *,
    state: dict | None = None,
    batch: int = BATCH,
    workers: int = 0,
    saves: dict[int, dict] | None = None,
) -> list[int]:
    """Sample ids of steps ``start`` to ``stop - 1``, drawn the way training draws them.

    Like the training entry, a resume applies the saved loader state, then the
    phases due at ``start``. Like the loop, a phase that fires drops the batch
    iterator, and a checkpoint taken after a step holds the state ``saves``
    keeps under the next step's number.
    """
    config = JobConfig(
        train=TrainConfig(batch_size=batch, seq_len=3),
        data=DataConfig(
            num_workers=workers,
            pin_memory=False,
            datasets=[DatasetSource(path=str(root / n), name=n, weight=0.5) for n in "ab"],
            phases=phases,
        ),
    )
    data = build_data_pipeline(config, CPU)
    loader = data.dataloader
    assert loader is not None
    if state is not None:
        loader.load_state_dict(state)
    phase_state = build_phase_state(config, data, step=start)
    batches = BatchStream(data)
    drawn: list[int] = []
    for step in range(start, stop):
        drawn.extend(batches.next_batch()["input_ids"][:, 0].tolist())
        if advance_phases(phase_state, data, step + 1):
            batches.reset()
        if saves is not None:
            saves[step + 1] = _roundtrip(loader.state_dict())
    return drawn


@pytest.mark.parametrize("workers", [0, 2])
def test_a_mid_epoch_change_draws_no_sample_twice_that_no_weighting_repeats(sources, workers):
    """0.5/0.5 to 0.6/0.4 halfway: ``b`` (1,000 then 800 per epoch) repeats nothing."""
    same = [TrainingPhase(start_step=100, dataset_weights={"a": 0.5})]
    new = [TrainingPhase(start_step=100, dataset_weights={"a": 0.6, "b": 0.4})]
    control = _run(sources, same, 0, 200, workers=workers)
    changed = _run(sources, new, 0, 200, workers=workers)

    assert changed[:1000] == control[:1000]
    assert max(Counter(i for i in changed if i >= SAMPLES).values()) == 1
    after = changed[1000:]
    assert sum(i >= SAMPLES for i in after) == 400
    assert sum(i < SAMPLES for i in after) == 600


TWO_CHANGES = [
    TrainingPhase(start_step=60, dataset_weights={"a": 0.7, "b": 0.3}),
    TrainingPhase(start_step=130, dataset_weights={"a": 0.2, "b": 0.8}),
]


@pytest.mark.parametrize("resume", [40, 60, 61, 100, 130, 131, 170, 200, 250])
def test_a_resume_draws_what_the_uninterrupted_run_draws(sources, resume):
    """Resumes before, at and after each of two changes in one epoch, and in the next epoch."""
    saves: dict[int, dict] = {}
    full = _run(sources, TWO_CHANGES, 0, 300, saves=saves)

    resumed = _run(sources, TWO_CHANGES, resume, 300, state=saves[resume])

    assert resumed == full[resume * BATCH :]
    recorded = "epoch_weights" in saves[resume]["sampler"]
    assert recorded == (60 <= resume <= 200)


def test_a_resume_with_dataloader_workers_draws_what_the_uninterrupted_run_draws(sources):
    saves: dict[int, dict] = {}
    full = _run(sources, TWO_CHANGES, 0, 250, workers=2, saves=saves)

    resumed = _run(sources, TWO_CHANGES, 131, 250, state=saves[131], workers=2)

    assert resumed == full[1310:]


def test_a_resume_at_another_batch_size_continues_the_sample_stream(sources):
    phases = [TrainingPhase(start_step=60, dataset_weights={"a": 0.7, "b": 0.3})]
    saves: dict[int, dict] = {}
    full = _run(sources, phases, 0, 200, saves=saves)

    resumed = _run(sources, phases, 80, 140, state=saves[80], batch=20)

    assert resumed == full[800:]


def test_a_run_without_a_change_saves_no_new_state(sources):
    saves: dict[int, dict] = {}
    _run(sources, [], 0, 250, saves=saves)

    assert all(set(state["sampler"]) == BASE_KEYS for state in saves.values())


def _reference(
    cumulative: list[int],
    weights: list[float],
    replicas: int,
    rank: int,
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
    drop_last: bool,
) -> list[int]:
    """One rank's epoch as the sampler draws it under fixed weights.

    Each dataset with a positive target is permuted with the epoch generator,
    split across ranks, and drawn to its target, wrapping around when the
    target exceeds the rank's share; the whole list is then shuffled.
    """
    sizes = [hi - lo for lo, hi in zip(cumulative, cumulative[1:], strict=False)]
    probs = [w / sum(weights) for w in weights]
    total = sum(s // replicas if drop_last else -(-s // replicas) for s in sizes)
    targets = [round(p * total) for p in probs]
    diff = total - sum(targets)
    by_prob = sorted(range(len(probs)), key=lambda i: -probs[i])
    for i in range(abs(diff)):
        targets[by_prob[i % len(probs)]] += 1 if diff > 0 else -1
    g = torch.Generator().manual_seed(seed + epoch)
    order: list[int] = []
    for size, offset, target in zip(sizes, cumulative, targets, strict=False):
        if target <= 0 or size == 0:
            continue
        indices = torch.randperm(size, generator=g).tolist() if shuffle else list(range(size))
        if drop_last:
            indices = indices[: size - size % replicas]
        else:
            indices += [indices[k % size] for k in range(-size % replicas)]
        share = indices[rank::replicas]
        if share:
            order.extend(offset + share[k % len(share)] for k in range(target))
    if shuffle:
        order = [order[p] for p in torch.randperm(len(order), generator=g).tolist()]
    return order


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("replicas", [1, 3])
def test_weights_that_change_only_between_epochs_draw_the_same_lists(shuffle, drop_last, replicas):
    """No change, a change at an epoch's first sample, and a skip: the fixed-weight lists."""
    cumulative = [0, 1, 41, 41, 101]
    before, after = [1.0, 3.0, 1.0, 0.5], [0.0, 1.0, 2.0, 4.0]
    for rank in range(replicas):
        sampler = MixtureSampler(
            cumulative, before, replicas, rank, shuffle=shuffle, seed=7, drop_last=drop_last
        )
        options = {"seed": 7, "shuffle": shuffle, "drop_last": drop_last}
        assert list(sampler) == _reference(cumulative, before, replicas, rank, epoch=0, **options)

        sampler.update_weights(after)
        assert list(sampler) == _reference(cumulative, after, replicas, rank, epoch=0, **options)

        sampler.set_epoch(1)
        sampler.set_skip(5)
        expected = _reference(cumulative, after, replicas, rank, epoch=1, **options)
        assert list(sampler) == expected[5:]
        assert set(sampler.state_dict()) == BASE_KEYS


def test_a_change_to_the_same_weights_inside_an_epoch_changes_nothing():
    sampler = MixtureSampler([0, 300, 900], [1.0, 2.0], num_replicas=1, rank=0, seed=3)
    full = list(sampler)

    sampler.update_weights([2.0, 4.0])
    sampler.set_skip(250)

    assert list(sampler) == full[250:]
    assert set(sampler.state_dict()) == BASE_KEYS


def _in_order(sizes: list[int], weights: list[float], position: int, new: list[float]) -> list[int]:
    """One rank's unshuffled epoch whose weights change at ``position``."""
    cumulative = [0]
    for size in sizes:
        cumulative.append(cumulative[-1] + size)
    sampler = MixtureSampler(cumulative, weights, num_replicas=1, rank=0, shuffle=False)
    drawn = list(sampler)[:position]
    sampler.update_weights(new)
    sampler.set_skip(position)
    return drawn + list(sampler)


def test_the_rest_of_the_epoch_takes_undrawn_samples_first_then_repeats():
    """``a`` 0.5 to 0.9 at 50: the 0.9 weighting oversamples it, so it wraps around."""
    epoch = _in_order([100, 100], [0.5, 0.5], 50, [0.9, 0.1])

    rest = epoch[50:]
    a = list(range(50, 100)) + list(range(85))
    b = list(range(100, 115))
    assert rest == a + b


def test_a_dataset_no_weighting_oversamples_draws_at_most_its_undrawn_samples():
    """``a`` is drawn out by the change; its 1/3 share goes to the oversampled ``b``."""
    epoch = _in_order([100, 100, 100], [1.0, 1.0, 1.0], 150, [2.0, 3.0, 1.0])

    counts = Counter(epoch)
    assert len(epoch) == 300
    assert all(counts[i] == 1 for i in range(100))
    assert sum(100 <= i < 200 for i in epoch[150:]) == 125
    assert sum(i >= 200 for i in epoch[150:]) == 25
    assert all(counts[i] <= 1 for i in range(200, 300))


def test_weights_that_oversample_nothing_draw_every_sample_once():
    """0.501/0.499 targets each dataset's size, so the rest is exactly the undrawn samples."""
    epoch = _in_order([100, 100], [0.5, 0.5], 50, [0.501, 0.499])

    assert sorted(epoch) == list(range(200))


def test_several_changes_never_repeat_a_dataset_no_weighting_oversamples():
    """``c`` keeps a third of every epoch while ``a`` and ``b`` trade weight twice."""
    sampler = MixtureSampler([0, 300, 600, 900], [1.0, 1.0, 1.0], num_replicas=1, rank=0, seed=11)
    epoch = list(sampler)[:200]
    for position, weights in ((200, [3.0, 1.0, 2.0]), (500, [1.0, 3.0, 2.0])):
        sampler.update_weights(weights)
        sampler.set_skip(position)
        epoch = epoch[:position] + list(sampler)

    c = [i for i in epoch if i >= 600]
    assert len(epoch) == 900
    assert len(c) == len(set(c))
    assert [position for position, _ in sampler.state_dict()["epoch_weights"]] == [0, 200, 500]


@pytest.mark.parametrize(
    ("cumulative", "replicas", "drop_last", "position", "after", "lengths", "drawn"),
    [
        ([0, 60, 120, 180], 2, True, 30, [2.0, 2.0, 1.0], [90, 90], 24),
        ([0, 1, 61, 121], 3, False, 5, [1.0, 1.0, 1.0], [41, 41, 41], 36),
    ],
    ids=["even", "one_sample_dataset"],
)
def test_a_dataset_entering_mid_epoch_draws_each_ranks_own_share(
    cumulative, replicas, drop_last, position, after, lengths, drawn
):
    """The last dataset has weight 0 until ``position``, then draws disjoint shares.

    A dataset of one sample padded across three ranks gives every rank a share
    of it, so the ranks' lists keep the same length through the change.
    """
    entering: list[int] = []
    for rank in range(replicas):
        sampler = MixtureSampler(
            cumulative, [1.0, 1.0, 0.0], replicas, rank, seed=5, drop_last=drop_last
        )
        epoch = list(sampler)[:position]
        sampler.update_weights(after)
        sampler.set_skip(position)
        epoch += list(sampler)
        assert len(epoch) == lengths[rank]
        entering += [i for i in epoch if i >= cumulative[2]]

    assert len(entering) == drawn
    assert len(entering) == len(set(entering))


def test_a_change_at_the_largest_seed_draws_the_rest_of_the_epoch():
    sampler = MixtureSampler([0, 100, 200], [1.0, 1.0], num_replicas=1, rank=0, seed=2**64 - 1)
    epoch = list(sampler)[:40]
    sampler.update_weights([3.0, 1.0])
    sampler.set_skip(40)
    epoch += list(sampler)

    b = [i for i in epoch if i >= 100]
    assert len(epoch) == 200
    assert len(b) == len(set(b))


def test_the_state_records_a_change_until_the_next_epoch():
    sampler = MixtureSampler([0, 100, 200], [1.0, 1.0], num_replicas=1, rank=0, seed=2)
    list(sampler)
    sampler.update_weights([3.0, 1.0])
    pending = sampler.state_dict()

    sampler.set_skip(40)
    list(sampler)
    applied = sampler.state_dict()
    sampler.set_epoch(1)
    list(sampler)

    assert pending["epoch_weights"] == [[0, [0.5, 0.5]]]
    assert applied["epoch_weights"] == [[0, [0.5, 0.5]], [40, [0.75, 0.25]]]
    assert set(sampler.state_dict()) == BASE_KEYS


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_a_resumed_sampler_replays_the_recorded_change(temperature):
    def build() -> MixtureSampler:
        return MixtureSampler(
            [0, 100, 200], [1.0, 1.0], num_replicas=1, rank=0, seed=2, temperature=temperature
        )

    sampler = build()
    list(sampler)
    sampler.update_weights([3.0, 1.0], temperature=temperature)
    sampler.set_skip(40)
    rest = list(sampler)
    state = _roundtrip(sampler.state_dict())

    resumed = build()
    resumed.load_state_dict(state)
    resumed.update_weights([3.0, 1.0], temperature=temperature)
    resumed.set_skip(70)

    assert list(resumed) == rest[30:]
    assert state["epoch_weights"][1][1] == pytest.approx(
        [0.75, 0.25] if temperature == 1.0 else [3**0.5 / (3**0.5 + 1), 1 / (3**0.5 + 1)]
    )


def test_a_state_without_a_record_rebuilds_the_epoch_with_the_current_weights():
    """A checkpoint from before the record resumes as it did then."""
    resumed = MixtureSampler([0, 100, 200], [1.0, 1.0], num_replicas=1, rank=0, seed=2)
    resumed.load_state_dict({"epoch": 0, "seed": 2, "num_replicas": 1, "rank": 0})
    resumed.update_weights([3.0, 1.0])
    resumed.set_skip(70)

    expected = _reference(
        [0, 100, 200], [3.0, 1.0], 1, 0, seed=2, epoch=0, shuffle=True, drop_last=True
    )
    assert list(resumed) == expected[70:]


def test_a_record_for_another_number_of_datasets_is_dropped(caplog, monkeypatch):
    state = {"epoch": 0, "epoch_weights": [[0, [0.5, 0.5]], [40, [0.75, 0.25]]]}
    sampler = MixtureSampler([0, 100, 200, 300], [1.0, 1.0, 1.0], num_replicas=1, rank=0)

    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="kempnerforge.data.sampler"):
        sampler.load_state_dict(state)
    sampler.set_skip(70)

    assert "for 2 datasets but this mixture has 3" in caplog.text
    expected = _reference(
        [0, 100, 200, 300], [1.0, 1.0, 1.0], 1, 0, seed=0, epoch=0, shuffle=True, drop_last=True
    )
    assert list(sampler) == expected[70:]
