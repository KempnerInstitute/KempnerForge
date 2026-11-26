"""A streaming HuggingFace run resumes at the batch it reached, not at the first document.

Offline: the corpus is local JSONL shards and the tokenizer a local WordLevel one, so
``datasets`` and ``transformers`` load both without the network. Document ``i`` is the
word ``w{i}`` repeated, and the tokenizer maps ``w{i}`` to id ``i + 2``, so a batch
names the documents it came from. Loaders that start workers are shut down through
``BatchStream.close``, since the workers ignore SIGTERM.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist

from kempnerforge.config.schema import CheckpointConfig, DataConfig, JobConfig, TrainConfig
from kempnerforge.data.dataloader import DataPositionError, StatefulDataLoader, StreamingDataLoader
from kempnerforge.data.dataset import StreamingHuggingFaceDataset
from kempnerforge.data.sampler import DistributedSampler
from kempnerforge.metrics.logger import _RankFilter
from kempnerforge.training.data_pipeline import DataPipeline, build_data_pipeline
from kempnerforge.training.loop import BatchStream
from kempnerforge.training.runtime import RuntimeContext
from tests.gloo_ranks import gloo_group, spawn
from tests.loader_workers import run_to_exit, worker_sockets_fit

SHARDS, DOCS_PER_SHARD = 4, 50
ALL_SHARDS = tuple(range(SHARDS))


def _write_corpus(root: Path, docs_per_shard: tuple[int, ...]) -> tuple[Path, Path]:
    """JSONL shards of ``docs_per_shard`` documents each, and a tokenizer for their words."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    data = root / "corpus"
    data.mkdir()
    doc = 0
    for shard, documents in enumerate(docs_per_shard):
        with open(data / f"shard_{shard}.jsonl", "w") as f:
            for _ in range(documents):
                f.write(json.dumps({"text": " ".join([f"w{doc}"] * (3 + doc * 7 % 11))}) + "\n")
                doc += 1
    vocab = {"<eos>": 0, "<unk>": 1, **{f"w{i}": i + 2 for i in range(doc)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>", unk_token="<unk>")
    fast.save_pretrained(str(root / "tokenizer"))
    return data, root / "tokenizer"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[Path, Path]:
    """Local JSONL shards and a local tokenizer that maps every word to its document."""
    return _write_corpus(tmp_path_factory.mktemp("stream"), (DOCS_PER_SHARD,) * SHARDS)


@pytest.fixture(scope="module")
def uneven_corpus(tmp_path_factory) -> tuple[Path, Path]:
    """A corpus whose first shard holds most documents, so data-parallel parts differ."""
    return _write_corpus(tmp_path_factory.mktemp("uneven"), (60, 10, 10, 10))


def _config(corpus, num_workers: int = 0, batch_size: int = 2, seq_len: int = 8) -> JobConfig:
    data, tokenizer = corpus
    return JobConfig(
        train=TrainConfig(batch_size=batch_size, seq_len=seq_len, compile_model=False),
        data=DataConfig(
            hf_dataset_name=str(data),
            hf_streaming=True,
            tokenizer_path=str(tokenizer),
            num_workers=num_workers,
            pin_memory=False,
        ),
        checkpoint=CheckpointConfig(dir="unused"),
    )


def _runtime() -> RuntimeContext:
    return RuntimeContext(
        rank=0, local_rank=0, world_size=1, device=torch.device("cpu"), device_mesh=None
    )


def _loader(config: JobConfig) -> StreamingDataLoader:
    loader = build_data_pipeline(config, _runtime()).dataloader
    assert isinstance(loader, StreamingDataLoader)
    return loader


def _close(loader) -> None:
    BatchStream(DataPipeline(dataloader=loader)).close()


def _stream_loader(corpus, *, num_workers=0, batch_size=2, **dataset_kwargs):
    """A loader over the corpus; ``dataset_kwargs`` override the dataset's arguments."""
    data, tokenizer = corpus
    kwargs = {"seq_len": 8, "tokenizer_path": str(tokenizer), **dataset_kwargs}
    dataset = StreamingHuggingFaceDataset(str(data), "train", "text", **kwargs)
    config = DataConfig(num_workers=num_workers, pin_memory=False)
    return StreamingDataLoader(dataset, batch_size=batch_size, config=config)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resume_continues_the_stream(corpus, num_workers):
    """After 5 batches, a resumed loader yields the uninterrupted run's batches 5 to 7."""
    if num_workers and not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    config = _config(corpus, num_workers)
    loader, resumed = _loader(config), _loader(config)
    try:
        iterator = iter(loader)
        for _ in range(5):
            next(iterator)
        state = loader.state_dict()
        expected = [next(iterator)["input_ids"] for _ in range(3)]
        resumed.load_state_dict(state)
        resumed_iterator = iter(resumed)
        got = [next(resumed_iterator)["input_ids"] for _ in range(3)]
    finally:
        _close(loader)
        _close(resumed)
    for want, have in zip(expected, got, strict=True):
        assert torch.equal(want, have)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resume_in_a_later_epoch_follows_that_epochs_shuffle(corpus, num_workers):
    """A resume in epoch 1 replays epoch 1's shuffle, also with persistent workers.

    Persistent workers keep their own copy of the dataset, so they reach epoch 1 by
    counting their own passes rather than through ``set_epoch``.
    """
    if num_workers and not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    config = _config(corpus, num_workers)
    loader, resumed = _loader(config), _loader(config)
    try:
        first_epoch = [batch["input_ids"] for batch in loader]
        iterator = iter(loader)
        for _ in range(4):
            next(iterator)
        state = loader.state_dict()
        expected = next(iterator)["input_ids"]
        resumed.load_state_dict(state)
        got = next(iter(resumed))["input_ids"]
    finally:
        _close(loader)
        _close(resumed)
    assert state["epoch"] == 1
    assert torch.equal(got, expected)
    assert not torch.equal(expected, first_epoch[4]), "epoch 1 should be shuffled anew"


def test_checkpoint_restores_the_stream_position(corpus, tmp_path):
    """Save through CheckpointManager mid-stream, resume the way the training entry does."""
    from kempnerforge.checkpoint.manager import CheckpointManager

    config = _config(corpus)
    config.checkpoint.dir = str(tmp_path / "ckpt")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = _loader(config)
    iterator = iter(loader)
    for _ in range(5):
        next(iterator)
    CheckpointManager(config.checkpoint, model, optimizer).save(step=5, dataloader=loader)
    expected = next(iterator)["input_ids"]

    manager = CheckpointManager(config.checkpoint, model, optimizer)
    assert manager.load()[0] == 5
    resumed = _loader(config)
    manager.apply_dataloader_state(resumed)
    assert torch.equal(next(iter(resumed))["input_ids"], expected)


@pytest.mark.parametrize(
    "change",
    [
        {"num_workers": 2},
        {"batch_size": 4},
        {"seq_len": 16},
        {"world_size": 2},
        {"seed": 7},
        {"shuffle_buffer_size": 5},
        {"pack_sequences": True},
        {"tokenizer_path": "another-tokenizer"},
    ],
    ids=[
        "num_workers",
        "batch_size",
        "seq_len",
        "data_parallel_size",
        "seed",
        "shuffle_buffer_size",
        "pack_sequences",
        "tokenizer",
    ],
)
def test_a_changed_stream_is_refused(corpus, change):
    """Replaying a count of batches is exact only on the stream that counted them."""
    loader = _stream_loader(corpus)
    next(iter(loader))
    state = loader.state_dict()
    with pytest.raises(DataPositionError, match="stream changed"):
        _stream_loader(corpus, **change).load_state_dict(state)


def test_map_style_and_stream_positions_do_not_mix(corpus):
    stream = _stream_loader(corpus)
    next(iter(stream))
    dataset = torch.utils.data.TensorDataset(torch.arange(64))
    mapped = StatefulDataLoader(
        dataset,
        batch_size=4,
        sampler=DistributedSampler(dataset, num_replicas=1, rank=0),
        config=DataConfig(num_workers=0, pin_memory=False),
    )
    with pytest.raises(DataPositionError, match="map-style dataset"):
        mapped.load_state_dict(stream.state_dict())
    with pytest.raises(DataPositionError, match="streams its data"):
        _stream_loader(corpus).load_state_dict(mapped.state_dict())


def test_check_state_refuses_what_load_state_dict_refuses_and_moves_nothing(corpus):
    """``CheckpointManager`` checks a position on every rank before any rank applies it.

    So ``check_state`` gives ``load_state_dict``'s verdict without applying the position.
    """
    loader = _stream_loader(corpus)
    iterator = iter(loader)
    for _ in range(3):
        next(iterator)
    state = loader.state_dict()
    fresh = _stream_loader(corpus)
    fresh.check_state(state)
    assert (fresh.state_dict()["epoch"], fresh.state_dict()["batches_yielded"]) == (0, 0)
    with pytest.raises(DataPositionError, match="stream changed"):
        _stream_loader(corpus, seed=7).check_state(state)
    with pytest.raises(DataPositionError, match="streams its data"):
        fresh.check_state({"epoch": 0, "batches_yielded": 3})


def test_a_replay_past_the_end_of_the_stream_starts_the_next_epoch(corpus, caplog, monkeypatch):
    """A position beyond the epoch's last batch resumes at the next epoch instead of raising.

    The ``kempnerforge`` logger stops propagation once configured, so the test lets
    records reach ``caplog`` for its duration.
    """
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    loader = _stream_loader(corpus)
    next_epoch = _stream_loader(corpus)
    next_epoch.load_state_dict({**loader.state_dict(), "epoch": 1})
    expected = next(iter(next_epoch))["input_ids"]

    state = {**loader.state_dict(), "batches_yielded": 10_000}
    with caplog.at_level(logging.WARNING, logger="kempnerforge.data.dataloader"):
        loader.load_state_dict(state)
        got = next(iter(loader))["input_ids"]
    assert "starting epoch 1" in caplog.text
    assert torch.equal(got, expected)
    assert loader.state_dict()["epoch"] == 1


def test_warm_start_from_a_map_style_checkpoint_starts_the_stream_fresh(
    corpus, tmp_path, caplog, monkeypatch
):
    """A ``load_path`` warm start into a streaming run warns instead of refusing.

    The ``kempnerforge`` logger stops propagation once configured, so the test lets
    records reach ``caplog``.
    """
    from kempnerforge.checkpoint.manager import CheckpointManager
    from kempnerforge.training.entry import restore_checkpoint

    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    dataset = torch.utils.data.TensorDataset(torch.arange(64))
    mapped = StatefulDataLoader(
        dataset,
        batch_size=4,
        sampler=DistributedSampler(dataset, num_replicas=1, rank=0),
        config=DataConfig(num_workers=0, pin_memory=False),
    )
    next(iter(mapped))
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    saved = CheckpointConfig(dir=str(tmp_path / "saved"))
    CheckpointManager(saved, model, optimizer).save(step=3, dataloader=mapped)

    config = _config(corpus)
    config.checkpoint = CheckpointConfig(
        dir=str(tmp_path / "new_run"),
        load_path=str(tmp_path / "saved" / "step_3"),
        exclude_from_loading=["optimizer"],
    )
    manager = CheckpointManager(config.checkpoint, model, optimizer)
    assert restore_checkpoint(config, model, None, manager)[0] == 3
    stream = _loader(config)
    with caplog.at_level(logging.WARNING, logger="kempnerforge.checkpoint.manager"):
        manager.apply_dataloader_state(stream)
    assert "starts at its first sample" in caplog.text and "streams its data" in caplog.text
    assert torch.equal(next(iter(stream))["input_ids"], next(iter(_loader(config)))["input_ids"])


def test_a_checkpoint_without_a_stream_position_warns_that_the_stream_restarts(
    corpus, tmp_path, caplog, monkeypatch
):
    """A streaming checkpoint saved before positions were recorded holds none.

    Its resume starts the stream at the first document and says so; a run that
    restored no checkpoint says nothing. The ``kempnerforge`` logger stops
    propagation once configured, so the test lets records reach ``caplog``.
    """
    from kempnerforge.checkpoint.manager import CheckpointManager
    from kempnerforge.training.entry import restore_checkpoint

    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    config = _config(corpus)
    config.checkpoint = CheckpointConfig(dir=str(tmp_path / "ckpt"))
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    CheckpointManager(config.checkpoint, model, optimizer).save(step=3, dataloader=None)

    with caplog.at_level(logging.WARNING, logger="kempnerforge.checkpoint.manager"):
        CheckpointManager(config.checkpoint, model, optimizer).apply_dataloader_state(
            _loader(config)
        )
    assert "holds no data position" not in caplog.text

    manager = CheckpointManager(config.checkpoint, model, optimizer)
    assert restore_checkpoint(config, model, None, manager)[0] == 3
    stream = _loader(config)
    with caplog.at_level(logging.WARNING, logger="kempnerforge.checkpoint.manager"):
        manager.apply_dataloader_state(stream)
    assert "holds no data position, so the stream restarts at its first document" in caplog.text
    assert torch.equal(next(iter(stream))["input_ids"], next(iter(_loader(config)))["input_ids"])


def _copy(corpus, directory: Path, shards=ALL_SHARDS) -> tuple[Path, Path]:
    """Copy some of the corpus shards into ``directory``; same tokenizer."""
    data, tokenizer = corpus
    directory.mkdir(parents=True)
    for shard in shards:
        shutil.copy(data / f"shard_{shard}.jsonl", directory)
    return directory, tokenizer


def test_an_added_data_file_is_refused(corpus, tmp_path):
    """A new shard changes every rank's stream, so the saved count no longer applies."""
    copied = _copy(corpus, tmp_path / "data")
    loader = _stream_loader(copied)
    next(iter(loader))
    state = loader.state_dict()
    shutil.copy(copied[0] / "shard_0.jsonl", copied[0] / "shard_9.jsonl")
    with pytest.raises(DataPositionError, match="source data files differ"):
        _stream_loader(copied).load_state_dict(state)


def test_the_same_files_in_another_directory_are_accepted(corpus, tmp_path):
    """Moving the data, say off scratch storage, keeps the stream and its position."""
    first = _copy(corpus, tmp_path / "scratch")
    loader = _stream_loader(first)
    iterator = iter(loader)
    for _ in range(5):
        next(iterator)
    state = loader.state_dict()
    expected = next(iterator)["input_ids"]
    moved = _copy(corpus, tmp_path / "lab" / "data")
    resumed = _stream_loader(moved)
    resumed.load_state_dict(state)
    assert torch.equal(next(iter(resumed))["input_ids"], expected)


def test_a_later_pass_on_another_ranks_position_warns(corpus, caplog, monkeypatch):
    """Past the first pass, another rank's count may not fit this rank; say so.

    A rank's own count, which an auto-resume restores, fits in every pass. In the
    first pass rank 0's count fits every rank still in that pass, and a rank that
    already finished it runs off the end of its part in the replay, which warns on
    its own (next test). A count saved before the rank was recorded may be another
    rank's. Rank 0 takes its own count, so the warning must pass the training log's
    rank-0 filter on the other ranks. The ``kempnerforge`` logger stops propagation
    once configured, so the test lets records reach ``caplog``.
    """
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    state = _stream_loader(corpus, world_size=2).state_dict()
    assert state["data_parallel_rank"] == 0
    later = {**state, "epoch": 1}
    with caplog.at_level(logging.WARNING, logger="kempnerforge.data.dataloader"):
        _stream_loader(corpus, rank=1, world_size=2).load_state_dict(state)
        _stream_loader(corpus, world_size=2).load_state_dict(later)
        assert "Resuming the stream in pass" not in caplog.text
        _stream_loader(corpus, rank=1, world_size=2).load_state_dict(later)
        assert "pass 1 on 2 data-parallel ranks from data-parallel rank 0's position" in (
            caplog.text
        )
        warned = [r for r in caplog.records if "Resuming the stream in pass" in r.getMessage()]
        monkeypatch.setenv("RANK", "1")
        assert _RankFilter(rank=0).filter(warned[0]), "the training log would drop it on rank 1"
        caplog.clear()
        unrecorded = {k: v for k, v in later.items() if k != "data_parallel_rank"}
        _stream_loader(corpus, world_size=2).load_state_dict(unrecorded)
    assert "from a position that predates the saved rank" in caplog.text


def test_a_replay_past_this_ranks_part_warns_in_its_own_log(corpus, caplog, monkeypatch):
    """Rank 0's count can outrun a shorter part of the first pass; that rank says so.

    In a load that gives every rank rank 0's count, a rank whose part of the stream is
    shorter may have finished the pass while rank 0 has not, so its replay runs off
    the end and it reads the next pass from its start again. Rank 0 replays its own
    count, so the warning must pass the training log's rank-0 filter on the other
    ranks. The ``kempnerforge`` logger stops propagation once configured, so the test
    lets records reach ``caplog``.
    """
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    part = sum(1 for _ in _stream_loader(corpus, rank=1, world_size=2))
    state = {**_stream_loader(corpus, world_size=2).state_dict(), "batches_yielded": part + 2}
    resumed = _stream_loader(corpus, rank=1, world_size=2)
    with caplog.at_level(logging.WARNING, logger="kempnerforge.data.dataloader"):
        resumed.load_state_dict(state)
        next(iter(resumed))
    ended = [r for r in caplog.records if "The stream ended" in r.getMessage()]
    assert len(ended) == 1
    assert f"after {part} of {part + 2} replayed batches of epoch 0" in ended[0].getMessage()
    monkeypatch.setenv("RANK", "1")
    assert _RankFilter(rank=0).filter(ended[0]), "the training log would drop it on rank 1"
    assert resumed.state_dict()["epoch"] == 1


def test_restarting_a_pass_midway_with_workers_reads_the_same_epoch(corpus, tmp_path):
    """A pass restarted after one worker finished its part still reads this epoch.

    With 3 shards and 2 workers, worker 1 reads one shard and finishes early; its
    persistent copy of the dataset then moves to the next epoch.
    """
    if not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    uneven = _copy(corpus, tmp_path / "three", shards=(0, 1, 2))
    reference = _stream_loader(uneven, num_workers=2)
    restarted = _stream_loader(uneven, num_workers=2)
    try:
        expected = [batch["input_ids"] for batch in reference]
        iterator = iter(restarted)
        for _ in range(len(expected) - 2):
            next(iterator)
        got = next(iter(restarted))["input_ids"]
    finally:
        _close(reference)
        _close(restarted)
    assert torch.equal(got, expected[-2])


LIVE_STREAM_AT_EXIT = """
import sys

from kempnerforge.config.schema import DataConfig
from kempnerforge.data.dataloader import StreamingDataLoader
from kempnerforge.data.dataset import StreamingHuggingFaceDataset

if __name__ == "__main__":
    data, tokenizer = sys.argv[1], sys.argv[2]
    dataset = StreamingHuggingFaceDataset(
        data, "train", "text", seq_len=8, tokenizer_path=tokenizer
    )
    loader = StreamingDataLoader(
        dataset, batch_size=2, config=DataConfig(num_workers=2, pin_memory=False)
    )
    batches = iter(loader)
    next(batches)
    print("served a batch", flush=True)
"""


def test_a_process_with_a_live_stream_exits(corpus, tmp_path):
    """A script whose streaming iterator, with workers, is alive at exit still exits.

    The stream's workers ignore SIGTERM as the map-style loader's do, so
    ``multiprocessing``'s exit hook would wait on them forever.
    """
    if not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    script = tmp_path / "live_stream.py"
    script.write_text(LIVE_STREAM_AT_EXIT)
    data, tokenizer = corpus

    result = run_to_exit(script, str(data), str(tokenizer))

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.splitlines()[-1:] == ["served a batch"]


def test_close_stops_the_workers_and_keeps_the_stream_position(corpus):
    """After ``close``, the next pass replays to the position and goes on as before."""
    if not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    closed = _stream_loader(corpus, num_workers=2)
    uninterrupted = _stream_loader(corpus, num_workers=2)
    try:
        batches = iter(closed)
        for _ in range(3):
            next(batches)
        state = closed.state_dict()
        workers = list(closed._dataloader._iterator._workers)

        closed.close()

        assert not any(worker.is_alive() for worker in workers)
        assert closed.state_dict() == state
        got = [next(iter(closed))["input_ids"], next(closed)["input_ids"]]
        reference = iter(uninterrupted)
        for _ in range(3):
            next(reference)
        expected = [next(reference)["input_ids"] for _ in range(2)]
    finally:
        closed.close()
        uninterrupted.close()
    for want, have in zip(expected, got, strict=True):
        assert torch.equal(want, have)


def _base_seed(loader: StreamingDataLoader) -> int:
    """The base seed of the loader's iterator; worker ``k`` takes ``base_seed + k``."""
    iterator = loader._dataloader._iterator if loader._num_workers else loader._iterator
    assert iterator is not None
    return iterator._base_seed


def _resume_parity(loader, resumed, saved_after: int, then: int):
    """Batches of ``loader`` from ``saved_after`` on, and of ``resumed`` from its saved state.

    ``loader`` has taken ``saved_after`` batches of its current pass. ``resumed``
    loads the state and the CPU RNG saved there. Returns each one's ``input_ids``,
    CPU RNG and base seed after ``then`` more batches.
    """
    rng, state = torch.random.get_rng_state(), loader.state_dict()
    assert state["batches_yielded"] == saved_after
    expected = [next(loader)["input_ids"] for _ in range(then)]
    uninterrupted = (expected, torch.random.get_rng_state(), _base_seed(loader))
    torch.random.set_rng_state(rng)
    resumed.load_state_dict(state)
    batches = iter(resumed)
    got = [next(batches)["input_ids"] for _ in range(then)]
    return uninterrupted, (got, torch.random.get_rng_state(), _base_seed(resumed))


def _assert_same_run(uninterrupted, resumed) -> None:
    (expected, expected_rng, expected_seed), (got, got_rng, got_seed) = uninterrupted, resumed
    assert torch.equal(got_rng, expected_rng), "the resume left the CPU RNG elsewhere"
    assert got_seed == expected_seed
    for want, have in zip(expected, got, strict=True):
        assert torch.equal(want, have)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_a_resumed_stream_continues_the_cpu_rng_and_the_base_seed(corpus, num_workers):
    """Resumed after 5 batches, the stream draws nothing new from the CPU RNG.

    Its iterator takes the uninterrupted run's base seed, from which its workers
    take theirs.
    """
    if num_workers and not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    torch.manual_seed(0)
    loader = _stream_loader(corpus, num_workers=num_workers)
    resumed = _stream_loader(corpus, num_workers=num_workers)
    try:
        iterator = iter(loader)
        for _ in range(5):
            next(iterator)
        uninterrupted, resume = _resume_parity(loader, resumed, 5, 3)
    finally:
        _close(loader)
        _close(resumed)
    _assert_same_run(uninterrupted, resume)


def test_a_pass_restarted_midway_draws_a_new_base_seed_that_a_resume_keeps(corpus):
    """New workers for a pass restarted midway draw their base seed from the CPU RNG.

    A resume saved after the restart gives its workers that seed.
    """
    if not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    torch.manual_seed(0)
    loader = _stream_loader(corpus, num_workers=2)
    resumed = _stream_loader(corpus, num_workers=2)
    try:
        iterator = iter(loader)
        for _ in range(3):
            next(iterator)
        first = _base_seed(loader)
        generator = torch.Generator()
        generator.set_state(torch.random.get_rng_state())
        iterator = iter(loader)
        next(iterator)
        drawn = torch.empty((), dtype=torch.int64).random_(generator=generator).item()
        assert _base_seed(loader) == drawn != first
        uninterrupted, resume = _resume_parity(loader, resumed, 4, 2)
    finally:
        _close(loader)
        _close(resumed)
    _assert_same_run(uninterrupted, resume)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_a_stream_saved_after_a_reset_resumes_as_the_run_restarts_the_pass(corpus, num_workers):
    """Saved after ``BatchStream.reset``, the resume replays as the run's restarted pass does.

    The restarted pass starts a new iterator, which draws a new base seed from the
    CPU RNG, in the run and in the resume. The reset leaves the live iterator, and
    its workers, to the next ``iter()``.
    """
    if num_workers and not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    torch.manual_seed(0)
    run = BatchStream(DataPipeline(dataloader=_stream_loader(corpus, num_workers=num_workers)))
    resumed = BatchStream(DataPipeline(dataloader=_stream_loader(corpus, num_workers=num_workers)))
    try:
        resumed.reset()
        for _ in range(4):
            run.next_batch()
        live = run.dataloader._iterator
        run.reset()
        assert run.dataloader._iterator is live is not None
        rng, state = torch.random.get_rng_state(), run.dataloader.state_dict()
        expected = [run.next_batch()["input_ids"] for _ in range(3)]
        uninterrupted = (expected, torch.random.get_rng_state(), _base_seed(run.dataloader))
        torch.random.set_rng_state(rng)
        resumed.dataloader.load_state_dict(state)
        got = [resumed.next_batch()["input_ids"] for _ in range(3)]
        resume = (got, torch.random.get_rng_state(), _base_seed(resumed.dataloader))
    finally:
        run.close()
        resumed.close()
    assert state["batches_yielded"] == 4
    assert state["base_seed_rng_state"] is None
    _assert_same_run(uninterrupted, resume)


def test_a_reset_whose_pass_runs_to_its_end_keeps_the_workers_base_seed(corpus):
    """A reset stream whose live pass runs on to its end has no pass left to restart.

    With persistent workers the next pass resets their iterator, which keeps its
    base seed, so the state stays saved and a resume keeps the seed too.
    """
    if not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    torch.manual_seed(0)
    loader = _stream_loader(corpus, num_workers=2)
    resumed = _stream_loader(corpus, num_workers=2)
    try:
        next(iter(loader))
        loader.restart()
        with pytest.raises(StopIteration):
            while True:
                next(loader)
        rng, state = torch.random.get_rng_state(), loader.state_dict()
        batches = iter(loader)
        expected = [next(batches)["input_ids"] for _ in range(2)]
        uninterrupted = (expected, torch.random.get_rng_state(), _base_seed(loader))
        torch.random.set_rng_state(rng)
        resumed.load_state_dict(state)
        batches = iter(resumed)
        got = [next(batches)["input_ids"] for _ in range(2)]
        resume = (got, torch.random.get_rng_state(), _base_seed(resumed))
    finally:
        _close(loader)
        _close(resumed)
    assert state["epoch"] == 1 and state["base_seed_rng_state"] is not None
    _assert_same_run(uninterrupted, resume)


def test_a_load_into_a_running_stream_starts_new_workers_from_the_loaded_state(corpus):
    """Workers that ran before ``load_state_dict`` stop; the next pass replays with new ones."""
    if not worker_sockets_fit():
        pytest.skip("the temp directory path is too long for DataLoader worker sockets")
    loader = _stream_loader(corpus, num_workers=2)
    running = _stream_loader(corpus, num_workers=2)
    try:
        torch.manual_seed(1)
        iterator = iter(running)
        for _ in range(6):
            next(iterator)
        workers = list(running._dataloader._iterator._workers)
        torch.manual_seed(0)
        iterator = iter(loader)
        for _ in range(4):
            next(iterator)
        uninterrupted, resume = _resume_parity(loader, running, 4, 3)
    finally:
        _close(loader)
        _close(running)
    assert not any(worker.is_alive() for worker in workers)
    _assert_same_run(uninterrupted, resume)


@pytest.mark.parametrize(("batches", "warns"), [(5, True), (0, False)])
def test_a_stream_position_without_the_saved_state_resumes_as_before(
    corpus, batches, warns, caplog, monkeypatch
):
    """A position saved without the state draws a new base seed from the CPU RNG.

    It warns when the stream had started. The ``kempnerforge`` logger stops
    propagation once configured, so the test lets records reach ``caplog``.
    """
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    loader = _stream_loader(corpus)
    iterator = iter(loader)
    for _ in range(batches):
        next(iterator)
    state = {k: v for k, v in loader.state_dict().items() if k != "base_seed_rng_state"}
    expected = next(iterator)["input_ids"]
    resumed = _stream_loader(corpus)
    with caplog.at_level(logging.WARNING, logger="kempnerforge.data.dataloader"):
        resumed.load_state_dict(state)
    before = torch.random.get_rng_state()
    got = next(iter(resumed))["input_ids"]
    assert not torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(got, expected)
    assert ("predates the saved base seed" in caplog.text) == warns


def test_the_replay_logs_its_progress(corpus, caplog, monkeypatch):
    """The replay reports how far it got, at what rate and how long is left.

    The ``kempnerforge`` logger stops propagation once configured, so the test lets
    records reach ``caplog``.
    """
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    loader = _stream_loader(corpus)
    iterator = iter(loader)
    for _ in range(10):
        next(iterator)
    resumed = _stream_loader(corpus)
    resumed.load_state_dict(loader.state_dict())
    with caplog.at_level(logging.INFO, logger="kempnerforge.data.dataloader"):
        next(iter(resumed))
    assert "Replaying 10 batches (160 tokens)" in caplog.text
    assert "Replayed 10 of 10 batches" in caplog.text and "tokens/s" in caplog.text


DP2_TP2 = [(0, 2), (0, 2), (1, 2), (1, 2)]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _draw() -> list[float]:
    """One draw from each generator a checkpoint captures on CPU."""
    return [random.random(), float(np.random.random()), float(torch.rand(()))]


class _Collect(logging.Handler):
    """Keeps the message of every warning it handles."""

    def __init__(self, messages: list[str]) -> None:
        super().__init__(logging.WARNING)
        self.messages = messages

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _part_loader(data: str, tokenizer: str, dp: tuple[int, int]) -> StreamingDataLoader:
    """A loader over data-parallel part ``dp[0]`` of ``dp[1]`` of the stream."""
    dataset = StreamingHuggingFaceDataset(
        data, "train", "text", seq_len=8, tokenizer_path=tokenizer, rank=dp[0], world_size=dp[1]
    )
    return StreamingDataLoader(
        dataset, batch_size=2, config=DataConfig(num_workers=0, pin_memory=False)
    )


def _next_batches(stream: BatchStream, count: int = 3) -> list:
    return [stream.next_batch()["input_ids"].tolist() for _ in range(count)]


def _stream_manager(root: str, dp: tuple[int, int], lenient: bool):
    """A manager that records this rank's data-parallel coordinates, as the training entry's."""
    from kempnerforge.checkpoint.manager import CheckpointManager

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = CheckpointConfig(dir=root, ignore_rank_state_mismatch=lenient)
    return CheckpointManager(config, model, optimizer, dp_info=dp)


def _save_stream_job(
    rank: int, world: int, port: int, root: str, data: str, tokenizer: str, steps: int, out: str
) -> None:
    """One rank of a 4-rank gloo job, laid out as dp 2 x tp 2, that saves its stream position.

    Global rank ``r`` reads data-parallel part ``r // 2`` of 2 and takes ``steps``
    batches, moving to the next pass where its part ends, as training does. After the
    save, rank 0 writes every rank's saved pass and count and its next 3 batches, which
    an exact resume of that data-parallel rank must read next.
    """
    with gloo_group(rank, world, port, timeout=timedelta(seconds=120)):
        loader = _part_loader(data, tokenizer, DP2_TP2[rank])
        stream = BatchStream(DataPipeline(dataloader=loader))
        for _ in range(steps):
            stream.next_batch()
        _stream_manager(root, DP2_TP2[rank], lenient=False).save(step=1, dataloader=loader)
        state = loader.state_dict()
        report = {"at": [state["epoch"], state["batches_yielded"]], "next": _next_batches(stream)}
        gathered: list = [None] * world
        dist.all_gather_object(gathered, report)
        if rank == 0:
            Path(out).write_text(json.dumps(gathered))


def _resume_stream_job(
    rank: int,
    world: int,
    port: int,
    root: str,
    views: list[str],
    tokenizer: str,
    layout: list[tuple[int, int]],
    lenient: bool,
    out: str,
    warm: bool = False,
) -> None:
    """One rank of a gloo job that auto-resumes the stream checkpoint, as the training entry does.

    Rank ``r`` reads data-parallel part ``layout[r]`` of the stream in ``views[r]``,
    loads with ``exact_resume=True`` and ``lenient`` as
    ``checkpoint.ignore_rank_state_mismatch``, or with ``warm`` as a warm start
    (``warm_start=True``), then applies the position with ``apply_dataloader_state``.
    It seeds its generators with ``555 + r`` just before the load. Rank 0 writes every
    rank's report: the error the resume raised or the next 3 batches, the next random
    draw and the one the fresh seeds give, and the warnings.
    """
    with gloo_group(rank, world, port, timeout=timedelta(seconds=120)):
        warnings: list[str] = []
        dp = (layout[rank][0], layout[rank][1])
        manager = _stream_manager(root, dp, lenient)
        loader = _part_loader(views[rank], tokenizer, dp)
        logging.getLogger("kempnerforge").addHandler(_Collect(warnings))
        _seed_all(555 + rank)
        report: dict = {"fresh": _draw(), "error": None, "next": []}
        _seed_all(555 + rank)
        try:
            if warm:
                manager.load(warm_start=True)
            else:
                manager.load(exact_resume=True)
            manager.apply_dataloader_state(loader)
        except Exception as e:
            report["error"] = f"{type(e).__name__}: {e}"
        report["draw"] = _draw()
        if report["error"] is None:
            report["next"] = _next_batches(BatchStream(DataPipeline(dataloader=loader)))
        report["warnings"] = warnings
        gathered: list = [None] * world
        dist.all_gather_object(gathered, report)
        if rank == 0:
            Path(out).write_text(json.dumps(gathered))


@pytest.fixture(scope="module")
def stream_checkpoint(uneven_corpus, tmp_path_factory) -> tuple[str, list[dict]]:
    """A stream position saved by 4 gloo ranks as dp 2 x tp 2, and the save job's reports.

    The ranks take a few batches more than the longer data-parallel part holds, so each
    part is in its second pass, at a count that differs between the parts.
    """
    data, tokenizer = (str(p) for p in uneven_corpus)
    parts = [sum(1 for _ in _part_loader(data, tokenizer, (d, 2))) for d in range(2)]
    assert parts[0] != parts[1], parts
    root = tmp_path_factory.mktemp("stream_checkpoint")
    out = str(root / "saved.json")
    spawn(_save_stream_job, 4, str(root / "ckpt"), data, tokenizer, max(parts) + 3, out)
    return str(root / "ckpt"), json.loads(Path(out).read_text())


def _resume_streams(tmp_path, root, views, tokenizer, layout, lenient, warm=False) -> list[dict]:
    out = str(tmp_path / "resumed.json")
    spawn(_resume_stream_job, len(layout), root, views, tokenizer, layout, lenient, out, warm)
    return json.loads(Path(out).read_text())


def _fresh_batches(data: str, tokenizer: str, dp: tuple[int, int]) -> list:
    return _next_batches(BatchStream(DataPipeline(dataloader=_part_loader(data, tokenizer, dp))))


def _stale_copy(corpus, directory: Path) -> str:
    """A copy of the corpus in which one data file has an extra document, so its size differs."""
    data, _ = _copy(corpus, directory)
    with open(data / "shard_3.jsonl", "a") as f:
        f.write(json.dumps({"text": "w0 w0 w0"}) + "\n")
    return str(data)


def test_a_flag_resume_on_the_saved_data_parallel_size_keeps_each_ranks_stream_position(
    uneven_corpus, stream_checkpoint, tmp_path
):
    """Saved as dp 2 x tp 2 on 4 ranks, auto-resumed as dp 2 on 2 with the flag.

    The two data-parallel parts differ in length, so their saved positions differ. Each
    rank takes the one its own data-parallel rank saved and reads what the uninterrupted
    run read next; the random streams restart, and no rank starts its data over.
    """
    data, tokenizer = (str(p) for p in uneven_corpus)
    root, saved = stream_checkpoint
    assert saved[0]["at"] != saved[2]["at"], "the saved positions should differ"
    resumed = _resume_streams(tmp_path, root, [data, data], tokenizer, [(0, 2), (1, 2)], True)
    for rank, report in enumerate(resumed):
        assert report["error"] is None, report["error"]
        assert report["next"] == saved[2 * rank]["next"], f"rank {rank} took another position"
        assert report["draw"] == report["fresh"], f"rank {rank} restored a saved RNG stream"
        kept = [w for w in report["warnings"] if "Kept: the saved data position" in w]
        assert kept, report["warnings"]
        assert not [w for w in report["warnings"] if "does not carry over" in w]


def test_a_flag_resume_on_another_data_parallel_size_restarts_the_stream_on_every_rank(
    uneven_corpus, stream_checkpoint, tmp_path
):
    """Resumed as dp 4 on 4 ranks, the positions saved on dp 2 have no counterpart.

    Every rank warns with the reason and starts its part of the stream at its first
    document, and the random streams restart.
    """
    data, tokenizer = (str(p) for p in uneven_corpus)
    root, _ = stream_checkpoint
    layout = [(r, 4) for r in range(4)]
    resumed = _resume_streams(tmp_path, root, [data] * 4, tokenizer, layout, True)
    for rank, report in enumerate(resumed):
        assert report["error"] is None, report["error"]
        assert report["next"] == _fresh_batches(data, tokenizer, layout[rank])
        assert report["draw"] == report["fresh"], f"rank {rank} restored a saved RNG stream"
        restarted = [w for w in report["warnings"] if "does not carry over" in w]
        assert restarted and "data_parallel_size: saved 2, now 4" in restarted[0], restarted


def test_a_flag_resume_restarts_every_rank_when_one_rank_sees_a_file_at_another_size(
    uneven_corpus, stream_checkpoint, tmp_path
):
    """Rank 1 reads a copy with one data file grown, so only rank 1 cannot place its position.

    The ranks agree before applying it: both start their parts of the stream at the
    first document, rank 0's warning names rank 1 and why, and the random streams
    restart. Before the agreement, rank 0 kept its position and rank 1 restarted.
    """
    data, tokenizer = (str(p) for p in uneven_corpus)
    root, _ = stream_checkpoint
    views = [data, _stale_copy(uneven_corpus, tmp_path / "stale")]
    resumed = _resume_streams(tmp_path, root, views, tokenizer, [(0, 2), (1, 2)], True)
    for rank, report in enumerate(resumed):
        assert report["error"] is None, report["error"]
        assert report["next"] == _fresh_batches(views[rank], tokenizer, (rank, 2)), f"rank {rank}"
        assert report["draw"] == report["fresh"], f"rank {rank} restored a saved RNG stream"
        restarted = [w for w in report["warnings"] if "does not carry over" in w]
        assert restarted and "source data files differ" in restarted[0], report["warnings"]
        assert "ranks [1]" in restarted[0], restarted[0]


def test_a_warm_start_restarts_every_rank_when_one_rank_sees_a_file_at_another_size(
    uneven_corpus, stream_checkpoint, tmp_path
):
    """A warm start on 2 ranks gives both rank 0's position, which only rank 1 cannot place.

    Rank 1 reads a copy with one data file grown. The ranks agree before applying the
    position: both start their parts of the stream at the first document, and the
    warning names rank 1. A warm start restores rank 0's RNG state on every rank, so the
    random streams are not checked.
    """
    data, tokenizer = (str(p) for p in uneven_corpus)
    root, _ = stream_checkpoint
    views = [data, _stale_copy(uneven_corpus, tmp_path / "stale")]
    resumed = _resume_streams(tmp_path, root, views, tokenizer, [(0, 2), (1, 2)], False, True)
    for rank, report in enumerate(resumed):
        assert report["error"] is None, report["error"]
        assert report["next"] == _fresh_batches(views[rank], tokenizer, (rank, 2)), f"rank {rank}"
        restarted = [w for w in report["warnings"] if "warm start's data position" in w]
        assert restarted and "on ranks [1]" in restarted[0], report["warnings"]


def test_without_the_flag_a_file_one_rank_sees_at_another_size_stops_every_rank(
    uneven_corpus, stream_checkpoint, tmp_path
):
    """On the layout that saved it, a position that rank 3 alone cannot place stops every rank.

    Every rank raises ``DataPositionError`` together, naming rank 3's reason, instead
    of rank 3 raising alone while the others go on into the next collective.
    """
    data, tokenizer = (str(p) for p in uneven_corpus)
    root, _ = stream_checkpoint
    views = [data, data, data, _stale_copy(uneven_corpus, tmp_path / "stale")]
    resumed = _resume_streams(tmp_path, root, views, tokenizer, DP2_TP2, False)
    for rank, report in enumerate(resumed):
        assert report["error"] is not None, f"rank {rank} resumed past rank 3's refusal"
        assert report["error"].startswith("DataPositionError: "), report["error"]
        assert "source data files differ" in report["error"], report["error"]
        assert report["next"] == [], f"rank {rank} read data"


def test_a_check_that_fails_on_one_rank_another_way_stops_every_rank(
    uneven_corpus, stream_checkpoint, tmp_path
):
    """Rank 1's data directory is missing, so its check fails with an error of its own.

    The ranks gather that verdict too: rank 1 raises its error and rank 0 a
    ``RuntimeError`` naming rank 1, instead of waiting for it in the next collective.
    """
    data, tokenizer = (str(p) for p in uneven_corpus)
    root, _ = stream_checkpoint
    views = [data, str(tmp_path / "missing")]
    resumed = _resume_streams(tmp_path, root, views, tokenizer, [(0, 2), (1, 2)], True)
    assert resumed[1]["error"] is not None, "rank 1 resumed from a missing directory"
    assert not resumed[1]["error"].startswith("RuntimeError: Checking"), resumed[1]["error"]
    assert resumed[0]["error"] is not None, "rank 0 resumed without rank 1"
    assert resumed[0]["error"].startswith(
        "RuntimeError: Checking the saved data position failed on ranks [1]"
    ), resumed[0]["error"]
    assert [r["next"] for r in resumed] == [[], []]
