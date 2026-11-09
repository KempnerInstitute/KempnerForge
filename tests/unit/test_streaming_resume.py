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
from pathlib import Path

import pytest
import torch

from kempnerforge.config.schema import CheckpointConfig, DataConfig, JobConfig, TrainConfig
from kempnerforge.data.dataloader import DataPositionError, StatefulDataLoader, StreamingDataLoader
from kempnerforge.data.dataset import StreamingHuggingFaceDataset
from kempnerforge.data.sampler import DistributedSampler
from kempnerforge.training.data_pipeline import DataPipeline, build_data_pipeline
from kempnerforge.training.loop import BatchStream
from kempnerforge.training.runtime import RuntimeContext

SHARDS, DOCS_PER_SHARD = 4, 50


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[Path, Path]:
    """Local JSONL shards and a local tokenizer that maps every word to its document."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("stream")
    data = root / "corpus"
    data.mkdir()
    doc = 0
    for shard in range(SHARDS):
        with open(data / f"shard_{shard}.jsonl", "w") as f:
            for _ in range(DOCS_PER_SHARD):
                f.write(json.dumps({"text": " ".join([f"w{doc}"] * (3 + doc * 7 % 11))}) + "\n")
                doc += 1
    vocab = {"<eos>": 0, "<unk>": 1, **{f"w{i}": i + 2 for i in range(doc)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>", unk_token="<unk>")
    fast.save_pretrained(str(root / "tokenizer"))
    return data, root / "tokenizer"


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


def _worker_sockets_fit() -> bool:
    """Whether DataLoader workers can hand batches to this process here.

    Workers send tensors over AF_UNIX sockets created in the multiprocessing temp
    directory, and a socket path may not exceed 107 bytes.
    """
    import multiprocessing.util

    return len(multiprocessing.util.get_temp_dir()) + len("/listener-12345678") <= 107


def _stream_loader(corpus, *, world_size=1, num_workers=0, batch_size=2, seq_len=8):
    data, tokenizer = corpus
    dataset = StreamingHuggingFaceDataset(
        str(data), "train", "text", seq_len, str(tokenizer), world_size=world_size
    )
    config = DataConfig(num_workers=num_workers, pin_memory=False)
    return StreamingDataLoader(dataset, batch_size=batch_size, config=config)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resume_continues_the_stream(corpus, num_workers):
    """After 5 batches, a resumed loader yields the uninterrupted run's batches 5 to 7."""
    if num_workers and not _worker_sockets_fit():
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


def test_resume_in_a_later_epoch_follows_that_epochs_shuffle(corpus):
    config = _config(corpus)
    loader = _loader(config)
    first_epoch = [batch["input_ids"] for batch in loader]
    iterator = iter(loader)
    for _ in range(4):
        next(iterator)
    state = loader.state_dict()
    expected = next(iterator)["input_ids"]
    assert state["epoch"] == 1

    resumed = _loader(config)
    resumed.load_state_dict(state)
    got = next(iter(resumed))["input_ids"]
    assert torch.equal(got, expected)
    assert not torch.equal(got, first_epoch[4]), "epoch 1 should be shuffled anew"


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
    [{"num_workers": 2}, {"batch_size": 4}, {"seq_len": 16}, {"world_size": 2}],
    ids=["num_workers", "batch_size", "seq_len", "data_parallel_size"],
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
