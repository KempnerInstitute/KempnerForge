"""DataComp streaming reader tests on small webdataset shards."""

from __future__ import annotations

import io
import itertools
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest
import torch
from image_text.datacomp import DataCompStream, shard_names, tar_samples
from image_text.document import Document, Turn
from image_text.render import ChatRenderer, RenderedStream
from image_text.stream import StreamLoader
from PIL import Image

PROMPT = "Describe the image."


def _jpeg(width: int = 6) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, 4), (120, 40, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _sample(key: str, caption: Any = None, **members: bytes) -> list[tuple[str, bytes]]:
    """A sample's members: a jpg and a json caption unless ``members`` overrides them."""
    files = {"jpg": _jpeg(), "json": json.dumps({"caption": caption or f"cap {key}"}).encode()}
    files.update(members)
    return [(f"{key}.{extension}", data) for extension, data in files.items()]


def _write_shard(path: Path, members: list[tuple[str, bytes]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _shards(root: Path, shards: int, per_shard: int) -> Path:
    """``shards`` shards of ``per_shard`` samples, captioned ``cap <shard>-<sample>``."""
    for s in range(shards):
        members = [m for i in range(per_shard) for m in _sample(f"{s:02d}-{i}")]
        _write_shard(root / f"{s:05d}.tar", members)
    return root


def _captions(documents: Any, count: int, stream: Any = None) -> list[str]:
    """The next ``count`` captions, each recorded as trained on ``stream`` when given."""
    captions = []
    for position, doc in itertools.islice(documents, count):
        if stream is not None:
            stream.advance(position)
        captions.append(doc.turns[0].answers[0])
    return captions


def test_tar_samples_group_members_by_key(tmp_path: Path) -> None:
    shard = _write_shard(
        tmp_path / "a.tar",
        [("x/1.jpg", b"i"), ("x/1.json", b"j"), ("x/2.JPG", b"k"), ("1.txt", b"t")],
    )
    assert list(tar_samples(shard)) == [
        ("x/1", {"jpg": b"i", "json": b"j"}),
        ("x/2", {"jpg": b"k"}),
        ("1", {"txt": b"t"}),
    ]
    assert list(tar_samples(shard, skip=2)) == [("1", {"txt": b"t"})]
    assert list(tar_samples(shard, skip=3)) == []


def test_only_tar_files_are_shards(tmp_path: Path) -> None:
    _shards(tmp_path, 2, 1)
    (tmp_path / "00000.parquet").write_bytes(b"")
    (tmp_path / "_tmp.tar").mkdir()
    assert shard_names(tmp_path) == ["00000.tar", "00001.tar"]


def test_reads_captions_with_the_prompt(tmp_path: Path) -> None:
    stream = DataCompStream(_shards(tmp_path, 1, 2), PROMPT)
    position, doc = next(iter(stream))
    assert position == [0, 0, 0, 1]
    assert doc.turns[0].user == PROMPT
    assert doc.turns[0].answers == ("cap 00-0",)
    assert doc.images[0].size == (6, 4) and doc.images[0].mode == "RGB"


def test_each_epoch_reads_every_sample_once_then_goes_on(tmp_path: Path) -> None:
    stream = DataCompStream(_shards(tmp_path, 4, 3), PROMPT, seed=1)
    captions = _captions(iter(stream), 24)
    first, second = captions[:12], captions[12:]
    assert (
        sorted(first)
        == sorted(second)
        == sorted(f"cap {s:02d}-{i}" for s in range(4) for i in range(3))
    )
    assert first != second


def test_ranks_and_workers_split_the_shards(tmp_path: Path) -> None:
    _shards(tmp_path, 12, 1)
    dealt = [
        DataCompStream(tmp_path, PROMPT, rank=r, world_size=2, num_workers=3).worker_shards(5, w)
        for r in range(2)
        for w in range(3)
    ]
    assert all(len(shards) == 2 for shards in dealt)
    assert sorted(itertools.chain(*dealt)) == shard_names(tmp_path)


def test_unusable_samples_and_a_broken_shard_are_passed_over(tmp_path: Path) -> None:
    members = (
        _sample("0", jpg=b"not an image")
        + [("1.jpg", _jpeg())]
        + _sample("2", json=b"[1, 2]")
        + _sample("3", caption="  ")
        + _sample("4")
    )
    _write_shard(tmp_path / "00000.tar", members)
    whole = _write_shard(tmp_path / "00001.tar", _sample("5") + _sample("6")).read_bytes()
    (tmp_path / "00001.tar").write_bytes(whole[:700])
    stream = DataCompStream(tmp_path, PROMPT)
    stream.worker_shards = lambda epoch, worker: ["00000.tar", "00001.tar"]
    assert _captions(iter(stream), 2) == ["cap 4", "cap 4"]
    assert stream.skipped >= 5


def test_a_damaged_header_after_the_first_raises(tmp_path: Path) -> None:
    shard = _write_shard(tmp_path / "a.tar", _sample("0") + _sample("1"))
    with tarfile.open(shard) as tar:
        offset = tar.getmembers()[3].offset
    data = bytearray(shard.read_bytes())
    data[offset + 148 : offset + 156] = b"garbage!"
    shard.write_bytes(bytes(data))
    samples = tar_samples(shard)
    assert next(samples)[0] == "0"
    with pytest.raises(tarfile.ReadError, match="damaged header"):
        next(samples)


def test_a_tar_cut_at_a_member_boundary_raises(tmp_path: Path) -> None:
    shard = _write_shard(tmp_path / "a.tar", _sample("0") + _sample("1"))
    with tarfile.open(shard) as tar:
        offset = tar.getmembers()[2].offset
    shard.write_bytes(shard.read_bytes()[:offset])
    with pytest.raises(tarfile.ReadError, match="without its end marker"):
        list(tar_samples(shard))


def test_empty_and_missing_shards_count_as_skips(tmp_path: Path) -> None:
    _write_shard(tmp_path / "00000.tar", [])
    _write_shard(tmp_path / "00001.tar", _sample("1"))
    stream = DataCompStream(tmp_path, PROMPT, max_attempts=4)
    stream.worker_shards = lambda epoch, worker: ["00000.tar", "gone.tar", "00001.tar"]
    assert _captions(iter(stream), 1) == ["cap 1"]
    assert stream.skipped == 2


def test_a_stream_of_empty_shards_raises(tmp_path: Path) -> None:
    for s in range(2):
        _write_shard(tmp_path / f"{s:05d}.tar", [])
    with pytest.raises(RuntimeError, match="4 DataComp samples in a row"):
        next(iter(DataCompStream(tmp_path, PROMPT, max_attempts=4)))


def test_a_run_of_unusable_samples_raises(tmp_path: Path) -> None:
    _write_shard(tmp_path / "00000.tar", [m for i in range(3) for m in _sample(str(i), jpg=b"x")])
    with pytest.raises(RuntimeError, match="3 DataComp samples in a row"):
        next(iter(DataCompStream(tmp_path, PROMPT, max_attempts=3)))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"rank": 2, "world_size": 2}, "outside a world"),
        ({"num_workers": -1}, "must not be negative"),
        ({"world_size": 2, "num_workers": 2}, "fewer than the 4"),
        ({"max_attempts": 0}, "must be positive"),
    ],
)
def test_bad_layouts_are_refused(tmp_path: Path, options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DataCompStream(_shards(tmp_path, 3, 1), PROMPT, **options)


def test_a_stream_read_by_another_worker_count_raises(tmp_path: Path) -> None:
    stream = DataCompStream(_shards(tmp_path, 2, 1), PROMPT, num_workers=2)
    with pytest.raises(RuntimeError, match="built for 2 loader workers but is read by 0"):
        next(iter(stream))


@pytest.mark.parametrize("read", [1, 4, 5, 9, 10])
def test_a_resume_starts_at_the_next_sample(tmp_path: Path, read: int) -> None:
    _shards(tmp_path, 3, 3)
    stream = DataCompStream(tmp_path, PROMPT, seed=4)
    reader = iter(stream)
    _captions(reader, read, stream)
    state = stream.state_dict()
    expected = _captions(reader, 4)
    assert stream.state_dict() == state
    resumed = DataCompStream(tmp_path, PROMPT, seed=4)
    resumed.load_state_dict(state)
    assert _captions(iter(resumed), 4) == expected


def test_a_resume_passes_over_unusable_samples_it_had_reached(tmp_path: Path) -> None:
    members = _sample("0") + _sample("1", jpg=b"x") + _sample("2") + _sample("3")
    _write_shard(tmp_path / "00000.tar", members)
    stream = DataCompStream(tmp_path, PROMPT)
    _captions(iter(stream), 2, stream)
    state = stream.state_dict()
    assert state["positions"] == [[0, 0, 3]]
    resumed = DataCompStream(tmp_path, PROMPT)
    resumed.load_state_dict(state)
    assert _captions(iter(resumed), 2) == ["cap 3", "cap 0"]
    assert resumed.skipped == 0


@pytest.mark.parametrize(
    ("name", "change"),
    [
        ("seed", {"seed": 5}),
        ("rank", {"rank": 1}),
        ("num_workers", {"num_workers": 2}),
        ("shard_digest", {}),
    ],
)
def test_a_resume_with_another_layout_is_refused(tmp_path: Path, name: str, change: dict) -> None:
    _shards(tmp_path, 4, 1)
    state = DataCompStream(tmp_path, PROMPT, world_size=2).state_dict()
    if name == "shard_digest":
        (tmp_path / "00003.tar").rename(tmp_path / "00009.tar")
    with pytest.raises(ValueError, match=f"{name} was"):
        DataCompStream(tmp_path, PROMPT, world_size=2, **change).load_state_dict(state)


def _batch_captions(batch: list[Document]) -> list[str]:
    return [doc.turns[0].answers[0] for doc in batch]


@pytest.mark.parametrize("context", ["fork", "spawn"])
@pytest.mark.parametrize("trained", [3, 4, 5])
def test_a_loader_resume_continues_where_training_stopped(
    tmp_path: Path, context: str, trained: int
) -> None:
    _shards(tmp_path, 6, 5)

    def loader() -> StreamLoader:
        stream = DataCompStream(tmp_path, PROMPT, num_workers=3, seed=2)
        options = {"num_workers": 3, "multiprocessing_context": context, "prefetch_factor": 4}
        return StreamLoader(stream, 2, collate_fn=list, **options)

    first = loader()
    reader = iter(first)
    for _ in range(trained):
        next(reader)
    state = first.state_dict()
    expected = [_batch_captions(next(reader)) for _ in range(6)]
    del reader
    resumed = loader()
    resumed.load_state_dict(state)
    assert [_batch_captions(batch) for batch in itertools.islice(resumed, 6)] == expected


def test_a_loader_refuses_persistent_workers(tmp_path: Path) -> None:
    stream = DataCompStream(_shards(tmp_path, 2, 1), PROMPT, num_workers=2)
    with pytest.raises(ValueError, match="persistent_workers"):
        StreamLoader(stream, 2, num_workers=2, persistent_workers=True)


def test_a_rendered_stream_trains_the_caption(tmp_path: Path, tokenizer: Any) -> None:
    _write_shard(tmp_path / "00000.tar", _sample("0", caption="a<|im_end|>b") + _sample("1"))
    stream = DataCompStream(tmp_path, PROMPT)
    renderer = ChatRenderer(tokenizer, max_text_len=64, image_size=8, answer_only=True)
    rendered = RenderedStream(stream, renderer)
    position, sample = next(iter(rendered))
    assert position == [0, 0, 0, 2]
    labels = sample["labels"]
    assert tokenizer.decode(labels[labels != -100]) == "cap 1<|im_end|>"
    assert sample["pixel_values"].shape[-2:] == torch.Size([8, 8])
    assert rendered.skipped == 1
    assert rendered.state_dict() == stream.state_dict()


class _Stream:
    def __init__(self, count: int) -> None:
        self.count = count

    def __iter__(self) -> Any:
        answers = tuple(f"answer {i}" for i in range(8))
        image = Image.new("RGB", (4, 4))
        document = Document(images=(image,), turns=(Turn(PROMPT, answers),))
        return iter([([0, 0, 0, i + 1], document) for i in range(self.count)])

    def advance(self, position: list[int]) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


def test_ranks_draw_replies_apart(tokenizer: Any) -> None:
    renderer = ChatRenderer(tokenizer, max_text_len=64, image_size=8, answer_only=True)

    def replies(rank: int) -> list[str]:
        stream = RenderedStream(_Stream(12), renderer, seed=3, rank=rank)
        return [tokenizer.decode(s["labels"][s["labels"] != -100]) for _, s in stream]

    assert replies(0) == replies(0) != replies(1)
