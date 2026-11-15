"""DataComp-1B captions, streamed from its webdataset tar shards without an index."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

from torch.utils.data import IterableDataset, get_worker_info

from .decode import decode_rgb
from .document import Document, Turn, is_text
from .skip import MAX_ATTEMPTS, SkipLog, UnusableRowError, check_max_attempts

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp")


def shard_names(shard_dir: str | Path) -> list[str]:
    """The ``.tar`` files directly under ``shard_dir``, sorted by name."""
    with os.scandir(shard_dir) as entries:
        return sorted(e.name for e in entries if e.name.endswith(".tar") and e.is_file())


def tar_samples(path: str | Path, skip: int = 0) -> Iterator[tuple[str, dict[str, bytes]]]:
    """A webdataset shard's samples in stored order, as each key and its members by extension.

    A sample is a run of consecutive members whose paths agree up to the first dot
    of the file name. The first ``skip`` samples are passed over without reading
    their members. A damaged header raises ``tarfile.ReadError`` where tarfile alone
    would end the archive there.
    """
    key: str | None = None
    index = -1
    members: dict[str, bytes] = {}
    with open(path, "rb") as file:
        with tarfile.open(fileobj=file, mode="r|") as tar:
            for info in tar:
                if not info.isfile():
                    continue
                folder, _, name = info.name.rpartition("/")
                stem, _, extension = name.partition(".")
                sample = f"{folder}/{stem}" if folder else stem
                if sample != key:
                    if key is not None and index >= skip:
                        yield key, members
                    key, members, index = sample, {}, index + 1
                if index >= skip:
                    data = tar.extractfile(info)
                    members[extension.lower()] = data.read() if data is not None else b""
            end = tar.offset
        _check_end(file, end)
    if key is not None and index >= skip:
        yield key, members


def _check_end(file: BinaryIO, offset: int) -> None:
    """Raise ``tarfile.ReadError`` unless only the zeros of an end marker follow ``offset``."""
    file.seek(offset)
    zeros = 0
    while chunk := file.read(1 << 20):
        if chunk.strip(b"\0"):
            raise tarfile.ReadError(f"a damaged header at byte {offset} ends the archive early")
        zeros += len(chunk)
    if zeros < 2 * tarfile.BLOCKSIZE:
        raise tarfile.ReadError(f"the archive stops at byte {offset} without its end marker")


class ShardEndError(UnusableRowError):
    """A shard that holds no samples or cannot be read further."""


class DataCompStream(IterableDataset):
    """DataComp image-caption pairs from its webdataset shards, each asked with ``prompt``.

    There is no index: the shards are listed once, reordered every epoch from
    ``seed``, dealt out to data-parallel ranks and then to this rank's
    ``num_workers`` loader workers. Each worker keeps ``open_shards`` of its shards
    open, reads each front to back, and takes every sample from one of them drawn
    at random, opening its next shard when one runs out or fails. The stream does not end;
    after its last shard a worker goes on to the next epoch.
    A sample without a decodable image or a caption is passed over, as is the rest
    of a shard that stops being a readable tar; ``skipped`` counts them, and
    ``max_attempts`` skips in a row raise.

    Each document comes with its position: the worker, then its epoch, the next of
    its shards to open, the draws it has made, and the ordinal and samples read of
    each open shard. The draws are seeded by the seed, rank, worker, epoch and draw
    count, so a position fixes what follows. ``advance`` records a position in the
    process that trains, which ``StreamLoader`` does for every batch it hands out,
    so ``state_dict`` holds where training stopped and a resume reopens each
    worker's shards past the samples it had read, without decoding them. The
    state also names the worker whose batch was due next, and a resumed loader's
    first worker takes up its shards, so batches come in the order they would have.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        prompt: str,
        *,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 0,
        open_shards: int = 4,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside a world of {world_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must not be negative, got {num_workers}")
        if open_shards < 1:
            raise ValueError(f"open_shards must be positive, got {open_shards}")
        self.shard_dir = Path(shard_dir)
        self.shards = shard_names(self.shard_dir)
        readers = world_size * max(num_workers, 1)
        if len(self.shards) < readers:
            raise ValueError(
                f"{self.shard_dir} holds {len(self.shards)} shards, fewer than the "
                f"{readers} ranks and workers that read them"
            )
        self.prompt = prompt
        self.rank = rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.seed = seed
        self.open_shards = open_shards
        self.max_attempts = check_max_attempts(max_attempts)
        self._skips = SkipLog("DataComp sample")
        self._positions = [self._state(0, 0, 0, []) for _ in range(max(num_workers, 1))]
        self._next_worker = 0

    @property
    def skipped(self) -> int:
        return self._skips.count

    def worker_shards(self, epoch: int, worker: int) -> list[str]:
        """The shards ``worker`` of this rank reads in ``epoch``, in reading order."""
        order = random.Random(f"{self.seed}:{epoch}").sample(self.shards, len(self.shards))
        return order[self.rank :: self.world_size][worker :: max(self.num_workers, 1)]

    def _state(self, epoch: int, next_shard: int, draws: int, active: list[list[int]]) -> list[int]:
        """A worker's position, with the unused places for open shards held by ``-1``."""
        padding = [-1, -1] * (self.open_shards - len(active))
        return [epoch, next_shard, draws, *(n for pair in active for n in pair), *padding]

    def _worker(self) -> int:
        info = get_worker_info()
        workers = 0 if info is None else info.num_workers
        if workers != self.num_workers:
            raise RuntimeError(
                f"the stream was built for {self.num_workers} loader workers "
                f"but is read by {workers}"
            )
        return 0 if info is None else (info.id + self._next_worker) % workers

    def _document(self, members: dict[str, bytes]) -> Document:
        image = next((members[e] for e in IMAGE_EXTENSIONS if e in members), None)
        if image is None or "json" not in members:
            raise UnusableRowError(f"has members {sorted(members)}, not an image and its json")
        try:
            caption = json.loads(members["json"]).get("caption")
        except (ValueError, AttributeError) as exc:
            raise UnusableRowError(f"json does not hold an object: {exc}") from None
        if not is_text(caption):
            raise UnusableRowError("has no caption")
        return Document(images=(decode_rgb(image),), turns=(Turn(self.prompt, (caption,)),))

    def _read_shard(self, name: str, skip: int) -> Iterator[Document | UnusableRowError]:
        found = False
        try:
            for key, members in tar_samples(self.shard_dir / name, skip):
                found = True
                try:
                    yield self._document(members)
                except UnusableRowError as exc:
                    yield UnusableRowError(f"{name} sample {key} {exc}")
        except (tarfile.TarError, EOFError, OSError) as exc:
            yield ShardEndError(f"{name} stops being a readable tar: {exc}")
            return
        if not found and skip == 0:
            yield ShardEndError(f"{name} holds no samples")

    def __iter__(self) -> Iterator[tuple[list[int], Document]]:
        worker = self._worker()
        epoch, next_shard, draws, *flat = self._positions[worker]
        active = [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2) if flat[i] >= 0]
        in_a_row = 0
        while True:
            shards = self.worker_shards(epoch, worker)
            readers = [self._read_shard(shards[ordinal], read) for ordinal, read in active]
            while True:
                while len(active) < self.open_shards and next_shard < len(shards):
                    active.append([next_shard, 0])
                    readers.append(self._read_shard(shards[next_shard], 0))
                    next_shard += 1
                if not active:
                    break
                draw = random.Random(f"{self.seed}:{self.rank}:{worker}:{epoch}:{draws}")
                pick = draw.randrange(len(active))
                item = next(readers[pick], None)
                if item is None:
                    del active[pick], readers[pick]
                    continue
                draws += 1
                active[pick][1] += 1
                if isinstance(item, ShardEndError):
                    del active[pick], readers[pick]
                if isinstance(item, UnusableRowError):
                    self._skips(draws - 1, item)
                    in_a_row += 1
                    if in_a_row >= self.max_attempts:
                        raise RuntimeError(
                            f"{self.max_attempts} DataComp samples in a row cannot be "
                            f"used, the last: {item}"
                        )
                    continue
                in_a_row = 0
                yield [worker, *self._state(epoch, next_shard, draws, active)], item
            epoch, next_shard, draws = epoch + 1, 0, 0

    def advance(self, position: list[int]) -> None:
        """Record that training has had the document at ``position``."""
        worker, *reached = position
        self._positions[worker] = list(reached)
        self._next_worker = (worker + 1) % len(self._positions)

    def state_dict(self) -> dict[str, Any]:
        """Each worker's position after the last document trained, and the layout."""
        return {
            "seed": self.seed,
            "world_size": self.world_size,
            "rank": self.rank,
            "num_workers": self.num_workers,
            "open_shards": self.open_shards,
            "shards": len(self.shards),
            "shard_digest": hashlib.sha256("\n".join(self.shards).encode()).hexdigest(),
            "positions": [list(position) for position in self._positions],
            "next_worker": self._next_worker,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Start each worker at the sample after the last one ``state`` says was trained."""
        current = self.state_dict()
        names = ("seed", "world_size", "rank", "num_workers", "open_shards", "shards")
        for name in (*names, "shard_digest"):
            if state[name] != current[name]:
                raise ValueError(
                    f"cannot resume the DataComp stream: {name} was {state[name]}, "
                    f"now {current[name]}"
                )
        self._positions = [list(position) for position in state["positions"]]
        self._next_worker = state["next_worker"]
