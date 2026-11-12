"""DataComp-1B captions, streamed from its webdataset tar shards without an index."""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from torch.utils.data import IterableDataset, get_worker_info

from .decode import decode_rgb
from .document import Document, Turn, is_text
from .skip import MAX_ATTEMPTS, SkipLog, UnusableRowError, check_max_attempts

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp")


def shard_names(shard_dir: str | Path) -> list[str]:
    """The ``.tar`` files directly under ``shard_dir``, sorted by name."""
    with os.scandir(shard_dir) as entries:
        return sorted(e.name for e in entries if e.name.endswith(".tar") and e.is_file())


def tar_samples(path: str | Path) -> Iterator[tuple[str, dict[str, bytes]]]:
    """A webdataset shard's samples in stored order, as each key and its members by extension.

    A sample is a run of consecutive members whose paths agree up to the first dot
    of the file name.
    """
    key: str | None = None
    members: dict[str, bytes] = {}
    with tarfile.open(path, mode="r|") as tar:
        for info in tar:
            if not info.isfile():
                continue
            folder, _, name = info.name.rpartition("/")
            stem, _, extension = name.partition(".")
            sample = f"{folder}/{stem}" if folder else stem
            if sample != key:
                if key is not None:
                    yield key, members
                key, members = sample, {}
            data = tar.extractfile(info)
            members[extension.lower()] = data.read() if data is not None else b""
    if key is not None:
        yield key, members


class DataCompStream(IterableDataset):
    """DataComp image-caption pairs from its webdataset shards, each asked with ``prompt``.

    There is no index: the shards are listed once, reordered every epoch from
    ``seed``, dealt out to data-parallel ranks and then to this rank's
    ``num_workers`` loader workers, and each shard is read front to back. The
    stream does not end; after its last shard a worker goes on to the next epoch.
    A sample without a decodable image or a caption is passed over, as is the rest
    of a shard that stops being a readable tar; ``skipped`` counts them, and
    ``max_attempts`` skips in a row raise.

    ``state_dict`` holds the shard each worker is reading, so a resume starts each
    worker at the beginning of that shard: it reads again what the worker had read
    of it, and samples of the previous shard still in the loader's prefetch queue
    are not seen. Loader workers must be forked from the process that calls
    ``state_dict``, which reads their positions from shared memory.
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
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside a world of {world_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must not be negative, got {num_workers}")
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
        self.max_attempts = check_max_attempts(max_attempts)
        self._skips = SkipLog("sample from DataComp shard")
        self._positions = multiprocessing.Array("q", 2 * max(num_workers, 1))

    @property
    def skipped(self) -> int:
        return self._skips.count

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_positions"] = list(self._positions)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        positions = state.pop("_positions")
        self.__dict__.update(state)
        self._positions = multiprocessing.Array("q", positions)

    def worker_shards(self, epoch: int, worker: int) -> list[str]:
        """The shards ``worker`` of this rank reads in ``epoch``, in reading order."""
        order = random.Random(f"{self.seed}:{epoch}").sample(self.shards, len(self.shards))
        return order[self.rank :: self.world_size][worker :: max(self.num_workers, 1)]

    def _worker(self) -> int:
        info = get_worker_info()
        workers = 0 if info is None else info.num_workers
        if workers != self.num_workers:
            raise RuntimeError(
                f"the stream was built for {self.num_workers} loader workers "
                f"but is read by {workers}"
            )
        return 0 if info is None else info.id

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

    def _read_shard(self, name: str) -> Iterator[Document | UnusableRowError]:
        try:
            for key, members in tar_samples(self.shard_dir / name):
                try:
                    yield self._document(members)
                except UnusableRowError as exc:
                    yield UnusableRowError(f"{name} sample {key} {exc}")
        except (tarfile.TarError, EOFError) as exc:
            yield UnusableRowError(f"{name} stops being a readable tar: {exc}")

    def __iter__(self) -> Iterator[Document]:
        worker = self._worker()
        with self._positions.get_lock():
            epoch, start = self._positions[2 * worker], self._positions[2 * worker + 1]
        in_a_row = 0
        while True:
            shards = self.worker_shards(epoch, worker)
            for ordinal in range(start, len(shards)):
                with self._positions.get_lock():
                    self._positions[2 * worker], self._positions[2 * worker + 1] = epoch, ordinal
                for item in self._read_shard(shards[ordinal]):
                    if isinstance(item, UnusableRowError):
                        self._skips(ordinal, item)
                        in_a_row += 1
                        if in_a_row >= self.max_attempts:
                            raise RuntimeError(
                                f"{self.max_attempts} DataComp samples in a row cannot be "
                                f"used, the last: {item}"
                            )
                        continue
                    in_a_row = 0
                    yield item
            epoch, start = epoch + 1, 0

    def state_dict(self) -> dict[str, Any]:
        """The ``(epoch, shard ordinal)`` each worker is reading, with the layout they assume."""
        with self._positions.get_lock():
            values = list(self._positions)
        return {
            "seed": self.seed,
            "world_size": self.world_size,
            "rank": self.rank,
            "num_workers": self.num_workers,
            "shards": len(self.shards),
            "positions": [values[i : i + 2] for i in range(0, len(values), 2)],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Start each worker at the beginning of the shard ``state`` says it was reading."""
        current = self.state_dict()
        for name in ("seed", "world_size", "rank", "num_workers", "shards"):
            if state[name] != current[name]:
                raise ValueError(
                    f"cannot resume the DataComp stream: {name} was {state[name]}, "
                    f"now {current[name]}"
                )
        with self._positions.get_lock():
            for worker, (epoch, ordinal) in enumerate(state["positions"]):
                self._positions[2 * worker], self._positions[2 * worker + 1] = epoch, ordinal
