"""Loading a resumable stream so that a resume continues where training stopped."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate


class ResumableStream(Protocol):
    """An iterable dataset of ``(position, item)`` pairs whose positions it can record."""

    def __iter__(self) -> Iterator[tuple[list[int], Any]]: ...

    def advance(self, position: list[int]) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


def _split(collate_fn: Callable[[list[Any]], Any], pairs: list[tuple[list[int], Any]]) -> Any:
    positions, items = zip(*pairs, strict=True)
    return list(positions), collate_fn(list(items))


class StreamLoader:
    """A ``DataLoader`` over ``stream`` that records the position of every batch it hands out.

    The positions stay in this process, so ``state_dict`` is where training
    stopped rather than where the loader workers, running ahead to fill their
    prefetch queues, had read to, and loader workers may be started by any method.
    Each batch is ``collate_fn`` over its items; ``options`` go to the
    ``DataLoader``, except persistent workers, which would carry on from the
    positions they were started with rather than the ones recorded here.
    """

    def __init__(
        self,
        stream: ResumableStream,
        batch_size: int,
        *,
        collate_fn: Callable[[list[Any]], Any] = default_collate,
        **options: Any,
    ) -> None:
        if not isinstance(stream, IterableDataset):
            raise TypeError(f"StreamLoader needs an IterableDataset, got {type(stream).__name__}")
        if options.get("persistent_workers"):
            raise ValueError(
                "StreamLoader starts its workers afresh, so persistent_workers is refused"
            )
        self.stream = stream
        self.loader = DataLoader(
            stream,
            batch_size=batch_size,
            collate_fn=functools.partial(_split, collate_fn),
            **options,
        )

    def __iter__(self) -> Iterator[Any]:
        for positions, batch in self.loader:
            for position in positions:
                self.stream.advance(position)
            yield batch

    def state_dict(self) -> dict[str, Any]:
        return self.stream.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.stream.load_state_dict(state)
