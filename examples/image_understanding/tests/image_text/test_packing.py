"""Tests for packing rendered documents into fixed-length rows."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
import torch
from image_text.datacomp import DataCompStream
from image_text.packing import PackedLoader, PackedStreamLoader, footprint
from image_text.render import ChatRenderer, RenderedStream
from test_datacomp import _shards
from torch.utils.data import Dataset, IterableDataset

TOKENS_PER_IMAGE = 2


class Docs(Dataset):
    """Documents of given (views, text tokens), whose tokens are their index plus one."""

    def __init__(self, sizes: list[tuple[int, int]]) -> None:
        self.sizes = sizes

    def __len__(self) -> int:
        return len(self.sizes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        views, text = self.sizes[index]
        return {
            "pixel_values": torch.zeros(views, 3, 2, 2),
            "input_ids": torch.full((text,), index + 1),
            "labels": torch.full((text,), index + 1),
            "text_len": torch.tensor(text),
        }


def _documents(batch: dict[str, torch.Tensor]) -> list[int]:
    seen: list[int] = []
    for token in batch["input_ids"][0].tolist():
        if token and (not seen or seen[-1] != token):
            seen.append(token)
    return [token - 1 for token in seen]


def _greedy(sizes: list[tuple[int, int]], order: list[int], seq_len: int) -> list[list[int]]:
    rows: list[list[int]] = []
    fill = 0
    for index in order:
        views, text = sizes[index]
        size = views * TOKENS_PER_IMAGE + text
        if size > seq_len:
            continue
        if rows and fill + size <= seq_len:
            rows[-1].append(index)
            fill += size
        else:
            rows.append([index])
            fill = size
    return rows


SIZES = [(1, 5), (0, 9), (3, 2), (2, 7), (0, 3), (1, 1), (4, 4), (0, 12), (2, 2), (1, 6)]


def _loader(sizes: list[tuple[int, int]] = SIZES, **options: object) -> PackedLoader:
    kwargs: dict = {"seq_len": 16, "tokens_per_image": TOKENS_PER_IMAGE, "pad_id": 0}
    kwargs.update(options)
    return PackedLoader(Docs(sizes), **kwargs)


class TestPackedLoader:
    def test_documents_fill_rows_in_order(self) -> None:
        loader = _loader()
        expected = _greedy(SIZES, loader.order(0).tolist(), 16)
        batches = list(itertools.islice(loader, len(expected)))
        assert [_documents(b) for b in batches] == expected
        assert all(b["input_ids"].shape == (1, 16) for b in batches)
        assert [b["pixel_values"].shape[0] for b in batches] == [
            sum(SIZES[i][0] for i in row) for row in expected
        ]

    def test_a_document_longer_than_a_row_is_passed_over(self) -> None:
        sizes = [*SIZES[:4], (5, 10), *SIZES[4:]]
        loader = _loader(sizes)
        rows = [_documents(b) for b in itertools.islice(loader, 6)]
        assert 4 not in itertools.chain(*rows)
        assert loader.oversize == 1
        assert footprint(Docs(sizes)[4], TOKENS_PER_IMAGE) == 20

    def test_passes_repeat_with_a_new_order(self) -> None:
        loader = _loader()
        first = _greedy(SIZES, loader.order(0).tolist(), 16)
        second = _greedy(SIZES, loader.order(1).tolist(), 16)
        batches = list(itertools.islice(loader, len(first) + len(second)))
        assert [_documents(b) for b in batches] == first + second
        assert loader.order(0).tolist() != loader.order(1).tolist()
        assert loader.state_dict()["pass"] == 2

    def test_ranks_split_each_pass_evenly(self) -> None:
        orders = [_loader(rank=r, world_size=3).order(0).tolist() for r in range(3)]
        assert [len(order) for order in orders] == [3, 3, 3]
        assert len(set(itertools.chain(*orders))) == 9

    @pytest.mark.parametrize("taken", [1, 3, 5])
    def test_a_resume_packs_the_rows_of_an_uninterrupted_run(self, taken: int) -> None:
        uninterrupted = [_documents(b) for b in itertools.islice(_loader(), 9)]
        first = _loader()
        before = [_documents(b) for b in itertools.islice(first, taken)]
        resumed = _loader()
        resumed.load_state_dict(first.state_dict())
        after = [_documents(b) for b in itertools.islice(resumed, 9 - taken)]
        assert before + after == uninterrupted

    def test_samples_render_in_loader_workers(self) -> None:
        loader = _loader(num_workers=2)
        expected = _greedy(SIZES, loader.order(0).tolist(), 16)
        assert [_documents(b) for b in itertools.islice(loader, len(expected))] == expected

    def test_the_training_loop_shuts_its_workers_down(self) -> None:
        from kempnerforge.training.data_pipeline import DataPipeline
        from kempnerforge.training.loop import BatchStream

        stream = BatchStream(DataPipeline(dataloader=_loader(num_workers=2)))
        stream.next_batch()
        workers = list(stream.dataloader._iterator._workers)
        stream.close()
        assert workers and not any(worker.is_alive() for worker in workers)

    def test_view_groups_give_each_view_its_own_group(self) -> None:
        loader = _loader([(2, 3), (0, 4), (1, 2)], view_groups=True, seq_len=32)
        loader.order = lambda _pass: torch.tensor([0, 1, 2]).numpy()
        batch = next(iter(loader))
        groups = batch["attention_group_ids"][0].tolist()
        assert groups[:4] == [0, 0, 1, 1] and groups[4:11] == [-1] * 7
        assert groups[11:13] == [2, 2] and set(groups[13:]) == {-1}
        assert "attention_group_ids" not in next(iter(_loader()))

    @pytest.mark.parametrize("option", ["batch_size", "sampler", "shuffle", "collate_fn"])
    def test_options_it_sets_are_refused(self, option: str) -> None:
        with pytest.raises(ValueError, match=f"sets {option} itself"):
            _loader(**{option: None})

    def test_arguments_are_checked(self) -> None:
        with pytest.raises(ValueError, match="outside world_size"):
            _loader(rank=2, world_size=2)
        with pytest.raises(ValueError, match="positive"):
            _loader(seq_len=0)
        with pytest.raises(ValueError, match="one per rank"):
            _loader(SIZES[:1], world_size=2)

    def test_a_pass_with_nothing_to_pack_raises(self) -> None:
        with pytest.raises(RuntimeError, match="packed no document"):
            next(iter(_loader([(9, 9), (9, 9)])))


class DocStream(IterableDataset):
    """``Docs`` read in index order without end, each with the count read so far as position."""

    def __init__(self, sizes: list[tuple[int, int]]) -> None:
        self.docs = Docs(sizes)
        self.next = 0

    def __iter__(self) -> Any:
        for count in itertools.count(self.next):
            yield [count + 1], self.docs[count % len(self.docs)]

    def advance(self, position: list[int]) -> None:
        self.next = position[0]

    def state_dict(self) -> dict[str, int]:
        return {"next": self.next}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.next = state["next"]


def _stream_loader(sizes: list[tuple[int, int]] = SIZES, **options: Any) -> PackedStreamLoader:
    kwargs: dict = {"seq_len": 16, "tokens_per_image": TOKENS_PER_IMAGE, "pad_id": 0}
    kwargs.update(options)
    return PackedStreamLoader(DocStream(sizes), **kwargs)


class TestPackedStreamLoader:
    def test_documents_fill_rows_in_stream_order(self) -> None:
        expected = _greedy(SIZES, list(range(len(SIZES))) * 3, 16)[:-1]
        batches = list(itertools.islice(_stream_loader(), len(expected)))
        assert [_documents(b) for b in batches] == [
            [i % len(SIZES) for i in row] for row in expected
        ]
        assert all(b["input_ids"].shape == (1, 16) for b in batches)

    def test_a_document_longer_than_a_row_is_passed_over(self) -> None:
        sizes = [*SIZES[:4], (5, 10), *SIZES[4:]]
        loader = _stream_loader(sizes)
        rows = [_documents(b) for b in itertools.islice(loader, 6)]
        assert 4 not in itertools.chain(*rows)
        assert loader.oversize == 1 and loader.state_dict()["oversize"] == 1

    @pytest.mark.parametrize("taken", [1, 3, 5])
    def test_a_resume_packs_the_rows_of_an_uninterrupted_run(self, taken: int) -> None:
        uninterrupted = [_documents(b) for b in itertools.islice(_stream_loader(), 9)]
        first = _stream_loader()
        before = [_documents(b) for b in itertools.islice(first, taken)]
        resumed = _stream_loader()
        resumed.load_state_dict(first.state_dict())
        after = [_documents(b) for b in itertools.islice(resumed, 9 - taken)]
        assert before + after == uninterrupted

    @pytest.mark.parametrize("taken", [1, 2])
    def test_a_datacomp_resume_packs_the_rows_of_an_uninterrupted_run(
        self, tmp_path: Path, tokenizer: Any, taken: int
    ) -> None:
        _shards(tmp_path, 4, 4)
        renderer = ChatRenderer(tokenizer, max_text_len=None, image_size=8, answer_only=False)

        def loader() -> PackedStreamLoader:
            stream = DataCompStream(tmp_path, "Caption it.", num_workers=2, seed=4)
            rendered = RenderedStream(stream, renderer, seed=4)
            return PackedStreamLoader(
                rendered, seq_len=96, tokens_per_image=4, pad_id=renderer.pad_id, num_workers=2
            )

        uninterrupted = [b["input_ids"] for b in itertools.islice(loader(), 5)]
        first = loader()
        before = [b["input_ids"] for b in itertools.islice(first, taken)]
        resumed = loader()
        resumed.load_state_dict(first.state_dict())
        after = [b["input_ids"] for b in itertools.islice(resumed, 5 - taken)]
        assert all(torch.equal(a, b) for a, b in zip(before + after, uninterrupted, strict=True))

    def test_view_groups_give_each_view_its_own_group(self) -> None:
        loader = _stream_loader([(2, 3), (0, 4), (1, 2)], view_groups=True, seq_len=32)
        groups = next(iter(loader))["attention_group_ids"][0].tolist()
        assert groups[:4] == [0, 0, 1, 1] and groups[4:11] == [-1] * 7
        assert groups[11:13] == [2, 2]

    def test_arguments_are_checked(self) -> None:
        with pytest.raises(TypeError, match="needs an IterableDataset, got Docs"):
            PackedStreamLoader(Docs(SIZES), seq_len=16, tokens_per_image=2, pad_id=0)
        with pytest.raises(ValueError, match="sets batch_size itself"):
            _stream_loader(batch_size=1)
        with pytest.raises(ValueError, match="persistent_workers"):
            _stream_loader(num_workers=2, persistent_workers=True)
        with pytest.raises(ValueError, match="positive"):
            _stream_loader(seq_len=0)
