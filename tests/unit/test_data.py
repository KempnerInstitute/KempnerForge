"""Unit tests for KempnerForge data pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kempnerforge.data.dataset import MemoryMappedDataset
from kempnerforge.data.sampler import DistributedSampler


@pytest.fixture
def mmap_data_dir(tmp_path):
    """Create temporary .npy files with fake token data."""
    seq_len = 16
    tokens_per_file = 160

    for i in range(3):
        tokens = np.arange(i * tokens_per_file, (i + 1) * tokens_per_file, dtype=np.uint16)
        np.save(tmp_path / f"shard_{i:03d}.npy", tokens)

    return tmp_path, seq_len


@pytest.fixture
def simple_dataset():
    """A trivial list-based dataset for sampler tests."""

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, n):
            self.data = list(range(n))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return ListDataset(100)


class TestMemoryMappedDataset:
    def test_len(self, mmap_data_dir):
        data_dir, seq_len = mmap_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        assert len(ds) == 30

    def test_getitem_shape(self, mmap_data_dir):
        data_dir, seq_len = mmap_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        sample = ds[0]
        assert sample["input_ids"].shape == (seq_len - 1,)
        assert sample["labels"].shape == (seq_len - 1,)

    def test_input_labels_offset(self, mmap_data_dir):
        """Labels should be input shifted by 1 (causal LM)."""
        data_dir, seq_len = mmap_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        sample = ds[0]
        assert sample["input_ids"][0].item() == 0
        assert sample["labels"][0].item() == 1
        assert (sample["labels"][:-1] == sample["input_ids"][1:]).all()

    def test_cross_file_boundary(self, mmap_data_dir):
        """Samples near file boundaries should work correctly."""
        data_dir, seq_len = mmap_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        sample = ds[10]
        assert sample["input_ids"][0].item() == 160

    def test_index_out_of_range(self, mmap_data_dir):
        data_dir, seq_len = mmap_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        with pytest.raises(IndexError):
            ds[30]
        with pytest.raises(IndexError):
            ds[-1]

    def test_no_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No files"):
            MemoryMappedDataset(str(tmp_path), seq_len=16)

    def test_state_dict_round_trip(self, mmap_data_dir):
        data_dir, seq_len = mmap_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        ds._epoch = 5
        state = ds.state_dict()
        assert state["epoch"] == 5

        ds2 = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        ds2.load_state_dict(state)
        assert ds2._epoch == 5

    def test_different_seq_len(self, tmp_path):
        """Different seq_len should produce different sample counts."""
        tokens = np.arange(256, dtype=np.uint16)
        np.save(tmp_path / "data.npy", tokens)

        ds8 = MemoryMappedDataset(str(tmp_path), seq_len=8)
        ds32 = MemoryMappedDataset(str(tmp_path), seq_len=32)
        assert len(ds8) == 32
        assert len(ds32) == 8

    def test_partial_sequence_dropped(self, tmp_path):
        """Tokens that don't fill a full seq_len are discarded."""
        tokens = np.arange(100, dtype=np.uint16)
        np.save(tmp_path / "data.npy", tokens)

        ds = MemoryMappedDataset(str(tmp_path), seq_len=16)
        assert len(ds) == 6


class TestDistributedSampler:
    def test_partition_no_overlap(self, simple_dataset):
        """Indices across ranks should not overlap."""
        num_replicas = 4
        all_indices = []
        for rank in range(num_replicas):
            sampler = DistributedSampler(
                simple_dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
            all_indices.append(list(sampler))

        flat = [idx for rank_indices in all_indices for idx in rank_indices]
        assert len(flat) == len(set(flat))

    def test_partition_covers_dataset(self, simple_dataset):
        """All indices should be covered (with drop_last=False)."""
        num_replicas = 3
        all_indices = set()
        for rank in range(num_replicas):
            sampler = DistributedSampler(
                simple_dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            all_indices.update(list(sampler))

        assert all_indices == set(range(len(simple_dataset)))

    def test_equal_partition_sizes(self, simple_dataset):
        """All ranks should get the same number of samples."""
        num_replicas = 4
        sizes = set()
        for rank in range(num_replicas):
            sampler = DistributedSampler(
                simple_dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
            sizes.add(len(list(sampler)))

        assert len(sizes) == 1

    def test_deterministic_shuffle(self, simple_dataset):
        """Same seed + epoch → same permutation."""
        s1 = DistributedSampler(
            simple_dataset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=42,
        )
        s2 = DistributedSampler(
            simple_dataset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=42,
        )
        s1.set_epoch(3)
        s2.set_epoch(3)
        assert list(s1) == list(s2)

    def test_different_epoch_different_order(self, simple_dataset):
        """Different epochs should produce different orderings."""
        s = DistributedSampler(
            simple_dataset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=42,
        )
        s.set_epoch(0)
        order_0 = list(s)
        s.set_epoch(1)
        order_1 = list(s)
        assert order_0 != order_1

    def test_skip_ahead(self, simple_dataset):
        """Skip-ahead should produce the tail of the full iteration."""
        s = DistributedSampler(
            simple_dataset,
            num_replicas=1,
            rank=0,
            shuffle=False,
            drop_last=True,
        )

        full = list(s)

        s.set_skip(10)
        skipped = list(s)
        assert skipped == full[10:]

    def test_state_dict_round_trip(self, simple_dataset):
        s = DistributedSampler(
            simple_dataset,
            num_replicas=2,
            rank=1,
            shuffle=True,
            seed=99,
        )
        s.set_epoch(7)
        state = s.state_dict()
        assert state["epoch"] == 7
        assert state["seed"] == 99
        assert state["rank"] == 1

        s2 = DistributedSampler(
            simple_dataset,
            num_replicas=2,
            rank=1,
            shuffle=True,
            seed=99,
        )
        s2.load_state_dict(state)
        assert s2._epoch == 7

    def test_drop_last_vs_pad(self, simple_dataset):
        """drop_last=True discards, drop_last=False pads."""
        s_drop = DistributedSampler(
            simple_dataset,
            num_replicas=3,
            rank=0,
            shuffle=False,
            drop_last=True,
        )
        s_pad = DistributedSampler(
            simple_dataset,
            num_replicas=3,
            rank=0,
            shuffle=False,
            drop_last=False,
        )
        assert len(list(s_drop)) == 33
        assert len(list(s_pad)) == 34


class TestStatefulDataLoader:
    def test_basic_iteration(self, mmap_data_dir):
        from kempnerforge.config.schema import DataConfig
        from kempnerforge.data.dataloader import StatefulDataLoader

        data_dir, seq_len = mmap_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        config = DataConfig(num_workers=0, pin_memory=False)
        loader = StatefulDataLoader(ds, batch_size=4, config=config)

        batches = list(loader)
        assert len(batches) > 0
        assert batches[0]["input_ids"].shape == (4, seq_len - 1)

    def test_state_dict(self, mmap_data_dir):
        from kempnerforge.config.schema import DataConfig
        from kempnerforge.data.dataloader import StatefulDataLoader

        data_dir, seq_len = mmap_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        config = DataConfig(num_workers=0, pin_memory=False)
        loader = StatefulDataLoader(ds, batch_size=4, config=config)

        it = iter(loader)
        next(it)
        next(it)

        state = loader.state_dict()
        assert state["batches_yielded"] == 2
        assert "sampler" in state

    def test_epoch_increments(self, mmap_data_dir):
        from kempnerforge.config.schema import DataConfig
        from kempnerforge.data.dataloader import StatefulDataLoader

        data_dir, seq_len = mmap_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        config = DataConfig(num_workers=0, pin_memory=False)
        loader = StatefulDataLoader(ds, batch_size=4, config=config)

        list(loader)
        assert loader._epoch == 1


class TestHuggingFaceDataset:
    """Tests for the non-streaming HuggingFace dataset using mocked HF internals."""

    def _make_dataset(self, seq_len, documents):
        """Create a HuggingFaceDataset with mocked load_dataset and tokenizer."""
        from unittest.mock import MagicMock, patch

        from kempnerforge.data.dataset import HuggingFaceDataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.encode = lambda text, **kwargs: [ord(c) for c in text]

        mock_raw_dataset = documents

        with (
            patch("datasets.load_dataset", return_value=mock_raw_dataset),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            ds = HuggingFaceDataset(
                dataset_name="mock/dataset",
                split="train",
                text_field="text",
                seq_len=seq_len,
                tokenizer_path="mock",
            )
        return ds

    def test_len(self):
        ds = self._make_dataset(seq_len=4, documents=[{"text": "abcdefghij"}])
        assert len(ds) == 2

    def test_getitem_shape(self):
        ds = self._make_dataset(seq_len=8, documents=[{"text": "a" * 30}])
        sample = ds[0]
        assert sample["input_ids"].shape == (8,)
        assert sample["labels"].shape == (8,)

    def test_input_label_offset(self):
        """Labels should be inputs shifted by 1."""
        ds = self._make_dataset(seq_len=4, documents=[{"text": "abcde"}])
        sample = ds[0]
        assert sample["input_ids"][0].item() == ord("a")
        assert sample["labels"][0].item() == ord("b")
        assert (sample["labels"][:-1] == sample["input_ids"][1:]).all()

    def test_sequence_packing(self):
        """Short docs should be concatenated with EOS separators."""
        docs = [{"text": "ab"}, {"text": "cd"}, {"text": "ef"}]
        ds = self._make_dataset(seq_len=4, documents=docs)
        assert len(ds) == 1

    def test_empty_docs_skipped(self):
        """Empty documents should not contribute tokens (except no EOS)."""
        ds = self._make_dataset(
            seq_len=4,
            documents=[{"text": ""}, {"text": "abcdefghij"}],
        )
        assert len(ds) == 2

    def test_state_dict_round_trip(self):
        ds = self._make_dataset(seq_len=4, documents=[{"text": "abcdefghij"}])
        ds._epoch = 3
        ds._sample_idx = 5

        state = ds.state_dict()
        assert state["epoch"] == 3
        assert state["sample_idx"] == 5
        assert state["total_sequences"] == len(ds)

        ds2 = self._make_dataset(seq_len=4, documents=[{"text": "abcdefghij"}])
        ds2.load_state_dict(state)
        assert ds2._epoch == 3
        assert ds2._sample_idx == 5


class TestStreamingHuggingFaceDataset:
    """Tests using a mock streaming dataset to avoid network dependencies."""

    def _make_mock_dataset(self, seq_len, rank=0, world_size=1, seed=42):
        """Create a StreamingHuggingFaceDataset with mocked HF internals."""
        from unittest.mock import MagicMock

        from kempnerforge.data.dataset import StreamingHuggingFaceDataset

        ds = StreamingHuggingFaceDataset(
            dataset_name="mock/dataset",
            split="train",
            text_field="text",
            seq_len=seq_len,
            tokenizer_path="mock",
            rank=rank,
            world_size=world_size,
            seed=seed,
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.encode = lambda text, **kwargs: [ord(c) for c in text]
        ds._tokenizer = mock_tokenizer
        ds._eos_id = 0

        return ds

    def _patch_stream(self, ds, documents):
        """Patch _load_stream to return mock documents."""

        class MockStream:
            def __init__(self, docs):
                self._docs = docs

            def shuffle(self, seed=0, buffer_size=1000):
                return self

            def __iter__(self):
                return iter(self._docs)

        ds._load_stream = lambda: MockStream(documents)

    def test_basic_iteration(self):
        ds = self._make_mock_dataset(seq_len=4)
        docs = [{"text": "abcdefghij"}]
        self._patch_stream(ds, docs)

        samples = list(ds)
        assert len(samples) == 2
        assert samples[0]["input_ids"].shape == (4,)
        assert samples[0]["labels"].shape == (4,)

    def test_sequence_packing(self):
        """Multiple short docs should be packed into sequences."""
        ds = self._make_mock_dataset(seq_len=8)
        docs = [{"text": "abc"}, {"text": "def"}, {"text": "ghi"}]
        self._patch_stream(ds, docs)

        samples = list(ds)
        assert len(samples) == 1

    @staticmethod
    def _read_as_rank(rank: int, world_size: int, n_shards: int) -> tuple[int, set[int]]:
        """Stream ``n_shards`` local shards of 5 documents as ``rank`` of ``world_size``.

        Returns how many documents this rank read from the source and which
        documents it yielded. Document ``i`` is the text ``str(i)`` and
        tokenizes to ``[i, i, i]``, so with ``seq_len=3`` a sample is one document.
        """
        from unittest.mock import patch

        from datasets import IterableDataset

        from kempnerforge.data.dataset import StreamingHuggingFaceDataset

        reads: list[int] = []

        def source(shards):
            for shard in shards:
                for j in range(5):
                    reads.append(shard * 5 + j)
                    yield {"text": str(shard * 5 + j)}

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [int(text)] * 3

        stream = IterableDataset.from_generator(
            source, gen_kwargs={"shards": list(range(n_shards))}
        )
        with patch("datasets.load_dataset", return_value=stream):
            ds = StreamingHuggingFaceDataset(
                "local",
                "train",
                "text",
                3,
                "tok",
                rank=rank,
                world_size=world_size,
                shuffle_buffer_size=1,
            )
            ds._tokenizer, ds._eos_id = Tokenizer(), 0
            docs = {int(sample["input_ids"][0]) for sample in ds}
        return len(reads), docs

    @pytest.mark.parametrize("n_shards", [4, 3])
    def test_distributed_sharding(self, n_shards):
        """Every document goes to exactly one rank, whether or not the shards divide evenly."""
        parts = [self._read_as_rank(rank, 2, n_shards)[1] for rank in range(2)]
        assert not parts[0] & parts[1], "a document went to both ranks"
        assert parts[0] | parts[1] == set(range(n_shards * 5))

    def test_each_rank_reads_only_its_shards(self):
        """With a shard count that divides by the ranks, a rank reads its own shards alone."""
        reads = [self._read_as_rank(rank, 2, 4)[0] for rank in range(2)]
        assert reads == [10, 10]

    def test_empty_docs_skipped(self):
        ds = self._make_mock_dataset(seq_len=4)
        docs = [{"text": ""}, {"text": "abcdefghij"}]
        self._patch_stream(ds, docs)

        samples = list(ds)
        assert len(samples) > 0

    def test_state_dict_round_trip(self):
        ds = self._make_mock_dataset(seq_len=4)
        docs = [{"text": "abcdefghij" * 5}]
        self._patch_stream(ds, docs)

        it = iter(ds)
        next(it)
        next(it)

        state = ds.state_dict()
        assert "epoch" in state
        assert "rank_docs_consumed" in state

    def test_load_state_dict(self):
        ds = self._make_mock_dataset(seq_len=4)

        ds.load_state_dict({"epoch": 3, "rank_docs_consumed": 100})
        assert ds._epoch == 3
        assert ds._skip_rank_docs == 100
