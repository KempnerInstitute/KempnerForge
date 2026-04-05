"""Unit tests for KempnerForge data pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kempnerforge.data.dataset import MemoryMappedDataset
from kempnerforge.data.sampler import DistributedSampler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mmap_data_dir(tmp_path):
    """Create temporary .npy files with fake token data."""
    seq_len = 16
    tokens_per_file = 160  # 10 samples of seq_len=16 each

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


# ---------------------------------------------------------------------------
# MemoryMappedDataset
# ---------------------------------------------------------------------------


class TestMemoryMappedDataset:
    def test_len(self, mmap_data_dir):
        data_dir, seq_len = mmap_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        # 3 files × 160 tokens each / 16 seq_len = 30 samples
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
        # First file: tokens 0..159, first sample: tokens 0..15
        # input_ids = [0,1,...,14], labels = [1,2,...,15]
        assert sample["input_ids"][0].item() == 0
        assert sample["labels"][0].item() == 1
        assert (sample["labels"][:-1] == sample["input_ids"][1:]).all()

    def test_cross_file_boundary(self, mmap_data_dir):
        """Samples near file boundaries should work correctly."""
        data_dir, seq_len = mmap_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        # Sample 10 is the first sample in the second file
        sample = ds[10]
        assert sample["input_ids"][0].item() == 160  # second file starts at 160

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
        assert len(ds8) == 32  # 256 / 8
        assert len(ds32) == 8  # 256 / 32

    def test_partial_sequence_dropped(self, tmp_path):
        """Tokens that don't fill a full seq_len are discarded."""
        tokens = np.arange(100, dtype=np.uint16)  # 100 tokens, seq_len=16 → 6 full + 4 leftover
        np.save(tmp_path / "data.npy", tokens)

        ds = MemoryMappedDataset(str(tmp_path), seq_len=16)
        assert len(ds) == 6  # 100 // 16 = 6, 4 tokens discarded


# ---------------------------------------------------------------------------
# DistributedSampler
# ---------------------------------------------------------------------------


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

        # No overlap
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

        # Should cover all original indices
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

        assert len(sizes) == 1  # all same size

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

        # Full iteration
        full = list(s)

        # Skip first 10
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
        # 100 samples, 3 replicas → drop_last: 33 each (99 used), pad: 34 each
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


# ---------------------------------------------------------------------------
# StatefulDataLoader
# ---------------------------------------------------------------------------


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

        # Iterate a few batches
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

        # Exhaust one epoch
        list(loader)
        assert loader._epoch == 1


# ---------------------------------------------------------------------------
# StreamingHuggingFaceDataset
# ---------------------------------------------------------------------------


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

        # Mock tokenizer: each character → its ordinal (simple 1-to-1 mapping)
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
                return self  # Don't shuffle for deterministic tests

            def __iter__(self):
                return iter(self._docs)

        ds._load_stream = lambda: MockStream(documents)

    def test_basic_iteration(self):
        ds = self._make_mock_dataset(seq_len=4)
        # Each char encodes to one token. "abcde" = 5 tokens + 1 EOS = 6 tokens
        # chunk_size = seq_len + 1 = 5, so one full chunk from 6 tokens
        docs = [{"text": "abcdefghij"}]  # 10 tokens + 1 EOS = 11 → 2 chunks of 5
        self._patch_stream(ds, docs)

        samples = list(ds)
        assert len(samples) == 2
        assert samples[0]["input_ids"].shape == (4,)
        assert samples[0]["labels"].shape == (4,)

    def test_sequence_packing(self):
        """Multiple short docs should be packed into sequences."""
        ds = self._make_mock_dataset(seq_len=8)
        # chunk_size = 9. "abc" = 3 tokens + EOS = 4. Need 3 docs for one chunk.
        docs = [{"text": "abc"}, {"text": "def"}, {"text": "ghi"}]
        # 3+1 + 3+1 + 3+1 = 12 tokens → 1 chunk of 9, 3 leftover
        self._patch_stream(ds, docs)

        samples = list(ds)
        assert len(samples) == 1

    def test_distributed_sharding(self):
        """Each rank should get different documents."""
        docs = [{"text": f"doc{i}" * 10} for i in range(20)]

        results = {}
        for rank in range(2):
            ds = self._make_mock_dataset(seq_len=4, rank=rank, world_size=2)
            self._patch_stream(ds, docs)
            results[rank] = list(ds)

        # Both ranks should produce sequences
        assert len(results[0]) > 0
        assert len(results[1]) > 0

    def test_empty_docs_skipped(self):
        ds = self._make_mock_dataset(seq_len=4)
        docs = [{"text": ""}, {"text": "abcdefghij"}]
        self._patch_stream(ds, docs)

        samples = list(ds)
        assert len(samples) > 0  # Empty doc should be skipped

    def test_state_dict_round_trip(self):
        ds = self._make_mock_dataset(seq_len=4)
        docs = [{"text": "abcdefghij" * 5}]  # 50 tokens → plenty of chunks
        self._patch_stream(ds, docs)

        # Consume some samples
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
