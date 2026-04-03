"""Integration test: DataLoader state save → load → same sequence.

Verifies that the stateful dataloader resumes from the exact position
after a checkpoint/restore cycle.
"""

from __future__ import annotations

import numpy as np

from kempnerforge.config.schema import DataConfig
from kempnerforge.data.dataloader import StatefulDataLoader
from kempnerforge.data.dataset import MemoryMappedDataset
from kempnerforge.data.sampler import DistributedSampler


def _make_dataset(tmp_path, n_tokens: int = 2000, seq_len: int = 32):
    """Create a small mmap dataset."""
    tokens = np.arange(n_tokens, dtype=np.uint16)
    np.save(tmp_path / "data.npy", tokens)
    return MemoryMappedDataset(data_dir=str(tmp_path), seq_len=seq_len)


# Zero-worker config for tests (avoids multiprocessing overhead)
_TEST_CONFIG = DataConfig(num_workers=0, pin_memory=False)


class TestDataResumption:
    def test_dataloader_state_exact_resumption(self, tmp_path):
        """Dataloader produces same sequence after state_dict round-trip."""
        ds = _make_dataset(tmp_path, n_tokens=5000, seq_len=32)
        sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=42)
        loader = StatefulDataLoader(ds, batch_size=4, sampler=sampler, config=_TEST_CONFIG)

        # Consume 3 batches
        it = iter(loader)
        for _ in range(3):
            next(it)

        # Save state
        state = loader.state_dict()

        # Also get the next 3 from the original iterator
        batches_continued = []
        for _ in range(3):
            batches_continued.append(next(it))

        # Create new loader, restore, consume 3 more batches
        sampler2 = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=42)
        loader2 = StatefulDataLoader(ds, batch_size=4, sampler=sampler2, config=_TEST_CONFIG)
        loader2.load_state_dict(state)
        batches_after_resume = []
        it2 = iter(loader2)
        for _ in range(3):
            batches_after_resume.append(next(it2))

        # Resumed batches should match the continued batches
        for i, (b_cont, b_res) in enumerate(
            zip(batches_continued, batches_after_resume, strict=True)
        ):
            assert (b_cont["input_ids"] == b_res["input_ids"]).all(), (
                f"Batch {i} input_ids differ after resume"
            )
            assert (b_cont["labels"] == b_res["labels"]).all(), (
                f"Batch {i} labels differ after resume"
            )

    def test_sampler_skip_ahead(self, tmp_path):
        """Sampler set_skip produces same indices as continued iteration."""
        ds = _make_dataset(tmp_path, n_tokens=3000, seq_len=32)
        sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=42)

        # Get all indices for epoch 0
        all_indices = list(sampler)

        # Create new sampler with skip (skip first 10)
        sampler2 = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=42)
        sampler2.set_skip(10)
        remaining = list(sampler2)

        # remaining should be all_indices[10:]
        assert remaining == all_indices[10:], "Sampler skip-ahead does not match"

    def test_epoch_boundary_resumption(self, tmp_path):
        """Resumption works across epoch boundaries."""
        ds = _make_dataset(tmp_path, n_tokens=1000, seq_len=32)
        sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=42)
        loader = StatefulDataLoader(ds, batch_size=8, sampler=sampler, config=_TEST_CONFIG)

        # Exhaust the first epoch
        batches_epoch1 = list(loader)
        assert len(batches_epoch1) > 0

        state = loader.state_dict()
        assert state["epoch"] == 1  # epoch incremented after exhaustion

        # Resume — should start next epoch
        sampler2 = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=42)
        loader2 = StatefulDataLoader(ds, batch_size=8, sampler=sampler2, config=_TEST_CONFIG)
        loader2.load_state_dict(state)

        # Should be able to iterate
        batches_epoch2 = list(loader2)
        assert len(batches_epoch2) > 0
