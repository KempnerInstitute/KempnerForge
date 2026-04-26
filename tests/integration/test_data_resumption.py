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

    def test_double_resume_within_same_epoch(self, tmp_path):
        """Two save/load cycles within a single epoch must preserve alignment.

        Reproduces the pre-fix bug where batches_yielded was reset to 0 on
        every iter() call and not restored from state_dict. The second resume
        would re-skip only the delta of the middle run rather than the total
        epoch position.
        """
        ds = _make_dataset(tmp_path, n_tokens=30000, seq_len=32)
        bs = 4

        # Ground truth: one continuous run, capture batches 40..49.
        s_gt = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=42)
        L_gt = StatefulDataLoader(ds, batch_size=bs, sampler=s_gt, config=_TEST_CONFIG)
        it_gt = iter(L_gt)
        for _ in range(40):
            next(it_gt)
        ground_truth = [next(it_gt) for _ in range(10)]

        # Run 1: consume 30 batches, save.
        s1 = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=42)
        L1 = StatefulDataLoader(ds, batch_size=bs, sampler=s1, config=_TEST_CONFIG)
        it1 = iter(L1)
        for _ in range(30):
            next(it1)
        state1 = L1.state_dict()
        assert state1["batches_yielded"] == 30

        # Run 2: resume from state1, consume 10 more, save again.
        s2 = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=42)
        L2 = StatefulDataLoader(ds, batch_size=bs, sampler=s2, config=_TEST_CONFIG)
        L2.load_state_dict(state1)
        it2 = iter(L2)
        for _ in range(10):
            next(it2)
        state2 = L2.state_dict()

        # Pre-fix: state2["batches_yielded"] == 10. Post-fix: 40.
        assert state2["batches_yielded"] == 40, (
            f"Expected total epoch position 40, got {state2['batches_yielded']} "
            "— save/load/iter lost prior resume offset"
        )

        # Run 3: resume from state2, consume 10 more — must match ground_truth.
        s3 = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=42)
        L3 = StatefulDataLoader(ds, batch_size=bs, sampler=s3, config=_TEST_CONFIG)
        L3.load_state_dict(state2)
        it3 = iter(L3)
        resumed = [next(it3) for _ in range(10)]

        for i, (gt, r) in enumerate(zip(ground_truth, resumed, strict=True)):
            assert (gt["input_ids"] == r["input_ids"]).all(), (
                f"Batch {i} misaligned after double-resume"
            )
