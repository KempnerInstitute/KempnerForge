"""Distributed tests for checkpointing.

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_checkpoint.py -v
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from kempnerforge.checkpoint import manager as mgr_mod
from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.schema import CheckpointConfig, ModelConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import build_optimizer

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

SMALL_CONFIG = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=512, max_seq_len=64)

# Shared filesystem temp directory (visible to all nodes)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TEST_TMP = _PROJECT_ROOT / ".test_tmp"


@pytest.fixture
def shared_tmp_dir():
    """Create a temp directory on the shared filesystem, cleaned up after use."""
    rank = dist.get_rank()
    # Rank 0 creates, then broadcasts path
    if rank == 0:
        _TEST_TMP.mkdir(exist_ok=True)
        import tempfile

        d = tempfile.mkdtemp(dir=_TEST_TMP)
    else:
        d = ""
    obj_list = [d]
    dist.broadcast_object_list(obj_list, src=0)
    d = obj_list[0]
    yield d
    dist.barrier()
    if rank == 0:
        shutil.rmtree(d, ignore_errors=True)


class TestCheckpointRoundTrip:
    def test_save_load_fsdp(self, distributed_env, shared_tmp_dir):
        """Save and load under FSDP — model output should be identical."""
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir

        # Build model + FSDP
        torch.manual_seed(42)
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        from kempnerforge.config.schema import OptimizerConfig

        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        # Do a forward + backward to get non-trivial optimizer state
        tokens = torch.randint(0, 512, (2, 32), device="cuda")
        out = model(tokens)
        out.sum().backward()
        opt.step()
        opt.zero_grad()

        # Capture reference output
        with torch.no_grad():
            ref_out = model(tokens).clone()

        # Save
        config = CheckpointConfig(dir=ckpt_dir, keep_last_n=2)
        mgr = CheckpointManager(config, model, opt)
        mgr.save(step=1, tokens_seen=64)

        # Perturb model to verify load actually restores
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Verify output changed
        with torch.no_grad():
            perturbed_out = model(tokens)
        assert not torch.allclose(ref_out, perturbed_out, atol=1e-3)

        # Load
        mgr.load()

        # Verify output matches reference
        with torch.no_grad():
            restored_out = model(tokens)
        assert torch.allclose(ref_out, restored_out, atol=1e-5), (
            f"Restored output differs: max diff={(ref_out - restored_out).abs().max().item()}"
        )

    def test_latest_symlink(self, distributed_env, shared_tmp_dir):
        """The 'latest' symlink should point to the most recent checkpoint."""
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir

        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        from kempnerforge.config.schema import OptimizerConfig

        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        config = CheckpointConfig(dir=ckpt_dir, keep_last_n=3)
        mgr = CheckpointManager(config, model, opt)

        mgr.save(step=10)
        mgr.save(step=20)

        if dist.get_rank() == 0:
            latest = Path(ckpt_dir) / "latest"
            assert latest.exists()
            assert latest.resolve().name == "step_20"


class TestCheckpointSaveBarrier:
    """save() must synchronize all ranks on the rank-0 metadata writes."""

    def test_save_waits_for_rank0_writes(self, distributed_env, shared_tmp_dir):
        """Non-rank-0 must not return from save() before rank 0 finishes.

        Without the end-of-save barrier, non-rank-0 returns immediately
        after the async DCP dispatch, while rank 0 is still writing
        train_state.pt and the latest symlink. We force the race to
        be measurable by slowing rank-0's torch.save.
        """
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir
        rank = dist.get_rank()

        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        from kempnerforge.config.schema import OptimizerConfig

        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        cfg = CheckpointConfig(dir=ckpt_dir, keep_last_n=2)
        mgr = CheckpointManager(cfg, model, opt)

        real_torch_save = torch.save
        sleep_sec = 0.5

        def slow_on_rank0(*args, **kwargs):
            if rank == 0:
                time.sleep(sleep_sec)
            return real_torch_save(*args, **kwargs)

        # Barrier before timing so all ranks start save() at roughly the
        # same instant; isolates the signal we care about.
        dist.barrier()
        t0 = time.perf_counter()
        with patch.object(mgr_mod.torch, "save", side_effect=slow_on_rank0):
            mgr.save(step=1, tokens_seen=100)
        elapsed = time.perf_counter() - t0

        # Every rank must have waited for rank-0's slow write.
        assert elapsed >= 0.4 * sleep_sec, (
            f"rank {rank}: save() returned after {elapsed:.3f}s — the "
            f"end-of-save barrier is missing. Expected >= {0.4 * sleep_sec:.3f}s."
        )

        # And every rank observes rank-0's writes afterwards.
        step_dir = Path(ckpt_dir) / "step_1"
        assert (step_dir / "train_state.pt").exists(), (
            f"rank {rank}: train_state.pt not visible after save()"
        )
        assert (step_dir / "metadata.json").exists(), (
            f"rank {rank}: metadata.json not visible after save()"
        )


class TestCheckpointLoadDivergentExistence:
    """load() must not hang if ranks disagree about train_state.pt existence.

    Simulates attribute-cache skew (NFS/Lustre) by patching Path.exists
    so non-rank-0 sees a missing file. Rank-0's answer must be authoritative
    via broadcast; otherwise only some ranks enter the torch.load branch
    and the subsequent broadcast_object_list deadlocks.
    """

    def test_load_does_not_hang_on_divergent_exists(self, distributed_env, shared_tmp_dir):
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir
        rank = dist.get_rank()

        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        from kempnerforge.config.schema import OptimizerConfig

        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        cfg = CheckpointConfig(dir=ckpt_dir, keep_last_n=2)
        mgr = CheckpointManager(cfg, model, opt)

        # Save so there's something to load.
        mgr.save(step=1, tokens_seen=100)
        dist.barrier()

        # Patch exists() so non-rank-0 sees the file as missing. Without
        # the authoritative broadcast, non-rank-0 skips torch.load but
        # rank 0 enters and calls broadcast_object_list — deadlock.
        real_exists = Path.exists

        def skewed_exists(self):
            if rank != 0 and self.name == "train_state.pt":
                return False
            return real_exists(self)

        # Wrap in a timeout via a CUDA event + sleep loop would be ideal;
        # simpler: rely on the PG default timeout to surface a deadlock as
        # a RuntimeError rather than blocking the test runner forever.
        # With the fix, load() completes promptly; without, the test hangs
        # until the PG timeout fires.
        with patch.object(Path, "exists", skewed_exists):
            step, tokens_seen, _ = mgr.load()

        assert step == 1, f"rank {rank}: expected step=1 after load, got {step}"
        assert tokens_seen == 100, (
            f"rank {rank}: expected tokens_seen=100 after load, got {tokens_seen}"
        )
