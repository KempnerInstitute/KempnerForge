"""Distributed tests for checkpointing.

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_checkpoint.py -v
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.schema import CheckpointConfig, ModelConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import build_optimizer

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

SMALL_CONFIG = ModelConfig(
    dim=128, n_layers=2, n_heads=2, vocab_size=512, max_seq_len=64
)

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
            f"Restored output differs: max diff={( ref_out - restored_out).abs().max().item()}"
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
