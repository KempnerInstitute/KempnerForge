"""Pipeline-parallel equivalence against a single-GPU forward.

Run with: torchrun --nproc_per_node=2 -m pytest tests/distributed/test_pp.py
"""

from __future__ import annotations

import os

import pytest
import torch

from kempnerforge.config.schema import ModelConfig
from kempnerforge.distributed.pipeline_parallel import build_stage_module
from kempnerforge.model.transformer import Transformer

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="requires torchrun with >= 2 GPUs",
)

CONFIG = ModelConfig(dim=128, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=64)
BATCH, SEQ, N_MICROBATCHES = 4, 32, 2


class TestPipelineParallelEquivalence:
    """Splitting a model across pipeline stages must not change its output.

    There is no packed counterpart here on purpose: ``PipelineStageModule.forward``
    takes only hidden states, so ``doc_ids`` never reaches the stages and a packed
    batch would train with cross-document attention. ``JobConfig.validate`` rejects
    that combination outright -- see
    ``tests/unit/test_config.py::TestJobConfig::test_validate_packing_with_pp_rejected``.
    """

    def _stage_and_reference(self, device):
        """One set of weights, materialized as a full model and as this rank's stage."""
        torch.manual_seed(0)
        reference = Transformer(CONFIG)
        ref_state = reference.state_dict()

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        stage_module = build_stage_module(CONFIG, pp_rank=rank, pp_size=world)

        own = stage_module.state_dict()
        subset = {k: v for k, v in ref_state.items() if k in own}
        # A partial load would silently look like agreement, so pin the key sets.
        assert set(subset) == set(own), f"rank {rank}: missing {set(own) - set(subset)}"
        stage_module.load_state_dict(subset, strict=True)

        reference.load_state_dict(ref_state)
        return stage_module.to(device).eval(), reference.to(device).eval()

    def test_pipeline_output_matches_single_gpu(self):
        """pp=2 logits match a single-GPU forward over the same weights."""
        from torch.distributed.pipelining import PipelineStage
        from torch.distributed.pipelining.schedules import ScheduleGPipe

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        stage_module, reference = self._stage_and_reference(device)
        tokens = torch.randint(
            0, CONFIG.vocab_size, (BATCH, SEQ), generator=torch.Generator().manual_seed(7)
        ).to(device)

        micro = BATCH // N_MICROBATCHES
        example = (
            torch.zeros(micro, SEQ, dtype=torch.long, device=device)
            if rank == 0
            else torch.zeros(micro, SEQ, CONFIG.dim, dtype=torch.float32, device=device)
        )
        stage = PipelineStage(
            submodule=stage_module,
            stage_index=rank,
            num_stages=world,
            device=device,
            input_args=(example,),
        )
        schedule = ScheduleGPipe(stage, n_microbatches=N_MICROBATCHES)

        with torch.no_grad():
            if rank == 0:
                schedule.step(tokens)
                return  # only the last stage produces logits
            pipeline_out = schedule.step()
            single_gpu_out = reference(tokens)

        if rank == world - 1:
            torch.testing.assert_close(pipeline_out, single_gpu_out, rtol=1e-5, atol=1e-5)
