"""Pipeline-parallel equivalence against a single-GPU forward.

Run with: torchrun --nproc_per_node=2 -m pytest tests/distributed/test_pp.py
(or with 4 ranks)
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

CONFIG = ModelConfig(dim=128, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=128)
N_MICROBATCHES = max(2, int(os.environ.get("WORLD_SIZE", "2")))
BATCH, SEQ = 2 * N_MICROBATCHES, 128


def _doc_ids(batch: int, seq_len: int, device) -> torch.Tensor:
    """Two documents per row, split down the middle."""
    half = seq_len // 2
    return torch.tensor([[0] * half + [1] * (seq_len - half)] * batch, device=device)


class TestPipelineParallelEquivalence:
    """Splitting a model across pipeline stages must not change its output.

    The packed cases check both halves of the contract separately: that the
    masking is right (forward parity against a single GPU) and that the run can
    actually train (a real step with backward). Only the second reaches the code
    that an integer tensor on the pipe breaks.
    """

    def _stage_and_reference(self, device, attention_backend: str = "sdpa"):
        """One set of weights, materialized as a full model and as this rank's stage."""
        from dataclasses import replace

        config = replace(CONFIG, attention_backend=attention_backend)
        torch.manual_seed(0)
        reference = Transformer(config)
        ref_state = reference.state_dict()

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        stage_module = build_stage_module(config, pp_rank=rank, pp_size=world)

        own = stage_module.state_dict()
        subset = {k: v for k, v in ref_state.items() if k in own}
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
                return
            pipeline_out = schedule.step()
            single_gpu_out = reference(tokens)

        if rank == world - 1:
            torch.testing.assert_close(pipeline_out, single_gpu_out, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("attention_backend", ["sdpa", "flex"])
    def test_packed_pipeline_matches_single_gpu(self, attention_backend):
        """pp=2 with doc_ids matches a single-GPU packed forward.

        This is the assertion the old ``pp > 1 + pack_sequences`` rejection
        existed to stand in for. Before doc_ids reached the stages, the pipeline
        silently produced *unpacked causal* output instead -- the loss looked
        fine because labels still carry -100 at boundaries, so nothing in a
        training curve would have revealed it.

        Forward only. ``test_packed_pipeline_training_step`` covers backward,
        and the pair is deliberate: a no-loss_fn schedule leaves
        ``has_backward=False``, which skips whole regions of ``PipelineStage``.
        """
        from torch.distributed.pipelining import PipelineStage
        from torch.distributed.pipelining.schedules import ScheduleGPipe

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        stage_module, reference = self._stage_and_reference(
            device, attention_backend=attention_backend
        )
        tokens = torch.randint(
            0, CONFIG.vocab_size, (BATCH, SEQ), generator=torch.Generator().manual_seed(7)
        ).to(device)
        doc_ids = _doc_ids(BATCH, SEQ, device)

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
                schedule.step(tokens, doc_ids=doc_ids)
                return
            pipeline_out = schedule.step(doc_ids=doc_ids)
            packed = reference(tokens, doc_ids=doc_ids)
            unpacked = reference(tokens)

        if rank == world - 1:
            torch.testing.assert_close(pipeline_out, packed, rtol=1e-5, atol=1e-5)
            assert not torch.allclose(packed, unpacked, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("attention_backend", ["sdpa", "flex"])
    def test_packed_pipeline_gradients_match_single_gpu(self, attention_backend):
        """Backward parity, not merely backward liveness.

        ``test_packed_pipeline_training_step`` asserts that gradients exist and
        are finite, which a backward applying the *wrong* mask would also
        satisfy. This pins the values: the mask has to be right in backward as
        well as forward.

        The reference is a single full-batch backward. That is what the pipeline
        reproduces: the schedule divides accumulated gradients by
        ``n_microbatches`` (``scale_grads`` defaults to True, which torch
        documents as the setting to use with a mean-reduction loss), so the
        result is the *average* over micro-batches, not the sum. For equal
        micro-batches that equals the gradient of the full-batch mean loss.

        An earlier version of this test summed per-micro-batch losses instead,
        and failed by exactly a factor of ``n_microbatches``.
        """
        import torch.nn.functional as F

        from kempnerforge.config.schema import DistributedConfig
        from kempnerforge.distributed.pipeline_parallel import (
            build_pipeline_schedule,
            build_pipeline_stage,
        )
        from kempnerforge.distributed.setup import init_distributed

        def loss_fn(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(out.flatten(0, 1).float(), target.flatten())

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        mesh = init_distributed(DistributedConfig(dp_shard=1, tp=1, pp=world), seed=42)

        stage_module, reference = self._stage_and_reference(
            device, attention_backend=attention_backend
        )
        stage = build_pipeline_stage(
            stage_module,
            mesh,
            device,
            batch_size=BATCH // N_MICROBATCHES,
            seq_len=SEQ,
            param_dtype=torch.float32,
        )
        schedule = build_pipeline_schedule(
            stage,
            n_microbatches=N_MICROBATCHES,
            loss_fn=loss_fn,
            schedule="1f1b",
        )

        gen = torch.Generator().manual_seed(7)
        tokens = torch.randint(0, CONFIG.vocab_size, (BATCH, SEQ), generator=gen).to(device)
        labels = torch.randint(0, CONFIG.vocab_size, (BATCH, SEQ), generator=gen).to(device)
        doc_ids = _doc_ids(BATCH, SEQ, device)

        stage_module.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        if rank == 0:
            schedule.step(tokens, doc_ids=doc_ids, target=labels, losses=losses)
        elif rank == world - 1:
            schedule.step(doc_ids=doc_ids, target=labels, losses=losses)
        else:
            schedule.step(doc_ids=doc_ids)

        reference.zero_grad(set_to_none=True)
        loss_fn(reference(tokens, doc_ids=doc_ids), labels).backward()

        compared, largest = 0, 0.0
        for name, param in stage_module.named_parameters():
            ref_grad = reference.get_parameter(name).grad
            assert param.grad is not None, f"rank {rank}: {name} has no gradient"
            assert ref_grad is not None, f"rank {rank}: reference {name} has no gradient"
            torch.testing.assert_close(
                param.grad, ref_grad, rtol=2e-4, atol=2e-5, msg=lambda m, n=name: f"{n}: {m}"
            )
            largest = max(largest, ref_grad.abs().max().item())
            compared += 1

        assert compared, f"rank {rank}: compared no parameters"
        assert largest > 1e-6, f"rank {rank}: reference gradients are ~zero ({largest:.2e})"

    @pytest.mark.parametrize("schedule_name", ["1f1b", "gpipe"])
    @pytest.mark.parametrize("attention_backend", ["sdpa", "flex"])
    def test_packed_pipeline_training_step(self, schedule_name, attention_backend):
        """A packed pp=2 step must complete forward *and* backward.

        Forward-only coverage cannot see this. A schedule built without
        ``loss_fn`` leaves ``has_backward=False``, and that flag gates
        ``_prepare_forward_infra``, which calls ``requires_grad_(True)`` on every
        received buffer with no dtype check -- so an integer tensor sent between
        stages raises only once a real backward is wired up. ``1f1b`` is the
        production default and is checked first for that reason.
        """
        import torch.nn.functional as F

        from kempnerforge.config.schema import DistributedConfig
        from kempnerforge.distributed.pipeline_parallel import (
            build_pipeline_schedule,
            build_pipeline_stage,
        )
        from kempnerforge.distributed.setup import init_distributed

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        mesh = init_distributed(DistributedConfig(dp_shard=1, tp=1, pp=world), seed=42)

        stage_module, _ = self._stage_and_reference(device, attention_backend=attention_backend)
        stage_module.train()
        stage = build_pipeline_stage(
            stage_module,
            mesh,
            device,
            batch_size=BATCH // N_MICROBATCHES,
            seq_len=SEQ,
            param_dtype=torch.float32,
        )
        schedule = build_pipeline_schedule(
            stage,
            n_microbatches=N_MICROBATCHES,
            loss_fn=lambda out, tgt: F.cross_entropy(out.flatten(0, 1).float(), tgt.flatten()),
            schedule=schedule_name,
        )

        gen = torch.Generator().manual_seed(7)
        tokens = torch.randint(0, CONFIG.vocab_size, (BATCH, SEQ), generator=gen).to(device)
        labels = torch.randint(0, CONFIG.vocab_size, (BATCH, SEQ), generator=gen).to(device)
        doc_ids = _doc_ids(BATCH, SEQ, device)

        losses: list[torch.Tensor] = []
        if rank == 0:
            schedule.step(tokens, doc_ids=doc_ids, target=labels, losses=losses)
        elif rank == world - 1:
            schedule.step(doc_ids=doc_ids, target=labels, losses=losses)
        else:
            schedule.step(doc_ids=doc_ids)

        grads = [p.grad for p in stage_module.parameters() if p.grad is not None]
        assert grads, f"rank {rank}: backward produced no gradients"
        assert all(torch.isfinite(g).all() for g in grads), f"rank {rank}: non-finite gradient"

        if rank == world - 1:
            assert losses, "last stage recorded no loss"
            assert all(torch.isfinite(loss) for loss in losses), f"non-finite loss: {losses}"

    def test_in_loop_eval_matches_single_gpu_and_training_continues(self):
        """run_eval drives the training schedule, built with a loss function, under
        no_grad, which torch allows only through eval(). Between two training steps,
        as in the loop, the eval batches go through it N_MICROBATCHES at a time, the
        shape the stages were built for, and eval computes no gradients."""
        import torch.nn.functional as F

        from kempnerforge.config.schema import DistributedConfig
        from kempnerforge.distributed.pipeline_parallel import (
            build_pipeline_schedule,
            build_pipeline_stage,
        )
        from kempnerforge.distributed.setup import init_distributed
        from kempnerforge.training.eval import run_eval

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        mesh = init_distributed(DistributedConfig(dp_shard=1, tp=1, pp=world), seed=42)

        def loss_fn(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(out.flatten(0, 1).float(), target.flatten())

        stage_module, reference = self._stage_and_reference(device)
        stage_module.train()
        micro = BATCH // N_MICROBATCHES
        stage = build_pipeline_stage(
            stage_module, mesh, device, batch_size=micro, seq_len=SEQ, param_dtype=torch.float32
        )
        schedule = build_pipeline_schedule(
            stage, n_microbatches=N_MICROBATCHES, loss_fn=loss_fn, schedule="1f1b"
        )

        gen = torch.Generator().manual_seed(7)
        batches = [
            {
                "input_ids": torch.randint(0, CONFIG.vocab_size, (micro, SEQ), generator=gen),
                "labels": torch.randint(0, CONFIG.vocab_size, (micro, SEQ), generator=gen),
            }
            for _ in range(2 * N_MICROBATCHES)
        ]
        tokens = torch.cat([b["input_ids"] for b in batches[:N_MICROBATCHES]]).to(device)
        labels = torch.cat([b["labels"] for b in batches[:N_MICROBATCHES]]).to(device)

        def train_step() -> None:
            if rank == 0:
                schedule.step(tokens, target=labels, losses=[])
            elif rank == world - 1:
                schedule.step(target=labels, losses=[])
            else:
                schedule.step()

        train_step()
        stage_module.zero_grad()
        metrics = run_eval(
            stage_module,
            batches,
            loss_fn,
            device,
            2 * N_MICROBATCHES,
            pp_schedule=schedule,
            pp_rank=rank,
            pp_size=world,
            pp_group=mesh["pp"].get_group(),
            pp_n_microbatches=N_MICROBATCHES,
        )
        with torch.no_grad():
            expected = sum(
                loss_fn(reference(b["input_ids"].to(device)), b["labels"].to(device)).item()
                for b in batches
            ) / len(batches)
        assert abs(metrics["eval/loss"] - expected) <= 1e-5 * abs(expected), (metrics, expected)
        assert all(p.grad is None for p in stage_module.parameters())

        train_step()
        assert all(p.grad is not None for p in stage_module.parameters()), f"rank {rank}"

    def test_fp32_master_weights_step_under_autocast(self) -> None:
        """A pp-only mesh has no FSDP2 wrap, so fp32 master weights train under autocast."""
        from kempnerforge.config.schema import DistributedConfig, JobConfig, TrainConfig
        from kempnerforge.distributed.parallel import autocast_dtype
        from kempnerforge.distributed.setup import init_distributed
        from kempnerforge.training.entry import build_model
        from kempnerforge.training.loss import build_loss_fn
        from kempnerforge.training.runtime import RuntimeContext

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        mesh = init_distributed(DistributedConfig(dp_shard=1, tp=1, pp=world), seed=42)
        config = JobConfig(
            model=CONFIG,
            train=TrainConfig(
                batch_size=BATCH // N_MICROBATCHES,
                seq_len=SEQ,
                grad_accum_steps=N_MICROBATCHES,
                compile_model=False,
                fp32_master_weights=True,
            ),
            distributed=DistributedConfig(dp_shard=1, tp=1, pp=world),
        )
        runtime = RuntimeContext(
            rank=rank, local_rank=rank, world_size=world, device=device, device_mesh=mesh
        )
        model, pipeline = build_model(config, runtime, build_loss_fn(config.train))
        assert pipeline is not None
        assert all(p.dtype == torch.float32 for p in model.parameters())
        amp_dtype = autocast_dtype(model, torch.float32, torch.bfloat16)
        assert amp_dtype == torch.bfloat16

        gen = torch.Generator().manual_seed(7)
        tokens = torch.randint(0, CONFIG.vocab_size, (BATCH, SEQ), generator=gen).to(device)
        labels = torch.randint(0, CONFIG.vocab_size, (BATCH, SEQ), generator=gen).to(device)
        losses: list[torch.Tensor] = []
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            if rank == 0:
                pipeline.schedule.step(tokens, target=labels, losses=losses)
            elif rank == world - 1:
                pipeline.schedule.step(target=labels, losses=losses)
            else:
                pipeline.schedule.step()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, f"rank {rank}: backward produced no gradients"
        assert all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in grads)
        if rank == world - 1:
            assert losses and all(torch.isfinite(loss) for loss in losses), f"losses: {losses}"
