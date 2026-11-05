"""Unit tests for the evaluation module."""

from __future__ import annotations

import math
import socket

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kempnerforge.training.eval import run_eval, should_build_eval_dataloader

VOCAB_SIZE = 64
DIM = 32
SEQ_LEN = 16


class _TinyLM(nn.Module):
    """Minimal LM: embedding → linear → logits. Accepts token IDs like a real model."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, dim: int = DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


def _make_eval_dataloader(
    batch_size: int = 4,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_SIZE,
    num_samples: int = 20,
) -> DataLoader:
    """Create a simple eval dataloader with random data."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, vocab_size, (num_samples, seq_len))
    dataset = TensorDataset(input_ids, labels)

    def collate_fn(batch):
        inputs = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        return {"input_ids": inputs, "labels": targets}

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def _loss_fn(logits, labels):
    return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


class TestRunEval:
    def test_returns_loss_and_perplexity(self):
        model = _TinyLM()
        dl = _make_eval_dataloader()
        metrics = run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=3)

        assert "eval/loss" in metrics
        assert "eval/perplexity" in metrics
        assert metrics["eval/loss"] > 0
        assert metrics["eval/perplexity"] > 1.0

    def test_perplexity_equals_exp_loss(self):
        model = _TinyLM()
        dl = _make_eval_dataloader()
        metrics = run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=3)
        expected_ppl = math.exp(min(metrics["eval/loss"], 20.0))
        assert abs(metrics["eval/perplexity"] - expected_ppl) < 1e-4

    def test_perfect_model_low_loss(self):
        """A model that always predicts the correct class should have low loss."""
        input_ids = torch.zeros(8, SEQ_LEN, dtype=torch.long)
        labels = torch.zeros(8, SEQ_LEN, dtype=torch.long)
        dataset = TensorDataset(input_ids, labels)

        def collate_fn(batch):
            inputs = torch.stack([b[0] for b in batch])
            targets = torch.stack([b[1] for b in batch])
            return {"input_ids": inputs, "labels": targets}

        dl = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        model = _TinyLM()
        with torch.no_grad():
            model.embed.weight.zero_()
            model.embed.weight[0, :] = 1.0
            model.head.weight.zero_()
            model.head.weight[0, :] = 10.0

        metrics = run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=2)
        assert metrics["eval/loss"] < 0.1
        assert metrics["eval/perplexity"] < 1.2

    def test_model_restored_to_train_mode(self):
        """run_eval should restore model.train() after evaluation."""
        model = _TinyLM()
        model.train()
        dl = _make_eval_dataloader()
        run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=1)
        assert model.training is True

    def test_wraps_around_short_dataloader(self):
        """If dataloader has fewer batches than eval_steps, it should wrap around."""
        model = _TinyLM()
        dl = _make_eval_dataloader(num_samples=4, batch_size=4)
        metrics = run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=5)
        assert metrics["eval/loss"] > 0

    def test_no_gradients_computed(self):
        """run_eval should not accumulate gradients."""
        model = _TinyLM()
        dl = _make_eval_dataloader()
        run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=2)
        for param in model.parameters():
            assert param.grad is None


class TestShouldBuildEvalDataloader:
    """Gating logic for `eval.enabled` across text-only vs VLM configs.

    VLM configs must not build an eval dataloader on this branch because
    ``run_eval`` calls ``model(input_ids)`` which does not match
    ``VLMWrapper.forward(pixel_values, input_ids, labels)``.
    """

    def test_text_only_enabled_builds(self):
        assert should_build_eval_dataloader(True, False) == (True, False)

    def test_text_only_disabled_skips(self):
        assert should_build_eval_dataloader(False, False) == (False, False)

    def test_vlm_enabled_warns_and_skips(self):
        """is_vlm=True + eval.enabled=True → skip eval, flag the warning."""
        assert should_build_eval_dataloader(True, True) == (False, True)

    def test_vlm_disabled_silent(self):
        """is_vlm=True + eval.enabled=False → silent no-op."""
        assert should_build_eval_dataloader(False, True) == (False, False)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _check_two_stage_eval(rank: int, world: int, port: int, schedule_name: str) -> None:
    """A training step, eval through the same two-stage schedule against the whole model,
    then another training step, as the training loop runs them."""
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.device_mesh import init_device_mesh

    from kempnerforge.config.schema import ModelConfig
    from kempnerforge.distributed.pipeline_parallel import (
        build_pipeline_schedule,
        build_pipeline_stage,
        build_stage_module,
    )
    from kempnerforge.model.transformer import Transformer

    def loss_fn(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out.flatten(0, 1).float(), target.flatten())

    n_micro, micro_batch, seq = 2, 2, 16
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("pp",))
        config = ModelConfig(dim=32, n_layers=2, n_heads=2, vocab_size=VOCAB_SIZE, max_seq_len=seq)
        torch.manual_seed(0)
        reference = Transformer(config)
        stage_module = build_stage_module(config, pp_rank=rank, pp_size=world)
        own = stage_module.state_dict()
        subset = {k: v for k, v in reference.state_dict().items() if k in own}
        stage_module.load_state_dict(subset, strict=True)
        stage = build_pipeline_stage(
            stage_module,
            mesh,
            torch.device("cpu"),
            batch_size=micro_batch,
            seq_len=seq,
            param_dtype=torch.float32,
        )
        schedule = build_pipeline_schedule(
            stage, n_microbatches=n_micro, loss_fn=loss_fn, schedule=schedule_name
        )

        generator = torch.Generator().manual_seed(7)
        batches = [
            {
                "input_ids": torch.randint(0, VOCAB_SIZE, (micro_batch, seq), generator=generator),
                "labels": torch.randint(0, VOCAB_SIZE, (micro_batch, seq), generator=generator),
            }
            for _ in range(2 * n_micro)
        ]
        tokens = torch.cat([b["input_ids"] for b in batches[:n_micro]])
        labels = torch.cat([b["labels"] for b in batches[:n_micro]])

        def train_step() -> None:
            if rank == 0:
                schedule.step(tokens, target=labels, losses=[])
            else:
                schedule.step(target=labels, losses=[])

        train_step()
        stage_module.zero_grad()
        metrics = run_eval(
            stage_module,
            batches,  # type: ignore[arg-type]
            loss_fn,
            torch.device("cpu"),
            2 * n_micro,
            pp_schedule=schedule,
            pp_rank=rank,
            pp_size=world,
            pp_group=mesh["pp"].get_group(),
            pp_n_microbatches=n_micro,
        )
        with torch.no_grad():
            expected = sum(
                loss_fn(reference(b["input_ids"]), b["labels"]).item() for b in batches
            ) / len(batches)
        assert math.isclose(metrics["eval/loss"], expected, rel_tol=1e-5), (metrics, expected)
        assert all(p.grad is None for p in stage_module.parameters())
        assert stage_module.training

        train_step()
        assert all(p.grad is not None for p in stage_module.parameters())
    finally:
        dist.destroy_process_group()


def _check_two_stage_fsdp_eval(rank: int, world: int, port: int, schedule_name: str) -> None:
    """PP 2 x FSDP2 dp_shard 2: a training step, eval, then a step whose gradients match the
    whole model's, so eval between steps leaves the sharded state and gradients alone."""
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from kempnerforge.config.schema import ModelConfig
    from kempnerforge.distributed.parallel import apply_fsdp2
    from kempnerforge.distributed.pipeline_parallel import (
        build_pipeline_schedule,
        build_pipeline_stage,
        build_stage_module,
    )
    from kempnerforge.model.transformer import Transformer

    def loss_fn(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out.flatten(0, 1).float(), target.flatten())

    n_micro, micro_batch, seq = 2, 2, 16
    per_dp_batch = n_micro * micro_batch
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("pp", "dp_shard"))
        pp_rank, dp_rank = mesh.get_local_rank("pp"), mesh.get_local_rank("dp_shard")
        config = ModelConfig(dim=32, n_layers=2, n_heads=2, vocab_size=VOCAB_SIZE, max_seq_len=seq)
        torch.manual_seed(0)
        reference = Transformer(config)
        stage_module = build_stage_module(config, pp_rank=pp_rank, pp_size=2)
        own = stage_module.state_dict()
        subset = {k: v for k, v in reference.state_dict().items() if k in own}
        stage_module.load_state_dict(subset, strict=True)
        fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        apply_fsdp2(stage_module, mesh, mp_policy=fp32)
        stage = build_pipeline_stage(
            stage_module,
            mesh,
            torch.device("cpu"),
            batch_size=micro_batch,
            seq_len=seq,
            param_dtype=torch.float32,
        )
        schedule = build_pipeline_schedule(
            stage, n_microbatches=n_micro, loss_fn=loss_fn, schedule=schedule_name
        )

        generator = torch.Generator().manual_seed(7)
        tokens = torch.randint(0, VOCAB_SIZE, (2 * per_dp_batch, seq), generator=generator)
        labels = torch.randint(0, VOCAB_SIZE, (2 * per_dp_batch, seq), generator=generator)
        rows = slice(dp_rank * per_dp_batch, (dp_rank + 1) * per_dp_batch)
        batches = [
            {
                "input_ids": torch.randint(0, VOCAB_SIZE, (micro_batch, seq), generator=generator),
                "labels": torch.randint(0, VOCAB_SIZE, (micro_batch, seq), generator=generator),
            }
            for _ in range(2 * n_micro)
        ]

        def train_step() -> None:
            if pp_rank == 0:
                schedule.step(tokens[rows], target=labels[rows], losses=[])
            else:
                schedule.step(target=labels[rows], losses=[])

        train_step()
        stage_module.zero_grad()
        metrics = run_eval(
            stage_module,
            batches,  # type: ignore[arg-type]
            loss_fn,
            torch.device("cpu"),
            2 * n_micro,
            pp_schedule=schedule,
            pp_rank=pp_rank,
            pp_size=2,
            pp_group=mesh["pp"].get_group(),
            pp_n_microbatches=n_micro,
        )
        with torch.no_grad():
            expected = sum(
                loss_fn(reference(b["input_ids"]), b["labels"]).item() for b in batches
            ) / len(batches)
        assert math.isclose(metrics["eval/loss"], expected, rel_tol=1e-5), (metrics, expected)
        assert all(p.grad is None for p in stage_module.parameters())

        train_step()
        loss_fn(reference(tokens), labels).backward()
        for name, p in stage_module.named_parameters():
            torch.testing.assert_close(
                p.grad.full_tensor(),  # type: ignore[union-attr]
                reference.get_parameter(name).grad,
                rtol=2e-4,
                atol=2e-5,
            )
    finally:
        dist.destroy_process_group()


class _NoBatchSchedule:
    """A pipeline schedule that must not be reached."""

    def step(self, *args, **kwargs):
        raise AssertionError("step() called")

    def eval(self, *args, **kwargs):
        raise AssertionError("eval() called")


class TestRunEvalPipeline:
    @pytest.mark.parametrize("schedule_name", ["1f1b", "gpipe"])
    def test_two_stage_eval_matches_the_whole_model_and_training_continues(self, schedule_name):
        """Under no_grad, run_eval drives the training schedule through eval(): torch refuses
        step() there for a schedule with a loss function. The batches go through the
        schedule two at a time, the training microbatch shape, eval computes no gradients,
        and the schedule trains again afterward."""
        import torch.multiprocessing as mp

        mp.spawn(_check_two_stage_eval, args=(2, _free_port(), schedule_name), nprocs=2)

    @pytest.mark.parametrize("schedule_name", ["1f1b", "gpipe"])
    def test_eval_between_fsdp2_sharded_steps_keeps_the_training_gradients(self, schedule_name):
        """With FSDP2 inside each stage, the schedule's gradient reduction still runs in
        eval(); with the gradients cleared it changes nothing, and the next step's sharded
        gradients equal the whole model's."""
        import torch.multiprocessing as mp

        mp.spawn(_check_two_stage_fsdp_eval, args=(4, _free_port(), schedule_name), nprocs=4)

    def test_eval_steps_must_fill_whole_schedule_calls(self):
        with pytest.raises(ValueError, match="multiple of the pipeline schedule's 2 microbatches"):
            run_eval(
                _TinyLM(),
                _make_eval_dataloader(),
                _loss_fn,
                torch.device("cpu"),
                eval_steps=3,
                pp_schedule=_NoBatchSchedule(),
                pp_rank=0,
                pp_size=2,
                pp_n_microbatches=2,
            )

    def test_a_loader_with_no_batch_raises(self):
        empty = DataLoader(
            TensorDataset(torch.zeros(1, SEQ_LEN, dtype=torch.long)), batch_size=4, drop_last=True
        )
        with pytest.raises(ValueError, match="yields no batch"):
            run_eval(
                _TinyLM(),
                empty,
                _loss_fn,
                torch.device("cpu"),
                eval_steps=2,
                pp_schedule=_NoBatchSchedule(),
                pp_rank=0,
                pp_size=2,
                pp_n_microbatches=2,
            )
