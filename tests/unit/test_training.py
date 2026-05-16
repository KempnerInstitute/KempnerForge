"""Unit tests for KempnerForge training components."""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.schema import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    SchedulerType,
)
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.scheduler import build_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TINY_CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=32)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_creates_adamw(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig())
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig())
        assert len(opt.param_groups) == 2

    def test_weight_decay_split(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig(weight_decay=0.1))
        # Group 0: with decay, Group 1: no decay
        assert opt.param_groups[0]["weight_decay"] == 0.1
        assert opt.param_groups[1]["weight_decay"] == 0.0

    def test_no_decay_for_norms(self):
        """Norm parameters (1D) should be in the no-decay group."""
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig())
        no_decay_params = set(id(p) for p in opt.param_groups[1]["params"])

        for name, param in model.named_parameters():
            if param.ndim <= 1:
                assert id(param) in no_decay_params, f"{name} should be in no-decay group"

    def test_lr_is_set(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-5))
        for group in opt.param_groups:
            assert group["lr"] == 1e-5

    def test_unknown_optimizer_raises(self):
        model = Transformer(TINY_CONFIG)
        with pytest.raises(KeyError, match="Unknown optimizer"):
            build_optimizer(model, OptimizerConfig(name="sgd"))

    def test_all_params_accounted(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig())
        opt_param_ids = set()
        for group in opt.param_groups:
            for p in group["params"]:
                opt_param_ids.add(id(p))
        model_param_ids = {id(p) for p in model.parameters() if p.requires_grad}
        assert opt_param_ids == model_param_ids


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TestScheduler:
    def test_cosine_warmup_start(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig(lr=1.0))
        sched = build_scheduler(opt, SchedulerConfig(warmup_steps=100), max_steps=1000)
        # At init, LambdaLR applies lr_fn(0) = 0/100 = 0
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0)
        sched.step()
        # After 1 step: lr_fn(1) = 1/100 = 0.01
        assert opt.param_groups[0]["lr"] == pytest.approx(0.01, abs=1e-5)

    def test_cosine_warmup_end(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig(lr=1.0))
        sched = build_scheduler(opt, SchedulerConfig(warmup_steps=100), max_steps=1000)
        for _ in range(100):
            sched.step()
        # At step 100, should be at peak LR
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=0.01)

    def test_cosine_decay_end(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig(lr=1.0))
        config = SchedulerConfig(warmup_steps=0, min_lr_ratio=0.1)
        sched = build_scheduler(opt, config, max_steps=1000)
        for _ in range(1000):
            sched.step()
        # At end, should be at min_lr
        assert opt.param_groups[0]["lr"] == pytest.approx(0.1, abs=0.02)

    def test_linear_decay(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig(lr=1.0))
        config = SchedulerConfig(name=SchedulerType.linear, warmup_steps=0, min_lr_ratio=0.0)
        sched = build_scheduler(opt, config, max_steps=100)
        for _ in range(50):
            sched.step()
        # Midpoint: should be ~0.5
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5, abs=0.02)

    def test_wsd_stable_phase(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig(lr=1.0))
        config = SchedulerConfig(
            name=SchedulerType.wsd,
            warmup_steps=10,
            stable_steps=80,
            decay_steps=10,
            min_lr_ratio=0.0,
        )
        sched = build_scheduler(opt, config, max_steps=100)
        # Step through warmup
        for _ in range(10):
            sched.step()
        # During stable phase, LR should be 1.0
        for _ in range(80):
            sched.step()
            assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=0.01)

    def test_wsd_decay_phase(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig(lr=1.0))
        config = SchedulerConfig(
            name=SchedulerType.wsd,
            warmup_steps=10,
            stable_steps=80,
            decay_steps=10,
            min_lr_ratio=0.0,
        )
        sched = build_scheduler(opt, config, max_steps=100)
        for _ in range(100):
            sched.step()
        # At end of decay, should be at min_lr
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=0.02)

    def test_unknown_scheduler_raises(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig())
        with pytest.raises(KeyError, match="Unknown scheduler"):
            build_scheduler(opt, SchedulerConfig(name="invalid"), max_steps=100)

    def test_zero_warmup(self):
        model = Transformer(TINY_CONFIG)
        opt = build_optimizer(model, OptimizerConfig(lr=1.0))
        sched = build_scheduler(opt, SchedulerConfig(warmup_steps=0), max_steps=100)
        sched.step()
        # Should start at peak LR (no warmup)
        assert opt.param_groups[0]["lr"] > 0.9


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------


class TestGradUtils:
    def test_maybe_no_sync_last_step_yields(self):
        """On the last micro-step, should yield normally."""
        from kempnerforge.training.grad import maybe_no_sync

        model = Transformer(TINY_CONFIG)
        with maybe_no_sync(model, micro_step=2, grad_accum_steps=3):
            pass  # Should not raise

    def test_maybe_no_sync_intermediate_step(self):
        """On intermediate steps, should yield (no-op for non-FSDP models)."""
        from kempnerforge.training.grad import maybe_no_sync

        model = Transformer(TINY_CONFIG)
        with maybe_no_sync(model, micro_step=0, grad_accum_steps=3):
            pass  # Should not raise (no set_requires_gradient_sync on vanilla model)

    def test_maybe_no_sync_fsdp2_branch(self):
        """When the model has set_requires_gradient_sync (FSDP2), the wrapper
        toggles sync off on entry and back on after exit."""
        from kempnerforge.training.grad import maybe_no_sync

        class _FakeFSDPModel:
            def __init__(self):
                self.calls = []

            def set_requires_gradient_sync(self, val):
                self.calls.append(val)

        model = _FakeFSDPModel()
        with maybe_no_sync(model, micro_step=0, grad_accum_steps=3):
            assert model.calls == [False]
        assert model.calls == [False, True]


# ---------------------------------------------------------------------------
# Integration: loss decreases
# ---------------------------------------------------------------------------


class TestTrainingIntegration:
    def test_loss_decreases(self):
        """Train a tiny model on a repeating batch — loss should decrease."""
        model = Transformer(TINY_CONFIG).to(DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        # Fixed repeating batch (same data each step → model can learn it)
        torch.manual_seed(123)
        tokens = torch.randint(0, 256, (4, 32), device=DEVICE)
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]

        model.train()
        losses = []

        for _ in range(100):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.reshape(-1)
            )
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())

        avg_first = sum(losses[:10]) / 10
        avg_last = sum(losses[-10:]) / 10
        assert avg_last < avg_first, (
            f"Loss did not decrease: first 10 avg={avg_first:.4f}, last 10 avg={avg_last:.4f}"
        )

    def test_memorization(self):
        """A tiny model should memorize a single batch."""
        model = Transformer(TINY_CONFIG).to(DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=3e-3, fused=False))

        # Fixed batch
        torch.manual_seed(42)
        tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]

        model.train()
        for _ in range(200):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.reshape(-1)
            )
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Loss should be very low after memorizing one batch
        assert loss.item() < 1.0, f"Model failed to memorize: loss={loss.item():.4f}"
