"""Unit tests for Lion and Schedule-Free AdamW optimizers (Phase 9)."""

from __future__ import annotations

import torch
import torch.nn as nn

from kempnerforge.config.schema import OptimizerConfig, SchedulerConfig, SchedulerType
from kempnerforge.training.optimizer import (
    Lion,
    ScheduleFreeAdamW,
    build_optimizer,
)
from kempnerforge.training.scheduler import build_scheduler


def _make_model():
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


# ---------------------------------------------------------------------------
# Lion optimizer
# ---------------------------------------------------------------------------


class TestLion:
    def test_build_via_registry(self):
        model = nn.Linear(32, 10)
        config = OptimizerConfig(name="lion", lr=1e-4)
        optimizer = build_optimizer(model, config)
        assert isinstance(optimizer, Lion)

    def test_decreases_loss(self):
        """Lion should decrease loss on a simple problem."""
        torch.manual_seed(42)
        model = _make_model()
        x = torch.randn(16, 32)
        y = torch.randint(0, 10, (16,))

        config = OptimizerConfig(name="lion", lr=1e-4, weight_decay=0.0)
        optimizer = build_optimizer(model, config)

        initial_loss = nn.functional.cross_entropy(model(x), y).item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.cross_entropy(model(x), y).item()
        assert final_loss < initial_loss

    def test_single_momentum_buffer(self):
        """Lion should use exactly one state tensor per parameter (half of AdamW)."""
        torch.manual_seed(0)
        model = nn.Linear(32, 32)
        config = OptimizerConfig(name="lion", lr=1e-4)
        optimizer = build_optimizer(model, config)

        x = torch.randn(4, 32)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

        for p in model.parameters():
            if p.grad is not None and p in optimizer.state:
                state = optimizer.state[p]
                # Only exp_avg — no exp_avg_sq like AdamW
                assert "exp_avg" in state
                assert len([k for k in state if isinstance(state[k], torch.Tensor)]) == 1

    def test_sign_based_updates(self):
        """All updates should have magnitude equal to lr (since sign is +-1)."""
        torch.manual_seed(42)
        model = nn.Linear(16, 16, bias=False)
        lr = 0.01
        optimizer = Lion([{"params": model.parameters(), "weight_decay": 0.0}], lr=lr)

        weight_before = model.weight.data.clone()

        x = torch.randn(4, 16)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

        diff = (model.weight.data - weight_before).abs()
        # Every element should have changed by exactly lr
        assert torch.allclose(diff, torch.full_like(diff, lr), atol=1e-7)

    def test_weight_decay(self):
        """Lion with weight decay should shrink parameters."""
        torch.manual_seed(42)
        model = nn.Linear(32, 32)
        weight_before = model.weight.data.norm().item()

        config = OptimizerConfig(name="lion", lr=1e-4, weight_decay=0.5)
        optimizer = build_optimizer(model, config)

        x = torch.randn(4, 32)
        for _ in range(10):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()

        weight_after = model.weight.data.norm().item()
        assert weight_after < weight_before

    def test_produces_finite_updates(self):
        torch.manual_seed(0)
        model = _make_model()
        config = OptimizerConfig(name="lion", lr=1e-4)
        optimizer = build_optimizer(model, config)

        x = torch.randn(8, 32)
        for _ in range(10):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()

        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_state_dict_roundtrip(self):
        """state_dict/load_state_dict should preserve momentum buffers."""
        torch.manual_seed(42)
        model = _make_model()
        config = OptimizerConfig(name="lion", lr=1e-4)
        optimizer = build_optimizer(model, config)

        x = torch.randn(8, 32)
        for _ in range(5):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x), torch.randint(0, 10, (8,))).backward()
            optimizer.step()

        sd = optimizer.state_dict()

        model2 = _make_model()
        optimizer2 = build_optimizer(model2, config)
        optimizer2.load_state_dict(sd)

        for p1, p2 in zip(
            [p for g in optimizer.param_groups for p in g["params"]],
            [p for g in optimizer2.param_groups for p in g["params"]],
            strict=True,
        ):
            if p1 in optimizer.state and "exp_avg" in optimizer.state[p1]:
                buf1 = optimizer.state[p1]["exp_avg"]
                buf2 = optimizer2.state[p2]["exp_avg"]
                torch.testing.assert_close(buf1, buf2)

    def test_betas_from_config(self):
        """Lion should use betas from OptimizerConfig."""
        model = nn.Linear(16, 16)
        config = OptimizerConfig(name="lion", lr=1e-4, betas=(0.95, 0.98))
        optimizer = build_optimizer(model, config)
        assert optimizer.defaults["betas"] == (0.95, 0.98)

    def test_weight_decay_groups(self):
        """2D params should get decay, 1D should not."""
        model = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64), nn.Linear(64, 10))
        config = OptimizerConfig(name="lion", lr=1e-4, weight_decay=0.1)
        optimizer = build_optimizer(model, config)

        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        assert decay_group["weight_decay"] == 0.1
        assert no_decay_group["weight_decay"] == 0.0

    def test_memory_half_of_adamw(self):
        """Lion optimizer state should use ~half the memory of AdamW."""
        torch.manual_seed(0)
        model = nn.Linear(256, 256, bias=False)
        param_size = model.weight.numel() * model.weight.element_size()

        # Lion: one buffer
        lion_opt = Lion([{"params": model.parameters(), "weight_decay": 0.0}], lr=1e-4)
        x = torch.randn(4, 256)
        lion_opt.zero_grad()
        model(x).sum().backward()
        lion_opt.step()

        lion_state_bytes = sum(
            v.numel() * v.element_size()
            for s in lion_opt.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )

        # AdamW: two buffers (exp_avg + exp_avg_sq)
        adamw_opt = torch.optim.AdamW([{"params": model.parameters(), "weight_decay": 0.0}])
        adamw_opt.zero_grad()
        model(x).sum().backward()
        adamw_opt.step()

        adamw_state_bytes = sum(
            v.numel() * v.element_size()
            for s in adamw_opt.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )

        # Lion should use ~half the state memory of AdamW
        assert lion_state_bytes == param_size  # one buffer
        assert adamw_state_bytes >= 2 * param_size  # two buffers
        assert lion_state_bytes < adamw_state_bytes


# ---------------------------------------------------------------------------
# Schedule-Free AdamW
# ---------------------------------------------------------------------------


class TestScheduleFreeAdamW:
    def test_build_via_registry(self):
        model = nn.Linear(32, 10)
        config = OptimizerConfig(name="schedule_free_adamw", lr=0.01)
        optimizer = build_optimizer(model, config)
        assert isinstance(optimizer, ScheduleFreeAdamW)

    def test_decreases_loss(self):
        """Schedule-Free AdamW should decrease loss."""
        torch.manual_seed(42)
        model = _make_model()
        x = torch.randn(16, 32)
        y = torch.randint(0, 10, (16,))

        config = OptimizerConfig(name="schedule_free_adamw", lr=1e-3, weight_decay=0.0)
        optimizer = build_optimizer(model, config)

        initial_loss = nn.functional.cross_entropy(model(x), y).item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.cross_entropy(model(x), y).item()
        assert final_loss < initial_loss

    def test_warmup(self):
        """With warmup, early steps should use smaller effective LR."""
        torch.manual_seed(42)
        model = nn.Linear(16, 16, bias=False)
        base_lr = 0.01

        # No warmup — full LR from step 1
        opt_no_warmup = ScheduleFreeAdamW(
            [{"params": list(model.parameters()), "weight_decay": 0.0}],
            lr=base_lr,
            warmup_steps=0,
        )
        weight_before = model.weight.data.clone()
        x = torch.randn(4, 16)
        opt_no_warmup.zero_grad()
        model(x).sum().backward()
        opt_no_warmup.step()
        delta_no_warmup = (model.weight.data - weight_before).norm().item()

        # Reset
        model.weight.data.copy_(weight_before)

        # With warmup — step 1 uses lr * (1/100) = 0.01%
        opt_warmup = ScheduleFreeAdamW(
            [{"params": list(model.parameters()), "weight_decay": 0.0}],
            lr=base_lr,
            warmup_steps=100,
        )
        opt_warmup.zero_grad()
        model(x).sum().backward()
        opt_warmup.step()
        delta_warmup = (model.weight.data - weight_before).norm().item()

        # Warmup step should produce much smaller updates
        assert delta_warmup < delta_no_warmup * 0.1

    def test_warmup_config_field(self):
        """schedule_free_warmup_steps should be passed to the optimizer."""
        model = nn.Linear(16, 16)
        config = OptimizerConfig(
            name="schedule_free_adamw", lr=0.01, schedule_free_warmup_steps=500
        )
        optimizer = build_optimizer(model, config)
        assert optimizer.warmup_steps == 500

    def test_eval_train_params(self):
        """eval_params/train_params should switch parameter values."""
        torch.manual_seed(42)
        model = nn.Linear(16, 16, bias=False)
        optimizer = ScheduleFreeAdamW(
            [{"params": list(model.parameters()), "weight_decay": 0.0}],
            lr=0.01,
        )

        # Run a few steps to diverge z and x
        x = torch.randn(4, 16)
        for _ in range(10):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()

        # After training, params are at y (interpolated)
        y_params = model.weight.data.clone()

        # Switch to eval (params = x = running average)
        optimizer.eval_params()
        eval_params = model.weight.data.clone()
        assert not torch.allclose(y_params, eval_params)

        # Switch back to train (params = y)
        optimizer.train_params()
        restored_params = model.weight.data.clone()
        assert torch.allclose(y_params, restored_params, atol=1e-6)

    def test_produces_finite_updates(self):
        torch.manual_seed(0)
        model = _make_model()
        config = OptimizerConfig(name="schedule_free_adamw", lr=1e-3)
        optimizer = build_optimizer(model, config)

        x = torch.randn(8, 32)
        for _ in range(10):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()

        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_state_dict_roundtrip(self):
        """state_dict should preserve z, v, x, step counter, and warmup_steps."""
        torch.manual_seed(42)
        model = _make_model()
        config = OptimizerConfig(
            name="schedule_free_adamw", lr=0.01, schedule_free_warmup_steps=100
        )
        optimizer = build_optimizer(model, config)

        x = torch.randn(8, 32)
        for _ in range(5):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x), torch.randint(0, 10, (8,))).backward()
            optimizer.step()

        sd = optimizer.state_dict()
        assert "_k" in sd
        assert sd["_k"] == 5
        assert "_warmup_steps" in sd
        assert sd["_warmup_steps"] == 100

        model2 = _make_model()
        config2 = OptimizerConfig(
            name="schedule_free_adamw", lr=0.01, schedule_free_warmup_steps=100
        )
        optimizer2 = build_optimizer(model2, config2)
        optimizer2.load_state_dict(sd)

        assert optimizer2._k == 5
        assert optimizer2.warmup_steps == 100

        # Verify per-parameter state was restored
        for p1, p2 in zip(
            [p for g in optimizer.param_groups for p in g["params"]],
            [p for g in optimizer2.param_groups for p in g["params"]],
            strict=True,
        ):
            if p1 in optimizer.state and "z" in optimizer.state[p1]:
                for key in ("z", "v"):
                    torch.testing.assert_close(optimizer.state[p1][key], optimizer2.state[p2][key])

    def test_weight_decay(self):
        torch.manual_seed(42)
        model = nn.Linear(32, 32)
        weight_before = model.weight.data.norm().item()

        config = OptimizerConfig(name="schedule_free_adamw", lr=1e-3, weight_decay=0.5)
        optimizer = build_optimizer(model, config)

        x = torch.randn(4, 32)
        for _ in range(20):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()

        weight_after = model.weight.data.norm().item()
        assert weight_after < weight_before

    def test_weight_decay_groups(self):
        """2D params should get decay, 1D should not."""
        model = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64), nn.Linear(64, 10))
        config = OptimizerConfig(name="schedule_free_adamw", lr=1e-3, weight_decay=0.1)
        optimizer = build_optimizer(model, config)

        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        assert decay_group["weight_decay"] == 0.1
        assert no_decay_group["weight_decay"] == 0.0


# ---------------------------------------------------------------------------
# No-op scheduler (for schedule-free optimizer)
# ---------------------------------------------------------------------------


class TestNoneScheduler:
    def test_constant_lr(self):
        """'none' scheduler should keep LR constant."""
        model = nn.Linear(16, 16)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = SchedulerConfig(name=SchedulerType.none)
        scheduler = build_scheduler(optimizer, config, max_steps=100)

        for _ in range(20):
            scheduler.step()

        assert optimizer.param_groups[0]["lr"] == 0.1

    def test_get_last_lr(self):
        """get_last_lr should return base LR."""
        model = nn.Linear(16, 16)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        config = SchedulerConfig(name=SchedulerType.none)
        scheduler = build_scheduler(optimizer, config, max_steps=100)

        scheduler.step()
        assert scheduler.get_last_lr()[0] == 0.05

    def test_schedule_free_with_none_scheduler(self):
        """Full integration: schedule-free optimizer + none scheduler."""
        torch.manual_seed(42)
        model = _make_model()
        config = OptimizerConfig(name="schedule_free_adamw", lr=1e-3)
        optimizer = build_optimizer(model, config)

        sched_config = SchedulerConfig(name=SchedulerType.none)
        scheduler = build_scheduler(optimizer, sched_config, max_steps=100)

        x = torch.randn(16, 32)
        y = torch.randint(0, 10, (16,))

        initial_loss = nn.functional.cross_entropy(model(x), y).item()

        for _ in range(30):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        final_loss = nn.functional.cross_entropy(model(x), y).item()
        assert final_loss < initial_loss


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestOptimizerConfig:
    def test_schedule_free_warmup_default(self):
        config = OptimizerConfig()
        assert config.schedule_free_warmup_steps == 0

    def test_schedule_free_warmup_set(self):
        config = OptimizerConfig(schedule_free_warmup_steps=1000)
        assert config.schedule_free_warmup_steps == 1000

    def test_scheduler_type_none(self):
        assert SchedulerType.none == "none"
        assert SchedulerType("none") == SchedulerType.none
