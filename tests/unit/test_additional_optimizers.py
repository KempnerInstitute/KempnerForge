"""Unit tests for Lion and Schedule-Free AdamW optimizers."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from kempnerforge.config.schema import OptimizerConfig, SchedulerConfig, SchedulerType
from kempnerforge.training.optimizer import (
    Lion,
    ScheduleFreeAdamW,
    build_optimizer,
    evaluation_point,
)
from kempnerforge.training.scheduler import build_scheduler


def _make_model():
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


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

        assert lion_state_bytes == param_size
        assert adamw_state_bytes >= 2 * param_size
        assert lion_state_bytes < adamw_state_bytes


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

        model.weight.data.copy_(weight_before)

        opt_warmup = ScheduleFreeAdamW(
            [{"params": list(model.parameters()), "weight_decay": 0.0}],
            lr=base_lr,
            warmup_steps=100,
        )
        opt_warmup.zero_grad()
        model(x).sum().backward()
        opt_warmup.step()
        delta_warmup = (model.weight.data - weight_before).norm().item()

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

        x = torch.randn(4, 16)
        for _ in range(10):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()

        y_params = model.weight.data.clone()

        optimizer.eval_params()
        eval_params = model.weight.data.clone()
        assert not torch.allclose(y_params, eval_params)

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
        """state_dict holds z, v, x and each parameter's step, and only state and groups."""
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
        assert set(sd) == {"state", "param_groups"}
        assert {state["step"].item() for state in sd["state"].values()} == {5.0}

        model2 = _make_model()
        optimizer2 = build_optimizer(model2, config)
        optimizer2.load_state_dict(sd)
        assert optimizer2.warmup_steps == 100

        for p1, p2 in zip(
            [p for g in optimizer.param_groups for p in g["params"]],
            [p for g in optimizer2.param_groups for p in g["params"]],
            strict=True,
        ):
            for key in ("step", "z", "v", "x"):
                torch.testing.assert_close(optimizer.state[p1][key], optimizer2.state[p2][key])

    def test_a_state_saved_with_one_step_count_loads(self):
        """A state from before each parameter kept its step resumes every parameter from ``_k``."""
        torch.manual_seed(0)
        model = nn.Linear(8, 8)
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.01)
        for _ in range(3):
            model(torch.randn(2, 8)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        sd = optimizer.state_dict()
        for state in sd["state"].values():
            del state["step"]
        sd["_k"], sd["_warmup_steps"] = 3, 7

        optimizer2 = ScheduleFreeAdamW(nn.Linear(8, 8).parameters(), lr=0.01)
        optimizer2.load_state_dict(sd)
        assert {state["step"].item() for state in optimizer2.state.values()} == {3.0}
        assert optimizer2.warmup_steps == 7

    def test_the_legacy_count_reaches_only_entries_with_state_and_no_step(self):
        """``_k`` leaves an empty entry empty and an entry's own ``step`` as it is.

        An empty entry, as the legacy ``train_params`` left for a parameter without state,
        stays empty, so that parameter's first gradient starts it at step 1.
        """
        torch.manual_seed(0)
        params = [nn.Parameter(torch.randn(4, 4)) for _ in range(3)]
        optimizer = ScheduleFreeAdamW(params, lr=0.01)
        for _ in range(3):
            for p in params[:2]:
                p.grad = torch.randn(4, 4)
            optimizer.step()
        sd = copy.deepcopy(optimizer.state_dict())
        del sd["state"][0]["step"]
        sd["state"][1]["step"] = torch.tensor(5.0)
        sd["state"][2] = {}
        sd["_k"] = 3

        params2 = [nn.Parameter(torch.randn(4, 4)) for _ in range(3)]
        optimizer2 = ScheduleFreeAdamW(params2, lr=0.01)
        optimizer2.load_state_dict(sd)
        steps = [optimizer2.state[p].get("step") for p in params2]
        assert [None if s is None else s.item() for s in steps] == [3.0, 5.0, None]
        assert optimizer2.state[params2[2]] == {}

        for p in params2:
            p.grad = torch.ones(4, 4)
        optimizer2.step()
        assert [optimizer2.state[p]["step"].item() for p in params2] == [4.0, 6.0, 1.0]

    def test_the_step_count_is_float32_under_any_default_dtype(self):
        """Under a bf16 default dtype a bf16 count would stop at 256; the count stays exact."""
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            torch.manual_seed(0)
            model = nn.Linear(4, 4)
            optimizer = ScheduleFreeAdamW(model.parameters(), lr=1e-3, warmup_steps=1000)
            for _ in range(300):
                model(torch.randn(2, 4)).sum().backward()
                optimizer.step()
                optimizer.zero_grad()
            counts = [state["step"].clone() for state in optimizer.state.values()]
            legacy = copy.deepcopy(optimizer.state_dict())
            for state in legacy["state"].values():
                del state["step"]
            legacy["_k"] = 300
            optimizer2 = ScheduleFreeAdamW(nn.Linear(4, 4).parameters(), lr=1e-3)
            optimizer2.load_state_dict(legacy)
        finally:
            torch.set_default_dtype(previous)
        counts += [state["step"] for state in optimizer2.state.values()]
        assert all(c.dtype == torch.float32 and c.item() == 300 for c in counts), counts

    def test_reading_a_fresh_optimizer_through_dcp_leaves_the_model_alone(self):
        """``get_optimizer_state_dict`` gives a fresh optimizer a zero-lr step; it must not fail."""
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

        torch.manual_seed(0)
        model = _make_model()
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = build_optimizer(model, OptimizerConfig(name="schedule_free_adamw", lr=0.01))
        state = get_optimizer_state_dict(model, optimizer)
        assert state["state"], "the template should hold every parameter's state"
        for b, p in zip(before, model.parameters(), strict=True):
            assert torch.equal(b, p)

    def test_a_zero_lr_step_is_a_no_op(self):
        """A step at lr 0, as under a warmup that starts at 0, leaves the weights unchanged."""
        torch.manual_seed(0)
        model = _make_model()
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.0)
        model(torch.randn(4, 32)).sum().backward()
        optimizer.step()
        for b, p in zip(before, model.parameters(), strict=True):
            assert torch.equal(b, p)

    @pytest.mark.parametrize("warmup_steps", [0, 10])
    @pytest.mark.parametrize("betas", [(0.9, 0.95), (0.9, 0.999)])
    def test_a_late_parameter_takes_a_fresh_first_update(self, betas, warmup_steps):
        """A parameter whose first gradient comes after 100 steps moves as it would on step 1."""

        def first_update(frozen_steps):
            torch.manual_seed(0)
            model = nn.ModuleDict(
                {"a": nn.Linear(16, 16, bias=False), "b": nn.Linear(16, 16, bias=False)}
            )
            optimizer = ScheduleFreeAdamW(
                model.parameters(), lr=1e-3, betas=betas, warmup_steps=warmup_steps
            )
            g = torch.Generator().manual_seed(1)
            ga, gb = torch.randn(16, 16, generator=g), torch.randn(16, 16, generator=g)
            for _ in range(frozen_steps):
                model["a"].weight.grad = ga.clone()
                optimizer.step()
            before = model["b"].weight.detach().clone()
            model["a"].weight.grad, model["b"].weight.grad = ga.clone(), gb.clone()
            optimizer.step()
            return model["b"].weight.detach() - before

        assert torch.equal(first_update(100), first_update(0))

    def test_evaluation_point_swaps_in_x_and_restores_y_bit_for_bit(self):
        """Inside ``evaluation_point`` the weights are x; after it they are y again, exactly."""
        torch.manual_seed(0)
        model = _make_model()
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.01)
        for _ in range(5):
            model(torch.randn(4, 32)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        y = [p.detach().clone() for p in model.parameters()]
        with evaluation_point(optimizer):
            inside = [p.detach().clone() for p in model.parameters()]
        params = list(model.parameters())
        assert all(
            torch.equal(i, optimizer.state[p]["x"]) for i, p in zip(inside, params, strict=True)
        )
        assert not all(torch.equal(i, b) for i, b in zip(inside, y, strict=True))
        assert all(torch.equal(b, p) for b, p in zip(y, params, strict=True))

    def test_evaluation_point_gives_no_state_to_a_fresh_optimizer(self):
        """The swap creates no optimizer state, so a step-0 save still writes no optimizer."""
        model = _make_model()
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.01)
        with evaluation_point(optimizer):
            pass
        assert not optimizer.state

    def test_train_params_gives_no_state_to_a_fresh_optimizer(self):
        """``train_params`` before the first step adds no empty state entries.

        An empty entry makes ``optimizer.state`` truthy, so a step-0 save would write the
        optimizer and DCP's load template would skip the step that builds every entry.
        """
        model = _make_model()
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.01)
        optimizer.eval_params()
        optimizer.train_params()
        assert not optimizer.state

    def test_evaluation_point_restores_y_when_the_block_raises(self):
        torch.manual_seed(0)
        model = nn.Linear(8, 8)
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.01)
        for _ in range(3):
            model(torch.randn(2, 8)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        y = [p.detach().clone() for p in model.parameters()]
        with pytest.raises(RuntimeError), evaluation_point(optimizer):
            raise RuntimeError("eval failed")
        assert all(torch.equal(a, p) for a, p in zip(y, model.parameters(), strict=True))

    def test_evaluation_point_restores_y_when_the_swap_in_fails_partway(self):
        """A swap that raises after moving some parameters to x leaves every one at y."""
        torch.manual_seed(0)
        model = _make_model()
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.01)
        for _ in range(3):
            model(torch.randn(4, 32)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        params = list(model.parameters())
        y = [p.detach().clone() for p in params]

        def swap_half_then_fail():
            for p in params[: len(params) // 2]:
                p.data.copy_(optimizer.state[p]["x"])
            raise RuntimeError("swap failed")

        optimizer.eval_params = swap_half_then_fail
        with pytest.raises(RuntimeError, match="swap failed"), evaluation_point(optimizer):
            pass
        assert all(torch.equal(a, p) for a, p in zip(y, params, strict=True))

    def test_evaluation_point_leaves_other_optimizers_alone(self):
        model = _make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before = [p.detach().clone() for p in model.parameters()]
        with evaluation_point(optimizer):
            pass
        assert all(torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))

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
