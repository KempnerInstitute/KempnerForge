"""Unit tests for LR schedule extensions (Phase 10)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from kempnerforge.config.schema import SchedulerConfig, SchedulerType
from kempnerforge.training.scheduler import build_scheduler


def _make_optimizer(lr: float = 1.0) -> torch.optim.Optimizer:
    model = nn.Linear(16, 16)
    return torch.optim.SGD(model.parameters(), lr=lr)


def _step_to(scheduler, n: int) -> None:
    """Advance scheduler by n steps."""
    for _ in range(n):
        scheduler.step()


# ---------------------------------------------------------------------------
# Constant with warmup
# ---------------------------------------------------------------------------


class TestConstantScheduler:
    def test_warmup_start(self):
        """LR should start at 0 during warmup."""
        opt = _make_optimizer()
        build_scheduler(opt, SchedulerConfig(name="constant", warmup_steps=100), 1000)
        # LambdaLR applies lr_fn(0) at init → 0/100 = 0
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0)

    def test_warmup_midpoint(self):
        opt = _make_optimizer()
        sched = build_scheduler(opt, SchedulerConfig(name="constant", warmup_steps=100), 1000)
        _step_to(sched, 50)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5, abs=0.02)

    def test_warmup_end(self):
        opt = _make_optimizer()
        sched = build_scheduler(opt, SchedulerConfig(name="constant", warmup_steps=100), 1000)
        _step_to(sched, 100)
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=0.01)

    def test_flat_after_warmup(self):
        """LR stays at peak after warmup ends."""
        opt = _make_optimizer()
        sched = build_scheduler(opt, SchedulerConfig(name="constant", warmup_steps=100), 1000)
        _step_to(sched, 500)
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=0.01)

    def test_flat_at_end(self):
        opt = _make_optimizer()
        sched = build_scheduler(opt, SchedulerConfig(name="constant", warmup_steps=100), 1000)
        _step_to(sched, 1000)
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=0.01)

    def test_no_warmup(self):
        """With warmup_steps=0, LR is constant from step 0."""
        opt = _make_optimizer()
        build_scheduler(opt, SchedulerConfig(name="constant", warmup_steps=0), 1000)
        # LambdaLR at init: lr_fn(0) with no warmup → 1.0
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0)

    def test_scheduler_type_enum(self):
        opt = _make_optimizer()
        sched = build_scheduler(
            opt, SchedulerConfig(name=SchedulerType.constant, warmup_steps=10), 100
        )
        _step_to(sched, 50)
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# WSD decay type variants
# ---------------------------------------------------------------------------


class TestWSDDecayTypes:
    def _make_wsd(self, decay_type: str) -> tuple[torch.optim.Optimizer, object]:
        opt = _make_optimizer()
        config = SchedulerConfig(
            name="wsd",
            warmup_steps=10,
            stable_steps=40,
            decay_steps=50,
            min_lr_ratio=0.0,
            wsd_decay_type=decay_type,
        )
        sched = build_scheduler(opt, config, max_steps=100)
        return opt, sched

    def test_cosine_default(self):
        """Default WSD uses cosine decay."""
        config = SchedulerConfig(name="wsd")
        assert config.wsd_decay_type == "cosine"

    def test_cosine_stable_phase(self):
        opt, sched = self._make_wsd("cosine")
        _step_to(sched, 30)  # In stable phase (step 10-50)
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=0.01)

    def test_cosine_decay_end(self):
        opt, sched = self._make_wsd("cosine")
        _step_to(sched, 100)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=0.02)

    def test_linear_decay_midpoint(self):
        """Linear decay at midpoint should be ~0.5."""
        opt, sched = self._make_wsd("linear")
        _step_to(sched, 75)  # Midpoint of decay phase (step 50-100)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5, abs=0.05)

    def test_linear_decay_end(self):
        opt, sched = self._make_wsd("linear")
        _step_to(sched, 100)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=0.02)

    def test_sqrt_stays_higher_than_linear(self):
        """Sqrt decay should stay higher than linear at midpoint."""
        opt_sqrt, sched_sqrt = self._make_wsd("sqrt")
        opt_linear, sched_linear = self._make_wsd("linear")
        _step_to(sched_sqrt, 75)
        _step_to(sched_linear, 75)
        # Sqrt stays higher for longer
        assert opt_sqrt.param_groups[0]["lr"] > opt_linear.param_groups[0]["lr"]

    def test_sqrt_decay_end(self):
        opt, sched = self._make_wsd("sqrt")
        _step_to(sched, 100)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=0.02)

    def test_all_variants_same_at_start(self):
        """All decay types should produce the same LR during warmup and stable phases."""
        for dtype in ("cosine", "linear", "sqrt"):
            opt, sched = self._make_wsd(dtype)
            _step_to(sched, 10)  # End of warmup
            assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=0.01), dtype
            _step_to(sched, 20)  # In stable phase
            assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=0.01), dtype

    def test_all_variants_same_at_end(self):
        """All decay types should reach min_lr at the end."""
        for dtype in ("cosine", "linear", "sqrt"):
            opt, sched = self._make_wsd(dtype)
            _step_to(sched, 100)
            assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=0.02), dtype

    def test_invalid_decay_type_rejected(self):
        with pytest.raises(ValueError, match="wsd_decay_type"):
            SchedulerConfig(name="wsd", wsd_decay_type="invalid")

    def test_different_curves(self):
        """The three decay types should produce different LR values during decay."""
        lrs = {}
        for dtype in ("cosine", "linear", "sqrt"):
            opt, sched = self._make_wsd(dtype)
            _step_to(sched, 65)  # 30% through decay
            lrs[dtype] = opt.param_groups[0]["lr"]
        # All should be different
        assert lrs["cosine"] != pytest.approx(lrs["linear"], abs=0.01)
        assert lrs["linear"] != pytest.approx(lrs["sqrt"], abs=0.01)


# ---------------------------------------------------------------------------
# REX scheduler
# ---------------------------------------------------------------------------


class TestREXScheduler:
    def test_warmup_start(self):
        opt = _make_optimizer()
        build_scheduler(opt, SchedulerConfig(name="rex", warmup_steps=100, rex_alpha=1.0), 1000)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0)

    def test_warmup_end(self):
        opt = _make_optimizer()
        sched = build_scheduler(
            opt, SchedulerConfig(name="rex", warmup_steps=100, rex_alpha=1.0), 1000
        )
        _step_to(sched, 100)
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=0.01)

    def test_alpha_1_is_linear(self):
        """alpha=1.0 should produce linear decay."""
        opt = _make_optimizer()
        config = SchedulerConfig(name="rex", warmup_steps=0, rex_alpha=1.0, min_lr_ratio=0.0)
        sched = build_scheduler(opt, config, max_steps=100)
        _step_to(sched, 50)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5, abs=0.02)

    def test_alpha_2_convex_decay(self):
        """alpha=2.0 should stay higher than linear at midpoint."""
        opt_rex = _make_optimizer()
        sched_rex = build_scheduler(
            opt_rex,
            SchedulerConfig(name="rex", warmup_steps=0, rex_alpha=2.0, min_lr_ratio=0.0),
            100,
        )
        opt_lin = _make_optimizer()
        sched_lin = build_scheduler(
            opt_lin,
            SchedulerConfig(name="rex", warmup_steps=0, rex_alpha=1.0, min_lr_ratio=0.0),
            100,
        )
        _step_to(sched_rex, 50)
        _step_to(sched_lin, 50)
        # alpha=2: (1-0.5)^2 = 0.25 vs alpha=1: (1-0.5) = 0.5
        assert opt_rex.param_groups[0]["lr"] < opt_lin.param_groups[0]["lr"]

    def test_alpha_half_stays_high(self):
        """alpha=0.5 (sqrt) should stay higher than linear."""
        opt_rex = _make_optimizer()
        sched_rex = build_scheduler(
            opt_rex,
            SchedulerConfig(name="rex", warmup_steps=0, rex_alpha=0.5, min_lr_ratio=0.0),
            100,
        )
        opt_lin = _make_optimizer()
        sched_lin = build_scheduler(
            opt_lin,
            SchedulerConfig(name="rex", warmup_steps=0, rex_alpha=1.0, min_lr_ratio=0.0),
            100,
        )
        _step_to(sched_rex, 50)
        _step_to(sched_lin, 50)
        # alpha=0.5: (1-0.5)^0.5 ≈ 0.707 vs alpha=1: 0.5
        assert opt_rex.param_groups[0]["lr"] > opt_lin.param_groups[0]["lr"]

    def test_decay_to_zero(self):
        opt = _make_optimizer()
        config = SchedulerConfig(name="rex", warmup_steps=0, rex_alpha=1.0, min_lr_ratio=0.0)
        sched = build_scheduler(opt, config, max_steps=100)
        _step_to(sched, 100)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=0.02)

    def test_min_lr_ratio_floor(self):
        """REX should respect min_lr_ratio as a floor."""
        opt = _make_optimizer()
        config = SchedulerConfig(name="rex", warmup_steps=0, rex_alpha=1.0, min_lr_ratio=0.1)
        sched = build_scheduler(opt, config, max_steps=100)
        _step_to(sched, 100)
        assert opt.param_groups[0]["lr"] >= 0.1 - 0.01

    def test_invalid_alpha_rejected(self):
        with pytest.raises(ValueError, match="rex_alpha"):
            SchedulerConfig(name="rex", rex_alpha=0.0)
        with pytest.raises(ValueError, match="rex_alpha"):
            SchedulerConfig(name="rex", rex_alpha=-1.0)

    def test_scheduler_type_enum(self):
        opt = _make_optimizer()
        sched = build_scheduler(
            opt, SchedulerConfig(name=SchedulerType.rex, warmup_steps=10, rex_alpha=0.5), 100
        )
        _step_to(sched, 50)
        assert opt.param_groups[0]["lr"] > 0

    def test_exact_values(self):
        """Verify exact REX values at specific points."""
        opt = _make_optimizer(lr=1.0)
        config = SchedulerConfig(name="rex", warmup_steps=0, rex_alpha=2.0, min_lr_ratio=0.0)
        sched = build_scheduler(opt, config, max_steps=100)
        # At step 0 (init): (1 - 0/100)^2 = 1.0
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0)
        _step_to(sched, 50)
        # At step 50: (1 - 50/100)^2 = 0.25
        assert opt.param_groups[0]["lr"] == pytest.approx(0.25, abs=0.01)


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------


class TestSchedulerTomlLoading:
    def test_constant_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            """
[scheduler]
name = "constant"
warmup_steps = 200
"""
        )
        from kempnerforge.config.loader import load_config

        config = load_config(str(toml_path), cli_args=[])
        assert config.scheduler.name == SchedulerType.constant
        assert config.scheduler.warmup_steps == 200

    def test_rex_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            """
[scheduler]
name = "rex"
warmup_steps = 100
rex_alpha = 0.5
min_lr_ratio = 0.05
"""
        )
        from kempnerforge.config.loader import load_config

        config = load_config(str(toml_path), cli_args=[])
        assert config.scheduler.name == SchedulerType.rex
        assert config.scheduler.rex_alpha == 0.5
        assert config.scheduler.min_lr_ratio == 0.05

    def test_wsd_decay_type_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            """
[scheduler]
name = "wsd"
warmup_steps = 10
stable_steps = 40
wsd_decay_type = "sqrt"
"""
        )
        from kempnerforge.config.loader import load_config

        config = load_config(str(toml_path), cli_args=[])
        assert config.scheduler.wsd_decay_type == "sqrt"

    def test_rex_alpha_cli_override(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('[scheduler]\nname = "rex"\n')
        from kempnerforge.config.loader import load_config

        config = load_config(str(toml_path), cli_args=["--scheduler.rex_alpha=2.0"])
        assert config.scheduler.rex_alpha == 2.0


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestSchedulerConfigValidation:
    def test_scheduler_type_constant(self):
        assert SchedulerType.constant == "constant"
        assert SchedulerType("constant") == SchedulerType.constant

    def test_scheduler_type_rex(self):
        assert SchedulerType.rex == "rex"
        assert SchedulerType("rex") == SchedulerType.rex

    def test_wsd_decay_type_default(self):
        config = SchedulerConfig()
        assert config.wsd_decay_type == "cosine"

    def test_rex_alpha_default(self):
        config = SchedulerConfig()
        assert config.rex_alpha == 1.0
