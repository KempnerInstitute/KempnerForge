"""Unit tests for parameter freezing helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from kempnerforge.config.optimizer import OptimizerConfig
from kempnerforge.config.registry import registry
from kempnerforge.config.vlm import DEFAULT_MODULE_PATTERNS, FreezeSpec, FreezeStage, VLMConfig
from kempnerforge.training.freeze import (
    apply_freeze_specs,
    canonical_freeze_meta,
    freeze_params,
    trainable_during_run,
    trainable_for_the_run,
)
from kempnerforge.training.optimizer import build_optimizer


class _MiniVLM(nn.Module):
    """Minimal stand-in for VLMWrapper, used to exercise FQN patterns."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.vision_encoder = nn.Sequential(nn.Linear(8, 8))
        self.adapter = nn.Linear(8, 4)


@pytest.fixture
def model() -> _MiniVLM:
    return _MiniVLM()


class TestFreezeParams:
    def test_freezes_matching(self, model):
        n = freeze_params(model, ["transformer.*"])
        assert n > 0
        for name, p in model.named_parameters():
            if name.startswith("transformer."):
                assert not p.requires_grad
            else:
                assert p.requires_grad

    def test_empty_patterns_is_noop(self, model):
        before = {n: p.requires_grad for n, p in model.named_parameters()}
        n = freeze_params(model, [])
        after = {n: p.requires_grad for n, p in model.named_parameters()}
        assert n == 0
        assert before == after

    def test_idempotent(self, model):
        first = freeze_params(model, ["adapter.*"])
        second = freeze_params(model, ["adapter.*"])
        assert first > 0
        assert second == 0

    def test_unfreeze(self, model):
        n_frozen = freeze_params(model, ["adapter.*"], frozen=True)
        n_unfrozen = freeze_params(model, ["adapter.*"], frozen=False)
        assert n_frozen == n_unfrozen
        for p in model.adapter.parameters():
            assert p.requires_grad

    def test_raw_pattern(self, model):
        """Raw fnmatch works; ``transformer.0.*`` hits only the first layer."""
        freeze_params(model, ["transformer.0.*"])
        for name, p in model.named_parameters():
            if name.startswith("transformer.0"):
                assert not p.requires_grad
            else:
                assert p.requires_grad

    def test_multiple_patterns_union(self, model):
        freeze_params(model, ["vision_encoder.*", "adapter.*"])
        for name, p in model.named_parameters():
            frozen = name.startswith("vision_encoder.") or name.startswith("adapter.")
            assert p.requires_grad == (not frozen)


class TestApplyFreezeSpecs:
    def test_alias_resolves_via_pattern_map(self, model):
        totals = apply_freeze_specs(
            model,
            [FreezeSpec("vision_encoder", True)],
            DEFAULT_MODULE_PATTERNS,
        )
        assert totals["vision_encoder"] > 0
        for p in model.vision_encoder.parameters():
            assert not p.requires_grad
        for p in model.transformer.parameters():
            assert p.requires_grad

    def test_unknown_module_falls_back_to_raw_pattern(self, model):
        totals = apply_freeze_specs(
            model,
            [FreezeSpec("transformer.1.*", True)],
            DEFAULT_MODULE_PATTERNS,
        )
        assert totals["transformer.1.*"] > 0
        for name, p in model.named_parameters():
            expected_trainable = not name.startswith("transformer.1")
            assert p.requires_grad == expected_trainable

    def test_multiple_specs_compose(self, model):
        apply_freeze_specs(
            model,
            [FreezeSpec("vision_encoder", True), FreezeSpec("adapter", True)],
            DEFAULT_MODULE_PATTERNS,
        )
        for p in model.vision_encoder.parameters():
            assert not p.requires_grad
        for p in model.adapter.parameters():
            assert not p.requires_grad
        for p in model.transformer.parameters():
            assert p.requires_grad


class TestCanonicalFreezeMeta:
    def test_reorder_invariant(self):
        a = canonical_freeze_meta(
            [FreezeSpec("vision_encoder", True), FreezeSpec("adapter", False)]
        )
        b = canonical_freeze_meta(
            [FreezeSpec("adapter", False), FreezeSpec("vision_encoder", True)]
        )
        assert a == b

    def test_dedup(self):
        out = canonical_freeze_meta(
            [
                FreezeSpec("vision_encoder", True),
                FreezeSpec("vision_encoder", True),
            ]
        )
        assert out == [{"module": "vision_encoder", "frozen": True}]

    def test_output_shape(self):
        out = canonical_freeze_meta([FreezeSpec("adapter", False)])
        assert out == [{"module": "adapter", "frozen": False}]

    def test_empty(self):
        assert canonical_freeze_meta([]) == []


class TestEffectiveFreeze:
    def test_empty_schedule_returns_base(self):
        from kempnerforge.training.freeze import effective_freeze

        base = [FreezeSpec("vision_encoder", True)]
        out = effective_freeze(step=10, base=base, schedule=[])
        assert out == base

    def test_single_stage_overrides_when_reached(self):
        from kempnerforge.config.vlm import FreezeStage
        from kempnerforge.training.freeze import effective_freeze

        base = [FreezeSpec("vision_encoder", True)]
        stage = FreezeStage(start_step=100, specs=(FreezeSpec("vision_encoder", False),))
        before = effective_freeze(step=99, base=base, schedule=[stage])
        assert before == base
        at = effective_freeze(step=100, base=base, schedule=[stage])
        assert at == [FreezeSpec("vision_encoder", False)]
        after = effective_freeze(step=200, base=base, schedule=[stage])
        assert after == [FreezeSpec("vision_encoder", False)]

    def test_multi_stage_last_write_wins_on_conflict(self):
        from kempnerforge.config.vlm import FreezeStage
        from kempnerforge.training.freeze import effective_freeze

        base = [FreezeSpec("vision_encoder", True), FreezeSpec("adapter", True)]
        stages = [
            FreezeStage(start_step=10, specs=(FreezeSpec("adapter", False),)),
            FreezeStage(start_step=20, specs=(FreezeSpec("vision_encoder", False),)),
            FreezeStage(start_step=30, specs=(FreezeSpec("adapter", True),)),
        ]
        out = effective_freeze(step=100, base=base, schedule=stages)
        out_map = {s.module: s.frozen for s in out}
        assert out_map["vision_encoder"] is False
        assert out_map["adapter"] is True

    def test_monotonic_order_independent_of_input(self):
        from kempnerforge.config.vlm import FreezeStage
        from kempnerforge.training.freeze import effective_freeze

        base = [FreezeSpec("vision_encoder", True)]
        stages_a = [
            FreezeStage(start_step=10, specs=(FreezeSpec("vision_encoder", False),)),
            FreezeStage(start_step=20, specs=(FreezeSpec("vision_encoder", True),)),
        ]
        stages_b = list(reversed(stages_a))
        out_a = effective_freeze(step=100, base=base, schedule=stages_a)
        out_b = effective_freeze(step=100, base=base, schedule=stages_b)
        assert out_a == out_b

    def test_module_key_validation_rejects_typo_in_base(self):
        from kempnerforge.training.freeze import effective_freeze

        base = [FreezeSpec("vison_encoder", True)]
        with pytest.raises(ValueError, match="vison_encoder"):
            effective_freeze(
                step=0,
                base=base,
                schedule=[],
                valid_modules={"vision_encoder", "adapter", "transformer"},
            )

    def test_module_key_validation_rejects_typo_in_stage(self):
        from kempnerforge.config.vlm import FreezeStage
        from kempnerforge.training.freeze import effective_freeze

        base = [FreezeSpec("vision_encoder", True)]
        stage = FreezeStage(
            start_step=10,
            specs=(FreezeSpec("adaptor", False),),
        )
        with pytest.raises(ValueError, match="adaptor"):
            effective_freeze(
                step=20,
                base=base,
                schedule=[stage],
                valid_modules={"vision_encoder", "adapter", "transformer"},
            )

    def test_module_key_validation_skipped_when_valid_modules_none(self):
        from kempnerforge.training.freeze import effective_freeze

        base = [FreezeSpec("anything", True)]
        out = effective_freeze(step=0, base=base, schedule=[], valid_modules=None)
        assert out == base


_ENCODER = {"vision_encoder.0.weight", "vision_encoder.0.bias"}
_ALL = {name for name, _ in _MiniVLM().named_parameters()}


def _unfreeze_encoder_at(step: int) -> VLMConfig:
    return VLMConfig(
        freeze=[FreezeSpec("vision_encoder")],
        freeze_schedule=[FreezeStage(step, (FreezeSpec("vision_encoder", False),))],
    )


class TestTrainableDuringRun:
    def test_the_build_freeze_alone(self, model):
        config = VLMConfig(freeze=[FreezeSpec("vision_encoder")])
        assert trainable_during_run(model, config) == _ALL - _ENCODER

    def test_a_stage_that_unfreezes_a_build_frozen_module_adds_it(self, model):
        assert trainable_during_run(model, _unfreeze_encoder_at(5)) == _ALL

    def test_a_module_that_trains_until_a_stage_freezes_it_counts(self, model):
        config = VLMConfig(freeze=[], freeze_schedule=[FreezeStage(5, (FreezeSpec("adapter"),))])
        assert trainable_during_run(model, config) == _ALL

    def test_a_module_frozen_for_the_whole_run_stays_out(self, model):
        config = VLMConfig(
            freeze=[FreezeSpec("vision_encoder")],
            freeze_schedule=[
                FreezeStage(5, (FreezeSpec("adapter"),)),
                FreezeStage(9, (FreezeSpec("adapter", False),)),
            ],
        )
        assert trainable_during_run(model, config) == _ALL - _ENCODER

    def test_the_trainable_part_of_a_partly_frozen_module_counts(self, model):
        patterns = {**DEFAULT_MODULE_PATTERNS, "encoder_bias": ["vision_encoder.0.bias"]}
        config = VLMConfig(
            freeze=[FreezeSpec("vision_encoder"), FreezeSpec("encoder_bias", False)],
            module_patterns=patterns,
        )
        assert trainable_during_run(model, config) == _ALL - {"vision_encoder.0.weight"}

    def test_a_parameter_that_only_a_resume_trains_counts(self, model):
        """The stage unfreezes the encoder's weight and then freezes the encoder, so the run
        keeps the weight frozen; effective_freeze orders the unfreeze last, so a resume past
        the stage trains it."""
        patterns = {**DEFAULT_MODULE_PATTERNS, "encoder_weight": ["vision_encoder.0.weight"]}
        config = VLMConfig(
            freeze=[FreezeSpec("vision_encoder")],
            freeze_schedule=[
                FreezeStage(5, (FreezeSpec("encoder_weight", False), FreezeSpec("vision_encoder")))
            ],
            module_patterns=patterns,
        )
        assert trainable_during_run(model, config) == _ALL - {"vision_encoder.0.bias"}

    def test_a_parameter_built_without_grad_and_never_unfrozen_stays_out(self, model):
        model.adapter.bias.requires_grad_(False)
        assert trainable_during_run(model, VLMConfig(freeze=[])) == _ALL - {"adapter.bias"}

    def test_the_model_is_left_as_it_is(self, model):
        trainable_during_run(model, _unfreeze_encoder_at(5))
        assert all(p.requires_grad for p in model.parameters())


def test_apply_freeze_specs_reaches_the_module_a_compiled_model_wraps(model):
    compiled = torch.compile(model, backend="eager")
    flipped = apply_freeze_specs(compiled, [FreezeSpec("vision_encoder")], DEFAULT_MODULE_PATTERNS)
    assert flipped == {"vision_encoder": 72}
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen == _ENCODER


class _Compiled(nn.Module):
    """Wraps a module under ``_orig_mod``, as ``torch.compile`` does."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._orig_mod = module


class TestTrainableForTheRun:
    def test_marks_the_parameters_a_stage_unfreezes_and_restores_them(self, model):
        config = _unfreeze_encoder_at(5)
        apply_freeze_specs(model, config.freeze, config.module_patterns)
        with trainable_for_the_run(model, config):
            assert all(p.requires_grad for p in model.parameters())
        frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
        assert frozen == _ENCODER

    def test_restores_them_when_the_body_raises(self, model):
        config = _unfreeze_encoder_at(5)
        apply_freeze_specs(model, config.freeze, config.module_patterns)
        with pytest.raises(RuntimeError, match="load failed"), trainable_for_the_run(model, config):
            raise RuntimeError("load failed")
        frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
        assert frozen == _ENCODER

    def test_leaves_a_module_frozen_for_the_whole_run_frozen(self, model):
        config = VLMConfig(freeze=[FreezeSpec("vision_encoder")])
        apply_freeze_specs(model, config.freeze, config.module_patterns)
        with trainable_for_the_run(model, config):
            frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
        assert frozen == _ENCODER

    def test_reads_a_compiled_model_through_the_module_it_wraps(self, model):
        config = _unfreeze_encoder_at(5)
        apply_freeze_specs(model, config.freeze, config.module_patterns)
        with trainable_for_the_run(_Compiled(model), config):
            assert all(p.requires_grad for p in model.parameters())


def _held(optimizer: torch.optim.Optimizer) -> set[int]:
    """Ids of the parameters an optimizer holds, with a Muon optimizer's AdamW fallback."""
    inner = getattr(optimizer, "_adam", None)
    optimizers = [optimizer] if inner is None else [optimizer, inner]
    return {id(p) for opt in optimizers for group in opt.param_groups for p in group["params"]}


@pytest.mark.parametrize("name", sorted(registry.list("optimizer")))
def test_a_module_the_build_froze_trains_once_a_stage_unfreezes_it(model, name):
    """The optimizer built under trainable_for_the_run holds the frozen encoder, leaves it
    untouched while it has no gradient, weight decay included, and updates it once the
    stage unfreezes it."""
    torch.manual_seed(0)
    config = _unfreeze_encoder_at(2)
    apply_freeze_specs(model, config.freeze, config.module_patterns)
    with trainable_for_the_run(model, config):
        optimizer = build_optimizer(
            model, OptimizerConfig(name=name, lr=1e-2, weight_decay=0.1, fused=False)
        )
    encoder = list(model.vision_encoder.parameters())
    assert {id(p) for p in encoder} <= _held(optimizer)
    before = [p.detach().clone() for p in encoder]

    def step() -> None:
        out = model.transformer(model.adapter(model.vision_encoder(torch.randn(4, 8))))
        out.square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    step()
    assert all(torch.equal(p, b) for p, b in zip(encoder, before, strict=True))
    apply_freeze_specs(model, config.freeze_schedule[0].specs, config.module_patterns)
    step()
    assert not any(torch.equal(p, b) for p, b in zip(encoder, before, strict=True))
