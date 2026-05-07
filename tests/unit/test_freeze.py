"""Unit tests for parameter freezing helpers."""

from __future__ import annotations

import pytest
import torch.nn as nn

from kempnerforge.config.vlm import DEFAULT_MODULE_PATTERNS, FreezeSpec
from kempnerforge.training.freeze import (
    apply_freeze_specs,
    canonical_freeze_meta,
    freeze_params,
)


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


# ---------------------------------------------------------------------------
# freeze_params
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# apply_freeze_specs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# canonical_freeze_meta
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# effective_freeze
# ---------------------------------------------------------------------------


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
        # Before transition.
        before = effective_freeze(step=99, base=base, schedule=[stage])
        assert before == base
        # At transition.
        at = effective_freeze(step=100, base=base, schedule=[stage])
        assert at == [FreezeSpec("vision_encoder", False)]
        # After transition.
        after = effective_freeze(step=200, base=base, schedule=[stage])
        assert after == [FreezeSpec("vision_encoder", False)]

    def test_multi_stage_last_write_wins_on_conflict(self):
        from kempnerforge.config.vlm import FreezeStage
        from kempnerforge.training.freeze import effective_freeze

        base = [FreezeSpec("vision_encoder", True), FreezeSpec("adapter", True)]
        stages = [
            FreezeStage(start_step=10, specs=(FreezeSpec("adapter", False),)),
            FreezeStage(start_step=20, specs=(FreezeSpec("vision_encoder", False),)),
            FreezeStage(start_step=30, specs=(FreezeSpec("adapter", True),)),  # re-freeze adapter
        ]
        out = effective_freeze(step=100, base=base, schedule=stages)
        out_map = {s.module: s.frozen for s in out}
        assert out_map["vision_encoder"] is False
        assert out_map["adapter"] is True  # last write wins

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

        base = [FreezeSpec("vison_encoder", True)]  # typo
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
            specs=(FreezeSpec("adaptor", False),),  # typo
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
