"""Unit tests for parameter freezing helpers."""

from __future__ import annotations

import pytest
import torch.nn as nn

from kempnerforge.config.vlm import DEFAULT_MODULE_PATTERNS, FreezeSpec, FreezeStage
from kempnerforge.training.freeze import (
    apply_freeze_specs,
    canonical_freeze_meta,
    effective_freeze,
    freeze_params,
    unmatched_patterns,
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


class TestFreezeNames:
    """A name is an alias or a parameter pattern on the build, save and resume paths (#514)."""

    def test_validation_accepts_parameter_patterns(self):
        from kempnerforge.config.vlm import FreezeStage
        from kempnerforge.training.freeze import effective_freeze

        base = [FreezeSpec("transformer", True), FreezeSpec("transformer.layers.1.*", False)]
        stage = FreezeStage(
            start_step=10,
            specs=(FreezeSpec("vision_encoder.proj.bias", False), FreezeSpec("adapter*", True)),
        )
        out = effective_freeze(
            step=20,
            base=base,
            schedule=[stage],
            valid_modules={"vision_encoder", "adapter", "transformer"},
        )
        assert [s.module for s in out] == [
            "transformer",
            "transformer.layers.1.*",
            "vision_encoder.proj.bias",
            "adapter*",
        ]

    def test_freeze_meta_takes_a_raw_pattern_the_build_applies(self):
        from kempnerforge.config.vlm import VLMConfig
        from kempnerforge.training.freeze import freeze_meta_at_step

        vlm = VLMConfig(
            max_text_len=8,
            freeze=[FreezeSpec("transformer"), FreezeSpec("transformer.layers.1.*", False)],
        )
        assert {"module": "transformer.layers.1.*", "frozen": False} in freeze_meta_at_step(1, vlm)

    def test_a_bare_word_that_is_no_alias_is_refused_at_config_time(self):
        from kempnerforge.config.vlm import VLMConfig

        with pytest.raises(ValueError, match=r"\['vison_encoder'\], which are neither module"):
            VLMConfig(freeze=[FreezeSpec("vison_encoder")])

    def test_a_typo_in_a_stage_is_refused_at_config_time(self):
        from kempnerforge.config.vlm import FreezeStage, VLMConfig

        with pytest.raises(ValueError, match="adaptor"):
            VLMConfig(freeze_schedule=[FreezeStage(5, (FreezeSpec("adaptor", True),))])

    def test_patterns_and_custom_aliases_pass_config(self):
        from kempnerforge.config.vlm import VLMConfig

        patterns = {**DEFAULT_MODULE_PATTERNS, "enc0": ["vision_encoder.layers.0.*"]}
        config = VLMConfig(
            freeze=[
                FreezeSpec("enc0"),
                FreezeSpec("adapter.*"),
                FreezeSpec("transformer.layers.[01].*", False),
            ],
            module_patterns=patterns,
        )
        assert [s.module for s in config.freeze] == [
            "enc0",
            "adapter.*",
            "transformer.layers.[01].*",
        ]

    @pytest.mark.parametrize("name", ["transformer.layers", "adapter*", "adapte?", "adapte[r]"])
    def test_each_pattern_marker_alone_makes_a_pattern(self, name):
        from kempnerforge.config.vlm import VLMConfig, is_parameter_pattern
        from kempnerforge.training.freeze import effective_freeze

        assert is_parameter_pattern(name)
        assert not is_parameter_pattern("adapter")
        assert [s.module for s in VLMConfig(freeze=[FreezeSpec(name)]).freeze] == [name]
        out = effective_freeze(
            step=0, base=[FreezeSpec(name)], schedule=[], valid_modules={"adapter"}
        )
        assert [s.module for s in out] == [name]

    def test_a_dotted_alias_resolves_as_its_alias(self, model):
        from kempnerforge.config.vlm import VLMConfig

        patterns = {**DEFAULT_MODULE_PATTERNS, "vision.tower": ["vision_encoder.*"]}
        config = VLMConfig(freeze=[FreezeSpec("vision.tower")], module_patterns=patterns)
        apply_freeze_specs(model, config.freeze, patterns)
        assert not any(p.requires_grad for p in model.vision_encoder.parameters())
        assert all(p.requires_grad for p in model.transformer.parameters())

    def test_a_pattern_in_a_later_stage_counts_from_its_start(self):
        from kempnerforge.config.vlm import FreezeStage, VLMConfig
        from kempnerforge.training.freeze import freeze_meta_at_step

        vlm = VLMConfig(
            max_text_len=8,
            freeze_schedule=[FreezeStage(5, (FreezeSpec("transformer.layers.1.*", True),))],
        )
        spec = {"module": "transformer.layers.1.*", "frozen": True}
        assert spec not in freeze_meta_at_step(4, vlm)
        assert spec in freeze_meta_at_step(5, vlm)

    def test_a_typo_in_toml_fails_when_the_config_loads(self, tmp_path):
        from kempnerforge.config.loader import load_config

        path = tmp_path / "config.toml"
        path.write_text('[vlm]\narch = "joint_decoder"\nfreeze = [{module = "adaptor"}]\n')
        with pytest.raises(ValueError, match="adaptor"):
            load_config(str(path), cli_args=[])


def _masks(model: nn.Module) -> dict[str, bool]:
    return {name: p.requires_grad for name, p in model.named_parameters()}


class TestResumedFreezeOrder:
    """A resume applies the freeze the uninterrupted run had at its step (#512)."""

    @pytest.mark.parametrize(
        ("base", "stages", "patterns"),
        [
            pytest.param(
                [FreezeSpec("transformer"), FreezeSpec("transformer.1.*", False)],
                [FreezeStage(4, (FreezeSpec("transformer"),))],
                DEFAULT_MODULE_PATTERNS,
                id="pattern_under_a_refrozen_alias",
            ),
            pytest.param(
                [FreezeSpec("vision_encoder")],
                [
                    FreezeStage(2, (FreezeSpec("transformer"),)),
                    FreezeStage(4, (FreezeSpec("top", False),)),
                    FreezeStage(6, (FreezeSpec("transformer"),)),
                ],
                {**DEFAULT_MODULE_PATTERNS, "top": ["transformer.1.*"]},
                id="custom_alias_of_issue_512",
            ),
        ],
    )
    def test_a_resume_at_each_step_matches_the_run(self, base, stages, patterns):
        live = _MiniVLM()
        apply_freeze_specs(live, base, patterns)
        for step in range(8):
            for stage in stages:
                if stage.start_step == step:
                    apply_freeze_specs(live, stage.specs, patterns)
            resumed = _MiniVLM()
            apply_freeze_specs(resumed, base, patterns)
            apply_freeze_specs(resumed, effective_freeze(step, base, stages), patterns)
            assert _masks(resumed) == _masks(live), step
        assert not any(v for k, v in _masks(live).items() if k.startswith("transformer.1."))

    def test_a_key_written_again_moves_to_the_end(self):
        base = [FreezeSpec("transformer"), FreezeSpec("transformer.1.*", False)]
        stages = [FreezeStage(4, (FreezeSpec("transformer"),))]
        assert [(s.module, s.frozen) for s in effective_freeze(4, base, stages)] == [
            ("transformer.1.*", False),
            ("transformer", True),
        ]


class TestUnmatchedPatterns:
    def test_lists_only_patterns_that_match_nothing(self, model):
        specs = [
            FreezeSpec("vision_encoder"),
            FreezeSpec("transformer.1.*"),
            FreezeSpec("transfomer.*"),
            FreezeSpec("adapter.bias", False),
            FreezeSpec("transfomer.*", False),
        ]
        assert unmatched_patterns(model, specs, DEFAULT_MODULE_PATTERNS) == ["transfomer.*"]

    def test_an_alias_without_parameters_is_not_checked(self):
        empty = nn.Module()
        empty.vision_encoder = nn.Module()
        specs = [FreezeSpec("vision_encoder")]
        assert unmatched_patterns(empty, specs, DEFAULT_MODULE_PATTERNS) == []


def _build_vlm(vlm):
    import torch

    from kempnerforge.config.adapter import AdapterConfig
    from kempnerforge.config.model import ModelConfig
    from kempnerforge.config.vision import VisionEncoderConfig
    from kempnerforge.distributed.parallel import build_parallel_model

    return build_parallel_model(
        ModelConfig(dim=32, n_layers=2, n_heads=2, n_kv_heads=2, vocab_size=64, max_seq_len=16),
        torch.device("cpu"),
        None,
        vision_config=VisionEncoderConfig(type="random", feature_dim=16, num_tokens=4),
        adapter_config=AdapterConfig(),
        vlm_config=vlm,
    )


class TestBuildRefusesUnmatchedPatterns:
    def test_a_misspelled_pattern_fails_the_build(self):
        from kempnerforge.config.vlm import VLMConfig

        vlm = VLMConfig(max_text_len=8, freeze=[FreezeSpec("transformer.layer.1.*", False)])
        with pytest.raises(ValueError, match=r"\['transformer.layer.1.\*'\]"):
            _build_vlm(vlm)

    def test_a_misspelled_pattern_in_a_stage_fails_the_build(self):
        from kempnerforge.config.vlm import VLMConfig

        stage = FreezeStage(5, (FreezeSpec("transfomer.*", True),))
        vlm = VLMConfig(max_text_len=8, freeze_schedule=[stage])
        with pytest.raises(ValueError, match="transfomer"):
            _build_vlm(vlm)

    def test_matching_patterns_and_an_encoder_without_parameters_build(self):
        from kempnerforge.config.vlm import VLMConfig

        vlm = VLMConfig(
            max_text_len=8,
            freeze=[
                FreezeSpec("vision_encoder"),
                FreezeSpec("transformer"),
                FreezeSpec("transformer.layers.1.*", False),
            ],
            freeze_schedule=[FreezeStage(5, (FreezeSpec("adapter.*", True),))],
        )
        model = _build_vlm(vlm)
        masks = _masks(model)
        assert all(v for k, v in masks.items() if k.startswith("transformer.layers.1."))
        assert not any(v for k, v in masks.items() if k.startswith("transformer.layers.0."))
