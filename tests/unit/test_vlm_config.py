"""Unit tests for VLMConfig (arch-only after the schema flip).

Vision-encoder fields (``type``/``path``/``feature_dim``/``num_tokens``)
moved to ``VisionEncoderConfig`` and live in ``tests/unit/test_vision_config.py``.
Adapter fields (``type``/``hidden_dim``/``activation``) moved to
``AdapterConfig`` and live in ``tests/unit/test_adapter.py``.
``JobConfig`` cross-section validation lives in ``tests/unit/test_config.py``.
"""

from __future__ import annotations

import pytest

from kempnerforge.config.registry import registry
from kempnerforge.config.vlm import (
    DEFAULT_MODULE_PATTERNS,
    CrossAttentionConfig,
    FreezeSpec,
    FreezeStage,
    JointDecoderConfig,
    MoTConfig,
    VLMConfig,
)


class TestVLMConfigValidation:
    def test_unknown_arch(self):
        with pytest.raises(ValueError, match="Unknown vlm.arch"):
            VLMConfig(arch="bogus")

    def test_zero_max_text_len(self):
        with pytest.raises(ValueError, match="max_text_len"):
            VLMConfig(max_text_len=0)

    def test_freeze_schedule_monotonic(self):
        with pytest.raises(ValueError, match="strictly monotonic"):
            VLMConfig(
                freeze_schedule=[
                    FreezeStage(start_step=100, specs=(FreezeSpec("transformer"),)),
                    FreezeStage(start_step=50, specs=(FreezeSpec("transformer"),)),
                ],
            )

    def test_freeze_schedule_valid_monotonic(self):
        cfg = VLMConfig(
            freeze_schedule=[
                FreezeStage(start_step=50, specs=(FreezeSpec("transformer"),)),
                FreezeStage(start_step=100, specs=(FreezeSpec("vision_encoder"),)),
            ],
        )
        assert [s.start_step for s in cfg.freeze_schedule] == [50, 100]

    def test_default_is_valid(self):
        cfg = VLMConfig()
        assert cfg.arch == "joint_decoder"
        assert cfg.freeze == [FreezeSpec("vision_encoder", True)]
        # module_patterns is a copy, not shared with the module-level default
        assert cfg.module_patterns == DEFAULT_MODULE_PATTERNS
        assert cfg.module_patterns is not DEFAULT_MODULE_PATTERNS


class TestVLMConfigSubclasses:
    """Subclassed VLMConfig: discriminated union via registry."""

    def test_joint_decoder_config_construction(self):
        cfg = JointDecoderConfig()
        assert cfg.arch == "joint_decoder"
        assert isinstance(cfg, VLMConfig)

    def test_cross_attention_config_construction(self):
        cfg = CrossAttentionConfig()
        assert cfg.arch == "cross_attention"
        assert cfg.cross_attention_every_n_layers == 4
        assert cfg.cross_attention_n_heads == 0
        assert cfg.cross_attention_n_kv_heads == 0
        # CA-specific module alias added to module_patterns
        assert "cross_attention" in cfg.module_patterns
        assert cfg.module_patterns["cross_attention"] == [
            "transformer.cross_attention_layers",
            "transformer.cross_attention_layers.*",
        ]
        # Base aliases still present
        assert "transformer" in cfg.module_patterns
        assert "vision_encoder" in cfg.module_patterns

    def test_cross_attention_invalid_cadence(self):
        with pytest.raises(ValueError, match="cross_attention_every_n_layers"):
            CrossAttentionConfig(cross_attention_every_n_layers=0)

    def test_cross_attention_negative_heads(self):
        with pytest.raises(ValueError, match="non-negative"):
            CrossAttentionConfig(cross_attention_n_heads=-1)

    def test_for_arch_joint_decoder(self):
        cfg = VLMConfig.for_arch("joint_decoder")
        assert isinstance(cfg, JointDecoderConfig)
        assert cfg.arch == "joint_decoder"

    def test_for_arch_cross_attention(self):
        cfg = VLMConfig.for_arch("cross_attention", cross_attention_every_n_layers=2)
        assert isinstance(cfg, CrossAttentionConfig)
        assert cfg.cross_attention_every_n_layers == 2

    def test_for_arch_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown vlm.arch"):
            VLMConfig.for_arch("bogus")

    def test_for_arch_error_lists_registered_arches(self):
        try:
            VLMConfig.for_arch("bogus")
        except ValueError as e:
            msg = str(e)
            assert "joint_decoder" in msg
            assert "cross_attention" in msg

    def test_registry_has_all_archs(self):
        archs = set(registry.list_vlm_configs())
        assert {"joint_decoder", "cross_attention", "mot"} <= archs


class TestCrossAttentionResolvedHeads:
    """`cross_attention_n_heads=0` and `cross_attention_n_kv_heads=0` both
    resolve against ModelConfig.n_heads at build time. The block
    constructor never observes 0.
    """

    def test_both_zero_resolve_to_model_n_heads_mha(self):
        cfg = CrossAttentionConfig()
        n, kv = cfg.resolved_heads(model_n_heads=32)
        assert n == 32
        assert kv == 32

    def test_n_heads_zero_n_kv_heads_explicit_gqa(self):
        cfg = CrossAttentionConfig(cross_attention_n_kv_heads=4)
        n, kv = cfg.resolved_heads(model_n_heads=32)
        assert n == 32
        assert kv == 4

    def test_n_heads_explicit_n_kv_heads_zero_mha(self):
        cfg = CrossAttentionConfig(cross_attention_n_heads=8)
        n, kv = cfg.resolved_heads(model_n_heads=32)
        assert n == 8
        assert kv == 8

    def test_both_explicit(self):
        cfg = CrossAttentionConfig(
            cross_attention_n_heads=16,
            cross_attention_n_kv_heads=4,
        )
        n, kv = cfg.resolved_heads(model_n_heads=32)
        assert n == 16
        assert kv == 4

    def test_zero_model_n_heads_raises(self):
        cfg = CrossAttentionConfig()
        with pytest.raises(ValueError, match="model_n_heads must be positive"):
            cfg.resolved_heads(model_n_heads=0)


class TestModalityStrategyRegistry:
    """Registry methods for ModalityStrategy. Strategies themselves land in
    Step 5; this only exercises the registry plumbing."""

    def test_register_get_roundtrip(self):
        @registry.register_modality_strategy("test_arch_register_get")
        class _Strategy:
            pass

        retrieved = registry.get_modality_strategy("test_arch_register_get")
        assert retrieved is _Strategy

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="modality_strategy"):
            registry.get_modality_strategy("nonexistent_arch")

    def test_list_includes_registered(self):
        @registry.register_modality_strategy("test_arch_list")
        class _Strategy:
            pass

        names = registry.list_modality_strategies()
        assert "test_arch_list" in names

    def test_double_register_raises(self):
        @registry.register_modality_strategy("test_arch_dup")
        class _Strategy:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @registry.register_modality_strategy("test_arch_dup")
            class _Other:
                pass


class TestMoTConfig:
    """`MoTConfig` (registered subclass for arch='mot')."""

    def test_construction_defaults(self):
        cfg = MoTConfig()
        assert cfg.arch == "mot"
        assert isinstance(cfg, VLMConfig)
        assert cfg.mot_modalities == ("image", "text")
        assert cfg.mot_image_n_heads == 0
        assert cfg.mot_image_n_kv_heads == 0
        assert cfg.mot_warm_start_from_text is False

    def test_module_patterns_has_mot_alias(self):
        cfg = MoTConfig()
        assert "mot" in cfg.module_patterns
        assert cfg.module_patterns["mot"] == [
            "transformer.layers",
            "transformer.layers.*",
        ]
        # Base aliases still present.
        assert "transformer" in cfg.module_patterns
        assert "vision_encoder" in cfg.module_patterns

    def test_for_arch_mot_returns_subclass(self):
        cfg = VLMConfig.for_arch("mot")
        assert isinstance(cfg, MoTConfig)
        assert cfg.arch == "mot"

    def test_for_arch_mot_with_overrides(self):
        cfg = VLMConfig.for_arch(
            "mot",
            mot_warm_start_from_text=True,
            mot_warm_start_path="/tmp/jd_ckpt.pt",
        )
        assert isinstance(cfg, MoTConfig)
        assert cfg.mot_warm_start_from_text is True

    def test_residual_stream_image_tokens_passes_through(self):
        cfg = MoTConfig()
        assert cfg.residual_stream_image_tokens(128) == 128

    def test_resolved_image_heads_zero_inherits_model(self):
        cfg = MoTConfig()
        n, kv = cfg.resolved_image_heads(model_n_heads=32)
        assert n == 32 and kv == 32  # MHA default (no model_n_kv_heads passed)

    def test_resolved_image_heads_inherits_model_gqa(self):
        """When the text backbone is GQA, image kv_heads inherits the text
        kv_heads default (so v1's equal-head-count check still passes)."""
        cfg = MoTConfig()
        n, kv = cfg.resolved_image_heads(model_n_heads=32, model_n_kv_heads=8)
        assert n == 32 and kv == 8  # inherits text-side GQA

    def test_resolved_image_heads_explicit_n_kv_heads(self):
        cfg = MoTConfig(mot_image_n_kv_heads=4)
        n, kv = cfg.resolved_image_heads(model_n_heads=32)
        assert n == 32 and kv == 4  # GQA on image path

    def test_resolved_image_heads_explicit_n_heads(self):
        cfg = MoTConfig(mot_image_n_heads=8)
        n, kv = cfg.resolved_image_heads(model_n_heads=32)
        assert n == 8 and kv == 8

    def test_resolved_image_heads_zero_model_raises(self):
        cfg = MoTConfig()
        with pytest.raises(ValueError, match="model_n_heads must be positive"):
            cfg.resolved_image_heads(model_n_heads=0)

    def test_mot_modalities_must_include_text(self):
        with pytest.raises(ValueError, match="must include 'text'"):
            MoTConfig(mot_modalities=("image", "audio"))

    def test_mot_modalities_must_include_image(self):
        with pytest.raises(ValueError, match="must include 'image'"):
            MoTConfig(mot_modalities=("text", "audio"))

    def test_mot_modalities_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2 entries"):
            MoTConfig(mot_modalities=("text",))

    def test_mot_modalities_duplicates_raise(self):
        with pytest.raises(ValueError, match="duplicates"):
            MoTConfig(mot_modalities=("image", "text", "text"))

    def test_negative_image_heads_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            MoTConfig(mot_image_n_heads=-1)

    def test_negative_image_kv_heads_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            MoTConfig(mot_image_n_kv_heads=-1)

    def test_warm_start_path_default_empty(self):
        cfg = MoTConfig()
        assert cfg.mot_warm_start_path == ""

    def test_warm_start_from_text_requires_path(self):
        with pytest.raises(ValueError, match="mot_warm_start_path"):
            MoTConfig(mot_warm_start_from_text=True)

    def test_warm_start_from_text_with_path(self):
        cfg = MoTConfig(
            mot_warm_start_from_text=True,
            mot_warm_start_path="/tmp/jd_ckpt.pt",
        )
        assert cfg.mot_warm_start_from_text is True
        assert cfg.mot_warm_start_path == "/tmp/jd_ckpt.pt"

    def test_warm_start_path_set_without_flag_is_valid(self):
        """Path can be set without the flag; the flag is the gate."""
        cfg = MoTConfig(
            mot_warm_start_from_text=False,
            mot_warm_start_path="/tmp/jd_ckpt.pt",
        )
        assert cfg.mot_warm_start_from_text is False
        assert cfg.mot_warm_start_path == "/tmp/jd_ckpt.pt"
