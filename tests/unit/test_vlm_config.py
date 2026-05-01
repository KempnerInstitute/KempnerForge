"""Unit tests for VLMConfig and its integration with ModelConfig."""

from __future__ import annotations

import pytest

from kempnerforge.config.model import ModelConfig
from kempnerforge.config.registry import registry
from kempnerforge.config.vlm import (
    DEFAULT_MODULE_PATTERNS,
    CrossAttentionConfig,
    FreezeSpec,
    FreezeStage,
    JointDecoderConfig,
    VLMConfig,
)


class TestVLMConfigValidation:
    def test_unknown_arch(self):
        with pytest.raises(ValueError, match="Unknown vlm.arch"):
            VLMConfig(arch="bogus", vision_encoder="random")

    def test_requires_vision_encoder(self):
        with pytest.raises(ValueError, match="vision_encoder must be set"):
            VLMConfig(arch="joint_decoder")

    def test_unknown_adapter_activation(self):
        with pytest.raises(ValueError, match="adapter_activation"):
            VLMConfig(vision_encoder="random", adapter_activation="tanh")

    def test_negative_feature_dim(self):
        with pytest.raises(ValueError, match="non-negative"):
            VLMConfig(vision_encoder="random", feature_dim=-1)

    def test_zero_max_text_len(self):
        with pytest.raises(ValueError, match="max_text_len"):
            VLMConfig(vision_encoder="random", max_text_len=0)

    def test_freeze_schedule_monotonic(self):
        with pytest.raises(ValueError, match="strictly monotonic"):
            VLMConfig(
                vision_encoder="random",
                freeze_schedule=[
                    FreezeStage(start_step=100, specs=(FreezeSpec("transformer"),)),
                    FreezeStage(start_step=50, specs=(FreezeSpec("transformer"),)),
                ],
            )

    def test_default_is_valid(self):
        cfg = VLMConfig(vision_encoder="random")
        assert cfg.arch == "joint_decoder"
        assert cfg.freeze == [FreezeSpec("vision_encoder", True)]
        # module_patterns is a copy, not shared with the module-level default
        assert cfg.module_patterns == DEFAULT_MODULE_PATTERNS
        assert cfg.module_patterns is not DEFAULT_MODULE_PATTERNS


class TestVLMConfigSubclasses:
    """Subclassed VLMConfig: discriminated union via registry."""

    def test_joint_decoder_config_construction(self):
        cfg = JointDecoderConfig(vision_encoder="random")
        assert cfg.arch == "joint_decoder"
        assert isinstance(cfg, VLMConfig)

    def test_cross_attention_config_construction(self):
        cfg = CrossAttentionConfig(vision_encoder="random")
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
            CrossAttentionConfig(vision_encoder="random", cross_attention_every_n_layers=0)

    def test_cross_attention_negative_heads(self):
        with pytest.raises(ValueError, match="non-negative"):
            CrossAttentionConfig(vision_encoder="random", cross_attention_n_heads=-1)

    def test_for_arch_joint_decoder(self):
        cfg = VLMConfig.for_arch("joint_decoder", vision_encoder="random")
        assert isinstance(cfg, JointDecoderConfig)
        assert cfg.arch == "joint_decoder"

    def test_for_arch_cross_attention(self):
        cfg = VLMConfig.for_arch(
            "cross_attention",
            vision_encoder="random",
            cross_attention_every_n_layers=2,
        )
        assert isinstance(cfg, CrossAttentionConfig)
        assert cfg.cross_attention_every_n_layers == 2

    def test_for_arch_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown vlm.arch"):
            VLMConfig.for_arch("bogus", vision_encoder="random")

    def test_for_arch_error_lists_registered_arches(self):
        try:
            VLMConfig.for_arch("bogus", vision_encoder="random")
        except ValueError as e:
            msg = str(e)
            assert "joint_decoder" in msg
            assert "cross_attention" in msg

    def test_for_arch_mot_still_reserved(self):
        """MoT is the remaining reserved arch on this branch; TOMLs aiming
        at it should get a clear NotImplementedError."""
        with pytest.raises(NotImplementedError, match="reserved"):
            VLMConfig.for_arch("mot", vision_encoder="random")

    def test_registry_has_jd_and_ca(self):
        archs = set(registry.list_vlm_configs())
        assert {"joint_decoder", "cross_attention"} <= archs


class TestCrossAttentionResolvedHeads:
    """`cross_attention_n_heads=0` and `cross_attention_n_kv_heads=0` both
    resolve against ModelConfig.n_heads at build time. The block
    constructor never observes 0.
    """

    def test_both_zero_resolve_to_model_n_heads_mha(self):
        cfg = CrossAttentionConfig(vision_encoder="random")
        n, kv = cfg.resolved_heads(model_n_heads=32)
        assert n == 32
        assert kv == 32

    def test_n_heads_zero_n_kv_heads_explicit_gqa(self):
        cfg = CrossAttentionConfig(vision_encoder="random", cross_attention_n_kv_heads=4)
        n, kv = cfg.resolved_heads(model_n_heads=32)
        assert n == 32
        assert kv == 4

    def test_n_heads_explicit_n_kv_heads_zero_mha(self):
        cfg = CrossAttentionConfig(vision_encoder="random", cross_attention_n_heads=8)
        n, kv = cfg.resolved_heads(model_n_heads=32)
        assert n == 8
        assert kv == 8

    def test_both_explicit(self):
        cfg = CrossAttentionConfig(
            vision_encoder="random",
            cross_attention_n_heads=16,
            cross_attention_n_kv_heads=4,
        )
        n, kv = cfg.resolved_heads(model_n_heads=32)
        assert n == 16
        assert kv == 4

    def test_zero_model_n_heads_raises(self):
        cfg = CrossAttentionConfig(vision_encoder="random")
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


class TestModelConfigWithVLM:
    def test_is_vlm_false_by_default(self):
        mc = ModelConfig()
        assert mc.is_vlm is False
        assert mc.vlm is None

    def test_is_vlm_true_when_set(self):
        mc = ModelConfig(
            max_seq_len=1024,
            vlm=VLMConfig(vision_encoder="random", num_tokens=64, max_text_len=512),
        )
        assert mc.is_vlm is True

    def test_max_seq_len_cross_check_raises(self):
        with pytest.raises(ValueError, match="max_seq_len.*insufficient"):
            ModelConfig(
                max_seq_len=128,
                vlm=VLMConfig(
                    vision_encoder="random",
                    num_tokens=64,
                    max_text_len=512,
                ),
            )

    def test_max_seq_len_ok_when_large_enough(self):
        mc = ModelConfig(
            max_seq_len=600,
            vlm=VLMConfig(
                vision_encoder="random",
                num_tokens=64,
                max_text_len=512,
            ),
        )
        assert mc.max_seq_len == 600

    def test_num_tokens_zero_defers_check(self):
        """num_tokens=0 means 'infer at build time'; config-time check is skipped."""
        mc = ModelConfig(
            max_seq_len=64,
            vlm=VLMConfig(
                vision_encoder="random",
                num_tokens=0,
                max_text_len=512,
            ),
        )
        assert mc.is_vlm is True
