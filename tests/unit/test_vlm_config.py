"""Unit tests for VLMConfig and its integration with ModelConfig."""

from __future__ import annotations

import pytest

from kempnerforge.config.model import ModelConfig
from kempnerforge.config.registry import registry
from kempnerforge.config.vlm import (
    DEFAULT_MODULE_PATTERNS,
    FreezeSpec,
    FreezeStage,
    JointDecoderConfig,
    MoTConfig,
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

    def test_for_arch_joint_decoder(self):
        cfg = VLMConfig.for_arch("joint_decoder", vision_encoder="random")
        assert isinstance(cfg, JointDecoderConfig)
        assert cfg.arch == "joint_decoder"

    def test_for_arch_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown vlm.arch"):
            VLMConfig.for_arch("bogus", vision_encoder="random")

    def test_for_arch_error_lists_registered_arches(self):
        try:
            VLMConfig.for_arch("bogus", vision_encoder="random")
        except ValueError as e:
            msg = str(e)
            assert "joint_decoder" in msg

    def test_for_arch_cross_attention_still_reserved(self):
        """Cross-Attention is the remaining reserved arch on this branch;
        TOMLs aiming at it should get a clear NotImplementedError."""
        with pytest.raises(NotImplementedError, match="reserved"):
            VLMConfig.for_arch("cross_attention", vision_encoder="random")

    def test_registry_has_jd_and_mot(self):
        archs = set(registry.list_vlm_configs())
        assert {"joint_decoder", "mot"} <= archs


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
        cfg = MoTConfig(vision_encoder="random")
        assert cfg.arch == "mot"
        assert isinstance(cfg, VLMConfig)
        assert cfg.mot_modalities == ("image", "text")
        assert cfg.mot_image_n_heads == 0
        assert cfg.mot_image_n_kv_heads == 0
        assert cfg.mot_warm_start_from_text is False

    def test_module_patterns_has_mot_alias(self):
        cfg = MoTConfig(vision_encoder="random")
        assert "mot" in cfg.module_patterns
        assert cfg.module_patterns["mot"] == [
            "transformer.layers",
            "transformer.layers.*",
        ]
        # Base aliases still present.
        assert "transformer" in cfg.module_patterns
        assert "vision_encoder" in cfg.module_patterns

    def test_for_arch_mot_returns_subclass(self):
        cfg = VLMConfig.for_arch("mot", vision_encoder="random")
        assert isinstance(cfg, MoTConfig)
        assert cfg.arch == "mot"

    def test_for_arch_mot_with_overrides(self):
        cfg = VLMConfig.for_arch(
            "mot",
            vision_encoder="random",
            num_tokens=64,
            mot_warm_start_from_text=True,
            mot_warm_start_path="/tmp/jd_ckpt.pt",
        )
        assert isinstance(cfg, MoTConfig)
        assert cfg.num_tokens == 64
        assert cfg.mot_warm_start_from_text is True

    def test_residual_stream_image_tokens_matches_num_tokens(self):
        cfg = MoTConfig(vision_encoder="random", num_tokens=128)
        assert cfg.residual_stream_image_tokens() == 128

    def test_resolved_image_heads_zero_inherits_model(self):
        cfg = MoTConfig(vision_encoder="random")
        n, kv = cfg.resolved_image_heads(model_n_heads=32)
        assert n == 32 and kv == 32  # MHA default (no model_n_kv_heads passed)

    def test_resolved_image_heads_inherits_model_gqa(self):
        """When the text backbone is GQA, image kv_heads inherits the text
        kv_heads default (so v1's equal-head-count check still passes)."""
        cfg = MoTConfig(vision_encoder="random")
        n, kv = cfg.resolved_image_heads(model_n_heads=32, model_n_kv_heads=8)
        assert n == 32 and kv == 8  # inherits text-side GQA

    def test_resolved_image_heads_explicit_n_kv_heads(self):
        cfg = MoTConfig(vision_encoder="random", mot_image_n_kv_heads=4)
        n, kv = cfg.resolved_image_heads(model_n_heads=32)
        assert n == 32 and kv == 4  # GQA on image path

    def test_resolved_image_heads_explicit_n_heads(self):
        cfg = MoTConfig(vision_encoder="random", mot_image_n_heads=8)
        n, kv = cfg.resolved_image_heads(model_n_heads=32)
        assert n == 8 and kv == 8

    def test_resolved_image_heads_zero_model_raises(self):
        cfg = MoTConfig(vision_encoder="random")
        with pytest.raises(ValueError, match="model_n_heads must be positive"):
            cfg.resolved_image_heads(model_n_heads=0)

    def test_mot_modalities_must_include_text(self):
        with pytest.raises(ValueError, match="must include 'text'"):
            MoTConfig(vision_encoder="random", mot_modalities=("image", "audio"))

    def test_mot_modalities_must_include_image(self):
        with pytest.raises(ValueError, match="must include 'image'"):
            MoTConfig(vision_encoder="random", mot_modalities=("text", "audio"))

    def test_mot_modalities_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2 entries"):
            MoTConfig(vision_encoder="random", mot_modalities=("text",))

    def test_mot_modalities_duplicates_raise(self):
        with pytest.raises(ValueError, match="duplicates"):
            MoTConfig(vision_encoder="random", mot_modalities=("image", "text", "text"))

    def test_negative_image_heads_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            MoTConfig(vision_encoder="random", mot_image_n_heads=-1)

    def test_negative_image_kv_heads_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            MoTConfig(vision_encoder="random", mot_image_n_kv_heads=-1)

    def test_inherits_base_validation(self):
        """Base-class validations still fire on MoT subclass."""
        with pytest.raises(ValueError, match="vision_encoder must be set"):
            MoTConfig()

    def test_warm_start_path_default_empty(self):
        cfg = MoTConfig(vision_encoder="random")
        assert cfg.mot_warm_start_path == ""

    def test_warm_start_from_text_requires_path(self):
        with pytest.raises(ValueError, match="mot_warm_start_path"):
            MoTConfig(vision_encoder="random", mot_warm_start_from_text=True)

    def test_warm_start_from_text_with_path(self):
        cfg = MoTConfig(
            vision_encoder="random",
            mot_warm_start_from_text=True,
            mot_warm_start_path="/tmp/jd_ckpt.pt",
        )
        assert cfg.mot_warm_start_from_text is True
        assert cfg.mot_warm_start_path == "/tmp/jd_ckpt.pt"

    def test_warm_start_path_set_without_flag_is_valid(self):
        """Path can be set without the flag; the flag is the gate."""
        cfg = MoTConfig(
            vision_encoder="random",
            mot_warm_start_from_text=False,
            mot_warm_start_path="/tmp/jd_ckpt.pt",
        )
        assert cfg.mot_warm_start_from_text is False
        assert cfg.mot_warm_start_path == "/tmp/jd_ckpt.pt"


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
