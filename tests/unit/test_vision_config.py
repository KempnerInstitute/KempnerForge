"""Unit tests for VisionEncoderConfig (config/vision.py)."""

from __future__ import annotations

import pytest
import torch

import kempnerforge.model.vision  # noqa: F401
from kempnerforge.config.loader import load_config
from kempnerforge.config.registry import registry
from kempnerforge.config.vision import VisionEncoderConfig


class TestVisionEncoderConfig:
    def test_defaults(self):
        cfg = VisionEncoderConfig()
        assert cfg.type == "random"
        assert cfg.path == ""
        assert cfg.feature_dim == 0
        assert cfg.num_tokens == 0

    def test_explicit_fields(self):
        cfg = VisionEncoderConfig(
            type="siglip2",
            path="google/siglip2-base-patch16-224",
            feature_dim=768,
            num_tokens=196,
        )
        assert cfg.type == "siglip2"
        assert cfg.path == "google/siglip2-base-patch16-224"
        assert cfg.feature_dim == 768
        assert cfg.num_tokens == 196

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown vision_encoder.type"):
            VisionEncoderConfig(type="not_a_real_encoder")

    def test_negative_feature_dim_raises(self):
        with pytest.raises(ValueError, match="feature_dim must be non-negative"):
            VisionEncoderConfig(feature_dim=-1)

    def test_negative_num_tokens_raises(self):
        with pytest.raises(ValueError, match="num_tokens must be non-negative"):
            VisionEncoderConfig(num_tokens=-1)

    def test_zero_feature_dim_is_inference_sentinel(self):
        """``feature_dim=0`` means 'infer from the encoder at build time';
        it is not a validation error."""
        cfg = VisionEncoderConfig(feature_dim=0)
        assert cfg.feature_dim == 0

    def test_zero_num_tokens_is_inference_sentinel(self):
        """``num_tokens=0`` means 'infer at build time'; the cross-check
        against ``model.max_seq_len`` then runs in ``build_vlm_wrapper``
        using the encoder's resolved value."""
        cfg = VisionEncoderConfig(num_tokens=0)
        assert cfg.num_tokens == 0

    def test_registered_types_include_known_encoders(self):
        """Sanity: the registry knows about the three encoders the
        codebase ships, so the type-validation error message is useful."""
        names = registry.list_vision_encoders()
        assert "random" in names
        assert "siglip2" in names
        assert "clip" in names


class TestVisionEncoderMixedPrecision:
    def test_unset_by_default(self):
        cfg = VisionEncoderConfig()
        assert cfg.mixed_precision is None
        assert cfg.compute_dtype is None

    @pytest.mark.parametrize(
        ("precision", "dtype"), [("bf16", torch.bfloat16), ("fp32", torch.float32)]
    )
    def test_resolves_to_the_compute_dtype(self, precision, dtype):
        assert VisionEncoderConfig(mixed_precision=precision).compute_dtype == dtype

    def test_loads_from_toml_and_cli(self, tmp_path):
        toml = tmp_path / "vlm.toml"
        toml.write_text(
            '[vision_encoder]\ntype = "random"\nmixed_precision = "fp32"\n\n'
            "[vlm]\nmax_text_len = 32\n"
        )
        config = load_config(str(toml), cli_args=[])
        assert config.vision_encoder is not None
        assert config.vision_encoder.mixed_precision == "fp32"
        assert config.train.mixed_precision == "bf16"

        override = load_config(str(toml), cli_args=["--vision_encoder.mixed_precision=bf16"])
        assert override.vision_encoder is not None
        assert override.vision_encoder.compute_dtype == torch.bfloat16

        unset = load_config(str(toml), cli_args=["--vision_encoder.mixed_precision=None"])
        assert unset.vision_encoder is not None
        assert unset.vision_encoder.mixed_precision is None

    @pytest.mark.parametrize("precision", ["fp16", "fp8", "float32", ""])
    def test_toml_rejects_other_values(self, tmp_path, precision):
        toml = tmp_path / "vlm.toml"
        toml.write_text(f'[vision_encoder]\ntype = "random"\nmixed_precision = "{precision}"\n')
        with pytest.raises(ValueError, match=r"not in allowed Literal values: \('bf16', 'fp32'\)"):
            load_config(str(toml), cli_args=[])
