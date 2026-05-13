"""Unit tests for VisionEncoderConfig (config/vision.py)."""

from __future__ import annotations

import pytest

import kempnerforge.model.vision  # noqa: F401  -- register encoder types
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
