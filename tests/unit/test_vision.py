"""Unit tests for vision encoder registry and stubs.

HF-backed encoders (SigLIP2, CLIP) require network access and the
``transformers`` package; tests for them are gated by the
``RUN_HF_TESTS`` env var. Default CI exercises only the random stub.
"""

from __future__ import annotations

import os

import pytest
import torch

# Importing the module registers the builders under the shared registry.
import kempnerforge.model.vision  # noqa: F401
from kempnerforge.config.registry import registry
from kempnerforge.model.vision import RandomVisionEncoder, VisionEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestRandomVisionEncoder:
    def test_output_shape(self):
        enc = RandomVisionEncoder(num_tokens=16, feature_dim=384).to(DEVICE)
        pixels = torch.randn(2, 3, 224, 224, device=DEVICE)
        out = enc(pixels)
        assert out.shape == (2, 16, 384)

    def test_output_on_input_device(self):
        enc = RandomVisionEncoder(num_tokens=8, feature_dim=64).to(DEVICE)
        pixels = torch.randn(1, 3, 32, 32, device=DEVICE)
        out = enc(pixels)
        assert out.device == pixels.device

    def test_deterministic_for_same_input(self):
        enc = RandomVisionEncoder(num_tokens=4, feature_dim=32, seed=123)
        pixels = torch.randn(1, 3, 16, 16)
        a = enc(pixels)
        b = enc(pixels)
        assert torch.equal(a, b)

    def test_different_inputs_yield_different_outputs(self):
        enc = RandomVisionEncoder(num_tokens=4, feature_dim=32, seed=123)
        a = enc(torch.ones(1, 3, 16, 16))
        b = enc(torch.ones(1, 3, 16, 16) * 2)
        assert not torch.equal(a, b)

    def test_dtype_follows_buffer(self):
        enc = RandomVisionEncoder(num_tokens=4, feature_dim=32).to(torch.bfloat16)
        out = enc(torch.randn(1, 3, 16, 16))
        assert out.dtype == torch.bfloat16

    def test_subclasses_visionencoder(self):
        assert issubclass(RandomVisionEncoder, VisionEncoder)


class TestVisionEncoderRegistry:
    def test_random_is_registered(self):
        builder = registry.get_vision_encoder("random")
        enc = builder("", num_tokens=8, feature_dim=128)
        assert isinstance(enc, VisionEncoder)
        assert enc.num_tokens == 8
        assert enc.feature_dim == 128

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="vision_encoder"):
            registry.get_vision_encoder("not_a_real_encoder")

    def test_siglip2_and_clip_are_registered(self):
        """Both HF builders are registered at import time, even if the
        underlying transformers dependency would fail at build time
        without a real HF model on disk."""
        assert "siglip2" in registry.list("vision_encoder")
        assert "clip" in registry.list("vision_encoder")


@pytest.mark.skipif(
    not os.environ.get("RUN_HF_TESTS"),
    reason="HF vision encoder tests require RUN_HF_TESTS=1 and network access",
)
class TestHFEncoders:
    def test_siglip2_load(self):
        builder = registry.get_vision_encoder("siglip2")
        # Use a tiny public checkpoint; swap to project-local if needed.
        enc = builder("google/siglip2-base-patch16-224")
        assert enc.num_tokens > 0
        assert enc.feature_dim > 0
