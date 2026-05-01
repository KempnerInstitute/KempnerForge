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


# ---------------------------------------------------------------------------
# _HFVisionEncoder via mocked transformers.AutoModel
# ---------------------------------------------------------------------------


class _FakeOutput:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.last_hidden_state = hidden


class _FakeVisionTower(torch.nn.Module):
    """Stands in for an HF vision tower. Honors a SimpleNamespace ``config``
    so ``_HFVisionEncoder.__init__`` can read ``image_size`` / ``patch_size`` /
    ``hidden_size`` off of it."""

    def __init__(self, image_size: int = 224, patch_size: int = 16, hidden_size: int = 64) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.config = SimpleNamespace(
            image_size=image_size, patch_size=patch_size, hidden_size=hidden_size
        )
        self._hidden_size = hidden_size

    def forward(self, pixel_values: torch.Tensor) -> _FakeOutput:  # type: ignore[override]
        B = pixel_values.shape[0]
        # SigLIP-style tower exposes (B, n_patches, hidden); CLIP-style tower
        # adds a CLS token at position 0. Either way, hand back something
        # plausible — the caller decides whether to strip.
        n_patches = (self.config.image_size // self.config.patch_size) ** 2
        return _FakeOutput(torch.zeros(B, n_patches + 1, self._hidden_size))


class _FakeAutoModel:
    """Substitute for ``transformers.AutoModel`` exposing ``from_pretrained``.

    The fake exposes a ``vision_model`` attribute, mimicking the CLIP/SigLIP
    family layout that ``_HFVisionEncoder`` probes for.
    """

    @staticmethod
    def from_pretrained(path: str) -> torch.nn.Module:  # noqa: ARG004
        wrapper = torch.nn.Module()
        wrapper.vision_model = _FakeVisionTower()  # type: ignore[attr-defined]
        return wrapper


class TestHFVisionEncoderMocked:
    """Exercise the HF-encoder build/forward paths without network access by
    swapping ``transformers.AutoModel`` for an in-process fake."""

    def test_siglip2_builder_no_strip_cls(self, monkeypatch):
        import sys

        fake_mod = type(sys)("transformers")
        fake_mod.AutoModel = _FakeAutoModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_mod)

        builder = registry.get_vision_encoder("siglip2")
        enc = builder("dummy/path")
        # Default 224/16: 14*14 = 196 patches; SigLIP path keeps the +1 token
        # because the probe sees ``hidden_size`` and strip_cls=False.
        assert enc.feature_dim == 64
        assert enc.num_tokens == 196 + 1
        out = enc(torch.zeros(2, 3, 224, 224))
        assert out.shape[0] == 2

    def test_clip_builder_strips_cls(self, monkeypatch):
        import sys

        fake_mod = type(sys)("transformers")
        fake_mod.AutoModel = _FakeAutoModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_mod)

        builder = registry.get_vision_encoder("clip")
        enc = builder("dummy/path")
        assert enc.feature_dim == 64
        assert enc.num_tokens == 196  # CLS stripped
        out = enc(torch.zeros(1, 3, 224, 224))
        # Fake tower returns 197 tokens; CLIP encoder strips the first.
        assert out.shape == (1, 196, 64)

    def test_builder_overrides_apply(self, monkeypatch):
        """num_tokens / feature_dim kwargs override the probed values."""
        import sys

        fake_mod = type(sys)("transformers")
        fake_mod.AutoModel = _FakeAutoModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_mod)

        builder = registry.get_vision_encoder("siglip2")
        enc = builder("dummy/path", num_tokens=42, feature_dim=128)
        assert enc.num_tokens == 42
        assert enc.feature_dim == 128

        builder = registry.get_vision_encoder("clip")
        enc = builder("dummy/path", num_tokens=33, feature_dim=99)
        assert enc.num_tokens == 33
        assert enc.feature_dim == 99

    def test_empty_path_raises(self, monkeypatch):
        import sys

        fake_mod = type(sys)("transformers")
        fake_mod.AutoModel = _FakeAutoModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_mod)

        builder = registry.get_vision_encoder("siglip2")
        with pytest.raises(ValueError, match="vision_encoder_path must be set"):
            builder("")

    def test_falls_back_to_top_level_model_when_no_vision_model(self, monkeypatch):
        """When the loaded HF model lacks a ``.vision_model`` attribute, the
        encoder treats the whole model as the vision tower."""
        import sys

        class _BareAuto:
            @staticmethod
            def from_pretrained(path: str) -> torch.nn.Module:  # noqa: ARG004
                # Top-level module IS the tower (no .vision_model attr).
                return _FakeVisionTower(image_size=64, patch_size=16, hidden_size=32)

        fake_mod = type(sys)("transformers")
        fake_mod.AutoModel = _BareAuto  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_mod)

        builder = registry.get_vision_encoder("siglip2")
        enc = builder("dummy/path")
        # 64 / 16 = 4; 4*4 = 16 patches.
        assert enc.feature_dim == 32
        assert enc.num_tokens == 16 + 1
