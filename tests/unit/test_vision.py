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
    ``hidden_size`` off of it.

    The ``has_cls_token`` flag controls the forward output shape so a test
    can mirror either a CLIP-style tower (prepends a CLS token at position 0,
    output length = n_patches + 1) or a SigLIP-style tower (no CLS, output
    length = n_patches). Tests must pair this flag with the right encoder
    builder so claim and actual shape agree.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 64,
        has_cls_token: bool = True,
    ) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.config = SimpleNamespace(
            image_size=image_size, patch_size=patch_size, hidden_size=hidden_size
        )
        self._hidden_size = hidden_size
        self._has_cls_token = has_cls_token

    def forward(self, pixel_values: torch.Tensor) -> _FakeOutput:  # type: ignore[override]
        B = pixel_values.shape[0]
        n_patches = (self.config.image_size // self.config.patch_size) ** 2
        n_out = n_patches + 1 if self._has_cls_token else n_patches
        return _FakeOutput(torch.zeros(B, n_out, self._hidden_size))


class _FakeAutoModel:
    """CLIP-style ``transformers.AutoModel`` substitute. Wraps a CLIP-shaped
    ``_FakeVisionTower`` (forward output prepends a CLS token at position 0)."""

    @staticmethod
    def from_pretrained(path: str) -> torch.nn.Module:  # noqa: ARG004
        wrapper = torch.nn.Module()
        wrapper.vision_model = _FakeVisionTower(has_cls_token=True)  # type: ignore[attr-defined]
        return wrapper


class _FakeAutoModelSigLIP:
    """SigLIP-style ``transformers.AutoModel`` substitute. Wraps a
    SigLIP-shaped ``_FakeVisionTower`` (forward output is patch tokens only,
    no CLS token at position 0)."""

    @staticmethod
    def from_pretrained(path: str) -> torch.nn.Module:  # noqa: ARG004
        wrapper = torch.nn.Module()
        wrapper.vision_model = _FakeVisionTower(has_cls_token=False)  # type: ignore[attr-defined]
        return wrapper


class TestHFVisionEncoderMocked:
    """Exercise the HF-encoder build/forward paths without network access by
    swapping ``transformers.AutoModel`` for an in-process fake."""

    def test_siglip2_no_cls_no_off_by_one(self, monkeypatch):
        """SigLIP2 has no CLS token. The builder must set num_tokens to
        n_patches (not n_patches + 1) and the forward output must agree.
        Regression test for the pre-fix off-by-one where the encoder
        unconditionally added +1 whenever ``hidden_size`` was truthy.
        """
        import sys

        fake_mod = type(sys)("transformers")
        fake_mod.AutoModel = _FakeAutoModelSigLIP  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_mod)

        builder = registry.get_vision_encoder("siglip2")
        enc = builder("dummy/path")
        # Default 224/16: 14*14 = 196 patches; no CLS, so num_tokens == n_patches.
        assert enc.feature_dim == 64
        assert enc.num_tokens == 196
        out = enc(torch.zeros(2, 3, 224, 224))
        assert out.shape == (2, 196, 64)

    def test_clip_no_strip_keeps_cls(self, monkeypatch):
        """When has_cls_token=True and strip_cls=False, num_tokens includes
        the CLS position (n_patches + 1). Not used by either registered
        builder today, but the branch is part of the contract.
        """
        import sys

        fake_mod = type(sys)("transformers")
        fake_mod.AutoModel = _FakeAutoModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_mod)

        from kempnerforge.model.vision import _HFVisionEncoder

        enc = _HFVisionEncoder("dummy/path", strip_cls=False, has_cls_token=True)
        assert enc.num_tokens == 196 + 1
        out = enc(torch.zeros(1, 3, 224, 224))
        assert out.shape == (1, 197, 64)

    def test_strip_cls_with_no_cls_token_raises(self):
        """``strip_cls=True`` paired with ``has_cls_token=False`` is
        meaningless (nothing to strip) and must raise at construction
        rather than silently underreporting ``num_tokens``.
        """
        from kempnerforge.model.vision import _HFVisionEncoder

        with pytest.raises(ValueError, match="strip_cls=True is meaningless"):
            _HFVisionEncoder("dummy/path", strip_cls=True, has_cls_token=False)

    @pytest.mark.parametrize(
        "builder_name,fake_auto,expected_tokens",
        [
            ("siglip2", _FakeAutoModelSigLIP, 196),
            ("clip", _FakeAutoModel, 196),
        ],
    )
    def test_dryrun_shape_matches_num_tokens(
        self, monkeypatch, builder_name, fake_auto, expected_tokens
    ):
        """``enc(pixels).shape[1] == enc.num_tokens`` for every registered
        HF-backed encoder. End-to-end regression guard: if a builder ever
        claims a token count that disagrees with its forward output, this
        test fires regardless of the underlying cause.
        """
        import sys

        fake_mod = type(sys)("transformers")
        fake_mod.AutoModel = fake_auto  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_mod)

        builder = registry.get_vision_encoder(builder_name)
        enc = builder("dummy/path")
        assert enc.num_tokens == expected_tokens
        out = enc(torch.zeros(1, 3, 224, 224))
        assert out.shape[1] == enc.num_tokens, (
            f"{builder_name}: encoder claims num_tokens={enc.num_tokens} "
            f"but forward output has shape[1]={out.shape[1]}"
        )

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
        encoder treats the whole model as the vision tower.
        Uses the SigLIP2 builder paired with a SigLIP-shape fake tower so
        the encoder's ``has_cls_token=False`` matches what the tower
        actually outputs.
        """
        import sys

        class _BareAuto:
            @staticmethod
            def from_pretrained(path: str) -> torch.nn.Module:  # noqa: ARG004
                # Top-level module IS the tower (no .vision_model attr).
                # SigLIP-shape: no CLS at position 0.
                return _FakeVisionTower(
                    image_size=64, patch_size=16, hidden_size=32, has_cls_token=False
                )

        fake_mod = type(sys)("transformers")
        fake_mod.AutoModel = _BareAuto  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_mod)

        builder = registry.get_vision_encoder("siglip2")
        enc = builder("dummy/path")
        # 64 / 16 = 4; 4*4 = 16 patches; SigLIP -> no +1.
        assert enc.feature_dim == 32
        assert enc.num_tokens == 16
