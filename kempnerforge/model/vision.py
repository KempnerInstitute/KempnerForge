"""Vision encoders for VLM training.

A vision encoder turns ``(B, 3, H, W)`` pixel values into a bag of
``(B, num_tokens, feature_dim)`` patch tokens that the VLM adapter maps
into the language-model embedding space.

Encoders register themselves via ``registry.register_vision_encoder``.
Currently shipped:

- ``random`` — small deterministic stub for tests and smoke configs. No
  network access required. Produces reproducible noise for a given seed.
- ``siglip2`` / ``clip`` — thin wrappers around HuggingFace
  ``AutoModel.from_pretrained``. The HF imports are deferred so the
  module is importable on machines without the ``transformers`` package,
  and failures are surfaced with a clear message.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kempnerforge.config.registry import registry


class VisionEncoder(nn.Module):
    """Base class for vision encoders.

    Subclasses must set ``feature_dim`` and ``num_tokens`` before returning
    from ``__init__`` and implement ``forward(pixel_values)`` to produce a
    ``(B, num_tokens, feature_dim)`` tensor.
    """

    feature_dim: int
    num_tokens: int

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class RandomVisionEncoder(VisionEncoder):
    """Deterministic random-token stub.

    The output is computed from a hash of ``pixel_values.sum()`` so the
    same image produces the same tokens across calls; independent of
    model weights so it works under FSDP2 without sharding a real encoder.

    Used in tests and the ``vlm_debug.toml`` smoke config.
    """

    def __init__(self, num_tokens: int = 16, feature_dim: int = 768, seed: int = 0) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        self._seed = seed
        # Carry a trivial buffer so .to(device) / .to(dtype) have something
        # to move; also lets VLMWrapper confirm the module actually lives
        # on the target device.
        self.register_buffer("_anchor", torch.zeros(1, dtype=torch.float32), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]
        # Derive a per-image seed from the input so the same image yields
        # the same tokens. Kept cheap: sum across spatial dims and cast.
        per_image = pixel_values.flatten(1).sum(dim=1)
        anchor = self.get_buffer("_anchor")
        out = torch.empty(
            B,
            self.num_tokens,
            self.feature_dim,
            device=pixel_values.device,
            dtype=anchor.dtype,
        )
        for i in range(B):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self._seed) + int(per_image[i].item() * 1e6))
            out[i] = torch.randn(
                self.num_tokens,
                self.feature_dim,
                generator=gen,
            ).to(device=pixel_values.device, dtype=anchor.dtype)
        return out


@registry.register_vision_encoder("random")
def _build_random(
    path: str = "",
    num_tokens: int | None = None,
    feature_dim: int | None = None,
    **_: Any,
) -> VisionEncoder:
    """Builder for the test stub.

    ``path`` is ignored. When ``num_tokens`` / ``feature_dim`` are None the
    defaults are used.
    """
    return RandomVisionEncoder(
        num_tokens=num_tokens if num_tokens is not None else 16,
        feature_dim=feature_dim if feature_dim is not None else 768,
    )


class _HFVisionEncoder(VisionEncoder):
    """Shared wrapper for HuggingFace vision encoders (SigLIP2, CLIP, ...).

    The HF model's vision tower produces patch tokens; we drop the CLS
    token if present and expose a flat ``(B, num_tokens, feature_dim)``
    output. The text tower and projection head (if any) are discarded.
    """

    def __init__(self, path: str, strip_cls: bool = False) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Loading HuggingFace vision encoders requires `transformers`. "
                "Install it or use the 'random' encoder for tests."
            ) from e
        if not path:
            raise ValueError("vlm.vision_encoder_path must be set for HF-backed vision encoders")
        model = AutoModel.from_pretrained(path)
        # Prefer .vision_model if present (CLIP, SigLIP family); otherwise
        # assume the whole loaded model is the vision tower.
        vision_tower = getattr(model, "vision_model", model)
        self.vision_tower = vision_tower
        self._strip_cls = strip_cls

        # Probe output shape with a tiny dummy input to resolve attributes
        # without relying on HF config fields that differ between models.
        cfg = getattr(vision_tower, "config", None)
        image_size = getattr(cfg, "image_size", 224) if cfg is not None else 224
        patch_size = getattr(cfg, "patch_size", 16) if cfg is not None else 16
        hidden = getattr(cfg, "hidden_size", None) if cfg is not None else None
        n_patches = (image_size // patch_size) ** 2
        self.feature_dim = int(hidden) if hidden else -1  # -1 => resolve via dry run
        self.num_tokens = n_patches if strip_cls else n_patches + (1 if hidden else 0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_tower(pixel_values=pixel_values)
        hidden = getattr(out, "last_hidden_state", out)
        if self._strip_cls:
            hidden = hidden[:, 1:, :]
        return hidden


@registry.register_vision_encoder("siglip2")
def _build_siglip2(
    path: str,
    num_tokens: int | None = None,
    feature_dim: int | None = None,
    **_: Any,
) -> VisionEncoder:
    """Builder for a SigLIP2 vision tower. SigLIP2 has no CLS token."""
    enc = _HFVisionEncoder(path, strip_cls=False)
    if num_tokens is not None:
        enc.num_tokens = num_tokens
    if feature_dim is not None:
        enc.feature_dim = feature_dim
    return enc


@registry.register_vision_encoder("clip")
def _build_clip(
    path: str,
    num_tokens: int | None = None,
    feature_dim: int | None = None,
    **_: Any,
) -> VisionEncoder:
    """Builder for a CLIP ViT vision tower. CLIP output includes a CLS
    token; we strip it so ``num_tokens`` matches the number of image
    patches.
    """
    enc = _HFVisionEncoder(path, strip_cls=True)
    if num_tokens is not None:
        enc.num_tokens = num_tokens
    if feature_dim is not None:
        enc.feature_dim = feature_dim
    return enc
