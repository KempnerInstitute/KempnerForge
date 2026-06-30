"""Per-frame timestamp embedding for the VLM video path.

A video clip enters the model as ``F`` frames in temporal order, but frame
*order* alone does not tell the model *when* each frame occurs: an 8-frame clip
spanning 2 seconds and one spanning 2 minutes both map to frame indices 0..7.
``FrameTimeEmbedding`` injects the actual per-frame timestamp (seconds) so the
model can reason about elapsed time (Molmo2-style temporal grounding).

The timestamp is expanded into sinusoidal features at log-spaced periods (à la
Transformer positional encodings, but over continuous seconds rather than
integer positions), then projected to the model dimension and added to that
frame's visual tokens. The output projection is zero-initialized so the
temporal signal starts at zero and is learned from there — matching the
``CrossAttention`` warm-start convention: at step 0 the module is a no-op on
the model's outputs (the added embedding is zero, so logits are unchanged),
while gradients still flow into the projection, so it begins learning from the
first step.

The frequencies are recomputed in ``forward`` on the input's device rather than
stored in a buffer, so the module is safe to construct under
``torch.device("meta")`` (the meta-device / FSDP build path): the only
parameters are the projection, which materializes like any other ``nn.Linear``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from kempnerforge.config.registry import registry


class TimeEmbedding(nn.Module):
    """Base for per-frame timestamp embeddings (the *additive* family).

    Contract: ``forward(times: (B, F) seconds) -> (B, F, dim)`` — an additive
    embedding added to each frame's visual tokens, with **no change to sequence
    length** — plus ``reset_parameters()`` so meta-device builds can re-init
    after ``to_empty``. Register a new technique with
    ``@registry.register_time_embedding`` and select it via
    ``[time_embedding].type``; ``build_time_embedding`` dispatches through the
    registry.

    Out of scope (a separate, future integration point): sequence-*modifying*
    time encodings — e.g. Molmo2-style textual time-tokens interleaved between
    frame groups — change the token sequence (count / ``output_slice`` /
    ``modality_ids`` / MoT split) and need tokenizer + interleaved-sequence
    support KF does not have yet. Those would hook the sequence-assembly layer
    (``ModalityStrategy.prepare``), not this additive registry; set
    ``[time_embedding].type = "none"`` to run them instead of an additive one.
    """

    def forward(self, times: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def reset_parameters(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class FrameTimeEmbedding(TimeEmbedding):
    """Sinusoidal embedding of a per-frame timestamp (seconds) -> model dim.

    Args:
        dim: Model dimension (the embedding is added to the visual tokens).
        num_bands: Number of sinusoidal frequency bands; the raw feature width
            is ``2 * num_bands`` (sin + cos).
        min_period: Shortest period in seconds (highest frequency); sets the
            finest temporal resolution.
        max_period: Longest period in seconds (lowest frequency); sets the
            coarsest temporal scale the embedding can represent.
    """

    def __init__(
        self,
        dim: int,
        num_bands: int = 16,
        min_period: float = 0.5,
        max_period: float = 256.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"FrameTimeEmbedding dim must be positive (got {dim})")
        if num_bands <= 0:
            raise ValueError(f"FrameTimeEmbedding num_bands must be positive (got {num_bands})")
        if not 0.0 < min_period < max_period:
            raise ValueError(
                f"FrameTimeEmbedding requires 0 < min_period < max_period "
                f"(got min_period={min_period}, max_period={max_period})"
            )
        self.dim = dim
        self.num_bands = num_bands
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        self.proj = nn.Linear(2 * num_bands, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Zero-init so the temporal signal starts at zero and is learned. Also
        # re-init contract for the meta-device build (to_empty -> reset).
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Embed per-frame timestamps.

        Args:
            times: Per-frame timestamps in seconds, shape ``(batch, frames)``.

        Returns:
            ``(batch, frames, dim)`` temporal embeddings to add to each frame's
            visual tokens.
        """
        # Angular frequencies for log-spaced periods, on the input's device so
        # this is safe regardless of where the module was constructed (meta/CPU).
        periods = torch.logspace(
            math.log10(self.min_period),
            math.log10(self.max_period),
            self.num_bands,
            device=times.device,
            dtype=torch.float32,
        )
        ang = times.to(torch.float32).unsqueeze(-1) * (2.0 * math.pi / periods)  # (B, F, bands)
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, F, 2*bands)
        return self.proj(feats.to(self.proj.weight.dtype))


@registry.register_time_embedding("sinusoidal")
def _build_sinusoidal(
    dim: int,
    *,
    num_bands: int = 16,
    min_period: float = 0.5,
    max_period: float = 256.0,
    **_: Any,
) -> FrameTimeEmbedding:
    """Registry builder for the sinusoidal time embedding."""
    return FrameTimeEmbedding(
        dim, num_bands=num_bands, min_period=min_period, max_period=max_period
    )


def build_time_embedding(time_embedding_config: Any, dim: int) -> TimeEmbedding | None:
    """Build the per-frame time embedding from a ``TimeEmbeddingConfig``.

    Returns ``None`` when disabled (``type == "none"``). A ``None`` config falls
    back to the default (sinusoidal) so video callers that pass nothing keep the
    default behavior. The config is duck-typed (``.enabled`` / ``.type`` /
    ``.extra_kwargs()``) to avoid a model->config import cycle, matching
    ``build_adapter``.
    """
    if time_embedding_config is None:
        from kempnerforge.config.time_embedding import TimeEmbeddingConfig  # noqa: PLC0415

        time_embedding_config = TimeEmbeddingConfig()
    if not time_embedding_config.enabled:
        return None
    builder = registry.get_time_embedding(time_embedding_config.type)
    return builder(dim, **time_embedding_config.extra_kwargs())
