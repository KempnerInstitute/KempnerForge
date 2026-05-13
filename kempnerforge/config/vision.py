"""Vision-encoder configuration.

``VisionEncoderConfig`` selects and parameterizes the vision encoder
that the ``VLMWrapper`` composes alongside the text backbone and
adapter. It is a top-level section in TOML (``[vision_encoder]``),
sibling to ``[model]``, ``[adapter]``, and ``[vlm]``.

Field summary:

- ``type`` selects the encoder by registry key
  (see ``registry.register_vision_encoder``). Defaults to ``"random"``
  for tests; production configs set ``"siglip2"`` / ``"clip"`` etc.
- ``path`` is the HF Hub id or local path passed to the encoder
  builder. Empty string is accepted for stub encoders (``"random"``).
- ``feature_dim`` is the output feature dim of the encoder. ``0`` means
  "infer from the encoder at build time".
- ``num_tokens`` is the number of image tokens the encoder produces per
  image. ``0`` means "infer at build time". When ``> 0`` it is cross-
  checked against ``model.max_seq_len`` at config time inside
  ``JobConfig.__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass

from kempnerforge.config.registry import registry


@dataclass
class VisionEncoderConfig:
    """Configuration for the vision encoder component of a VLM."""

    type: str = "random"
    path: str = ""
    feature_dim: int = 0
    num_tokens: int = 0

    def __post_init__(self) -> None:
        # Late import: importing the encoder module triggers the
        # ``@registry.register_vision_encoder`` decorators that populate the
        # registry. Without this, ``list_vision_encoders()`` would return an
        # empty list when this dataclass is constructed before any encoder
        # module has been imported (e.g. in unit-test isolation).
        import kempnerforge.model.vision  # noqa: F401, PLC0415

        registered = tuple(registry.list_vision_encoders())
        if self.type not in registered:
            raise ValueError(
                f"Unknown vision_encoder.type: {self.type!r}. Registered: {sorted(registered)}."
            )
        if self.feature_dim < 0:
            raise ValueError(
                f"vision_encoder.feature_dim must be non-negative (got {self.feature_dim})"
            )
        if self.num_tokens < 0:
            raise ValueError(
                f"vision_encoder.num_tokens must be non-negative (got {self.num_tokens})"
            )
