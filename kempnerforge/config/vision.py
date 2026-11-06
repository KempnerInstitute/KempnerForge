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
- ``mixed_precision`` is the dtype the encoder computes in (``"bf16"`` or
  ``"fp32"``), independent of ``train.mixed_precision``, for image and video
  runs alike, also under the ``torch.autocast`` that
  ``train.fp32_master_weights`` runs without an FSDP2 wrap. Its parameters,
  gradients and optimizer state keep the dtype the encoder was loaded in, on
  any number of GPUs. Its floating-point buffers are cast to the set dtype,
  so a checkpoint holds them in it. Unset, the encoder keeps the default:
  an FSDP2-wrapped encoder computes in the language model's dtype, and an
  unwrapped one in its loaded dtype, with that autocast reaching its
  matmuls where it runs. It applies to the training build
  (``build_parallel_model``); ``build_vlm_wrapper`` leaves every dtype to
  its caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from kempnerforge.config.registry import registry

if TYPE_CHECKING:
    import torch

_MIXED_PRECISIONS = ("bf16", "fp32")


@dataclass
class VisionEncoderConfig:
    """Configuration for the vision encoder component of a VLM."""

    type: str = "random"
    path: str = ""
    feature_dim: int = 0
    num_tokens: int = 0
    mixed_precision: Literal["bf16", "fp32"] | None = None

    def __post_init__(self) -> None:
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
        if self.mixed_precision is not None and self.mixed_precision not in _MIXED_PRECISIONS:
            raise ValueError(
                f"vision_encoder.mixed_precision must be one of {list(_MIXED_PRECISIONS)} "
                f"or unset (got {self.mixed_precision!r})"
            )

    @property
    def param_dtype(self) -> torch.dtype | None:
        """The dtype the encoder computes in, or ``None`` when ``mixed_precision`` is unset."""
        if self.mixed_precision is None:
            return None
        import torch

        return {"bf16": torch.bfloat16, "fp32": torch.float32}[self.mixed_precision]
