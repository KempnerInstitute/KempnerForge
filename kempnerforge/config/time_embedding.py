"""Time-embedding (per-frame timestamp) configuration.

``TimeEmbeddingConfig`` selects which per-frame timestamp embedding the VLM
video path uses and parameterizes it. Dispatched via the ``time_embedding``
registry at build time (see ``kempnerforge/model/frame_time.py``).

In TOML, ``[time_embedding]`` is a top-level section parallel to ``[adapter]``.
It is only consumed for video (``frames_per_clip > 1``); the image and text
paths never build one. ``type = "none"`` disables the embedding even for video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kempnerforge.config.registry import registry


@dataclass
class TimeEmbeddingConfig:
    """Selects the time-embedding type and parameterizes it.

    Register a new technique via ``@registry.register_time_embedding`` and select
    it with ``type``; ``type = "none"`` disables the embedding entirely.

    Fields:
        type: Registry key for the builder (``"sinusoidal"`` default, or ``"none"``).
        num_bands: Number of sinusoidal frequency bands (``"sinusoidal"`` only).
        min_period: Shortest period in seconds (finest temporal resolution).
        max_period: Longest period in seconds (coarsest temporal scale).
    """

    type: str = "sinusoidal"
    num_bands: int = 16
    min_period: float = 0.5
    max_period: float = 256.0

    def __post_init__(self) -> None:
        if self.type == "none":
            return
        # Late import: importing the module triggers the
        # ``@registry.register_time_embedding`` decorators. Doing it at module
        # scope would create a circular import via the config/model graph.
        import kempnerforge.model.frame_time  # noqa: F401, PLC0415

        registered = tuple(registry.list_time_embeddings())
        if self.type not in registered:
            raise ValueError(
                f"Unknown time_embedding.type: {self.type!r}. "
                f"Registered: {sorted(registered)} (or 'none' to disable)."
            )
        if self.num_bands <= 0:
            raise ValueError(f"time_embedding.num_bands must be positive (got {self.num_bands})")
        if not 0.0 < self.min_period < self.max_period:
            raise ValueError(
                f"time_embedding requires 0 < min_period < max_period "
                f"(got min_period={self.min_period}, max_period={self.max_period})"
            )

    @property
    def enabled(self) -> bool:
        """Whether a module should be built (``type != "none"``)."""
        return self.type != "none"

    def extra_kwargs(self) -> dict[str, Any]:
        """Builder kwargs beyond ``dim``. Type-specific builders take what they
        need and swallow the rest via ``**_`` (mirrors ``AdapterConfig``)."""
        return {
            "num_bands": self.num_bands,
            "min_period": self.min_period,
            "max_period": self.max_period,
        }
