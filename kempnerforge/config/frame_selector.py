"""Query-aware frame-selection configuration.

``FrameSelectorConfig`` selects which frame selector the video path uses and
parameterizes it. Dispatched via the ``frame_selector`` registry at build time
(see ``kempnerforge/data/frame_selection.py``).

In TOML, ``[frame_selector]`` is a top-level section parallel to ``[video]`` /
``[adapter]``. When present (and ``[video]`` is set) a clip is decoded to a pool
of ``candidate_frames`` and the selector keeps ``[video].max_frames`` of them for
the sample's query; when absent, the video path decodes ``max_frames`` uniformly
as before (bit-identical). It requires ``[video]`` — a selector with no video has
nothing to select from.

The candidate pool is sized solely by ``candidate_frames`` / ``candidate_fps``.
``[video].fps`` and ``[video].min_frames`` describe the plain (non-selector)
decode and are inert on the selector path — only ``[video].max_frames`` (the
count kept) still applies. ``JobConfig`` warns if ``fps``/``min_frames`` are set
to non-defaults alongside a selector, so a tuned-but-ignored knob is never silent.

The default scorer is SigLIP2-so400m-patch14-224. The mDP3 paper used
SigLIP(v1)-so400m @ 384; SigLIP2 @ 224 is chosen for stack consistency with the
VLM vision towers and cheaper scoring. The paper-exact tower is one
``scorer_path`` away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kempnerforge.config.registry import registry

_SCORERS = ("siglip2", "clip")
_KERNELS = ("rkhs", "cosine")


@dataclass
class FrameSelectorConfig:
    """Selects the frame-selector type and parameterizes it.

    Register a new selector via ``@registry.register_frame_selector`` and select
    it with ``type``. Absence of the ``[frame_selector]`` section (rather than any
    ``"none"`` value) is the off switch.

    Fields:
        type: Registry key for the selector builder (``"topk"`` / ``"qframe"`` /
            ``"mdp3"``).
        scorer: Dual-encoder family for scoring (``"siglip2"`` / ``"clip"``).
        scorer_path: HF hub id or local path for the scorer weights.
        candidate_frames: Size of the candidate pool decoded before selection.
            Must be >= ``[video].max_frames`` (checked in ``JobConfig``).
        candidate_fps: Candidate sampling rate. ``0.0`` (default) samples exactly
            ``candidate_frames`` uniformly over the clip; ``> 0`` samples at that
            rate, capped at ``candidate_frames``.
        gumbel_tau: Softmax temperature for ``"qframe"`` Gumbel sampling.
        seed: Base seed for ``"qframe"`` (mixed per-sample with the sample key).
        mdp3_lambda: Relevance/diversity trade-off for ``"mdp3"`` (paper ``lambda``).
        mdp3_segment_size: Temporal segment length for ``"mdp3"`` sequentiality;
            ``0`` = single-segment plain conditional DPP (no sequentiality).
        mdp3_kernel: Similarity kernel for ``"mdp3"`` (``"rkhs"`` multi-Gaussian, or
            ``"cosine"`` ablation).
    """

    type: str = "topk"
    scorer: str = "siglip2"
    scorer_path: str = "google/siglip2-so400m-patch14-224"
    candidate_frames: int = 128
    candidate_fps: float = 0.0
    gumbel_tau: float = 0.8
    seed: int = 0
    mdp3_lambda: float = 0.2
    mdp3_segment_size: int = 32
    mdp3_kernel: str = "rkhs"

    def __post_init__(self) -> None:
        # Validate scalar fields unconditionally (fail-loud at construction) so a
        # bad value can't lie dormant and only surface under a later type flip.
        if self.scorer not in _SCORERS:
            raise ValueError(
                f"frame_selector.scorer must be one of {_SCORERS} (got {self.scorer!r})"
            )
        if not self.scorer_path:
            raise ValueError("frame_selector.scorer_path must be non-empty")
        if self.candidate_frames < 1:
            raise ValueError(
                f"frame_selector.candidate_frames must be >= 1 (got {self.candidate_frames})"
            )
        if self.candidate_fps < 0.0:
            raise ValueError(
                f"frame_selector.candidate_fps must be non-negative (got {self.candidate_fps})"
            )
        if self.gumbel_tau <= 0.0:
            raise ValueError(f"frame_selector.gumbel_tau must be positive (got {self.gumbel_tau})")
        if self.mdp3_lambda <= 0.0:
            raise ValueError(
                f"frame_selector.mdp3_lambda must be positive (got {self.mdp3_lambda})"
            )
        if self.mdp3_segment_size < 0:
            raise ValueError(
                f"frame_selector.mdp3_segment_size must be non-negative "
                f"(got {self.mdp3_segment_size})"
            )
        if self.mdp3_kernel not in _KERNELS:
            raise ValueError(
                f"frame_selector.mdp3_kernel must be one of {_KERNELS} (got {self.mdp3_kernel!r})"
            )
        # Late import: importing the module triggers the
        # ``@registry.register_frame_selector`` decorators. Doing it at module
        # scope would create a circular import via the config/data graph.
        import kempnerforge.data.frame_selection  # noqa: F401, PLC0415

        registered = tuple(registry.list_frame_selectors())
        if self.type not in registered:
            raise ValueError(
                f"Unknown frame_selector.type: {self.type!r}. Registered: {sorted(registered)}."
            )

    def extra_kwargs(self) -> dict[str, Any]:
        """Builder kwargs beyond ``scorer``. ``candidate_frames`` / ``candidate_fps``
        configure the base selector; per-algorithm knobs are consumed by their own
        selector; foreign keys are swallowed via ``**_`` (mirrors ``AdapterConfig``).
        ``scorer`` / ``scorer_path`` build the scorer, so neither appears here.
        """
        return {
            "candidate_frames": self.candidate_frames,
            "candidate_fps": self.candidate_fps,
            "gumbel_tau": self.gumbel_tau,
            "seed": self.seed,
            "mdp3_lambda": self.mdp3_lambda,
            "mdp3_segment_size": self.mdp3_segment_size,
            "mdp3_kernel": self.mdp3_kernel,
        }
