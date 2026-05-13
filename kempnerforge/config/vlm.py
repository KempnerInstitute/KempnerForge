"""VLM (vision-language model) configuration.

``VLMConfig`` carries the arch-level knobs of the vision-language
model: which architecture to wire (``arch``), the fixed text padding
length, and the freeze policy. The vision encoder and adapter are
described by sibling top-level sections (``VisionEncoderConfig`` in
``config/vision.py``, ``AdapterConfig`` in ``config/adapter.py``).

In TOML, ``[vlm]`` is a top-level section, parallel to ``[model]``,
``[vision_encoder]``, and ``[adapter]``. When ``[vlm]`` is absent the
job is a pure text run.

Architecture is a discriminated union on the ``arch`` field:

- ``"joint_decoder"`` image tokens prepended to the text sequence.
- ``"cross_attention"`` image K/V flows in via separate
  cross-attention blocks at a configurable cadence.
- ``"mot"`` Mixture-of-Transformers: per-modality Q/K/V/O + per-
  modality FFN at every layer, single global self-attention.

Each arch gets its own ``VLMConfig`` subclass, registered via
``registry.register_vlm_config``. The TOML loader dispatches on
``arch`` to instantiate the right subclass; programmatic callers use
``VLMConfig.for_arch(arch_name, **fields)``.

``FreezeSpec`` / ``FreezeStage`` are consumed by
``kempnerforge/training/freeze.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kempnerforge.config.registry import registry

# Default aliases: module-name keys expand to fnmatch patterns. Each alias
# matches both the bare module attribute (a hypothetical param directly on
# ``adapter``/``vision_encoder`` etc.) and all nested children, so freezing
# by alias cannot silently miss a parameter.
DEFAULT_MODULE_PATTERNS: dict[str, list[str]] = {
    "transformer": ["transformer", "transformer.*"],
    "vision_encoder": ["vision_encoder", "vision_encoder.*"],
    "adapter": ["adapter", "adapter.*"],
}


@dataclass(frozen=True)
class FreezeSpec:
    """A single freeze directive.

    ``module`` is an alias (key in a pattern map such as
    ``DEFAULT_MODULE_PATTERNS``) or a raw fnmatch pattern matching
    fully-qualified parameter names.
    """

    module: str
    frozen: bool = True


@dataclass(frozen=True)
class FreezeStage:
    """A freeze directive that applies from ``start_step`` onward.

    Used for staged training recipes where the trainable subset
    changes across training phases. The list of stages on
    ``VLMConfig`` is expected to be in strictly monotonic
    ``start_step`` order.
    """

    start_step: int
    specs: tuple[FreezeSpec, ...]


# Reserved arches: known names not yet implemented. Loader/for_arch raise
# ``NotImplementedError`` rather than ``ValueError`` so TOMLs that aim at
# a future arch get a clear message about it.
_RESERVED_ARCHS: tuple[str, ...] = ()


@dataclass
class VLMConfig:
    """Base VLM configuration.

    Subclasses register themselves via ``@registry.register_vlm_config``
    and override the ``arch`` field's default. Use
    ``VLMConfig.for_arch(arch_name, **fields)`` to construct
    programmatically; the TOML loader dispatches on ``arch``
    automatically.

    Field summary (full per-field docs are picked up from autodoc):

    - ``arch`` VLM architecture discriminator. Subclasses set this via
      field default; direct construction with an arch name not backed by
      a registered subclass raises.
    - ``max_text_len`` fixed text padding length used by ``VLMCollator``.
      Enforces rank-consistent batches under FSDP2.
    - ``freeze`` static freeze specs applied once at build time.
    - ``freeze_schedule`` step-boundary freeze transitions.
    - ``module_patterns`` map of module alias (``"transformer"``,
      ``"vision_encoder"``, ``"adapter"``, plus arch-specific additions)
      to fnmatch pattern list.
    """

    arch: str = "joint_decoder"
    max_text_len: int = 512
    freeze: list[FreezeSpec] = field(default_factory=lambda: [FreezeSpec("vision_encoder", True)])
    freeze_schedule: list[FreezeStage] = field(default_factory=list)
    module_patterns: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_MODULE_PATTERNS.items()}
    )

    def __post_init__(self) -> None:
        if self.arch in _RESERVED_ARCHS:
            raise NotImplementedError(
                f"vlm.arch={self.arch!r} is reserved; not yet implemented. "
                f"Reserved: {sorted(_RESERVED_ARCHS)}."
            )
        registered = tuple(registry.list_vlm_configs())
        if self.arch not in registered:
            raise ValueError(
                f"Unknown vlm.arch: {self.arch!r}. "
                f"Registered: {sorted(registered)}. "
                f"Reserved (not yet implemented): {sorted(_RESERVED_ARCHS)}."
            )
        if self.max_text_len <= 0:
            raise ValueError("vlm.max_text_len must be positive")
        if self.freeze_schedule:
            steps = [s.start_step for s in self.freeze_schedule]
            if steps != sorted(steps) or len(steps) != len(set(steps)):
                raise ValueError("vlm.freeze_schedule start_steps must be strictly monotonic")

    def residual_stream_image_tokens(self, num_tokens: int) -> int:
        """Number of image tokens this arch places in the residual stream.

        Used by ``JobConfig`` to validate that ``model.max_seq_len`` and
        ``train.seq_len`` are large enough to fit
        ``residual_stream_image_tokens + max_text_len`` along the
        attention sequence dimension.

        - Joint-Decoder / MoT: ``num_tokens`` (image tokens prepended to
          text).
        - Cross-Attention: ``0`` (residual stream is text-only; image
          features flow side-channel into CA blocks).

        Args:
            num_tokens: The vision encoder's resolved ``num_tokens``.
                Pass ``0`` when it is not known yet (the "infer at build
                time" sentinel); cross-checks that depend on a concrete
                value will skip and re-run at build time.
        """
        return num_tokens

    @classmethod
    def for_arch(cls, arch: str, **kwargs: Any) -> VLMConfig:
        """Resolve ``arch`` to a registered subclass and instantiate.

        Raises:
            ValueError: ``arch`` is not registered.
            NotImplementedError: ``arch`` is reserved (in
                ``_RESERVED_ARCHS``) matches loader semantics so the
                error type is independent of construction site.

        Example:
            >>> cfg = VLMConfig.for_arch(
            ...     "cross_attention",
            ...     max_text_len=2048,
            ...     cross_attention_every_n_layers=4,
            ... )
        """
        if arch in _RESERVED_ARCHS:
            raise NotImplementedError(
                f"vlm.arch={arch!r} is reserved; not yet implemented. "
                f"Reserved: {sorted(_RESERVED_ARCHS)}."
            )
        try:
            sub = registry.get_vlm_config(arch)
        except KeyError as e:
            raise ValueError(
                f"Unknown vlm.arch: {arch!r}. "
                f"Registered: {sorted(registry.list_vlm_configs())}. "
                f"Reserved (not yet implemented): {sorted(_RESERVED_ARCHS)}."
            ) from e
        return sub(**kwargs)


@registry.register_vlm_config("joint_decoder")
@dataclass
class JointDecoderConfig(VLMConfig):
    """Joint-Decoder: image tokens prepended to the text sequence.

    No additional fields beyond ``VLMConfig``. The arch is wired
    through ``VLMWrapper`` + ``ModalityContext.prefix_embeds`` +
    ``output_slice``.
    """

    arch: str = "joint_decoder"


@registry.register_vlm_config("cross_attention")
@dataclass
class CrossAttentionConfig(VLMConfig):
    """Cross-Attention: image K/V flows into separate cross-attention
    blocks inserted at a configurable cadence.

    The CA-specific module alias ``"cross_attention"`` is added to
    ``module_patterns`` so freeze targeting works out of the box.
    """

    arch: str = "cross_attention"
    cross_attention_every_n_layers: int = 4
    cross_attention_n_heads: int = 0
    cross_attention_n_kv_heads: int = 0
    module_patterns: dict[str, list[str]] = field(
        default_factory=lambda: {
            **{k: list(v) for k, v in DEFAULT_MODULE_PATTERNS.items()},
            "cross_attention": [
                "transformer.cross_attention_layers",
                "transformer.cross_attention_layers.*",
            ],
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cross_attention_every_n_layers <= 0:
            raise ValueError(
                "vlm.cross_attention_every_n_layers must be positive "
                f"(got {self.cross_attention_every_n_layers})"
            )
        if self.cross_attention_n_heads < 0 or self.cross_attention_n_kv_heads < 0:
            raise ValueError(
                "vlm.cross_attention_n_heads and cross_attention_n_kv_heads must be non-negative"
            )

    def residual_stream_image_tokens(self, num_tokens: int) -> int:  # noqa: ARG002
        """Cross-Attention does not extend the residual stream.

        Image features flow as K/V into separate CrossAttentionBlocks;
        the residual itself carries text only. So the seq_len cross-check
        skips ``num_tokens`` and just enforces ``seq_len >= max_text_len``.
        The ``num_tokens`` argument is accepted for signature parity with
        the base method but ignored.
        """
        return 0

    def resolved_heads(self, model_n_heads: int) -> tuple[int, int]:
        """Resolve zero-defaults against the text backbone's head count.

        Returns ``(n_heads, n_kv_heads)`` such that the
        ``CrossAttentionBlock`` constructor never observes 0.

        Resolution rule:

        - ``n_heads = self.cross_attention_n_heads or model_n_heads``
        - ``n_kv_heads = self.cross_attention_n_kv_heads or n_heads``
        """
        if model_n_heads <= 0:
            raise ValueError(f"model_n_heads must be positive (got {model_n_heads})")
        n_heads = self.cross_attention_n_heads or model_n_heads
        n_kv_heads = self.cross_attention_n_kv_heads or n_heads
        return n_heads, n_kv_heads


@registry.register_vlm_config("mot")
@dataclass
class MoTConfig(VLMConfig):
    """Mixture-of-Transformers: per-modality Q/K/V/O projections + per-
    modality FFN at every layer; single global self-attention mixes all
    modality streams (Liang et al. 2024, Algorithm 1).

    Image tokens are prepended to the text sequence in the residual
    stream (image-then-text concat order). ``modality_ids`` tags every
    position with its source modality; the operator routes per-token
    through the per-modality projection / FFN copy for that position.

    The MoT-specific module alias ``"mot"`` is added to
    ``module_patterns`` so freeze targeting works out of the box:
    ``FreezeSpec("mot", True)`` freezes the per-modality main stack
    (``transformer.layers.*``) without touching the embedding /
    output head / final norms.
    """

    arch: str = "mot"
    mot_modalities: tuple[str, ...] = ("image", "text")
    mot_image_n_heads: int = 0
    mot_image_n_kv_heads: int = 0
    mot_warm_start_from_text: bool = False
    mot_warm_start_path: str = ""
    module_patterns: dict[str, list[str]] = field(
        default_factory=lambda: {
            **{k: list(v) for k, v in DEFAULT_MODULE_PATTERNS.items()},
            "mot": [
                "transformer.layers",
                "transformer.layers.*",
            ],
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.mot_modalities) < 2:
            raise ValueError(
                f"vlm.mot_modalities must have at least 2 entries (got {self.mot_modalities!r})"
            )
        if "text" not in self.mot_modalities:
            raise ValueError(
                f"vlm.mot_modalities must include 'text' (got {self.mot_modalities!r})"
            )
        if "image" not in self.mot_modalities:
            raise ValueError(
                f"vlm.mot_modalities must include 'image' (got {self.mot_modalities!r})"
            )
        if len(set(self.mot_modalities)) != len(self.mot_modalities):
            raise ValueError(
                f"vlm.mot_modalities must not contain duplicates (got {self.mot_modalities!r})"
            )
        if self.mot_image_n_heads < 0 or self.mot_image_n_kv_heads < 0:
            raise ValueError("vlm.mot_image_n_heads and mot_image_n_kv_heads must be non-negative")
        if self.mot_warm_start_from_text and not self.mot_warm_start_path:
            raise ValueError(
                "vlm.mot_warm_start_from_text=True requires vlm.mot_warm_start_path to be a "
                "non-empty filesystem path to a torch-saved JD or text-only state dict"
            )

    def residual_stream_image_tokens(self, num_tokens: int) -> int:
        """MoT prepends ``num_tokens`` image tokens to the text sequence
        (same residual-stream layout as Joint-Decoder).
        """
        return num_tokens

    def resolved_image_heads(
        self, model_n_heads: int, model_n_kv_heads: int = 0
    ) -> tuple[int, int]:
        """Resolve zero-defaults against the text backbone's head counts.

        Returns ``(n_heads, n_kv_heads)`` such that the operator's
        per-modality projection sizes are never built from 0.

        Resolution rule:

        - ``n_heads = self.mot_image_n_heads or model_n_heads``
        - ``n_kv_heads = self.mot_image_n_kv_heads or model_n_kv_heads or n_heads``

        v1 note: the global-SDPA design requires equal head counts
        across modalities; ``Transformer.__init__`` asserts the
        resolved tuple matches the text backbone (raise on per-modality
        override). Field is present so a future per-modality relaxation
        can land without a config-shape change.
        """
        if model_n_heads <= 0:
            raise ValueError(f"model_n_heads must be positive (got {model_n_heads})")
        n_heads = self.mot_image_n_heads or model_n_heads
        n_kv_heads = self.mot_image_n_kv_heads or model_n_kv_heads or n_heads
        return n_heads, n_kv_heads
