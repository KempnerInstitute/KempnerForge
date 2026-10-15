"""Typed contract for the two-stage image-understanding recipe.

This module pins the recipe as configuration: a registered ``VLMConfig``
subclass (``arch = "image_understanding"``) carrying the stage settings,
pack geometry, tower/connector selection, precision policy, and
provenance state of the two pinned source training runs, plus a loader
for the recipe manifest that stores the stage-2 ordered source list and
the sha256 of each pinned source artifact. The config pins the manifest
by sha256 too, checked against the file's bytes when the config is built
and again when the manifest is loaded.

Reach it from TOML through the existing seams: ``plugins = ["recipe"]``
imports this module (registering the arch) before ``[vlm]`` is parsed,
and ``[vlm] arch = "image_understanding"`` dispatches to
``ImageUnderstandingConfig``.

Three contract rules are enforced here rather than documented:

- Nominal LM metadata is separate from packed residual budgets.
  ``nominal_text_context`` (the source LM's nominal text context) and
  ``lm_tie_embeddings_metadata`` (the published LM ties embeddings) are
  metadata; the packer budget lives in ``PackGeometry.pack_budget`` and
  the runtime trains untied (``lm_untie_at_runtime``). No validator
  couples the two sides.
- Token budgets are validated, not hard-coded. ``tokens_per_view`` must
  equal the count the selected connector yields for the declared geometry
  (a pooling connector reduces by its window, a projection connector keeps
  every patch), so the per-view cost follows the connector rather than a
  constant.
- Unresolved provenance blocks a production launch. Fields whose
  historical value is not established hold the ``OPEN`` sentinel; a
  config with ``production_launch = true`` refuses to construct while
  any field required for its stage is unresolved.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.registry import registry
from kempnerforge.config.vlm import DEFAULT_MODULE_PATTERNS, FreezeSpec, VLMConfig

ARCH_NAME = "image_understanding"

# Sentinel for a provenance value that is not established yet. A production
# launch refuses while any required field still holds it.
OPEN = "OPEN"

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Modules the freeze policy is defined over, and the ones each stage trains; every other
# recipe module is frozen in that stage.
_RECIPE_MODULES: tuple[str, ...] = ("transformer", "vision_encoder", "adapter")
_STAGE_TRAINABLE: dict[int, frozenset[str]] = {
    1: frozenset({"vision_encoder", "adapter"}),
    2: frozenset({"transformer", "vision_encoder", "adapter"}),
}

# OPEN-provenance fields that must be resolved before a production launch,
# per stage. Stage 2 additionally requires a non-empty ``init_checkpoint``.
_LAUNCH_REQUIRED_OPEN_FIELDS: dict[int, tuple[str, ...]] = {
    1: (
        "stage1_shard_listing",
        "stage1_shard_hashes",
        "stage1_loader_workers",
        "source_dependency_versions",
        "source_runtime_commit",
    ),
    2: (
        "source_dependency_versions",
        "source_runtime_commit",
        "endpoint_config_hash_resolution",
        "vision_mixed_precision_impl",
        "resume_reset_impl",
    ),
}
_OPEN_FIELD_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(_LAUNCH_REQUIRED_OPEN_FIELDS[1] + _LAUNCH_REQUIRED_OPEN_FIELDS[2])
)


def _is_unresolved(value: str) -> bool:
    return value == OPEN or not value.strip()


@dataclass(frozen=True)
class PackGeometry:
    """Per-stage visual geometry and the packer budget derived from it.

    ``pool_window`` is the connector's pooling window, 1 for a connector
    that keeps every patch. ``tokens_per_view`` is the declared packer
    cost of one view; ``ImageUnderstandingConfig`` refuses any
    disagreement with ``connector_tokens_per_view`` so the budget tracks
    the selected connector.
    """

    image_size: int
    patch_size: int
    pool_window: int
    tile_grid: int
    include_global_view: bool
    tokens_per_view: int
    pack_budget: int

    def __post_init__(self) -> None:
        for name in (
            "image_size",
            "patch_size",
            "pool_window",
            "tile_grid",
            "tokens_per_view",
            "pack_budget",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"geometry.{name} must be positive (got {getattr(self, name)})")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"geometry.image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        if self.patches_per_side % self.pool_window != 0:
            raise ValueError(
                f"geometry: patches per side ({self.patches_per_side}) must be divisible by "
                f"pool_window ({self.pool_window})"
            )
        if self.tile_grid == 1 and self.include_global_view:
            raise ValueError(
                "geometry.include_global_view requires tiling (tile_grid > 1); a single "
                "pass already is the global view"
            )
        if self.pack_budget <= self.tokens_per_image:
            raise ValueError(
                f"geometry.pack_budget ({self.pack_budget}) must exceed tokens_per_image "
                f"({self.tokens_per_image}); a pack must fit one maximal image plus text"
            )

    @property
    def patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    def connector_tokens_per_view(self, connector_type: str) -> int:
        """Tokens one view yields through ``connector_type``, as the runtime counts them."""
        adapter = AdapterConfig(type=connector_type, pool_window=self.pool_window)
        return adapter.output_num_tokens(self.patches_per_side**2)

    @property
    def views_per_image(self) -> int:
        return self.tile_grid**2 + (1 if self.include_global_view else 0)

    @property
    def tokens_per_image(self) -> int:
        return self.tokens_per_view * self.views_per_image


@dataclass(frozen=True)
class StageOptimGroup:
    """Peak LR and schedule shape for one parameter group."""

    lr: float
    warmup_steps: int
    schedule_steps: int
    min_lr: float
    start_factor: float

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"optim group lr must be positive (got {self.lr})")
        if self.warmup_steps < 0:
            raise ValueError(f"optim group warmup_steps must be >= 0 (got {self.warmup_steps})")
        if self.schedule_steps <= 0:
            raise ValueError(
                f"optim group schedule_steps must be positive (got {self.schedule_steps})"
            )
        if not 0 <= self.min_lr < self.lr:
            raise ValueError(
                f"optim group min_lr ({self.min_lr}) must satisfy 0 <= min_lr < lr ({self.lr})"
            )
        if not 0 < self.start_factor <= 1:
            raise ValueError(f"optim group start_factor ({self.start_factor}) must be in (0, 1]")


@dataclass(frozen=True)
class StageSettings:
    """The settings one stage of the recipe runs with.

    ``data_parallel_ranks`` records the data-parallel world the pinned
    run split its data over; the data-order contract depends on it, so
    it is recipe metadata, not a runtime parallelism knob.
    ``grad_clip_norm`` is 0.0 for no clipping (the recipe value).
    ``train_lm`` declares the stage's LM policy and must match the stage
    contract: stage 1 freezes the LM, stage 2 trains it.
    """

    stage: int
    max_steps: int
    epochs: int
    seed: int
    accumulation_steps: int
    data_parallel_ranks: int
    train_lm: bool
    answer_only_loss: bool
    image_attention: str
    bidirectional_scope: str
    vision_tower_precision: str
    vision_attn_implementation: str
    resume_reset_data_order: bool
    grad_clip_norm: float
    geometry: PackGeometry
    optim_vision: StageOptimGroup
    optim_language: StageOptimGroup
    optim_mixer: StageOptimGroup

    def __post_init__(self) -> None:
        if self.stage not in (1, 2):
            raise ValueError(f"stage.stage must be 1 or 2 (got {self.stage})")
        for name in ("max_steps", "epochs", "accumulation_steps", "data_parallel_ranks"):
            if getattr(self, name) <= 0:
                raise ValueError(f"stage.{name} must be positive (got {getattr(self, name)})")
        if self.seed < 0:
            raise ValueError(f"stage.seed must be >= 0 (got {self.seed})")
        if self.grad_clip_norm < 0:
            raise ValueError(
                f"stage.grad_clip_norm must be >= 0 (got {self.grad_clip_norm}); 0 disables"
            )
        if self.image_attention not in ("causal", "bidirectional"):
            raise ValueError(
                f"stage.image_attention must be 'causal' or 'bidirectional' "
                f"(got {self.image_attention!r})"
            )
        if self.image_attention == "causal" and self.bidirectional_scope:
            raise ValueError("stage.bidirectional_scope requires image_attention='bidirectional'")
        if self.image_attention == "bidirectional" and not self.bidirectional_scope:
            raise ValueError("stage.image_attention='bidirectional' requires a scope")
        if self.vision_tower_precision not in ("float32", "mixed_bfloat16"):
            raise ValueError(
                f"stage.vision_tower_precision must be 'float32' or 'mixed_bfloat16' "
                f"(got {self.vision_tower_precision!r})"
            )
        if self.vision_attn_implementation not in ("", "sdpa"):
            raise ValueError(
                f"stage.vision_attn_implementation must be '' or 'sdpa' "
                f"(got {self.vision_attn_implementation!r})"
            )
        lm_trained = "transformer" in _STAGE_TRAINABLE[self.stage]
        if self.train_lm != lm_trained:
            raise ValueError(
                f"stage {self.stage} {'trains' if lm_trained else 'freezes'} the LM; "
                f"train_lm must be {'true' if lm_trained else 'false'}"
            )
        if self.stage == 1:
            if self.geometry.tile_grid != 1 or self.geometry.include_global_view:
                raise ValueError("stage 1 is single-view; tiling is a stage-2 setting")
            if self.resume_reset_data_order:
                raise ValueError(
                    "resume_reset_data_order records a stage-2 exception; stage 1 must not set it"
                )
        for group_name in ("optim_vision", "optim_language", "optim_mixer"):
            group: StageOptimGroup = getattr(self, group_name)
            total = group.warmup_steps + group.schedule_steps
            if total != self.max_steps:
                raise ValueError(
                    f"stage.{group_name}: warmup_steps + schedule_steps ({total}) must equal "
                    f"stage.max_steps ({self.max_steps})"
                )


@registry.register_vlm_config(ARCH_NAME)
@dataclass
class ImageUnderstandingConfig(VLMConfig):
    """Contract for one stage of the two-stage image-understanding recipe.

    The residual-stream layout is joint-decoder (image tokens prepended
    to text), inherited from the base class. ``freeze`` defaults to
    empty; each stage contract declares its freeze policy explicitly and
    the validator checks the effective freeze state, at build time and
    after every ``freeze_schedule`` transition, against the stage
    contract: stage 1 freezes the LM and trains the tower and connector,
    stage 2 trains all three. ``stage.train_lm`` must agree with the
    contract but never decides the check.
    """

    arch: str = ARCH_NAME
    # Tower / connector / LM selection.
    lm_backbone_id: str = ""
    tokenizer_id: str = ""
    vision_tower_id: str = ""
    connector_type: str = ""
    connector_hidden_dim: int = 0
    # Rendering contract.
    use_chat_template: bool = True
    caption_prompt: str = ""
    vision_border: bool = False
    # Nominal LM metadata. These describe the published LM, not packer
    # budgets: the packed path never caps documents at
    # ``nominal_text_context``, and the runtime unties the head even
    # though the published LM ties it.
    nominal_text_context: int = 0
    lm_tie_embeddings_metadata: bool = True
    lm_untie_at_runtime: bool = True
    loss_reduction: str = "sample_mean"
    # Source-occurrence manifest reference (data file, pinned by hash).
    source_manifest: str = ""
    source_manifest_sha256: str = ""
    # The stage this job runs.
    stage: StageSettings | None = None
    # Launch gating.
    production_launch: bool = False
    init_checkpoint: str = ""
    # OPEN provenance: values of the pinned source runs that are not
    # established. Each holds ``OPEN`` until resolved to a real value.
    stage1_shard_listing: str = OPEN
    stage1_shard_hashes: str = OPEN
    stage1_loader_workers: str = OPEN
    source_dependency_versions: str = OPEN
    source_runtime_commit: str = OPEN
    endpoint_config_hash_resolution: str = OPEN
    vision_mixed_precision_impl: str = OPEN
    resume_reset_impl: str = OPEN
    freeze: list[FreezeSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("lm_backbone_id", "tokenizer_id", "vision_tower_id", "caption_prompt"):
            if not getattr(self, name).strip():
                raise ValueError(f"vlm.{name} must be set for the image-understanding recipe")
        self._validate_connector()
        if self.nominal_text_context <= 0:
            raise ValueError(
                f"vlm.nominal_text_context must be positive (got {self.nominal_text_context})"
            )
        if not self.loss_reduction.strip():
            raise ValueError("vlm.loss_reduction must be set")
        if not self.source_manifest.strip():
            raise ValueError("vlm.source_manifest must reference the recipe manifest file")
        if not _SHA256_RE.fullmatch(self.source_manifest_sha256):
            raise ValueError(
                "vlm.source_manifest_sha256 must be 64 lowercase hex characters "
                f"(got {self.source_manifest_sha256!r})"
            )
        _read_pinned_manifest(manifest_path(self), self.source_manifest_sha256)
        if self.stage is None:
            raise ValueError("the image-understanding contract requires a [vlm.stage] table")
        self._validate_token_budget()
        for name in _OPEN_FIELD_NAMES:
            if not getattr(self, name).strip():
                raise ValueError(
                    f"vlm.{name} must be a resolved value or the literal {OPEN!r}, not empty"
                )
        self._validate_freeze_policy()
        if self.stage.stage == 1 and self.init_checkpoint:
            raise ValueError(
                "stage 1 starts from the published tower and LM weights with a fresh "
                "connector; init_checkpoint must be empty"
            )
        if self.production_launch:
            unresolved = self.unresolved_launch_fields()
            if unresolved:
                raise ValueError(
                    "production_launch=true refused; unresolved provenance fields for "
                    f"stage {self.stage.stage}: {', '.join(unresolved)}. Resolve each "
                    f"(replace {OPEN!r} with the established value) or launch without "
                    "the production mark."
                )

    def _validate_connector(self) -> None:
        # Late import mirrors AdapterConfig: the decorators that populate the
        # adapter registry live in the model module.
        import kempnerforge.model.adapter  # noqa: F401, PLC0415

        registered = tuple(registry.list_adapters())
        if self.connector_type not in registered:
            raise ValueError(
                f"Unknown vlm.connector_type: {self.connector_type!r}. "
                f"Registered adapters: {sorted(registered)}."
            )
        if self.connector_hidden_dim < 0:
            raise ValueError(
                f"vlm.connector_hidden_dim must be >= 0 (got {self.connector_hidden_dim})"
            )

    def _validate_token_budget(self) -> None:
        assert self.stage is not None
        # Late import, as in _validate_connector: the model package pulls in torch.
        from kempnerforge.model.adapter import POOLING_ADAPTER_TYPES  # noqa: PLC0415

        geometry = self.stage.geometry
        if self.connector_type not in POOLING_ADAPTER_TYPES and geometry.pool_window != 1:
            raise ValueError(
                f"geometry.pool_window ({geometry.pool_window}) must be 1 for connector "
                f"{self.connector_type!r}, which keeps every patch; pooling connectors: "
                f"{sorted(POOLING_ADAPTER_TYPES)}"
            )
        derived = geometry.connector_tokens_per_view(self.connector_type)
        if geometry.tokens_per_view != derived:
            raise ValueError(
                f"geometry.tokens_per_view ({geometry.tokens_per_view}) must equal the "
                f"connector-derived count ({derived}) for connector {self.connector_type!r} "
                f"with image_size={geometry.image_size}, patch_size={geometry.patch_size}, "
                f"pool_window={geometry.pool_window}"
            )

    def _validate_freeze_policy(self) -> None:
        assert self.stage is not None
        for alias in _RECIPE_MODULES:
            if self.module_patterns.get(alias) != DEFAULT_MODULE_PATTERNS[alias]:
                raise ValueError(
                    f"vlm.module_patterns[{alias!r}] must be {DEFAULT_MODULE_PATTERNS[alias]} "
                    "so a freeze directive covers the whole module "
                    f"(got {self.module_patterns.get(alias)!r})"
                )
        # Late import, as in _validate_connector: the training package pulls in torch.
        from kempnerforge.training.freeze import effective_freeze  # noqa: PLC0415

        # The runtime keeps the last directive per module and applies scheduled stages
        # on top, so check the state the run starts in and the state after each stage.
        stage = self.stage.stage
        trainable = _STAGE_TRAINABLE[stage]
        known = set(_RECIPE_MODULES)
        states = [("build time", effective_freeze(0, self.freeze, [], known))]
        for transition in self.freeze_schedule:
            step = transition.start_step
            specs = effective_freeze(step, self.freeze, self.freeze_schedule, known)
            states.append((f"step {step}", specs))
        for where, specs in states:
            frozen = {spec.module for spec in specs if spec.frozen}
            for module in _RECIPE_MODULES:
                if module in trainable and module in frozen:
                    raise ValueError(
                        f"stage {stage} trains {module!r}; freeze must not include it "
                        f"(frozen at {where})"
                    )
                if module not in trainable and module not in frozen:
                    raise ValueError(
                        f"stage {stage} requires freeze to include "
                        f"{{ module = {module!r}, frozen = true }} (trainable at {where})"
                    )

    def unresolved_launch_fields(self) -> list[str]:
        """Names of fields a production launch of this stage still needs."""
        assert self.stage is not None
        unresolved = [
            name
            for name in _LAUNCH_REQUIRED_OPEN_FIELDS[self.stage.stage]
            if _is_unresolved(getattr(self, name))
        ]
        if self.stage.stage == 2 and not self.init_checkpoint.strip():
            unresolved.append("init_checkpoint")
        return unresolved


@dataclass(frozen=True)
class ManifestArtifact:
    """One pinned source artifact: recorded location and content hash."""

    label: str
    path: str
    sha256: str
    byte_size: int


@dataclass(frozen=True)
class Stage1Source:
    dataset_class: str
    name: str
    epoch_size: int


@dataclass(frozen=True)
class SourceManifest:
    """The recipe manifest: provenance block plus the ordered source list.

    ``stage2_sources`` is the authority for the stage-2 mixture: the
    ordered occurrence list, with every repeat contributing its full
    rows again. It is never deduplicated, sorted, or normalized.
    """

    version: int
    recipe: str
    artifacts: tuple[ManifestArtifact, ...]
    open_provenance: tuple[str, ...]
    notes: tuple[str, ...]
    stage1_source: Stage1Source
    stage2_sources: tuple[str, ...]

    def occurrence_count(self) -> int:
        return len(self.stage2_sources)

    def unique_count(self) -> int:
        return len(set(self.stage2_sources))

    def multiplicity(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in self.stage2_sources:
            counts[name] = counts.get(name, 0) + 1
        return counts


def _require_keys(block: dict, allowed: set[str], where: str) -> None:
    unknown = set(block) - allowed
    if unknown:
        raise ValueError(f"manifest {where}: unknown keys {sorted(unknown)}")
    missing = allowed - set(block)
    if missing:
        raise ValueError(f"manifest {where}: missing keys {sorted(missing)}")


def _require_str_list(value: object, where: str) -> list[str]:
    """A JSON array of non-empty strings; iterating a string or object also yields strings."""
    if not isinstance(value, list):
        raise ValueError(f"manifest {where} must be a list, got {type(value).__name__}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"manifest {where} entries must be non-empty strings")
    return value


def _read_pinned_manifest(path: Path, expected_sha256: str) -> bytes:
    """The manifest's bytes, refused unless they hash to the pin."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"vlm.source_manifest: cannot read {path}: {exc}") from exc
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"manifest content hash mismatch for {path}: vlm.source_manifest_sha256 is "
            f"{expected_sha256}, the file hashes to {digest}"
        )
    return data


def manifest_path(config: ImageUnderstandingConfig, base_dir: Path | None = None) -> Path:
    """Resolve the manifest reference; relative paths anchor at ``configs/``."""
    path = Path(config.source_manifest)
    if not path.is_absolute():
        path = (base_dir or CONFIGS_DIR) / path
    return path


def load_source_manifest(
    config: ImageUnderstandingConfig, base_dir: Path | None = None
) -> SourceManifest:
    """Load and verify the manifest the config references.

    Refuses rather than degrades: a content-hash mismatch, a malformed
    block, or a source list that disagrees with its recorded counts each
    raise ``ValueError`` so a truncated or edited manifest can never
    feed a run.
    """
    path = manifest_path(config, base_dir)
    raw = json.loads(_read_pinned_manifest(path, config.source_manifest_sha256))
    _require_keys(
        raw,
        {
            "manifest_version",
            "recipe",
            "provenance",
            "stage1_source",
            "stage2_sources",
            "source_counts",
        },
        "top level",
    )
    _require_keys(raw["provenance"], {"artifacts", "open", "notes"}, "provenance")
    artifacts = []
    for entry in raw["provenance"]["artifacts"]:
        _require_keys(entry, {"label", "path", "sha256", "byte_size"}, "provenance artifact")
        if not _SHA256_RE.fullmatch(entry["sha256"]):
            raise ValueError(f"manifest artifact {entry['label']!r}: malformed sha256")
        if entry["byte_size"] <= 0:
            raise ValueError(f"manifest artifact {entry['label']!r}: byte_size must be positive")
        artifacts.append(ManifestArtifact(**entry))
    open_items = _require_str_list(raw["provenance"]["open"], "provenance.open")
    notes = _require_str_list(raw["provenance"]["notes"], "provenance.notes")
    _require_keys(raw["stage1_source"], {"dataset_class", "name", "epoch_size"}, "stage1_source")
    sources = _require_str_list(raw["stage2_sources"], "stage2_sources")
    if not sources:
        raise ValueError("manifest stage2_sources must be a non-empty list of non-empty names")
    _require_keys(raw["source_counts"], {"occurrences", "unique"}, "source_counts")
    if raw["source_counts"]["occurrences"] != len(sources):
        raise ValueError(
            f"manifest source_counts.occurrences ({raw['source_counts']['occurrences']}) "
            f"disagrees with the literal list ({len(sources)}); refusing a truncated or "
            "padded manifest"
        )
    if raw["source_counts"]["unique"] != len(set(sources)):
        raise ValueError(
            f"manifest source_counts.unique ({raw['source_counts']['unique']}) disagrees "
            f"with the literal list ({len(set(sources))})"
        )
    return SourceManifest(
        version=raw["manifest_version"],
        recipe=raw["recipe"],
        artifacts=tuple(artifacts),
        open_provenance=tuple(open_items),
        notes=tuple(notes),
        stage1_source=Stage1Source(**raw["stage1_source"]),
        stage2_sources=tuple(sources),
    )
