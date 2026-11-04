"""Top-level job configuration aggregating all sub-configs."""

from __future__ import annotations

from dataclasses import dataclass, field

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.checkpoint import CheckpointConfig
from kempnerforge.config.data import DataConfig
from kempnerforge.config.distributed import DistributedConfig
from kempnerforge.config.eval import EvalConfig
from kempnerforge.config.metrics import MetricsConfig
from kempnerforge.config.model import FLEX_BLOCK_SIZE, ModelConfig
from kempnerforge.config.optimizer import OptimizerConfig
from kempnerforge.config.profiling import ProfilingConfig
from kempnerforge.config.scheduler import SchedulerConfig
from kempnerforge.config.training import TrainConfig
from kempnerforge.config.video import VideoConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import MoMaConfig, VLMConfig

_HF_VISION_ENCODER_TYPES = frozenset({"siglip2", "clip"})


@dataclass
class JobConfig:
    """Top-level configuration aggregating all sub-configs.

    The VLM stack lives as three sibling top-level sections,
    ``[vision_encoder]`` / ``[adapter]`` / ``[vlm]``. A pure text run
    leaves all three at ``None`` and writes only ``[model]`` (plus the
    other operational sections). A VLM run sets ``[vlm]`` and is
    required to provide ``[vision_encoder]``; ``[adapter]`` defaults to
    the registered 2-layer MLP and may be omitted.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    vision_encoder: VisionEncoderConfig | None = None
    adapter: AdapterConfig | None = None
    vlm: VLMConfig | None = None
    video: VideoConfig | None = None
    plugins: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Cross-section invariants that fire at construction time.

        ``validate(world_size)`` covers checks that depend on the
        distributed world size; everything that is decidable from the
        config alone runs here so loader-time mistakes surface before
        any distributed init.
        """
        if not isinstance(self.plugins, list) or not all(isinstance(p, str) for p in self.plugins):
            raise ValueError(f"plugins must be a list of module paths (got {self.plugins!r})")

        if self.vlm is not None:
            if self.vision_encoder is None:
                raise ValueError(
                    "[vlm] is set but [vision_encoder] is missing; a VLM run requires a "
                    "vision encoder section."
                )
            if self.adapter is None:
                self.adapter = AdapterConfig()
            if self.vision_encoder.type in _HF_VISION_ENCODER_TYPES and (
                self.vision_encoder.feature_dim > 0 or self.vision_encoder.num_tokens > 0
            ):
                import logging

                overrides = []
                if self.vision_encoder.feature_dim > 0:
                    overrides.append(f"feature_dim={self.vision_encoder.feature_dim}")
                if self.vision_encoder.num_tokens > 0:
                    overrides.append(f"num_tokens={self.vision_encoder.num_tokens}")
                logging.getLogger(__name__).warning(
                    "vision_encoder.type=%r is HF-backed and probes its own dims; "
                    "setting %s overrides the probe and may desync from the loaded "
                    "model. Leave both at 0 to use the probed values.",
                    self.vision_encoder.type,
                    " and ".join(overrides),
                )
            if self.vision_encoder.num_tokens > 0:
                assert self.adapter is not None
                per_frame = self.adapter.output_num_tokens(self.vision_encoder.num_tokens)
                frames = self.views_per_sample
                visual_tokens = frames * per_frame
                residual_image_tokens = self.vlm.residual_stream_image_tokens(visual_tokens)
                required = residual_image_tokens + self.vlm.max_text_len
                if self.model.max_seq_len < required:
                    raise ValueError(
                        f"model.max_seq_len ({self.model.max_seq_len}) insufficient for VLM: "
                        f"residual_image_tokens ({residual_image_tokens}) + "
                        f"vlm.max_text_len ({self.vlm.max_text_len}) = {required}"
                    )

        if self.video is not None and self.vlm is None:
            raise ValueError(
                "[video] is set but [vlm] is missing; video training runs through "
                "the VLM wrapper, so a [vlm] section (and [vision_encoder]) is required."
            )
        if self.video is not None and self.vlm is not None and self.vlm.tile_grid:
            raise ValueError(
                "vlm.tile_grid tiles the image path, but [video] is set; a video run "
                "samples frames instead, so leave vlm.tile_grid at 0"
            )

        if self.model.attention_backend == "flex" and self.train.seq_len < FLEX_BLOCK_SIZE:
            raise ValueError(
                f"attention_backend='flex' requires train.seq_len >= {FLEX_BLOCK_SIZE}, "
                f"got {self.train.seq_len}. Use attention_backend='sdpa' for shorter sequences."
            )

    @property
    def is_vlm(self) -> bool:
        """Whether this job builds a ``VLMWrapper`` around the text backbone."""
        return self.vlm is not None

    @property
    def is_video(self) -> bool:
        """Whether this job trains on a video dataset (a VLM sub-mode)."""
        return self.video is not None

    @property
    def views_per_sample(self) -> int:
        """Images the encoder sees per sample: a clip's frames, a tiled image's views, or one."""
        if self.video is not None:
            return self.video.max_frames
        if self.vlm is not None:
            return self.vlm.views_per_image
        return 1

    def validate(self, world_size: int = 1) -> None:
        """Run cross-config validations that depend on the world size."""
        self.distributed.validate_world_size(world_size)

        if self.train.seq_len > self.model.max_seq_len:
            raise ValueError(
                f"train.seq_len ({self.train.seq_len}) exceeds "
                f"model.max_seq_len ({self.model.max_seq_len})"
            )

        if self.model.tie_embeddings and self.distributed.pp > 1:
            raise ValueError(
                "Tied embeddings are not supported with pipeline parallelism "
                "(embedding and output head must be on different stages)"
            )

        if self.eval.enabled and not self.eval.dataset_path and not self.eval.hf_dataset_name:
            raise ValueError(
                "eval.enabled is True but no eval data source is configured. "
                "Set eval.dataset_path or eval.hf_dataset_name."
            )

        if self.model.is_moe and self.train.compile_model:
            import logging

            logging.getLogger(__name__).warning(
                "torch.compile is not yet optimized for MoE dispatch (data-dependent shapes "
                "cause graph breaks). Set compile_model=false for MoE models."
            )

        if self.distributed.tp > 1:
            if self.model.n_heads % self.distributed.tp != 0:
                raise ValueError(
                    f"n_heads ({self.model.n_heads}) must be divisible by "
                    f"tp ({self.distributed.tp})"
                )
            if self.model.n_kv_heads and self.model.n_kv_heads % self.distributed.tp != 0:
                raise ValueError(
                    f"n_kv_heads ({self.model.n_kv_heads}) must be divisible by "
                    f"tp ({self.distributed.tp})"
                )

        if self.train.is_fp8 and self.distributed.tp > 1:
            raise ValueError(
                "FP8 + Tensor Parallelism is not yet supported (torchao Float8Linear "
                "does not compose with DTensor sharding). Use FP8 with FSDP only, "
                "or TP without FP8."
            )

        if self.model.is_moe and self.distributed.pp > 1:
            raise ValueError(
                "MoE + Pipeline Parallelism is not supported. MoE layers use "
                "data-dependent routing that is incompatible with pipeline stage "
                "splitting. Use FSDP, TP, or EP instead."
            )

        if self.distributed.ep > 1:
            if not self.model.is_moe:
                raise ValueError("ep > 1 requires an MoE model (num_experts > 0)")
            if self.model.num_experts % self.distributed.ep != 0:
                raise ValueError(
                    f"num_experts ({self.model.num_experts}) must be divisible by "
                    f"ep ({self.distributed.ep})"
                )

        if self.is_vlm:
            assert self.vlm is not None
            assert self.vision_encoder is not None
            if self.vision_encoder.num_tokens > 0:
                assert self.adapter is not None
                per_frame = self.adapter.output_num_tokens(self.vision_encoder.num_tokens)
                frames = self.views_per_sample
                visual_tokens = frames * per_frame
                residual_image_tokens = self.vlm.residual_stream_image_tokens(visual_tokens)
                required = residual_image_tokens + self.vlm.max_text_len
                if self.train.seq_len < required:
                    raise ValueError(
                        f"train.seq_len ({self.train.seq_len}) insufficient for VLM: "
                        f"residual_image_tokens ({residual_image_tokens}) + "
                        f"vlm.max_text_len ({self.vlm.max_text_len}) = {required}"
                    )
            if self.distributed.pp > 1:
                raise ValueError(
                    "VLM + Pipeline Parallelism is not supported on this branch. "
                    "The vlm-cross-attention branch will address PP integration."
                )

            if isinstance(self.vlm, MoMaConfig):
                if self.model.num_experts > 0:
                    raise ValueError(
                        "MoMa + model.num_experts > 0 is rejected. MoMa derives expert "
                        "counts per modality from vlm.moma_experts_per_modality; set "
                        "model.num_experts=0 (the dense-LLM default) when using arch='moma'."
                    )
                if self.distributed.ep > 1:
                    raise ValueError(
                        "MoMa + Expert Parallelism is not supported in v1. Per-modality "
                        "expert groups need EP-aware dispatch that is not yet wired."
                    )
                if self.train.compile_model:
                    import logging

                    logging.getLogger(__name__).warning(
                        "torch.compile is not yet validated for MoMa dispatch "
                        "(modality_ids-based scatter/gather + expert-choice top-k cause "
                        "graph breaks). Set compile_model=false for MoMa models."
                    )

                if self.train.activation_checkpointing == "full":
                    import logging

                    logging.getLogger(__name__).warning(
                        "AC=full is currently a no-op for MoMa (apply_ac matches "
                        "TransformerBlock only; MoMaBlock is a sibling nn.Module). "
                        "Use ac='selective' (wraps the Attention submodule, still works) "
                        "or reduce moma_experts_per_modality until the apply_ac refactor lands."
                    )
