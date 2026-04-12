"""Data pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetSource:
    """A single data source in a multi-dataset mixture.

    Either ``path`` (pre-tokenized) or ``hf_name`` (HuggingFace) must be set.
    ``weight`` controls the relative sampling probability.
    """

    path: str = ""  # Pre-tokenized data directory
    weight: float = 1.0  # Relative sampling weight
    name: str = ""  # Name for per-dataset metrics (auto-derived if empty)
    hf_name: str = ""  # HuggingFace dataset name
    hf_config: str = ""  # HuggingFace dataset config

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"Dataset weight must be positive, got {self.weight}")


@dataclass
class TrainingPhase:
    """A training phase with custom dataset weights and LR scaling.

    Used for data annealing: at ``start_step``, the mixture sampler switches
    to ``dataset_weights`` and the learning rate is multiplied by ``lr_scale``.
    """

    start_step: int = 0
    dataset_weights: dict[str, float] = field(default_factory=dict)
    lr_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("TrainingPhase.start_step must be non-negative")
        if self.lr_scale <= 0:
            raise ValueError("TrainingPhase.lr_scale must be positive")
        for name, w in self.dataset_weights.items():
            if w < 0:
                raise ValueError(f"TrainingPhase.dataset_weights['{name}'] must be non-negative")


@dataclass
class DataConfig:
    """Data pipeline settings."""

    dataset_path: str = ""
    file_pattern: str = "*.npy"  # Glob pattern for data files (e.g., "*.npy", "tokenized_*.bin")
    tokenizer_path: str = ""
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    # For HuggingFace datasets
    hf_dataset_name: str | None = None
    hf_dataset_config: str | None = None  # Dataset config name (e.g., "wikitext-2-raw-v1")
    hf_dataset_split: str = "train"
    hf_dataset_text_field: str = "text"
    hf_streaming: bool = False  # Use streaming (IterableDataset) for large HF datasets
    pack_sequences: bool = False  # Document-aware packing with cross-doc isolation
    # Multi-dataset mixing (overrides dataset_path/hf_dataset_name when non-empty)
    datasets: list[DatasetSource] = field(default_factory=list)
    mix_temperature: float = 1.0  # Temperature for weight scaling (1.0=as-is, >1=uniform)
    # Phase scheduling (step-triggered weight/LR transitions)
    phases: list[TrainingPhase] = field(default_factory=list)
    # Annealing shortcut (syntactic sugar for a common 2-phase pattern)
    anneal_start_step: int = 0  # 0 = disabled
    anneal_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be >= 1")
        if self.mix_temperature <= 0:
            raise ValueError("mix_temperature must be positive")
        for src in self.datasets:
            if not src.path and not src.hf_name:
                raise ValueError(f"DatasetSource '{src.name}' must have either path or hf_name")
        # Phase validation
        if self.phases:
            steps = [p.start_step for p in self.phases]
            if steps != sorted(steps) or len(steps) != len(set(steps)):
                raise ValueError(
                    "data.phases start_steps must be strictly monotonically increasing"
                )
        if self.anneal_start_step < 0:
            raise ValueError("anneal_start_step must be non-negative")
        if self.phases and self.anneal_start_step > 0:
            raise ValueError(
                "Cannot use both data.phases and data.anneal_start_step — "
                "use phases for multi-phase scheduling"
            )
