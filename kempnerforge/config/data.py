"""Data pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

TOKEN_DTYPES = ("uint16", "uint32")


def check_token_dtype(value: str, key: str) -> None:
    """Reject a ``token_dtype`` that is neither ``""`` nor one of ``TOKEN_DTYPES``."""
    if value != "" and value not in TOKEN_DTYPES:
        raise ValueError(
            f"{key} must be 'uint16' or 'uint32', the integer type the raw .bin token "
            f"shards were written with, or empty to take it from the metadata.yaml next "
            f"to the shards; got {value!r}"
        )


@dataclass
class DatasetSource:
    """A single data source in a multi-dataset mixture.

    Either ``path`` (pre-tokenized) or ``hf_name`` (HuggingFace) must be set.
    ``weight`` controls the relative sampling probability. ``token_dtype``
    overrides ``data.token_dtype`` for the ``.bin`` shards under ``path``.
    The mixture knows the source by ``mixture_name``.
    """

    path: str = ""
    weight: float = 1.0
    name: str = ""
    hf_name: str = ""
    hf_config: str = ""
    token_dtype: str = ""

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"Dataset weight must be positive, got {self.weight}")
        check_token_dtype(self.token_dtype, "[[data.datasets]] token_dtype")

    @property
    def mixture_name(self) -> str:
        """The name the mixture gives this source: ``name``, else ``path``, else ``hf_name``.

        Its per-source metrics carry it, and a key of a phase's ``dataset_weights`` or of
        ``anneal_weights`` must equal it to reweight the source.
        """
        return self.name or self.path or self.hf_name


@dataclass
class TrainingPhase:
    """A training phase with custom dataset weights and LR scaling.

    From ``start_step``, the learning rate is multiplied by ``lr_scale`` on any run, and
    the sampler of a ``[[data.datasets]]`` mixture switches to ``dataset_weights``, keyed by
    ``DatasetSource.mixture_name``. ``JobConfig.validate`` refuses the weights on a run that
    builds no mixture, and a key that names no source.
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
    """Data pipeline settings.

    ``token_dtype`` is the integer type of the raw ``.bin`` token shards,
    ``"uint16"`` or ``"uint32"``, for ``dataset_path`` and every ``datasets``
    source that does not set its own. Left empty, the ``metadata.yaml`` next to
    the shards decides, as ``MemoryMappedDataset`` describes. ``.npy`` shards
    carry their dtype in their header and ignore it.
    """

    dataset_path: str = ""
    file_pattern: str = "*.npy"
    token_dtype: str = ""
    tokenizer_path: str = ""
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    hf_dataset_name: str | None = None
    hf_dataset_config: str | None = None
    hf_dataset_split: str = "train"
    hf_dataset_text_field: str = "text"
    hf_dataset_image_field: str = "image"
    hf_dataset_prompt_field: str = ""
    hf_image_size: int = 224
    hf_max_images: int = 1
    hf_streaming: bool = False
    pack_sequences: bool = False
    datasets: list[DatasetSource] = field(default_factory=list)
    mix_temperature: float = 1.0
    phases: list[TrainingPhase] = field(default_factory=list)
    anneal_start_step: int = 0
    anneal_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be >= 1")
        if self.hf_max_images < 1:
            raise ValueError("hf_max_images must be >= 1")
        if self.mix_temperature <= 0:
            raise ValueError("mix_temperature must be positive")
        check_token_dtype(self.token_dtype, "data.token_dtype")
        for src in self.datasets:
            if not src.path and not src.hf_name:
                raise ValueError(f"DatasetSource '{src.name}' must have either path or hf_name")
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
                "Cannot use both data.phases and data.anneal_start_step; "
                "use phases for multi-phase scheduling"
            )

    def mixture_weight_fields(self) -> list[str]:
        """The settings that reweight a ``[[data.datasets]]`` mixture, named for an error.

        Each phase that sets ``dataset_weights``, and ``anneal_start_step`` with
        ``anneal_weights`` when the two make a phase. A run without a mixture refuses them.
        """
        fields = [
            f"data.phases[{i}].dataset_weights"
            for i, phase in enumerate(self.phases)
            if phase.dataset_weights
        ]
        if self.anneal_start_step > 0 and self.anneal_weights:
            fields.append("data.anneal_start_step and data.anneal_weights")
        return fields

    def unknown_mixture_weight_keys(self) -> dict[str, list[str]]:
        """The keys of each setting ``mixture_weight_fields`` lists that name no source.

        Maps ``data.phases[i].dataset_weights``, and ``data.anneal_weights`` when
        ``anneal_start_step`` makes it a phase, to its sorted keys that match no
        ``DatasetSource.mixture_name`` in ``datasets``. A setting whose keys all match is left
        out, so the result is empty when every key names a source.
        """
        names = {source.mixture_name for source in self.datasets}
        settings = {
            f"data.phases[{i}].dataset_weights": phase.dataset_weights
            for i, phase in enumerate(self.phases)
        }
        if self.anneal_start_step > 0:
            settings["data.anneal_weights"] = self.anneal_weights
        unknown = {field: sorted(set(weights) - names) for field, weights in settings.items()}
        return {field: keys for field, keys in unknown.items() if keys}
