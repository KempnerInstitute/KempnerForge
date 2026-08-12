"""Video input configuration.

``VideoConfig`` is the ``[video]`` top-level section. When present, the job
trains on a video dataset through the VLM wrapper: a clip is decoded into an
ordered set of frames, each preprocessed like an image and fed to the vision
encoder. The section is a sibling of ``[vision_encoder]`` / ``[adapter]`` /
``[vlm]`` and requires ``[vlm]`` to be set.

Frame-sampling defaults: sample at ``fps`` per second, include the first and
last frame, and cap at ``max_frames``. ``max_frames`` is the
per-clip frame budget; the number of visual tokens it implies
(``max_frames * tokens_per_frame``) feeds the residual-stream / sequence-length
math once the model consumes video.

One corpus is configured with the flat fields; several are mixed by listing
``[[video.datasets]]`` entries (see ``VideoDatasetSource``), which share the
global frame geometry and are sampled by weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

_VIDEO_SPLITS = ("train", "validation", "test")

# Fields a ``[[video.datasets]]`` entry may override per corpus. Frame geometry
# is deliberately absent — see ``VideoDatasetSource``.
_SOURCE_OVERRIDES = (
    "dataset_type",
    "data_root",
    "dataset_name",
    "subset",
    "split",
    "prompt",
    "prompt_pool",
    "text_source",
    "qa_format",
    "max_samples",
    "require_video_file",
)


@dataclass
class VideoDatasetSource:
    """One corpus in a video mixture (a ``[[video.datasets]]`` entry).

    Frame geometry (``fps`` / ``max_frames`` / ``min_frames`` / ``frame_size`` /
    ``sampling_policy``) is deliberately *not* settable here: it sizes the
    visual-token budget that ``model.max_seq_len`` is validated against, so it
    stays global on ``[video]`` and every source shares it. Setting one of those
    keys on a source is rejected by the loader as an unknown key.

    Fields left empty inherit the corresponding ``[video]`` value.

    Fields:
        dataset_type: Registry key for the dataset builder (required).
        data_root: Root directory of this corpus (required).
        dataset_name: On-disk corpus name within a style (WebVid-style layouts).
        subset: Within-corpus variant (e.g. NExT-QA ``"MC"`` / ``"OE"``).
        split: Which split to read; inherits ``[video].split`` when empty.
        prompt: Instruction prepended to the target text and masked from loss.
        text_source: Which on-disk text field supplies the target (corpus-specific).
        qa_format: Registry key for question/answer rendering (QA corpora only);
            inherits ``[video].qa_format`` when empty.
        max_samples: Cap this corpus to N examples (``0`` = all).
        require_video_file: Drop manifest rows whose video is missing. ``None``
            lets the builder pick its own default (on for partially-downloaded
            corpora, off where an existence scan would be prohibitive).
        weight: Relative sampling weight within the mixture.
        name: Label for per-dataset metrics (defaults to ``dataset_type``).
    """

    dataset_type: str = ""
    data_root: str = ""
    dataset_name: str = ""
    subset: str = ""
    split: str = ""
    prompt: str = ""
    prompt_pool: list[str] = field(default_factory=list)
    text_source: str = ""
    qa_format: str = ""
    max_samples: int = 0
    require_video_file: bool | None = None
    weight: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_type:
            raise ValueError("video.datasets entries require a dataset_type")
        if not self.data_root:
            raise ValueError(f"video.datasets entry {self.dataset_type!r} requires a data_root")
        if self.weight <= 0:
            raise ValueError(
                f"video.datasets[{self.dataset_type!r}].weight must be positive (got {self.weight})"
            )
        if self.max_samples < 0:
            raise ValueError(
                f"video.datasets[{self.dataset_type!r}].max_samples must be "
                f"non-negative (got {self.max_samples})"
            )
        if self.split and self.split not in _VIDEO_SPLITS:
            raise ValueError(
                f"video.datasets[{self.dataset_type!r}].split must be one of "
                f"{_VIDEO_SPLITS} (got {self.split!r})"
            )

    @property
    def metrics_name(self) -> str:
        """Label used for per-dataset metrics."""
        return self.name or self.dataset_type


@dataclass
class VideoConfig:
    """Video dataset location and frame-sampling knobs.

    A single-corpus run sets the flat fields (``data_root`` / ``dataset_type`` /
    ...). A mixture instead lists ``[[video.datasets]]`` entries, each a
    :class:`VideoDatasetSource`; the two forms are mutually exclusive. Frame
    geometry is global either way.

    Fields:
        data_root: Root directory of the on-disk video dataset.
        dataset_type: Registry key for the dataset builder (``"webvid"`` default).
        dataset_name: On-disk corpus name within a style (e.g. ``"webvid-10M"``).
        subset: Within-corpus variant (e.g. NExT-QA ``"MC"`` / ``"OE"``).
        sampling_policy: Registry key for the frame-sampling policy (``"uniform"``).
        split: Which split to read (``"train"``, ``"validation"`` or ``"test"``).
        max_samples: Cap the manifest to this many examples (``0`` = all).
        max_frames: Maximum frames sampled per clip (the per-clip budget).
        min_frames: Minimum frames sampled per clip; short clips pad up to this.
        fps: Target sampling rate in frames per second.
        frame_size: Square pixel size each frame is resized to.
        prompt: Optional instruction prepended to the target text, masked from loss.
        text_source: Which on-disk text field supplies the target (corpus-specific).
        qa_format: Registry key for question/answer rendering (QA corpora only).
        require_video_file: Drop manifest rows whose video file is missing.
        datasets: Mixture sources; when non-empty the flat corpus fields are unused.
    """

    data_root: str = ""
    dataset_type: str = "webvid"
    dataset_name: str = "webvid-10M"
    subset: str = ""
    sampling_policy: str = "uniform"
    split: str = "train"
    max_samples: int = 0
    max_frames: int = 16
    min_frames: int = 4
    fps: float = 2.0
    frame_size: int = 224
    prompt: str = ""
    # Caption prompts sampled one per example, seeded by (index, epoch);
    # non-empty wins over ``prompt``. A pool of paraphrases keeps the model from
    # tying one exact wording to "produce a caption".
    prompt_pool: list[str] = field(default_factory=list)
    text_source: str = ""
    qa_format: str = "mcq_letter"
    require_video_file: bool | None = None
    datasets: list[VideoDatasetSource] = field(default_factory=list)

    def sources(self) -> list[VideoDatasetSource]:
        """Normalize both config forms to a list of sources.

        A flat single-corpus config yields one source; a ``[[video.datasets]]``
        mixture yields its entries with empty fields filled in from ``[video]``.
        """
        if not self.datasets:
            return [
                VideoDatasetSource(
                    dataset_type=self.dataset_type,
                    data_root=self.data_root,
                    dataset_name=self.dataset_name,
                    subset=self.subset,
                    split=self.split,
                    prompt=self.prompt,
                    text_source=self.text_source,
                    qa_format=self.qa_format,
                    max_samples=self.max_samples,
                    require_video_file=self.require_video_file,
                )
            ]
        return [
            replace(
                src,
                dataset_name=src.dataset_name or self.dataset_name,
                subset=src.subset or self.subset,
                split=src.split or self.split,
                prompt=src.prompt or self.prompt,
                prompt_pool=src.prompt_pool or self.prompt_pool,
                text_source=src.text_source or self.text_source,
                qa_format=src.qa_format or self.qa_format,
                max_samples=src.max_samples or self.max_samples,
                # Tri-state: a source's explicit True/False wins; None inherits
                # [video]'s value (itself possibly None -> builder default).
                require_video_file=(
                    src.require_video_file
                    if src.require_video_file is not None
                    else self.require_video_file
                ),
            )
            for src in self.datasets
        ]

    def for_source(self, source: VideoDatasetSource) -> VideoConfig:
        """A single-corpus view of this config with ``source``'s fields applied.

        Builders keep their ``(video_config, tokenizer_path, max_text_len)``
        signature and read one corpus off a plain ``VideoConfig``; frame geometry
        comes from ``self`` so every source in a mixture shares it.
        """
        overrides = {f: getattr(source, f) for f in _SOURCE_OVERRIDES}
        # ``qa_format`` is the one override with a non-empty default, so an
        # unresolved source must not blank it out.
        if not overrides["qa_format"]:
            overrides["qa_format"] = self.qa_format
        return replace(self, datasets=[], **overrides)

    def __post_init__(self) -> None:
        # Late imports populate the dataset/sampling registries (their decorators
        # run on import) and avoid a config->data import cycle; only hit for a
        # video job. ``av`` is not required here (it is lazy inside the decoder).
        import kempnerforge.data.qa_format  # noqa: F401, PLC0415
        import kempnerforge.data.video_dataset  # noqa: F401, PLC0415
        import kempnerforge.data.video_io  # noqa: F401, PLC0415
        import kempnerforge.data.video_qa_datasets  # noqa: F401, PLC0415
        from kempnerforge.config.registry import registry  # noqa: PLC0415

        known = registry.list_video_datasets()
        if self.datasets:
            if self.data_root:
                raise ValueError(
                    "video.data_root and [[video.datasets]] are mutually exclusive; "
                    "set the corpus root on each source instead."
                )
            for src in self.datasets:
                if src.dataset_type not in known:
                    raise ValueError(
                        f"video.datasets dataset_type must be one of {sorted(known)} "
                        f"(got {src.dataset_type!r})"
                    )
            names = [src.metrics_name for src in self.datasets]
            if len(set(names)) != len(names):
                raise ValueError(
                    f"video.datasets names must be unique (got {names}); set an "
                    "explicit `name` when mixing two sources of the same type."
                )
        elif self.dataset_type not in known:
            raise ValueError(
                f"video.dataset_type must be one of {sorted(known)} (got {self.dataset_type!r})"
            )
        formats = registry.list_qa_formats()
        for value in [self.qa_format, *(s.qa_format for s in self.datasets)]:
            if value and value not in formats:
                raise ValueError(
                    f"video.qa_format must be one of {sorted(formats)} (got {value!r})"
                )
        if self.sampling_policy not in registry.list_sampling_policies():
            raise ValueError(
                "video.sampling_policy must be one of "
                f"{sorted(registry.list_sampling_policies())} (got {self.sampling_policy!r})"
            )
        if self.split not in _VIDEO_SPLITS:
            raise ValueError(f"video.split must be one of {_VIDEO_SPLITS} (got {self.split!r})")
        if self.max_samples < 0:
            raise ValueError(f"video.max_samples must be non-negative (got {self.max_samples})")
        for pool, where in [(self.prompt_pool, "video")] + [
            (s.prompt_pool, f"video.datasets[{s.dataset_type!r}]") for s in self.datasets
        ]:
            if any(not p.strip() for p in pool):
                raise ValueError(f"{where}.prompt_pool entries must be non-empty strings")
        if self.min_frames < 1:
            raise ValueError(f"video.min_frames must be >= 1 (got {self.min_frames})")
        if self.max_frames < 1:
            raise ValueError(f"video.max_frames must be >= 1 (got {self.max_frames})")
        if self.min_frames > self.max_frames:
            raise ValueError(
                f"video.min_frames ({self.min_frames}) must be <= video.max_frames "
                f"({self.max_frames})"
            )
        if self.fps <= 0:
            raise ValueError(f"video.fps must be positive (got {self.fps})")
        if self.frame_size <= 0:
            raise ValueError(f"video.frame_size must be positive (got {self.frame_size})")
