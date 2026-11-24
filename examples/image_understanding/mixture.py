"""A packed training loader over a mixture of image-text sources described in a TOML file.

A mixture file lists ``[[source]]`` tables, each naming a ``reader`` (a key of
``READERS``, ``"finevision"`` or ``"datacomp"``), a ``repeat`` count (default 1) and
the reader's arguments; a ``finevision`` source reads every subset under ``root``
that holds an ``index.npz``, or those in ``subsets``, less those in ``exclude``,
each repeated. A ``datacomp`` source streams ``DataCompStream``'s shards from
``shard_dir`` and is the mixture's only source, without repeats. ``[render]`` holds ``ChatRenderer`` options and ``[packing]`` ``PackedLoader``
options (``view_groups``). ``check_mixture`` refuses any other table, an unknown
reader and an argument its reader, the renderer or the packer does not take, and
``check_run`` a run the mixture cannot train, both before the run starts.
"""

from __future__ import annotations

import inspect
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from image_text import pixmo, testbed
from image_text.datacomp import DataCompStream
from image_text.finevision import FineVisionArrow
from image_text.packing import PackedLoader, PackedStreamLoader
from image_text.render import ChatRenderer, RenderedDataset, RenderedStream
from torch.utils.data import ConcatDataset, Dataset

from kempnerforge.config.job import JobConfig
from kempnerforge.data.vlm_dataset import build_tokenizer
from kempnerforge.distributed.utils import get_dp_info
from kempnerforge.training.data_pipeline import DataFn, DataPipeline
from kempnerforge.training.runtime import RuntimeContext

READERS: dict[str, Callable[..., Dataset]] = {
    "coco_captions": testbed.coco_captions,
    "vqav2": testbed.vqav2,
    "textvqa": testbed.textvqa,
    "docvqa": testbed.docvqa,
    "chartqa": testbed.chartqa,
    "pixmo_captions": pixmo.pixmo_captions,
    "pixmo_cap_qa": pixmo.pixmo_cap_qa,
    "pixmo_pointing": pixmo.pixmo_pointing,
    "pixmo_counting": pixmo.pixmo_counting,
    "pixmo_point_explanations": pixmo.pixmo_point_explanations,
}


TABLES = ("source", "render", "packing")
PACKING_OPTIONS = ("view_groups",)
RUN_RENDER_OPTIONS = (
    "tokenizer",
    "max_text_len",
    "image_size",
    "max_images",
    "tile_grid",
    "tile_global",
)
FINEVISION_OPTIONS = ("root", "subsets", "exclude")
STREAM_RUN_OPTIONS = ("rank", "world_size", "num_workers", "seed")
SOURCE_OPTIONS = ("reader", "repeat", "name")


def _accepted(factory: Callable[..., Any]) -> tuple[set[str] | None, set[str]]:
    """The keyword arguments ``factory`` takes (None for any) and those it requires."""
    names: set[str] = set()
    required: set[str] = set()
    for parameter in inspect.signature(factory).parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return None, required
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.POSITIONAL_ONLY):
            continue
        names.add(parameter.name)
        if parameter.default is inspect.Parameter.empty:
            required.add(parameter.name)
    return names, required


def _check_arguments(
    where: str, given: set[str], factory: Callable[..., Any], fixed: set[str]
) -> None:
    accepted, required = _accepted(factory)
    allowed = None if accepted is None else accepted - fixed
    unknown = sorted(given - allowed) if allowed is not None else []
    if unknown:
        raise ValueError(f"{where}: unknown option(s) {unknown}; expected some of {sorted(allowed)}")
    missing = sorted(required - given - fixed)
    if missing:
        raise ValueError(f"{where}: missing option(s) {missing}")


def check_mixture(mixture: dict[str, Any]) -> None:
    """Refuse a mixture with an unknown table, reader or option, reading no data.

    Raises:
        ValueError: naming the first table, source or option at fault.
    """
    unknown = sorted(set(mixture) - set(TABLES))
    if unknown:
        raise ValueError(f"unknown mixture table(s) {unknown}; expected some of {list(TABLES)}")
    render = mixture.get("render", {})
    fixed = sorted(set(render) & set(RUN_RENDER_OPTIONS))
    if fixed:
        raise ValueError(f"[render]: {fixed} are set from the run config, not the mixture")
    _check_arguments("[render]", set(render), ChatRenderer, set(RUN_RENDER_OPTIONS))
    packing = sorted(set(mixture.get("packing", {})) - set(PACKING_OPTIONS))
    if packing:
        raise ValueError(f"[packing]: unknown option(s) {packing}; expected some of {list(PACKING_OPTIONS)}")
    sources = mixture.get("source", [])
    if not sources:
        raise ValueError("the mixture lists no source")
    for index, entry in enumerate(sources):
        reader = entry.get("reader")
        where = f"source {entry.get('name', reader)!r} (number {index + 1})"
        options = set(entry) - set(SOURCE_OPTIONS)
        repeat = entry.get("repeat", 1)
        if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
            raise ValueError(f"{where}: repeat must be a positive integer, got {repeat!r}")
        if reader == "finevision":
            _check_arguments(
                where,
                options - set(FINEVISION_OPTIONS),
                FineVisionArrow,
                {"subset_dir"},
            )
            if "root" not in options:
                raise ValueError(f"{where}: missing option(s) ['root']")
        elif reader == "datacomp":
            if len(sources) != 1:
                raise ValueError(
                    f"{where}: a datacomp source streams, so it must be the mixture's only source"
                )
            if repeat != 1:
                raise ValueError(f"{where}: a stream does not repeat, got repeat={repeat}")
            _check_arguments(where, options, DataCompStream, set(STREAM_RUN_OPTIONS))
        elif reader in READERS:
            _check_arguments(where, options, READERS[reader], set())
        else:
            raise ValueError(
                f"{where}: unknown reader {reader!r}; expected finevision or one of {sorted(READERS)}"
            )


def check_run(config: JobConfig) -> None:
    """Refuse a run a mixture cannot train: one row of a packed VLM per micro-batch.

    Raises:
        ValueError: without ``[vlm]``, ``data.pack_sequences`` or ``train.batch_size = 1``.
    """
    if config.vlm is None or not config.data.pack_sequences:
        raise ValueError("a mixture trains a packed VLM: set [vlm] and data.pack_sequences")
    if config.train.batch_size != 1:
        raise ValueError(
            f"a mixture yields one packed row per micro-batch: set train.batch_size = 1, "
            f"got {config.train.batch_size}"
        )


def load_mixture(path: str | Path) -> dict[str, Any]:
    """The mixture file at ``path``, checked with ``check_mixture``."""
    with open(path, "rb") as file:
        mixture = tomllib.load(file)
    check_mixture(mixture)
    return mixture


def finevision_subsets(
    root: str | Path, subsets: list[str] | None = None, exclude: list[str] | None = None
) -> list[str]:
    """The FineVision subsets under ``root`` to read: ``subsets`` (default all) less ``exclude``."""
    present = sorted(p.name for p in Path(root).iterdir() if (p / "index.npz").exists())
    chosen = present if subsets is None else list(subsets)
    excluded = set(exclude or [])
    missing = sorted((set(chosen) | excluded) - set(present))
    if missing:
        raise ValueError(f"FineVision subsets not indexed under {root}: {missing}")
    return [name for name in chosen if name not in excluded]


def stream_source(mixture: dict[str, Any]) -> dict[str, Any] | None:
    """The options of the mixture's ``datacomp`` source, or None when it has none."""
    check_mixture(mixture)
    entry = mixture["source"][0]
    if entry.get("reader") != "datacomp":
        return None
    return {key: value for key, value in entry.items() if key not in SOURCE_OPTIONS}


def mixture_sources(mixture: dict[str, Any]) -> list[tuple[str, Dataset, int]]:
    """Each map-style source's name, reader and repeat count, in the file's order."""
    check_mixture(mixture)
    if stream_source(mixture) is not None:
        raise ValueError("a datacomp source streams alone; read it with stream_source")
    sources: list[tuple[str, Dataset, int]] = []
    for entry in mixture.get("source", []):
        options = dict(entry)
        reader = options.pop("reader", None)
        repeat = options.pop("repeat", 1)
        name = options.pop("name", reader)
        if reader == "finevision":
            root = Path(options.pop("root"))
            chosen = finevision_subsets(
                root, options.pop("subsets", None), options.pop("exclude", None)
            )
            for subset in chosen:
                sources.append(
                    (f"{name}/{subset}", FineVisionArrow(root / subset, **options), repeat)
                )
        else:
            sources.append((str(name), READERS[reader](**options), repeat))
    return sources


def mixture_data_fn(path: str | Path) -> DataFn:
    """A ``run_training`` ``data_fn`` packing the mixture at ``path`` into rows of seq_len.

    Documents render uncut and with every image, each image at ``data.hf_image_size``
    and cut into views as ``vlm.tile_grid`` and ``vlm.tile_global`` say, and pack with
    the model's ``tokens_per_image``, one row per micro-batch, so the run needs
    ``data.pack_sequences`` and ``train.batch_size = 1`` (``check_run``). The file is
    read and checked here, before the run starts.
    """
    mixture = load_mixture(path)

    def data_fn(
        config: JobConfig, runtime: RuntimeContext, *, tokens_per_image: int | None = None
    ) -> DataPipeline:
        check_run(config)
        if tokens_per_image is None:
            raise ValueError("a mixture trains a packed VLM: set [vlm] and data.pack_sequences")
        assert config.vlm is not None
        dp_rank, dp_size = get_dp_info(runtime.device_mesh)
        renderer = ChatRenderer(
            build_tokenizer(config.data.tokenizer_path),
            max_text_len=None,
            image_size=config.data.hf_image_size,
            max_images=None,
            tile_grid=config.vlm.tile_grid,
            tile_global=config.vlm.tile_global,
            **mixture.get("render", {}),
        )
        seed = config.train.effective_data_seed
        workers = config.data.num_workers
        packing = {
            "seq_len": config.train.seq_len,
            "tokens_per_image": tokens_per_image,
            "pad_id": renderer.pad_id,
            **mixture.get("packing", {}),
            "num_workers": workers,
            "pin_memory": config.data.pin_memory,
            "prefetch_factor": config.data.prefetch_factor if workers else None,
        }
        stream = stream_source(mixture)
        if stream is not None:
            shards = DataCompStream(
                stream.pop("shard_dir"),
                rank=dp_rank,
                world_size=dp_size,
                num_workers=workers,
                seed=seed,
                **stream,
            )
            rendered = RenderedStream(shards, renderer, seed=seed, rank=dp_rank)
            loader: Any = PackedStreamLoader(rendered, **packing)
        else:
            documents = ConcatDataset(
                [
                    RenderedDataset(reader, renderer, seed=f"{seed}:{name}:{copy}")
                    for name, reader, repeat in mixture_sources(mixture)
                    for copy in range(repeat)
                ]
            )
            loader = PackedLoader(documents, rank=dp_rank, world_size=dp_size, seed=seed, **packing)
        return DataPipeline(dataloader=loader, dp_rank=dp_rank, dp_size=dp_size)

    return data_fn
