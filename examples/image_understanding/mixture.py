"""A packed training loader over a mixture of image-text sources described in a TOML file.

A mixture file lists ``[[source]]`` tables, each naming a ``reader`` (a key of
``READERS``, or ``"finevision"``), a ``repeat`` count (default 1) and the reader's
arguments; a ``finevision`` source reads every subset under ``root`` that holds an
``index.npz``, or those in ``subsets``, less those in ``exclude``, each repeated.
``[render]`` holds ``ChatRenderer`` options and ``[packing]`` ``PackedLoader``
options (``view_groups``).
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from image_text import pixmo, testbed
from image_text.finevision import FineVisionArrow
from image_text.packing import PackedLoader
from image_text.render import ChatRenderer, RenderedDataset
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


def load_mixture(path: str | Path) -> dict[str, Any]:
    """The mixture file at ``path``."""
    with open(path, "rb") as file:
        return tomllib.load(file)


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


def mixture_sources(mixture: dict[str, Any]) -> list[tuple[str, Dataset, int]]:
    """Each source's name, reader and repeat count, in the file's order."""
    sources: list[tuple[str, Dataset, int]] = []
    for entry in mixture.get("source", []):
        options = dict(entry)
        reader = options.pop("reader", None)
        repeat = options.pop("repeat", 1)
        name = options.pop("name", reader)
        if not isinstance(repeat, int) or repeat < 1:
            raise ValueError(f"source {name!r}: repeat must be a positive integer, got {repeat!r}")
        if reader == "finevision":
            root = Path(options.pop("root"))
            chosen = finevision_subsets(
                root, options.pop("subsets", None), options.pop("exclude", None)
            )
            for subset in chosen:
                sources.append(
                    (f"{name}/{subset}", FineVisionArrow(root / subset, **options), repeat)
                )
        elif reader in READERS:
            sources.append((name, READERS[reader](**options), repeat))
        else:
            raise ValueError(
                f"unknown reader {reader!r}; expected finevision or one of {sorted(READERS)}"
            )
    if not sources:
        raise ValueError("the mixture lists no source")
    return sources


def mixture_data_fn(path: str | Path) -> DataFn:
    """A ``run_training`` ``data_fn`` packing the mixture at ``path`` into rows of seq_len.

    Documents render uncut and with every image, each image at ``data.hf_image_size``
    and cut into views as ``vlm.tile_grid`` and ``vlm.tile_global`` say, and pack with
    the model's ``tokens_per_image``, so the run needs ``data.pack_sequences``.
    """
    mixture = load_mixture(path)

    def data_fn(
        config: JobConfig, runtime: RuntimeContext, *, tokens_per_image: int | None = None
    ) -> DataPipeline:
        if tokens_per_image is None or config.vlm is None:
            raise ValueError("a mixture trains a packed VLM: set [vlm] and data.pack_sequences")
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
        documents = ConcatDataset(
            [
                RenderedDataset(reader, renderer, seed=f"{seed}:{name}:{copy}")
                for name, reader, repeat in mixture_sources(mixture)
                for copy in range(repeat)
            ]
        )
        workers = config.data.num_workers
        loader = PackedLoader(
            documents,
            seq_len=config.train.seq_len,
            tokens_per_image=tokens_per_image,
            pad_id=renderer.pad_id,
            rank=dp_rank,
            world_size=dp_size,
            seed=seed,
            **mixture.get("packing", {}),
            num_workers=workers,
            pin_memory=config.data.pin_memory,
            prefetch_factor=config.data.prefetch_factor if workers else None,
        )
        return DataPipeline(dataloader=loader, dp_rank=dp_rank, dp_size=dp_size)

    return data_fn
