"""Tests for building the packed loader of a mixture of image-text sources."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from image_text.packing import PackedStreamLoader
from mixture import finevision_subsets, mixture_data_fn, mixture_sources, stream_source
from test_datacomp import _shards
from test_finevision import _row, _run, _write_parquet

from kempnerforge.config.loader import load_config
from kempnerforge.training.runtime import RuntimeContext

CONFIG = """
[vision_encoder]
type = "random"
[vlm]
arch = "joint_decoder"
tile_grid = 2
[data]
tokenizer_path = "Qwen/Qwen3-0.6B"
hf_image_size = 8
pack_sequences = true
num_workers = 0
[train]
seq_len = 384
seed = 3
"""


@pytest.fixture
def finevision(tmp_path: Path) -> Path:
    src, dst = tmp_path / "parquet", tmp_path / "arrow"
    subsets = {
        "images": [_row(1, [("Q0", "A0")], 5), _row(2, [("Q1", "A1"), ("Q1b", "A1b")], 5)],
        "text_chat": [_row(0, [("T0", "U0")], 5)],
        "leaky": [_row(1, [("L0", "M0")], 5)],
    }
    for name, rows in subsets.items():
        _write_parquet(src / name / "train-00000-of-00001.parquet", [rows])
    _run("convert", "--src", str(src), "--dst", str(dst))
    _run("index", "--src", str(src), "--dst", str(dst))
    return dst


@pytest.fixture
def captions(tmp_path: Path, write_image: Callable[..., Path]) -> dict[str, str]:
    write_image("coco/a.png")
    annotations = tmp_path / "captions.json"
    annotations.write_text(json.dumps([{"image": "a.png", "caption": ["a red square"]}]))
    return {"annotation_file": str(annotations), "image_root": str(tmp_path / "coco")}


def _mixture(tmp_path: Path, finevision: Path, captions: dict[str, str]) -> Path:
    path = tmp_path / "mixture.toml"
    path.write_text(
        "[render]\nanswer_only = true\ntrain_reply_prefix = true\nlead_space = true\n"
        "[packing]\nview_groups = true\n"
        '[[source]]\nreader = "coco_captions"\nrepeat = 3\n'
        f'annotation_file = "{captions["annotation_file"]}"\n'
        f'image_root = "{captions["image_root"]}"\nprompt = "Caption it."\n'
        f'[[source]]\nreader = "finevision"\nroot = "{finevision}"\nexclude = ["leaky"]\n'
    )
    return path


def _runtime() -> RuntimeContext:
    return RuntimeContext(
        rank=0, local_rank=0, world_size=1, device=torch.device("cpu"), device_mesh=None
    )


class TestMixture:
    def test_finevision_subsets_are_chosen_and_excluded(self, finevision: Path) -> None:
        assert finevision_subsets(finevision) == ["images", "leaky", "text_chat"]
        assert finevision_subsets(finevision, exclude=["leaky"]) == ["images", "text_chat"]
        assert finevision_subsets(finevision, ["text_chat"]) == ["text_chat"]
        with pytest.raises(ValueError, match="not indexed"):
            finevision_subsets(finevision, exclude=["missing"])

    def test_sources_keep_their_repeats(
        self, tmp_path: Path, finevision: Path, captions: dict[str, str]
    ) -> None:
        from mixture import load_mixture

        sources = mixture_sources(load_mixture(_mixture(tmp_path, finevision, captions)))
        assert [(name, len(reader), repeat) for name, reader, repeat in sources] == [
            ("coco_captions", 1, 3),
            ("finevision/images", 2, 1),
            ("finevision/text_chat", 1, 1),
        ]

    @pytest.mark.parametrize(
        ("source", "message"),
        [
            ({"reader": "nope"}, "unknown reader"),
            ({"reader": "coco_captions", "repeat": 0}, "repeat must be a positive"),
        ],
    )
    def test_bad_sources_are_refused(self, source: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            mixture_sources({"source": [source]})
        with pytest.raises(ValueError, match="no source"):
            mixture_sources({})

    def test_the_data_fn_packs_every_source(
        self, tmp_path: Path, finevision: Path, captions: dict[str, str]
    ) -> None:
        config_path = tmp_path / "run.toml"
        config_path.write_text(CONFIG)
        config = load_config(str(config_path), cli_args=[])
        data_fn = mixture_data_fn(_mixture(tmp_path, finevision, captions))
        pipeline = data_fn(config, _runtime(), tokens_per_image=4)
        loader = pipeline.dataloader
        assert len(loader.dataset) == 3 + 2 + 1
        assert (loader.seq_len, loader.tokens_per_image, loader.seed) == (384, 4, 3)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == batch["attention_group_ids"].shape == (1, 384)
        assert batch["pixel_values"].shape == (30, 3, 8, 8)
        assert int(batch["doc_ids"].max()) == 6
        assert int(batch["attention_group_ids"].max()) == 29
        text = loader.dataset[0]
        assert int(text["text_len"]) == len(text["input_ids"])

    def test_a_mixture_needs_a_packed_vlm_run(
        self, tmp_path: Path, finevision: Path, captions: dict[str, str]
    ) -> None:
        config_path = tmp_path / "run.toml"
        config_path.write_text(CONFIG)
        data_fn = mixture_data_fn(_mixture(tmp_path, finevision, captions))
        with pytest.raises(ValueError, match="packed VLM"):
            data_fn(load_config(str(config_path), cli_args=[]), _runtime())


def _datacomp_mixture(tmp_path: Path, shards: Path, extra: str = "") -> Path:
    path = tmp_path / "datacomp.toml"
    path.write_text(
        "[render]\nanswer_only = false\nlead_space = true\n"
        f'[[source]]\nreader = "datacomp"\nshard_dir = "{shards}"\n'
        'prompt = "Caption it."\nopen_shards = 2\n' + extra
    )
    return path


class TestDataCompMixture:
    def test_the_data_fn_packs_the_stream(self, tmp_path: Path) -> None:
        config_path = tmp_path / "run.toml"
        config_path.write_text(CONFIG)
        config = load_config(str(config_path), cli_args=[])
        shards = _shards(tmp_path / "shards", 2, 6)
        data_fn = mixture_data_fn(_datacomp_mixture(tmp_path, shards))
        loader = data_fn(config, _runtime(), tokens_per_image=4).dataloader
        assert isinstance(loader, PackedStreamLoader)
        assert (loader.seq_len, loader.tokens_per_image) == (384, 4)
        assert loader.stream.documents.seed == 3
        batch = next(iter(loader))
        documents = int(batch["doc_ids"].max())
        assert documents > 1 and batch["pixel_values"].shape == (5 * documents, 3, 8, 8)
        assert loader.state_dict()["stream"]["positions"][0][2] >= documents

    def test_a_stream_is_the_only_source_and_does_not_repeat(
        self, tmp_path: Path, captions: dict[str, str]
    ) -> None:
        shards = _shards(tmp_path / "shards", 1, 1)
        coco = (
            '[[source]]\nreader = "coco_captions"\n'
            f'annotation_file = "{captions["annotation_file"]}"\n'
            f'image_root = "{captions["image_root"]}"\n'
        )
        from mixture import load_mixture

        with pytest.raises(ValueError, match="only source"):
            stream_source(load_mixture(_datacomp_mixture(tmp_path, shards, coco)))
        with pytest.raises(ValueError, match="does not repeat"):
            stream_source(load_mixture(_datacomp_mixture(tmp_path, shards, "repeat = 2\n")))
        with pytest.raises(ValueError, match="datacomp streams alone"):
            mixture_sources({"source": [{"reader": "datacomp"}]})
