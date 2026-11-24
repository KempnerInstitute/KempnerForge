"""Tests for building the packed loader of a mixture of image-text sources."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from mixture import check_mixture, check_run, finevision_subsets, mixture_data_fn, mixture_sources
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
batch_size = 1
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

    @pytest.mark.parametrize(
        ("mixture", "message"),
        [
            ({"packng": {"view_groups": True}}, "unknown mixture table"),
            ({"render": {"answer_onlyy": True}}, r"\[render\]: unknown option"),
            ({"render": {"max_images": 1}}, "set from the run config"),
            ({"packing": {"view_group": True}}, r"\[packing\]: unknown option"),
            ({"source": [{"reader": "vqav2", "annotation_file": "a"}]}, "missing option"),
            (
                {"source": [{"reader": "vqav2", "annotation_file": "a", "image_rot": "b"}]},
                r"unknown option\(s\) \['image_rot'\]",
            ),
            ({"source": [{"reader": "finevision"}]}, r"missing option\(s\) \['root'\]"),
            (
                {"source": [{"reader": "finevision", "root": "r", "max_image": 1}]},
                r"unknown option\(s\) \['max_image'\]",
            ),
            ({"source": [{"reader": "finevision", "root": "r", "subset_dir": "s"}]}, "unknown option"),
        ],
    )
    def test_a_mistaken_mixture_is_refused_before_reading(
        self, mixture: dict[str, Any], message: str
    ) -> None:
        mixture = {"source": [{"reader": "vqav2", "annotation_file": "a", "image_root": "b"}]} | mixture
        with pytest.raises(ValueError, match=message):
            check_mixture(mixture)

    def test_the_mixture_file_is_checked_when_the_data_fn_is_made(self, tmp_path: Path) -> None:
        path = tmp_path / "mixture.toml"
        path.write_text('[packng]\nview_groups = true\n[[source]]\nreader = "vqav2"\n')
        with pytest.raises(ValueError, match="unknown mixture table"):
            mixture_data_fn(path)

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ("--train.batch_size=2", "batch_size = 1"),
            ("--data.pack_sequences=false", "pack_sequences"),
        ],
    )
    def test_a_run_the_mixture_cannot_train_is_refused(
        self, tmp_path: Path, override: str, message: str
    ) -> None:
        config_path = tmp_path / "run.toml"
        config_path.write_text(CONFIG)
        check_run(load_config(str(config_path), cli_args=[]))
        with pytest.raises(ValueError, match=message):
            check_run(load_config(str(config_path), cli_args=[override]))

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
