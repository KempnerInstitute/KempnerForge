"""Tests for writing a ``.pt`` joint-decoder checkpoint as a DCP to warm-start from."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed.checkpoint as dcp

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_pt_checkpoint.py"

CONFIG = """
[model]
dim = 32
n_layers = 2
n_heads = 2
n_kv_heads = 1
head_dim_override = 16
vocab_size = 64
ffn_hidden_dim = 64
norm_type = "rmsnorm"
norm_eps = 1e-6
activation = "silu"
qk_norm = true
tie_embeddings = false
max_seq_len = 1024
[vision_encoder]
type = "siglip2"
path = "{encoder}"
[adapter]
type = "avgpool"
pool_window = 2
hidden_dim = 24
activation = "gelu"
pre_norm = "rmsnorm"
[vlm]
arch = "joint_decoder"
"""


def _converter() -> Any:
    spec = importlib.util.spec_from_file_location("convert_pt_checkpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_key(converter: Any, target: str) -> str:
    renames = {v: k for k, v in converter._RENAMES.items()}
    if target in renames:
        return renames[target]
    source, prefix = converter._VISION_PREFIX
    if target.startswith(prefix):
        return source + target[len(prefix) :]
    _, _, layer, *leaf = target.split(".")
    leaves = {v: k for k, v in converter._LAYER_LEAVES.items()}
    return f"multimodal_core.layers.{layer}.{leaves['.'.join(leaf)]}"


def _config(tmp_path: Path, *, use_head: bool) -> Path:
    from transformers import SiglipVisionConfig, SiglipVisionModel

    encoder = tmp_path / f"siglip-{use_head}"
    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=4,
        vision_use_head=use_head,
    )
    SiglipVisionModel(vision).save_pretrained(encoder)
    path = tmp_path / f"run-{use_head}.toml"
    path.write_text(CONFIG.format(encoder=encoder))
    return path


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return _config(tmp_path, use_head=True)


def _target(config: Path) -> dict[str, torch.Tensor]:
    from kempnerforge.config.loader import load_config
    from kempnerforge.model.vlm import build_vlm_wrapper

    job = load_config(str(config), cli_args=[])
    model = build_vlm_wrapper(job.model, job.vision_encoder, job.adapter, job.vlm)
    return model.state_dict()


def _source(converter: Any, config: Path) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        _source_key(converter, key): torch.randn(value.shape, generator=generator)
        for key, value in _target(config).items()
    }


class TestConvertPtCheckpoint:
    def test_every_tensor_reaches_its_place(self, tmp_path: Path, config: Path) -> None:
        converter = _converter()
        source = _source(converter, config)
        assert "text_head.weight" in source and "adapter.norm.weight" in source
        torch.save(source, tmp_path / "stage1.pt")
        out = converter.convert(str(tmp_path / "stage1.pt"), str(config), str(tmp_path / "dcp"))
        state = {key: torch.empty_like(value) for key, value in _target(config).items()}
        dcp.load({"model": state}, checkpoint_id=str(out))
        for key, tensor in source.items():
            assert torch.equal(state[converter.map_key(key)], tensor), key

    def test_a_pooling_head_the_model_lacks_is_dropped(self, tmp_path: Path) -> None:
        converter = _converter()
        source = _source(converter, _config(tmp_path, use_head=True))
        headless = _config(tmp_path, use_head=False)
        head = [key for key in source if ".vision_model.head." in key]
        assert head and not any(".head." in key for key in _target(headless))
        torch.save(source, tmp_path / "stage1.pt")
        out = converter.convert(str(tmp_path / "stage1.pt"), str(headless), str(tmp_path / "dcp"))
        state = {key: torch.empty_like(value) for key, value in _target(headless).items()}
        dcp.load({"model": state}, checkpoint_id=str(out))
        for key, tensor in source.items():
            if key not in head:
                assert torch.equal(state[converter.map_key(key)], tensor), key

    def test_a_source_key_without_a_target_is_refused(self, tmp_path: Path, config: Path) -> None:
        converter = _converter()
        source = _source(converter, config)
        source["multimodal_core.layers.0.self_attn.rotary.inv_freq"] = torch.zeros(4)
        torch.save(source, tmp_path / "stage1.pt")
        with pytest.raises(ValueError, match="have no target"):
            converter.convert(str(tmp_path / "stage1.pt"), str(config), str(tmp_path / "dcp"))

    def test_a_missing_tensor_is_refused(self, tmp_path: Path, config: Path) -> None:
        converter = _converter()
        source = _source(converter, config)
        del source["text_head.weight"]
        torch.save(source, tmp_path / "stage1.pt")
        with pytest.raises(ValueError, match="1 model tensors missing"):
            converter.convert(str(tmp_path / "stage1.pt"), str(config), str(tmp_path / "dcp"))
        assert not (tmp_path / "dcp").exists()

    def test_a_shape_mismatch_is_refused(self, tmp_path: Path, config: Path) -> None:
        converter = _converter()
        source = _source(converter, config)
        source["adapter.proj.2.bias"] = torch.zeros(7)
        torch.save(source, tmp_path / "stage1.pt")
        with pytest.raises(ValueError, match="shape mismatch"):
            converter.convert(str(tmp_path / "stage1.pt"), str(config), str(tmp_path / "dcp"))

    def test_an_occupied_output_is_refused(self, tmp_path: Path, config: Path) -> None:
        converter = _converter()
        (tmp_path / "dcp").mkdir()
        (tmp_path / "dcp" / "file").write_text("x")
        with pytest.raises(FileExistsError):
            converter.convert(str(tmp_path / "missing.pt"), str(config), str(tmp_path / "dcp"))
