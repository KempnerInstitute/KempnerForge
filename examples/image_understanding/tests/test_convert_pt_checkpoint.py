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


NAMED = {
    "adapter.norm.weight": "adapter.proj.ln_q.weight",
    "adapter.proj.0.weight": "adapter.proj.proj1.weight",
    "adapter.proj.0.bias": "adapter.proj.proj1.bias",
    "adapter.proj.2.weight": "adapter.proj.proj2.weight",
    "adapter.proj.2.bias": "adapter.proj.proj2.bias",
    "multimodal_core.norm.weight": "transformer.norm.weight",
    "text_preprocessor.embed.weight": "transformer.token_embedding.embedding.weight",
    "text_head.weight": "transformer.output_head.proj.weight",
}
OTHER_ADAPTER = {
    "adapter.model.ln_q.weight": "adapter.norm.weight",
    "adapter.model.mlp.0.weight": "adapter.proj.0.weight",
    "adapter.model.mlp.0.bias": "adapter.proj.0.bias",
    "adapter.model.mlp.2.weight": "adapter.proj.2.weight",
    "adapter.model.mlp.2.bias": "adapter.proj.2.bias",
}
LAYER = {
    "self_attn.q_proj.weight": "attention.q_proj.weight",
    "self_attn.k_proj.weight": "attention.k_proj.weight",
    "self_attn.v_proj.weight": "attention.v_proj.weight",
    "self_attn.o_proj.weight": "attention.o_proj.weight",
    "self_attn.q_norm.weight": "attention.q_norm.weight",
    "self_attn.k_norm.weight": "attention.k_norm.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "mlp_norm.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
}
VISION = ("image_encoder.model.vision_model.", "vision_encoder.vision_tower.")


def _target_key(source: str) -> str:
    """Where a source tensor belongs, spelled out here rather than read from the converter."""
    source = OTHER_ADAPTER.get(source, source)
    if source in NAMED:
        return NAMED[source]
    if source.startswith(VISION[0]):
        return VISION[1] + source[len(VISION[0]) :]
    _, _, layer, *leaf = source.split(".")
    return f"transformer.layers.{layer}.{LAYER['.'.join(leaf)]}"


def _source_key(target: str) -> str:
    named = {v: k for k, v in NAMED.items()}
    if target in named:
        return named[target]
    if target.startswith(VISION[1]):
        return VISION[0] + target[len(VISION[1]) :]
    _, _, layer, *leaf = target.split(".")
    leaves = {v: k for k, v in LAYER.items()}
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
        _source_key(key): torch.randn(value.shape, generator=generator)
        for key, value in _target(config).items()
    }


def _converted(config: Path, out: Path) -> dict[str, torch.Tensor]:
    state = {key: torch.empty_like(value) for key, value in _target(config).items()}
    dcp.load({"model": state}, checkpoint_id=str(out))
    return state


class TestConvertPtCheckpoint:
    def test_every_tensor_reaches_its_place(self, tmp_path: Path, config: Path) -> None:
        converter = _converter()
        source = _source(converter, config)
        assert "text_head.weight" in source and "adapter.norm.weight" in source
        torch.save(source, tmp_path / "stage1.pt")
        out = converter.convert(str(tmp_path / "stage1.pt"), str(config), str(tmp_path / "dcp"))
        state = _converted(config, out)
        assert len(state) == len(source)
        for key, tensor in source.items():
            assert torch.equal(state[_target_key(key)], tensor), key

    def test_the_other_adapter_names_convert_alike(self, tmp_path: Path, config: Path) -> None:
        converter = _converter()
        source = _source(converter, config)
        renamed = {v: k for k, v in OTHER_ADAPTER.items()}
        source = {renamed.get(key, key): tensor for key, tensor in source.items()}
        assert "adapter.model.mlp.2.bias" in source and "adapter.proj.2.bias" not in source
        torch.save(source, tmp_path / "stage1.pt")
        out = converter.convert(str(tmp_path / "stage1.pt"), str(config), str(tmp_path / "dcp"))
        state = _converted(config, out)
        for key, tensor in source.items():
            assert torch.equal(state[_target_key(key)], tensor), key

    def test_both_adapter_names_at_once_are_refused(self, tmp_path: Path, config: Path) -> None:
        converter = _converter()
        source = _source(converter, config)
        source["adapter.model.ln_q.weight"] = source["adapter.norm.weight"].clone()
        torch.save(source, tmp_path / "stage1.pt")
        with pytest.raises(ValueError, match="two source keys map to adapter.proj.ln_q.weight"):
            converter.convert(str(tmp_path / "stage1.pt"), str(config), str(tmp_path / "dcp"))

    def test_main_converts_from_the_command_line(self, tmp_path: Path, config: Path) -> None:
        converter = _converter()
        source = _source(converter, config)
        torch.save(source, tmp_path / "stage1.pt")
        converter.main(
            [
                "--checkpoint",
                str(tmp_path / "stage1.pt"),
                "--config",
                str(config),
                "--out",
                str(tmp_path / "dcp"),
            ]
        )
        state = _converted(config, tmp_path / "dcp")
        assert torch.equal(state["transformer.output_head.proj.weight"], source["text_head.weight"])

    def test_a_tied_target_is_refused(self, tmp_path: Path, config: Path) -> None:
        converter = _converter()
        tied = tmp_path / "tied.toml"
        text = config.read_text()
        tied.write_text(text.replace("tie_embeddings = false", "tie_embeddings = true"))
        with pytest.raises(ValueError, match="ties its embeddings"):
            converter.convert(str(tmp_path / "missing.pt"), str(tied), str(tmp_path / "dcp"))

    def test_a_config_without_a_vlm_is_refused(self, tmp_path: Path) -> None:
        converter = _converter()
        text = tmp_path / "text.toml"
        text.write_text("[train]\nmax_steps = 1\n")
        with pytest.raises(ValueError, match="does not describe a VLM"):
            converter.convert(str(tmp_path / "missing.pt"), str(text), str(tmp_path / "dcp"))

    def test_a_source_that_is_not_a_flat_state_dict_is_refused(
        self, tmp_path: Path, config: Path
    ) -> None:
        converter = _converter()
        torch.save({"model": {"a": torch.zeros(1)}}, tmp_path / "nested.pt")
        with pytest.raises(ValueError, match="not a flat state dict"):
            converter.convert(str(tmp_path / "nested.pt"), str(config), str(tmp_path / "dcp"))

    def test_a_pooling_head_the_model_lacks_is_dropped(self, tmp_path: Path) -> None:
        converter = _converter()
        source = _source(converter, _config(tmp_path, use_head=True))
        headless = _config(tmp_path, use_head=False)
        head = [key for key in source if ".vision_model.head." in key]
        assert head and not any(".head." in key for key in _target(headless))
        torch.save(source, tmp_path / "stage1.pt")
        out = converter.convert(str(tmp_path / "stage1.pt"), str(headless), str(tmp_path / "dcp"))
        state = _converted(headless, out)
        for key, tensor in source.items():
            if key not in head:
                assert torch.equal(state[_target_key(key)], tensor), key

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
