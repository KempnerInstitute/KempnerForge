"""Unit tests for checkpoint format conversion key mapping."""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import convert_checkpoint
from convert_checkpoint import (
    _build_hf_config,
    _hf_to_kf_key,
    _kf_to_hf_key,
    dcp_to_hf,
    hf_to_dcp,
    main,
)

from kempnerforge.config.loader import load_config
from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.transformer import Transformer


class TestKFtoHFKeyMapping:
    def test_embedding(self):
        assert _kf_to_hf_key("token_embedding.embedding.weight") == "model.embed_tokens.weight"

    def test_output_head(self):
        assert _kf_to_hf_key("output_head.proj.weight") == "lm_head.weight"

    def test_final_norm(self):
        assert _kf_to_hf_key("norm.weight") == "model.norm.weight"

    def test_attention_proj(self):
        assert (
            _kf_to_hf_key("layers.0.attention.q_proj.weight")
            == "model.layers.0.self_attn.q_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.5.attention.k_proj.weight")
            == "model.layers.5.self_attn.k_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.31.attention.v_proj.weight")
            == "model.layers.31.self_attn.v_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.31.attention.o_proj.weight")
            == "model.layers.31.self_attn.o_proj.weight"
        )

    def test_attention_norm(self):
        assert (
            _kf_to_hf_key("layers.0.attention_norm.weight")
            == "model.layers.0.input_layernorm.weight"
        )

    def test_mlp_norm(self):
        assert (
            _kf_to_hf_key("layers.0.mlp_norm.weight")
            == "model.layers.0.post_attention_layernorm.weight"
        )

    def test_mlp_projections(self):
        assert (
            _kf_to_hf_key("layers.0.mlp.gate_proj.weight") == "model.layers.0.mlp.gate_proj.weight"
        )
        assert _kf_to_hf_key("layers.0.mlp.up_proj.weight") == "model.layers.0.mlp.up_proj.weight"
        assert (
            _kf_to_hf_key("layers.0.mlp.down_proj.weight") == "model.layers.0.mlp.down_proj.weight"
        )


class TestHFtoKFKeyMapping:
    def test_embedding(self):
        assert _hf_to_kf_key("model.embed_tokens.weight") == "token_embedding.embedding.weight"

    def test_output_head(self):
        assert _hf_to_kf_key("lm_head.weight") == "output_head.proj.weight"

    def test_final_norm(self):
        assert _hf_to_kf_key("model.norm.weight") == "norm.weight"

    def test_attention_proj(self):
        assert (
            _hf_to_kf_key("model.layers.0.self_attn.q_proj.weight")
            == "layers.0.attention.q_proj.weight"
        )

    def test_attention_norm(self):
        assert (
            _hf_to_kf_key("model.layers.0.input_layernorm.weight")
            == "layers.0.attention_norm.weight"
        )

    def test_mlp_norm(self):
        assert (
            _hf_to_kf_key("model.layers.0.post_attention_layernorm.weight")
            == "layers.0.mlp_norm.weight"
        )


class TestRoundTrip:
    @pytest.mark.parametrize(
        "kf_key",
        [
            "token_embedding.embedding.weight",
            "output_head.proj.weight",
            "norm.weight",
            "layers.0.attention.q_proj.weight",
            "layers.0.attention.k_proj.weight",
            "layers.0.attention.v_proj.weight",
            "layers.0.attention.o_proj.weight",
            "layers.0.attention_norm.weight",
            "layers.0.mlp_norm.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "layers.31.attention.q_proj.weight",
        ],
    )
    def test_kf_to_hf_to_kf(self, kf_key):
        assert _hf_to_kf_key(_kf_to_hf_key(kf_key)) == kf_key


class TestMoEKFtoHFKeyMapping:
    def test_router_gate(self):
        assert (
            _kf_to_hf_key("layers.0.mlp.router.gate.weight")
            == "model.layers.0.block_sparse_moe.gate.weight"
        )

    def test_router_expert_bias(self):
        assert _kf_to_hf_key("layers.0.mlp.router.expert_bias") == "model.layers.0.mlp.expert_bias"

    def test_expert_projections(self):
        """Mixtral's w1, w2 and w3 are the gate, down and up projections."""
        assert (
            _kf_to_hf_key("layers.0.mlp.experts.0.gate_proj.weight")
            == "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        )
        assert (
            _kf_to_hf_key("layers.0.mlp.experts.3.up_proj.weight")
            == "model.layers.0.block_sparse_moe.experts.3.w3.weight"
        )
        assert (
            _kf_to_hf_key("layers.0.mlp.experts.7.down_proj.weight")
            == "model.layers.0.block_sparse_moe.experts.7.w2.weight"
        )

    def test_shared_expert(self):
        assert (
            _kf_to_hf_key("layers.0.mlp.shared_expert.gate_proj.weight")
            == "model.layers.0.mlp.shared_experts.gate_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.0.mlp.shared_expert.up_proj.weight")
            == "model.layers.0.mlp.shared_experts.up_proj.weight"
        )

    def test_router_gate_different_layers(self):
        assert (
            _kf_to_hf_key("layers.15.mlp.router.gate.weight")
            == "model.layers.15.block_sparse_moe.gate.weight"
        )


class TestMoEHFtoKFKeyMapping:
    def test_mixtral_router_gate(self):
        assert (
            _hf_to_kf_key("model.layers.3.block_sparse_moe.gate.weight")
            == "layers.3.mlp.router.gate.weight"
        )

    @pytest.mark.parametrize(
        "mixtral,kf", [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]
    )
    def test_mixtral_expert_projections(self, mixtral, kf):
        assert (
            _hf_to_kf_key(f"model.layers.0.block_sparse_moe.experts.5.{mixtral}.weight")
            == f"layers.0.mlp.experts.5.{kf}.weight"
        )

    def test_router_gate(self):
        assert _hf_to_kf_key("model.layers.0.mlp.gate.weight") == "layers.0.mlp.router.gate.weight"

    def test_router_expert_bias(self):
        assert _hf_to_kf_key("model.layers.0.mlp.expert_bias") == "layers.0.mlp.router.expert_bias"

    def test_expert_projections(self):
        assert (
            _hf_to_kf_key("model.layers.0.mlp.experts.0.gate_proj.weight")
            == "layers.0.mlp.experts.0.gate_proj.weight"
        )

    def test_shared_experts(self):
        assert (
            _hf_to_kf_key("model.layers.0.mlp.shared_experts.gate_proj.weight")
            == "layers.0.mlp.shared_expert.gate_proj.weight"
        )

    def test_gate_proj_not_confused_with_router_gate(self):
        """mlp.gate_proj should NOT become mlp.router.gate_proj."""
        assert (
            _hf_to_kf_key("model.layers.0.mlp.gate_proj.weight") == "layers.0.mlp.gate_proj.weight"
        )


class TestMoERoundTrip:
    @pytest.mark.parametrize(
        "kf_key",
        [
            "layers.0.mlp.router.gate.weight",
            "layers.0.mlp.router.expert_bias",
            "layers.0.mlp.experts.0.gate_proj.weight",
            "layers.0.mlp.experts.0.up_proj.weight",
            "layers.0.mlp.experts.0.down_proj.weight",
            "layers.0.mlp.experts.7.gate_proj.weight",
            "layers.0.mlp.shared_expert.gate_proj.weight",
            "layers.0.mlp.shared_expert.up_proj.weight",
            "layers.0.mlp.shared_expert.down_proj.weight",
            "layers.15.mlp.router.gate.weight",
            "layers.15.mlp.experts.3.down_proj.weight",
        ],
    )
    def test_moe_kf_to_hf_to_kf(self, kf_key):
        assert _hf_to_kf_key(_kf_to_hf_key(kf_key)) == kf_key


class TestBuildHFConfig:
    def test_config_fields(self):
        mc = ModelConfig(
            dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=32000,
            max_seq_len=4096,
        )
        hf_cfg = _build_hf_config(mc)

        assert hf_cfg["model_type"] == "llama"
        assert hf_cfg["architectures"] == ["LlamaForCausalLM"]
        assert hf_cfg["hidden_size"] == 4096
        assert hf_cfg["num_hidden_layers"] == 32
        assert hf_cfg["num_attention_heads"] == 32
        assert hf_cfg["num_key_value_heads"] == 8
        assert hf_cfg["vocab_size"] == 32000
        assert hf_cfg["max_position_embeddings"] == 4096
        assert hf_cfg["torch_dtype"] == "bfloat16"

    def test_tie_embeddings_propagated(self):
        mc = ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256, tie_embeddings=True)
        hf_cfg = _build_hf_config(mc)
        assert hf_cfg["tie_word_embeddings"] is True

        mc2 = ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256, tie_embeddings=False)
        hf_cfg2 = _build_hf_config(mc2)
        assert hf_cfg2["tie_word_embeddings"] is False

    def test_moe_config(self):
        mc = ModelConfig(
            dim=512,
            n_layers=8,
            n_heads=8,
            vocab_size=32000,
            num_experts=8,
            moe_top_k=2,
            moe_aux_loss_weight=0.01,
        )
        hf_cfg = _build_hf_config(mc)

        assert hf_cfg["model_type"] == "mixtral"
        assert hf_cfg["architectures"] == ["MixtralForCausalLM"]
        assert hf_cfg["num_local_experts"] == 8
        assert hf_cfg["num_experts_per_tok"] == 2
        assert hf_cfg["router_aux_loss_coef"] == 0.01

    def test_moe_intermediate_size_is_the_expert_width(self):
        """Mixtral reads intermediate_size as each expert's width; a dense model keeps the
        dense width."""
        mc = ModelConfig(
            dim=64,
            n_layers=2,
            n_heads=2,
            vocab_size=256,
            num_experts=4,
            moe_expert_ffn_multiplier=0.5,
        )
        assert mc.computed_expert_ffn_hidden_dim != mc.computed_ffn_hidden_dim
        assert _build_hf_config(mc)["intermediate_size"] == mc.computed_expert_ffn_hidden_dim
        dense = replace(mc, num_experts=0)
        assert _build_hf_config(dense)["intermediate_size"] == mc.computed_ffn_hidden_dim

    def test_head_dim_is_exported(self):
        """A decoupled width is not recoverable from dim // n_heads, so the
        export must carry it explicitly."""
        coupled = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256)
        assert _build_hf_config(coupled)["head_dim"] == 16

        decoupled = ModelConfig(
            dim=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim_override=32, vocab_size=256
        )
        hf_cfg = _build_hf_config(decoupled)
        assert hf_cfg["head_dim"] == 32 != decoupled.dim // decoupled.n_heads

    def test_moe_head_dim_is_exported(self):
        mc = ModelConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim_override=32,
            vocab_size=256,
            num_experts=4,
            moe_top_k=2,
        )
        assert _build_hf_config(mc)["head_dim"] == 32

    def test_dim_indivisible_by_n_heads_is_rejected(self):
        """``head_dim_override`` lifts that requirement for training, but the
        target architectures reject it even with an explicit head_dim."""
        mc = ModelConfig(
            dim=100, n_layers=2, n_heads=8, n_kv_heads=8, head_dim_override=16, vocab_size=256
        )
        with pytest.raises(ValueError, match="not divisible by n_heads"):
            _build_hf_config(mc)

    @pytest.mark.parametrize("override", [0, 32])
    def test_exported_config_rebuilds_the_attention_shapes(self, override):
        """The end the export exists for: a model built from config.json must
        take the checkpoint's attention weights."""
        import torch

        transformers = pytest.importorskip("transformers")

        from kempnerforge.model.transformer import Transformer

        mc = ModelConfig(
            dim=64,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            head_dim_override=override,
            vocab_size=256,
            max_seq_len=64,
        )
        hf_cfg = _build_hf_config(mc)
        cfg = transformers.LlamaConfig(
            hidden_size=hf_cfg["hidden_size"],
            num_attention_heads=hf_cfg["num_attention_heads"],
            num_key_value_heads=hf_cfg["num_key_value_heads"],
            head_dim=hf_cfg["head_dim"],
            intermediate_size=hf_cfg["intermediate_size"],
            num_hidden_layers=hf_cfg["num_hidden_layers"],
            vocab_size=hf_cfg["vocab_size"],
            max_position_embeddings=hf_cfg["max_position_embeddings"],
        )
        with torch.device("meta"):
            kf = Transformer(mc).state_dict()
            hf = transformers.LlamaForCausalLM(cfg).state_dict()
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert (
                hf[f"model.layers.0.self_attn.{proj}.weight"].shape
                == kf[f"layers.0.attention.{proj}.weight"].shape
            ), proj

    def test_dense_has_no_moe_fields(self):
        mc = ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256)
        hf_cfg = _build_hf_config(mc)
        assert "num_local_experts" not in hf_cfg
        assert hf_cfg["model_type"] == "llama"


def _tiny_dcp(path: Path, model_config: ModelConfig) -> None:
    """Save a freshly initialized model as a single-process DCP checkpoint."""
    dcp.save({"model": Transformer(model_config).state_dict()}, checkpoint_id=str(path))


def _written(hf_dir: Path) -> tuple[set[torch.dtype], str]:
    """The dtypes of an export's tensors and the ``torch_dtype`` its config.json names."""
    weights = load_file(hf_dir / "model.safetensors")
    hf_cfg = json.loads((hf_dir / "config.json").read_text())
    return {t.dtype for t in weights.values()}, hf_cfg["torch_dtype"]


class TestExportDtypeMetadata:
    @pytest.mark.parametrize(
        ("dtype", "name"),
        [(torch.float32, "float32"), (torch.bfloat16, "bfloat16"), (torch.float16, "float16")],
    )
    def test_config_json_names_the_written_dtype(self, tmp_path, tiny_model_config, dtype, name):
        """torch_dtype names the dtype the tensors were written in, not a fixed bfloat16."""
        _tiny_dcp(tmp_path / "dcp", tiny_model_config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), tiny_model_config, dtype=dtype)
        assert _written(tmp_path / "hf") == ({dtype}, name)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_transformers_loads_the_written_dtype(self, tmp_path, tiny_model_config, dtype):
        """``from_pretrained(dtype="auto")`` follows torch_dtype, so a fixed
        bfloat16 there would load these weights as bf16."""
        transformers = pytest.importorskip("transformers")
        _tiny_dcp(tmp_path / "dcp", tiny_model_config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), tiny_model_config, dtype=dtype)
        model = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "hf", dtype="auto")
        assert {p.dtype for p in model.parameters()} == {dtype}


_TINY_TOML = """\
[model]
dim = 64
n_layers = 2
n_heads = 2
n_kv_heads = 2
vocab_size = 256
max_seq_len = 64
"""


def _run_dcp_to_hf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export_dtype: str | None, flag: str | None
) -> tuple[set[torch.dtype], str]:
    """Export a tiny checkpoint through the CLI and return what it wrote.

    ``export_dtype`` goes in the TOML's ``[checkpoint]`` table and ``flag`` is
    passed as ``--dtype``; ``None`` leaves either out.
    """
    toml = tmp_path / "job.toml"
    table = "" if export_dtype is None else f'[checkpoint]\nexport_dtype = "{export_dtype}"\n'
    toml.write_text(_TINY_TOML + table)
    _tiny_dcp(tmp_path / "dcp", load_config(toml, cli_args=[]).model)
    argv = ["convert_checkpoint.py", "dcp-to-hf", "--config", str(toml)]
    argv += ["--dcp-dir", str(tmp_path / "dcp"), "--hf-dir", str(tmp_path / "hf")]
    if flag is not None:
        argv += ["--dtype", flag]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    return _written(tmp_path / "hf")


class TestExportDtypePrecedence:
    """dcp-to-hf takes ``--dtype``, then ``checkpoint.export_dtype``, then bfloat16."""

    def test_flag_alone_sets_the_dtype(self, tmp_path, monkeypatch):
        written = _run_dcp_to_hf(tmp_path, monkeypatch, None, "float16")
        assert written == ({torch.float16}, "float16")

    def test_config_alone_sets_the_dtype(self, tmp_path, monkeypatch):
        written = _run_dcp_to_hf(tmp_path, monkeypatch, "float32", None)
        assert written == ({torch.float32}, "float32")

    @pytest.mark.parametrize(
        ("flag", "dtype"), [("bfloat16", torch.bfloat16), ("float16", torch.float16)]
    )
    def test_flag_overrides_the_config(self, tmp_path, monkeypatch, flag, dtype):
        """Even a flag that names the default wins over the TOML."""
        written = _run_dcp_to_hf(tmp_path, monkeypatch, "float32", flag)
        assert written == ({dtype}, flag)

    def test_bfloat16_without_flag_or_config(self, tmp_path, monkeypatch):
        written = _run_dcp_to_hf(tmp_path, monkeypatch, None, None)
        assert written == ({torch.bfloat16}, "bfloat16")

    @pytest.mark.parametrize(
        ("export_dtype", "flag", "source"),
        [(None, "float32", "--dtype"), ("float32", None, "checkpoint.export_dtype")],
    )
    def test_log_names_where_the_dtype_came_from(
        self, tmp_path, monkeypatch, caplog, export_dtype, flag, source
    ):
        """The log line says whether the flag or the config chose the export dtype."""
        with caplog.at_level(logging.INFO, logger="convert_checkpoint"):
            _run_dcp_to_hf(tmp_path, monkeypatch, export_dtype, flag)
        assert f"Export dtype: float32 (from {source})" in caplog.text


def _random_dcp(path: Path, model_config: ModelConfig, std: float = 1.0) -> Transformer:
    """Save a model whose every weight, norms included, is drawn at random; return it."""
    model = Transformer(model_config)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(std=std, generator=generator)
    dcp.save({"model": model.state_dict()}, checkpoint_id=str(path))
    return model


class TestTiedExport:
    """A model with tied embeddings exports at every dtype, float32 included."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_head_is_written_as_a_copy_of_the_embedding(self, tmp_path, tiny_model_config, dtype):
        """At float32 the cast leaves the tied head on the embedding's storage, which
        safetensors refuses, so the export writes the head as its own copy."""
        config = replace(tiny_model_config, tie_embeddings=True)
        source = _random_dcp(tmp_path / "dcp", config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), config, dtype=dtype)
        weights = load_file(tmp_path / "hf" / "model.safetensors")
        embedding = source.state_dict()["token_embedding.embedding.weight"].to(dtype)
        assert torch.equal(weights["model.embed_tokens.weight"], embedding)
        assert torch.equal(weights["lm_head.weight"], embedding)
        assert json.loads((tmp_path / "hf" / "config.json").read_text())["tie_word_embeddings"]

    def test_transformers_loads_a_float32_export_tied(self, tmp_path, tiny_model_config):
        """transformers ties the copied head back to the embedding, at float32."""
        transformers = pytest.importorskip("transformers")
        config = replace(tiny_model_config, tie_embeddings=True)
        source = _random_dcp(tmp_path / "dcp", config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), config, dtype=torch.float32)
        model = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "hf", dtype="auto")
        head, embedding = model.get_output_embeddings().weight, model.get_input_embeddings().weight
        assert head is embedding
        assert embedding.dtype == torch.float32
        assert torch.equal(embedding, source.state_dict()["token_embedding.embedding.weight"])


def _llama_weights(model: Transformer) -> dict[str, torch.Tensor]:
    """A dense model's weights under the names ``LlamaForCausalLM`` loads them by."""
    assert model.token_embedding is not None and model.output_head is not None
    weights = {
        "model.embed_tokens.weight": model.token_embedding.embedding.weight,
        "model.norm.weight": model.norm.weight,
        "lm_head.weight": model.output_head.proj.weight,
    }
    for i, block in model.layers.items():
        prefix = f"model.layers.{i}"
        weights[f"{prefix}.input_layernorm.weight"] = block.get_parameter("attention_norm.weight")
        weights[f"{prefix}.post_attention_layernorm.weight"] = block.get_parameter(
            "mlp_norm.weight"
        )
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{prefix}.self_attn.{proj}.weight"] = getattr(block.attention, proj).weight
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weights[f"{prefix}.mlp.{proj}.weight"] = getattr(block.mlp, proj).weight
    return weights


class TestExportValues:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_exported_tensors_are_the_source_weights(self, tmp_path, tiny_model_config, dtype):
        """Each exported tensor is the source weight its name stands for, cast to the
        export dtype, so a key mapping that swaps or drops weights fails here even
        though every dtype comes out right."""
        source = _random_dcp(tmp_path / "dcp", tiny_model_config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), tiny_model_config, dtype=dtype)
        weights = load_file(tmp_path / "hf" / "model.safetensors")
        expected = _llama_weights(source)
        assert weights.keys() == expected.keys()
        for name, tensor in expected.items():
            assert torch.equal(weights[name], tensor.detach().to(dtype)), name


def _moe_config(**overrides) -> ModelConfig:
    """A small MoE model that differs from Mixtral's defaults where an export could slip:
    grouped-query attention, an expert width other than the dense width, and rope_theta."""
    fields = {
        "dim": 64,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "vocab_size": 256,
        "max_seq_len": 64,
        "num_experts": 4,
        "moe_top_k": 2,
        "moe_expert_ffn_multiplier": 1.5,
        "rope_theta": 10000.0,
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def _read_back(hf_dir: Path, dcp_dir: Path, config: ModelConfig) -> dict[str, torch.Tensor]:
    """Import an export with ``hf_to_dcp`` and return the model state it saved."""
    hf_to_dcp(str(hf_dir), str(dcp_dir), config)
    state = {"model": Transformer(config).state_dict()}
    dcp.load(state, checkpoint_id=str(dcp_dir))
    return state["model"]


class TestMixtralExport:
    """An MoE model exports as a Mixtral checkpoint, which transformers loads as trained."""

    @pytest.mark.parametrize("packed", [False, True])
    def test_transformers_loads_every_weight_and_matches_the_logits(self, tmp_path, packed):
        transformers = pytest.importorskip("transformers")
        config = _moe_config(moe_packed_experts=packed)
        source = _random_dcp(tmp_path / "dcp", config, std=0.05).eval()
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), config, dtype=torch.float32)
        model, info = transformers.AutoModelForCausalLM.from_pretrained(
            tmp_path / "hf", dtype=torch.float32, output_loading_info=True
        )
        assert type(model).__name__ == "MixtralForCausalLM"
        assert not info["missing_keys"]
        assert not info["unexpected_keys"]
        assert not info["mismatched_keys"]
        tokens = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            torch.testing.assert_close(model.eval()(tokens).logits, source(tokens))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_packed_experts_are_written_one_tensor_per_expert(self, tmp_path, dtype):
        """A packed model writes the names an unpacked one does, each expert's slice of the
        packed weight transposed to the (out, in) layout of a linear layer."""
        config = _moe_config(moe_packed_experts=True)
        source = _random_dcp(tmp_path / "dcp", config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), config, dtype=dtype)
        weights = load_file(tmp_path / "hf" / "model.safetensors")
        unpacked = Transformer(replace(config, moe_packed_experts=False)).state_dict()
        assert weights.keys() == {_kf_to_hf_key(name) for name in unpacked}
        for i, block in source.layers.items():
            for proj, mixtral in (("gate", "w1"), ("down", "w2"), ("up", "w3")):
                packed = block.mlp.get_parameter(f"{proj}_w")
                for j in range(config.num_experts):
                    name = f"model.layers.{i}.block_sparse_moe.experts.{j}.{mixtral}.weight"
                    assert torch.equal(weights[name], packed[j].t().to(dtype)), name

    @pytest.mark.parametrize("packed", [False, True])
    def test_hf_to_dcp_reads_an_export_back(self, tmp_path, packed):
        config = _moe_config(moe_packed_experts=packed)
        source = _random_dcp(tmp_path / "dcp", config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), config, dtype=torch.float32)
        back = _read_back(tmp_path / "hf", tmp_path / "back", config)
        for name, tensor in source.state_dict().items():
            assert torch.equal(back[name], tensor), name

    def test_hf_to_dcp_reads_the_expert_names_written_before(self, tmp_path):
        """An export written with mlp.gate and mlp.experts.{j}.*_proj still imports."""
        config = _moe_config()
        source = _random_dcp(tmp_path / "dcp", config)

        def written_before(name: str) -> str:
            if ".mlp.experts." in name:
                return "model." + name
            return _kf_to_hf_key(name).replace(".block_sparse_moe.gate.", ".mlp.gate.")

        (tmp_path / "hf").mkdir()
        old = {written_before(name): tensor for name, tensor in source.state_dict().items()}
        assert any(".mlp.experts.0.gate_proj." in name for name in old)
        save_file(old, tmp_path / "hf" / "model.safetensors")
        back = _read_back(tmp_path / "hf", tmp_path / "back", config)
        for name, tensor in source.state_dict().items():
            assert torch.equal(back[name], tensor), name

    @pytest.mark.parametrize(
        "overrides,named",
        [
            ({"moe_shared_experts": 1}, "moe_shared_experts=1: Mixtral has none"),
            ({"moe_frequency": 2}, "moe_frequency=2: every Mixtral layer is an MoE layer"),
            (
                {"moe_router": "sigmoid_topk"},
                "moe_router='sigmoid_topk': Mixtral routes with softmax top-k",
            ),
            ({"activation": "gelu"}, "activation='gelu': Mixtral experts are SwiGLU"),
            ({"norm_type": "layernorm"}, "norm_type='layernorm': Mixtral norms are RMSNorm"),
            ({"qk_norm": True}, "qk_norm=True: Mixtral has no query or key norm"),
        ],
    )
    def test_refuses_what_mixtral_cannot_hold_before_reading(self, tmp_path, overrides, named):
        """The refusal names the setting before the checkpoint is read, here one that does
        not exist, or anything is written."""
        config = _moe_config(**overrides)
        with pytest.raises(ValueError, match=re.escape(named)):
            dcp_to_hf(str(tmp_path / "missing"), str(tmp_path / "hf"), config)
        assert not (tmp_path / "hf").exists()

    def test_hf_to_dcp_reads_the_experts_a_loaded_mixtral_packs(self, tmp_path, monkeypatch):
        """A hub ID goes through a loaded model's state dict, where transformers 5 packs each
        layer's experts; the import splits them back into every expert's weights."""
        transformers = pytest.importorskip("transformers")
        config = _moe_config()
        source = _random_dcp(tmp_path / "dcp", config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), config, dtype=torch.float32)
        loaded = transformers.AutoModelForCausalLM.from_pretrained(
            tmp_path / "hf", dtype=torch.float32
        ).state_dict()
        assert "model.layers.0.mlp.experts.gate_up_proj" in loaded
        monkeypatch.setattr(convert_checkpoint, "_load_hf_hub", lambda model_id: loaded)
        back = _read_back(tmp_path / "some-org-mixtral", tmp_path / "back", config)
        for name, tensor in source.state_dict().items():
            assert torch.equal(back[name], tensor), name

    def test_capacity_warns_that_transformers_drops_no_tokens(self, tmp_path, caplog):
        config = _moe_config(moe_capacity_factor=1.25)
        _random_dcp(tmp_path / "dcp", config)
        with caplog.at_level(logging.WARNING, logger="convert_checkpoint"):
            dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), config, dtype=torch.float32)
        assert "moe_capacity_factor=1.25: transformers has no expert capacity" in caplog.text
        assert (tmp_path / "hf" / "model.safetensors").exists()

    def test_every_refused_setting_is_named(self):
        config = _moe_config(moe_shared_experts=2, moe_router="sigmoid_topk")
        with pytest.raises(ValueError, match="moe_shared_experts=2.*; moe_router='sigmoid_topk'"):
            _build_hf_config(config)

    def test_without_the_check_a_model_exports_for_hf_to_dcp(self, tmp_path, caplog):
        """A DeepSeek-style model exports with a warning that names what Mixtral lacks, and
        hf-to-dcp reads every weight back, as the resharding round trip needs."""
        config = _moe_config(moe_router="sigmoid_topk", moe_shared_experts=1, moe_frequency=2)
        source = _random_dcp(tmp_path / "dcp", config)
        with caplog.at_level(logging.WARNING, logger="convert_checkpoint"):
            dcp_to_hf(
                str(tmp_path / "dcp"),
                str(tmp_path / "hf"),
                config,
                dtype=torch.float32,
                check_architecture=False,
            )
        assert "moe_shared_experts=1: Mixtral has none" in caplog.text
        assert "read it only with hf-to-dcp" in caplog.text
        back = _read_back(tmp_path / "hf", tmp_path / "back", config)
        for name, tensor in source.state_dict().items():
            assert torch.equal(back[name], tensor), name

    def test_the_cli_flag_turns_the_check_off(self, tmp_path, monkeypatch):
        toml = tmp_path / "job.toml"
        toml.write_text(_TINY_TOML + "num_experts = 2\nmoe_shared_experts = 1\n")
        _tiny_dcp(tmp_path / "dcp", load_config(toml, cli_args=[]).model)
        argv = ["convert_checkpoint.py", "dcp-to-hf", "--config", str(toml)]
        argv += ["--dcp-dir", str(tmp_path / "dcp"), "--hf-dir", str(tmp_path / "hf")]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(ValueError, match="pass --no-architecture-check"):
            main()
        monkeypatch.setattr(sys, "argv", [*argv, "--no-architecture-check"])
        main()
        assert (tmp_path / "hf" / "model.safetensors").exists()


class TestLlamaExport:
    """A dense model exports as Llama, and one that Llama cannot hold is refused."""

    @pytest.mark.parametrize(
        "overrides,named",
        [
            ({"qk_norm": True}, "qk_norm=True: Llama has no query or key norm"),
            ({"norm_type": "layernorm"}, "norm_type='layernorm': Llama norms are RMSNorm"),
            ({"activation": "gelu"}, "activation='gelu': Llama MLPs are SwiGLU"),
            ({"activation": "relu"}, "activation='relu': Llama MLPs are SwiGLU"),
        ],
    )
    def test_refuses_what_llama_cannot_hold_before_reading(
        self, tmp_path, tiny_model_config, overrides, named
    ):
        """The refusal names the setting before the checkpoint is read, here one that does
        not exist, or anything is written."""
        config = replace(tiny_model_config, **overrides)
        with pytest.raises(
            ValueError, match="Cannot export this model as Llama: " + re.escape(named)
        ):
            dcp_to_hf(str(tmp_path / "missing"), str(tmp_path / "hf"), config)
        assert not (tmp_path / "hf").exists()

    def test_without_the_check_a_dense_model_exports_for_hf_to_dcp(
        self, tmp_path, tiny_model_config, caplog
    ):
        """A model with a query and key norm and LayerNorm exports with a warning, and
        hf-to-dcp reads every weight back."""
        config = replace(tiny_model_config, qk_norm=True, norm_type="layernorm")
        source = _random_dcp(tmp_path / "dcp", config)
        with caplog.at_level(logging.WARNING, logger="convert_checkpoint"):
            dcp_to_hf(
                str(tmp_path / "dcp"),
                str(tmp_path / "hf"),
                config,
                dtype=torch.float32,
                check_architecture=False,
            )
        assert "Exporting a model that Llama cannot hold" in caplog.text
        assert "qk_norm=True: Llama has no query or key norm" in caplog.text
        back = _read_back(tmp_path / "hf", tmp_path / "back", config)
        for name, tensor in source.state_dict().items():
            assert torch.equal(back[name], tensor), name
