"""Unit tests for checkpoint format conversion key mapping."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

# Add scripts/ to path so we can import the conversion module
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from convert_checkpoint import _build_hf_config, _hf_to_kf_key, _kf_to_hf_key, dcp_to_hf, main

from kempnerforge.config.loader import load_config
from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.transformer import Transformer

# ---------------------------------------------------------------------------
# Key mapping: KempnerForge → HuggingFace
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Key mapping: HuggingFace → KempnerForge
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Round-trip: KF → HF → KF should be identity
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# HuggingFace config generation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MoE key mapping: KempnerForge → HuggingFace
# ---------------------------------------------------------------------------


class TestMoEKFtoHFKeyMapping:
    def test_router_gate(self):
        assert _kf_to_hf_key("layers.0.mlp.router.gate.weight") == "model.layers.0.mlp.gate.weight"

    def test_router_expert_bias(self):
        assert _kf_to_hf_key("layers.0.mlp.router.expert_bias") == "model.layers.0.mlp.expert_bias"

    def test_expert_projections(self):
        assert (
            _kf_to_hf_key("layers.0.mlp.experts.0.gate_proj.weight")
            == "model.layers.0.mlp.experts.0.gate_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.0.mlp.experts.3.up_proj.weight")
            == "model.layers.0.mlp.experts.3.up_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.0.mlp.experts.7.down_proj.weight")
            == "model.layers.0.mlp.experts.7.down_proj.weight"
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
            _kf_to_hf_key("layers.15.mlp.router.gate.weight") == "model.layers.15.mlp.gate.weight"
        )


# ---------------------------------------------------------------------------
# MoE key mapping: HuggingFace → KempnerForge
# ---------------------------------------------------------------------------


class TestMoEHFtoKFKeyMapping:
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


# ---------------------------------------------------------------------------
# MoE round-trip: KF → HF → KF should be identity
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# HuggingFace config generation
# ---------------------------------------------------------------------------


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
