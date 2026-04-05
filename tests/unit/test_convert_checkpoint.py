"""Unit tests for checkpoint format conversion key mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import the conversion module
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from convert_checkpoint import _build_hf_config, _hf_to_kf_key, _kf_to_hf_key

from kempnerforge.config.schema import ModelConfig

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
