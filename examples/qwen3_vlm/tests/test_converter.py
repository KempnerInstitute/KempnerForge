"""Standalone tests for the multimodal -> KempnerForge checkpoint key mapping.

Not part of the main ``tests/`` suite. Run explicitly:

    uv run pytest examples/qwen3_vlm/tests/

The mapping (``map_key`` / ``map_state_dict``) is pure, so it is tested on key
strings alone. The completeness test builds a tiny KF ``Transformer`` + adapter
(no vision, no network) for Joint-Decoder and MoT, synthesizes the matching
multimodal source keys, and asserts the mapping covers every target key exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Add the example dir to path so we can import the conversion module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convert_multimodal_checkpoint import map_key, map_state_dict

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.schema import ModelConfig
from kempnerforge.config.vlm import JointDecoderConfig, MoTConfig
from kempnerforge.model.adapter import build_adapter
from kempnerforge.model.transformer import Transformer

IMG_TEXT = {"0": "image", "1": "text"}  # source index -> KF modality name


# ---------------------------------------------------------------------------
# map_key — shared / vision / embed / head / adapter
# ---------------------------------------------------------------------------


class TestMapKeyShared:
    def test_vision_prefix_swap(self):
        assert map_key(
            "image_encoder.model.vision_model.embeddings.patch_embedding.weight", "joint_decoder"
        ) == ["vision_encoder.vision_tower.embeddings.patch_embedding.weight"]
        assert map_key(
            "image_encoder.model.vision_model.encoder.layers.7.self_attn.k_proj.bias", "mot"
        ) == ["vision_encoder.vision_tower.encoder.layers.7.self_attn.k_proj.bias"]

    def test_token_embedding(self):
        assert map_key("text_preprocessor.embed.weight", "joint_decoder") == [
            "transformer.token_embedding.embedding.weight"
        ]

    def test_output_head_weight(self):
        assert map_key("text_head.weight", "joint_decoder") == [
            "transformer.output_head.proj.weight"
        ]

    def test_output_head_bias_dropped(self):
        # KF OutputHead is bias-free; the source bias (older ckpts) is dropped.
        assert map_key("text_head.bias", "joint_decoder") == []

    def test_adapter_patch_merger(self):
        assert map_key("adapter.model.ln_q.weight", "joint_decoder") == ["adapter.ln_q.weight"]
        assert map_key("adapter.model.mlp.0.weight", "mot") == ["adapter.proj1.weight"]
        assert map_key("adapter.model.mlp.0.bias", "mot") == ["adapter.proj1.bias"]
        assert map_key("adapter.model.mlp.2.weight", "joint_decoder") == ["adapter.proj2.weight"]
        assert map_key("adapter.model.mlp.2.bias", "joint_decoder") == ["adapter.proj2.bias"]

    def test_unrecognized_key_returns_none(self):
        assert map_key("something.unexpected", "joint_decoder") is None
        assert map_key("multimodal_core.mystery.weight", "joint_decoder") is None


# ---------------------------------------------------------------------------
# map_key — Joint-Decoder layers + final norm
# ---------------------------------------------------------------------------


class TestMapKeyJointDecoder:
    def test_attention_projections(self):
        p = "multimodal_core.layers.5.self_attn"
        assert map_key(f"{p}.q_proj.weight", "joint_decoder") == [
            "transformer.layers.5.attention.q_proj.weight"
        ]
        assert map_key(f"{p}.o_proj.weight", "joint_decoder") == [
            "transformer.layers.5.attention.o_proj.weight"
        ]
        assert map_key(f"{p}.q_norm.weight", "joint_decoder") == [
            "transformer.layers.5.attention.q_norm.weight"
        ]

    def test_layernorms(self):
        assert map_key("multimodal_core.layers.0.input_layernorm.weight", "joint_decoder") == [
            "transformer.layers.0.attention_norm.weight"
        ]
        assert map_key(
            "multimodal_core.layers.0.post_attention_layernorm.weight", "joint_decoder"
        ) == ["transformer.layers.0.mlp_norm.weight"]

    def test_mlp(self):
        assert map_key("multimodal_core.layers.2.mlp.gate_proj.weight", "joint_decoder") == [
            "transformer.layers.2.mlp.gate_proj.weight"
        ]

    def test_final_norm_single(self):
        assert map_key("multimodal_core.norm.weight", "joint_decoder") == [
            "transformer.norm.weight"
        ]


# ---------------------------------------------------------------------------
# map_key — MoT per-modality + fanned-out final norm
# ---------------------------------------------------------------------------


class TestMapKeyMoT:
    def test_attn_norm_per_modality(self):
        assert map_key(
            "multimodal_core.layers.1.self_attn.input_layer_norm.0.weight", "mot", IMG_TEXT
        ) == ["transformer.layers.1.attn_norm.image.weight"]
        assert map_key(
            "multimodal_core.layers.1.self_attn.input_layer_norm.1.weight", "mot", IMG_TEXT
        ) == ["transformer.layers.1.attn_norm.text.weight"]

    def test_attn_projections_per_modality(self):
        assert map_key("multimodal_core.layers.3.self_attn.q_proj.0.weight", "mot", IMG_TEXT) == [
            "transformer.layers.3.attn.q_proj.image.weight"
        ]
        assert map_key("multimodal_core.layers.3.self_attn.k_norm.1.weight", "mot", IMG_TEXT) == [
            "transformer.layers.3.attn.k_norm.text.weight"
        ]

    def test_feed_forward_per_modality(self):
        assert map_key(
            "multimodal_core.layers.0.feed_forward.mlp.0.down_proj.weight", "mot", IMG_TEXT
        ) == ["transformer.layers.0.mlp.image.down_proj.weight"]
        assert map_key(
            "multimodal_core.layers.0.feed_forward.post_attention_layernorm.1.weight",
            "mot",
            IMG_TEXT,
        ) == ["transformer.layers.0.mlp_norm.text.weight"]

    def test_modality_order_swaps_with_index(self):
        swapped = {"0": "text", "1": "image"}
        assert map_key("multimodal_core.layers.0.self_attn.q_proj.0.weight", "mot", swapped) == [
            "transformer.layers.0.attn.q_proj.text.weight"
        ]

    def test_final_norm_fans_out(self):
        assert map_key("multimodal_core.norm.weight", "mot", IMG_TEXT) == [
            "transformer.norm.weight",
            "transformer.mot_norms.image.weight",
            "transformer.mot_norms.text.weight",
        ]


class TestMapKeyUnsupportedArch:
    def test_cross_attention_layer_key_raises(self):
        with pytest.raises(NotImplementedError):
            map_key("multimodal_core.layers.0.self_attn.q_proj.weight", "cross_attention")


# ---------------------------------------------------------------------------
# map_key — JD source -> MoT target (init MoT from a dense JD checkpoint)
# ---------------------------------------------------------------------------


class TestMapKeyJDtoMoT:
    """Dense JD weights duplicated into both MoT modality copies."""

    def test_attention_duplicated_into_both_modalities(self):
        assert map_key(
            "multimodal_core.layers.3.self_attn.q_proj.weight", "joint_decoder", target_arch="mot"
        ) == [
            "transformer.layers.3.attn.q_proj.image.weight",
            "transformer.layers.3.attn.q_proj.text.weight",
        ]
        assert map_key(
            "multimodal_core.layers.3.self_attn.q_norm.weight", "joint_decoder", target_arch="mot"
        ) == [
            "transformer.layers.3.attn.q_norm.image.weight",
            "transformer.layers.3.attn.q_norm.text.weight",
        ]

    def test_layernorms_duplicated(self):
        assert map_key(
            "multimodal_core.layers.0.input_layernorm.weight", "joint_decoder", target_arch="mot"
        ) == [
            "transformer.layers.0.attn_norm.image.weight",
            "transformer.layers.0.attn_norm.text.weight",
        ]
        assert map_key(
            "multimodal_core.layers.0.post_attention_layernorm.weight",
            "joint_decoder",
            target_arch="mot",
        ) == [
            "transformer.layers.0.mlp_norm.image.weight",
            "transformer.layers.0.mlp_norm.text.weight",
        ]

    def test_mlp_duplicated(self):
        assert map_key(
            "multimodal_core.layers.1.mlp.down_proj.weight", "joint_decoder", target_arch="mot"
        ) == [
            "transformer.layers.1.mlp.image.down_proj.weight",
            "transformer.layers.1.mlp.text.down_proj.weight",
        ]

    def test_final_norm_fans_out(self):
        assert map_key("multimodal_core.norm.weight", "joint_decoder", target_arch="mot") == [
            "transformer.norm.weight",
            "transformer.mot_norms.image.weight",
            "transformer.mot_norms.text.weight",
        ]

    def test_shared_parts_unchanged(self):
        # embedding / head / adapter map the same regardless of target arch.
        assert map_key("text_preprocessor.embed.weight", "joint_decoder", target_arch="mot") == [
            "transformer.token_embedding.embedding.weight"
        ]
        assert map_key("adapter.model.ln_q.weight", "joint_decoder", target_arch="mot") == [
            "adapter.ln_q.weight"
        ]


# ---------------------------------------------------------------------------
# Completeness: mapping covers every target key for a tiny JD / MoT model
# ---------------------------------------------------------------------------


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,  # decoupled (4 * 16 = 64 here, but exercises the field)
        qk_norm=True,
        ffn_hidden_dim=128,
        vocab_size=100,
        max_seq_len=64,
    )


def _target_keys(model: Transformer, adapter: torch.nn.Module) -> set[str]:
    return {f"transformer.{k}" for k in model.state_dict()} | {
        f"adapter.{k}" for k in adapter.state_dict()
    }


def _adapter():
    cfg = AdapterConfig(type="mlp_2layer", pre_norm="rmsnorm", hidden_dim=32, activation="gelu")
    return build_adapter(cfg, in_dim=32, out_dim=64)


def _jd_source_keys(n_layers: int) -> list[str]:
    keys = [
        "text_preprocessor.embed.weight",
        "text_head.weight",
        "multimodal_core.norm.weight",
        "adapter.model.ln_q.weight",
        "adapter.model.mlp.0.weight",
        "adapter.model.mlp.0.bias",
        "adapter.model.mlp.2.weight",
        "adapter.model.mlp.2.bias",
    ]
    for i in range(n_layers):
        p = f"multimodal_core.layers.{i}"
        keys += [
            f"{p}.self_attn.q_proj.weight",
            f"{p}.self_attn.k_proj.weight",
            f"{p}.self_attn.v_proj.weight",
            f"{p}.self_attn.o_proj.weight",
            f"{p}.self_attn.q_norm.weight",
            f"{p}.self_attn.k_norm.weight",
            f"{p}.input_layernorm.weight",
            f"{p}.post_attention_layernorm.weight",
            f"{p}.mlp.gate_proj.weight",
            f"{p}.mlp.up_proj.weight",
            f"{p}.mlp.down_proj.weight",
        ]
    return keys


def _mot_source_keys(n_layers: int) -> list[str]:
    keys = [
        "text_preprocessor.embed.weight",
        "text_head.weight",
        "multimodal_core.norm.weight",
        "adapter.model.ln_q.weight",
        "adapter.model.mlp.0.weight",
        "adapter.model.mlp.0.bias",
        "adapter.model.mlp.2.weight",
        "adapter.model.mlp.2.bias",
    ]
    for i in range(n_layers):
        p = f"multimodal_core.layers.{i}"
        for m in ("0", "1"):
            keys += [
                f"{p}.self_attn.input_layer_norm.{m}.weight",
                f"{p}.self_attn.q_proj.{m}.weight",
                f"{p}.self_attn.k_proj.{m}.weight",
                f"{p}.self_attn.v_proj.{m}.weight",
                f"{p}.self_attn.o_proj.{m}.weight",
                f"{p}.self_attn.q_norm.{m}.weight",
                f"{p}.self_attn.k_norm.{m}.weight",
                f"{p}.feed_forward.mlp.{m}.gate_proj.weight",
                f"{p}.feed_forward.mlp.{m}.up_proj.weight",
                f"{p}.feed_forward.mlp.{m}.down_proj.weight",
                f"{p}.feed_forward.post_attention_layernorm.{m}.weight",
            ]
    return keys


class TestMappingCompleteness:
    def test_joint_decoder_covers_all_target_keys(self):
        cfg = _tiny_config()
        model = Transformer(cfg, vlm_config=JointDecoderConfig())
        adapter = _adapter()
        target = _target_keys(model, adapter)

        src = {k: torch.zeros(1) for k in _jd_source_keys(cfg.n_layers)}
        converted, unmapped = map_state_dict(src, "joint_decoder")

        assert unmapped == []
        assert set(converted) == target  # no missing, no unexpected

    def test_mot_covers_all_target_keys(self):
        cfg = _tiny_config()
        model = Transformer(cfg, vlm_config=MoTConfig(), num_image_tokens=8)
        adapter = _adapter()
        target = _target_keys(model, adapter)

        src = {k: torch.zeros(1) for k in _mot_source_keys(cfg.n_layers)}
        converted, unmapped = map_state_dict(src, "mot", IMG_TEXT)

        assert unmapped == []
        assert set(converted) == target

    def test_jd_to_mot_covers_all_target_keys(self):
        # Init MoT from a dense JD checkpoint: duplicate each dense weight into
        # both modality copies. The JD source must fully seed the MoT target.
        cfg = _tiny_config()
        model = Transformer(cfg, vlm_config=MoTConfig(), num_image_tokens=8)
        adapter = _adapter()
        target = _target_keys(model, adapter)

        src = {k: torch.zeros(1) for k in _jd_source_keys(cfg.n_layers)}
        converted, unmapped = map_state_dict(src, "joint_decoder", target_arch="mot")

        assert unmapped == []
        assert set(converted) == target

    def test_jd_to_mot_symmetric_duplication_shares_source_tensor(self):
        # Symmetric init: the image and text copies are the SAME source tensor.
        src = {"multimodal_core.layers.0.self_attn.o_proj.weight": torch.zeros(3)}
        converted, _ = map_state_dict(src, "joint_decoder", target_arch="mot")
        assert (
            converted["transformer.layers.0.attn.o_proj.image.weight"]
            is converted["transformer.layers.0.attn.o_proj.text.weight"]
        )
