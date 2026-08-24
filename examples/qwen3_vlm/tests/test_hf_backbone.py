#!/usr/bin/env python3
"""Unit tests for the HF -> KempnerForge transformer key mapping.

The mapping (``map_key`` / ``map_state_dict``) and the tied-head fill are pure,
so they are tested on key strings / tensors alone (no GPU, network, or I/O).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add the example dir to path so we can import the conversion module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convert_hf_backbone import (  # noqa: E402
    _EMBED_KEY,
    _HEAD_KEY,
    fill_tied_head,
    map_key,
    map_state_dict,
)


class TestMapKey:
    def test_embedding_head_norm(self):
        assert map_key("model.embed_tokens.weight") == _EMBED_KEY
        assert map_key("lm_head.weight") == _HEAD_KEY
        assert map_key("model.norm.weight") == "norm.weight"

    def test_attention_projections(self):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert (
                map_key(f"model.layers.3.self_attn.{proj}.weight")
                == f"layers.3.attention.{proj}.weight"
            )

    def test_qk_norm_rides_along(self):
        assert (
            map_key("model.layers.0.self_attn.q_norm.weight") == "layers.0.attention.q_norm.weight"
        )
        assert (
            map_key("model.layers.0.self_attn.k_norm.weight") == "layers.0.attention.k_norm.weight"
        )

    def test_layernorm_renames(self):
        assert map_key("model.layers.1.input_layernorm.weight") == "layers.1.attention_norm.weight"
        assert (
            map_key("model.layers.1.post_attention_layernorm.weight") == "layers.1.mlp_norm.weight"
        )

    def test_mlp_unchanged(self):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert map_key(f"model.layers.2.mlp.{proj}.weight") == f"layers.2.mlp.{proj}.weight"

    def test_unmapped_returns_none(self):
        assert map_key("model.rotary_emb.inv_freq") is None
        assert map_key("something.else") is None


class TestMapStateDict:
    def test_splits_converted_and_unmapped(self):
        sd = {
            "model.embed_tokens.weight": torch.zeros(4, 3),
            "model.layers.0.self_attn.q_norm.weight": torch.zeros(3),
            "model.rotary_emb.inv_freq": torch.zeros(2),
        }
        converted, unmapped = map_state_dict(sd)
        assert set(converted) == {_EMBED_KEY, "layers.0.attention.q_norm.weight"}
        assert unmapped == ["model.rotary_emb.inv_freq"]


class TestFillTiedHead:
    def test_fills_head_from_embedding(self):
        embed = torch.zeros(4, 3)
        converted = {_EMBED_KEY: embed}
        assert fill_tied_head(converted, {_EMBED_KEY, _HEAD_KEY}) is True
        assert converted[_HEAD_KEY] is embed

    def test_noop_when_head_present(self):
        embed, head = torch.zeros(4, 3), torch.ones(4, 3)
        converted = {_EMBED_KEY: embed, _HEAD_KEY: head}
        assert fill_tied_head(converted, {_EMBED_KEY, _HEAD_KEY}) is False
        assert converted[_HEAD_KEY] is head

    def test_noop_when_target_untied(self):
        converted = {_EMBED_KEY: torch.zeros(4, 3)}
        assert fill_tied_head(converted, {_EMBED_KEY}) is False
        assert _HEAD_KEY not in converted


class TestUnmappedKeysAreReported:
    def test_unrecognized_layer_naming_is_surfaced(self):
        sd = {
            "transformer.h.0.attn.c_attn.weight": torch.zeros(2, 2),  # GPT-2 style
            "model.embed_tokens.weight": torch.zeros(4, 3),
        }
        converted, unmapped = map_state_dict(sd)
        assert set(converted) == {_EMBED_KEY}
        assert unmapped == ["transformer.h.0.attn.c_attn.weight"]
