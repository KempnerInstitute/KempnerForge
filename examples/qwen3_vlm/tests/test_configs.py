"""Standalone tests for the Qwen3-0.6B VLM example configs.

Not part of the main ``tests/`` suite (testpaths). Run explicitly:

    uv run pytest examples/qwen3_vlm/tests/

Each config loads, selects the right arch, builds a Qwen3-0.6B backbone block
with the decoupled 2048-wide attention, uses the SigLIP2 so400m encoder +
a pre-norm mlp_2layer adapter, and wires the video pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kempnerforge.config.loader import load_config
from kempnerforge.model.transformer import TransformerBlock

EXAMPLE = Path(__file__).resolve().parents[1]
ATTN_WIDTH = 16 * 128  # 2048, decoupled from dim=1024

QWEN3_06B = {
    "dim": 1024,
    "n_layers": 28,
    "n_heads": 16,
    "n_kv_heads": 8,
    "head_dim": 128,
    "ffn_hidden_dim": 3072,
    "vocab_size": 151936,
    "rope_theta": 1000000.0,
    "norm_eps": 1e-6,
}

CONFIGS = {
    "joint_decoder": EXAMPLE / "configs/vlm_qwen3_0.6b_joint_decoder.toml",
    "cross_attention": EXAMPLE / "configs/vlm_qwen3_0.6b_cross_attn.toml",
    "mot": EXAMPLE / "configs/vlm_qwen3_0.6b_mot.toml",
    "moma": EXAMPLE / "configs/vlm_qwen3_0.6b_moma.toml",
}


class TestQwen3VLMConfigs:
    @pytest.mark.parametrize("arch", list(CONFIGS))
    def test_loads_and_selects_arch(self, arch):
        cfg = load_config(str(CONFIGS[arch]), cli_args=[])
        assert cfg.vlm is not None
        assert cfg.vlm.arch == arch

    @pytest.mark.parametrize("arch", list(CONFIGS))
    def test_model_is_qwen3_06b_shape(self, arch):
        model = load_config(str(CONFIGS[arch]), cli_args=[]).model
        assert model.dim == QWEN3_06B["dim"]
        assert model.n_layers == QWEN3_06B["n_layers"]
        assert model.n_heads == QWEN3_06B["n_heads"]
        assert model.n_kv_heads == QWEN3_06B["n_kv_heads"]
        assert model.head_dim == QWEN3_06B["head_dim"]
        assert model.computed_ffn_hidden_dim == QWEN3_06B["ffn_hidden_dim"]
        assert model.vocab_size == QWEN3_06B["vocab_size"]
        assert model.qk_norm is True
        assert model.tie_embeddings is False
        assert model.rope_theta == QWEN3_06B["rope_theta"]
        assert model.norm_eps == QWEN3_06B["norm_eps"]

    @pytest.mark.parametrize("arch", list(CONFIGS))
    def test_uses_prenorm_mlp_and_siglip2(self, arch):
        cfg = load_config(str(CONFIGS[arch]), cli_args=[])
        assert cfg.adapter is not None
        assert cfg.vision_encoder is not None
        assert cfg.adapter.type == "mlp_2layer"
        assert cfg.adapter.pre_norm == "rmsnorm"
        assert cfg.vision_encoder.type == "siglip2"
        assert "siglip2-so400m-patch14-224" in cfg.vision_encoder.path
        # feature_dim / num_tokens use the 0 = "probe at build time" sentinel.
        assert cfg.vision_encoder.feature_dim == 0
        assert cfg.vision_encoder.num_tokens == 0

    @pytest.mark.parametrize("arch", list(CONFIGS))
    def test_video_pipeline_configured(self, arch):
        cfg = load_config(str(CONFIGS[arch]), cli_args=[])
        assert cfg.video is not None
        assert cfg.video.dataset_type == "webvid"
        assert cfg.video.max_frames >= 1

    @pytest.mark.parametrize("arch", ["joint_decoder", "mot"])
    def test_warm_start_configs_set_load_path(self, arch):
        # JD and MoT warm-start from a converted checkpoint; cross/moma are
        # from scratch (no source) and intentionally leave load_path unset.
        cfg = load_config(str(CONFIGS[arch]), cli_args=[])
        assert cfg.checkpoint.load_path
        assert cfg.checkpoint.exclude_from_loading == ["optimizer", "dataloader"]

    @pytest.mark.parametrize("arch", ["cross_attention", "moma"])
    def test_from_scratch_configs_have_no_load_path(self, arch):
        cfg = load_config(str(CONFIGS[arch]), cli_args=[])
        assert not cfg.checkpoint.load_path

    @pytest.mark.parametrize("arch", list(CONFIGS))
    def test_backbone_block_builds_with_qwen3_attention_shape(self, arch):
        # Build ONE block from the parsed config (cheap: no embedding/head, no
        # full 28-layer stack) and confirm the decoupled 2048-wide attention.
        model = load_config(str(CONFIGS[arch]), cli_args=[]).model
        block = TransformerBlock(model, layer_idx=0)
        assert block.attention.q_proj.out_features == ATTN_WIDTH  # 2048
        assert block.attention.o_proj.out_features == QWEN3_06B["dim"]  # 1024
        assert block.attention.q_norm is not None  # qk_norm on
        assert block.attention.q_norm.weight.shape == (QWEN3_06B["head_dim"],)  # (128,)
