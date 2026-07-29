"""Tests for the Qwen3-0.6B video-training config.

Standalone (not in ``testpaths``): ``uv run pytest examples/qwen3_vlm/tests/``
"""

from __future__ import annotations

from pathlib import Path

from kempnerforge.config.loader import load_config
from kempnerforge.model.transformer import TransformerBlock

EXAMPLE = Path(__file__).resolve().parents[1]
CONFIG = EXAMPLE / "configs/vlm_qwen3_0.6b_joint_decoder_webvid.toml"
ATTN_WIDTH = 16 * 128  # 2048, decoupled from dim=1024
SIGLIP2_PATCH = 14

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


def _cfg():
    return load_config(str(CONFIG), cli_args=[])


class TestQwen3VLMConfig:
    def test_loads_and_selects_joint_decoder(self):
        cfg = _cfg()
        assert cfg.vlm is not None
        assert cfg.vlm.arch == "joint_decoder"

    def test_model_is_qwen3_06b_shape(self):
        model = _cfg().model
        assert model.dim == QWEN3_06B["dim"]
        assert model.n_layers == QWEN3_06B["n_layers"]
        assert model.n_heads == QWEN3_06B["n_heads"]
        assert model.n_kv_heads == QWEN3_06B["n_kv_heads"]
        assert model.head_dim == QWEN3_06B["head_dim"]
        assert model.computed_ffn_hidden_dim == QWEN3_06B["ffn_hidden_dim"]
        assert model.vocab_size == QWEN3_06B["vocab_size"]
        assert model.qk_norm is True
        assert model.rope_theta == QWEN3_06B["rope_theta"]
        assert model.norm_eps == QWEN3_06B["norm_eps"]
        # HF Qwen3-0.6B ties the output head to the embedding.
        assert model.tie_embeddings is True

    def test_backbone_block_builds_with_qwen3_attention_shape(self):
        # One block only — cheap, and enough to check the decoupled width.
        block = TransformerBlock(_cfg().model, layer_idx=0)
        assert block.attention.q_proj.out_features == ATTN_WIDTH  # 2048
        assert block.attention.o_proj.out_features == QWEN3_06B["dim"]  # 1024
        assert block.attention.q_norm is not None  # qk_norm on
        assert block.attention.q_norm.weight.shape == (QWEN3_06B["head_dim"],)  # (128,)

    def test_uses_siglip2_with_avgpool_connector(self):
        cfg = _cfg()
        assert cfg.vision_encoder is not None and cfg.adapter is not None
        assert cfg.vision_encoder.type == "siglip2"
        assert "siglip2-so400m-patch14-224" in cfg.vision_encoder.path
        # feature_dim / num_tokens use the 0 = "probe at build time" sentinel.
        assert cfg.vision_encoder.feature_dim == 0
        assert cfg.vision_encoder.num_tokens == 0
        assert cfg.adapter.type == "avgpool"
        assert cfg.adapter.pool_window == 2  # 16x16 patches -> 8x8 = 64/frame

    def test_video_pipeline_configured(self):
        cfg = _cfg()
        assert cfg.video is not None
        assert cfg.video.dataset_type == "webvid"
        assert cfg.video.max_frames == 18
        assert cfg.video.prompt  # masked prompt supervises the first caption token

    def test_visual_plus_text_fits_max_seq_len(self):
        cfg = _cfg()
        patches = (cfg.video.frame_size // SIGLIP2_PATCH) ** 2
        per_frame = patches // (cfg.adapter.pool_window**2)
        total = cfg.video.max_frames * per_frame + cfg.vlm.max_text_len
        assert total <= cfg.model.max_seq_len, f"{total} > {cfg.model.max_seq_len}"

    def test_warm_start_is_weights_only(self):
        cfg = _cfg()
        assert cfg.checkpoint.load_path  # the DCP from convert_hf_backbone.py
        assert cfg.checkpoint.exclude_from_loading == ["optimizer", "dataloader"]

    def test_llm_frozen_vision_and_adapter_train(self):
        frozen = {s.module for s in _cfg().vlm.freeze if s.frozen}
        assert "transformer" in frozen
        assert "vision_encoder" not in frozen
