"""Tests for ``configs/vlm_qwen3_0.6b_joint_decoder_webvid.toml``."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from kempnerforge.config.job import JobConfig
from kempnerforge.config.loader import load_config
from kempnerforge.model.transformer import TransformerBlock

CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "vlm_qwen3_0.6b_joint_decoder_webvid.toml"
)

# From the source model's config.json; the conversion refuses any disagreement.
SOURCE = {
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000,
    "tie_word_embeddings": True,
    "hidden_act": "silu",
}
PATCH = 14  # the vision encoder's patch size at 224 px


@pytest.fixture(scope="module")
def config() -> JobConfig:
    return load_config(str(CONFIG), cli_args=[])


def test_backbone_matches_the_source(config: JobConfig) -> None:
    model = config.model
    assert {
        "hidden_size": model.dim,
        "num_hidden_layers": model.n_layers,
        "num_attention_heads": model.n_heads,
        "num_key_value_heads": model.n_kv_heads,
        "head_dim": model.head_dim,
        "intermediate_size": model.computed_ffn_hidden_dim,
        "vocab_size": model.vocab_size,
        "rms_norm_eps": model.norm_eps,
        "rope_theta": model.rope_theta,
        "tie_word_embeddings": model.tie_embeddings,
        "hidden_act": str(model.activation),
    } == SOURCE
    assert model.qk_norm is True
    assert str(model.norm_type) == "rmsnorm"
    assert model.is_moe is False


def test_attention_is_wider_than_the_model(config: JobConfig) -> None:
    attention = TransformerBlock(config.model, layer_idx=0).attention
    assert attention.q_proj.out_features == 16 * 128
    assert attention.k_proj.out_features == attention.v_proj.out_features == 8 * 128
    assert attention.o_proj.in_features == 16 * 128
    assert attention.o_proj.out_features == 1024
    assert attention.q_norm is not None and attention.q_norm.weight.shape == (128,)


def test_the_decoder_normalises_with_the_source_epsilon(config: JobConfig) -> None:
    """Every norm the block runs must use the epsilon the backbone was trained with.

    The config value alone does not settle this: a norm that takes its epsilon
    from somewhere else normalises differently while the config still reads as
    intended, so the built block is asked directly.
    """
    block = TransformerBlock(config.model, layer_idx=0)
    epsilons = {
        name: module.eps
        for name, module in block.named_modules()
        if isinstance(getattr(module, "eps", None), float)
    }
    assert set(epsilons) >= {"attention.q_norm", "attention.k_norm", "attention_norm", "mlp_norm"}
    assert epsilons == dict.fromkeys(epsilons, SOURCE["rms_norm_eps"])


def test_vision_tokens_and_caption_fit_the_context(config: JobConfig) -> None:
    assert config.vision_encoder is not None and config.adapter is not None
    assert config.video is not None and config.vlm is not None
    assert config.vision_encoder.type == "siglip2"
    assert config.vision_encoder.path == "google/siglip2-so400m-patch14-224"
    assert (config.vision_encoder.num_tokens, config.vision_encoder.feature_dim) == (0, 0)
    assert (config.adapter.type, config.adapter.pool_window) == ("avgpool", 2)
    per_frame = config.adapter.output_num_tokens((config.video.frame_size // PATCH) ** 2)
    assert per_frame == 64
    assert config.video.max_frames * per_frame + config.vlm.max_text_len == 1248
    assert config.model.max_seq_len == config.train.seq_len == 1280


def test_video_captioning_setup(config: JobConfig) -> None:
    assert config.vlm is not None and config.video is not None
    assert config.vlm.arch == "joint_decoder"
    assert config.video.dataset_type == "webvid"
    assert config.video.prompt == "Describe the video:"
    assert config.data.tokenizer_path == "Qwen/Qwen3-0.6B"


def test_only_the_vision_side_trains(config: JobConfig) -> None:
    assert config.vlm is not None
    assert [(spec.module, spec.frozen) for spec in config.vlm.freeze] == [("transformer", True)]


def test_warm_start_is_weights_only(config: JobConfig) -> None:
    assert config.checkpoint.load_path == "path-to-init-checkpoint"
    assert config.checkpoint.exclude_from_loading == ["optimizer"]


def test_paths_are_placeholders_and_wandb_is_off(config: JobConfig) -> None:
    assert config.video is not None
    assert config.video.data_root == "path-to-webvid-10m"
    assert config.checkpoint.dir == "path-to-output-dir"
    assert config.checkpoint.load_path == "path-to-init-checkpoint"
    # Left at its default, every run would write its events into the same
    # relative directory, mixing a shakedown's with the real run's.
    assert config.metrics.tensorboard_dir == "path-to-tensorboard-dir"
    assert config.metrics.enable_wandb is False
    raw = tomllib.loads(CONFIG.read_text())
    assert "wandb_project" not in raw["metrics"] and "wandb_run_name" not in raw["metrics"]


@pytest.mark.parametrize(
    ("world_size", "cli_args"), [(16, []), (4, ["--train.grad_accum_steps=4"])]
)
def test_validates_at_a_global_batch_of_256(world_size: int, cli_args: list[str]) -> None:
    config = load_config(str(CONFIG), cli_args=cli_args)
    config.validate(world_size=world_size)
    assert config.train.batch_size * config.train.grad_accum_steps * world_size == 256
