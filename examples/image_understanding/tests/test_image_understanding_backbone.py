"""The stage presets build the Qwen3-0.6B backbone the converter fills."""

from __future__ import annotations

from pathlib import Path

import pytest

from kempnerforge.config.job import JobConfig
from kempnerforge.config.loader import load_config
from kempnerforge.model.transformer import TransformerBlock

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
STAGES = ["qwen3_0.6b_siglip2_stage1.toml", "qwen3_0.6b_siglip2_stage2.toml"]

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


@pytest.fixture(scope="module", params=STAGES, ids=STAGES)
def config(request: pytest.FixtureRequest) -> JobConfig:
    return load_config(str(CONFIG_DIR / request.param), cli_args=[])


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
        "hidden_act": str(model.activation),
    } == {k: v for k, v in SOURCE.items() if k != "tie_word_embeddings"}


def test_the_output_head_is_untied_from_the_tied_source(config: JobConfig) -> None:
    """The converter fills the untied head from the source's embedding."""
    assert SOURCE["tie_word_embeddings"] is True
    assert config.model.tie_embeddings is False


def test_attention_is_wider_than_the_model(config: JobConfig) -> None:
    attention = TransformerBlock(config.model, layer_idx=0).attention
    assert attention.q_proj.out_features == 16 * 128
    assert attention.k_proj.out_features == attention.v_proj.out_features == 8 * 128
    assert attention.o_proj.in_features == 16 * 128
    assert attention.o_proj.out_features == 1024
    assert attention.q_norm is not None and attention.q_norm.weight.shape == (128,)


def test_the_decoder_normalises_with_the_source_epsilon(config: JobConfig) -> None:
    block = TransformerBlock(config.model, layer_idx=0)
    epsilons = {
        name: module.eps
        for name, module in block.named_modules()
        if isinstance(getattr(module, "eps", None), float)
    }
    assert set(epsilons) >= {"attention.q_norm", "attention.k_norm", "attention_norm", "mlp_norm"}
    assert epsilons == dict.fromkeys(epsilons, SOURCE["rms_norm_eps"])
