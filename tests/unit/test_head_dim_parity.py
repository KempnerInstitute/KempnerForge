"""Tests that an odd head_dim, which rotary embeddings cannot rotate, fails as a ValueError."""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.loader import load_config
from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.position import precompute_rope_frequencies
from kempnerforge.model.transformer import Transformer

TINY = {"n_layers": 1, "vocab_size": 64, "max_seq_len": 16}


def _odd_head_dim_message(dim: int, n_heads: int) -> str:
    return (
        f"dim ({dim}) // n_heads ({n_heads}) gives an odd head_dim ({dim // n_heads}), and "
        "rotary embeddings need an even one; change dim or n_heads, or set an even "
        "head_dim_override"
    )


class TestModelConfigHeadDimParity:
    @pytest.mark.parametrize(("dim", "n_heads"), [(68, 4), (6, 2), (72, 8), (17, 1)])
    def test_an_odd_inferred_head_dim_is_rejected(self, dim, n_heads):
        with pytest.raises(ValueError) as excinfo:
            ModelConfig(dim=dim, n_heads=n_heads, n_kv_heads=n_heads, **TINY)
        assert str(excinfo.value) == _odd_head_dim_message(dim, n_heads)

    def test_a_toml_with_an_odd_inferred_head_dim_fails_to_load(self, tmp_path):
        toml = tmp_path / "odd.toml"
        toml.write_text("[model]\ndim = 68\nn_heads = 4\nn_kv_heads = 4\n")
        with pytest.raises(ValueError, match=r"odd head_dim \(17\)"):
            load_config(str(toml), cli_args=[])

    def test_a_cli_override_to_an_odd_inferred_head_dim_fails_to_load(self):
        args = ["--model.dim=68", "--model.n_heads=4", "--model.n_kv_heads=4"]
        with pytest.raises(ValueError, match=r"odd head_dim \(17\)"):
            load_config(cli_args=args)

    def test_a_dim_that_does_not_divide_still_reports_divisibility(self):
        with pytest.raises(ValueError, match=r"dim \(70\) must be divisible by n_heads \(4\)"):
            ModelConfig(dim=70, n_heads=4, n_kv_heads=4, **TINY)

    @pytest.mark.parametrize(
        ("dim", "n_heads", "head_dim_override", "head_dim"),
        [(64, 4, 0, 16), (68, 4, 16, 16), (70, 4, 18, 18)],
    )
    def test_an_even_head_dim_builds_and_runs(self, dim, n_heads, head_dim_override, head_dim):
        """An odd ``dim // n_heads`` is fine once ``head_dim_override`` sets an even width."""
        config = ModelConfig(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            head_dim_override=head_dim_override,
            **TINY,
        )
        assert config.head_dim == head_dim
        logits = Transformer(config)(torch.randint(0, TINY["vocab_size"], (2, 8)))
        assert logits.shape == (2, 8, TINY["vocab_size"])


class TestRopeHeadDim:
    def test_an_odd_head_dim_raises_value_error(self):
        with pytest.raises(
            ValueError, match="^head_dim must be even for rotary embeddings, got 17$"
        ):
            precompute_rope_frequencies(head_dim=17, max_seq_len=8)

    def test_an_even_head_dim_gives_half_width_tables(self):
        cos, sin = precompute_rope_frequencies(head_dim=18, max_seq_len=8)
        assert cos.shape == sin.shape == (8, 9)
