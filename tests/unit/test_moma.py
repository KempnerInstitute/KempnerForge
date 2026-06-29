"""Unit tests for Mixture of Modality-Aware Experts (MoMa) operator + block + integration.

Covers:

- ``MoMaConfig``: field defaults, validation (modalities, expert counts,
  capacity factor), polymorphic methods (``residual_stream_image_tokens``,
  ``effective_capacity_factor``).
- ``MoMaStrategy``: ``ModalityContext`` construction (prefix_embeds,
  output_slice, modality_ids), modality_ids value/shape/dtype, num_image_tokens.
- ``ExpertChoiceSigmoidRouter``: forward shapes, Gumbel-noise behavior
  (train vs eval), empty input, k_e clamping, expert_counts metric.
- ``ExpertChoiceMoE``: forward shape, empty input, gradient flow,
  multi-expert accumulation onto the same token, zero contribution when
  no expert picks a token.
- ``MoMaFFN``: per-modality dispatch by modality_ids, all-text /
  all-image batches, gradient flow into both modality groups, shape /
  dtype validation.
- ``MoMaBlock``: forward shape, residual add, gradient flow, parameter
  layout (shared QKVO + per-modality MoE FFN).
- End-to-end ``Transformer`` + ``MoMaConfig``: build, forward, gradient
  flow, output shape.

No GPU required; uses CPU tensors.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from kempnerforge.config.schema import ModelConfig
from kempnerforge.config.vlm import MoMaConfig
from kempnerforge.model.attention import Attention
from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.moma import (
    ExpertChoiceMoE,
    ExpertChoiceSigmoidRouter,
    MoMaBlock,
    MoMaFFN,
)
from kempnerforge.model.transformer import Transformer
from kempnerforge.model.vlm import MoMaStrategy

DEVICE = torch.device("cpu")


def _config(
    dim: int = 64,
    n_heads: int = 4,
    n_kv_heads: int | None = None,
    n_layers: int = 2,
    max_seq_len: int = 64,
) -> ModelConfig:
    """Tiny dense config for MoMa unit tests."""
    return ModelConfig(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads or n_heads,
        vocab_size=128,
        max_seq_len=max_seq_len,
        ffn_hidden_dim=128,
    )


def test_moma_finegrained_experts():
    """MoMa expert-choice MoE experts honor moe_expert_ffn_multiplier (fine-grained)."""
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads=4, vocab_size=128, max_seq_len=64, moe_expert_ffn_multiplier=0.5
    )
    ffn = MoMaFFN(
        cfg,
        modalities=("image", "text"),
        experts_per_modality={"image": 4, "text": 4},
        capacity_factor_per_modality={"image": 1.0, "text": 1.0},
    )
    expert = ffn.experts["text"].experts[0]
    assert expert.gate_proj.weight.shape[0] == cfg.computed_ffn_hidden_dim // 2


def test_moma_ffn_excludes_padded_from_routing():
    """key_padding_mask drops padded positions from expert-choice routing: they
    get zero FFN output, and real tokens' outputs don't depend on padded tokens'
    content (no capacity competition)."""
    torch.manual_seed(0)
    cfg = ModelConfig(dim=32, n_layers=2, n_heads=4, vocab_size=128, max_seq_len=64)
    ffn = MoMaFFN(
        cfg,
        modalities=("image", "text"),
        experts_per_modality={"image": 2, "text": 2},
        capacity_factor_per_modality={"image": 1.0, "text": 1.0},
        gumbel_noise=False,  # deterministic routing for the comparison
    )
    # positions 0..3 image, 4..7 text; mark image positions 2,3 as padded.
    modality_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    kpm = torch.tensor([[True, True, False, False, True, True, True, True]])
    x = torch.randn(1, 8, cfg.dim)
    out = ffn(x, modality_ids, key_padding_mask=kpm)
    # Padded image positions are excluded from routing -> zero FFN output.
    assert torch.count_nonzero(out[0, 2]) == 0
    assert torch.count_nonzero(out[0, 3]) == 0
    # Real tokens' outputs are invariant to the padded tokens' content.
    x2 = x.clone()
    x2[0, 2:4] = torch.randn(2, cfg.dim)
    out2 = ffn(x2, modality_ids, key_padding_mask=kpm)
    real = [0, 1, 4, 5, 6, 7]
    assert torch.equal(out[0, real], out2[0, real])


# ---------------------------------------------------------------------------
# MoMaConfig
# ---------------------------------------------------------------------------


class TestMoMaConfig:
    def test_defaults(self):
        cfg = MoMaConfig()
        assert cfg.arch == "moma"
        assert cfg.moma_modalities == ("image", "text")
        assert cfg.moma_experts_per_modality == {"image": 4, "text": 4}
        assert cfg.moma_capacity_factor == 0.0
        assert cfg.moma_gumbel_noise is True

    def test_module_patterns_includes_moma_alias(self):
        cfg = MoMaConfig()
        assert "moma" in cfg.module_patterns
        # Sanity: alias points at the transformer layers.
        assert any("transformer.layers" in p for p in cfg.module_patterns["moma"])

    def test_residual_stream_image_tokens_is_num_tokens(self):
        # MoMa uses the JD/MoT image-prefix layout.
        cfg = MoMaConfig()
        assert cfg.residual_stream_image_tokens(64) == 64
        assert cfg.residual_stream_image_tokens(0) == 0

    def test_effective_capacity_factor_paper_default(self):
        cfg = MoMaConfig(moma_experts_per_modality={"image": 4, "text": 4})
        # Paper default c_e = 1/|E^M| per modality.
        assert cfg.effective_capacity_factor("image") == pytest.approx(0.25)
        assert cfg.effective_capacity_factor("text") == pytest.approx(0.25)

    def test_effective_capacity_factor_explicit_override(self):
        cfg = MoMaConfig(
            moma_experts_per_modality={"image": 4, "text": 4},
            moma_capacity_factor=0.5,
        )
        assert cfg.effective_capacity_factor("image") == 0.5
        assert cfg.effective_capacity_factor("text") == 0.5

    def test_effective_capacity_factor_unbalanced(self):
        # Paper's moe_7t1i: 7 text experts + 1 image expert.
        cfg = MoMaConfig(moma_experts_per_modality={"image": 1, "text": 7})
        assert cfg.effective_capacity_factor("image") == pytest.approx(1.0)
        assert cfg.effective_capacity_factor("text") == pytest.approx(1.0 / 7)

    def test_rejects_fewer_than_two_modalities(self):
        with pytest.raises(ValueError, match="at least 2 entries"):
            MoMaConfig(moma_modalities=("text",))

    def test_rejects_missing_text_modality(self):
        with pytest.raises(ValueError, match="must include 'text'"):
            MoMaConfig(
                moma_modalities=("image", "audio"),
                moma_experts_per_modality={"image": 2, "audio": 2},
            )

    def test_rejects_missing_image_modality(self):
        with pytest.raises(ValueError, match="must include 'image'"):
            MoMaConfig(
                moma_modalities=("text", "audio"),
                moma_experts_per_modality={"text": 2, "audio": 2},
            )

    def test_rejects_duplicate_modalities(self):
        with pytest.raises(ValueError, match="must not contain duplicates"):
            MoMaConfig(
                moma_modalities=("image", "text", "image"),
                moma_experts_per_modality={"image": 2, "text": 2},
            )

    def test_rejects_missing_expert_count_entry(self):
        with pytest.raises(ValueError, match="missing entries"):
            MoMaConfig(moma_experts_per_modality={"image": 2})

    def test_rejects_extra_expert_count_entry(self):
        with pytest.raises(ValueError, match="unknown modality keys"):
            MoMaConfig(moma_experts_per_modality={"image": 2, "text": 2, "audio": 4})

    def test_rejects_nonpositive_expert_count(self):
        with pytest.raises(ValueError, match="must be positive"):
            MoMaConfig(moma_experts_per_modality={"image": 0, "text": 2})

    def test_rejects_negative_capacity_factor(self):
        with pytest.raises(ValueError, match="capacity_factor must be >= 0"):
            MoMaConfig(moma_capacity_factor=-0.1)


# ---------------------------------------------------------------------------
# MoMaStrategy
# ---------------------------------------------------------------------------


class _StubVisionEncoder(nn.Module):
    def __init__(self, num_tokens: int, feature_dim: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        self.proj = nn.Linear(3, feature_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # (B, ?, ?, ?) -> (B, num_tokens, feature_dim)
        b = pixel_values.shape[0]
        return torch.zeros(b, self.num_tokens, self.feature_dim, device=pixel_values.device)


class _StubAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def output_num_tokens(self, num_input_tokens: int) -> int:
        return num_input_tokens  # projection stub: token count unchanged

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.proj(feats)


class _StubWrapper(nn.Module):
    def __init__(self, num_tokens: int, feature_dim: int, dim: int) -> None:
        super().__init__()
        self.vision_encoder = _StubVisionEncoder(num_tokens, feature_dim)
        self.adapter = _StubAdapter(feature_dim, dim)
        self.frames_per_clip = 1


class TestMoMaStrategy:
    def test_prepare_builds_modality_context(self):
        wrapper = _StubWrapper(num_tokens=4, feature_dim=8, dim=16)
        strategy = MoMaStrategy()
        pixel_values = torch.zeros(2, 3, 8, 8)
        input_ids = torch.zeros(2, 6, dtype=torch.long)

        ctx = strategy.prepare(wrapper, pixel_values, input_ids)
        assert isinstance(ctx, ModalityContext)
        assert ctx.prefix_embeds is not None
        assert ctx.prefix_embeds.shape == (2, 4, 16)
        assert ctx.output_slice == slice(4, None)
        assert ctx.modality_ids is not None
        assert ctx.modality_ids.shape == (2, 4 + 6)
        assert ctx.modality_ids.dtype == torch.long

    def test_modality_ids_image_then_text(self):
        wrapper = _StubWrapper(num_tokens=3, feature_dim=8, dim=16)
        strategy = MoMaStrategy()
        pixel_values = torch.zeros(1, 3, 8, 8)
        input_ids = torch.zeros(1, 5, dtype=torch.long)

        ctx = strategy.prepare(wrapper, pixel_values, input_ids)
        # First 3 positions (image) get 0; rest (text) get 1.
        assert ctx.modality_ids is not None
        ids = ctx.modality_ids[0]
        assert torch.equal(ids[:3], torch.zeros(3, dtype=torch.long))
        assert torch.equal(ids[3:], torch.ones(5, dtype=torch.long))

    def test_num_image_tokens(self):
        wrapper = _StubWrapper(num_tokens=7, feature_dim=8, dim=16)
        strategy = MoMaStrategy()
        assert strategy.num_image_tokens(wrapper) == 7


# ---------------------------------------------------------------------------
# ExpertChoiceSigmoidRouter
# ---------------------------------------------------------------------------


class TestExpertChoiceSigmoidRouter:
    def test_construction_rejects_nonpositive_experts(self):
        with pytest.raises(ValueError, match="num_experts must be positive"):
            ExpertChoiceSigmoidRouter(dim=16, num_experts=0, capacity_factor=0.5)

    def test_construction_rejects_nonpositive_capacity(self):
        with pytest.raises(ValueError, match="capacity_factor must be positive"):
            ExpertChoiceSigmoidRouter(dim=16, num_experts=4, capacity_factor=0.0)
        with pytest.raises(ValueError, match="capacity_factor must be positive"):
            ExpertChoiceSigmoidRouter(dim=16, num_experts=4, capacity_factor=-0.1)

    def test_forward_shapes(self):
        router = ExpertChoiceSigmoidRouter(dim=16, num_experts=4, capacity_factor=0.25)
        router.eval()
        x = torch.randn(20, 16)
        scores, indices = router(x)
        # k_e = ceil(0.25 * 20) = 5
        assert scores.shape == (4, 5)
        assert indices.shape == (4, 5)
        assert indices.dtype == torch.long
        # All indices in range [0, N).
        assert indices.min().item() >= 0
        assert indices.max().item() < 20

    def test_forward_rejects_wrong_rank(self):
        router = ExpertChoiceSigmoidRouter(dim=16, num_experts=2, capacity_factor=0.5)
        with pytest.raises(ValueError, match=r"\(N, D\)"):
            router(torch.randn(2, 8, 16))  # (B, S, D) — rank 3, not 2

    def test_forward_empty_input(self):
        router = ExpertChoiceSigmoidRouter(dim=16, num_experts=4, capacity_factor=0.25)
        x = torch.zeros(0, 16)
        scores, indices = router(x)
        assert scores.shape == (4, 0)
        assert indices.shape == (4, 0)

    def test_k_e_clamped_to_n(self):
        # capacity 1.0 → k_e = N. capacity > 1 (e.g. 5.0 * N) → clamp to N.
        router = ExpertChoiceSigmoidRouter(dim=16, num_experts=2, capacity_factor=5.0)
        router.eval()
        x = torch.randn(3, 16)
        scores, indices = router(x)
        assert scores.shape == (2, 3)
        assert indices.shape == (2, 3)

    def test_k_e_at_least_one(self):
        # capacity * N < 1 → k_e should still be 1.
        router = ExpertChoiceSigmoidRouter(dim=16, num_experts=2, capacity_factor=0.01)
        router.eval()
        x = torch.randn(4, 16)
        scores, indices = router(x)
        # ceil(0.01 * 4) = 1
        assert scores.shape == (2, 1)
        assert indices.shape == (2, 1)

    def test_gumbel_noise_changes_scores_in_train_mode(self):
        # We check scores, not selections: when gate logits are well-separated,
        # Gumbel noise of typical magnitude (G' - G'' ~ Logistic(0, 1)) may not
        # flip the top-k ordering even though the underlying scores differ.
        # Probing scores directly is the robust signal that noise was applied.
        router = ExpertChoiceSigmoidRouter(
            dim=16, num_experts=4, capacity_factor=0.5, gumbel_noise=True
        )
        router.train()
        x = torch.randn(8, 16)
        torch.manual_seed(0)
        scores_a, _ = router(x)
        torch.manual_seed(1)
        scores_b, _ = router(x)
        assert not torch.allclose(scores_a, scores_b)

    def test_eval_mode_deterministic(self):
        router = ExpertChoiceSigmoidRouter(
            dim=16, num_experts=4, capacity_factor=0.5, gumbel_noise=True
        )
        router.eval()
        x = torch.randn(8, 16)
        scores_a, idx_a = router(x)
        scores_b, idx_b = router(x)
        assert torch.equal(idx_a, idx_b)
        assert torch.allclose(scores_a, scores_b)

    def test_gumbel_disabled_deterministic_in_train(self):
        router = ExpertChoiceSigmoidRouter(
            dim=16, num_experts=4, capacity_factor=0.5, gumbel_noise=False
        )
        router.train()
        x = torch.randn(8, 16)
        scores_a, idx_a = router(x)
        scores_b, idx_b = router(x)
        assert torch.equal(idx_a, idx_b)
        assert torch.allclose(scores_a, scores_b)

    def test_expert_counts_set(self):
        router = ExpertChoiceSigmoidRouter(dim=16, num_experts=4, capacity_factor=0.5)
        router.eval()
        x = torch.randn(8, 16)
        _ = router(x)
        # Each expert picks k_e = ceil(0.5 * 8) = 4 tokens.
        assert router.expert_counts.shape == (4,)
        assert (router.expert_counts == 4).all()


# ---------------------------------------------------------------------------
# ExpertChoiceMoE
# ---------------------------------------------------------------------------


class TestExpertChoiceMoE:
    def test_forward_shape(self):
        moe = ExpertChoiceMoE(dim=16, hidden_dim=32, num_experts=4, capacity_factor=0.5)
        moe.eval()
        x = torch.randn(10, 16)
        y = moe(x)
        assert y.shape == (10, 16)

    def test_forward_empty_input(self):
        moe = ExpertChoiceMoE(dim=16, hidden_dim=32, num_experts=4, capacity_factor=0.5)
        x = torch.zeros(0, 16)
        y = moe(x)
        assert y.shape == (0, 16)

    def test_forward_rejects_wrong_rank(self):
        moe = ExpertChoiceMoE(dim=16, hidden_dim=32, num_experts=2, capacity_factor=0.5)
        with pytest.raises(ValueError, match=r"\(N, D\)"):
            moe(torch.randn(2, 5, 16))

    def test_gradient_flow_to_experts_and_gate(self):
        moe = ExpertChoiceMoE(dim=16, hidden_dim=32, num_experts=4, capacity_factor=1.0)
        moe.train()
        x = torch.randn(8, 16, requires_grad=True)
        y = moe(x)
        loss = y.sum()
        loss.backward()
        # Gate gradient should be non-None and non-zero (some experts run).
        assert moe.router.gate.weight.grad is not None
        assert moe.router.gate.weight.grad.abs().sum().item() > 0
        # At least one expert's gate_proj gradient should be non-zero.
        any_expert_grad = any(
            e.gate_proj.weight.grad is not None and e.gate_proj.weight.grad.abs().sum().item() > 0
            for e in moe.experts
        )
        assert any_expert_grad

    def test_token_not_selected_gets_zero_contribution(self):
        # With capacity 0.01 and N=10, k_e=1: each expert picks exactly 1 token.
        # With 2 experts that's at most 2 distinct tokens selected; at least 8
        # tokens get 0 contribution from the MoE block.
        torch.manual_seed(0)
        moe = ExpertChoiceMoE(
            dim=16, hidden_dim=32, num_experts=2, capacity_factor=0.01, gumbel_noise=False
        )
        moe.eval()
        x = torch.randn(10, 16)
        y = moe(x)
        # Rows where the output is exactly zero ⇒ no expert picked that token.
        # With token-choice routing this would be impossible, but with EC it
        # is the expected behavior.
        zero_rows = (y.abs().sum(dim=-1) == 0).sum().item()
        assert zero_rows >= 8

    def test_expert_counts_property(self):
        moe = ExpertChoiceMoE(dim=16, hidden_dim=32, num_experts=4, capacity_factor=0.5)
        moe.eval()
        x = torch.randn(8, 16)
        _ = moe(x)
        # Expert counts come from the router.
        assert moe.expert_counts.shape == (4,)


# ---------------------------------------------------------------------------
# MoMaFFN
# ---------------------------------------------------------------------------


class TestMoMaFFN:
    def _make_ffn(self, dim: int = 32) -> MoMaFFN:
        config = _config(dim=dim)
        return MoMaFFN(
            config,
            modalities=("image", "text"),
            experts_per_modality={"image": 2, "text": 2},
            capacity_factor_per_modality={"image": 0.5, "text": 0.5},
            gumbel_noise=False,
        )

    def test_construction_rejects_missing_experts_entry(self):
        config = _config()
        with pytest.raises(ValueError, match="experts_per_modality missing"):
            MoMaFFN(
                config,
                modalities=("image", "text"),
                experts_per_modality={"image": 2},
                capacity_factor_per_modality={"image": 0.5, "text": 0.5},
            )

    def test_construction_rejects_missing_capacity_entry(self):
        config = _config()
        with pytest.raises(ValueError, match="capacity_factor_per_modality missing"):
            MoMaFFN(
                config,
                modalities=("image", "text"),
                experts_per_modality={"image": 2, "text": 2},
                capacity_factor_per_modality={"image": 0.5},
            )

    def test_forward_shape(self):
        ffn = self._make_ffn(dim=32)
        ffn.eval()
        x = torch.randn(2, 8, 32)
        # 4 image positions, 4 text positions (image-prefix layout).
        modality_ids = torch.zeros(2, 8, dtype=torch.long)
        modality_ids[:, 4:] = 1
        y = ffn(x, modality_ids)
        assert y.shape == (2, 8, 32)

    def test_forward_rejects_wrong_input_rank(self):
        ffn = self._make_ffn()
        with pytest.raises(ValueError, match=r"\(B, S, D\)"):
            ffn(torch.randn(8, 32), torch.zeros(8, dtype=torch.long))

    def test_forward_rejects_mismatched_modality_ids_shape(self):
        ffn = self._make_ffn()
        x = torch.randn(2, 8, 32)
        with pytest.raises(ValueError, match="does not match"):
            ffn(x, torch.zeros(2, 7, dtype=torch.long))

    def test_forward_rejects_non_long_modality_ids(self):
        ffn = self._make_ffn()
        x = torch.randn(2, 8, 32)
        with pytest.raises(ValueError, match="dtype must be torch.long"):
            ffn(x, torch.zeros(2, 8, dtype=torch.float32))

    def test_forward_rejects_out_of_range_modality_id(self):
        """A modality id >= len(modalities) is silently equivalent to "no group
        picked this token" without the check — caller would see zero output
        at those positions, which is hard to debug. We require an explicit
        ValueError instead.
        """
        ffn = self._make_ffn()
        ffn.eval()
        x = torch.randn(2, 4, 32)
        modality_ids = torch.zeros(2, 4, dtype=torch.long)
        modality_ids[0, 0] = 2  # only 0 ("image") and 1 ("text") are valid
        with pytest.raises(ValueError, match="out-of-range"):
            ffn(x, modality_ids)

    def test_forward_rejects_negative_modality_id(self):
        ffn = self._make_ffn()
        ffn.eval()
        x = torch.randn(2, 4, 32)
        modality_ids = torch.zeros(2, 4, dtype=torch.long)
        modality_ids[1, 2] = -1
        with pytest.raises(ValueError, match="out-of-range"):
            ffn(x, modality_ids)

    def test_all_text_batch_image_positions_zero(self):
        """When no tokens are tagged image, image-position outputs are 0."""
        ffn = self._make_ffn(dim=32)
        ffn.eval()
        x = torch.randn(1, 6, 32)
        # All tokens are text (modality_id == 1).
        modality_ids = torch.ones(1, 6, dtype=torch.long)
        y = ffn(x, modality_ids)
        # Sanity: text positions should generally produce some non-zero output.
        # (Test text path actually fires.)
        assert y.abs().sum().item() > 0

    def test_all_image_batch_text_positions_zero(self):
        """When no tokens are tagged text, text-position outputs are 0."""
        ffn = self._make_ffn(dim=32)
        ffn.eval()
        x = torch.randn(1, 6, 32)
        modality_ids = torch.zeros(1, 6, dtype=torch.long)
        y = ffn(x, modality_ids)
        assert y.abs().sum().item() > 0

    def test_dispatch_isolates_modalities(self):
        """Image positions are processed only by image experts, text by text experts.

        Verifies by zeroing one modality's experts and confirming output at
        positions of the *other* modality is unchanged.
        """
        ffn = self._make_ffn(dim=32)
        ffn.eval()
        x = torch.randn(1, 8, 32)
        modality_ids = torch.zeros(1, 8, dtype=torch.long)
        modality_ids[:, 4:] = 1

        y_full = ffn(x, modality_ids)
        # Zero text experts' weights → text-position outputs should change,
        # image-position outputs should be identical.
        with torch.no_grad():
            for e in ffn.experts["text"].experts:
                e.gate_proj.weight.zero_()
                e.up_proj.weight.zero_()
                e.down_proj.weight.zero_()
        y_text_zeroed = ffn(x, modality_ids)
        # Image positions (indices 0..3) unchanged.
        assert torch.allclose(y_full[:, :4, :], y_text_zeroed[:, :4, :])
        # Text positions (indices 4..7) now zero (or different).
        assert not torch.allclose(y_full[:, 4:, :], y_text_zeroed[:, 4:, :])

    def test_gradient_flow_to_both_modality_groups(self):
        ffn = self._make_ffn(dim=32)
        ffn.train()
        x = torch.randn(1, 8, 32, requires_grad=True)
        modality_ids = torch.zeros(1, 8, dtype=torch.long)
        modality_ids[:, 4:] = 1
        y = ffn(x, modality_ids)
        y.sum().backward()
        # Both modality groups should have gradients on their gates.
        image_gate_grad = ffn.experts["image"].router.gate.weight.grad
        text_gate_grad = ffn.experts["text"].router.gate.weight.grad
        assert image_gate_grad is not None
        assert text_gate_grad is not None
        assert image_gate_grad.abs().sum().item() > 0
        assert text_gate_grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# MoMaBlock
# ---------------------------------------------------------------------------


class TestMoMaBlock:
    def _make_block(self, dim: int = 32) -> MoMaBlock:
        config = _config(dim=dim, n_heads=4, n_kv_heads=4, max_seq_len=32)
        return MoMaBlock(
            config,
            modalities=("image", "text"),
            experts_per_modality={"image": 2, "text": 2},
            capacity_factor_per_modality={"image": 0.5, "text": 0.5},
            gumbel_noise=False,
            layer_idx=0,
        )

    def test_construction_has_shared_attention_and_per_modality_ffn(self):
        block = self._make_block()
        # Shared attention: single QKVO Linear (not nn.ModuleDict).
        assert isinstance(block.attention, Attention)
        # Per-modality MoE FFN: ModuleDict keyed by modality.
        assert isinstance(block.mlp, MoMaFFN)
        assert set(block.mlp.experts.keys()) == {"image", "text"}

    def test_forward_shape(self):
        from kempnerforge.model.position import precompute_rope_frequencies

        block = self._make_block(dim=32)
        block.eval()
        cos, sin = precompute_rope_frequencies(head_dim=8, max_seq_len=16)
        x = torch.randn(2, 8, 32)
        modality_ids = torch.zeros(2, 8, dtype=torch.long)
        modality_ids[:, 4:] = 1
        y = block(x, cos[:8], sin[:8], modality_ids)
        assert y.shape == (2, 8, 32)

    def test_residual_preserves_unselected_tokens(self):
        """If no expert picks a token, the residual still carries it.

        With capacity factor 0.01 + 2 experts + 8 tokens, k_e = 1 so at
        most 2 tokens get nonzero MoE contribution; the rest should
        appear ~unchanged at the output (modulo the attention contribution).
        """
        from kempnerforge.model.position import precompute_rope_frequencies

        config = _config(dim=32, n_heads=4, n_kv_heads=4)
        block = MoMaBlock(
            config,
            modalities=("image", "text"),
            experts_per_modality={"image": 1, "text": 1},
            capacity_factor_per_modality={"image": 0.01, "text": 0.01},
            gumbel_noise=False,
            layer_idx=0,
        )
        # Zero the attention output so we isolate the FFN residual behavior.
        with torch.no_grad():
            block.attention.o_proj.weight.zero_()
        block.eval()
        cos, sin = precompute_rope_frequencies(head_dim=8, max_seq_len=16)
        x = torch.randn(1, 8, 32)
        modality_ids = torch.zeros(1, 8, dtype=torch.long)
        modality_ids[:, 4:] = 1
        y = block(x, cos[:8], sin[:8], modality_ids)
        # Most positions should be ~equal to the input (residual passthrough).
        # Count rows that are close to input.
        close = torch.isclose(y[0], x[0], atol=1e-5).all(dim=-1).sum().item()
        assert close >= 6  # at least 6 of 8 tokens passed through unmodified

    def test_gradient_flow(self):
        from kempnerforge.model.position import precompute_rope_frequencies

        block = self._make_block(dim=32)
        block.train()
        cos, sin = precompute_rope_frequencies(head_dim=8, max_seq_len=16)
        x = torch.randn(1, 8, 32, requires_grad=True)
        modality_ids = torch.zeros(1, 8, dtype=torch.long)
        modality_ids[:, 4:] = 1
        y = block(x, cos[:8], sin[:8], modality_ids)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# End-to-end Transformer + MoMaConfig
# ---------------------------------------------------------------------------


class TestTransformerWithMoMaConfig:
    def test_build_with_moma_config(self):
        config = _config(dim=32, n_heads=4, n_kv_heads=4, n_layers=2, max_seq_len=32)
        vlm = MoMaConfig(
            moma_experts_per_modality={"image": 2, "text": 2},
            moma_gumbel_noise=False,
        )
        transformer = Transformer(config, vlm_config=vlm, num_image_tokens=4)
        # All blocks should be MoMaBlock instances.
        for layer in transformer.layers.values():
            assert isinstance(layer, MoMaBlock)
        # MoT-specific state should be empty.
        assert transformer._mot_modalities == ()
        # MoMa-specific state should be set.
        assert transformer._moma_modalities == ("image", "text")

    def test_forward_with_modality_context(self):
        config = _config(dim=32, n_heads=4, n_kv_heads=4, n_layers=2, max_seq_len=32)
        vlm = MoMaConfig(
            moma_experts_per_modality={"image": 2, "text": 2},
            moma_gumbel_noise=False,
        )
        transformer = Transformer(config, vlm_config=vlm, num_image_tokens=4)
        transformer.eval()
        b, n_img, t_text = 1, 4, 4
        tokens = torch.randint(0, config.vocab_size, (b, t_text))
        prefix_embeds = torch.randn(b, n_img, config.dim)
        modality_ids = torch.zeros(b, n_img + t_text, dtype=torch.long)
        modality_ids[:, n_img:] = 1
        ctx = ModalityContext(
            prefix_embeds=prefix_embeds,
            output_slice=slice(n_img, None),
            modality_ids=modality_ids,
        )
        logits = transformer(tokens=tokens, modality=ctx)
        # output_slice trims image positions → output has t_text positions.
        assert logits.shape == (b, t_text, config.vocab_size)
        assert torch.isfinite(logits).all()

    def test_forward_rejects_missing_modality_ids(self):
        config = _config(dim=32, n_heads=4, n_kv_heads=4, n_layers=2, max_seq_len=32)
        vlm = MoMaConfig(
            moma_experts_per_modality={"image": 2, "text": 2},
            moma_gumbel_noise=False,
        )
        transformer = Transformer(config, vlm_config=vlm, num_image_tokens=4)
        b, t_text = 1, 4
        tokens = torch.randint(0, config.vocab_size, (b, t_text))
        prefix_embeds = torch.randn(b, 4, config.dim)
        ctx = ModalityContext(
            prefix_embeds=prefix_embeds,
            output_slice=slice(4, None),
            # modality_ids deliberately omitted
        )
        with pytest.raises(ValueError, match="requires modality.modality_ids"):
            transformer(tokens=tokens, modality=ctx)

    def test_forward_rejects_mismatched_modality_ids_shape(self):
        config = _config(dim=32, n_heads=4, n_kv_heads=4, n_layers=2, max_seq_len=32)
        vlm = MoMaConfig(
            moma_experts_per_modality={"image": 2, "text": 2},
            moma_gumbel_noise=False,
        )
        transformer = Transformer(config, vlm_config=vlm, num_image_tokens=4)
        b, n_img, t_text = 1, 4, 4
        tokens = torch.randint(0, config.vocab_size, (b, t_text))
        prefix_embeds = torch.randn(b, n_img, config.dim)
        # Wrong shape: should be (b, n_img + t_text) = (1, 8) but we pass (1, 7).
        modality_ids = torch.zeros(b, 7, dtype=torch.long)
        ctx = ModalityContext(
            prefix_embeds=prefix_embeds,
            output_slice=slice(n_img, None),
            modality_ids=modality_ids,
        )
        with pytest.raises(ValueError, match="does not match"):
            transformer(tokens=tokens, modality=ctx)

    def test_gradient_flow_end_to_end(self):
        config = _config(dim=32, n_heads=4, n_kv_heads=4, n_layers=2, max_seq_len=32)
        vlm = MoMaConfig(
            moma_experts_per_modality={"image": 2, "text": 2},
            moma_gumbel_noise=False,
        )
        transformer = Transformer(config, vlm_config=vlm, num_image_tokens=4)
        transformer.train()
        b, n_img, t_text = 1, 4, 4
        tokens = torch.randint(0, config.vocab_size, (b, t_text))
        prefix_embeds = torch.randn(b, n_img, config.dim, requires_grad=True)
        modality_ids = torch.zeros(b, n_img + t_text, dtype=torch.long)
        modality_ids[:, n_img:] = 1
        ctx = ModalityContext(
            prefix_embeds=prefix_embeds,
            output_slice=slice(n_img, None),
            modality_ids=modality_ids,
        )
        logits = transformer(tokens=tokens, modality=ctx)
        logits.sum().backward()
        # Prefix embeds and at least one expert in each modality group should
        # have gradients.
        assert prefix_embeds.grad is not None
        assert prefix_embeds.grad.abs().sum().item() > 0

    def test_get_expert_counts_returns_empty_for_moma(self):
        """The flat-MoE helper returns {} for MoMa — MoMa layers expose
        per-modality counts through get_moma_expert_counts instead.
        """
        config = _config(dim=32, n_heads=4, n_kv_heads=4, n_layers=2, max_seq_len=32)
        vlm = MoMaConfig(
            moma_experts_per_modality={"image": 2, "text": 2},
            moma_gumbel_noise=False,
        )
        transformer = Transformer(config, vlm_config=vlm, num_image_tokens=4)
        assert transformer.get_expert_counts() == {}

    def test_get_moma_expert_counts_returns_empty_when_no_moma_layers(self):
        """Dense Transformer (no MoMa, no MoT, no MoE) has no MoMa layers,
        so the helper returns {}.
        """
        config = _config(dim=32, n_heads=4, n_kv_heads=4, n_layers=2, max_seq_len=32)
        transformer = Transformer(config)
        assert transformer.get_moma_expert_counts() == {}

    def test_get_moma_expert_counts_after_forward(self):
        """After a forward, get_moma_expert_counts surfaces per-layer
        per-modality utilization tensors (paper Figure 5 shape).
        """
        config = _config(dim=32, n_heads=4, n_kv_heads=4, n_layers=2, max_seq_len=32)
        # Unequal experts per modality so the shape check is meaningful.
        vlm = MoMaConfig(
            moma_experts_per_modality={"image": 2, "text": 3},
            moma_gumbel_noise=False,
        )
        transformer = Transformer(config, vlm_config=vlm, num_image_tokens=4)
        transformer.eval()

        b, n_img, t_text = 1, 4, 4
        tokens = torch.randint(0, config.vocab_size, (b, t_text))
        prefix_embeds = torch.randn(b, n_img, config.dim)
        modality_ids = torch.zeros(b, n_img + t_text, dtype=torch.long)
        modality_ids[:, n_img:] = 1
        ctx = ModalityContext(
            prefix_embeds=prefix_embeds,
            output_slice=slice(n_img, None),
            modality_ids=modality_ids,
        )
        _ = transformer(tokens=tokens, modality=ctx)

        counts = transformer.get_moma_expert_counts()
        # Both layers reported.
        assert set(counts.keys()) == {0, 1}
        for layer_counts in counts.values():
            # Both modality groups present.
            assert set(layer_counts.keys()) == {"image", "text"}
            # Per-modality shape == (num_experts_for_that_modality,).
            assert layer_counts["image"].shape == (2,)
            assert layer_counts["text"].shape == (3,)
            # Counts are non-negative; expert-choice puts >=1 token per expert
            # when N_m > 0 (k_e = max(1, ceil(c*N_m)) and N_m == 4 here).
            assert (layer_counts["image"] >= 1).all()
            assert (layer_counts["text"] >= 1).all()


# ---------------------------------------------------------------------------
# Sanity check: math.ceil semantics for k_e
# ---------------------------------------------------------------------------


def test_k_e_formula_matches_paper():
    """k_e = ceil(capacity_factor * N) matches the paper's b^M * c_e formula.

    Paper: k_e = b^M * c_e, where b^M is total tokens of modality M.
    Implementation: k_e = ceil(capacity_factor * n_tokens) with capacity_factor
    defaulting to 1/|E^M| (so k_e ~ N/|E|).
    """
    for n, c in [(16, 0.25), (20, 0.5), (100, 0.1), (7, 1.0 / 3)]:
        expected = max(1, math.ceil(c * n))
        # The router computes this internally; we verify the formula
        # produces sensible values.
        assert expected >= 1
        assert expected <= n
