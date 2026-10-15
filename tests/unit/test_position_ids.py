"""Unit tests for explicit per-token position ids in Transformer.forward.

Covers the gathered-RoPE math against hand-computed rotations, per-row
position independence, the packed position-reset acceptance property, strict
input validation, gradient flow, and the unchanged default path.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.attention import KVCache
from kempnerforge.model.position import apply_rope, precompute_rope_frequencies
from kempnerforge.model.transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DIM = 64
_N_HEADS = 4
_HEAD_DIM = _DIM // _N_HEADS
_VOCAB = 97
_MAX_SEQ_LEN = 32


def _tiny_model(n_layers: int = 2, seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    config = ModelConfig(
        dim=_DIM,
        n_layers=n_layers,
        n_heads=_N_HEADS,
        vocab_size=_VOCAB,
        max_seq_len=_MAX_SEQ_LEN,
    )
    return Transformer(config).to(DEVICE).eval()


# ---------------------------------------------------------------------------
# apply_rope with gathered (batch, seq_len, head_dim // 2) frequencies
# ---------------------------------------------------------------------------


class TestApplyRopeGathered:
    def test_hand_computed_rotation_at_gathered_positions(self):
        """Gathered cos/sin rotate each token by its own position's angles.

        head_dim=4 and theta=10000 give pair frequencies f0 = 1.0 and
        f1 = 10000^(-2/4) = 0.01; a token at position p with values
        (a, b, c, e) must become
        (a cos(p f0) - c sin(p f0), b cos(p f1) - e sin(p f1),
         c cos(p f0) + a sin(p f0), e cos(p f1) + b sin(p f1)).
        """
        head_dim = 4
        freqs = [1.0, 0.01]
        cos_table, sin_table = precompute_rope_frequencies(head_dim, max_seq_len=8)
        position_ids = torch.tensor([[5, 2, 0], [1, 7, 3]])

        # x: (batch=2, heads=1, seq=3, head_dim=4), distinct values per token
        x = torch.arange(1.0, 1.0 + 2 * 3 * head_dim).reshape(2, 1, 3, head_dim)

        out = apply_rope(x, cos_table[position_ids], sin_table[position_ids])

        expected = torch.empty_like(x)
        for row in range(2):
            for t in range(3):
                p = int(position_ids[row, t])
                a, b, c, e = (float(v) for v in x[row, 0, t])
                expected[row, 0, t] = torch.tensor(
                    [
                        a * math.cos(p * freqs[0]) - c * math.sin(p * freqs[0]),
                        b * math.cos(p * freqs[1]) - e * math.sin(p * freqs[1]),
                        c * math.cos(p * freqs[0]) + a * math.sin(p * freqs[0]),
                        e * math.cos(p * freqs[1]) + b * math.sin(p * freqs[1]),
                    ]
                )
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_batched_matches_per_row_shared(self):
        """(B, S, d) frequencies equal applying each row's (S, d) separately."""
        torch.manual_seed(1)
        cos_table, sin_table = precompute_rope_frequencies(8, max_seq_len=16)
        position_ids = torch.tensor([[0, 1, 2, 3], [7, 3, 0, 11]])
        x = torch.randn(2, 3, 4, 8)

        batched = apply_rope(x, cos_table[position_ids], sin_table[position_ids])
        for row in range(2):
            per_row = apply_rope(
                x[row : row + 1], cos_table[position_ids[row]], sin_table[position_ids[row]]
            )
            assert torch.equal(batched[row : row + 1], per_row)

    def test_rejects_wrong_frequency_ndim(self):
        x = torch.randn(1, 1, 4, 8)
        bad = torch.randn(4)
        with pytest.raises(ValueError, match="got 1 dims"):
            apply_rope(x, bad, bad)


# ---------------------------------------------------------------------------
# Transformer.forward(position_ids=...)
# ---------------------------------------------------------------------------


class TestTransformerPositionIds:
    def test_contiguous_position_ids_match_default(self):
        """Explicit arange positions reproduce the default path bit-exactly."""
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (2, 10), device=DEVICE)
        position_ids = torch.arange(10, device=DEVICE).expand(2, -1).contiguous()
        with torch.no_grad():
            default = model(tokens)
            explicit = model(tokens, position_ids=position_ids)
        assert torch.equal(default, explicit)

    def test_repeated_document_at_two_pack_offsets(self):
        """A document packed at offsets 0 and L has identical fp32 outputs
        when positions reset per document, and different ones when they
        keep counting through the pack (the control that makes the
        equality meaningful)."""
        model = _tiny_model()
        L = 8
        doc = torch.randint(0, _VOCAB, (1, L), device=DEVICE)
        packed = torch.cat([doc, doc], dim=1)
        doc_ids = torch.cat(
            [torch.zeros(1, L, dtype=torch.long), torch.ones(1, L, dtype=torch.long)], dim=1
        ).to(DEVICE)
        reset_positions = torch.arange(L, device=DEVICE).repeat(2).unsqueeze(0)

        with torch.no_grad():
            out_reset = model(packed, doc_ids=doc_ids, position_ids=reset_positions)
            out_absolute = model(packed, doc_ids=doc_ids)

        assert torch.equal(out_reset[:, :L], out_reset[:, L:])
        assert not torch.equal(out_absolute[:, :L], out_absolute[:, L:])

    def test_per_row_positions_are_independent(self):
        """Rows with different positions match the same rows run alone, and
        positions change the output (restarting vs contiguous rows differ)."""
        model = _tiny_model()
        S = 8
        row = torch.randint(0, _VOCAB, (1, S), device=DEVICE)
        tokens = row.expand(2, -1).contiguous()
        position_ids = torch.stack(
            [
                torch.arange(S, device=DEVICE),
                torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=DEVICE),
            ]
        )
        with torch.no_grad():
            batched = model(tokens, position_ids=position_ids)
            row0 = model(tokens[:1], position_ids=position_ids[:1])
            row1 = model(tokens[1:], position_ids=position_ids[1:])
        torch.testing.assert_close(batched[0], row0[0])
        torch.testing.assert_close(batched[1], row1[0])
        assert not torch.allclose(batched[0], batched[1])

    def test_gradient_flows_through_gathered_rope(self):
        model = _tiny_model().train()
        tokens = torch.randint(0, _VOCAB, (2, 6), device=DEVICE)
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2], [4, 5, 6, 7, 8, 9]], device=DEVICE)
        model(tokens, position_ids=position_ids).pow(2).mean().backward()

        emb_grad = model.token_embedding.embedding.weight.grad  # type: ignore[union-attr]
        q_grad = model.layers["0"].attention.q_proj.weight.grad
        assert emb_grad is not None and q_grad is not None
        assert torch.isfinite(emb_grad).all() and torch.isfinite(q_grad).all()
        assert emb_grad.abs().sum() > 0 and q_grad.abs().sum() > 0

    def test_default_state_dict_has_no_position_entries(self):
        model = _tiny_model()
        assert all(
            "position" not in key and "rope" not in key.lower() for key in model.state_dict()
        )

    def test_position_ids_defaults_to_none(self):
        assert inspect.signature(Transformer.forward).parameters["position_ids"].default is None


class TestTransformerPositionIdsValidation:
    def _tokens(self, batch: int = 2, seq: int = 6) -> torch.Tensor:
        return torch.randint(0, _VOCAB, (batch, seq), device=DEVICE)

    def test_rejects_wrong_dtype(self):
        model = _tiny_model(n_layers=1)
        bad = torch.zeros(2, 6, dtype=torch.int32, device=DEVICE)
        with pytest.raises(ValueError, match="dtype must be torch.int64"):
            model(self._tokens(), position_ids=bad)

    def test_rejects_wrong_ndim(self):
        model = _tiny_model(n_layers=1)
        bad = torch.zeros(6, dtype=torch.long, device=DEVICE)
        with pytest.raises(ValueError, match=r"must have shape \(batch, seq_len\)"):
            model(self._tokens(), position_ids=bad)

    @pytest.mark.parametrize("shape", [(2, 5), (3, 6)])
    def test_rejects_shape_mismatch(self, shape):
        model = _tiny_model(n_layers=1)
        bad = torch.zeros(*shape, dtype=torch.long, device=DEVICE)
        with pytest.raises(ValueError, match="does not match the residual shape"):
            model(self._tokens(), position_ids=bad)

    @pytest.mark.parametrize("value", [-1, _MAX_SEQ_LEN])
    def test_rejects_out_of_bounds_values(self, value):
        model = _tiny_model(n_layers=1)
        bad = torch.zeros(2, 6, dtype=torch.long, device=DEVICE)
        bad[1, 3] = value
        with pytest.raises(ValueError, match=r"must be in \[0, "):
            model(self._tokens(), position_ids=bad)

    def test_rejects_kv_caches_combination(self):
        model = _tiny_model(n_layers=1)
        caches = [
            KVCache(
                batch_size=1,
                max_seq_len=_MAX_SEQ_LEN,
                n_kv_heads=_N_HEADS,
                head_dim=_HEAD_DIM,
                dtype=torch.float32,
                device=DEVICE,
            )
        ]
        tokens = self._tokens(batch=1, seq=4)
        position_ids = torch.arange(4, device=DEVICE).unsqueeze(0)
        with pytest.raises(ValueError, match="cannot be combined with kv_caches"):
            model(tokens, kv_caches=caches, position_ids=position_ids)

    def test_rejects_mot_models(self):
        from kempnerforge.config.vlm import MoTConfig

        config = ModelConfig(
            dim=_DIM,
            n_layers=1,
            n_heads=_N_HEADS,
            n_kv_heads=_N_HEADS,
            vocab_size=_VOCAB,
            max_seq_len=_MAX_SEQ_LEN,
            ffn_hidden_dim=128,
        )
        model = Transformer(config, vlm_config=MoTConfig(max_text_len=8), num_image_tokens=4).to(
            DEVICE
        )
        tokens = self._tokens(batch=1, seq=8)
        position_ids = torch.arange(8, device=DEVICE).unsqueeze(0)
        with pytest.raises(ValueError, match="not supported for MoT"):
            model(tokens, position_ids=position_ids)
