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
from kempnerforge.config.vlm import MoMaConfig
from kempnerforge.model.attention import KVCache
from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.position import apply_rope, precompute_rope_frequencies
from kempnerforge.model.transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DIM = 64
_N_HEADS = 4
_HEAD_DIM = _DIM // _N_HEADS
_VOCAB = 97
_MAX_SEQ_LEN = 32


def _tiny_model(n_layers: int = 2, seed: int = 0, **config_kw) -> Transformer:
    torch.manual_seed(seed)
    config = ModelConfig(
        dim=_DIM,
        n_layers=n_layers,
        n_heads=_N_HEADS,
        vocab_size=_VOCAB,
        max_seq_len=_MAX_SEQ_LEN,
        **config_kw,
    )
    return Transformer(config).to(DEVICE).eval()


def _embedding_calls(model: Transformer) -> list[int]:
    """Record each call to the token embedding, the first compute in forward."""
    calls: list[int] = []
    model.token_embedding.register_forward_pre_hook(  # type: ignore[union-attr]
        lambda module, args: calls.append(1)
    )
    return calls


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


class TestPositionIdsFromTheContext:
    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_context_positions_match_the_argument(self, backend):
        """With a group mask too, so the flex backend builds its BlockMask."""
        model = _tiny_model(attention_backend=backend)
        tokens = torch.randint(0, _VOCAB, (2, 10), device=DEVICE)
        gen = torch.Generator().manual_seed(1)
        position_ids = torch.randint(0, _MAX_SEQ_LEN, (2, 10), generator=gen).to(DEVICE)
        groups = torch.full((2, 10), -1, device=DEVICE)
        groups[:, :4] = 0
        with torch.no_grad():
            argument = model(tokens, position_ids=position_ids, attention_group_ids=groups)
            context = model(
                tokens,
                modality=ModalityContext(position_ids=position_ids, attention_group_ids=groups),
            )
            default = model(tokens, attention_group_ids=groups)
        assert torch.equal(argument, context)
        assert not torch.equal(default, context)

    def test_context_positions_cover_the_prefix(self):
        """A strategy's positions span the residual, prefix included, and are checked so."""
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (1, 6), device=DEVICE)
        prefix = torch.randn(1, 4, _DIM, device=DEVICE)
        positions = torch.arange(10, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            default = model(tokens, modality=ModalityContext(prefix_embeds=prefix))
            explicit = model(
                tokens, modality=ModalityContext(prefix_embeds=prefix, position_ids=positions)
            )
        assert torch.equal(default, explicit)
        with pytest.raises(ValueError, match="does not match the residual"):
            model(
                tokens,
                modality=ModalityContext(prefix_embeds=prefix, position_ids=positions[:, :6]),
            )

    def test_argument_and_context_together_raise(self):
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (1, 5), device=DEVICE)
        positions = torch.arange(5, device=DEVICE).unsqueeze(0)
        with pytest.raises(ValueError, match="position_ids was passed to Transformer.forward"):
            model(tokens, position_ids=positions, modality=ModalityContext(position_ids=positions))


def _caches(model: Transformer, batch: int = 1) -> list[KVCache]:
    return [
        KVCache(
            batch_size=batch,
            max_seq_len=_MAX_SEQ_LEN,
            n_kv_heads=_N_HEADS,
            head_dim=_HEAD_DIM,
            dtype=torch.float32,
            device=DEVICE,
        )
        for _ in range(len(model.layers))
    ]


class TestPositionIdsWithKVCache:
    """Caller positions apply to the appended tokens, so cached decoding matches a full forward."""

    @staticmethod
    def _positions() -> torch.Tensor:
        return torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]], device=DEVICE)

    def test_prefill_then_decode_matches_the_full_forward(self):
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (1, 8), device=DEVICE)
        positions = self._positions()
        caches = _caches(model)
        with torch.no_grad():
            full = model(tokens, position_ids=positions)
            prefill = model(tokens[:, :5], kv_caches=caches, position_ids=positions[:, :5])
            steps = [
                model(tokens[:, i : i + 1], kv_caches=caches, position_ids=positions[:, i : i + 1])
                for i in range(5, 8)
            ]
        cached = torch.cat([prefill, *steps], dim=1)
        assert caches[0].seq_len == 8
        torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-5)

    def test_context_positions_decode_the_same_way(self):
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (1, 8), device=DEVICE)
        positions = self._positions()
        caches = _caches(model)
        with torch.no_grad():
            full = model(tokens, modality=ModalityContext(position_ids=positions))
            prefill = model(
                tokens[:, :5],
                kv_caches=caches,
                modality=ModalityContext(position_ids=positions[:, :5]),
            )
            steps = [
                model(
                    tokens[:, i : i + 1],
                    kv_caches=caches,
                    modality=ModalityContext(position_ids=positions[:, i : i + 1]),
                )
                for i in range(5, 8)
            ]
        torch.testing.assert_close(torch.cat([prefill, *steps], dim=1), full, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("route", ["default", "argument", "context", "rotation"])
    @pytest.mark.parametrize("capture", [False, True], ids=["sdpa", "weights"])
    def test_appending_several_tokens_matches_the_full_forward(self, route, capture):
        """Tokens appended to a non-empty cache see every cached key, on both attention paths."""
        model = _tiny_model()
        for layer in model.layers.values():
            layer.attention.capture_attention_weights = capture
        tokens = torch.randint(0, _VOCAB, (2, 16), device=DEVICE)
        positions = torch.arange(16, device=DEVICE).repeat(2, 1)
        if route != "default":
            positions = torch.tensor([[0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11]] * 2).to(
                DEVICE
            )

        def kwargs(sl: slice) -> dict:
            if route == "default":
                return {}
            if route == "argument":
                return {"position_ids": positions[:, sl]}
            if route == "context":
                return {"modality": ModalityContext(position_ids=positions[:, sl])}
            return {"modality": TestRopeFromTheContext._rotation(model, positions[:, sl])}

        caches = _caches(model, batch=2)
        with torch.no_grad():
            full = model(tokens, **kwargs(slice(None)))
            prefill = model(tokens[:, :10], kv_caches=caches, **kwargs(slice(0, 10)))
            appended = model(tokens[:, 10:], kv_caches=caches, **kwargs(slice(10, 16)))
        assert caches[0].seq_len == 16
        torch.testing.assert_close(
            torch.cat([prefill, appended], dim=1), full, rtol=1e-5, atol=1e-5
        )

    def test_decode_step_positions_are_checked_against_the_step(self):
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (1, 4), device=DEVICE)
        caches = _caches(model)
        with torch.no_grad():
            model(
                tokens[:, :3], kv_caches=caches, position_ids=torch.arange(3, device=DEVICE)[None]
            )
        with pytest.raises(ValueError, match="does not match the residual"):
            model(
                tokens[:, 3:], kv_caches=caches, position_ids=torch.arange(4, device=DEVICE)[None]
            )


class TestRopeFromTheContext:
    """A strategy may hand the transformer the rotation itself, built from the same table."""

    @staticmethod
    def _rotation(model: Transformer, positions: torch.Tensor) -> ModalityContext:
        cos, sin = precompute_rope_frequencies(
            head_dim=_HEAD_DIM, max_seq_len=_MAX_SEQ_LEN, theta=model.config.rope_theta
        )
        cos, sin = cos.to(DEVICE), sin.to(DEVICE)
        return ModalityContext(rope_cos=cos[positions], rope_sin=sin[positions])

    def test_arange_rotation_matches_the_default(self):
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (2, 10), device=DEVICE)
        positions = torch.arange(10, device=DEVICE).expand(2, -1)
        with torch.no_grad():
            default = model(tokens)
            given = model(tokens, modality=self._rotation(model, positions))
        torch.testing.assert_close(given, default, rtol=0, atol=0)

    def test_gathered_rotation_matches_position_ids(self):
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (2, 10), device=DEVICE)
        gen = torch.Generator().manual_seed(2)
        positions = torch.randint(0, _MAX_SEQ_LEN, (2, 10), generator=gen).to(DEVICE)
        with torch.no_grad():
            by_ids = model(tokens, position_ids=positions)
            by_rotation = model(tokens, modality=self._rotation(model, positions))
        torch.testing.assert_close(by_rotation, by_ids, rtol=0, atol=0)

    def test_a_two_axis_rotation_differs_from_any_one_axis(self):
        """Pair sections rotated by different axes are a rotation no position id can give."""
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (1, 6), device=DEVICE)
        cos, sin = precompute_rope_frequencies(
            head_dim=_HEAD_DIM, max_seq_len=_MAX_SEQ_LEN, theta=model.config.rope_theta
        )
        rows = torch.tensor([0, 0, 0, 1, 1, 1])
        cols = torch.tensor([0, 1, 2, 0, 1, 2])
        half = _HEAD_DIM // 4
        two_axis_cos = torch.cat([cos[rows, :half], cos[cols, half:]], dim=-1).unsqueeze(0)
        two_axis_sin = torch.cat([sin[rows, :half], sin[cols, half:]], dim=-1).unsqueeze(0)
        modality = ModalityContext(
            rope_cos=two_axis_cos.to(DEVICE), rope_sin=two_axis_sin.to(DEVICE)
        )
        with torch.no_grad():
            two_axis = model(tokens, modality=modality)
            one_axis = [model(tokens, position_ids=p.unsqueeze(0).to(DEVICE)) for p in (rows, cols)]
        assert two_axis.shape == (1, 6, _VOCAB)
        assert all(not torch.allclose(two_axis, o) for o in one_axis)

    def test_rotation_covers_the_prefix_and_is_checked(self):
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (1, 6), device=DEVICE)
        prefix = torch.randn(1, 4, _DIM, device=DEVICE)
        ctx = self._rotation(model, torch.arange(10, device=DEVICE).unsqueeze(0))
        with torch.no_grad():
            default = model(tokens, modality=ModalityContext(prefix_embeds=prefix))
            given = model(
                tokens,
                modality=ModalityContext(
                    prefix_embeds=prefix, rope_cos=ctx.rope_cos, rope_sin=ctx.rope_sin
                ),
            )
        torch.testing.assert_close(given, default, rtol=0, atol=0)
        short = self._rotation(model, torch.arange(6, device=DEVICE).unsqueeze(0))
        with pytest.raises(ValueError, match="does not match the residual rotation shape"):
            model(
                tokens,
                modality=ModalityContext(
                    prefix_embeds=prefix, rope_cos=short.rope_cos, rope_sin=short.rope_sin
                ),
            )

    def test_gradient_flows_through_a_given_rotation(self):
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (1, 5), device=DEVICE)
        ctx = self._rotation(model, torch.arange(5, device=DEVICE).unsqueeze(0))
        ctx.rope_cos.requires_grad_(True)  # type: ignore[union-attr]
        ctx.rope_sin.requires_grad_(True)  # type: ignore[union-attr]
        model(tokens, modality=ctx).sum().backward()
        assert ctx.rope_cos.grad is not None and ctx.rope_cos.grad.abs().sum() > 0  # type: ignore[union-attr]
        assert ctx.rope_sin.grad is not None and ctx.rope_sin.grad.abs().sum() > 0  # type: ignore[union-attr]

    def test_a_rotation_needs_no_rope_tables(self):
        """A model whose tables were never built still applies a given rotation."""
        model = _tiny_model()
        model._rope_cos = model._rope_sin = None
        tokens = torch.randint(0, _VOCAB, (1, 5), device=DEVICE)
        ctx = self._rotation(model, torch.arange(5, device=DEVICE).unsqueeze(0))
        with torch.no_grad():
            model(tokens, modality=ctx)
        with pytest.raises(ValueError, match="RoPE tables not initialized"):
            model(tokens)

    def test_rejects_position_ids_alongside(self):
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (1, 5), device=DEVICE)
        ctx = self._rotation(model, torch.arange(5, device=DEVICE).unsqueeze(0))
        with pytest.raises(ValueError, match="cannot be combined with position_ids"):
            model(tokens, position_ids=torch.arange(5, device=DEVICE).unsqueeze(0), modality=ctx)

    def test_prefill_then_decode_matches_the_full_forward(self):
        model = _tiny_model()
        tokens = torch.randint(0, _VOCAB, (1, 8), device=DEVICE)
        positions = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]], device=DEVICE)
        caches = _caches(model)
        with torch.no_grad():
            full = model(tokens, modality=self._rotation(model, positions))
            prefill = model(
                tokens[:, :5], kv_caches=caches, modality=self._rotation(model, positions[:, :5])
            )
            steps = [
                model(
                    tokens[:, i : i + 1],
                    kv_caches=caches,
                    modality=self._rotation(model, positions[:, i : i + 1]),
                )
                for i in range(5, 8)
            ]
        torch.testing.assert_close(torch.cat([prefill, *steps], dim=1), full, rtol=1e-5, atol=1e-5)

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
        tokens = torch.randint(0, _VOCAB, (1, 8), device=DEVICE)
        ctx = self._rotation(model, torch.arange(8, device=DEVICE).unsqueeze(0))
        with pytest.raises(ValueError, match="not supported for MoT"):
            model(tokens, modality=ctx)


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

    def _assert_rejected_before_embedding(self, bad, match, n_prefix=0):
        model = _tiny_model(n_layers=1)
        calls = _embedding_calls(model)
        tokens = self._tokens()
        modality = (
            ModalityContext(prefix_embeds=torch.randn(2, n_prefix, _DIM, device=DEVICE))
            if n_prefix
            else None
        )
        with pytest.raises(ValueError, match=match):
            model(tokens, modality=modality, position_ids=bad)
        assert calls == []

        good = torch.arange(n_prefix + tokens.shape[1], device=DEVICE).expand(2, -1).contiguous()
        with torch.no_grad():
            model(tokens, modality=modality, position_ids=good)
        assert calls == [1]

    def test_shape_mismatch_rejected_before_embedding(self):
        bad = torch.zeros(2, 5, dtype=torch.long, device=DEVICE)
        self._assert_rejected_before_embedding(bad, r"residual shape \(2, 6\)")

    def test_out_of_range_id_rejected_before_embedding(self):
        bad = torch.zeros(2, 6, dtype=torch.long, device=DEVICE)
        bad[1, 3] = _MAX_SEQ_LEN
        self._assert_rejected_before_embedding(bad, rf"must be in \[0, {_MAX_SEQ_LEN}\)")

    def test_text_length_ids_rejected_when_prefix_embeds_set(self):
        """With 3 prefix positions and 6 text tokens the residual is 9 long."""
        bad = torch.zeros(2, 6, dtype=torch.long, device=DEVICE)
        self._assert_rejected_before_embedding(bad, r"residual shape \(2, 9\)", n_prefix=3)

    def test_uninitialized_rope_tables_rejected_before_embedding(self):
        """A model materialized with to_empty() has no RoPE tables until
        init_weights_and_freqs() runs, so position_ids cannot be validated or used."""
        config = ModelConfig(
            dim=_DIM, n_layers=1, n_heads=_N_HEADS, vocab_size=_VOCAB, max_seq_len=_MAX_SEQ_LEN
        )
        with torch.device("meta"):
            model = Transformer(config)
        model.to_empty(device=DEVICE)
        calls = _embedding_calls(model)
        tokens = self._tokens()
        good = torch.arange(6, device=DEVICE).expand(2, -1).contiguous()

        msg = r"RoPE tables not initialized; call init_weights_and_freqs\(\)"
        with pytest.raises(ValueError, match=msg):
            model(tokens, position_ids=good)
        assert calls == []

        model.init_weights_and_freqs()
        with torch.no_grad():
            model(tokens, position_ids=good)
        assert calls == [1]

    def test_inputs_embeds_route_checks_its_own_length(self):
        model = _tiny_model(n_layers=1)
        modality = ModalityContext(inputs_embeds=torch.randn(2, 6, _DIM, device=DEVICE))
        bad = torch.zeros(2, 5, dtype=torch.long, device=DEVICE)
        with pytest.raises(ValueError, match=r"residual shape \(2, 6\)"):
            model(modality=modality, position_ids=bad)

        contiguous = torch.arange(6, device=DEVICE).expand(2, -1).contiguous()
        with torch.no_grad():
            explicit = model(modality=modality, position_ids=contiguous)
            default = model(modality=modality)
        assert torch.equal(default, explicit)

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


class TestPositionIdsMoMa:
    def test_contiguous_positions_match_default_and_restart_differs(self):
        """On the MoMa arch (image prefix, per-modality MoE FFN) explicit arange
        positions reproduce the default forward bit-exactly, and positions that
        restart at the text segment change the output (the control that makes
        the equality meaningful). CPU only, where the expert-choice index_add
        accumulation is deterministic."""
        n_img, n_text = 4, 4
        config = ModelConfig(
            dim=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            vocab_size=_VOCAB,
            max_seq_len=_MAX_SEQ_LEN,
            ffn_hidden_dim=128,
        )
        vlm = MoMaConfig(moma_experts_per_modality={"image": 2, "text": 2}, moma_gumbel_noise=False)
        torch.manual_seed(0)
        model = Transformer(config, vlm_config=vlm, num_image_tokens=n_img).eval()

        tokens = torch.randint(0, _VOCAB, (1, n_text))
        modality_ids = torch.zeros(1, n_img + n_text, dtype=torch.long)
        modality_ids[:, n_img:] = 1
        modality = ModalityContext(
            prefix_embeds=torch.randn(1, n_img, config.dim),
            output_slice=slice(n_img, None),
            modality_ids=modality_ids,
        )
        contiguous = torch.arange(n_img + n_text).unsqueeze(0)
        restart = torch.cat([torch.arange(n_img), torch.arange(n_text)]).unsqueeze(0)

        with torch.no_grad():
            default = model(tokens, modality=modality)
            explicit = model(tokens, modality=modality, position_ids=contiguous)
            restarted = model(tokens, modality=modality, position_ids=restart)
        assert torch.equal(default, explicit)
        assert not torch.allclose(default, restarted)
