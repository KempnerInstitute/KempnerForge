"""Unit tests for caller-defined bidirectional visibility groups.

The predicate under test, on both attention backends:

    visible(q, k) = same_document(q, k) AND (k <= q OR same_nonnegative_group(q, k))

Group ids are a runtime input ``attention_group_ids`` of shape
``(batch, seq_len)``, int64; negative means no group. Expectations are
enumerated by hand (``PACK_VISIBLE`` and the per-test tables) or by
``oracle_mask``, a plain-Python loop over every (q, k) pair that is itself
pinned to the hand tables.

FlexAttention has no CPU backward in torch 2.11, so flex coverage here is
forward-only; dense-vs-flex gradient parity is covered on GPU in
``tests/integration/test_attention_groups.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask

from kempnerforge.config.schema import ModelConfig
from kempnerforge.config.vlm import MoMaConfig, MoTConfig
from kempnerforge.model.attention import Attention, KVCache
from kempnerforge.model.masking import (
    build_doc_causal_block_mask,
    flex_attention_fn,
    same_group_mask,
)
from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.transformer import Transformer

CPU = torch.device("cpu")


def _t(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def _from_visible(visible: list[set[int]]) -> torch.Tensor:
    """(S, S) bool matrix from the set of visible keys of each query."""
    seq_len = len(visible)
    return torch.tensor([[k in keys for k in range(seq_len)] for keys in visible])


def _block_diag(*blocks: torch.Tensor) -> torch.Tensor:
    return torch.block_diag(*(b.int() for b in blocks)).bool()


def _ones(n: int) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool)


# One pack covering every case: documents 0 and 1, groups 0 and 1, group 1
# reused in both documents, repeated negative ids (-1 in document 0, -3 in
# document 1), and a pad tail under its own sentinel document id (-1).
PACK_DOCS = _t([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1]])
PACK_GROUPS = _t([[0, 0, -1, 1, 1, -1, 1, 1, -3, -3, -1, -1]])
# Keys visible from each query, enumerated by hand.
PACK_VISIBLE = [
    {0, 1},  # q0: group 0 sees its later member
    {0, 1},
    {0, 1, 2},  # q2: ungrouped; the equal id at q5 does not group them
    {0, 1, 2, 3, 4},  # q3: group 1, document 0 members only
    {0, 1, 2, 3, 4},
    {0, 1, 2, 3, 4, 5},
    {6, 7},  # q6: document 1 reuses id 1 without joining document 0
    {6, 7},
    {6, 7, 8},  # q8: equal negative ids stay causal
    {6, 7, 8, 9},
    {10},  # q10: pad sentinel document, causal
    {10, 11},
]


def oracle_mask(doc_ids: torch.Tensor, group_ids: torch.Tensor | None) -> torch.Tensor:
    """Evaluate the predicate pair by pair in plain Python, shape (B, S, S)."""
    docs = doc_ids.tolist()
    groups = group_ids.tolist() if group_ids is not None else [[-1] * len(r) for r in docs]
    out = []
    for doc_row, group_row in zip(docs, groups, strict=True):
        seq_len = len(doc_row)
        out.append(
            [
                [
                    doc_row[q] == doc_row[k]
                    and (k <= q or (group_row[q] >= 0 and group_row[q] == group_row[k]))
                    for k in range(seq_len)
                ]
                for q in range(seq_len)
            ]
        )
    return torch.tensor(out, dtype=torch.bool)


FIXTURES = {
    "pack": (PACK_DOCS, PACK_GROUPS),
    # Membership is id equality alone; a group need not be contiguous.
    "noncontiguous_group": (_t([[0] * 8]), _t([[3, -1, 3, -1, 3, -1, -1, -1]])),
    # Prefix-LM shape: a bidirectional prefix, causal text, then a second document.
    "prefix_lm": (_t([[0, 0, 0, 0, 0, 1, 1, 1]]), _t([[5, 5, 5, -1, -1, -1, -1, -1]])),
    # Id 0 groups: "no group" is any negative id, not a falsy one.
    "group_id_zero": (_t([[0] * 8]), _t([[-1, -1, 0, 0, -1, -1, -1, -1]])),
    # One span of ids running across a document boundary splits in two.
    "span_across_doc_boundary": (
        _t([[0, 0, 0, 0, 1, 1, 1, 1]]),
        _t([[-1, 2, 2, 2, 2, 2, -1, -1]]),
    ),
    "batched": (
        _t([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2, 2]]),
        _t([[0, 0, -1, -1, 1, 1, -1, -1], [-1, -1, 4, 4, -1, 9, 9, -1]]),
    ),
}


def _materialize(doc_ids: torch.Tensor, group_ids: torch.Tensor | None) -> torch.Tensor:
    """Expand the BlockMask predicate to a dense (B, S, S) bool tensor."""
    batch, seq_len = doc_ids.shape
    block_mask = build_doc_causal_block_mask(doc_ids, CPU, group_ids)
    return create_mask(
        block_mask.mask_mod, B=batch, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=CPU
    )[:, 0]


class TestOracle:
    def test_matches_the_hand_enumerated_pack(self):
        assert torch.equal(oracle_mask(PACK_DOCS, PACK_GROUPS)[0], _from_visible(PACK_VISIBLE))

    def test_without_groups_is_block_diagonal_causal(self):
        expected = _block_diag(_ones(6).tril(), _ones(4).tril(), _ones(2).tril())
        assert torch.equal(oracle_mask(PACK_DOCS, None)[0], expected)


class TestSameGroupMask:
    """The dense group term used by the SDPA path."""

    def test_pack_pairs(self):
        # Group 0 = {0, 1}, group 1 = {3, 4, 6, 7}: this term ignores documents.
        members = [{0, 1}, {3, 4, 6, 7}]
        expected = torch.tensor(
            [[any(q in m and k in m for m in members) for k in range(12)] for q in range(12)]
        )
        out = same_group_mask(PACK_GROUPS)
        assert out.shape == (1, 1, 12, 12)
        assert torch.equal(out[0, 0], expected)

    def test_equal_negative_ids_never_match(self):
        assert not same_group_mask(_t([[-1, -1, -5, -5]])).any()

    def test_full_int64_range(self):
        # An int32 view would make 2**31 negative and fold 2**32 + 5 onto 5.
        out = same_group_mask(_t([[2**31, 2**31, 2**32 + 5, 5]]))
        expected = _block_diag(_ones(2), _ones(1), _ones(1))
        assert torch.equal(out[0, 0], expected)


class TestBlockMaskPredicate:
    """The FlexAttention mask_mod, expanded over every (q, k) pair."""

    def test_pack_matches_the_hand_table(self):
        assert torch.equal(_materialize(PACK_DOCS, PACK_GROUPS)[0], _from_visible(PACK_VISIBLE))

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_matches_oracle(self, name):
        doc_ids, group_ids = FIXTURES[name]
        assert torch.equal(_materialize(doc_ids, group_ids), oracle_mask(doc_ids, group_ids))

    def test_document_isolation_dominates_a_shared_group(self):
        """Every token its own document, all in group 0: only the diagonal survives."""
        doc_ids = torch.arange(8).unsqueeze(0)
        out = _materialize(doc_ids, torch.zeros_like(doc_ids))
        assert torch.equal(out[0], torch.eye(8, dtype=torch.bool))

    def test_group_spanning_a_document_makes_it_bidirectional(self):
        doc_ids = _t([[0, 0, 0, 1, 1, 1, 1, 1]])
        group_ids = _t([[2, 2, 2, -1, -1, -1, -1, -1]])
        expected = _block_diag(_ones(3), _ones(5).tril())
        assert torch.equal(_materialize(doc_ids, group_ids)[0], expected)

    def test_full_int64_group_ids(self):
        doc_ids = torch.zeros(1, 6, dtype=torch.long)
        group_ids = _t([[5, 2**32 + 5, 2**31, -1, 2**31, 5]])
        expected = _from_visible(
            [{0, 5}, {0, 1}, {0, 1, 2, 4}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}]
        )
        assert torch.equal(_materialize(doc_ids, group_ids)[0], expected)

    def test_all_negative_groups_equal_no_groups(self):
        all_negative = torch.full_like(PACK_DOCS, -1)
        assert torch.equal(_materialize(PACK_DOCS, all_negative), _materialize(PACK_DOCS, None))

    def test_no_query_row_fully_masked(self):
        for doc_ids, group_ids in FIXTURES.values():
            assert _materialize(doc_ids, group_ids).any(dim=-1).all()

    @pytest.mark.parametrize("seq_len", [7, 127, 128, 129, 200, 257])
    def test_lengths_around_the_block_size(self, seq_len):
        """A reused group id straddles the document and the 128 block boundary."""
        doc_ids = torch.zeros(1, seq_len, dtype=torch.long)
        doc_ids[0, seq_len // 2 :] = 1
        group_ids = torch.full((1, seq_len), -1, dtype=torch.long)
        group_ids[0, seq_len // 3 : 2 * seq_len // 3] = 4
        assert torch.equal(_materialize(doc_ids, group_ids), oracle_mask(doc_ids, group_ids))


class TestDenseAttentionPath:
    """The SDPA dense branch applies exactly the expected mask."""

    @staticmethod
    def _rope_identity(seq_len: int, head_dim: int):
        return torch.ones(seq_len, head_dim // 2), torch.zeros(seq_len, head_dim // 2)

    @staticmethod
    def _reference(attn: Attention, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Recompose attention from the module's own projections and a given mask."""
        batch, seq_len, _ = x.shape
        q = attn.q_proj(x).view(batch, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(batch, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(batch, seq_len, -1, attn.head_dim).transpose(1, 2)
        if attn.n_rep > 1:
            k = k.repeat_interleave(attn.n_rep, dim=1)
            v = v.repeat_interleave(attn.n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1))
        return attn.o_proj(out.transpose(1, 2).reshape(batch, seq_len, -1))

    def _setup(self, doc_ids: torch.Tensor, seed: int, n_kv_heads: int = 4):
        batch, seq_len = doc_ids.shape
        torch.manual_seed(seed)
        attn = Attention(dim=32, n_heads=4, n_kv_heads=n_kv_heads).eval()
        x = torch.randn(batch, seq_len, 32)
        cos, sin = self._rope_identity(seq_len, attn.head_dim)
        return attn, x, cos, sin

    @pytest.mark.parametrize("n_kv_heads", [4, 2], ids=["mha", "gqa"])
    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_forward_matches_the_reference(self, name, n_kv_heads):
        doc_ids, group_ids = FIXTURES[name]
        attn, x, cos, sin = self._setup(doc_ids, seed=0, n_kv_heads=n_kv_heads)
        with torch.no_grad():
            out = attn(x, cos, sin, doc_ids=doc_ids, attention_group_ids=group_ids)
            expected = self._reference(attn, x, oracle_mask(doc_ids, group_ids))
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_groups_without_doc_ids_are_one_document(self):
        attn, x, cos, sin = self._setup(PACK_DOCS, seed=1)
        expected_mask = oracle_mask(torch.zeros_like(PACK_GROUPS), PACK_GROUPS)
        with torch.no_grad():
            out = attn(x, cos, sin, attention_group_ids=PACK_GROUPS)
            expected = self._reference(attn, x, expected_mask)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_all_negative_groups_bitwise_equal_no_groups(self):
        attn, x, cos, sin = self._setup(PACK_DOCS, seed=2)
        all_negative = torch.full_like(PACK_DOCS, -1)
        with torch.no_grad():
            base = attn(x, cos, sin, doc_ids=PACK_DOCS)
            grouped = attn(x, cos, sin, doc_ids=PACK_DOCS, attention_group_ids=all_negative)
        torch.testing.assert_close(grouped, base, rtol=0, atol=0)

    def test_key_padding_drops_keys_inside_a_group(self):
        attn, x, cos, sin = self._setup(PACK_DOCS, seed=3)
        key_padding = torch.ones_like(PACK_DOCS, dtype=torch.bool)
        key_padding[0, 4] = False  # a group-1 member becomes an invalid key
        visible = PACK_VISIBLE.copy()
        visible[3] = {0, 1, 2, 3}
        visible[4] = {0, 1, 2, 3}
        visible[5] = {0, 1, 2, 3, 5}
        with torch.no_grad():
            out = attn(
                x,
                cos,
                sin,
                doc_ids=PACK_DOCS,
                attention_group_ids=PACK_GROUPS,
                key_padding_mask=key_padding,
            )
            expected = self._reference(attn, x, _from_visible(visible).unsqueeze(0))
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_captured_weights_follow_the_hand_table(self):
        attn, x, cos, sin = self._setup(PACK_DOCS, seed=4)
        attn.capture_attention_weights = True
        with torch.no_grad():
            attn(x, cos, sin, doc_ids=PACK_DOCS, attention_group_ids=PACK_GROUPS)
        weights = attn.last_attention_weights
        assert weights is not None
        visible = _from_visible(PACK_VISIBLE).expand_as(weights)
        assert torch.equal(weights != 0, visible)

    def test_captured_weights_without_groups_are_block_diagonal_causal(self):
        attn, x, cos, sin = self._setup(PACK_DOCS, seed=4)
        attn.capture_attention_weights = True
        with torch.no_grad():
            attn(x, cos, sin, doc_ids=PACK_DOCS)
        weights = attn.last_attention_weights
        assert weights is not None
        expected = _block_diag(_ones(6).tril(), _ones(4).tril(), _ones(2).tril())
        assert torch.equal(weights != 0, expected.expand_as(weights))


class TestFlexForwardParity:
    """Flex output against SDPA under the expected dense mask, fp64 on CPU."""

    @staticmethod
    def _qkv(doc_ids: torch.Tensor, n_kv_heads: int = 4, seed: int = 0):
        batch, seq_len = doc_ids.shape
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(batch, 4, seq_len, 16, dtype=torch.float64, generator=gen)
        k = torch.randn(batch, n_kv_heads, seq_len, 16, dtype=torch.float64, generator=gen)
        v = torch.randn(batch, n_kv_heads, seq_len, 16, dtype=torch.float64, generator=gen)
        return q, k, v

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_matches_dense_mask_sdpa(self, name):
        doc_ids, group_ids = FIXTURES[name]
        q, k, v = self._qkv(doc_ids)
        block_mask = build_doc_causal_block_mask(doc_ids, CPU, group_ids)
        out_flex = flex_attention_fn(False)(q, k, v, block_mask=block_mask, scale=16**-0.5)
        out_sdpa = F.scaled_dot_product_attention(
            q, k, v, attn_mask=oracle_mask(doc_ids, group_ids).unsqueeze(1)
        )
        torch.testing.assert_close(out_flex, out_sdpa, rtol=1e-12, atol=1e-12)

    def test_enable_gqa_with_groups(self):
        q, k, v = self._qkv(PACK_DOCS, n_kv_heads=2, seed=1)
        block_mask = build_doc_causal_block_mask(PACK_DOCS, CPU, PACK_GROUPS)
        out_flex = flex_attention_fn(False)(
            q, k, v, block_mask=block_mask, scale=16**-0.5, enable_gqa=True
        )
        out_sdpa = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(2, dim=1),
            v.repeat_interleave(2, dim=1),
            attn_mask=_from_visible(PACK_VISIBLE).view(1, 1, 12, 12),
        )
        torch.testing.assert_close(out_flex, out_sdpa, rtol=1e-12, atol=1e-12)


class TestTransformerForward:
    """End-to-end through Transformer.forward on both backends (CPU, forward only)."""

    @staticmethod
    def _model(backend: str) -> Transformer:
        torch.manual_seed(0)
        config = ModelConfig(
            dim=64,  # head_dim 16, the flex backend's floor
            n_layers=2,
            n_heads=4,
            vocab_size=64,
            max_seq_len=16,
            attention_backend=backend,
        )
        return Transformer(config).eval()

    @staticmethod
    def _tokens(seq: int = 12, seed: int = 0) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        return torch.randint(0, 64, (1, seq), generator=gen)

    @staticmethod
    def _perturb(tokens: torch.Tensor, positions: slice) -> torch.Tensor:
        moved = tokens.clone()
        moved[:, positions] = (moved[:, positions] + 1) % 64
        return moved

    def test_flex_matches_sdpa(self):
        tokens = self._tokens()
        with torch.no_grad():
            out = {
                backend: self._model(backend)(
                    tokens, doc_ids=PACK_DOCS, attention_group_ids=PACK_GROUPS
                )
                for backend in ("sdpa", "flex")
            }
        torch.testing.assert_close(out["flex"], out["sdpa"], rtol=1e-5, atol=1e-5)

    def test_flex_matches_sdpa_without_doc_ids(self):
        tokens = self._tokens(seed=1)
        with torch.no_grad():
            out = {
                backend: self._model(backend)(tokens, attention_group_ids=PACK_GROUPS)
                for backend in ("sdpa", "flex")
            }
        torch.testing.assert_close(out["flex"], out["sdpa"], rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_all_negative_groups_bitwise_equal_no_groups(self, backend):
        tokens = self._tokens(seed=2)
        model = self._model(backend)
        all_negative = torch.full_like(PACK_DOCS, -1)
        with torch.no_grad():
            base = model(tokens, doc_ids=PACK_DOCS)
            grouped = model(tokens, doc_ids=PACK_DOCS, attention_group_ids=all_negative)
        torch.testing.assert_close(grouped, base, rtol=0, atol=0)

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_group_member_sees_a_later_member(self, backend):
        tokens = self._tokens(seed=3)
        moved = self._perturb(tokens, slice(1, 2))  # q1: group 0, after q0
        model = self._model(backend)
        with torch.no_grad():
            causal = model(tokens, doc_ids=PACK_DOCS), model(moved, doc_ids=PACK_DOCS)
            grouped = [
                model(t, doc_ids=PACK_DOCS, attention_group_ids=PACK_GROUPS)
                for t in (tokens, moved)
            ]
        torch.testing.assert_close(causal[1][:, 0], causal[0][:, 0], rtol=0, atol=0)
        assert not torch.allclose(grouped[1][:, 0], grouped[0][:, 0])

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_later_group_stays_invisible_to_earlier_tokens(self, backend):
        """Moving q4 (group 1) reaches q3 (same group) and q5 (causal), not q0..q2."""
        tokens = self._tokens(seed=4)
        model = self._model(backend)
        with torch.no_grad():
            base, out = (
                model(t, doc_ids=PACK_DOCS, attention_group_ids=PACK_GROUPS)
                for t in (tokens, self._perturb(tokens, slice(4, 5)))
            )
        torch.testing.assert_close(out[:, :3], base[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(out[:, 6:], base[:, 6:], rtol=0, atol=0)
        assert not torch.allclose(out[:, 3], base[:, 3])
        assert not torch.allclose(out[:, 5], base[:, 5])

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_no_cross_document_leakage_with_a_reused_group_id(self, backend):
        """Document 1 shares group id 1 with document 0 and still never touches it."""
        tokens = self._tokens(seed=5)
        model = self._model(backend)
        with torch.no_grad():
            base, out = (
                model(t, doc_ids=PACK_DOCS, attention_group_ids=PACK_GROUPS)
                for t in (tokens, self._perturb(tokens, slice(6, 10)))
            )
        torch.testing.assert_close(out[:, :6], base[:, :6], rtol=0, atol=0)
        torch.testing.assert_close(out[:, 10:], base[:, 10:], rtol=0, atol=0)
        assert not torch.allclose(out[:, 6:10], base[:, 6:10])

    def test_key_padding_mask_keeps_groups_under_flex(self):
        """A key_padding_mask sends flex to the dense path with the groups intact.

        The ids cover the residual stream, prefix included.
        """
        tokens = self._tokens(seq=8, seed=6)
        prefix = torch.randn(1, 4, 64, generator=torch.Generator().manual_seed(6))
        key_padding = torch.ones(1, 12, dtype=torch.bool)
        key_padding[0, 2] = False
        modality = ModalityContext(prefix_embeds=prefix, key_padding_mask=key_padding)
        with torch.no_grad():
            out = {
                backend: self._model(backend)(
                    tokens, modality=modality, doc_ids=PACK_DOCS, attention_group_ids=PACK_GROUPS
                )
                for backend in ("sdpa", "flex")
            }
            ungrouped = self._model("flex")(tokens, modality=modality, doc_ids=PACK_DOCS)
        torch.testing.assert_close(out["flex"], out["sdpa"], rtol=0, atol=0)
        assert not torch.allclose(out["flex"], ungrouped)


class TestValidation:
    @staticmethod
    def _model() -> Transformer:
        config = ModelConfig(dim=32, n_layers=1, n_heads=4, vocab_size=64, max_seq_len=16)
        return Transformer(config)

    @staticmethod
    def _spy_embedding(model: Transformer) -> list[None]:
        """One entry per token-embedding call; every rejection must leave it empty."""
        assert model.token_embedding is not None
        calls: list[None] = []
        model.token_embedding.register_forward_pre_hook(lambda *_: calls.append(None))
        return calls

    @pytest.mark.parametrize("dtype", [torch.int32, torch.float32, torch.bool])
    def test_rejects_non_int64_dtype(self, dtype):
        model = self._model()
        calls = self._spy_embedding(model)
        tokens = torch.randint(0, 64, (1, 8))
        with pytest.raises(ValueError, match="must be torch.long"):
            model(tokens, attention_group_ids=torch.zeros(1, 8, dtype=dtype))
        assert calls == []

    @pytest.mark.parametrize("shape", [(1, 7), (2, 8), (8,), (1, 8, 1)])
    def test_rejects_shape_mismatch(self, shape):
        model = self._model()
        calls = self._spy_embedding(model)
        tokens = torch.randint(0, 64, (1, 8))
        with pytest.raises(ValueError, match=r"does not match the residual shape \(1, 8\)"):
            model(tokens, attention_group_ids=torch.zeros(shape, dtype=torch.long))
        assert calls == []

    def test_shape_follows_the_residual_not_the_tokens(self):
        model = self._model()
        calls = self._spy_embedding(model)
        tokens = torch.randint(0, 64, (1, 8))
        modality = ModalityContext(prefix_embeds=torch.zeros(1, 4, 32))
        with pytest.raises(ValueError, match=r"does not match the residual shape \(1, 12\)"):
            model(
                tokens, modality=modality, attention_group_ids=torch.zeros(1, 8, dtype=torch.long)
            )
        assert calls == []

    def test_valid_residual_shape_runs_and_embeds_once(self):
        model = self._model()
        calls = self._spy_embedding(model)
        tokens = torch.randint(0, 64, (1, 8))
        modality = ModalityContext(prefix_embeds=torch.zeros(1, 4, 32))
        groups = torch.tensor([[0] * 4 + [-1] * 8])
        out = model(tokens, modality=modality, attention_group_ids=groups)
        assert out.shape == (1, 12, 64)
        assert len(calls) == 1

    def test_shape_follows_inputs_embeds(self):
        model = self._model()
        modality = ModalityContext(inputs_embeds=torch.zeros(1, 8, 32))
        with pytest.raises(ValueError, match=r"does not match the residual shape \(1, 8\)"):
            model(modality=modality, attention_group_ids=torch.zeros(1, 7, dtype=torch.long))
        out = model(modality=modality, attention_group_ids=torch.zeros(1, 8, dtype=torch.long))
        assert out.shape == (1, 8, 64)

    def test_rejects_kv_caches(self):
        caches = [
            KVCache(
                batch_size=1,
                max_seq_len=16,
                n_kv_heads=4,
                head_dim=8,
                dtype=torch.float32,
                device=CPU,
            )
        ]
        model = self._model()
        calls = self._spy_embedding(model)
        tokens = torch.randint(0, 64, (1, 8))
        with pytest.raises(ValueError, match="training-only"):
            model(tokens, kv_caches=caches, attention_group_ids=torch.zeros(1, 8, dtype=torch.long))
        assert calls == []

    @pytest.mark.parametrize(
        "vlm_config",
        [
            MoMaConfig(moma_experts_per_modality={"image": 2, "text": 2}, moma_gumbel_noise=False),
            MoTConfig(max_text_len=8),
        ],
        ids=["moma", "mot"],
    )
    def test_rejects_moma_and_mot_paths(self, vlm_config):
        config = ModelConfig(dim=32, n_layers=1, n_heads=4, vocab_size=64, max_seq_len=16)
        model = Transformer(config, vlm_config=vlm_config, num_image_tokens=4)
        calls = self._spy_embedding(model)
        tokens = torch.randint(0, 64, (1, 4))
        with pytest.raises(ValueError, match="MoMa/MoT"):
            model(tokens, attention_group_ids=torch.zeros(1, 8, dtype=torch.long))
        assert calls == []
