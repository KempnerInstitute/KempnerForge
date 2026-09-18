"""Unit tests for FlexAttention BlockMask construction.

Covers the mask predicate itself (against a dense reference) and forward
parity between the flex path and the dense-mask SDPA path it replaces.

FlexAttention has no CPU backward in torch 2.11, so everything here is
forward-only; gradient parity is covered on GPU in
``tests/integration/test_compile.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask

from kempnerforge.model.masking import build_doc_causal_block_mask, flex_attention_fn

CPU = torch.device("cpu")


def dense_doc_causal(doc_ids: torch.Tensor) -> torch.Tensor:
    """Reference mask: causal AND same-document, as Attention builds it today."""
    seq_len = doc_ids.shape[1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    return causal.unsqueeze(0) & (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1))


def elementwise_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Expand the BlockMask's predicate to a dense (B, S, S) bool tensor."""
    batch, seq_len = doc_ids.shape
    block_mask = build_doc_causal_block_mask(doc_ids, CPU)
    return create_mask(
        block_mask.mask_mod, B=batch, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=CPU
    )[:, 0]


class TestBlockMaskStructure:
    @pytest.mark.parametrize(
        "doc_ids",
        [
            pytest.param([[0, 0, 0, 0, 0, 0, 0, 0]], id="single_document"),
            pytest.param([[0, 0, 0, 1, 1, 1, 1, 1]], id="two_documents"),
            pytest.param([[0, 0, 1, 1, 1, 2, 2, 2]], id="three_documents"),
            pytest.param([[0, 1, 2, 3, 4, 5, 6, 7]], id="every_token_its_own_document"),
            pytest.param([[0, 0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 2, 2, 2, 2]], id="batched"),
        ],
    )
    def test_matches_dense_reference(self, doc_ids):
        ids = torch.tensor(doc_ids)
        assert torch.equal(elementwise_mask(ids), dense_doc_causal(ids))

    def test_single_document_is_plain_causal(self):
        ids = torch.zeros(1, 16, dtype=torch.long)
        expected = torch.ones(16, 16, dtype=torch.bool).tril()
        assert torch.equal(elementwise_mask(ids)[0], expected)

    def test_every_token_attends_to_itself(self):
        """No query row is ever fully masked, so softmax cannot produce NaN."""
        ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        assert elementwise_mask(ids)[0].diagonal().all()
        assert elementwise_mask(ids).any(dim=-1).all()

    @pytest.mark.parametrize("seq_len", [7, 63, 129, 200])
    def test_sequence_length_not_a_multiple_of_block_size(self, seq_len):
        """BLOCK_SIZE is 128; ragged lengths must still mask exactly."""
        ids = torch.zeros(1, seq_len, dtype=torch.long)
        ids[0, seq_len // 2 :] = 1
        assert torch.equal(elementwise_mask(ids), dense_doc_causal(ids))

    def test_accepts_int64_doc_ids_from_the_dataset(self):
        """_compute_packed_output emits int64; the builder casts to int32 internally."""
        ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.int64)
        assert build_doc_causal_block_mask(ids, CPU) is not None
        assert torch.equal(elementwise_mask(ids), dense_doc_causal(ids))


class TestFlexForwardParity:
    """Flex output vs the dense-mask SDPA it replaces, in fp64 for a tight bound."""

    def _qkv(self, batch, n_heads, seq_len, head_dim, seed=0):
        gen = torch.Generator().manual_seed(seed)
        shape = (batch, n_heads, seq_len, head_dim)
        return tuple(torch.randn(shape, dtype=torch.float64, generator=gen) for _ in range(3))

    def test_matches_dense_mask_sdpa(self):
        doc_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
        q, k, v = self._qkv(1, 4, 8, 16)
        block_mask = build_doc_causal_block_mask(doc_ids, CPU)
        out_flex = flex_attention_fn(False)(q, k, v, block_mask=block_mask, scale=16**-0.5)
        out_sdpa = F.scaled_dot_product_attention(
            q, k, v, attn_mask=dense_doc_causal(doc_ids).unsqueeze(1)
        )
        torch.testing.assert_close(out_flex, out_sdpa, rtol=1e-12, atol=1e-12)

    def test_single_document_matches_is_causal_sdpa(self):
        doc_ids = torch.zeros(1, 8, dtype=torch.long)
        q, k, v = self._qkv(1, 4, 8, 16, seed=1)
        block_mask = build_doc_causal_block_mask(doc_ids, CPU)
        out_flex = flex_attention_fn(False)(q, k, v, block_mask=block_mask, scale=16**-0.5)
        out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.testing.assert_close(out_flex, out_sdpa, rtol=1e-12, atol=1e-12)

    def test_enable_gqa_matches_manual_kv_expansion(self):
        """Licenses skipping repeat_interleave on the flex path (attention.py)."""
        doc_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
        gen = torch.Generator().manual_seed(2)
        q = torch.randn(1, 4, 8, 16, dtype=torch.float64, generator=gen)
        k = torch.randn(1, 2, 8, 16, dtype=torch.float64, generator=gen)
        v = torch.randn(1, 2, 8, 16, dtype=torch.float64, generator=gen)
        block_mask = build_doc_causal_block_mask(doc_ids, CPU)
        flex = flex_attention_fn(False)
        out_gqa = flex(q, k, v, block_mask=block_mask, scale=16**-0.5, enable_gqa=True)
        out_expanded = flex(
            q,
            k.repeat_interleave(2, dim=1),
            v.repeat_interleave(2, dim=1),
            block_mask=block_mask,
            scale=16**-0.5,
        )
        torch.testing.assert_close(out_gqa, out_expanded, rtol=0, atol=0)


class TestFlexAttentionFn:
    def test_returns_the_same_callable_for_repeated_calls(self):
        """Cached so the compiled artifact is built once, not once per forward."""
        assert flex_attention_fn(False) is flex_attention_fn(False)

    def test_cpu_path_is_uncompiled(self):
        from torch.nn.attention.flex_attention import flex_attention

        assert flex_attention_fn(False) is flex_attention
