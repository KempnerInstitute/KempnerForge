"""Unit tests for ``CrossAttention`` and ``CrossAttentionBlock``.

Coverage groups:

- Structural: shape, dtype, backward, image_mask shape, MHA vs GQA.
- Numerical correctness: reference-impl cross-check, gradcheck, image
  features actually affect the output, image_mask is functional, image
  axis is uncausal.
- Block-level: shape, zero-init residual identity (warm-start
  guarantee), residual non-zero after a step, frozen no-grad.
- Variable shapes: parametrize over (B, seq_len, num_image_tokens).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kempnerforge.model.cross_attention import CrossAttention, CrossAttentionBlock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


class TestCrossAttention:
    def test_output_shape_dtype_backward(self):
        ca = CrossAttention(dim=64, n_heads=4, n_kv_heads=4).to(DEVICE)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        img = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
        out = ca(x, img)
        assert out.shape == (2, 8, 64)
        assert out.dtype == x.dtype
        out.sum().backward()
        assert x.grad is not None
        assert img.grad is not None
        # All projection weights receive gradients (Wo starts at zero so its
        # grad reflects the reverse-mode product, not the value).
        assert ca.q_proj.weight.grad is not None
        assert ca.k_proj.weight.grad is not None
        assert ca.v_proj.weight.grad is not None
        assert ca.o_proj.weight.grad is not None

    def test_image_mask_shape(self):
        """Accepts (B, num_image_tokens) bool; broadcasts across heads / Q."""
        ca = CrossAttention(dim=32, n_heads=4, n_kv_heads=4).to(DEVICE)
        x = torch.randn(2, 6, 32, device=DEVICE)
        img = torch.randn(2, 8, 32, device=DEVICE)
        mask = torch.ones(2, 8, dtype=torch.bool, device=DEVICE)
        out = ca(x, img, image_mask=mask)
        assert out.shape == (2, 6, 32)

    def test_mha_path(self):
        """n_kv_heads == n_heads (MHA): no head replication."""
        ca = CrossAttention(dim=32, n_heads=4, n_kv_heads=4).to(DEVICE)
        x = torch.randn(1, 4, 32, device=DEVICE)
        img = torch.randn(1, 6, 32, device=DEVICE)
        out = ca(x, img)
        assert out.shape == (1, 4, 32)
        assert ca.n_rep == 1

    def test_gqa_path(self):
        """n_kv_heads < n_heads (GQA): heads replicated up to n_heads."""
        ca = CrossAttention(dim=32, n_heads=4, n_kv_heads=2).to(DEVICE)
        x = torch.randn(1, 4, 32, device=DEVICE)
        img = torch.randn(1, 6, 32, device=DEVICE)
        out = ca(x, img)
        assert out.shape == (1, 4, 32)
        assert ca.n_rep == 2


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def _reference_cross_attention(
    ca: CrossAttention, x: torch.Tensor, image_features: torch.Tensor
) -> torch.Tensor:
    """Hand-written einsum-based reference. MHA only; GQA replicates
    K/V on the head axis after projection.
    """
    batch, seq_len, _ = x.shape
    n_image = image_features.shape[1]

    q = ca.q_proj(x).view(batch, seq_len, ca.n_heads, ca.head_dim)
    k = ca.k_proj(image_features).view(batch, n_image, ca.n_kv_heads, ca.head_dim)
    v = ca.v_proj(image_features).view(batch, n_image, ca.n_kv_heads, ca.head_dim)

    if ca.n_rep > 1:
        k = k.repeat_interleave(ca.n_rep, dim=2)
        v = v.repeat_interleave(ca.n_rep, dim=2)

    # (B, T, H, d), (B, N, H, d) -> einsum to (B, H, T, N) attention scores
    scale = ca.head_dim**-0.5
    attn = torch.einsum("bthd,bnhd->bhtn", q, k) * scale
    attn = F.softmax(attn, dim=-1)
    out = torch.einsum("bhtn,bnhd->bthd", attn, v)
    out = out.reshape(batch, seq_len, ca.n_heads * ca.head_dim)
    return ca.o_proj(out)


class TestCrossAttentionNumerical:
    def test_ca_matches_reference_implementation(self):
        """SDPA path matches a hand-written einsum reference within tolerance.

        Catches kernel-selection bugs, head-dim transpose mistakes, K/V
        projection ordering errors. Re-init Wo to non-zero so the test
        actually compares non-trivial outputs.
        """
        torch.manual_seed(0)
        ca = CrossAttention(dim=16, n_heads=2, n_kv_heads=2).to(DEVICE)
        # Wo is zero-init by default; nudge it for this test so outputs
        # are non-trivial. The reference uses the same weights.
        with torch.no_grad():
            ca.o_proj.weight.normal_(std=0.02)
        x = torch.randn(1, 4, 16, device=DEVICE)
        img = torch.randn(1, 8, 16, device=DEVICE)
        with torch.no_grad():
            out_sdpa = ca(x, img)
            out_ref = _reference_cross_attention(ca, x, img)
        torch.testing.assert_close(out_sdpa, out_ref, atol=1e-5, rtol=1e-5)

    def test_ca_matches_reference_implementation_gqa(self):
        """Same check on the GQA path (n_kv_heads != n_heads)."""
        torch.manual_seed(0)
        ca = CrossAttention(dim=16, n_heads=4, n_kv_heads=2).to(DEVICE)
        with torch.no_grad():
            ca.o_proj.weight.normal_(std=0.02)
        x = torch.randn(1, 3, 16, device=DEVICE)
        img = torch.randn(1, 5, 16, device=DEVICE)
        with torch.no_grad():
            out_sdpa = ca(x, img)
            out_ref = _reference_cross_attention(ca, x, img)
        torch.testing.assert_close(out_sdpa, out_ref, atol=1e-5, rtol=1e-5)

    def test_ca_gradcheck(self):
        """torch.autograd.gradcheck on a tiny CA in fp64.

        Asserts analytical gradients match numerical within default
        gradcheck tolerance. Only run on CPU because CUDA gradcheck is
        finicky with low-precision floats; the math is the same.
        """
        torch.manual_seed(0)
        ca = CrossAttention(dim=8, n_heads=2, n_kv_heads=2).double()
        with torch.no_grad():
            ca.o_proj.weight.normal_(std=0.05)
        x = torch.randn(1, 3, 8, dtype=torch.float64, requires_grad=True)
        img = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
        # Use torch.autograd.gradcheck on the input gradients only; param
        # gradients are exercised by test_output_shape_dtype_backward.
        assert torch.autograd.gradcheck(
            lambda x_, img_: ca(x_, img_),
            (x, img),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            check_undefined_grad=False,
        )

    def test_ca_image_features_affect_output(self):
        """Swapping image features changes the output. Rules out the
        silent residual-short-circuit failure mode where the block is
        wired but image input is functionally ignored."""
        torch.manual_seed(0)
        ca = CrossAttention(dim=16, n_heads=2, n_kv_heads=2).to(DEVICE)
        with torch.no_grad():
            ca.o_proj.weight.normal_(std=0.02)
        x = torch.randn(1, 4, 16, device=DEVICE)
        img_a = torch.randn(1, 6, 16, device=DEVICE)
        img_b = torch.randn(1, 6, 16, device=DEVICE)
        with torch.no_grad():
            out_a = ca(x, img_a)
            out_b = ca(x, img_b)
        assert not torch.allclose(out_a, out_b, atol=1e-4)

    def test_ca_image_mask_all_true_matches_none(self):
        """image_mask = all-True produces identical output to image_mask = None."""
        torch.manual_seed(0)
        ca = CrossAttention(dim=16, n_heads=2, n_kv_heads=2).to(DEVICE)
        with torch.no_grad():
            ca.o_proj.weight.normal_(std=0.02)
        x = torch.randn(1, 4, 16, device=DEVICE)
        img = torch.randn(1, 6, 16, device=DEVICE)
        mask_all = torch.ones(1, 6, dtype=torch.bool, device=DEVICE)
        with torch.no_grad():
            out_none = ca(x, img)
            out_mask = ca(x, img, image_mask=mask_all)
        torch.testing.assert_close(out_none, out_mask, atol=1e-6, rtol=1e-6)

    def test_ca_image_mask_drop_position_matches_remove(self):
        """Masking position j produces the same output as removing
        position j from image_features entirely (so the masked position
        contributes zero to the attention output)."""
        torch.manual_seed(0)
        ca = CrossAttention(dim=16, n_heads=2, n_kv_heads=2).to(DEVICE)
        with torch.no_grad():
            ca.o_proj.weight.normal_(std=0.02)
        x = torch.randn(1, 4, 16, device=DEVICE)
        img = torch.randn(1, 6, 16, device=DEVICE)
        # Mask out position 2.
        mask = torch.ones(1, 6, dtype=torch.bool, device=DEVICE)
        mask[0, 2] = False
        # Equivalent: drop position 2 from image_features entirely.
        img_dropped = torch.cat([img[:, :2, :], img[:, 3:, :]], dim=1)
        with torch.no_grad():
            out_masked = ca(x, img, image_mask=mask)
            out_dropped = ca(x, img_dropped)
        torch.testing.assert_close(out_masked, out_dropped, atol=1e-5, rtol=1e-5)

    def test_ca_image_axis_uncausal(self):
        """Text token at position 0 and at seq_len-1 attend to the FULL
        image K/V set (no causal masking on the image axis). With a
        constant text input across positions and a constant W_q, the
        per-position attention weights over the image axis must be
        bit-equal across all text positions.
        """
        torch.manual_seed(0)
        ca = CrossAttention(dim=16, n_heads=2, n_kv_heads=2).to(DEVICE)
        with torch.no_grad():
            ca.o_proj.weight.normal_(std=0.02)
        # Constant text input: every position is the same vector. So Q is the
        # same per-head vector at every text position. With identical Q across
        # text positions, the per-position attention weights over the image
        # axis are identical, hence the per-position outputs are identical.
        x_const = torch.randn(1, 1, 16, device=DEVICE).expand(1, 5, 16).contiguous()
        img = torch.randn(1, 8, 16, device=DEVICE)
        with torch.no_grad():
            out = ca(x_const, img)
        # All 5 text positions should produce the same output vector.
        for t in range(1, 5):
            torch.testing.assert_close(out[:, 0, :], out[:, t, :], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Block-level
# ---------------------------------------------------------------------------


class TestCrossAttentionBlock:
    def _block(self) -> CrossAttentionBlock:
        return CrossAttentionBlock(dim=32, n_heads=4, n_kv_heads=4, ffn_hidden_dim=64).to(DEVICE)

    def test_block_forward_shape(self):
        block = self._block()
        x = torch.randn(2, 6, 32, device=DEVICE)
        img = torch.randn(2, 8, 32, device=DEVICE)
        out = block(x, img)
        assert out.shape == (2, 6, 32)

    def test_ca_zero_init_residual_identity(self):
        """With Wo and MLP.down_proj zero-initialized (the construction
        defaults), block(x, img) is bit-equal to x. Required for warm-start
        parity with Llama-3-V: adding a CA arch to an existing text-only
        checkpoint must not regress text loss at step 0."""
        block = self._block().eval()
        x = torch.randn(2, 6, 32, device=DEVICE)
        img = torch.randn(2, 8, 32, device=DEVICE)
        with torch.no_grad():
            out = block(x, img)
        torch.testing.assert_close(out, x, atol=0.0, rtol=0.0)

    def test_block_residual_nonzero_after_step(self):
        """After one optimizer step (synthetic loss), block(x, img) != x.
        Confirms the block actually learns from non-zero gradients."""
        block = self._block()
        opt = torch.optim.SGD(block.parameters(), lr=1e-2)
        x = torch.randn(2, 6, 32, device=DEVICE)
        img = torch.randn(2, 8, 32, device=DEVICE)
        out = block(x, img)
        # Loss that depends on ALL the block's outputs (so gradients reach
        # every parameter, not just Wo). Sum of squared output works.
        loss = (out**2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            out_after = block(x, img)
        assert not torch.equal(out_after, x)

    def test_block_frozen_no_grad(self):
        """With all block params requires_grad=False, backward populates
        no grads on block params."""
        block = self._block()
        for p in block.parameters():
            p.requires_grad = False
        x = torch.randn(2, 6, 32, device=DEVICE, requires_grad=True)
        img = torch.randn(2, 8, 32, device=DEVICE, requires_grad=True)
        out = block(x, img)
        out.sum().backward()
        for p in block.parameters():
            assert p.grad is None
        # Inputs still receive gradients.
        assert x.grad is not None
        assert img.grad is not None


# ---------------------------------------------------------------------------
# Variable shapes (parametrize)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("seq_len", [16, 128])
@pytest.mark.parametrize("num_image_tokens", [64, 256])
def test_cross_attention_shape_parametrize(batch, seq_len, num_image_tokens):
    """Shape, dtype, and backward complete cleanly across a 2x2x2 matrix."""
    ca = CrossAttention(dim=32, n_heads=4, n_kv_heads=4).to(DEVICE)
    x = torch.randn(batch, seq_len, 32, device=DEVICE, requires_grad=True)
    img = torch.randn(batch, num_image_tokens, 32, device=DEVICE, requires_grad=True)
    out = ca(x, img)
    assert out.shape == (batch, seq_len, 32)
    assert out.dtype == x.dtype
    out.sum().backward()
    assert x.grad is not None
    assert img.grad is not None
