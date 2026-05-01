"""Unit tests for Mixture-of-Transformers (MoT) operator + block + warm-start helper.

Covers:
- ``MoTAttention``: per-modality projections, global SDPA, RoPE per modality,
  GQA / MHA paths, QK-norm path, gradient flow, Algorithm-1 reference parity,
  cross-modality information flow under causal masking.
- ``MoTBlock``: per-modality forward, zero-init residual identity at
  construction, residual non-identity after a step, frozen-grad behavior.
- ``mot_warm_start_from_text_stack``: dense -> per-modality copy, idempotency,
  no-op on non-MoT modules, shape-mismatch rejection, optional
  ``transformer.`` prefix on source keys.

No GPU required; uses CPU tensors.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.mot import MoTAttention, MoTBlock, mot_warm_start_from_text_stack
from kempnerforge.model.position import apply_rope, precompute_rope_frequencies

DEVICE = torch.device("cpu")


def _rope_for(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = precompute_rope_frequencies(head_dim=head_dim, max_seq_len=seq_len)
    return cos.to(DEVICE), sin.to(DEVICE)


def _config(
    dim: int = 64,
    n_heads: int = 4,
    n_kv_heads: int | None = None,
    qk_norm: bool = False,
    n_layers: int = 2,
) -> ModelConfig:
    """Tiny dense config for MoT unit tests."""
    return ModelConfig(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads or n_heads,
        vocab_size=128,
        max_seq_len=64,
        qk_norm=qk_norm,
        ffn_hidden_dim=128,
    )


# ---------------------------------------------------------------------------
# MoTAttention — structural
# ---------------------------------------------------------------------------


class TestMoTAttentionStructural:
    def test_construction_zero_init_o_proj(self):
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"))
        for m in ("image", "text"):
            assert torch.equal(attn.o_proj[m].weight, torch.zeros_like(attn.o_proj[m].weight))

    def test_per_modality_qkvo_projections_distinct(self):
        """After re-init, image and text Q-projections produce different Q for the same input."""
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"))
        # Re-init each q_proj with different random weights.
        torch.manual_seed(0)
        nn.init.normal_(attn.q_proj["image"].weight)
        torch.manual_seed(1)
        nn.init.normal_(attn.q_proj["text"].weight)
        x = torch.randn(2, 4, 64)
        q_img = attn.q_proj["image"](x)
        q_txt = attn.q_proj["text"](x)
        assert not torch.allclose(q_img, q_txt)

    def test_qk_norm_modules_present_and_per_modality(self):
        attn = MoTAttention(
            dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"), qk_norm=True
        )
        assert attn.q_norm is not None and attn.k_norm is not None
        assert set(attn.q_norm.keys()) == {"image", "text"}
        assert set(attn.k_norm.keys()) == {"image", "text"}

    def test_qk_norm_disabled_by_default(self):
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"))
        assert attn.q_norm is None and attn.k_norm is None

    def test_invalid_modalities_raises(self):
        with pytest.raises(ValueError, match="at least one modality"):
            MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=())

    def test_invalid_kv_head_divisibility_raises(self):
        with pytest.raises(ValueError, match="n_heads"):
            MoTAttention(dim=64, n_heads=4, n_kv_heads=3, modalities=("image", "text"))

    def test_forward_streams_keys_must_match(self):
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"))
        # Re-init o_proj so output is non-trivially zero.
        for m in ("image", "text"):
            nn.init.normal_(attn.o_proj[m].weight)
        rope = {"image": _rope_for(4, 16), "text": _rope_for(4, 16)}
        with pytest.raises(ValueError, match="streams keys"):
            attn({"image": torch.randn(1, 4, 64)}, rope)


# ---------------------------------------------------------------------------
# MoTAttention — forward shapes / dtypes / paths
# ---------------------------------------------------------------------------


class TestMoTAttentionForward:
    @pytest.mark.parametrize("batch", [1, 4])
    @pytest.mark.parametrize("t_image,t_text", [(8, 16), (16, 64)])
    def test_output_shape_per_modality(self, batch: int, t_image: int, t_text: int):
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"))
        for m in ("image", "text"):
            nn.init.normal_(attn.o_proj[m].weight)
        streams = {
            "image": torch.randn(batch, t_image, 64),
            "text": torch.randn(batch, t_text, 64),
        }
        rope = {"image": _rope_for(t_image, 16), "text": _rope_for(t_text, 16)}
        out = attn(streams, rope)
        assert out["image"].shape == (batch, t_image, 64)
        assert out["text"].shape == (batch, t_text, 64)

    def test_output_dtype_matches_input(self):
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text")).to(
            dtype=torch.float32
        )
        for m in ("image", "text"):
            nn.init.normal_(attn.o_proj[m].weight)
        streams = {"image": torch.randn(1, 4, 64), "text": torch.randn(1, 4, 64)}
        rope = {"image": _rope_for(4, 16), "text": _rope_for(4, 16)}
        out = attn(streams, rope)
        assert out["image"].dtype == torch.float32
        assert out["text"].dtype == torch.float32

    def test_mha_path(self):
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"))
        for m in ("image", "text"):
            nn.init.normal_(attn.o_proj[m].weight)
        streams = {"image": torch.randn(2, 6, 64), "text": torch.randn(2, 10, 64)}
        rope = {"image": _rope_for(6, 16), "text": _rope_for(10, 16)}
        out = attn(streams, rope)
        assert out["image"].shape == (2, 6, 64)
        assert out["text"].shape == (2, 10, 64)

    def test_gqa_path(self):
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=2, modalities=("image", "text"))
        for m in ("image", "text"):
            nn.init.normal_(attn.o_proj[m].weight)
        streams = {"image": torch.randn(1, 4, 64), "text": torch.randn(1, 4, 64)}
        rope = {"image": _rope_for(4, 16), "text": _rope_for(4, 16)}
        out = attn(streams, rope)
        assert out["image"].shape == (1, 4, 64)
        assert out["text"].shape == (1, 4, 64)
        assert torch.isfinite(out["image"]).all()
        assert torch.isfinite(out["text"]).all()

    def test_qk_norm_path_runs(self):
        attn = MoTAttention(
            dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"), qk_norm=True
        )
        for m in ("image", "text"):
            nn.init.normal_(attn.o_proj[m].weight)
        streams = {"image": torch.randn(1, 4, 64) * 100.0, "text": torch.randn(1, 4, 64) * 100.0}
        rope = {"image": _rope_for(4, 16), "text": _rope_for(4, 16)}
        out = attn(streams, rope)
        assert torch.isfinite(out["image"]).all()
        assert torch.isfinite(out["text"]).all()

    def test_backward_grads_flow_to_all_per_modality_projections(self):
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"))
        for m in ("image", "text"):
            nn.init.normal_(attn.o_proj[m].weight)
            nn.init.normal_(attn.q_proj[m].weight)
            nn.init.normal_(attn.k_proj[m].weight)
            nn.init.normal_(attn.v_proj[m].weight)
        streams = {"image": torch.randn(1, 4, 64), "text": torch.randn(1, 4, 64)}
        rope = {"image": _rope_for(4, 16), "text": _rope_for(4, 16)}
        out = attn(streams, rope)
        (out["image"].sum() + out["text"].sum()).backward()
        for m in ("image", "text"):
            for proj in (attn.q_proj[m], attn.k_proj[m], attn.v_proj[m], attn.o_proj[m]):
                assert proj.weight.grad is not None
                assert proj.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# MoTAttention — Algorithm 1 reference parity + cross-modality flow
# ---------------------------------------------------------------------------


def _algorithm_1_reference(
    attn: MoTAttention,
    streams: dict[str, torch.Tensor],
    rope: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Hand-written reference for MoT Algorithm 1.

    Per-modality projections (one Linear per modality), per-modality RoPE,
    concat along seq, single global SDPA, split, per-modality o_proj.
    """
    batch = next(iter(streams.values())).shape[0]
    qs: list[torch.Tensor] = []
    ks: list[torch.Tensor] = []
    vs: list[torch.Tensor] = []
    lens: dict[str, int] = {}
    for m in attn.modalities:
        x_m = streams[m]
        t_m = x_m.shape[1]
        lens[m] = t_m
        q_m = attn.q_proj[m](x_m).view(batch, t_m, attn.n_heads, attn.head_dim).transpose(1, 2)
        k_m = attn.k_proj[m](x_m).view(batch, t_m, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
        v_m = attn.v_proj[m](x_m).view(batch, t_m, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
        if attn.q_norm is not None:
            q_m = attn.q_norm[m](q_m.transpose(1, 2)).transpose(1, 2)
            k_m = attn.k_norm[m](k_m.transpose(1, 2)).transpose(1, 2)  # type: ignore[reportOptionalSubscript]
        cos_m, sin_m = rope[m]
        q_m = apply_rope(q_m, cos_m, sin_m)
        k_m = apply_rope(k_m, cos_m, sin_m)
        qs.append(q_m)
        ks.append(k_m)
        vs.append(v_m)
    q = torch.cat(qs, dim=2)
    k = torch.cat(ks, dim=2)
    v = torch.cat(vs, dim=2)
    if attn.n_rep > 1:
        k = k.repeat_interleave(attn.n_rep, dim=1)
        v = v.repeat_interleave(attn.n_rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = out.transpose(1, 2).contiguous()
    out_streams: dict[str, torch.Tensor] = {}
    offset = 0
    for m in attn.modalities:
        t_m = lens[m]
        o_m = out[:, offset : offset + t_m, :, :].reshape(batch, t_m, -1)
        out_streams[m] = attn.o_proj[m](o_m)
        offset += t_m
    return out_streams


class TestMoTAttentionNumerical:
    def test_matches_algorithm_1_reference(self):
        torch.manual_seed(42)
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=2, modalities=("image", "text"))
        for m in ("image", "text"):
            nn.init.normal_(attn.o_proj[m].weight)
        streams = {"image": torch.randn(2, 6, 64), "text": torch.randn(2, 10, 64)}
        rope = {"image": _rope_for(6, 16), "text": _rope_for(10, 16)}
        out_actual = attn(streams, rope)
        out_ref = _algorithm_1_reference(attn, streams, rope)
        for m in ("image", "text"):
            assert torch.allclose(out_actual[m], out_ref[m], atol=1e-5, rtol=1e-5)

    def test_swapping_image_stream_changes_text_output(self):
        """Global SA mixes streams: text output must depend on image input."""
        torch.manual_seed(0)
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"))
        for m in ("image", "text"):
            nn.init.normal_(attn.o_proj[m].weight)
        text = torch.randn(1, 8, 64)
        rope = {"image": _rope_for(4, 16), "text": _rope_for(8, 16)}
        img_a = torch.randn(1, 4, 64)
        img_b = torch.randn(1, 4, 64)
        out_a = attn({"image": img_a, "text": text}, rope)
        out_b = attn({"image": img_b, "text": text}, rope)
        assert not torch.allclose(out_a["text"], out_b["text"], atol=1e-6)

    def test_image_position_invariance_under_text_perturbation(self):
        """Causality: text appended after image => image output independent of text."""
        torch.manual_seed(0)
        attn = MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"))
        for m in ("image", "text"):
            nn.init.normal_(attn.o_proj[m].weight)
        image = torch.randn(1, 6, 64)
        rope = {"image": _rope_for(6, 16), "text": _rope_for(8, 16)}
        text_a = torch.randn(1, 8, 64)
        text_b = torch.randn(1, 8, 64)
        out_a = attn({"image": image, "text": text_a}, rope)
        out_b = attn({"image": image, "text": text_b}, rope)
        assert torch.allclose(out_a["image"], out_b["image"], atol=1e-6)

    def test_text_output_independent_of_image_projection_when_image_is_zero(self):
        """When image input is zero, image K/V are zero; text V contribution to
        text positions does not depend on the image-modality projection weights.
        Image positions still 'absorb' softmax mass (contribute 0 to V), so the
        text output is not equal to dense attention's text output, but it IS
        invariant under any change to image-modality projection weights.
        """
        torch.manual_seed(0)
        attn1 = MoTAttention(dim=64, n_heads=4, n_kv_heads=4, modalities=("image", "text"))
        # Re-init: text weights random; image weights random initial.
        for m in ("image", "text"):
            nn.init.normal_(attn1.q_proj[m].weight)
            nn.init.normal_(attn1.k_proj[m].weight)
            nn.init.normal_(attn1.v_proj[m].weight)
            nn.init.normal_(attn1.o_proj[m].weight)
        attn2 = copy.deepcopy(attn1)
        # Reset image-modality weights only on attn2.
        for proj in (attn2.q_proj, attn2.k_proj, attn2.v_proj, attn2.o_proj):
            nn.init.normal_(proj["image"].weight)

        zero_image = torch.zeros(1, 4, 64)
        text = torch.randn(1, 8, 64)
        rope = {"image": _rope_for(4, 16), "text": _rope_for(8, 16)}
        out1 = attn1({"image": zero_image, "text": text}, rope)
        out2 = attn2({"image": zero_image, "text": text}, rope)
        assert torch.allclose(out1["text"], out2["text"], atol=1e-6)


# ---------------------------------------------------------------------------
# MoTBlock
# ---------------------------------------------------------------------------


class TestMoTBlock:
    def test_block_forward_shape_per_modality(self):
        cfg = _config(dim=64, n_heads=4)
        block = MoTBlock(cfg, modalities=("image", "text"), layer_idx=0)
        # Re-init so block isn't identity.
        for m in ("image", "text"):
            nn.init.normal_(block.attn.o_proj[m].weight)
            nn.init.normal_(block.mlp[m].down_proj.weight)  # type: ignore[union-attr]
        streams = {"image": torch.randn(2, 6, 64), "text": torch.randn(2, 10, 64)}
        rope = {"image": _rope_for(6, 16), "text": _rope_for(10, 16)}
        out = block(streams, rope)
        assert out["image"].shape == (2, 6, 64)
        assert out["text"].shape == (2, 10, 64)

    def test_block_zero_init_residual_identity(self):
        """At construction, MoTBlock(streams) bit-equal to streams."""
        cfg = _config(dim=64, n_heads=4)
        block = MoTBlock(cfg, modalities=("image", "text"), layer_idx=0)
        streams = {"image": torch.randn(1, 4, 64), "text": torch.randn(1, 6, 64)}
        rope = {"image": _rope_for(4, 16), "text": _rope_for(6, 16)}
        with torch.no_grad():
            out = block(streams, rope)
        for m in ("image", "text"):
            assert torch.equal(out[m], streams[m])

    def test_block_residual_nonzero_after_step(self):
        """After a small training step, block is no longer identity."""
        cfg = _config(dim=64, n_heads=4)
        block = MoTBlock(cfg, modalities=("image", "text"), layer_idx=0)
        streams = {
            "image": torch.randn(1, 4, 64, requires_grad=False),
            "text": torch.randn(1, 4, 64, requires_grad=False),
        }
        rope = {"image": _rope_for(4, 16), "text": _rope_for(4, 16)}
        opt = torch.optim.SGD(block.parameters(), lr=1e-2)
        out = block(streams, rope)
        loss = sum(o.pow(2).mean() for o in out.values())
        loss.backward()
        opt.step()
        with torch.no_grad():
            out2 = block(streams, rope)
        for m in ("image", "text"):
            assert not torch.allclose(out2[m], streams[m])

    def test_block_frozen_no_grad(self):
        """Frozen block: forward runs and backward (via input grad) does not
        accumulate grads on any block parameter."""
        cfg = _config(dim=64, n_heads=4)
        block = MoTBlock(cfg, modalities=("image", "text"), layer_idx=0)
        for p in block.parameters():
            p.requires_grad_(False)
        # Re-init so block isn't identity (otherwise backward through identity
        # produces uninteresting input grads).
        with torch.no_grad():
            for m in ("image", "text"):
                nn.init.normal_(block.attn.o_proj[m].weight)
                nn.init.normal_(block.mlp[m].down_proj.weight)  # type: ignore[union-attr]
        streams = {
            "image": torch.randn(1, 4, 64, requires_grad=True),
            "text": torch.randn(1, 4, 64, requires_grad=True),
        }
        rope = {"image": _rope_for(4, 16), "text": _rope_for(4, 16)}
        out = block(streams, rope)
        loss = sum(o.sum() for o in out.values())
        loss.backward()
        for p in block.parameters():
            assert p.grad is None
        # Sanity: input grads do flow.
        assert streams["text"].grad is not None and streams["text"].grad.abs().sum() > 0

    def test_block_streams_keys_must_match(self):
        cfg = _config(dim=64, n_heads=4)
        block = MoTBlock(cfg, modalities=("image", "text"), layer_idx=0)
        rope = {"image": _rope_for(4, 16), "text": _rope_for(4, 16)}
        with pytest.raises(ValueError, match="streams keys"):
            block({"image": torch.randn(1, 4, 64)}, rope)

    def test_block_with_qk_norm(self):
        cfg = _config(dim=64, n_heads=4, qk_norm=True)
        block = MoTBlock(cfg, modalities=("image", "text"), layer_idx=0)
        assert block.attn.q_norm is not None
        assert "image" in block.attn.q_norm and "text" in block.attn.q_norm
        # Re-init so block is non-identity, then sanity-check forward.
        for m in ("image", "text"):
            nn.init.normal_(block.attn.o_proj[m].weight)
            nn.init.normal_(block.mlp[m].down_proj.weight)  # type: ignore[union-attr]
        streams = {"image": torch.randn(1, 4, 64) * 50.0, "text": torch.randn(1, 4, 64) * 50.0}
        rope = {"image": _rope_for(4, 16), "text": _rope_for(4, 16)}
        out = block(streams, rope)
        assert torch.isfinite(out["image"]).all()
        assert torch.isfinite(out["text"]).all()


# ---------------------------------------------------------------------------
# Warm-start helper
# ---------------------------------------------------------------------------


class _MoTHarness(nn.Module):
    """Minimal Transformer-like module: ``layers`` ModuleDict of MoTBlocks,
    optional ``mot_norms`` ModuleDict, optional ``norm`` (final).

    Mimics the structure that ``Transformer.__init__`` will produce in
    Step 4 so the warm-start helper can be tested without needing the
    full Transformer build path (which depends on MoTConfig from Step 3).
    """

    def __init__(
        self,
        config: ModelConfig,
        modalities: tuple[str, ...],
        n_layers: int,
        with_mot_norms: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {str(i): MoTBlock(config, modalities, layer_idx=i) for i in range(n_layers)}
        )
        if with_mot_norms:
            self.mot_norms = nn.ModuleDict(
                {m: nn.RMSNorm(config.dim) for m in modalities}  # type: ignore[attr-defined]
            )


def _synth_jd_state_dict(
    config: ModelConfig,
    n_layers: int,
    qk_norm: bool = False,
    include_final_norm: bool = True,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Synthesize a JD/text-only state dict with random tensors of the right shapes."""
    head_dim = config.dim // config.n_heads
    n_kv = config.n_kv_heads or config.n_heads
    h = config.computed_ffn_hidden_dim
    state: dict[str, torch.Tensor] = {}
    for i in range(n_layers):
        state[f"{prefix}layers.{i}.attention_norm.weight"] = torch.randn(config.dim)
        state[f"{prefix}layers.{i}.attention.q_proj.weight"] = torch.randn(
            config.n_heads * head_dim, config.dim
        )
        state[f"{prefix}layers.{i}.attention.k_proj.weight"] = torch.randn(
            n_kv * head_dim, config.dim
        )
        state[f"{prefix}layers.{i}.attention.v_proj.weight"] = torch.randn(
            n_kv * head_dim, config.dim
        )
        state[f"{prefix}layers.{i}.attention.o_proj.weight"] = torch.randn(
            config.dim, config.n_heads * head_dim
        )
        if qk_norm:
            state[f"{prefix}layers.{i}.attention.q_norm.weight"] = torch.randn(head_dim)
            state[f"{prefix}layers.{i}.attention.k_norm.weight"] = torch.randn(head_dim)
        state[f"{prefix}layers.{i}.mlp_norm.weight"] = torch.randn(config.dim)
        state[f"{prefix}layers.{i}.mlp.gate_proj.weight"] = torch.randn(h, config.dim)
        state[f"{prefix}layers.{i}.mlp.up_proj.weight"] = torch.randn(h, config.dim)
        state[f"{prefix}layers.{i}.mlp.down_proj.weight"] = torch.randn(config.dim, h)
    if include_final_norm:
        state[f"{prefix}norm.weight"] = torch.randn(config.dim)
    return state


class TestWarmStartHelper:
    def test_no_op_on_non_mot_model(self):
        """Module without MoTBlock layers: helper does not modify anything."""
        # Build a non-MoT module: one nn.Linear named "layers" (not a ModuleDict of MoTBlocks).
        non_mot = nn.Module()
        non_mot.layers = nn.ModuleDict({"0": nn.Linear(32, 32)})
        sentinel = non_mot.layers["0"].weight.detach().clone()
        # Source state dict with arbitrary content.
        source = {"layers.0.foo.weight": torch.randn(32, 32)}
        mot_warm_start_from_text_stack(non_mot, source)
        assert torch.equal(non_mot.layers["0"].weight, sentinel)

    def test_no_op_when_module_has_no_layers(self):
        empty = nn.Module()
        # Should not raise.
        mot_warm_start_from_text_stack(empty, {"layers.0.attention.q_proj.weight": torch.zeros(1)})

    def test_copies_dense_block_weights_to_per_modality_copies(self):
        cfg = _config(dim=32, n_heads=2, n_kv_heads=2)
        n_layers = 2
        modalities = ("image", "text")
        harness = _MoTHarness(cfg, modalities=modalities, n_layers=n_layers)
        source = _synth_jd_state_dict(cfg, n_layers=n_layers)
        mot_warm_start_from_text_stack(harness, source)
        for i in range(n_layers):
            for m in modalities:
                assert torch.equal(
                    harness.layers[str(i)].attn_norm[m].weight,
                    source[f"layers.{i}.attention_norm.weight"],
                )
                assert torch.equal(
                    harness.layers[str(i)].attn.q_proj[m].weight,
                    source[f"layers.{i}.attention.q_proj.weight"],
                )
                assert torch.equal(
                    harness.layers[str(i)].attn.k_proj[m].weight,
                    source[f"layers.{i}.attention.k_proj.weight"],
                )
                assert torch.equal(
                    harness.layers[str(i)].attn.v_proj[m].weight,
                    source[f"layers.{i}.attention.v_proj.weight"],
                )
                assert torch.equal(
                    harness.layers[str(i)].attn.o_proj[m].weight,
                    source[f"layers.{i}.attention.o_proj.weight"],
                )
                assert torch.equal(
                    harness.layers[str(i)].mlp_norm[m].weight,
                    source[f"layers.{i}.mlp_norm.weight"],
                )
                assert torch.equal(
                    harness.layers[str(i)].mlp[m].gate_proj.weight,  # type: ignore[union-attr]
                    source[f"layers.{i}.mlp.gate_proj.weight"],
                )
                assert torch.equal(
                    harness.layers[str(i)].mlp[m].up_proj.weight,  # type: ignore[union-attr]
                    source[f"layers.{i}.mlp.up_proj.weight"],
                )
                assert torch.equal(
                    harness.layers[str(i)].mlp[m].down_proj.weight,  # type: ignore[union-attr]
                    source[f"layers.{i}.mlp.down_proj.weight"],
                )
                assert torch.equal(
                    harness.mot_norms[m].weight,
                    source["norm.weight"],
                )

    def test_qk_norm_translation(self):
        cfg = _config(dim=32, n_heads=2, n_kv_heads=2, qk_norm=True)
        harness = _MoTHarness(cfg, modalities=("image", "text"), n_layers=1)
        source = _synth_jd_state_dict(cfg, n_layers=1, qk_norm=True)
        mot_warm_start_from_text_stack(harness, source)
        for m in ("image", "text"):
            assert torch.equal(
                harness.layers["0"].attn.q_norm[m].weight,  # type: ignore[index]
                source["layers.0.attention.q_norm.weight"],
            )
            assert torch.equal(
                harness.layers["0"].attn.k_norm[m].weight,  # type: ignore[index]
                source["layers.0.attention.k_norm.weight"],
            )

    def test_idempotent(self):
        cfg = _config(dim=32, n_heads=2, n_kv_heads=2)
        harness = _MoTHarness(cfg, modalities=("image", "text"), n_layers=2)
        source = _synth_jd_state_dict(cfg, n_layers=2)
        mot_warm_start_from_text_stack(harness, source)
        snapshot = {k: v.detach().clone() for k, v in harness.state_dict().items()}
        mot_warm_start_from_text_stack(harness, source)
        for k, v in harness.state_dict().items():
            assert torch.equal(v, snapshot[k]), f"key {k} changed on second call"

    def test_accepts_transformer_prefix_on_source_keys(self):
        cfg = _config(dim=32, n_heads=2, n_kv_heads=2)
        harness = _MoTHarness(cfg, modalities=("image", "text"), n_layers=1)
        source = _synth_jd_state_dict(cfg, n_layers=1, prefix="transformer.")
        mot_warm_start_from_text_stack(harness, source)
        for m in ("image", "text"):
            assert torch.equal(
                harness.layers["0"].attn.q_proj[m].weight,
                source["transformer.layers.0.attention.q_proj.weight"],
            )

    def test_shape_mismatch_raises(self):
        cfg = _config(dim=32, n_heads=2, n_kv_heads=2)
        harness = _MoTHarness(cfg, modalities=("image", "text"), n_layers=1)
        source = _synth_jd_state_dict(cfg, n_layers=1)
        # Corrupt one source tensor's shape.
        source["layers.0.attention.q_proj.weight"] = torch.randn(8, 8)
        with pytest.raises(ValueError, match="shape mismatch"):
            mot_warm_start_from_text_stack(harness, source)

    def test_partial_source_skips_missing_layers_silently(self):
        """Source missing layer 1 keys: layer 0 is warm-started, layer 1 keeps
        construction-time (zero-init) weights."""
        cfg = _config(dim=32, n_heads=2, n_kv_heads=2)
        harness = _MoTHarness(cfg, modalities=("image", "text"), n_layers=2)
        # Only synth layer 0; layer 1 absent from source.
        source_full = _synth_jd_state_dict(cfg, n_layers=1)
        zero_o_proj = harness.layers["1"].attn.o_proj["image"].weight.detach().clone()
        mot_warm_start_from_text_stack(harness, source_full)
        # Layer 0 warm-started.
        assert torch.equal(
            harness.layers["0"].attn.o_proj["image"].weight,
            source_full["layers.0.attention.o_proj.weight"],
        )
        # Layer 1 untouched.
        assert torch.equal(harness.layers["1"].attn.o_proj["image"].weight, zero_o_proj)
