"""Unit tests for VLM adapters and the adapter registry."""

from __future__ import annotations

import dataclasses
import inspect
import math
import re

import pytest
import torch
import torch.nn.functional as F

import kempnerforge.model.adapter
from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.loader import load_config
from kempnerforge.config.registry import registry
from kempnerforge.model.adapter import (
    POOLING_ADAPTER_TYPES,
    AttentionalPoolAdapter,
    AvgPoolAdapter,
    LinearAdapter,
    MLP2LayerAdapter,
    VisionAdapter,
    _pad_grid_to_windows,
    build_adapter,
    pooled_token_count,
)
from kempnerforge.model.norm import build_norm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_LOUD_PAD_VALUE = 1e4


class _LoudPadF:
    """Stand-in for ``torch.nn.functional`` that pads floats with a huge value.

    Lets a test prove the pooling adapters *mask* padded patches rather than
    relying on them being zero: with padding masked out, the pooled output must
    not move when the pad value changes.
    """

    def __getattr__(self, name):
        return getattr(F, name)

    def pad(self, t, pad, mode="constant", value=None):
        if t.dtype.is_floating_point:
            return F.pad(t, pad, mode, _LOUD_PAD_VALUE)
        return F.pad(t, pad, mode, value)


def _reference_pool_windows(num_tokens: int, window: int) -> list[list[int]]:
    """Flat patch indices per output window, computed independently of the module.

    ``ceil(grid/window)**2`` windows in row-major order; each lists only the real
    (in-grid) patches it covers, so a ragged edge window is short.
    """
    grid = math.isqrt(num_tokens)
    assert grid * grid == num_tokens
    per = math.ceil(grid / window)
    windows = []
    for i in range(per):
        for j in range(per):
            windows.append(
                [
                    r * grid + c
                    for r in range(i * window, min((i + 1) * window, grid))
                    for c in range(j * window, min((j + 1) * window, grid))
                ]
            )
    return windows


def _reference_attentional_pool(
    adapter: AttentionalPoolAdapter, x: torch.Tensor, window: int
) -> torch.Tensor:
    """Brute-force attentional pool: per window, attend over its real patches only.

    Builds no padded tensor at all, so it is an independent check of the padded +
    masked implementation rather than a restatement of it.
    """
    b, _, c = x.shape
    heads, head_dim = adapter.pool_heads, adapter.head_dim
    outs = []
    for idx in _reference_pool_windows(x.shape[1], window):
        patches = x[:, idx]
        k_real = patches.shape[1]
        q = adapter.q_proj(patches.mean(dim=1, keepdim=True))
        q = q.view(b, 1, heads, head_dim).transpose(1, 2)
        k = adapter.k_proj(patches).view(b, k_real, heads, head_dim).transpose(1, 2)
        v = adapter.v_proj(patches).view(b, k_real, heads, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, c)
        outs.append(adapter.out_proj(adapter.o_proj(attn)))
    return torch.stack(outs, dim=1)


class TestMLP2LayerAdapter:
    def test_forward_shape(self):
        adapter = MLP2LayerAdapter(in_dim=384, out_dim=256).to(DEVICE)
        x = torch.randn(2, 16, 384, device=DEVICE)
        out = adapter(x)
        assert out.shape == (2, 16, 256)

    def test_hidden_dim_defaults_to_out_dim(self):
        adapter = MLP2LayerAdapter(in_dim=128, out_dim=64)
        assert adapter.proj1.out_features == 64

    def test_hidden_dim_override(self):
        adapter = MLP2LayerAdapter(in_dim=128, out_dim=64, hidden_dim=256)
        assert adapter.proj1.out_features == 256
        assert adapter.proj2.in_features == 256

    def test_activations(self):
        for act in ("gelu", "silu", "relu"):
            adapter = MLP2LayerAdapter(in_dim=8, out_dim=8, activation=act)
            out = adapter(torch.randn(1, 4, 8))
            assert out.shape == (1, 4, 8)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown adapter activation"):
            MLP2LayerAdapter(in_dim=8, out_dim=8, activation="tanh")

    def test_rejects_zero_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            MLP2LayerAdapter(in_dim=0, out_dim=8)

    def test_backward_grads_flow(self):
        adapter = MLP2LayerAdapter(in_dim=32, out_dim=16).to(DEVICE)
        x = torch.randn(1, 4, 32, device=DEVICE, requires_grad=True)
        adapter(x).sum().backward()
        for p in adapter.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_reset_parameters_reinitializes(self):
        adapter = MLP2LayerAdapter(in_dim=8, out_dim=4)
        w1 = adapter.proj1.weight.clone()
        adapter.reset_parameters()
        assert torch.isfinite(adapter.proj1.weight).all()
        assert adapter.proj1.weight.shape == w1.shape


class TestLinearAdapter:
    def test_forward_shape(self):
        adapter = LinearAdapter(in_dim=384, out_dim=256).to(DEVICE)
        x = torch.randn(2, 16, 384, device=DEVICE)
        out = adapter(x)
        assert out.shape == (2, 16, 256)

    def test_rejects_zero_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            LinearAdapter(in_dim=0, out_dim=8)

    def test_backward_grads_flow(self):
        adapter = LinearAdapter(in_dim=32, out_dim=16).to(DEVICE)
        x = torch.randn(1, 4, 32, device=DEVICE, requires_grad=True)
        adapter(x).sum().backward()
        for p in adapter.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_reset_parameters_reinitializes(self):
        adapter = LinearAdapter(in_dim=8, out_dim=4)
        w1 = adapter.proj.weight.clone()
        adapter.reset_parameters()
        assert torch.isfinite(adapter.proj.weight).all()
        assert adapter.proj.weight.shape == w1.shape

    def test_no_hidden_dim_or_activation(self):
        """LinearAdapter has a single ``proj`` Linear and no hidden layer, so
        the config rejects the MLP fields rather than dropping them."""
        adapter = build_adapter(AdapterConfig(type="linear"), in_dim=8, out_dim=4)
        assert isinstance(adapter, LinearAdapter)
        assert hasattr(adapter, "proj")
        assert not hasattr(adapter, "proj1")
        assert not hasattr(adapter, "proj2")
        with pytest.raises(
            ValueError,
            match=r"adapter\.type = 'linear' does not use adapter\.hidden_dim = 42, "
            r"adapter\.activation = 'silu'",
        ):
            AdapterConfig(type="linear", hidden_dim=42, activation="silu")


class TestAdapterRegistry:
    def test_mlp_2layer_is_registered(self):
        builder = registry.get_adapter("mlp_2layer")
        adapter = builder(in_dim=8, out_dim=4)
        assert isinstance(adapter, MLP2LayerAdapter)

    def test_linear_is_registered(self):
        builder = registry.get_adapter("linear")
        adapter = builder(in_dim=8, out_dim=4)
        assert isinstance(adapter, LinearAdapter)

    def test_list_includes_both(self):
        names = registry.list_adapters()
        assert "mlp_2layer" in names
        assert "linear" in names

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="adapter"):
            registry.get_adapter("not_a_real_adapter")

    def test_double_register_raises(self):
        @registry.register_adapter("adapter_smoke_test_dup")
        def _build_first(in_dim, out_dim, **_):  # noqa: ARG001
            return MLP2LayerAdapter(in_dim=in_dim, out_dim=out_dim)

        with pytest.raises(ValueError, match="already registered"):

            @registry.register_adapter("adapter_smoke_test_dup")
            def _build_second(in_dim, out_dim, **_):  # noqa: ARG001
                return MLP2LayerAdapter(in_dim=in_dim, out_dim=out_dim)


class TestAdapterConfig:
    def test_defaults(self):
        cfg = AdapterConfig()
        assert cfg.type == "mlp_2layer"
        assert cfg.hidden_dim == 0
        assert cfg.activation == "gelu"

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown adapter.type"):
            AdapterConfig(type="not_a_real_type")

    def test_negative_hidden_dim_rejected(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            AdapterConfig(hidden_dim=-1)

    def test_bad_activation_rejected(self):
        with pytest.raises(ValueError, match="Unknown adapter.activation"):
            AdapterConfig(activation="tanh")

    def test_extra_kwargs_zero_hidden_dim_becomes_none(self):
        cfg = AdapterConfig(hidden_dim=0)
        kwargs = cfg.extra_kwargs()
        assert kwargs["hidden_dim"] is None

    def test_extra_kwargs_nonzero_hidden_dim_passes_through(self):
        cfg = AdapterConfig(hidden_dim=256)
        kwargs = cfg.extra_kwargs()
        assert kwargs["hidden_dim"] == 256

    def test_extra_kwargs_activation_passes_through(self):
        cfg = AdapterConfig(activation="silu")
        kwargs = cfg.extra_kwargs()
        assert kwargs["activation"] == "silu"


class TestBuildAdapter:
    def test_dispatches_to_mlp_2layer(self):
        cfg = AdapterConfig(type="mlp_2layer")
        adapter = build_adapter(cfg, in_dim=8, out_dim=4)
        assert isinstance(adapter, MLP2LayerAdapter)

    def test_dispatches_to_linear(self):
        cfg = AdapterConfig(type="linear")
        adapter = build_adapter(cfg, in_dim=8, out_dim=4)
        assert isinstance(adapter, LinearAdapter)

    def test_passes_in_out_dim(self):
        cfg = AdapterConfig(type="mlp_2layer")
        adapter = build_adapter(cfg, in_dim=32, out_dim=16)
        out = adapter(torch.randn(1, 4, 32))
        assert out.shape == (1, 4, 16)

    def test_honors_hidden_dim_zero_sentinel(self):
        """hidden_dim=0 (the sentinel) -> MLP2LayerAdapter falls back to
        out_dim for the hidden width."""
        cfg = AdapterConfig(type="mlp_2layer", hidden_dim=0)
        adapter = build_adapter(cfg, in_dim=8, out_dim=64)
        assert adapter.proj1.out_features == 64

    def test_honors_hidden_dim_explicit(self):
        cfg = AdapterConfig(type="mlp_2layer", hidden_dim=128)
        adapter = build_adapter(cfg, in_dim=8, out_dim=64)
        assert adapter.proj1.out_features == 128

    def test_activation_passes_through_to_mlp(self):
        cfg = AdapterConfig(type="mlp_2layer", activation="silu")
        adapter = build_adapter(cfg, in_dim=8, out_dim=4)
        import torch.nn as nn

        assert isinstance(adapter.act, nn.SiLU)


class TestPooledTokenCount:
    @pytest.mark.parametrize(
        ("n_in", "window", "expected"),
        [
            (196, 2, 49),
            (256, 2, 64),
            (729, 3, 81),
            (16, 2, 4),
            (16, 3, 4),
            (100, 3, 16),
            (16, 1, 16),
            (16, 4, 1),
        ],
    )
    def test_counts(self, n_in, window, expected):
        assert pooled_token_count(n_in, window) == expected

    def test_non_square_grid_raises(self):
        with pytest.raises(ValueError, match="square patch grid"):
            pooled_token_count(8, 2)

    def test_non_positive_window_raises(self):
        with pytest.raises(ValueError, match="window must be positive"):
            pooled_token_count(16, 0)

    def test_non_positive_tokens_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            pooled_token_count(0, 2)


class TestPadGridToWindows:
    """The shared ``_pad_grid_to_windows`` helper decides the divisible-vs-ragged
    branch for BOTH pooling adapters. A divisible grid MUST return ``valid=None``
    so the connectors take the unmasked, bit-exact, no-mask-built fast path; a
    ragged grid returns a validity mask over the real patches. Regressing this
    (e.g. an all-True mask for a divisible grid) would silently route divisible
    grids through the masked SDPA path instead of the unmasked one."""

    def test_divisible_grid_builds_no_mask(self):
        x = torch.randn(2, 16, 8)
        padded, per, valid = _pad_grid_to_windows(x, 2)
        assert per == 2
        assert valid is None
        assert padded.shape == (2, 4, 4, 8)

    def test_ragged_grid_builds_mask_over_real_patches(self):
        x = torch.randn(2, 16, 8)
        padded, per, valid = _pad_grid_to_windows(x, 3)
        assert per == 2
        assert padded.shape == (2, 6, 6, 8)
        assert valid is not None and valid.shape == (2, 6, 6, 1)
        assert valid.sum().item() == 2 * 16
        assert bool(valid[:, :4, :4].all())
        assert not bool(valid[:, 4:, :].any()) and not bool(valid[:, :, 4:].any())

    @pytest.mark.parametrize(
        ("n_in", "window"),
        [(16, 3), (49, 3), (4, 3), (100, 3), (25, 2), (196, 3)],
    )
    def test_every_window_keeps_at_least_one_real_patch(self, n_in, window):
        x = torch.randn(1, n_in, 4)
        padded, per, valid = _pad_grid_to_windows(x, window)
        if valid is None:
            return
        counts = valid.view(1, per, window, per, window, 1).sum(dim=(2, 4))
        assert int(counts.min()) >= 1


class TestAvgPoolAdapter:
    def test_forward_shape_divisible(self):
        adapter = AvgPoolAdapter(in_dim=96, out_dim=64, pool_window=2).to(DEVICE)
        x = torch.randn(2, 16, 96, device=DEVICE)
        out = adapter(x)
        assert out.shape == (2, 4, 64)

    def test_forward_shape_ragged(self):
        adapter = AvgPoolAdapter(in_dim=96, out_dim=64, pool_window=3).to(DEVICE)
        x = torch.randn(2, 16, 96, device=DEVICE)
        out = adapter(x)
        assert out.shape == (2, 4, 64)

    def test_is_vision_adapter(self):
        assert isinstance(AvgPoolAdapter(in_dim=8, out_dim=4), VisionAdapter)

    @pytest.mark.parametrize(("n_in", "window"), [(16, 2), (16, 3), (256, 2), (729, 3), (100, 3)])
    def test_output_num_tokens_matches_forward(self, n_in, window):
        """The static count MUST equal the actual forward length — MoT's
        positional split relies on this agreement."""
        adapter = AvgPoolAdapter(in_dim=32, out_dim=16, pool_window=window)
        x = torch.randn(1, n_in, 32)
        assert adapter(x).shape[1] == adapter.output_num_tokens(n_in)

    def test_pool_window_override_in_forward(self):
        adapter = AvgPoolAdapter(in_dim=32, out_dim=16, pool_window=2)
        x = torch.randn(1, 16, 32)
        assert adapter(x, pool_window=4).shape == (1, 1, 16)
        assert adapter(x).shape == (1, 4, 16)

    def test_non_square_input_raises(self):
        adapter = AvgPoolAdapter(in_dim=8, out_dim=4, pool_window=2)
        with pytest.raises(ValueError, match="square patch grid"):
            adapter(torch.randn(1, 8, 8))

    def test_rejects_zero_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            AvgPoolAdapter(in_dim=0, out_dim=8)

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="pool_window must be positive"):
            AvgPoolAdapter(in_dim=8, out_dim=8, pool_window=0)

    def test_backward_grads_flow(self):
        adapter = AvgPoolAdapter(in_dim=32, out_dim=16, pool_window=2).to(DEVICE)
        x = torch.randn(1, 16, 32, device=DEVICE, requires_grad=True)
        adapter(x).sum().backward()
        for p in adapter.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_reset_parameters_reinitializes(self):
        adapter = AvgPoolAdapter(in_dim=8, out_dim=4)
        adapter.reset_parameters()
        assert torch.isfinite(adapter.proj.weight).all()

    def test_divisible_pool_is_plain_mean(self):
        """With a divisible grid, the pooled value is the exact window mean."""
        adapter = AvgPoolAdapter(in_dim=4, out_dim=4, pool_window=2)
        with torch.no_grad():
            adapter.proj.weight.copy_(torch.eye(4))
            adapter.proj.bias.zero_()
        x = torch.arange(16.0).view(1, 16, 1).expand(1, 16, 4).contiguous()
        out = adapter(x)
        assert torch.allclose(out[0, 0], torch.full((4,), 2.5))

    def test_ragged_edge_window_averages_only_real_patches(self):
        """A ragged edge window averages over its real patches only (divide by the
        real-patch count, NOT by w*w)."""
        adapter = AvgPoolAdapter(in_dim=4, out_dim=4, pool_window=3)
        with torch.no_grad():
            adapter.proj.weight.copy_(torch.eye(4))
            adapter.proj.bias.zero_()
        x = torch.arange(16.0).view(1, 16, 1).expand(1, 16, 4).contiguous()
        out = adapter(x)
        assert torch.allclose(out[0, 3], torch.full((4,), 15.0))

    @pytest.mark.parametrize(("n_in", "window"), [(16, 3), (49, 3), (4, 3), (196, 3)])
    def test_pad_value_does_not_leak(self, monkeypatch, n_in, window):
        """Padded cells are excluded from the average, not merely zero-valued."""
        torch.manual_seed(0)
        adapter = AvgPoolAdapter(in_dim=8, out_dim=4, pool_window=window)
        x = torch.randn(2, n_in, 8)
        with torch.no_grad():
            baseline = adapter(x)
            monkeypatch.setattr(kempnerforge.model.adapter, "F", _LoudPadF())
            loud = adapter(x)
        assert torch.isfinite(loud).all()
        assert torch.allclose(baseline, loud, atol=1e-5)

    def test_forward_rejects_nonpositive_window_override(self):
        adapter = AvgPoolAdapter(in_dim=8, out_dim=4, pool_window=2)
        with pytest.raises(ValueError, match="pool_window must be positive"):
            adapter(torch.randn(1, 16, 8), pool_window=0)


class TestAttentionalPoolAdapter:
    def test_forward_shape(self):
        adapter = AttentionalPoolAdapter(in_dim=96, out_dim=64, pool_window=2, pool_heads=16).to(
            DEVICE
        )
        x = torch.randn(2, 16, 96, device=DEVICE)
        out = adapter(x)
        assert out.shape == (2, 4, 64)

    def test_is_vision_adapter(self):
        assert isinstance(AttentionalPoolAdapter(in_dim=16, out_dim=8, pool_heads=4), VisionAdapter)

    @pytest.mark.parametrize(("n_in", "window"), [(16, 2), (256, 2), (729, 3), (196, 3)])
    def test_output_num_tokens_matches_forward(self, n_in, window):
        adapter = AttentionalPoolAdapter(in_dim=32, out_dim=16, pool_window=window, pool_heads=4)
        x = torch.randn(1, n_in, 32)
        assert adapter(x).shape[1] == adapter.output_num_tokens(n_in)

    def test_ragged_grid_supported(self):
        adapter = AttentionalPoolAdapter(in_dim=96, out_dim=64, pool_window=3, pool_heads=16)
        out = adapter(torch.randn(2, 16, 96))
        assert out.shape == (2, 4, 64)
        assert torch.isfinite(out).all()

    def test_output_num_tokens_ragged(self):
        adapter = AttentionalPoolAdapter(in_dim=96, out_dim=64, pool_window=3, pool_heads=16)
        assert adapter.output_num_tokens(16) == 4
        assert adapter.output_num_tokens(196) == 25

    def test_ragged_edge_window_pools_only_real_patches(self):
        torch.manual_seed(0)
        adapter = AttentionalPoolAdapter(in_dim=32, out_dim=16, pool_window=3, pool_heads=4).eval()
        x = torch.randn(1, 16, 32)
        with torch.no_grad():
            out = adapter(x)
            real = x[:, 15:16]
            ref = adapter.out_proj(adapter.o_proj(adapter.v_proj(real)))
        assert torch.allclose(out[:, 3:4], ref, atol=1e-5)

    def test_worst_ragged_edge_remainder_one(self):
        torch.manual_seed(0)
        adapter = AttentionalPoolAdapter(in_dim=32, out_dim=16, pool_window=3, pool_heads=4).eval()
        x = torch.randn(1, 49, 32)
        with torch.no_grad():
            out = adapter(x)
            ref = adapter.out_proj(adapter.o_proj(adapter.v_proj(x[:, 48:49])))
        assert out.shape == (1, 9, 16)
        assert torch.allclose(out[:, 8:9], ref, atol=1e-5)

    def test_window_larger_than_grid(self):
        adapter = AttentionalPoolAdapter(in_dim=32, out_dim=16, pool_window=3, pool_heads=4)
        out = adapter(torch.randn(2, 4, 32))
        assert out.shape == (2, 1, 16)
        assert adapter.output_num_tokens(4) == 1
        assert torch.isfinite(out).all()

    def test_window_equals_grid(self):
        adapter = AttentionalPoolAdapter(in_dim=32, out_dim=16, pool_window=4, pool_heads=4)
        out = adapter(torch.randn(2, 16, 32))
        assert out.shape == (2, 1, 16)
        assert adapter.output_num_tokens(16) == 1

    def test_non_square_input_raises(self):
        adapter = AttentionalPoolAdapter(in_dim=32, out_dim=16, pool_window=3, pool_heads=4)
        with pytest.raises(ValueError, match="square patch grid"):
            adapter(torch.randn(1, 8, 32))

    @pytest.mark.parametrize(
        ("n_in", "window"), [(16, 3), (49, 3), (4, 3), (196, 3), (196, 2), (16, 4), (25, 2)]
    )
    def test_matches_independent_per_window_reference(self, n_in, window):
        """Output equals a brute-force pool that attends over real patches only.

        Cross-checks the padded + masked implementation against a formulation that
        never materializes padding, so a leak or a window-ordering error shows up.
        """
        torch.manual_seed(0)
        adapter = AttentionalPoolAdapter(
            in_dim=32, out_dim=16, pool_window=window, pool_heads=4
        ).eval()
        x = torch.randn(2, n_in, 32)
        with torch.no_grad():
            got = adapter(x)
            want = _reference_attentional_pool(adapter, x, window)
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-5)

    @pytest.mark.parametrize(("n_in", "window"), [(16, 3), (49, 3), (4, 3), (196, 3)])
    def test_pad_value_does_not_leak(self, monkeypatch, n_in, window):
        """Padded patches are masked out, not merely zero.

        Padding with a huge value instead of zero must not move the output; if the
        mask were dropped, the padded keys would dominate every edge window.
        """
        torch.manual_seed(0)
        adapter = AttentionalPoolAdapter(
            in_dim=32, out_dim=16, pool_window=window, pool_heads=4
        ).eval()
        x = torch.randn(2, n_in, 32)
        with torch.no_grad():
            baseline = adapter(x)
            monkeypatch.setattr(kempnerforge.model.adapter, "F", _LoudPadF())
            loud = adapter(x)
        assert torch.isfinite(loud).all()
        assert torch.allclose(baseline, loud, atol=1e-5)

    def test_heads_must_divide_dim(self):
        with pytest.raises(ValueError, match="divisible by"):
            AttentionalPoolAdapter(in_dim=96, out_dim=64, pool_heads=7)

    def test_rejects_zero_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            AttentionalPoolAdapter(in_dim=0, out_dim=8)

    def test_backward_grads_flow(self):
        adapter = AttentionalPoolAdapter(in_dim=32, out_dim=16, pool_window=2, pool_heads=4).to(
            DEVICE
        )
        x = torch.randn(1, 16, 32, device=DEVICE, requires_grad=True)
        adapter(x).sum().backward()
        for p in adapter.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_backward_grads_flow_ragged(self):
        adapter = AttentionalPoolAdapter(in_dim=32, out_dim=16, pool_window=3, pool_heads=4).to(
            DEVICE
        )
        x = torch.randn(1, 16, 32, device=DEVICE, requires_grad=True)
        adapter(x).sum().backward()
        for p in adapter.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_reset_parameters_reinitializes(self):
        adapter = AttentionalPoolAdapter(in_dim=16, out_dim=8, pool_heads=4)
        adapter.reset_parameters()
        assert torch.isfinite(adapter.out_proj.weight).all()

    def test_forward_rejects_nonpositive_window_override(self):
        adapter = AttentionalPoolAdapter(in_dim=16, out_dim=8, pool_window=2, pool_heads=4)
        with pytest.raises(ValueError, match="pool_window must be positive"):
            adapter(torch.randn(1, 16, 16), pool_window=0)


class TestPoolingAdapterRegistry:
    def test_avgpool_registered(self):
        adapter = registry.get_adapter("avgpool")(in_dim=8, out_dim=4, pool_window=2)
        assert isinstance(adapter, AvgPoolAdapter)

    def test_attentional_pool_registered(self):
        adapter = registry.get_adapter("attentional_pool")(in_dim=8, out_dim=4, pool_heads=2)
        assert isinstance(adapter, AttentionalPoolAdapter)

    def test_pooling_types_constant_matches_registry(self):
        for name in POOLING_ADAPTER_TYPES:
            assert name in registry.list_adapters()


class TestAdapterConfigPooling:
    def test_pool_defaults(self):
        cfg = AdapterConfig()
        assert cfg.pool_window == 2
        assert cfg.pool_heads == 16

    def test_rejects_zero_pool_window(self):
        with pytest.raises(ValueError, match="pool_window must be positive"):
            AdapterConfig(pool_window=0)

    def test_rejects_zero_pool_heads(self):
        with pytest.raises(ValueError, match="pool_heads must be positive"):
            AdapterConfig(pool_heads=0)

    def test_extra_kwargs_includes_pool_fields(self):
        kwargs = AdapterConfig(type="attentional_pool", pool_window=3, pool_heads=8).extra_kwargs()
        assert kwargs["pool_window"] == 3
        assert kwargs["pool_heads"] == 8

    def test_output_num_tokens_identity_for_projection(self):
        assert AdapterConfig(type="mlp_2layer").output_num_tokens(196) == 196
        assert AdapterConfig(type="linear").output_num_tokens(196) == 196

    def test_output_num_tokens_pools_for_avgpool(self):
        assert AdapterConfig(type="avgpool", pool_window=2).output_num_tokens(196) == 49

    def test_output_num_tokens_pools_for_attentional(self):
        assert AdapterConfig(type="attentional_pool", pool_window=3).output_num_tokens(729) == 81

    def test_attentional_output_num_tokens_allows_ragged(self):
        assert AdapterConfig(type="attentional_pool", pool_window=3).output_num_tokens(196) == 25

    def test_avgpool_output_num_tokens_allows_ragged(self):
        assert AdapterConfig(type="avgpool", pool_window=3).output_num_tokens(196) == 25

    def test_output_num_tokens_passthrough_on_sentinel(self):
        assert AdapterConfig(type="avgpool").output_num_tokens(0) == 0

    @pytest.mark.parametrize(
        ("adapter_type", "fields"),
        [
            ("avgpool", {}),
            ("avgpool", {"hidden_dim": 24, "pre_norm": "rmsnorm"}),
            ("attentional_pool", {"pool_heads": 4}),
        ],
        ids=["avgpool", "avgpool-mlp-head", "attentional_pool"],
    )
    def test_config_count_matches_module_count(self, adapter_type, fields):
        """Config-time estimate must equal the built module's count so the
        config-time seq-len check matches the build-time budget."""
        cfg = AdapterConfig(type=adapter_type, pool_window=2, **fields)
        module = build_adapter(cfg, in_dim=32, out_dim=16)
        assert cfg.output_num_tokens(256) == module.output_num_tokens(256)
        assert module(torch.randn(1, 256, 32)).shape[1] == cfg.output_num_tokens(256)


class TestBuildAdapterPooling:
    def test_dispatches_to_avgpool_with_window(self):
        cfg = AdapterConfig(type="avgpool", pool_window=2)
        adapter = build_adapter(cfg, in_dim=32, out_dim=16)
        assert isinstance(adapter, AvgPoolAdapter)
        assert adapter.pool_window == 2
        out = adapter(torch.randn(1, 16, 32))
        assert out.shape == (1, 4, 16)

    def test_dispatches_to_attentional_with_heads(self):
        cfg = AdapterConfig(type="attentional_pool", pool_window=2, pool_heads=8)
        adapter = build_adapter(cfg, in_dim=32, out_dim=16)
        assert isinstance(adapter, AttentionalPoolAdapter)
        assert adapter.pool_heads == 8


_REGISTERED_NORMS = sorted(registry.list("norm"))
_PROJECTION_KEYS = {"proj1.weight", "proj1.bias", "proj2.weight", "proj2.bias"}


def _reference_projections(seed: int, in_dim: int, hidden: int, out_dim: int):
    """The two ``nn.Linear`` draws a plain 2-layer MLP makes at ``seed``; every
    ``mlp_2layer`` build must reproduce them for its projections."""
    torch.manual_seed(seed)
    return torch.nn.Linear(in_dim, hidden), torch.nn.Linear(hidden, out_dim)


class TestMLP2LayerPreNorm:
    def test_registry_has_norms_to_test(self):
        """Refuse rather than pass vacuously if the norm registry is empty."""
        assert _REGISTERED_NORMS

    @pytest.mark.parametrize("kwargs", [{}, {"pre_norm": ""}], ids=["absent", "empty"])
    def test_off_builds_no_module_and_no_state(self, kwargs):
        adapter = MLP2LayerAdapter(in_dim=32, out_dim=16, **kwargs)
        assert adapter.ln_q is None
        assert "ln_q" not in dict(adapter.named_modules())
        assert set(adapter.state_dict()) == _PROJECTION_KEYS

    @pytest.mark.parametrize("pre_norm", [None, "", *_REGISTERED_NORMS])
    def test_projections_draw_the_same_rng_as_a_bare_mlp(self, pre_norm):
        """Off, the knob builds nothing; on, the registered norms are
        constant-initialized -- either way proj1/proj2 get the draw a plain
        2-layer MLP gets, so no existing config's init shifts."""
        ref1, ref2 = _reference_projections(11, 32, 16, 16)
        torch.manual_seed(11)
        adapter = MLP2LayerAdapter(in_dim=32, out_dim=16, pre_norm=pre_norm)
        for ours, ref in ((adapter.proj1, ref1), (adapter.proj2, ref2)):
            assert torch.equal(ours.weight, ref.weight)
            assert torch.equal(ours.bias, ref.bias)

    @pytest.mark.parametrize("norm", _REGISTERED_NORMS)
    def test_norm_is_built_over_the_input_dim(self, norm):
        adapter = MLP2LayerAdapter(in_dim=32, out_dim=16, pre_norm=norm)
        assert adapter.ln_q is not None
        assert all(p.shape == (32,) for p in adapter.ln_q.parameters())
        assert set(adapter.state_dict()) == _PROJECTION_KEYS | {
            f"ln_q.{k}" for k in adapter.ln_q.state_dict()
        }

    @pytest.mark.parametrize("norm", _REGISTERED_NORMS)
    def test_forward_is_norm_then_mlp(self, norm):
        torch.manual_seed(0)
        adapter = MLP2LayerAdapter(in_dim=32, out_dim=16, pre_norm=norm).to(DEVICE)
        x = torch.randn(2, 4, 32, device=DEVICE) * 10.0
        with torch.no_grad():
            expected = adapter.proj2(adapter.act(adapter.proj1(adapter.ln_q(x))))
            without_norm = adapter.proj2(adapter.act(adapter.proj1(x)))
            out = adapter(x)
        assert out.shape == (2, 4, 16)
        assert torch.equal(out, expected)
        assert not torch.allclose(out, without_norm)

    @pytest.mark.parametrize("norm", _REGISTERED_NORMS)
    def test_norm_receives_gradient(self, norm):
        adapter = MLP2LayerAdapter(in_dim=32, out_dim=16, pre_norm=norm).to(DEVICE)
        adapter(torch.randn(2, 4, 32, device=DEVICE)).sum().backward()
        for name, p in adapter.ln_q.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert (adapter.ln_q.weight.grad != 0).any()

    @pytest.mark.parametrize("norm", _REGISTERED_NORMS)
    def test_every_registered_norm_has_reset_parameters(self, norm):
        """The adapter re-initializes its norm by calling reset_parameters()."""
        assert callable(getattr(build_norm(norm, 32), "reset_parameters", None))

    @pytest.mark.parametrize("norm", _REGISTERED_NORMS)
    def test_reset_parameters_restores_the_norm_after_a_meta_build(self, norm):
        """The meta-device path runs to_empty() then reset_parameters(); the
        norm must come back to its constructor init, not keep whatever memory
        to_empty handed it."""
        fresh = MLP2LayerAdapter(in_dim=32, out_dim=16, pre_norm=norm)
        with torch.device("meta"):
            adapter = MLP2LayerAdapter(in_dim=32, out_dim=16, pre_norm=norm)
        adapter.to_empty(device=torch.device("cpu"))
        with torch.no_grad():
            for p in adapter.ln_q.parameters():
                p.fill_(float("nan"))
        adapter.reset_parameters()
        for k, v in adapter.ln_q.state_dict().items():
            assert torch.equal(v, fresh.ln_q.state_dict()[k]), k
        assert all(torch.isfinite(p).all() for p in adapter.parameters())

    def test_reset_parameters_delegates_to_the_norm(self):
        """The norm's own reset_parameters() is the only init applied to it."""
        calls: list[str] = []

        class _Norm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.full((32,), 0.5))

            def reset_parameters(self):
                calls.append("reset")

        adapter = MLP2LayerAdapter(in_dim=32, out_dim=16, pre_norm="rmsnorm")
        adapter.ln_q = _Norm()
        adapter.reset_parameters()
        assert calls == ["reset"]
        assert torch.equal(adapter.ln_q.weight, torch.full((32,), 0.5))


class _PluginNorm(torch.nn.LayerNorm):
    pass


@pytest.fixture
def plugin_norm():
    name = "adapter_pre_norm_test_plugin"
    registry.register("norm", name, _PluginNorm)
    yield name
    registry.unregister("norm", name)


class TestAdapterConfigPreNorm:
    def test_default_is_off(self):
        cfg = AdapterConfig()
        assert cfg.pre_norm == ""
        assert cfg.extra_kwargs()["pre_norm"] is None
        assert build_adapter(cfg, in_dim=32, out_dim=16).ln_q is None

    @pytest.mark.parametrize("norm", _REGISTERED_NORMS)
    def test_every_registered_norm_is_accepted_and_built(self, norm):
        cfg = AdapterConfig(pre_norm=norm)
        assert cfg.extra_kwargs()["pre_norm"] == norm
        adapter = build_adapter(cfg, in_dim=32, out_dim=16)
        assert isinstance(adapter, MLP2LayerAdapter)
        assert adapter.ln_q is not None
        assert adapter(torch.randn(1, 4, 32)).shape == (1, 4, 16)

    def test_a_norm_registered_at_runtime_is_accepted_and_built(self, plugin_norm):
        adapter = build_adapter(AdapterConfig(pre_norm=plugin_norm), in_dim=32, out_dim=16)
        assert type(adapter.ln_q) is _PluginNorm

    @pytest.mark.parametrize("adapter_type", ["mlp_2layer", "avgpool"])
    def test_unknown_norm_is_rejected_naming_the_registered_ones(self, adapter_type):
        with pytest.raises(ValueError, match=r"Unknown adapter\.pre_norm: 'nope'") as info:
            AdapterConfig(type=adapter_type, pre_norm="nope")
        for norm in _REGISTERED_NORMS:
            assert norm in str(info.value)

    @pytest.mark.parametrize("adapter_type", ["linear", "avgpool", "attentional_pool"])
    def test_adapters_without_a_norm_reject_it(self, adapter_type):
        """An adapter that builds no norm rejects pre_norm instead of training
        without it; avgpool builds one only for its MLP head."""
        with pytest.raises(
            ValueError,
            match=rf"adapter\.type = '{adapter_type}' does not use adapter\.pre_norm = 'rmsnorm'",
        ):
            AdapterConfig(type=adapter_type, pre_norm="rmsnorm")


_ACTIVATION_MODULES = {"gelu": torch.nn.GELU, "silu": torch.nn.SiLU, "relu": torch.nn.ReLU}
_HEAD_KEYS = {
    "proj.ln_q.weight",
    "proj.proj1.weight",
    "proj.proj1.bias",
    "proj.proj2.weight",
    "proj.proj2.bias",
}


def _reference_avg_pool(x: torch.Tensor, window: int) -> torch.Tensor:
    """Window means over the real patches, computed independently of the module."""
    windows = _reference_pool_windows(x.shape[1], window)
    return torch.stack([x[:, idx].mean(dim=1) for idx in windows], dim=1)


class TestAvgPoolNormMLPHead:
    @pytest.mark.parametrize("hidden_dim", [None, 0])
    def test_default_head_is_the_single_linear(self, hidden_dim):
        """Without a hidden width the head is the ``proj`` Linear it always was:
        the same module tree and state-dict keys, so existing checkpoints load."""
        adapter = AvgPoolAdapter(in_dim=32, out_dim=16, hidden_dim=hidden_dim)
        assert type(adapter.proj) is torch.nn.Linear
        assert [name for name, _ in adapter.named_modules()] == ["", "proj"]
        assert list(adapter.state_dict()) == ["proj.weight", "proj.bias"]

    def test_default_head_draws_the_same_rng_as_a_bare_linear(self):
        torch.manual_seed(5)
        ref = torch.nn.Linear(32, 16)
        torch.manual_seed(5)
        adapter = build_adapter(AdapterConfig(type="avgpool"), in_dim=32, out_dim=16)
        assert torch.equal(adapter.proj.weight, ref.weight)
        assert torch.equal(adapter.proj.bias, ref.bias)

    @pytest.mark.parametrize("activation", sorted(_ACTIVATION_MODULES))
    @pytest.mark.parametrize("pre_norm", [None, *_REGISTERED_NORMS])
    def test_positive_hidden_dim_builds_the_norm_mlp_head(self, pre_norm, activation):
        adapter = AvgPoolAdapter(
            in_dim=32,
            out_dim=16,
            hidden_dim=24,
            activation=activation,
            pre_norm=pre_norm,
            norm_eps=1e-3,
        )
        head = adapter.proj
        assert isinstance(head, MLP2LayerAdapter)
        assert (head.proj1.in_features, head.proj1.out_features) == (32, 24)
        assert (head.proj2.in_features, head.proj2.out_features) == (24, 16)
        assert isinstance(head.act, _ACTIVATION_MODULES[activation])
        if pre_norm is None:
            assert head.ln_q is None
        else:
            assert head.ln_q.eps == 1e-3
            assert all(p.shape == (32,) for p in head.ln_q.parameters())

    def test_head_state_dict_keys(self):
        adapter = AvgPoolAdapter(in_dim=32, out_dim=16, hidden_dim=24, pre_norm="rmsnorm")
        assert set(adapter.state_dict()) == _HEAD_KEYS

    def test_head_draws_the_same_init_as_an_mlp_2layer_adapter(self):
        torch.manual_seed(7)
        ref = MLP2LayerAdapter(in_dim=32, out_dim=16, hidden_dim=24, pre_norm="rmsnorm")
        torch.manual_seed(7)
        adapter = AvgPoolAdapter(in_dim=32, out_dim=16, hidden_dim=24, pre_norm="rmsnorm")
        for key, value in ref.state_dict().items():
            assert torch.equal(adapter.proj.state_dict()[key], value), key

    @pytest.mark.parametrize("norm", _REGISTERED_NORMS)
    @pytest.mark.parametrize(("n_in", "window"), [(16, 2), (25, 2), (49, 3)])
    def test_forward_is_pool_then_norm_then_mlp(self, n_in, window, norm):
        """The norm sees the pooled features, also on the ragged edge windows."""
        torch.manual_seed(0)
        adapter = AvgPoolAdapter(
            in_dim=32, out_dim=16, pool_window=window, hidden_dim=24, pre_norm=norm
        ).to(DEVICE)
        head = adapter.proj
        x = torch.randn(2, n_in, 32, device=DEVICE) * 10.0
        with torch.no_grad():
            expected = head.proj2(head.act(head.proj1(head.ln_q(_reference_avg_pool(x, window)))))
            norm_first = head.proj2(head.act(head.proj1(_reference_avg_pool(head.ln_q(x), window))))
            out = adapter(x)
        assert out.shape == (2, adapter.output_num_tokens(n_in), 16)
        assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)
        assert not torch.allclose(out, norm_first, atol=1e-3)

    def test_parameter_counts_follow_the_layer_shapes(self):
        """At 1152 -> 1024 the bare head is one Linear, 1152 * 1024 + 1024; an
        RMSNorm + MLP head of width 1152 adds 1152 norm weights and a
        1152 -> 1152 Linear, 1152 + 1152 * 1152 + 1152."""
        with torch.device("meta"):
            bare = AvgPoolAdapter(in_dim=1152, out_dim=1024)
            head = AvgPoolAdapter(in_dim=1152, out_dim=1024, hidden_dim=1152, pre_norm="rmsnorm")
        assert sum(p.numel() for p in bare.parameters()) == 1_180_672
        assert sum(p.numel() for p in head.parameters()) == 2_510_080

    def test_reset_parameters_restores_the_head_after_a_meta_build(self):
        fresh = AvgPoolAdapter(in_dim=32, out_dim=16, hidden_dim=24, pre_norm="rmsnorm")
        with torch.device("meta"):
            adapter = AvgPoolAdapter(in_dim=32, out_dim=16, hidden_dim=24, pre_norm="rmsnorm")
        adapter.to_empty(device=torch.device("cpu"))
        with torch.no_grad():
            for p in adapter.parameters():
                p.fill_(float("nan"))
        adapter.reset_parameters()
        assert torch.equal(adapter.proj.ln_q.weight, fresh.proj.ln_q.weight)
        assert all(torch.isfinite(p).all() for p in adapter.parameters())

    def test_every_head_parameter_receives_gradient(self):
        adapter = AvgPoolAdapter(in_dim=32, out_dim=16, hidden_dim=24, pre_norm="rmsnorm").to(
            DEVICE
        )
        adapter(torch.randn(2, 25, 32, device=DEVICE)).sum().backward()
        for name, p in adapter.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert (p.grad != 0).any(), name

    def test_pool_window_override_keeps_the_head(self):
        adapter = AvgPoolAdapter(in_dim=32, out_dim=16, pool_window=2, hidden_dim=24)
        x = torch.randn(1, 16, 32)
        assert adapter(x, pool_window=4).shape == (1, 1, 16)
        assert adapter(x).shape == (1, 4, 16)

    def test_rejects_negative_hidden_dim(self):
        with pytest.raises(ValueError, match="hidden_dim must be non-negative"):
            AvgPoolAdapter(in_dim=8, out_dim=4, hidden_dim=-1)

    def test_config_builds_the_head(self):
        cfg = AdapterConfig(
            type="avgpool", pool_window=3, hidden_dim=24, activation="silu", pre_norm="layernorm"
        )
        adapter = build_adapter(cfg, in_dim=32, out_dim=16, norm_eps=1e-3)
        assert isinstance(adapter, AvgPoolAdapter)
        assert adapter.pool_window == 3
        head = adapter.proj
        assert isinstance(head, MLP2LayerAdapter)
        assert head.proj1.out_features == 24
        assert isinstance(head.act, torch.nn.SiLU)
        assert isinstance(head.ln_q, torch.nn.LayerNorm)
        assert head.ln_q.eps == 1e-3


_BUILTIN_ADAPTERS = ("mlp_2layer", "linear", "avgpool", "attentional_pool")


@pytest.fixture
def plugin_adapter():
    """Register throwaway adapters that record their kwargs, removed after the test."""
    names: list[str] = []

    def register(accepts=None):
        name = f"adapter_accepts_test_{len(names)}"
        seen: dict = {}

        @registry.register_adapter(name, accepts=accepts)
        def _build(in_dim, out_dim, **kwargs):
            seen.update(kwargs)
            return LinearAdapter(in_dim=in_dim, out_dim=out_dim)

        names.append(name)
        return name, seen

    yield register
    for name in names:
        registry.unregister("adapter", name)


class TestAdapterAccepts:
    @pytest.mark.parametrize("name", _BUILTIN_ADAPTERS)
    def test_every_builtin_adapter_declares_its_fields(self, name):
        assert registry.get_adapter_accepts(name) is not None

    @pytest.mark.parametrize(
        ("name", "fields"),
        [
            ("mlp_2layer", {}),
            ("linear", {}),
            ("avgpool", {}),
            ("avgpool", {"hidden_dim": 8}),
            ("attentional_pool", {}),
        ],
        ids=["mlp_2layer", "linear", "avgpool", "avgpool-mlp-head", "attentional_pool"],
    )
    def test_declared_fields_are_builder_parameters(self, name, fields):
        """A declared field reaches a named builder parameter, never only ``**_``."""
        declared = registry.get_adapter_accepts(name)
        cfg = AdapterConfig(type=name, **fields)
        used = declared(cfg) if callable(declared) else declared
        params = inspect.signature(registry.get_adapter(name)).parameters
        for field in used:
            assert field in params, field
            assert params[field].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, field

    def test_a_string_declaration_is_rejected(self):
        with pytest.raises(TypeError, match="collection of field names"):
            registry.register_adapter("adapter_accepts_test_str", accepts="pool_window")
        assert "adapter_accepts_test_str" not in registry.list_adapters()

    def test_a_callable_returning_a_string_is_rejected(self, plugin_adapter):
        """A bare string would be read letter by letter as field names."""
        name, _ = plugin_adapter(accepts=lambda cfg: "pool_window")
        with pytest.raises(
            TypeError,
            match=rf"accepts for adapter '{name}' must return a collection of field names, "
            r"not the string 'pool_window'",
        ):
            AdapterConfig(type=name)

    def test_a_declared_collection_is_enforced(self, plugin_adapter):
        name, seen = plugin_adapter(accepts=("pool_window",))
        build_adapter(AdapterConfig(type=name, pool_window=3), in_dim=8, out_dim=4)
        assert seen["pool_window"] == 3
        with pytest.raises(
            ValueError, match=rf"adapter\.type = '{name}' does not use adapter\.hidden_dim = 8"
        ):
            AdapterConfig(type=name, hidden_dim=8)

    def test_a_declared_callable_sees_the_config(self, plugin_adapter):
        name, _ = plugin_adapter(
            accepts=lambda cfg: ("hidden_dim", "activation") if cfg.hidden_dim else ("hidden_dim",)
        )
        assert AdapterConfig(type=name, hidden_dim=8, activation="silu").activation == "silu"
        with pytest.raises(
            ValueError, match=rf"adapter\.type = '{name}' does not use adapter\.activation = 'silu'"
        ):
            AdapterConfig(type=name, activation="silu")

    def test_an_undeclared_adapter_accepts_every_field(self, plugin_adapter):
        """Declaring nothing keeps the behavior from before declarations: no field
        is rejected and the builder receives every field."""
        name, seen = plugin_adapter()
        settings = {
            "hidden_dim": 8,
            "activation": "silu",
            "pre_norm": "rmsnorm",
            "pool_window": 3,
            "pool_heads": 8,
        }
        build_adapter(AdapterConfig(type=name, **settings), in_dim=8, out_dim=4)
        assert seen == {**settings, "norm_eps": 1e-5}

    def test_a_declaration_naming_an_unknown_field_is_rejected(self, plugin_adapter):
        name, _ = plugin_adapter(accepts=("hiden_dim",))
        with pytest.raises(
            ValueError, match=r"declares fields AdapterConfig does not have: \['hiden_dim'\]"
        ):
            AdapterConfig(type=name)


_UNUSED_SETTINGS = [
    ("linear", {}, "hidden_dim", 8),
    ("linear", {}, "activation", "silu"),
    ("linear", {}, "pre_norm", "rmsnorm"),
    ("linear", {}, "pool_window", 3),
    ("linear", {}, "pool_heads", 8),
    ("mlp_2layer", {}, "pool_window", 3),
    ("mlp_2layer", {}, "pool_heads", 8),
    ("avgpool", {}, "activation", "silu"),
    ("avgpool", {}, "pre_norm", "rmsnorm"),
    ("avgpool", {}, "pool_heads", 8),
    ("avgpool", {"hidden_dim": 8}, "pool_heads", 8),
    ("attentional_pool", {}, "hidden_dim", 8),
    ("attentional_pool", {}, "activation", "silu"),
    ("attentional_pool", {}, "pre_norm", "rmsnorm"),
]


class TestAdapterConfigRejectsUnusedFields:
    @pytest.mark.parametrize(("adapter_type", "base", "field", "value"), _UNUSED_SETTINGS)
    def test_a_field_the_adapter_does_not_use_is_rejected(self, adapter_type, base, field, value):
        with pytest.raises(
            ValueError,
            match=rf"adapter\.type = '{adapter_type}' does not use "
            rf"adapter\.{field} = {re.escape(repr(value))}",
        ):
            AdapterConfig(type=adapter_type, **base, **{field: value})

    @pytest.mark.parametrize("adapter_type", _BUILTIN_ADAPTERS)
    def test_fields_at_their_defaults_never_raise(self, adapter_type):
        defaults = {
            f.name: f.default for f in dataclasses.fields(AdapterConfig) if f.name != "type"
        }
        assert AdapterConfig(type=adapter_type) == AdapterConfig(type=adapter_type, **defaults)

    @pytest.mark.parametrize(
        ("adapter_type", "settings"),
        [
            ("mlp_2layer", {"hidden_dim": 8, "activation": "silu", "pre_norm": "rmsnorm"}),
            ("avgpool", {"pool_window": 3}),
            (
                "avgpool",
                {"pool_window": 3, "hidden_dim": 8, "activation": "silu", "pre_norm": "rmsnorm"},
            ),
            ("attentional_pool", {"pool_window": 3, "pool_heads": 8}),
        ],
        ids=["mlp_2layer", "avgpool", "avgpool-mlp-head", "attentional_pool"],
    )
    def test_fields_the_adapter_uses_are_accepted(self, adapter_type, settings):
        cfg = AdapterConfig(type=adapter_type, **settings)
        assert {k: getattr(cfg, k) for k in settings} == settings

    def test_every_unused_field_is_named_with_its_default(self):
        with pytest.raises(ValueError) as info:
            AdapterConfig(type="linear", hidden_dim=8, pool_heads=8)
        message = str(info.value)
        assert "adapter.hidden_dim = 8, adapter.pool_heads = 8" in message
        assert "it uses no adapter fields" in message
        assert "(hidden_dim = 0, pool_heads = 16)" in message

    def test_avgpool_lists_hidden_dim_among_the_fields_it_uses(self):
        with pytest.raises(ValueError, match="with these settings it uses hidden_dim, pool_window"):
            AdapterConfig(type="avgpool", pre_norm="rmsnorm")

    def test_load_config_rejects_it(self, tmp_path):
        path = tmp_path / "adapter.toml"
        path.write_text('[adapter]\ntype = "avgpool"\npool_heads = 8\n')
        with pytest.raises(
            ValueError, match=r"adapter\.type = 'avgpool' does not use adapter\.pool_heads = 8"
        ):
            load_config(path, cli_args=[])

    def test_cli_overrides_turn_on_the_head(self, tmp_path):
        path = tmp_path / "adapter.toml"
        path.write_text('[adapter]\ntype = "avgpool"\n')
        cfg = load_config(path, cli_args=["--adapter.hidden_dim=24", "--adapter.pre_norm=rmsnorm"])
        assert cfg.adapter is not None
        assert (cfg.adapter.hidden_dim, cfg.adapter.pre_norm) == (24, "rmsnorm")

    def test_a_cli_override_of_an_unused_field_is_rejected(self, tmp_path):
        path = tmp_path / "adapter.toml"
        path.write_text('[adapter]\ntype = "mlp_2layer"\n')
        with pytest.raises(
            ValueError, match=r"adapter\.type = 'mlp_2layer' does not use adapter\.pool_window = 3"
        ):
            load_config(path, cli_args=["--adapter.pool_window=3"])
