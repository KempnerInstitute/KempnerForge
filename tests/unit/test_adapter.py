"""Unit tests for VLM adapters and the adapter registry."""

from __future__ import annotations

import pytest
import torch

# Importing the module registers the builders under the shared registry.
import kempnerforge.model.adapter  # noqa: F401
from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.registry import registry
from kempnerforge.model.adapter import (
    POOLING_ADAPTER_TYPES,
    AttentionalPoolAdapter,
    AvgPoolAdapter,
    LinearAdapter,
    MLP2LayerAdapter,
    VisionAdapter,
    build_adapter,
    pooled_token_count,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# MLP2LayerAdapter
# ---------------------------------------------------------------------------


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
        # Not guaranteed bit-different (unlikely but possible), so just
        # check it ran without error and weights are finite.
        assert torch.isfinite(adapter.proj1.weight).all()
        assert adapter.proj1.weight.shape == w1.shape


# ---------------------------------------------------------------------------
# LinearAdapter
# ---------------------------------------------------------------------------


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
        """LinearAdapter has a single ``proj`` Linear; no hidden_dim or
        activation parameters exist. The builder must ignore them."""
        # The dispatcher (build_adapter) passes hidden_dim/activation via
        # AdapterConfig.extra_kwargs(); the linear builder swallows them.
        cfg = AdapterConfig(type="linear", hidden_dim=42, activation="silu")
        adapter = build_adapter(cfg, in_dim=8, out_dim=4)
        assert isinstance(adapter, LinearAdapter)
        # No proj1/proj2 attributes; only proj.
        assert hasattr(adapter, "proj")
        assert not hasattr(adapter, "proj1")
        assert not hasattr(adapter, "proj2")


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# AdapterConfig
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# build_adapter dispatch
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# pooled_token_count (pure helper, single source of truth for the post-pool
# token count)
# ---------------------------------------------------------------------------


class TestPooledTokenCount:
    @pytest.mark.parametrize(
        ("n_in", "window", "expected"),
        [
            (196, 2, 49),  # 14x14 grid, divisible -> 7x7
            (256, 2, 64),  # 16x16 -> 8x8
            (729, 3, 81),  # 27x27 (Molmo2 SigLIP 378/14) -> 9x9
            (16, 2, 4),  # 4x4 -> 2x2
            (16, 3, 4),  # 4x4 ragged: ceil(4/3)=2 -> 2x2
            (100, 3, 16),  # 10x10 ragged: ceil(10/3)=4 -> 4x4
            (16, 1, 16),  # window 1 == identity
            (16, 4, 1),  # whole grid into one token
        ],
    )
    def test_counts(self, n_in, window, expected):
        assert pooled_token_count(n_in, window) == expected

    def test_non_square_grid_raises(self):
        with pytest.raises(ValueError, match="square patch grid"):
            pooled_token_count(8, 2)  # 8 is not a perfect square

    def test_non_positive_window_raises(self):
        with pytest.raises(ValueError, match="window must be positive"):
            pooled_token_count(16, 0)

    def test_non_positive_tokens_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            pooled_token_count(0, 2)

    def test_require_divisible_raises_on_ragged(self):
        # attentional_pool path: a ragged grid is rejected up front, not at forward.
        with pytest.raises(ValueError, match="ragged grid"):
            pooled_token_count(196, 3, require_divisible=True)  # 14x14 not divisible by 3

    def test_require_divisible_ok_when_divisible(self):
        assert pooled_token_count(196, 2, require_divisible=True) == 49  # 14x14 -> 7x7


# ---------------------------------------------------------------------------
# AvgPoolAdapter
# ---------------------------------------------------------------------------


class TestAvgPoolAdapter:
    def test_forward_shape_divisible(self):
        adapter = AvgPoolAdapter(in_dim=96, out_dim=64, pool_window=2).to(DEVICE)
        x = torch.randn(2, 16, 96, device=DEVICE)  # 4x4 grid
        out = adapter(x)
        assert out.shape == (2, 4, 64)  # 2x2 windows

    def test_forward_shape_ragged(self):
        adapter = AvgPoolAdapter(in_dim=96, out_dim=64, pool_window=3).to(DEVICE)
        x = torch.randn(2, 16, 96, device=DEVICE)  # 4x4 grid, ceil(4/3)=2
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
        x = torch.randn(1, 16, 32)  # 4x4 grid
        # Override to window 4 -> ceil(4/4)=1 -> single token.
        assert adapter(x, pool_window=4).shape == (1, 1, 16)
        # Default still 2x2 -> 4 tokens.
        assert adapter(x).shape == (1, 4, 16)

    def test_non_square_input_raises(self):
        adapter = AvgPoolAdapter(in_dim=8, out_dim=4, pool_window=2)
        with pytest.raises(ValueError, match="square patch grid"):
            adapter(torch.randn(1, 8, 8))  # 8 not a perfect square

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
        # Make proj an identity so we can read the pooled values directly.
        with torch.no_grad():
            adapter.proj.weight.copy_(torch.eye(4))
            adapter.proj.bias.zero_()
        x = torch.arange(16.0).view(1, 16, 1).expand(1, 16, 4).contiguous()  # 4x4 grid
        out = adapter(x)
        # Top-left window = mean of grid cells (0,0),(0,1),(1,0),(1,1) =
        # tokens 0,1,4,5 -> mean 2.5.
        assert torch.allclose(out[0, 0], torch.full((4,), 2.5))

    def test_forward_rejects_nonpositive_window_override(self):
        adapter = AvgPoolAdapter(in_dim=8, out_dim=4, pool_window=2)
        with pytest.raises(ValueError, match="pool_window must be positive"):
            adapter(torch.randn(1, 16, 8), pool_window=0)


# ---------------------------------------------------------------------------
# AttentionalPoolAdapter
# ---------------------------------------------------------------------------


class TestAttentionalPoolAdapter:
    def test_forward_shape(self):
        adapter = AttentionalPoolAdapter(in_dim=96, out_dim=64, pool_window=2, pool_heads=16).to(
            DEVICE
        )
        x = torch.randn(2, 16, 96, device=DEVICE)  # 4x4 grid
        out = adapter(x)
        assert out.shape == (2, 4, 64)

    def test_is_vision_adapter(self):
        assert isinstance(AttentionalPoolAdapter(in_dim=16, out_dim=8, pool_heads=4), VisionAdapter)

    @pytest.mark.parametrize(("n_in", "window"), [(16, 2), (256, 2), (729, 3)])
    def test_output_num_tokens_matches_forward(self, n_in, window):
        adapter = AttentionalPoolAdapter(in_dim=32, out_dim=16, pool_window=window, pool_heads=4)
        x = torch.randn(1, n_in, 32)
        assert adapter(x).shape[1] == adapter.output_num_tokens(n_in)

    def test_ragged_grid_raises(self):
        adapter = AttentionalPoolAdapter(in_dim=96, out_dim=64, pool_window=3, pool_heads=16)
        with pytest.raises(ValueError, match="divisible"):
            adapter(torch.randn(1, 16, 96))  # 4x4 grid, not divisible by 3

    def test_output_num_tokens_rejects_ragged(self):
        # The static count must mirror forward()'s ragged rejection so an invalid
        # config fails at build / seq-len-check time, not at the first step.
        adapter = AttentionalPoolAdapter(in_dim=96, out_dim=64, pool_window=3, pool_heads=16)
        with pytest.raises(ValueError, match="ragged grid"):
            adapter.output_num_tokens(16)  # 4x4 grid, not divisible by 3

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

    def test_reset_parameters_reinitializes(self):
        adapter = AttentionalPoolAdapter(in_dim=16, out_dim=8, pool_heads=4)
        adapter.reset_parameters()
        assert torch.isfinite(adapter.out_proj.weight).all()

    def test_forward_rejects_nonpositive_window_override(self):
        adapter = AttentionalPoolAdapter(in_dim=16, out_dim=8, pool_window=2, pool_heads=4)
        with pytest.raises(ValueError, match="pool_window must be positive"):
            adapter(torch.randn(1, 16, 16), pool_window=0)


# ---------------------------------------------------------------------------
# Pooling adapters: registry + config wiring
# ---------------------------------------------------------------------------


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
        kwargs = AdapterConfig(type="avgpool", pool_window=3, pool_heads=8).extra_kwargs()
        assert kwargs["pool_window"] == 3
        assert kwargs["pool_heads"] == 8

    def test_output_num_tokens_identity_for_projection(self):
        assert AdapterConfig(type="mlp_2layer").output_num_tokens(196) == 196
        assert AdapterConfig(type="linear").output_num_tokens(196) == 196

    def test_output_num_tokens_pools_for_avgpool(self):
        assert AdapterConfig(type="avgpool", pool_window=2).output_num_tokens(196) == 49

    def test_output_num_tokens_pools_for_attentional(self):
        assert AdapterConfig(type="attentional_pool", pool_window=3).output_num_tokens(729) == 81

    def test_attentional_output_num_tokens_rejects_ragged(self):
        # Config-time check rejects a ragged attentional_pool grid (mirrors forward),
        # so the misconfig fails at config load, not at the first training step.
        with pytest.raises(ValueError, match="ragged grid"):
            AdapterConfig(type="attentional_pool", pool_window=3).output_num_tokens(196)

    def test_avgpool_output_num_tokens_allows_ragged(self):
        # avgpool pools ragged edges, so the same ragged grid is fine (ceil math).
        assert AdapterConfig(type="avgpool", pool_window=3).output_num_tokens(196) == 25

    def test_output_num_tokens_passthrough_on_sentinel(self):
        # num_tokens=0 ("infer at build time") must not trigger the square check.
        assert AdapterConfig(type="avgpool").output_num_tokens(0) == 0

    @pytest.mark.parametrize("adapter_type", ["avgpool", "attentional_pool"])
    def test_config_count_matches_module_count(self, adapter_type):
        """Config-time estimate must equal the built module's count so the
        config-time seq-len check matches the build-time budget."""
        cfg = AdapterConfig(type=adapter_type, pool_window=2, pool_heads=4)
        module = build_adapter(cfg, in_dim=32, out_dim=16)
        assert cfg.output_num_tokens(256) == module.output_num_tokens(256)


class TestBuildAdapterPooling:
    def test_dispatches_to_avgpool_with_window(self):
        cfg = AdapterConfig(type="avgpool", pool_window=2)
        adapter = build_adapter(cfg, in_dim=32, out_dim=16)
        assert isinstance(adapter, AvgPoolAdapter)
        assert adapter.pool_window == 2
        out = adapter(torch.randn(1, 16, 32))  # 4x4 grid -> 2x2
        assert out.shape == (1, 4, 16)

    def test_dispatches_to_attentional_with_heads(self):
        cfg = AdapterConfig(type="attentional_pool", pool_window=2, pool_heads=8)
        adapter = build_adapter(cfg, in_dim=32, out_dim=16)
        assert isinstance(adapter, AttentionalPoolAdapter)
        assert adapter.pool_heads == 8
