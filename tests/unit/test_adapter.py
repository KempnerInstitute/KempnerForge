"""Unit tests for VLM adapters and the adapter registry."""

from __future__ import annotations

import pytest
import torch

# Importing the module registers the builders under the shared registry.
import kempnerforge.model.adapter  # noqa: F401
from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.registry import registry
from kempnerforge.model.adapter import (
    LinearAdapter,
    MLP2LayerAdapter,
    build_adapter,
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
