"""Unit tests for optimizers."""

from __future__ import annotations

import torch
import torch.nn as nn

from kempnerforge.config.schema import OptimizerConfig
from kempnerforge.training.optimizer import Muon, _is_muon_eligible, _newton_schulz, build_optimizer


class TestNewtonSchulz:
    def test_produces_orthogonal_matrix(self):
        """Newton-Schulz output should be approximately orthogonal."""
        torch.manual_seed(42)
        G = torch.randn(64, 64)
        U = _newton_schulz(G, steps=5)

        # U @ U^T should be close to identity (scaled)
        product = U @ U.T
        diag = torch.diag(product)
        off_diag = product - torch.diag(diag)

        # Off-diagonal elements should be small (5 iterations is approximate)
        assert off_diag.abs().max() < 0.15

    def test_rectangular_matrix(self):
        """Newton-Schulz should work with non-square matrices."""
        torch.manual_seed(0)
        G = torch.randn(32, 128)
        U = _newton_schulz(G, steps=5)
        assert U.shape == (32, 128)
        assert torch.isfinite(U).all()

    def test_deterministic(self):
        G = torch.randn(16, 16)
        U1 = _newton_schulz(G, steps=5)
        U2 = _newton_schulz(G, steps=5)
        torch.testing.assert_close(U1, U2)


class TestMuon:
    def _make_model(self):
        return nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def test_decreases_loss(self):
        """Muon should decrease loss on a simple problem."""
        torch.manual_seed(42)
        model = self._make_model()
        x = torch.randn(16, 32)
        y = torch.randint(0, 10, (16,))

        config = OptimizerConfig(name="muon", lr=0.01, weight_decay=0.0)
        optimizer = build_optimizer(model, config)

        initial_loss = nn.functional.cross_entropy(model(x), y).item()

        for _ in range(20):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.cross_entropy(model(x), y).item()
        assert final_loss < initial_loss

    def test_1d_params_use_adam(self):
        """Biases and norm params (1D) should be handled by internal AdamW."""
        model = self._make_model()
        config = OptimizerConfig(name="muon", lr=0.01)
        optimizer = build_optimizer(model, config)

        assert isinstance(optimizer, Muon)
        assert optimizer._adam is not None

        # Muon params should all be eligible (2D with reasonable aspect ratio)
        muon_params = [p for g in optimizer.param_groups for p in g["params"]]
        for p in muon_params:
            assert _is_muon_eligible(p), "Muon should only get NS-eligible params"

    def test_rectangular_params_use_adam(self):
        """Highly rectangular params (embeddings) should use AdamW, not Muon."""
        model = nn.Sequential(
            nn.Embedding(32000, 64),  # aspect ratio 500:1 — too rectangular
            nn.Linear(64, 64),
        )
        config = OptimizerConfig(name="muon", lr=0.01)
        optimizer = build_optimizer(model, config)

        muon_params = [p for g in optimizer.param_groups for p in g["params"]]
        adam_params = [p for g in optimizer._adam.param_groups for p in g["params"]]

        # Embedding weight (32000x64) should be in AdamW
        assert any(p.shape == (32000, 64) for p in adam_params)
        # Linear weight (64x64) should be in Muon
        assert any(p.shape == (64, 64) for p in muon_params)

    def test_produces_finite_updates(self):
        """All parameters should remain finite after Muon steps."""
        torch.manual_seed(0)
        model = self._make_model()
        config = OptimizerConfig(name="muon", lr=0.01)
        optimizer = build_optimizer(model, config)

        x = torch.randn(8, 32)
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(x).sum()
            loss.backward()
            optimizer.step()

        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_weight_decay(self):
        """Muon with weight decay should shrink parameters."""
        torch.manual_seed(42)
        model = nn.Linear(32, 32)
        weight_before = model.weight.data.norm().item()

        config = OptimizerConfig(name="muon", lr=0.01, weight_decay=0.5)
        optimizer = build_optimizer(model, config)

        x = torch.randn(4, 32)
        for _ in range(10):
            optimizer.zero_grad()
            loss = model(x).sum()
            loss.backward()
            optimizer.step()

        weight_after = model.weight.data.norm().item()
        assert weight_after < weight_before

    def test_momentum_config(self):
        config = OptimizerConfig(name="muon", muon_momentum=0.9, muon_ns_steps=3)
        model = nn.Linear(16, 16)
        optimizer = build_optimizer(model, config)
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["ns_steps"] == 3

    def test_adam_lr_separate(self):
        """muon_adam_lr should set a different LR for the internal AdamW."""
        model = self._make_model()
        config = OptimizerConfig(name="muon", lr=0.02, muon_adam_lr=3e-4)
        optimizer = build_optimizer(model, config)

        assert optimizer._initial_lr == 0.02
        assert optimizer._initial_adam_lr == 3e-4
        assert optimizer._adam.param_groups[0]["lr"] == 3e-4

    def test_adam_lr_defaults_to_muon_lr(self):
        """When muon_adam_lr is None, internal AdamW should use the same LR as Muon."""
        model = self._make_model()
        config = OptimizerConfig(name="muon", lr=0.01, muon_adam_lr=None)
        optimizer = build_optimizer(model, config)

        assert optimizer._initial_adam_lr == 0.01
        assert optimizer._adam.param_groups[0]["lr"] == 0.01

    def test_adam_lr_scales_with_scheduler(self):
        """When the scheduler changes Muon's LR, internal Adam LR should scale."""
        model = self._make_model()
        config = OptimizerConfig(name="muon", lr=0.02, muon_adam_lr=3e-4)
        optimizer = build_optimizer(model, config)

        # Simulate scheduler halving the Muon LR
        optimizer.param_groups[0]["lr"] = 0.01  # halved from 0.02

        # Run a step — Adam LR should scale proportionally
        x = torch.randn(4, 32)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

        # Adam LR should be halved too: 3e-4 * (0.01/0.02) = 1.5e-4
        expected_adam_lr = 3e-4 * (0.01 / 0.02)
        assert abs(optimizer._adam.param_groups[0]["lr"] - expected_adam_lr) < 1e-10

    def test_momentum_buffer_matches_grad_type(self):
        """Momentum buffer should be same type as p.grad (regular tensor here)."""
        torch.manual_seed(0)
        model = nn.Linear(16, 16)
        config = OptimizerConfig(name="muon", lr=0.01)
        optimizer = build_optimizer(model, config)

        x = torch.randn(4, 16)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

        # In non-FSDP case, momentum buffer should be a regular tensor
        for p in optimizer.param_groups[0]["params"]:
            if p in optimizer.state:
                buf = optimizer.state[p]["momentum_buffer"]
                assert isinstance(buf, torch.Tensor)
                assert buf.shape == p.grad.shape

    def test_state_dict_roundtrip(self):
        """state_dict/load_state_dict must preserve both Muon and internal AdamW state."""
        torch.manual_seed(42)
        model = self._make_model()
        config = OptimizerConfig(name="muon", lr=0.02, muon_adam_lr=3e-4)
        optimizer = build_optimizer(model, config)

        # Run a few steps to populate optimizer state
        x = torch.randn(8, 32)
        for _ in range(5):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x), torch.randint(0, 10, (8,))).backward()
            optimizer.step()

        # Save state
        sd = optimizer.state_dict()
        assert "_adam_state" in sd, "Internal AdamW state missing from state_dict"
        assert "_initial_lr" in sd
        assert "_initial_adam_lr" in sd

        # Create a fresh optimizer and load state
        model2 = self._make_model()
        config2 = OptimizerConfig(name="muon", lr=0.02, muon_adam_lr=3e-4)
        optimizer2 = build_optimizer(model2, config2)
        optimizer2.load_state_dict(sd)

        # Verify Muon momentum buffers match
        for p1, p2 in zip(
            [p for g in optimizer.param_groups for p in g["params"]],
            [p for g in optimizer2.param_groups for p in g["params"]],
            strict=True,
        ):
            if p1 in optimizer.state and "momentum_buffer" in optimizer.state[p1]:
                buf1 = optimizer.state[p1]["momentum_buffer"]
                buf2 = optimizer2.state[p2]["momentum_buffer"]
                torch.testing.assert_close(buf1, buf2)

        # Verify internal AdamW state was restored (non-empty)
        adam_states = [s for s in optimizer2._adam.state.values() if s]
        assert len(adam_states) > 0, "Internal AdamW state was not restored"

    def test_zero_grad_clears_delegated_params(self):
        """Params routed to the inner AdamW must be zeroed too.

        Only Muon's half is registered with the base class, so the inherited
        zero_grad leaves everything delegated to _adam accumulating. The shape
        matters: at an aspect ratio under 10 the embedding stays on Muon's side
        and this passes even when the delegation is missing.
        """
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Embedding(32000, 64),  # aspect ratio 500:1 -> inner AdamW
            nn.Linear(64, 64),
        )
        # lr tiny and no decay, so parameters barely move and the per-step
        # gradient is constant unless it is accumulating.
        config = OptimizerConfig(name="muon", lr=1e-12, weight_decay=0.0)
        optimizer = build_optimizer(model, config)
        embedding = model[0].weight
        assert any(p is embedding for g in optimizer._adam.param_groups for p in g["params"]), (
            "test needs the embedding on the AdamW side"
        )

        x = torch.randint(0, 32000, (8, 16))
        grad_sums = []
        for _ in range(3):
            optimizer.zero_grad()
            model(x).square().sum().backward()
            grad_sums.append(embedding.grad.abs().sum().item())
            optimizer.step()

        torch.testing.assert_close(grad_sums[1], grad_sums[0], rtol=1e-5, atol=0.0)
        torch.testing.assert_close(grad_sums[2], grad_sums[0], rtol=1e-5, atol=0.0)

    def test_zero_grad_still_clears_muon_params(self):
        """Regression guard: the Muon half must keep being zeroed."""
        torch.manual_seed(0)
        model = self._make_model()
        optimizer = build_optimizer(model, OptimizerConfig(name="muon", lr=0.01))

        x = torch.randn(8, 32)
        nn.functional.cross_entropy(model(x), torch.randint(0, 10, (8,))).backward()
        muon_params = [p for g in optimizer.param_groups for p in g["params"]]
        assert all(p.grad is not None for p in muon_params)

        optimizer.zero_grad()
        assert all(p.grad is None for p in muon_params)

    def test_zero_grad_forwards_set_to_none_false(self):
        """set_to_none=False has to reach the inner AdamW, not just Muon."""
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Embedding(32000, 64),
            nn.Linear(64, 64),
        )
        optimizer = build_optimizer(model, OptimizerConfig(name="muon", lr=0.01))
        embedding = model[0].weight

        x = torch.randint(0, 32000, (8, 16))
        model(x).square().sum().backward()
        assert embedding.grad is not None

        optimizer.zero_grad(set_to_none=False)
        assert embedding.grad is not None, "set_to_none=False must keep the tensor"
        assert embedding.grad.abs().sum().item() == 0.0


class TestAdamW:
    def test_build_adamw(self):
        model = nn.Linear(32, 10)
        config = OptimizerConfig(name="adamw", lr=1e-3)
        optimizer = build_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_weight_decay_groups(self):
        """2D params should get decay, 1D should not."""
        model = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64), nn.Linear(64, 10))
        config = OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.1)
        optimizer = build_optimizer(model, config)

        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        assert decay_group["weight_decay"] == 0.1
        assert no_decay_group["weight_decay"] == 0.0

        for p in decay_group["params"]:
            assert p.ndim >= 2
        for p in no_decay_group["params"]:
            assert p.ndim <= 1
