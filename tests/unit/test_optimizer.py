"""Unit tests for optimizers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

from kempnerforge.config.schema import ModelConfig, OptimizerConfig
from kempnerforge.model.moma import ExpertChoiceMoE
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import (
    Muon,
    _muon_update_scale,
    _newton_schulz,
    build_optimizer,
)


def _split(optimizer: Muon) -> tuple[set[torch.Tensor], set[torch.Tensor]]:
    """The parameters Muon orthogonalizes and the ones its inner AdamW updates."""
    muon = {p for group in optimizer.param_groups for p in group["params"]}
    adam = {
        p
        for inner in optimizer.inner_optimizers
        for group in inner.param_groups
        for p in group["params"]
    }
    return muon, adam


class _OutputShapes(TorchFunctionMode):
    """Records the shape of every tensor a torch function returns."""

    def __init__(self) -> None:
        super().__init__()
        self.shapes: set[tuple[int, ...]] = set()

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor):
            self.shapes.add(tuple(out.shape))
        return out


class TestNewtonSchulz:
    def test_produces_orthogonal_matrix(self):
        """Newton-Schulz output should be approximately orthogonal."""
        torch.manual_seed(42)
        G = torch.randn(64, 64)
        U = _newton_schulz(G, steps=5)

        product = U @ U.T
        diag = torch.diag(product)
        off_diag = product - torch.diag(diag)

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

    def test_tall_matrix_matches_iterating_on_it(self):
        """A tall matrix gets what iterating on it untransposed gives."""
        torch.manual_seed(0)
        G = torch.randn(48, 16, dtype=torch.float64)
        X = G / (G.norm() + 1e-7)
        for _ in range(5):
            A = X @ X.T
            X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
        torch.testing.assert_close(_newton_schulz(G, steps=5), X)

    def test_tall_matrix_forms_nothing_rows_by_rows(self):
        """A tall matrix iterates on its transpose, so its Gram matrix is cols x cols."""
        with _OutputShapes() as recorded:
            U = _newton_schulz(torch.randn(64, 8), steps=5)
        assert U.shape == (64, 8)
        assert (8, 8) in recorded.shapes
        assert (64, 64) not in recorded.shapes


class TestMuon:
    @pytest.mark.parametrize(
        ("adjust_lr", "tall", "wide"),
        [("original", 2.0, 1.0), ("match_rms_adamw", 0.8, 0.8), ("spectral_unclamped", 2.0, 0.5)],
    )
    def test_update_scale_modes(self, adjust_lr, tall, wide):
        """Each adjust_lr mode's factor for a tall 16x4 and a wide 4x16 matrix."""
        assert _muon_update_scale(adjust_lr, 16, 4) == pytest.approx(tall)
        assert _muon_update_scale(adjust_lr, 4, 16) == pytest.approx(wide)

    @pytest.mark.parametrize(("rows", "cols"), [(16, 4), (4, 16)])
    @pytest.mark.parametrize("adjust_lr", ["original", "match_rms_adamw", "spectral_unclamped"])
    def test_adjust_lr_scales_the_step(self, adjust_lr, rows, cols):
        """One step moves a tall or a wide weight by lr times the mode's scale times NS(grad)."""
        torch.manual_seed(0)
        weight = nn.Parameter(torch.zeros(rows, cols))
        weight.grad = torch.randn(rows, cols)
        optimizer = Muon([{"params": [weight]}], [{"params": []}], lr=0.1, adjust_lr=adjust_lr)
        optimizer.step()
        expected = -0.1 * _muon_update_scale(adjust_lr, rows, cols) * _newton_schulz(weight.grad)
        torch.testing.assert_close(weight.data, expected)

    def test_steps_a_4d_weight_whose_matrix_is_tall(self):
        """A 4-D weight flattening to a tall matrix steps by NS of that matrix, in its own shape."""
        torch.manual_seed(0)
        weight = nn.Parameter(torch.zeros(64, 3, 2, 2))
        weight.grad = torch.randn(64, 3, 2, 2)
        Muon([{"params": [weight]}], [{"params": []}], lr=0.1).step()
        matrix = weight.grad.view(64, 12)
        expected = -0.1 * _muon_update_scale("original", 64, 12) * _newton_schulz(matrix)
        torch.testing.assert_close(weight.data.view(64, 12), expected)

    def test_nesterov_orthogonalizes_the_lookahead(self):
        """With nesterov, Newton-Schulz sees grad + momentum * buffer."""
        torch.manual_seed(0)
        grads = [torch.randn(8, 8), torch.randn(8, 8)]
        weight = nn.Parameter(torch.zeros(8, 8))
        optimizer = Muon([{"params": [weight]}], [{"params": []}], lr=0.1, nesterov=True)
        expected = torch.zeros(8, 8)
        buf = torch.zeros(8, 8)
        for grad in grads:
            weight.grad = grad.clone()
            optimizer.step()
            buf = 0.95 * buf + grad
            expected -= 0.1 * _newton_schulz(grad + 0.95 * buf)
        torch.testing.assert_close(weight.data, expected)

    def test_nesterov_off_matches_the_plain_buffer(self):
        """Without nesterov, Newton-Schulz sees the momentum buffer."""
        torch.manual_seed(0)
        grads = [torch.randn(8, 8), torch.randn(8, 8)]
        weight = nn.Parameter(torch.zeros(8, 8))
        optimizer = Muon([{"params": [weight]}], [{"params": []}], lr=0.1)
        expected = torch.zeros(8, 8)
        buf = torch.zeros(8, 8)
        for grad in grads:
            weight.grad = grad.clone()
            optimizer.step()
            buf = 0.95 * buf + grad
            expected -= 0.1 * _newton_schulz(buf)
        torch.testing.assert_close(weight.data, expected)

    @pytest.mark.parametrize("adjust_lr", ["original", "match_rms_adamw"])
    def test_tracks_torch_muon(self, adjust_lr):
        """With nesterov, three steps land close to torch.optim.Muon's (which runs NS in bf16)."""
        if not hasattr(torch.optim, "Muon"):
            pytest.skip("torch.optim.Muon needs torch >= 2.9")
        torch.manual_seed(0)
        init = torch.randn(16, 8)
        grads = [torch.randn(16, 8) for _ in range(3)]
        ours = nn.Parameter(init.clone())
        theirs = nn.Parameter(init.clone())
        kf = Muon(
            [{"params": [ours]}], [{"params": []}], lr=0.02, nesterov=True, adjust_lr=adjust_lr
        )
        ref = torch.optim.Muon(
            [theirs],
            lr=0.02,
            weight_decay=0.0,
            nesterov=True,
            adjust_lr_fn=None if adjust_lr == "original" else adjust_lr,
        )
        for grad in grads:
            ours.grad, theirs.grad = grad.clone(), grad.clone()
            kf.step()
            ref.step()
        error = (ours.data - theirs.data).norm() / (theirs.data - init).norm()
        assert error < 0.05

    def test_rejects_unknown_adjust_lr(self):
        with pytest.raises(ValueError, match="adjust_lr must be one of"):
            Muon([{"params": [nn.Parameter(torch.zeros(2, 2))]}], [{"params": []}], adjust_lr="x")

    def test_build_optimizer_passes_nesterov_and_adjust_lr(self):
        config = OptimizerConfig(name="muon", muon_nesterov=True, muon_adjust_lr="match_rms_adamw")
        optimizer = build_optimizer(nn.Linear(8, 8), config)
        assert optimizer.param_groups[0]["nesterov"] is True
        assert optimizer.param_groups[0]["adjust_lr"] == "match_rms_adamw"

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

        muon, adam = _split(optimizer)
        assert muon == {model[0].weight, model[2].weight}
        assert adam == {model[0].bias, model[2].bias}

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

        optimizer.param_groups[0]["lr"] = 0.01

        x = torch.randn(4, 32)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

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

        x = torch.randn(8, 32)
        for _ in range(5):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x), torch.randint(0, 10, (8,))).backward()
            optimizer.step()

        sd = optimizer.state_dict()
        assert "_adam_state" in sd, "Internal AdamW state missing from state_dict"
        assert "_initial_lr" in sd
        assert "_initial_adam_lr" in sd

        model2 = self._make_model()
        config2 = OptimizerConfig(name="muon", lr=0.02, muon_adam_lr=3e-4)
        optimizer2 = build_optimizer(model2, config2)
        optimizer2.load_state_dict(sd)

        for p1, p2 in zip(
            [p for g in optimizer.param_groups for p in g["params"]],
            [p for g in optimizer2.param_groups for p in g["params"]],
            strict=True,
        ):
            if p1 in optimizer.state and "momentum_buffer" in optimizer.state[p1]:
                buf1 = optimizer.state[p1]["momentum_buffer"]
                buf2 = optimizer2.state[p2]["momentum_buffer"]
                torch.testing.assert_close(buf1, buf2)

        adam_states = [s for s in optimizer2._adam.state.values() if s]
        assert len(adam_states) > 0, "Internal AdamW state was not restored"

    def test_inner_optimizers_lists_the_inner_adamw(self):
        """``inner_optimizers`` returns Muon's inner AdamW."""
        optimizer = build_optimizer(self._make_model(), OptimizerConfig(name="muon", lr=0.01))
        assert optimizer.inner_optimizers == (optimizer._adam,)

    def test_zero_grad_clears_delegated_params(self):
        """Params routed to the inner AdamW must be zeroed too.

        Only Muon's half is registered with the base class, so the inherited
        zero_grad leaves everything delegated to _adam accumulating. The
        embedding is not an nn.Linear, so it is delegated and would be missed.
        """
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Embedding(32000, 64),
            nn.Linear(64, 64),
        )
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

    def test_zero_grad_without_delegated_params(self):
        """A model whose parameters are all Muon-eligible leaves _adam as None."""
        torch.manual_seed(0)
        model = nn.Linear(32, 32, bias=False)
        optimizer = build_optimizer(model, OptimizerConfig(name="muon", lr=0.01))
        assert optimizer._adam is None

        model(torch.randn(4, 32)).sum().backward()
        assert model.weight.grad is not None

        optimizer.zero_grad()
        assert model.weight.grad is None

    def test_no_delegated_params_step_and_roundtrip(self):
        """step/state_dict/load_state_dict must tolerate _adam being None."""
        torch.manual_seed(0)
        model = nn.Linear(32, 32, bias=False)
        optimizer = build_optimizer(model, OptimizerConfig(name="muon", lr=0.01))
        assert optimizer._adam is None
        assert optimizer.inner_optimizers == ()

        optimizer.zero_grad()
        model(torch.randn(4, 32)).sum().backward()
        optimizer.step()

        sd = optimizer.state_dict()
        assert "_adam_state" not in sd

        model2 = nn.Linear(32, 32, bias=False)
        optimizer2 = build_optimizer(model2, OptimizerConfig(name="muon", lr=0.01))
        optimizer2.load_state_dict(sd)
        assert optimizer2._adam is None

    def test_exact_orthogonalize_matches_local_when_unsharded(self):
        """Unsharded, exact and local orthogonalization step identically."""
        torch.manual_seed(0)
        grads = [torch.randn(16, 8), torch.randn(16, 8)]
        weights = [nn.Parameter(torch.zeros(16, 8)) for _ in range(2)]
        optimizers = [
            Muon([{"params": [weight]}], [{"params": []}], lr=0.1, orthogonalize=mode)
            for weight, mode in zip(weights, ("local", "exact"), strict=True)
        ]
        for grad in grads:
            for weight, optimizer in zip(weights, optimizers, strict=True):
                weight.grad = grad.clone()
                optimizer.step()
        assert torch.equal(weights[0].data, weights[1].data)

    def test_rejects_unknown_orthogonalize(self):
        with pytest.raises(ValueError, match="orthogonalize must be one of"):
            Muon(
                [{"params": [nn.Parameter(torch.zeros(2, 2))]}],
                [{"params": []}],
                orthogonalize="x",
            )

    def test_build_optimizer_passes_orthogonalize(self):
        config = OptimizerConfig(name="muon", muon_orthogonalize="exact")
        optimizer = build_optimizer(nn.Linear(8, 8), config)
        assert optimizer.param_groups[0]["orthogonalize"] == "exact"

    def test_state_saved_before_orthogonalize_loads_as_local(self):
        """A state saved before ``orthogonalize`` existed loads with "local", whatever is set."""
        weight = nn.Parameter(torch.zeros(4, 4))
        saved = Muon([{"params": [weight]}], [{"params": []}]).state_dict()
        for group in saved["param_groups"]:
            del group["orthogonalize"]
        optimizer = Muon([{"params": [weight]}], [{"params": []}], orthogonalize="exact")
        optimizer.load_state_dict(saved)
        assert optimizer.param_groups[0]["orthogonalize"] == "local"

    def test_state_saved_before_nesterov_and_adjust_lr_loads_the_old_update(self):
        """A state saved before ``nesterov`` and ``adjust_lr`` loads with False and "original"."""
        weight = nn.Parameter(torch.zeros(4, 4))
        saved = Muon([{"params": [weight]}], [{"params": []}]).state_dict()
        for group in saved["param_groups"]:
            del group["nesterov"], group["adjust_lr"]
        optimizer = Muon(
            [{"params": [weight]}], [{"params": []}], nesterov=True, adjust_lr="match_rms_adamw"
        )
        optimizer.load_state_dict(saved)
        assert optimizer.param_groups[0]["nesterov"] is False
        assert optimizer.param_groups[0]["adjust_lr"] == "original"


class TestMuonRouting:
    """Muon takes the hidden ``nn.Linear`` weights; its inner AdamW takes the rest."""

    @staticmethod
    def _model(**overrides: Any) -> Transformer:
        config = {"dim": 64, "n_layers": 1, "n_heads": 2, "vocab_size": 128, "max_seq_len": 16}
        return Transformer(ModelConfig(**(config | overrides)))

    @pytest.mark.parametrize("tie_embeddings", [False, True])
    @pytest.mark.parametrize("vocab_size", [128, 4096])
    def test_embedding_and_head_use_adam_at_any_vocab(self, vocab_size, tie_embeddings):
        """A vocab within or beyond 10x the width leaves the embedding and head to AdamW."""
        model = self._model(vocab_size=vocab_size, tie_embeddings=tie_embeddings)
        assert model.token_embedding is not None and model.output_head is not None
        head = model.output_head.proj.weight
        hidden = {m.weight for m in model.modules() if isinstance(m, nn.Linear)} - {head}
        muon, adam = _split(build_optimizer(model, OptimizerConfig(name="muon")))
        assert muon == hidden
        assert adam == set(model.parameters()) - hidden
        assert {model.token_embedding.embedding.weight, head} <= adam

    def test_vision_input_layers_use_adam(self):
        """A patch-embedding conv and a position table go to AdamW at shapes NS could take."""
        tower = nn.ModuleDict(
            {
                "patch_embedding": nn.Conv2d(3, 48, kernel_size=4, stride=4),
                "position_embedding": nn.Embedding(16, 48),
                "fc": nn.Linear(48, 48),
            }
        )
        muon, adam = _split(build_optimizer(tower, OptimizerConfig(name="muon")))
        assert muon == {tower["fc"].weight}
        assert {tower["patch_embedding"].weight, tower["position_embedding"].weight} <= adam

    def test_narrow_and_tall_linear_weights_use_muon(self):
        """A Linear weight goes to Muon however rectangular it is."""
        model = nn.Sequential(nn.Linear(256, 8), nn.Linear(8, 256))
        muon, _ = _split(build_optimizer(model, OptimizerConfig(name="muon")))
        assert muon == {model[0].weight, model[1].weight}

    @pytest.mark.parametrize("muon_router", [None, False, True])
    def test_moe_router_gate_follows_muon_router(self, muon_router):
        """The router gate goes to Muon only with muon_router; the experts go to Muon always."""
        model = self._model(num_experts=4, moe_top_k=2, moe_router="sigmoid_topk")
        moe = model.layers["0"].mlp
        options = {} if muon_router is None else {"muon_router": muon_router}
        muon, adam = _split(build_optimizer(model, OptimizerConfig(name="muon", **options)))
        on_muon = bool(muon_router)
        assert (moe.router.gate.weight in muon) is on_muon
        assert (moe.router.gate.weight in adam) is not on_muon
        assert moe.router.expert_bias in adam
        assert {expert.up_proj.weight for expert in moe.experts} <= muon

    @pytest.mark.parametrize("muon_router", [False, True])
    def test_moma_router_gate_follows_muon_router(self, muon_router):
        """MoMa's expert-choice router gate follows muon_router too."""
        moe = ExpertChoiceMoE(dim=32, hidden_dim=64, num_experts=4, capacity_factor=1.0)
        config = OptimizerConfig(name="muon", muon_router=muon_router)
        muon, adam = _split(build_optimizer(moe, config))
        assert (moe.router.gate.weight in muon) is muon_router
        assert (moe.router.gate.weight in adam) is not muon_router

    def test_packed_experts_use_adam(self):
        """Packed 3-D expert weights go to AdamW."""
        model = self._model(num_experts=4, moe_top_k=2, moe_packed_experts=True)
        moe = model.layers["0"].mlp
        _, adam = _split(build_optimizer(model, OptimizerConfig(name="muon")))
        assert {moe.up_w, moe.down_w, moe.gate_w} <= adam


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


class TestWeightDecayScope:
    """`optimizer.weight_decay_scope` selects which parameters receive weight decay."""

    @staticmethod
    def _model() -> nn.Module:
        """A model covering every shape the rule distinguishes.

        ``scale`` is 0-D and ``proj_bias`` is 2-D but named like a bias, so the two
        halves of the default rule are each exercised on their own.
        """
        model = nn.Module()
        model.embedding = nn.Embedding(16, 8)
        model.linear = nn.Linear(8, 8, bias=True)
        model.norm = nn.LayerNorm(8)
        model.scale = nn.Parameter(torch.ones(()))
        model.proj_bias = nn.Parameter(torch.zeros(4, 4))
        return model

    def _groups(self, scope: str, name: str = "adamw"):
        model = self._model()
        config = OptimizerConfig(name=name, lr=0.01, weight_decay=0.1, weight_decay_scope=scope)
        optimizer = build_optimizer(model, config)
        named = {id(p): n for n, p in model.named_parameters()}
        decay = {named[id(p)] for p in optimizer.param_groups[0]["params"]}
        no_decay = {named[id(p)] for p in optimizer.param_groups[1]["params"]}
        return decay, no_decay

    def test_default_scope_is_backward_compatible(self):
        assert OptimizerConfig(lr=0.01).weight_decay_scope == "exclude_norms_and_biases"

    def test_default_excludes_norms_biases_and_scalars(self):
        decay, no_decay = self._groups("exclude_norms_and_biases")
        assert decay == {"embedding.weight", "linear.weight"}
        assert no_decay == {"linear.bias", "norm.weight", "norm.bias", "scale", "proj_bias"}

    def test_all_decays_every_parameter(self):
        decay, no_decay = self._groups("all")
        assert decay == {
            "embedding.weight",
            "linear.weight",
            "linear.bias",
            "norm.weight",
            "norm.bias",
            "scale",
            "proj_bias",
        }
        assert no_decay == set()

    def test_group_weight_decay_values_are_unchanged(self):
        model = self._model()
        config = OptimizerConfig(lr=0.01, weight_decay=0.1, weight_decay_scope="all")
        optimizer = build_optimizer(model, config)
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    @pytest.mark.parametrize("name", ["adamw", "muon", "lion", "schedule_free_adamw"])
    @pytest.mark.parametrize(
        ("scope", "shrinks"), [("exclude_norms_and_biases", False), ("all", True)]
    )
    def test_scope_reaches_norm_weights_in_every_optimizer(self, name, scope, shrinks):
        """A 1-D norm weight is decayed only under "all", whichever optimizer runs it.

        Asserted through the parameter itself rather than through ``param_groups``,
        because Muon re-splits the groups and keeps its AdamW fallback separately.
        """
        torch.manual_seed(0)
        model = self._model()
        config = OptimizerConfig(name=name, lr=0.1, weight_decay=0.5, weight_decay_scope=scope)
        optimizer = build_optimizer(model, config)
        before = model.norm.weight.detach().clone()
        for param in model.parameters():
            param.grad = torch.zeros_like(param)
        optimizer.step()
        after = model.norm.weight.detach()
        if shrinks:
            assert (after.abs() < before.abs()).all()
        else:
            assert torch.allclose(after, before)

    def test_frozen_parameters_stay_out_of_every_group(self):
        """Under "all" every trained parameter is decayed, not every parameter."""
        model = self._model()
        model.norm.weight.requires_grad_(False)
        config = OptimizerConfig(lr=0.01, weight_decay=0.1, weight_decay_scope="all")
        optimizer = build_optimizer(model, config)
        held = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert id(model.norm.weight) not in held
        assert id(model.linear.weight) in held

    def test_a_positional_call_keeps_its_meaning(self):
        """Fields that predate the scope keep their positions."""
        config = OptimizerConfig("adamw", 0.001, 0.1, (0.9, 0.95))
        assert (config.lr, config.weight_decay, config.betas) == (0.001, 0.1, (0.9, 0.95))
        assert config.weight_decay_scope == "exclude_norms_and_biases"

    def test_invalid_scope_is_rejected_with_the_valid_options(self):
        with pytest.raises(ValueError, match="weight_decay_scope must be one of"):
            OptimizerConfig(lr=0.01, weight_decay_scope="norms")
