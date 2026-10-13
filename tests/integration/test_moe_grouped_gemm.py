"""GPU checks for the ragged grouped-GEMM expert dispatch.

The ragged path must reproduce the padded grouped GEMM it replaced bit for bit, in the
forward and in every gradient, and the full layer must behave the same way on CUDA.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import kempnerforge.model.moe as moe_mod
from kempnerforge.model.mlp import StandardMLP, SwiGLUMLP
from kempnerforge.model.moe import (
    build_moe,
    grouped_expert_forward,
    grouped_expert_forward_packed,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def _padded_reference(x_sorted, tokens_per_expert, experts):
    """The previous dispatch: every expert padded to the busiest expert's token count."""
    counts = torch.as_tensor(tokens_per_expert).tolist()
    num_experts, dim = len(experts), x_sorted.shape[1]
    max_tokens = max(counts)
    if max_tokens == 0 or x_sorted.shape[0] == 0:
        return torch.zeros_like(x_sorted)
    up_w = torch.stack([e.up_proj.weight.t() for e in experts])
    down_w = torch.stack([e.down_proj.weight.t() for e in experts])
    x_padded = x_sorted.new_zeros(num_experts, max_tokens, dim)
    offset = 0
    for i, count in enumerate(counts):
        if count > 0:
            x_padded[i, :count] = x_sorted[offset : offset + count]
        offset += count
    if hasattr(experts[0], "gate_proj"):
        gate_w = torch.stack([e.gate_proj.weight.t() for e in experts])
        hidden = F.silu(torch._grouped_mm(x_padded, gate_w)) * torch._grouped_mm(x_padded, up_w)
    else:
        hidden = experts[0]._activation(torch._grouped_mm(x_padded, up_w))
    out_padded = torch._grouped_mm(hidden, down_w)
    output = torch.zeros_like(x_sorted)
    offset = 0
    for i, count in enumerate(counts):
        if count > 0:
            output[offset : offset + count] = out_padded[i, :count]
        offset += count
    return output


def _experts(kind: str, num_experts: int, dim: int, hidden: int) -> torch.nn.ModuleList:
    make = (
        (lambda: SwiGLUMLP(dim, hidden))
        if kind == "swiglu"
        else (lambda: StandardMLP(dim, hidden, activation="gelu"))
    )
    return torch.nn.ModuleList([make() for _ in range(num_experts)]).to(DEVICE, DTYPE)


def _routing(num_tokens: int, top_k: int, num_experts: int, hot_share: float | None):
    """Sorted per-assignment tokens and counts; ``hot_share`` sends that share to expert 0."""
    idx = torch.randint(0, num_experts, (num_tokens * top_k,), device=DEVICE)
    if hot_share is not None:
        idx[: int(hot_share * idx.numel())] = 0
    idx[idx == num_experts - 1] = num_experts - 2  # keep one expert empty
    order = torch.argsort(idx, stable=True)
    return order, torch.bincount(idx[order], minlength=num_experts)


def _grads(out, x, experts, grad_out):
    params = [p for e in experts for p in e.parameters()]
    return torch.autograd.grad(out, [x, *params], grad_out, retain_graph=True)


class TestRaggedMatchesPadded:
    @pytest.mark.parametrize("kind", ["swiglu", "gelu"])
    @pytest.mark.parametrize(
        "num_experts,top_k,hot_share", [(8, 2, None), (32, 2, 0.6), (64, 8, None), (64, 8, 0.6)]
    )
    def test_forward_and_gradients_bit_identical(self, kind, num_experts, top_k, hot_share):
        torch.manual_seed(0)
        dim, hidden, num_tokens = 128, 256, 2048
        experts = _experts(kind, num_experts, dim, hidden)
        order, counts = _routing(num_tokens, top_k, num_experts, hot_share)
        assert (counts == 0).any(), "the routing must leave one expert empty"

        x = torch.randn(num_tokens, dim, device=DEVICE, dtype=DTYPE, requires_grad=True)
        tok = torch.arange(num_tokens, device=DEVICE).repeat_interleave(top_k)
        x_sorted = x[tok[order]]
        grad_out = torch.randn_like(x_sorted)

        ragged = grouped_expert_forward(x_sorted, counts, experts)
        padded = _padded_reference(x_sorted, counts, experts)
        assert torch.equal(ragged, padded)
        for got, want in zip(
            _grads(ragged, x, experts, grad_out),
            _grads(padded, x, experts, grad_out),
            strict=True,
        ):
            assert torch.equal(got, want)

    def test_packed_matches_unpacked(self):
        torch.manual_seed(0)
        dim, hidden, num_experts = 128, 256, 16
        experts = _experts("swiglu", num_experts, dim, hidden)
        order, counts = _routing(1024, 2, num_experts, None)
        x = torch.randn(1024 * 2, dim, device=DEVICE, dtype=DTYPE)[order]

        up_w = torch.stack([e.up_proj.weight.t().contiguous() for e in experts])
        down_w = torch.stack([e.down_proj.weight.t().contiguous() for e in experts])
        gate_w = torch.stack([e.gate_proj.weight.t().contiguous() for e in experts])
        packed = grouped_expert_forward_packed(x, counts, up_w, down_w, gate_w, F.silu)
        assert torch.equal(packed, grouped_expert_forward(x, counts, experts))

    def test_backward_with_expanded_grad(self):
        """``out.sum().backward()`` hands the kernel a stride-0 gradient; it must still work."""
        experts = _experts("swiglu", 4, 64, 128)
        x = torch.randn(40, 64, device=DEVICE, dtype=DTYPE, requires_grad=True)
        grouped_expert_forward(x, torch.tensor([10, 0, 20, 10]), experts).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_no_padding_memory(self):
        """A single hot expert no longer inflates the activation footprint."""
        experts = _experts("swiglu", 64, 128, 256)
        num_tokens, top_k = 4096, 8
        peaks = []
        for hot_share in (None, 0.9):
            torch.manual_seed(0)
            order, counts = _routing(num_tokens, top_k, 64, hot_share)
            x = torch.randn(num_tokens * top_k, 128, device=DEVICE, dtype=DTYPE)[order]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            grouped_expert_forward(x, counts, experts)
            peaks.append(torch.cuda.max_memory_allocated() - base)
        assert peaks[1] <= 1.05 * peaks[0], f"balanced {peaks[0]} vs skewed {peaks[1]} bytes"


class TestLayerOnCuda:
    def _layer(self, top_k: int, num_experts: int = 8):
        torch.manual_seed(0)
        moe = build_moe(dim=128, hidden_dim=256, num_experts=num_experts, top_k=top_k)
        return moe.to(DEVICE, DTYPE)

    def test_layer_matches_padded_dispatch_top1(self, monkeypatch):
        """With top-1 routing the combine has no additions, so the layer must be bit-identical."""
        moe = self._layer(top_k=1)
        x = torch.randn(4, 256, 128, device=DEVICE, dtype=DTYPE)
        out = moe(x)
        monkeypatch.setattr(moe_mod, "grouped_expert_forward", _padded_reference)
        assert torch.equal(moe(x), out)

    def test_layer_matches_padded_dispatch_topk(self, monkeypatch):
        moe = self._layer(top_k=2)
        x = torch.randn(4, 256, 128, device=DEVICE, dtype=DTYPE)
        out = moe(x)
        monkeypatch.setattr(moe_mod, "grouped_expert_forward", _padded_reference)
        torch.testing.assert_close(moe(x), out, atol=1e-2, rtol=1e-2)

    def test_layer_backward_all_params(self):
        moe = self._layer(top_k=2, num_experts=32)
        x = torch.randn(2, 64, 128, device=DEVICE, dtype=DTYPE)
        moe(x).float().pow(2).mean().backward()
        for name, p in moe.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name

    def test_compile_matches_eager(self):
        moe = self._layer(top_k=2)
        x = torch.randn(2, 128, 128, device=DEVICE, dtype=DTYPE)
        eager = moe(x)
        compiled = torch.compile(moe, fullgraph=False)(x)
        torch.testing.assert_close(compiled, eager, atol=2e-2, rtol=2e-2)
