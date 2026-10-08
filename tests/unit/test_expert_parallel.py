"""CPU checks of the EP dispatch / compute step with a single-rank stand-in for the all-to-all.

With one rank every expert is local and ``all_to_all_single`` is the identity, so
``ep_dispatch_and_compute`` must reproduce ``MoEMLP._local_forward`` exactly. Multi-rank
behaviour is covered by ``tests/distributed/test_ep.py``.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.distributed as dist

from kempnerforge.distributed.expert_parallel import ep_dispatch_and_compute
from kempnerforge.model.moe import _HAS_GROUPED_MM, MoEMLP, build_moe

pytestmark = pytest.mark.skipif(not _HAS_GROUPED_MM, reason="torch._grouped_mm not available")

DTYPE = torch.bfloat16  # bf16 takes the grouped path
DIM, HIDDEN, NUM_EXPERTS, TOP_K = 64, 128, 4, 2


def _identity_all_to_all(
    output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False
):
    output.copy_(input)


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setattr(dist, "all_to_all_single", _identity_all_to_all)


def _moe(packed: bool, **kwargs) -> MoEMLP:
    torch.manual_seed(0)
    moe = build_moe(
        dim=DIM,
        hidden_dim=HIDDEN,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        packed_experts=packed,
        **kwargs,
    )
    return moe.to(DTYPE)


def _routing(num_tokens: int):
    torch.manual_seed(1)
    x = torch.randn(num_tokens, DIM, dtype=DTYPE)
    indices = torch.rand(num_tokens, NUM_EXPERTS).topk(TOP_K, dim=-1).indices
    weights = torch.softmax(torch.randn(num_tokens, TOP_K), dim=-1).to(DTYPE)
    return x, weights, indices


def _ep(moe, x, weights, indices, gradient_scale=False):
    return ep_dispatch_and_compute(
        x,
        weights,
        indices,
        moe,
        None,  # type: ignore[arg-type]  # no process group behind the identity all-to-all
        local_expert_start=0,
        num_local_experts=NUM_EXPERTS,
        ep_world_size=1,
        gradient_scale=gradient_scale,
    )


def _expert_params(moe: MoEMLP) -> list[torch.nn.Parameter]:
    if moe.packed_experts:
        return [moe.up_w, moe.down_w, moe.gate_w]
    return list(moe.experts.parameters())


class TestSingleRankDispatch:
    @pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
    def test_matches_local_forward(self, single_rank, packed):
        moe = _moe(packed).eval()
        x, weights, indices = _routing(48)
        with torch.no_grad():
            out = _ep(moe, x, weights, indices)
            expected = moe._local_forward(x, weights, indices)
        assert torch.equal(out, expected)

    def test_gradient_scale_matches_local_forward(self, single_rank):
        moe = _moe(False, gradient_scale=True).train()
        x, weights, indices = _routing(48)
        x.requires_grad_(True)
        out = _ep(moe, x, weights, indices, gradient_scale=True)
        assert torch.equal(out, moe._local_forward(x, weights, indices))

        out.float().pow(2).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for p in _expert_params(moe):
            assert p.grad is not None

    @pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
    def test_no_tokens_keeps_expert_params_in_graph(self, single_rank, packed):
        """A rank that receives nothing must still produce a gradient for every expert."""
        moe = _moe(packed).train()
        x, weights, indices = _routing(0)
        x.requires_grad_(True)
        out = _ep(moe, x, weights, indices, gradient_scale=True)
        assert out.shape == (0, DIM)

        out.sum().backward()
        for p in _expert_params(moe):
            assert p.grad is not None
            assert torch.count_nonzero(p.grad) == 0

    def test_moe_forward_takes_ep_branch(self, single_rank):
        """``MoEMLP.forward`` routes through EP once ``ep_world_size > 1``."""
        moe = _moe(False).eval()
        reference = copy.deepcopy(moe)
        moe.ep_world_size = 2  # every expert is still local, so rank 1 gets nothing
        moe.num_local_experts = NUM_EXPERTS
        moe.local_expert_start = 0

        torch.manual_seed(2)
        x = torch.randn(2, 16, DIM, dtype=DTYPE)
        assert torch.equal(moe(x), reference(x))
