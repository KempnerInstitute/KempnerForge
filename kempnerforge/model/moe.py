"""Mixture-of-Experts feed-forward layer for KempnerForge models."""

from __future__ import annotations

import torch
import torch.nn as nn

import kempnerforge.model.router  # noqa: F401 — triggers router registration
from kempnerforge.config.registry import registry
from kempnerforge.model.mlp import build_mlp


class MoEMLP(nn.Module):
    """Mixture-of-Experts feed-forward layer.

    Composes a router (from "router" registry) with N expert MLPs (from "mlp"
    registry). Drop-in replacement for dense MLP — same forward signature.

    Stores aux_loss after each forward for collection by the training loop.
    """

    def __init__(
        self,
        router: nn.Module,
        experts: nn.ModuleList,
        shared_expert: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.router = router
        self.experts = experts
        self.shared_expert = shared_expert
        self.num_experts = len(experts)

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.router.aux_loss

    @property
    def expert_counts(self) -> torch.Tensor:
        return self.router.expert_counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass dispatching tokens to experts.

        Args:
            x: (batch, seq_len, dim)

        Returns:
            (batch, seq_len, dim)
        """
        B, L, D = x.shape
        x_flat = x.view(B * L, D)

        # Route tokens → stores aux_loss as side effect
        weights, indices = self.router(x_flat)

        output = torch.zeros_like(x_flat)

        for i in range(self.num_experts):
            # Which tokens chose expert i (in any top_k slot)
            mask = (indices == i).any(dim=-1)  # (num_tokens,)
            if not mask.any():
                continue

            expert_input = x_flat[mask]  # (num_selected, D)
            expert_output = self.experts[i](expert_input)  # (num_selected, D)

            # Weight each token assigned to expert i
            weight_for_i = (weights * (indices == i).float()).sum(dim=-1)  # (num_tokens,)
            output[mask] += weight_for_i[mask].unsqueeze(-1) * expert_output

        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        return output.view(B, L, D)


def build_moe(
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    activation: str = "silu",
    router_type: str = "softmax_topk",
    shared_experts: int = 0,
) -> MoEMLP:
    """Build an MoE layer, composing router + experts from Registry.

    Args:
        dim: Model dimension.
        hidden_dim: Expert FFN hidden dimension.
        num_experts: Number of routed experts.
        top_k: Experts selected per token.
        activation: MLP activation (registry key).
        router_type: Router registry key.
        shared_experts: Number of shared experts (always active).
    """
    router_builder = registry.get("router", router_type)
    router = router_builder(dim, num_experts, top_k)

    experts = nn.ModuleList([
        build_mlp(dim, hidden_dim, activation) for _ in range(num_experts)
    ])

    shared_expert = None
    if shared_experts > 0:
        shared_expert = build_mlp(dim, hidden_dim, activation)

    return MoEMLP(router, experts, shared_expert)
