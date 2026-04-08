"""MoE router implementations for KempnerForge models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kempnerforge.config.registry import registry


class SoftmaxTopKRouter(nn.Module):
    """Mixtral-style softmax top-k router with auxiliary load-balancing loss.

    Each token independently selects top_k experts. Routing weights are
    softmax-normalized, then renormalized over the selected top_k.

    Stores ``aux_loss`` and ``expert_counts`` as side effects of each forward
    call for collection by the training loop and metrics.
    """

    def __init__(self, dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss = torch.tensor(0.0)
        self.expert_counts: torch.Tensor = torch.zeros(num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: (num_tokens, dim) — flattened token representations.

        Returns:
            weights:        (num_tokens, top_k) — renormalized routing weights.
            expert_indices: (num_tokens, top_k) — selected expert indices.
        """
        # x: (num_tokens, dim)
        logits = self.gate(x)  # (num_tokens, num_experts)
        probs = F.softmax(logits, dim=-1)  # (num_tokens, num_experts)

        # Select top-k experts per token
        weights, indices = torch.topk(probs, k=self.top_k, dim=-1)

        # Renormalize weights over selected experts
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Auxiliary load-balancing loss (Switch Transformer style):
        #   L_aux = num_experts * sum_i(f_i * P_i)
        # where f_i = fraction of tokens routed to expert i,
        #       P_i = mean routing probability for expert i.
        num_tokens = x.shape[0]

        # f_i: fraction of tokens assigned to each expert (via one_hot)
        one_hot = F.one_hot(indices, num_classes=self.num_experts).float()  # (T, K, E)
        tokens_per_expert = one_hot.sum(dim=(0, 1))  # (E,)
        f = tokens_per_expert / (num_tokens * self.top_k)

        # P_i: mean routing probability for each expert
        p = probs.mean(dim=0)  # (E,)

        self.aux_loss = self.num_experts * (f * p).sum()
        self.expert_counts = tokens_per_expert.detach()

        return weights, indices


def _build_softmax_topk(dim: int, num_experts: int, top_k: int) -> SoftmaxTopKRouter:
    return SoftmaxTopKRouter(dim, num_experts, top_k)


registry.register("router", "softmax_topk", _build_softmax_topk)
