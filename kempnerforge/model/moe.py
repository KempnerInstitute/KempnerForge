"""Mixture-of-Experts feed-forward layer for KempnerForge models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import kempnerforge.model.router  # noqa: F401 — triggers router registration
from kempnerforge.config.registry import registry
from kempnerforge.model.mlp import build_mlp

_HAS_GROUPED_MM = hasattr(torch, "_grouped_mm")

# torch._grouped_mm under torch.compile requires bf16/fp16 inputs
# (the meta registration rejects fp32). Guard against this.
_GROUPED_MM_DTYPES = {torch.bfloat16, torch.float16}


def grouped_expert_forward(
    x_sorted: torch.Tensor,
    tokens_per_expert: list[int],
    experts: nn.ModuleList,
) -> torch.Tensor:
    """Batched expert computation using ``torch._grouped_mm``.

    Replaces the sequential expert loop with 2-3 grouped matrix multiplies
    (one CUDA kernel each), giving significant speedup when many experts are
    active.

    Args:
        x_sorted: (total_tokens, dim) token features sorted by expert index.
        tokens_per_expert: Number of tokens assigned to each expert, in order.
        experts: Expert modules whose weights are stacked for the grouped GEMM.

    Returns:
        (total_tokens, dim) expert outputs in the same sorted order as input.
    """
    num_experts = len(experts)
    total_tokens, dim = x_sorted.shape
    max_tokens = max(tokens_per_expert)

    if max_tokens == 0 or total_tokens == 0:
        return torch.zeros_like(x_sorted)

    is_swiglu = hasattr(experts[0], "gate_proj")

    # Stack expert weights into (E, in, out) for grouped matmul.
    # nn.Linear stores weight as (out, in), so transpose to (in, out).
    up_w = torch.stack([e.up_proj.weight.t() for e in experts])      # (E, dim, H)
    down_w = torch.stack([e.down_proj.weight.t() for e in experts])  # (E, H, dim)
    if is_swiglu:
        gate_w = torch.stack([e.gate_proj.weight.t() for e in experts])  # (E, dim, H)

    # Pad token groups into (E, max_tokens, dim) for uniform batch size.
    x_padded = x_sorted.new_zeros(num_experts, max_tokens, dim)
    offset = 0
    for i, count in enumerate(tokens_per_expert):
        if count > 0:
            x_padded[i, :count] = x_sorted[offset:offset + count]
        offset += count

    # Grouped matmuls — 3 for SwiGLU, 2 for StandardMLP.
    if is_swiglu:
        gate = torch._grouped_mm(x_padded, gate_w)   # (E, M, H)
        up = torch._grouped_mm(x_padded, up_w)       # (E, M, H)
        hidden = F.silu(gate) * up                     # (E, M, H)
    else:
        hidden = torch._grouped_mm(x_padded, up_w)    # (E, M, H)
        act_fn = experts[0]._activation
        hidden = act_fn(hidden)

    out_padded = torch._grouped_mm(hidden, down_w)     # (E, M, dim)

    # Unpad back to flat sorted order.
    output = torch.zeros_like(x_sorted)
    offset = 0
    for i, count in enumerate(tokens_per_expert):
        if count > 0:
            output[offset:offset + count] = out_padded[i, :count]
        offset += count

    return output


def _apply_capacity(
    weights: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
    capacity_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero routing weights for tokens that exceed per-expert capacity.

    Capacity = ceil(num_tokens * top_k / num_experts * capacity_factor).
    For each expert, only the first ``capacity`` tokens (in sequence order)
    are kept; the rest get weight=0 and are effectively dropped.

    Args:
        weights: (num_tokens, top_k) routing weights.
        indices: (num_tokens, top_k) expert indices.
        num_experts: Total number of experts.
        capacity_factor: Multiplier for capacity (1.0 = exact average, 1.25 = 25% headroom).

    Returns:
        (weights, indices) with overflow entries zeroed out. Tensors are cloned
        to avoid mutating the router's output.
    """
    import math

    num_tokens, top_k = indices.shape
    capacity = max(1, math.ceil(num_tokens * top_k / num_experts * capacity_factor))

    weights = weights.clone()
    for k in range(top_k):
        # Count per-expert assignments in this top_k slot.
        for e in range(num_experts):
            assigned = (indices[:, k] == e).nonzero(as_tuple=True)[0]
            if assigned.numel() <= capacity:
                continue
            drop = assigned[capacity:]
            weights[drop, k] = 0.0

    return weights, indices


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
        capacity_factor: float = 0.0,
    ) -> None:
        super().__init__()
        self.router = router
        self.experts = experts
        self.shared_expert = shared_expert
        self.num_experts = len(experts)
        self.capacity_factor = capacity_factor

        # EP attributes — set by apply_expert_parallel(); defaults = no EP
        self.ep_world_size: int = 1
        self.ep_group = None
        self.local_expert_start: int = 0
        self.num_local_experts: int = len(experts)

    def _local_forward(
        self,
        x_flat: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to local experts and weighted-combine results.

        Uses grouped GEMM when available (torch._grouped_mm) for batched expert
        computation. Falls back to sequential loop otherwise.
        """
        num_tokens, dim = x_flat.shape
        top_k = indices.shape[1]

        use_grouped = _HAS_GROUPED_MM and x_flat.dtype in _GROUPED_MM_DTYPES
        if use_grouped:
            # Expand (token, k) pairs → flat entries sorted by expert.
            flat_indices = indices.reshape(-1)           # (T*K,)
            flat_weights = weights.reshape(-1)           # (T*K,)
            token_ids = (
                torch.arange(num_tokens, device=x_flat.device)
                .unsqueeze(1).expand(-1, top_k).reshape(-1)
            )

            sort_order = torch.argsort(flat_indices, stable=True)
            sorted_expert_ids = flat_indices[sort_order]
            sorted_token_ids = token_ids[sort_order]
            sorted_weights = flat_weights[sort_order]

            x_sorted = x_flat[sorted_token_ids]
            tokens_per_expert = torch.bincount(
                sorted_expert_ids, minlength=self.num_experts
            ).tolist()

            expert_out = grouped_expert_forward(x_sorted, tokens_per_expert, self.experts)

            # Weighted scatter-add back to output.
            expert_out = expert_out * sorted_weights.unsqueeze(-1)
            output = torch.zeros(num_tokens, dim, dtype=x_flat.dtype, device=x_flat.device)
            output.scatter_add_(
                0,
                sorted_token_ids.unsqueeze(-1).expand_as(expert_out),
                expert_out,
            )
        else:
            output = torch.zeros_like(x_flat)
            for i in range(self.num_experts):
                mask = (indices == i).any(dim=-1)
                if not mask.any():
                    continue
                expert_input = x_flat[mask]
                expert_output = self.experts[i](expert_input)
                weight_for_i = (weights * (indices == i).float()).sum(dim=-1)
                output[mask] += weight_for_i[mask].unsqueeze(-1) * expert_output

        return output

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

        # Capacity factor: cap tokens per expert, drop overflow.
        # Dropped tokens get zero routing weight → no expert contribution,
        # carried through unchanged by the residual connection.
        if self.capacity_factor > 0:
            weights, indices = _apply_capacity(
                weights, indices, self.num_experts, self.capacity_factor,
            )

        if self.ep_world_size > 1:
            from kempnerforge.distributed.expert_parallel import ep_dispatch_and_compute

            output = ep_dispatch_and_compute(
                x_flat, weights, indices,
                self.experts, self.ep_group,
                self.local_expert_start, self.num_local_experts,
                self.ep_world_size,
            )
        else:
            output = self._local_forward(x_flat, weights, indices)

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
    capacity_factor: float = 0.0,
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
        capacity_factor: Token capacity per expert (0=unlimited, >0=cap).
    """
    router_builder = registry.get("router", router_type)
    router = router_builder(dim, num_experts, top_k)

    experts = nn.ModuleList([
        build_mlp(dim, hidden_dim, activation) for _ in range(num_experts)
    ])

    shared_expert = None
    if shared_experts > 0:
        shared_expert = build_mlp(dim, hidden_dim, activation)

    return MoEMLP(router, experts, shared_expert, capacity_factor=capacity_factor)
