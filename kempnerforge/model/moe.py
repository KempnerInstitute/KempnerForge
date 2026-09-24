"""Mixture-of-Experts feed-forward layer for KempnerForge models."""

from __future__ import annotations

from collections.abc import Sequence

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


def _contiguous_grad(grad: torch.Tensor) -> torch.Tensor:
    return grad.contiguous()


def _grouped_mm(x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """``(M, K) @ (E, K, N) -> (M, N)`` with the rows of ``x`` grouped by ``offs``.

    ``offs`` is the int32 inclusive prefix sum of the per-group row counts: group ``i``
    is ``x[offs[i-1]:offs[i]] @ weight[i]``. Empty groups are allowed. One ragged
    ``torch._grouped_mm`` call on any device.
    """
    out = torch._grouped_mm(x, weight, offs=offs)
    if out.requires_grad and not torch.compiler.is_compiling():
        # The kernel's backward rejects expanded (stride-0) gradients, e.g. from
        # ``out.sum().backward()``; materialise them before they reach it.
        out.register_hook(_contiguous_grad)
    return out


def _group_offsets(
    x_sorted: torch.Tensor, tokens_per_expert: torch.Tensor | Sequence[int]
) -> torch.Tensor:
    counts = torch.as_tensor(tokens_per_expert, dtype=torch.int32, device=x_sorted.device)
    return torch.cumsum(counts, 0, dtype=torch.int32)


def grouped_expert_forward(
    x_sorted: torch.Tensor,
    tokens_per_expert: torch.Tensor | Sequence[int],
    experts: nn.ModuleList,
) -> torch.Tensor:
    """Run every expert on its own token group with ragged grouped GEMMs.

    Each expert sees exactly its tokens -- nothing is padded to the busiest expert --
    so memory and time follow the total number of routed tokens, not the imbalance.

    Args:
        x_sorted: (total_tokens, dim) token features sorted by expert index.
        tokens_per_expert: (E,) token count per expert, in expert order.
        experts: Expert modules whose weights are stacked for the grouped GEMM.

    Returns:
        (total_tokens, dim) expert outputs in the same sorted order as the input.
    """
    if x_sorted.shape[0] == 0:
        return torch.zeros_like(x_sorted)
    offs = _group_offsets(x_sorted, tokens_per_expert)
    # nn.Linear stores weight as (out, in); the grouped GEMM wants (E, in, out).
    up_w = torch.stack([e.up_proj.weight.t() for e in experts])  # type: ignore[reportCallIssue, reportAttributeAccessIssue]  # (E, dim, H)
    down_w = torch.stack([e.down_proj.weight.t() for e in experts])  # type: ignore[reportCallIssue, reportAttributeAccessIssue]  # (E, H, dim)
    if hasattr(experts[0], "gate_proj"):
        gate_w = torch.stack([e.gate_proj.weight.t() for e in experts])  # type: ignore[reportCallIssue, reportAttributeAccessIssue]  # (E, dim, H)
        hidden = F.silu(_grouped_mm(x_sorted, gate_w, offs)) * _grouped_mm(x_sorted, up_w, offs)
    else:
        act_fn = experts[0]._activation
        hidden = act_fn(_grouped_mm(x_sorted, up_w, offs))  # type: ignore[reportCallIssue]
    return _grouped_mm(hidden, down_w, offs)


def grouped_expert_forward_packed(
    x_sorted: torch.Tensor,
    tokens_per_expert: torch.Tensor | Sequence[int],
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    gate_w: torch.Tensor | None,
    activation,
) -> torch.Tensor:
    """Same as ``grouped_expert_forward`` over pre-packed ``(E, in, out)`` weights.

    Args:
        x_sorted: (total_tokens, dim) token features sorted by expert index.
        tokens_per_expert: (E,) token count per expert, in expert order.
        up_w: (E, dim, hidden) packed up-projection weights.
        down_w: (E, hidden, dim) packed down-projection weights.
        gate_w: (E, dim, hidden) packed gate weights for SwiGLU, else None.
        activation: Applied to the up-projection when ``gate_w`` is None.

    Returns:
        (total_tokens, dim) expert outputs in the same sorted order as the input.
    """
    if x_sorted.shape[0] == 0:
        return torch.zeros_like(x_sorted)
    offs = _group_offsets(x_sorted, tokens_per_expert)
    if gate_w is not None:
        hidden = F.silu(_grouped_mm(x_sorted, gate_w, offs)) * _grouped_mm(x_sorted, up_w, offs)
    else:
        hidden = activation(_grouped_mm(x_sorted, up_w, offs))
    return _grouped_mm(hidden, down_w, offs)


def scale_by_expert_load(
    expert_out: torch.Tensor, tokens_per_expert: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Rescale each expert's rows (sorted by expert) by ``avg_tokens / its token count``."""
    total = expert_out.shape[0]
    avg_tokens = total / max(num_experts, 1)
    scale = (avg_tokens / tokens_per_expert.clamp(min=1).double()).float()
    rows = scale.repeat_interleave(tokens_per_expert, output_size=total)
    return (expert_out.float() * rows.unsqueeze(-1)).to(expert_out.dtype)


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
        gradient_scale: bool = False,
        packed_experts: bool = False,
    ) -> None:
        super().__init__()
        self.router = router
        self.shared_expert = shared_expert
        self.num_experts = len(experts)
        self.capacity_factor = capacity_factor
        self.gradient_scale = gradient_scale
        self.packed_experts = packed_experts

        # EP attributes — set by apply_expert_parallel(); defaults = no EP
        self.ep_world_size: int = 1
        self.ep_group = None
        self.local_expert_start: int = 0
        self.num_local_experts: int = len(experts)

        if packed_experts:
            # Packed expert weights: stack per-expert (out, in) Linear weights into
            # (E, in, out) tensors so grouped GEMM can consume them zero-copy.
            # Drop the per-expert nn.ModuleList — the packed tensors are the
            # sole source of truth. Tests / EP / FSDP2 read `self.up_w` etc.
            self._is_swiglu = hasattr(experts[0], "gate_proj")
            self.up_w = nn.Parameter(
                torch.stack([e.up_proj.weight.t().contiguous() for e in experts])  # type: ignore[reportCallIssue, reportAttributeAccessIssue]
            )
            self.down_w = nn.Parameter(
                torch.stack([e.down_proj.weight.t().contiguous() for e in experts])  # type: ignore[reportCallIssue, reportAttributeAccessIssue]
            )
            if self._is_swiglu:
                self.gate_w = nn.Parameter(
                    torch.stack([e.gate_proj.weight.t().contiguous() for e in experts])  # type: ignore[reportCallIssue, reportAttributeAccessIssue]
                )
                self._packed_activation = F.silu
            else:
                self._packed_activation = experts[0]._activation
        else:
            self.experts = experts

    def _apply_packed_expert(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Apply packed expert ``i`` to ``x`` without grouped GEMM.

        Used by the sequential fallback path. Matches the unpacked
        SwiGLU/StandardMLP forward exactly (no bias, same matmul order).
        """
        up = x @ self.up_w[i]
        if self._is_swiglu:
            gate = x @ self.gate_w[i]
            hidden = F.silu(gate) * up
        else:
            hidden = self._packed_activation(up)  # type: ignore[reportCallIssue]
        return hidden @ self.down_w[i]

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
            flat_indices = indices.reshape(-1)  # (T*K,)
            flat_weights = weights.reshape(-1)  # (T*K,)
            token_ids = (
                torch.arange(num_tokens, device=x_flat.device)
                .unsqueeze(1)
                .expand(-1, top_k)
                .reshape(-1)
            )

            sort_order = torch.argsort(flat_indices, stable=True)
            sorted_expert_ids = flat_indices[sort_order]
            sorted_token_ids = token_ids[sort_order]
            sorted_weights = flat_weights[sort_order]

            x_sorted = x_flat[sorted_token_ids]
            tokens_per_expert = torch.bincount(sorted_expert_ids, minlength=self.num_experts)

            if self.packed_experts:
                expert_out = grouped_expert_forward_packed(
                    x_sorted,
                    tokens_per_expert,
                    self.up_w,
                    self.down_w,
                    self.gate_w if self._is_swiglu else None,
                    self._packed_activation,
                )
            else:
                expert_out = grouped_expert_forward(x_sorted, tokens_per_expert, self.experts)

            # Per-expert gradient scaling: normalize by utilization ratio so
            # high-traffic experts don't dominate learning (DeepSeek-V3 Sec 3.2).
            if self.gradient_scale and self.training:
                expert_out = scale_by_expert_load(expert_out, tokens_per_expert, self.num_experts)

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
            # Precompute average tokens for gradient scaling
            if self.gradient_scale and self.training:
                avg_tokens = num_tokens * top_k / max(self.num_experts, 1)
            for i in range(self.num_experts):
                mask = (indices == i).any(dim=-1)
                if not mask.any():
                    continue
                expert_input = x_flat[mask]
                expert_output = (
                    self._apply_packed_expert(expert_input, i)
                    if self.packed_experts
                    else self.experts[i](expert_input)
                )
                # Per-expert gradient scaling (DeepSeek-V3 Sec 3.2)
                if self.gradient_scale and self.training:
                    tokens_i = (indices == i).sum().detach().float()
                    scale = avg_tokens / tokens_i.clamp(min=1.0)
                    expert_output = expert_output * scale
                weight_for_i = (weights * (indices == i).float()).sum(dim=-1)
                output[mask] += weight_for_i[mask].unsqueeze(-1) * expert_output

        return output

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.router.aux_loss  # type: ignore[reportReturnType]

    @property
    def z_loss(self) -> torch.Tensor:
        return self.router.z_loss  # type: ignore[reportReturnType]

    @property
    def expert_counts(self) -> torch.Tensor:
        return self.router.expert_counts  # type: ignore[reportReturnType]

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
                weights,
                indices,
                self.num_experts,
                self.capacity_factor,
            )

        if self.ep_world_size > 1:
            from kempnerforge.distributed.expert_parallel import ep_dispatch_and_compute

            output = ep_dispatch_and_compute(
                x_flat,
                weights,
                indices,
                self,
                self.ep_group,  # type: ignore[reportArgumentType]
                self.local_expert_start,
                self.num_local_experts,
                self.ep_world_size,
                gradient_scale=self.gradient_scale,
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
    gradient_scale: bool = False,
    sequence_aux_loss_weight: float = 0.0,
    bias_schedule: str = "constant",
    packed_experts: bool = False,
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
        gradient_scale: Per-expert gradient normalization.
        sequence_aux_loss_weight: Sequence-level balance loss weight (sigmoid router only).
        bias_schedule: Bias update rate schedule (sigmoid router only).
        packed_experts: Pack expert weights into one tensor per projection.
    """
    router_builder = registry.get("router", router_type)
    router_kwargs: dict[str, object] = {}
    if router_type == "sigmoid_topk":
        router_kwargs = {
            "sequence_aux_loss_weight": sequence_aux_loss_weight,
            "bias_schedule": bias_schedule,
        }
    router = router_builder(dim, num_experts, top_k, **router_kwargs)

    experts = nn.ModuleList([build_mlp(dim, hidden_dim, activation) for _ in range(num_experts)])

    shared_expert = None
    if shared_experts > 0:
        shared_expert = build_mlp(dim, hidden_dim, activation)

    return MoEMLP(
        router,
        experts,
        shared_expert,
        capacity_factor=capacity_factor,
        gradient_scale=gradient_scale,
        packed_experts=packed_experts,
    )
