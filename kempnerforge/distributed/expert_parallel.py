"""Expert Parallelism: partition MoE experts across an EP process group.

Each EP rank holds ``num_experts // ep_size`` experts. Tokens are shuffled
between ranks via all-to-all so every token reaches its assigned expert,
then results are returned to the originating rank.

When ``ep=1`` (default), this module is a no-op and the model runs with
all experts replicated on every rank (the pre-EP behavior).
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from kempnerforge.model.moe import MoEMLP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Differentiable all-to-all
# ---------------------------------------------------------------------------


class _AllToAll(torch.autograd.Function):
    """Differentiable wrapper around ``dist.all_to_all_single``.

    Forward sends tokens to the correct EP rank; backward reverses the
    communication (same all-to-all, swapped split/gather sizes).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        # Save for backward: reverse the all-to-all direction
        # backward receives what forward sent, sends what forward received
        ctx.bwd_output_splits = input_split_sizes
        ctx.bwd_input_splits = output_split_sizes
        ctx.group = group
        x = x.contiguous()
        out = torch.empty(
            sum(output_split_sizes), *x.shape[1:], dtype=x.dtype, device=x.device
        )
        dist.all_to_all_single(
            out, x,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        grad_input = torch.empty(
            sum(ctx.bwd_output_splits),
            *grad_output.shape[1:],
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.all_to_all_single(
            grad_input, grad_output,
            output_split_sizes=ctx.bwd_output_splits,
            input_split_sizes=ctx.bwd_input_splits,
            group=ctx.group,
        )
        return grad_input, None, None, None


def _all_to_all(
    x: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    return _AllToAll.apply(x, output_split_sizes, input_split_sizes, group)


# ---------------------------------------------------------------------------
# EP dispatch / combine
# ---------------------------------------------------------------------------


def ep_dispatch_and_compute(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    experts: torch.nn.ModuleList,
    ep_group: dist.ProcessGroup,
    local_expert_start: int,
    num_local_experts: int,
    ep_world_size: int,
) -> torch.Tensor:
    """All-to-all dispatch, local expert compute, all-to-all combine.

    Args:
        x: (num_tokens, dim) flattened token representations.
        weights: (num_tokens, top_k) routing weights.
        indices: (num_tokens, top_k) global expert indices.
        experts: Local expert ModuleList (length = num_local_experts).
        ep_group: EP process group.
        local_expert_start: First global expert index on this rank.
        num_local_experts: Number of experts on this rank.
        ep_world_size: Number of ranks in the EP group.

    Returns:
        output: (num_tokens, dim) weighted combination of expert outputs.
    """
    num_tokens, top_k = indices.shape
    dim = x.shape[-1]

    # --- 1. Build per-token, per-expert-selection dispatch info ---
    # Expand tokens: each (token, k) pair becomes one dispatch entry
    flat_indices = indices.reshape(-1)  # (num_tokens * top_k,)
    flat_weights = weights.reshape(-1)  # (num_tokens * top_k,)
    # Token id for each entry
    token_ids = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, top_k).reshape(-1)

    # Which EP rank owns each expert
    target_rank = flat_indices // num_local_experts  # (num_tokens * top_k,)

    # --- 2. Sort entries by target rank (stable sort preserves order within rank) ---
    sort_indices = torch.argsort(target_rank, stable=True)
    sorted_token_ids = token_ids[sort_indices]
    sorted_flat_indices = flat_indices[sort_indices]
    sorted_flat_weights = flat_weights[sort_indices]

    # Gather token features in send order
    x_sorted = x[sorted_token_ids]  # (num_tokens * top_k, dim)

    # Count how many entries go to each rank
    sorted_target_rank = target_rank[sort_indices]
    send_counts = torch.bincount(sorted_target_rank, minlength=ep_world_size)
    send_counts_list = send_counts.tolist()

    # --- 3. Exchange counts so each rank knows what it will receive ---
    recv_counts = torch.zeros_like(send_counts)
    dist.all_to_all_single(
        recv_counts, send_counts, group=ep_group
    )
    recv_counts_list = recv_counts.tolist()

    # --- 4. All-to-all: send tokens to expert-owning ranks ---
    received_tokens = _all_to_all(x_sorted, recv_counts_list, send_counts_list, ep_group)

    # Also exchange expert indices (as float for autograd compatibility, detached)
    sorted_expert_ids_float = sorted_flat_indices.float().unsqueeze(-1)
    received_expert_ids = torch.empty(
        sum(recv_counts_list), 1, dtype=torch.float32, device=x.device
    )
    dist.all_to_all_single(
        received_expert_ids, sorted_expert_ids_float,
        output_split_sizes=recv_counts_list,
        input_split_sizes=send_counts_list,
        group=ep_group,
    )
    received_expert_ids = received_expert_ids.squeeze(-1).long()

    # --- 5. Local expert computation ---
    # Sort received tokens by expert for grouped GEMM.
    from kempnerforge.model.moe import (
        _GROUPED_MM_DTYPES,
        _HAS_GROUPED_MM,
        grouped_expert_forward,
    )

    use_grouped = _HAS_GROUPED_MM and received_tokens.dtype in _GROUPED_MM_DTYPES
    if use_grouped:
        sort_by_expert = torch.argsort(received_expert_ids, stable=True)
        sorted_recv = received_tokens[sort_by_expert]
        sorted_ids = received_expert_ids[sort_by_expert]

        # Map global expert IDs to local indices for bincount.
        local_ids = sorted_ids - local_expert_start
        tokens_per_expert = torch.bincount(
            local_ids, minlength=num_local_experts
        ).tolist()

        local_output_sorted = grouped_expert_forward(
            sorted_recv, tokens_per_expert, experts,
        )

        # Unsort back to received order.
        unsort_by_expert = torch.argsort(sort_by_expert)
        local_output = local_output_sorted[unsort_by_expert]

        # Ensure all expert params are in the autograd graph for FSDP2.
        # grouped_expert_forward touches all expert weights via stacked matmul,
        # but experts with zero tokens contribute only padding zeros. Add an
        # explicit zero-valued contribution to guarantee AccumulateGrad fires.
        for i in range(num_local_experts):
            if tokens_per_expert[i] == 0:
                for p in experts[i].parameters():
                    local_output = local_output + p.sum() * 0
    else:
        local_output = torch.zeros_like(received_tokens)
        unused_expert_params: list[torch.nn.Parameter] = []
        for i in range(num_local_experts):
            global_expert_id = local_expert_start + i
            mask = received_expert_ids == global_expert_id
            if not mask.any():
                unused_expert_params.extend(experts[i].parameters())
                continue
            local_output[mask] = experts[i](received_tokens[mask])

        # FSDP2 requires every parameter's AccumulateGrad hook to fire during
        # backward for reduce-scatter to complete.
        if unused_expert_params:
            _zero = sum(p.sum() for p in unused_expert_params) * 0
            local_output = local_output + _zero

    # Keep dispatch all-to-all in the autograd graph. When all local experts
    # are unused, local_output has no gradient path to received_tokens, so
    # the dispatch _AllToAll.backward never fires on this rank. Since NCCL
    # matches all-to-all ops by position in the communicator, the missing
    # backward causes a misalignment with the peer EP rank → deadlock.
    # Adding a zero-valued contribution preserves the graph edge without
    # changing any gradient values.
    local_output = local_output + received_tokens.sum() * 0

    # --- 6. All-to-all: return processed tokens to originating ranks ---
    # Reverse the all-to-all (swap send/recv counts)
    returned_tokens = _all_to_all(local_output, send_counts_list, recv_counts_list, ep_group)

    # --- 7. Unsort and weighted combine ---
    # returned_tokens is in the same order as x_sorted (sorted by target rank)
    # Unsort back to original (token_id, k) order
    unsort_indices = torch.argsort(sort_indices)
    returned_unsorted = returned_tokens[unsort_indices]
    weights_unsorted = sorted_flat_weights[unsort_indices]

    # Weighted sum per token
    returned_unsorted = returned_unsorted * weights_unsorted.unsqueeze(-1)
    output = torch.zeros(num_tokens, dim, dtype=x.dtype, device=x.device)
    output.scatter_add_(
        0,
        token_ids[torch.arange(len(token_ids), device=x.device)].unsqueeze(-1).expand_as(
            returned_unsorted
        ),
        returned_unsorted,
    )

    return output


# ---------------------------------------------------------------------------
# Apply EP to model
# ---------------------------------------------------------------------------


def apply_expert_parallel(model: torch.nn.Module, device_mesh: DeviceMesh | None) -> None:
    """Partition MoE experts across the EP dimension of the DeviceMesh.

    For each MoEMLP in the model:
      - Prunes ``experts`` to the local subset for this EP rank
      - Stores EP metadata (group, rank, world_size, local expert range)

    Must be called AFTER tensor parallelism and BEFORE FSDP2.

    No-op when ``ep`` is not in the mesh or has size 1.
    """
    if device_mesh is None:
        return
    if "ep" not in device_mesh.mesh_dim_names:
        return

    ep_mesh = device_mesh["ep"]
    ep_size = ep_mesh.size()
    if ep_size <= 1:
        return

    ep_group = ep_mesh.get_group()
    ep_rank = ep_mesh.get_local_rank()

    applied = 0
    for module in model.modules():
        if not isinstance(module, MoEMLP):
            continue

        num_experts = module.num_experts
        assert num_experts % ep_size == 0, (
            f"num_experts ({num_experts}) must be divisible by ep ({ep_size})"
        )
        experts_per_rank = num_experts // ep_size
        start = ep_rank * experts_per_rank
        end = start + experts_per_rank

        # Prune to local experts only
        local_experts = torch.nn.ModuleList(
            [module.experts[i] for i in range(start, end)]
        )
        module.experts = local_experts

        # Store EP metadata
        module.ep_world_size = ep_size
        module.ep_group = ep_group
        module.local_expert_start = start
        module.num_local_experts = experts_per_rank

        applied += 1

    logger.info(
        f"Applied expert parallelism: ep_size={ep_size}, ep_rank={ep_rank}, "
        f"layers={applied}"
    )
