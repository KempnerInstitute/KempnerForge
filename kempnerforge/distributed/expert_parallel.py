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
        ctx.bwd_output_splits = input_split_sizes
        ctx.bwd_input_splits = output_split_sizes
        ctx.group = group
        x = x.contiguous()
        out = torch.empty(sum(output_split_sizes), *x.shape[1:], dtype=x.dtype, device=x.device)
        dist.all_to_all_single(
            out,
            x,
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
            grad_input,
            grad_output,
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


class _ScaleGrad(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by ``scale`` in the backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def ep_dispatch_and_compute(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    moe: MoEMLP,
    ep_group: dist.ProcessGroup,
    local_expert_start: int,
    num_local_experts: int,
    ep_world_size: int,
    gradient_scale: bool = False,
) -> torch.Tensor:
    """All-to-all dispatch, local expert compute, all-to-all combine.

    When the EP ranks hold the same batch, as they do in training, the backward pass
    gives the local experts the mean of the ranks' gradients, which is one process's,
    and gives ``x`` and ``weights`` the same gradients as without EP.

    Args:
        x: (num_tokens, dim) flattened token representations.
        weights: (num_tokens, top_k) routing weights.
        indices: (num_tokens, top_k) global expert indices.
        moe: The MoEMLP module. Holds either an nn.ModuleList of local experts
            (unpacked) or packed expert tensors already sliced to the local range.
        ep_group: EP process group.
        local_expert_start: First global expert index on this rank.
        num_local_experts: Number of experts on this rank.
        ep_world_size: Number of ranks in the EP group.
        gradient_scale: When ``moe`` is in training mode, scale each local expert's
            output by the average expert load over its own load, as
            ``MoEMLP._local_forward`` does.

    Returns:
        output: (num_tokens, dim) weighted combination of expert outputs.
    """
    num_tokens, top_k = indices.shape
    dim = x.shape[-1]

    flat_indices = indices.reshape(-1)
    flat_weights = weights.reshape(-1)
    token_ids = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, top_k).reshape(-1)

    target_rank = flat_indices // num_local_experts

    sort_indices = torch.argsort(target_rank, stable=True)
    sorted_token_ids = token_ids[sort_indices]
    sorted_flat_indices = flat_indices[sort_indices]
    sorted_flat_weights = flat_weights[sort_indices]

    x_sorted = x[sorted_token_ids]

    sorted_target_rank = target_rank[sort_indices]
    send_counts = torch.bincount(sorted_target_rank, minlength=ep_world_size)
    send_counts_list = send_counts.tolist()

    recv_counts = torch.zeros_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    recv_counts_list = recv_counts.tolist()

    received_tokens = _all_to_all(x_sorted, recv_counts_list, send_counts_list, ep_group)
    received_tokens = _ScaleGrad.apply(received_tokens, float(ep_world_size))

    sorted_expert_ids_float = sorted_flat_indices.float().unsqueeze(-1)
    received_expert_ids = torch.empty(
        sum(recv_counts_list), 1, dtype=torch.float32, device=x.device
    )
    dist.all_to_all_single(
        received_expert_ids,
        sorted_expert_ids_float,
        output_split_sizes=recv_counts_list,
        input_split_sizes=send_counts_list,
        group=ep_group,
    )
    received_expert_ids = received_expert_ids.squeeze(-1).long()

    from kempnerforge.model.moe import (
        _GROUPED_MM_DTYPES,
        _HAS_GROUPED_MM,
        grouped_expert_forward,
        grouped_expert_forward_packed,
    )

    use_grouped = _HAS_GROUPED_MM and received_tokens.dtype in _GROUPED_MM_DTYPES
    if use_grouped:
        sort_by_expert = torch.argsort(received_expert_ids, stable=True)
        sorted_recv = received_tokens[sort_by_expert]
        sorted_ids = received_expert_ids[sort_by_expert]

        local_ids = sorted_ids - local_expert_start
        tokens_per_expert = torch.bincount(local_ids, minlength=num_local_experts)

        if moe.packed_experts:
            local_output_sorted = grouped_expert_forward_packed(
                sorted_recv,
                tokens_per_expert,
                moe.up_w,
                moe.down_w,
                moe.gate_w if moe._is_swiglu else None,
                moe._packed_activation,
            )
        else:
            local_output_sorted = grouped_expert_forward(
                sorted_recv,
                tokens_per_expert,
                moe.experts,
            )

        unsort_by_expert = torch.argsort(sort_by_expert)
        local_output = local_output_sorted[unsort_by_expert]

        if sorted_recv.shape[0] == 0:
            params = (
                [moe.up_w, moe.down_w] + ([moe.gate_w] if moe._is_swiglu else [])
                if moe.packed_experts
                else list(moe.experts.parameters())
            )
            local_output = local_output + sum(p.sum() for p in params) * 0
    else:
        local_output = torch.zeros_like(received_tokens)
        if moe.packed_experts:
            any_computed = False
            for i in range(num_local_experts):
                global_expert_id = local_expert_start + i
                mask = received_expert_ids == global_expert_id
                if not mask.any():
                    continue
                local_output[mask] = moe._apply_packed_expert(received_tokens[mask], i)
                any_computed = True

            if not any_computed:
                _zero = moe.up_w.sum() * 0 + moe.down_w.sum() * 0
                if moe._is_swiglu:
                    _zero = _zero + moe.gate_w.sum() * 0
                local_output = local_output + _zero
        else:
            unused_expert_params: list[torch.nn.Parameter] = []
            for i in range(num_local_experts):
                global_expert_id = local_expert_start + i
                mask = received_expert_ids == global_expert_id
                if not mask.any():
                    unused_expert_params.extend(moe.experts[i].parameters())
                    continue
                local_output[mask] = moe.experts[i](received_tokens[mask])

            if unused_expert_params:
                _zero = sum(p.sum() for p in unused_expert_params) * 0
                local_output = local_output + _zero

    if gradient_scale and moe.training:
        total_recv = sum(recv_counts_list)
        if total_recv > 0:
            local_ids = received_expert_ids - local_expert_start
            tpe = torch.bincount(local_ids, minlength=num_local_experts)
            avg_tokens = num_tokens * top_k / max(num_local_experts, 1)
            scale = (avg_tokens / tpe.clamp(min=1).double()).float()
            local_output = (local_output.float() * scale[local_ids].unsqueeze(-1)).to(
                local_output.dtype
            )

    local_output = local_output + received_tokens.sum() * 0
    local_output = _ScaleGrad.apply(local_output, 1.0 / ep_world_size)

    returned_tokens = _all_to_all(local_output, send_counts_list, recv_counts_list, ep_group)

    unsort_indices = torch.argsort(sort_indices)
    returned_unsorted = returned_tokens[unsort_indices]
    weights_unsorted = sorted_flat_weights[unsort_indices]

    returned_unsorted = returned_unsorted * weights_unsorted.unsqueeze(-1)
    output = torch.zeros(num_tokens, dim, dtype=x.dtype, device=x.device)
    output.scatter_add_(
        0,
        token_ids[torch.arange(len(token_ids), device=x.device)]
        .unsqueeze(-1)
        .expand_as(returned_unsorted),
        returned_unsorted,
    )

    return output


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
    if "ep" not in device_mesh.mesh_dim_names:  # type: ignore[reportOperatorIssue]
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

        if module.packed_experts:
            module.up_w = torch.nn.Parameter(module.up_w.data[start:end].clone().contiguous())
            module.down_w = torch.nn.Parameter(module.down_w.data[start:end].clone().contiguous())
            if module._is_swiglu:
                module.gate_w = torch.nn.Parameter(
                    module.gate_w.data[start:end].clone().contiguous()
                )
        else:
            local_experts = torch.nn.ModuleList([module.experts[i] for i in range(start, end)])
            module.experts = local_experts

        module.ep_world_size = ep_size
        module.ep_group = ep_group  # type: ignore[reportAttributeAccessIssue]
        module.local_expert_start = start
        module.num_local_experts = experts_per_rank

        applied += 1

    logger.info(
        f"Applied expert parallelism: ep_size={ep_size}, ep_rank={ep_rank}, layers={applied}"
    )


def expert_parallel_params(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], dist.ProcessGroup | None]:
    """The routed-expert parameters that EP splits across ranks, and the EP group.

    Each EP rank holds its own slice of these and a copy of every other parameter.
    Returns ``([], None)`` when no ``MoEMLP`` has ``ep_world_size > 1``.

    Raises:
        ValueError: Two ``MoEMLP`` layers record different EP groups, so no single
            group covers every routed expert.
    """
    params: list[torch.nn.Parameter] = []
    group: dist.ProcessGroup | None = None
    for module in model.modules():
        if not isinstance(module, MoEMLP) or module.ep_world_size <= 1:
            continue
        if group is not None and module.ep_group is not group:
            raise ValueError(
                "MoE layers record different expert-parallel groups, so no single group "
                "covers every routed expert; apply_expert_parallel gives all layers one group"
            )
        group = module.ep_group
        if module.packed_experts:
            params += [module.up_w, module.down_w]
            if module._is_swiglu:
                params.append(module.gate_w)
        else:
            params += module.experts.parameters()
    return params, group
