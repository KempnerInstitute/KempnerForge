"""Tensor parallelism sharding plans for model components.

Applies column-parallel and row-parallel sharding to attention and MLP
projections using PyTorch's DTensor-based tensor parallelism.

Sharding strategy (with SequenceParallel):
  - Token embedding:  Replicated (SequenceParallel norm handles Replicate→Shard(1))
  - Attention Q/K/V:  ColwiseParallel (split heads across TP ranks)
  - Attention O:      RowwiseParallel (gather heads, reduce-scatter to Shard(1))
  - MLP gate/up:      ColwiseParallel (split hidden dim)
  - MLP down:         RowwiseParallel (gather hidden dim, reduce-scatter to Shard(1))
  - Norm layers:      SequenceParallel (operate on sequence-sharded activations)
  - Final norm:       SequenceParallel
  - Output head:      ColwiseParallel (split vocab, gather to Replicate for loss)

SequenceParallel keeps activations sharded along the sequence dimension
between blocks, reducing activation memory at norm layers by 1/tp and
replacing all-reduce with reduce-scatter in RowwiseParallel.

The token embedding stays replicated because RowwiseParallel on nn.Embedding
doesn't correctly redistribute output to Shard(1) — it relabels without
scattering, inflating the global sequence dimension. The first block's
SequenceParallel norm naturally handles the Replicate → Shard(1) transition.

SequenceParallel is disabled when tie_embeddings=True (ColwiseParallel on
the output head imposes incompatible sharding on the shared weight).

Pipeline parallel stages use basic TP (no SequenceParallel) to avoid
DTensors at PP stage boundaries.
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from kempnerforge.model.mlp import SwiGLUMLP

logger = logging.getLogger(__name__)


def _build_block_tp_plan(block, *, sequence_parallel: bool = True) -> dict:
    """Build a TP sharding plan for a single TransformerBlock.

    When sequence_parallel=True, norm layers use SequenceParallel and
    projections use Shard(1) input/output layouts so activations stay
    sequence-sharded between blocks. When False, activations are
    Replicate between blocks (basic TP).
    """
    col_kw = {"input_layouts": Shard(1)} if sequence_parallel else {}
    row_kw = {"output_layouts": Shard(1)} if sequence_parallel else {}

    plan = {}

    if sequence_parallel:
        plan["attention_norm"] = SequenceParallel()
        plan["mlp_norm"] = SequenceParallel()

    plan["attention.q_proj"] = ColwiseParallel(**col_kw)
    plan["attention.k_proj"] = ColwiseParallel(**col_kw)
    plan["attention.v_proj"] = ColwiseParallel(**col_kw)
    plan["attention.o_proj"] = RowwiseParallel(**row_kw)

    if isinstance(block.mlp, SwiGLUMLP):
        plan["mlp.gate_proj"] = ColwiseParallel(**col_kw)
        plan["mlp.up_proj"] = ColwiseParallel(**col_kw)
    else:
        plan["mlp.up_proj"] = ColwiseParallel(**col_kw)
    plan["mlp.down_proj"] = RowwiseParallel(**row_kw)

    return plan


def get_tp_mesh(device_mesh: DeviceMesh) -> DeviceMesh | None:
    """Extract the TP sub-mesh from a DeviceMesh.

    Returns None if no 'tp' dimension exists.
    """
    if "tp" not in device_mesh.mesh_dim_names:
        return None
    return device_mesh["tp"]


def apply_tensor_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    """Apply tensor parallelism to a Transformer or PipelineStageModule.

    Parallelizes attention/MLP projections, norm layers (SequenceParallel),
    and output head. Token embedding stays replicated. Should be called
    BEFORE apply_ac and apply_fsdp2.

    PP stages use basic TP without SequenceParallel to avoid DTensors at
    stage boundaries. SequenceParallel is also disabled when weights are
    tied.

    Args:
        model: Transformer or PipelineStageModule to parallelize.
        device_mesh: Full DeviceMesh with a 'tp' dimension.
    """
    tp_mesh = get_tp_mesh(device_mesh)
    if tp_mesh is None:
        logger.warning("No 'tp' dimension in DeviceMesh — skipping tensor parallelism")
        return

    tie = getattr(getattr(model, "config", None), "tie_embeddings", False)
    is_pp_stage = hasattr(model, "stage_id")
    # SequenceParallel requires TP on embedding/head (otherwise the unsharded
    # output head can't consume Shard(1) input from the final norm).
    seq_parallel = not is_pp_stage and not tie

    # Token embedding: wrap output as DTensor Replicate so the first block's
    # SequenceParallel norm properly redistributes to Shard(1). Without this,
    # SequenceParallel receives a plain tensor and labels it Shard(1) without
    # actually scattering, inflating the global sequence dimension.
    if seq_parallel and getattr(model, "token_embedding", None) is not None:
        from torch.distributed.tensor import DTensor

        def _wrap_replicate(module, input, output, mesh=tp_mesh):
            return DTensor.from_local(output, device_mesh=mesh, placements=[Replicate()])

        model.token_embedding.register_forward_hook(_wrap_replicate)

    # Transformer blocks
    for block in model.layers.values():
        plan = _build_block_tp_plan(block, sequence_parallel=seq_parallel)
        parallelize_module(block, tp_mesh, plan)

        # Re-wrap attention/MLP outputs as DTensor Shard(1). Operations inside
        # attention (SDPA, view, contiguous) strip DTensor metadata, causing
        # "mixed torch.Tensor and DTensor" errors at the residual connection.
        if seq_parallel:
            from torch.distributed.tensor import DTensor

            def _rewrap_shard1(module, input, output, mesh=tp_mesh):
                if not isinstance(output, DTensor):
                    return DTensor.from_local(output, device_mesh=mesh, placements=[Shard(1)])
                return output

            block.attention.register_forward_hook(_rewrap_shard1)
            block.mlp.register_forward_hook(_rewrap_shard1)

    # Final norm: SequenceParallel (non-PP, non-tied only)
    if seq_parallel and getattr(model, "norm", None) is not None:
        parallelize_module(model, tp_mesh, {"norm": SequenceParallel()})

    # Output head: split vocab dim, gather to Replicate for loss computation.
    # Only when seq_parallel=True — matches the Shard(1) data flow from the final norm.
    if seq_parallel and not tie and getattr(model, "output_head", None) is not None:
        parallelize_module(
            model.output_head,
            tp_mesh,
            {"proj": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate())},
        )

    logger.info(
        f"Applied tensor parallelism: tp_degree={tp_mesh.size()}, "
        f"layers={len(model.layers)}, sequence_parallel={seq_parallel}"
    )
