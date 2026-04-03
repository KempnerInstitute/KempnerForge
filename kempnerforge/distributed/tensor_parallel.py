"""Tensor parallelism sharding plans for model components.

Applies column-parallel and row-parallel sharding to attention and MLP
projections using PyTorch's DTensor-based tensor parallelism.

Sharding strategy:
  - Attention Q/K/V: ColwiseParallel (split heads across TP ranks)
  - Attention O:     RowwiseParallel (gather heads, all-reduce)
  - MLP gate/up:     ColwiseParallel (split hidden dim)
  - MLP down:        RowwiseParallel (gather hidden dim, all-reduce)

Embedding and output head are left replicated. SequenceParallel for norms
can be added as a future optimization.
"""

from __future__ import annotations

import logging

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from kempnerforge.model.mlp import SwiGLUMLP
from kempnerforge.model.transformer import Transformer

logger = logging.getLogger(__name__)


def _build_block_tp_plan(block) -> dict:
    """Build a TP sharding plan for a single TransformerBlock.

    Returns a dict mapping submodule paths to parallel styles,
    suitable for ``parallelize_module()``.
    """
    plan = {
        # Attention: split Q/K/V heads across TP ranks, gather at output
        "attention.q_proj": ColwiseParallel(),
        "attention.k_proj": ColwiseParallel(),
        "attention.v_proj": ColwiseParallel(),
        "attention.o_proj": RowwiseParallel(),
    }

    # MLP: split hidden dim across TP ranks
    if isinstance(block.mlp, SwiGLUMLP):
        # SwiGLU: gate + up are both column-parallel
        plan["mlp.gate_proj"] = ColwiseParallel()
        plan["mlp.up_proj"] = ColwiseParallel()
    else:
        # Standard MLP: only up is column-parallel
        plan["mlp.up_proj"] = ColwiseParallel()
    plan["mlp.down_proj"] = RowwiseParallel()

    return plan


def get_tp_mesh(device_mesh: DeviceMesh) -> DeviceMesh | None:
    """Extract the TP sub-mesh from a DeviceMesh.

    Returns None if no 'tp' dimension exists.
    """
    if "tp" not in device_mesh.mesh_dim_names:
        return None
    return device_mesh["tp"]


def apply_tensor_parallel(model: Transformer, device_mesh: DeviceMesh) -> None:
    """Apply tensor parallelism to a Transformer model.

    Parallelizes attention and MLP projections within each TransformerBlock.
    Should be called BEFORE apply_fsdp2.

    Args:
        model: Transformer model to parallelize.
        device_mesh: Full DeviceMesh (tp sub-mesh is extracted automatically).
            Must contain a 'tp' dimension.
    """
    tp_mesh = get_tp_mesh(device_mesh)
    if tp_mesh is None:
        logger.warning("No 'tp' dimension in DeviceMesh — skipping tensor parallelism")
        return

    for block in model.layers.values():
        plan = _build_block_tp_plan(block)
        parallelize_module(block, tp_mesh, plan)

    logger.info(
        f"Applied tensor parallelism: tp_degree={tp_mesh.size()}, "
        f"layers={len(model.layers)}"
    )
