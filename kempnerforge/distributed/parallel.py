"""FSDP2 and activation checkpointing application.

Applies fully_shard (FSDP2 composable API) and activation checkpointing
to Transformer models. FSDP2 should always be applied LAST, after TP and AC.

Recommended application order:
  1. Tensor parallelism (apply_tensor_parallel) — must see raw blocks
  2. Activation checkpointing (apply_ac) — wraps blocks in CheckpointWrapper
  3. FSDP2 (apply_fsdp2) — shards everything
"""

from __future__ import annotations

import logging
from functools import partial

import torch
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh

from kempnerforge.config.schema import ActivationCheckpointing
from kempnerforge.model.transformer import Transformer, TransformerBlock

logger = logging.getLogger(__name__)


def get_dp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Extract the data-parallel sub-mesh from a DeviceMesh.

    Returns a 1D mesh (pure sharding) or 2D mesh (replicate + shard / HSDP)
    depending on which dimensions are present.
    """
    dim_names = device_mesh.mesh_dim_names
    has_replicate = "dp_replicate" in dim_names
    has_shard = "dp_shard" in dim_names

    if has_replicate and has_shard:
        # 2D HSDP: first dim = replicate, second dim = shard
        return device_mesh["dp_replicate", "dp_shard"]
    elif has_shard:
        return device_mesh["dp_shard"]
    elif has_replicate:
        return device_mesh["dp_replicate"]
    else:
        # No DP dimensions (e.g., pure TP) — return full mesh
        return device_mesh


def default_mp_policy() -> MixedPrecisionPolicy:
    """Default mixed-precision policy: bf16 compute, fp32 gradient reduction."""
    return MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )


def apply_ac(model: Transformer, mode: ActivationCheckpointing) -> None:
    """Apply activation checkpointing to the model.

    Must be called BEFORE apply_fsdp2.

    Args:
        model: Transformer model.
        mode: Checkpointing mode — "none", "full", or "selective".
            full: checkpoint every TransformerBlock (maximum memory savings).
            selective: checkpoint only Attention modules (balanced trade-off).
    """
    if mode == ActivationCheckpointing.none:
        return

    wrapper_fn = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    if mode == ActivationCheckpointing.full:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrapper_fn,
            check_fn=lambda m: isinstance(m, TransformerBlock),
        )
        logger.info("Applied full activation checkpointing (per TransformerBlock)")

    elif mode == ActivationCheckpointing.selective:
        from kempnerforge.model.attention import Attention

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrapper_fn,
            check_fn=lambda m: isinstance(m, Attention),
        )
        logger.info("Applied selective activation checkpointing (Attention only)")


def apply_fsdp2(
    model: Transformer,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None = None,
    reshard_after_forward: bool | int = True,
) -> None:
    """Apply FSDP2 (fully_shard) to a Transformer model.

    Shards each TransformerBlock independently, then wraps the top-level
    model for remaining parameters (embeddings, final norm, output head).

    Must be called AFTER apply_ac and apply_tensor_parallel.

    Args:
        model: Transformer model to shard.
        device_mesh: Full DeviceMesh (dp sub-mesh is extracted automatically).
        mp_policy: Mixed precision policy. Defaults to bf16 params + fp32 reduce.
        reshard_after_forward: Whether to free gathered params after forward.
            True = always reshard (saves memory, default).
            False = keep gathered (useful when PP needs params across microbatches).
            int = rate-limit the number of concurrent all-gathers.
    """
    dp_mesh = get_dp_mesh(device_mesh)
    policy = mp_policy or default_mp_policy()

    # Shard each transformer block as an independent FSDP unit
    for layer in model.layers.values():
        fully_shard(
            layer,
            mesh=dp_mesh,
            mp_policy=policy,
            reshard_after_forward=reshard_after_forward,
        )

    # Top-level shard covers remaining params (embeddings, final norm, output head)
    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=policy,
        reshard_after_forward=reshard_after_forward,
    )

    logger.info(
        f"Applied FSDP2: dp_mesh={dp_mesh.mesh_dim_names}, blocks={len(model.layers)}, mp={policy}"
    )
