"""Parallelism application: TP, AC, FSDP2, and model building.

Applies parallelism to Transformer models in the correct order.

Application order (critical — wrong order causes silent correctness bugs):
  1. Tensor parallelism (apply_tensor_parallel) — must see raw blocks
  2. Activation checkpointing (apply_ac) — wraps blocks in CheckpointWrapper
  3. FSDP2 (apply_fsdp2) — shards everything

For convenience, ``build_parallel_model`` combines all steps including
model creation, meta-device initialization, and optional torch.compile.
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


def has_dp_mesh(device_mesh: DeviceMesh) -> bool:
    """Check whether the DeviceMesh contains any data-parallel dimensions."""
    dim_names = device_mesh.mesh_dim_names
    return "dp_shard" in dim_names or "dp_replicate" in dim_names


def get_dp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Extract the data-parallel sub-mesh from a DeviceMesh.

    Returns a 1D mesh (pure sharding) or 2D mesh (replicate + shard / HSDP)
    depending on which dimensions are present.

    Raises ValueError if no DP dimensions exist (e.g., pure TP mesh).
    Use ``has_dp_mesh`` to check first.
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
        raise ValueError(
            f"No DP dimensions in mesh {dim_names}. "
            "Use has_dp_mesh() to check before calling get_dp_mesh()."
        )


def default_mp_policy(param_dtype: torch.dtype = torch.bfloat16) -> MixedPrecisionPolicy:
    """Mixed-precision policy: param_dtype compute, fp32 gradient reduction."""
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
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
    if not has_dp_mesh(device_mesh):
        logger.info("No DP dimensions in mesh — skipping FSDP2")
        return

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


def build_parallel_model(
    model_config,
    device: torch.device,
    device_mesh: DeviceMesh | None,
    *,
    ac_mode: ActivationCheckpointing = ActivationCheckpointing.none,
    mp_policy: MixedPrecisionPolicy | None = None,
    param_dtype: torch.dtype = torch.bfloat16,
    compile_model: bool = False,
) -> torch.nn.Module:
    """Build a Transformer with parallelism applied in the correct order.

    Handles four configurations automatically:
      - TP enabled:  meta-device init → TP → AC → FSDP → materialize → init weights
      - TP disabled: create on device → AC → FSDP

    This is the non-PP model building path. For pipeline parallelism,
    use ``build_stage_module`` + apply parallelism directly.

    Args:
        model_config: ModelConfig for the Transformer.
        device: Target device for the model.
        device_mesh: Full DeviceMesh (may contain tp, dp_shard, dp_replicate dims).
        ac_mode: Activation checkpointing mode.
        mp_policy: FSDP2 mixed-precision policy. Defaults to bf16 params + fp32 reduce.
        param_dtype: Dtype for model parameters.
        compile_model: Whether to torch.compile the model.

    Returns:
        The parallelized model, ready for training.
    """
    from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel

    tp_enabled = device_mesh is not None and "tp" in device_mesh.mesh_dim_names

    if tp_enabled:
        # Meta-device init: create model with zero memory, apply parallelisms,
        # then materialize only the local shards on GPU.
        with torch.device("meta"):
            model = Transformer(model_config)
        apply_tensor_parallel(model, device_mesh)
        apply_ac(model, ac_mode)
        if device_mesh is not None:
            apply_fsdp2(model, device_mesh, mp_policy=mp_policy)
        model.to_empty(device=device)
        model.init_weights_and_freqs()
        model.to(dtype=param_dtype)
    else:
        model = Transformer(model_config).to(device=device, dtype=param_dtype)
        apply_ac(model, ac_mode)
        if device_mesh is not None:
            apply_fsdp2(model, device_mesh, mp_policy=mp_policy)

    if compile_model:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    return model
