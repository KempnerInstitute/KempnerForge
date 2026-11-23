"""Parallelism application: TP, AC, Float8, FSDP2, and model building.

Applies parallelism to Transformer models in the correct order.

Application order (critical — wrong order causes silent correctness bugs):
  1. Tensor parallelism (apply_tensor_parallel) — must see raw blocks
  2. Expert parallelism (apply_expert_parallel) — partitions MoE experts
  3. Float8 training (apply_float8) — wraps Linear → Float8Linear
  4. Activation checkpointing (apply_ac) — wraps blocks in CheckpointWrapper
  5. FSDP2 (apply_fsdp2) — shards everything (uses float8 all-gather if enabled)

For convenience, ``build_parallel_model`` combines all steps including
model creation, meta-device initialization, and optional torch.compile.
It also dispatches to a VLM branch when ``vlm_config`` is provided.
"""

from __future__ import annotations

import contextlib
import logging
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch.distributed._composable.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh

from kempnerforge.config.registry import registry
from kempnerforge.config.schema import ActivationCheckpointing
from kempnerforge.config.training import FP8_DIM_MULTIPLE
from kempnerforge.model.attention import Attention
from kempnerforge.model.cross_attention import CrossAttention
from kempnerforge.model.mot import MoTAttention
from kempnerforge.model.transformer import Transformer

if TYPE_CHECKING:
    from kempnerforge.distributed.pipeline_parallel import PipelineStageModule

logger = logging.getLogger(__name__)


def has_dp_mesh(device_mesh: DeviceMesh) -> bool:
    """Check whether the DeviceMesh contains any data-parallel dimensions."""
    dim_names = device_mesh.mesh_dim_names
    return "dp_shard" in dim_names or "dp_replicate" in dim_names  # type: ignore[reportOperatorIssue]


def get_dp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Extract the data-parallel sub-mesh from a DeviceMesh.

    Returns a 1D mesh (pure sharding) or 2D mesh (replicate + shard / HSDP)
    depending on which dimensions are present.

    Raises ValueError if no DP dimensions exist (e.g., pure TP mesh).
    Use ``has_dp_mesh`` to check first.
    """
    dim_names = device_mesh.mesh_dim_names
    has_replicate = "dp_replicate" in dim_names  # type: ignore[reportOperatorIssue]
    has_shard = "dp_shard" in dim_names  # type: ignore[reportOperatorIssue]

    if has_replicate and has_shard:
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
    """Mixed-precision policy: param_dtype compute, fp32 gradient reduction.

    ``cast_forward_inputs=True`` ensures FSDP2 casts input tensors to the
    declared ``param_dtype`` at each wrapped module's forward boundary.
    The VLM path relies on this so image embeddings produced by the
    adapter (bf16) reach the sharded transformer with matching dtype
    without needing the caller to do manual casts. The default on
    ``MixedPrecisionPolicy`` is False, so we set it explicitly here to
    pin the contract.
    """
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )


def is_fsdp_wrapped(model: torch.nn.Module) -> bool:
    """Whether ``fully_shard`` was applied to ``model`` or any of its submodules."""
    return any(isinstance(m, FSDPModule) for m in model.modules())


def autocast_dtype(
    model: torch.nn.Module, master_dtype: torch.dtype, compute_dtype: torch.dtype
) -> torch.dtype | None:
    """``compute_dtype`` if no FSDP2 wrap casts ``master_dtype`` parameters to it, else None."""
    if master_dtype == compute_dtype or is_fsdp_wrapped(model):
        return None
    return compute_dtype


_ATTENTION_TYPES = (Attention, MoTAttention, CrossAttention)


def _ac_blocks(model: torch.nn.Module) -> list[tuple[torch.nn.ModuleDict, str]]:
    """``(container, key)`` of every child of ``layers`` and ``cross_attention_layers``.

    A pipeline stage has no ``cross_attention_layers``, and a model without
    cross-attention has an empty one.
    """
    blocks = []
    for name in ("layers", "cross_attention_layers"):
        container = getattr(model, name, None)
        if container is not None:
            blocks.extend((container, key) for key in container)
    return blocks


def _is_checkpointed(module: torch.nn.Module) -> bool:
    """Whether ``module`` is, or holds, a ``CheckpointWrapper``."""
    return any(isinstance(m, CheckpointWrapper) for m in module.modules())


def apply_ac(model: Transformer | PipelineStageModule, mode: ActivationCheckpointing | str) -> int:
    """Apply activation checkpointing to the model's blocks.

    Must be called BEFORE apply_fsdp2.

    The blocks are the children of ``model.layers`` and, when present,
    ``model.cross_attention_layers``, selected by position, so every block
    class is covered: ``TransformerBlock``, ``MoTBlock``, ``MoMaBlock`` and
    ``CrossAttentionBlock`` alike. A block that already holds a
    ``CheckpointWrapper`` is left as it is, so calling this twice wraps
    nothing new.

    Args:
        model: Transformer model or pipeline stage.
        mode: Checkpointing mode, "none", "full", or "selective", as an
            ``ActivationCheckpointing`` member or its string value. ``full``
            checkpoints every block (maximum memory savings); ``selective``
            checkpoints the ``Attention``, ``MoTAttention`` and
            ``CrossAttention`` modules inside each block (balanced trade-off)
            and warns, naming the class, about a block that holds none of them.

    Returns:
        The number of modules this call wrapped.
    """
    mode = ActivationCheckpointing(mode)
    if mode == ActivationCheckpointing.none:
        return 0

    wrapper_fn = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    blocks = _ac_blocks(model)
    wrapped: dict[str, int] = {}
    for container, key in blocks:
        if _is_checkpointed(container[key]):
            continue
        if mode == ActivationCheckpointing.full:
            container[key] = wrapper_fn(container[key])
        else:
            apply_activation_checkpointing(
                container[key],
                checkpoint_wrapper_fn=wrapper_fn,
                check_fn=lambda m: isinstance(m, _ATTENTION_TYPES),
            )
        for m in container[key].modules():
            if isinstance(m, CheckpointWrapper):
                name = type(m._checkpoint_wrapped_module).__name__
                wrapped[name] = wrapped.get(name, 0) + 1

    uncovered = [type(c[k]).__name__ for c, k in blocks if not _is_checkpointed(c[k])]
    if not blocks:
        logger.warning(
            f"activation_checkpointing={mode.value!r} checkpoints nothing: the model has no "
            "blocks in layers or cross_attention_layers"
        )
    elif uncovered:
        logger.warning(
            f"activation_checkpointing={mode.value!r} leaves {len(uncovered)} of {len(blocks)} "
            f"blocks unwrapped ({', '.join(sorted(set(uncovered)))}): they hold no Attention, "
            "MoTAttention or CrossAttention module. Use 'full' to checkpoint them whole."
        )
    if wrapped:
        counts = ", ".join(f"{n} {name}" for name, n in wrapped.items())
        logger.info(f"Applied {mode.value} activation checkpointing: wrapped {counts}")
    elif blocks and not uncovered:
        logger.info(
            f"Activation checkpointing ({mode.value}) wrapped nothing new: every block "
            "was already checkpointed"
        )
    return sum(wrapped.values())


def apply_float8(model: Transformer, enable_fsdp_float8_all_gather: bool = True) -> None:
    """Apply Float8 training (torchao) to the model.

    Converts nn.Linear modules to Float8Linear for E4M3 forward / E5M2 backward
    with dynamic tensorwise scaling. Master weights keep their dtype (bf16, or fp32).

    Must be called AFTER apply_tensor_parallel / apply_expert_parallel
    and BEFORE apply_ac / apply_fsdp2. The shape check below reads each Linear's
    global ``in_features`` and ``out_features``, which are its weight's dimensions
    only without TP; config validation rejects FP8 with TP.

    Routed MoE experts are excluded because they use grouped GEMM
    (torch._grouped_mm), which bypasses Float8Linear.forward(). The shared expert
    is a plain Linear call; the same name filter excludes it, by convention.

    A Linear whose ``in_features`` or ``out_features`` is not a multiple of
    ``FP8_DIM_MULTIPLE`` (16) stays in high precision, because torch._scaled_mm
    rejects it on CUDA. One warning names each such layer with its shape, and the
    info line counts the converted Linears.

    Args:
        model: Transformer model.
        enable_fsdp_float8_all_gather: If True, FSDP2 all-gathers use float8
            (halves communication volume). Requires FSDP2 to be applied after.
            Must be False when TP is active — the float8 weight wrapper calls
            aten.is_pinned on DTensors, which has no sharding strategy yet.

    Raises:
        ValueError: If the model has Linear layers but none of them converts, so an
            FP8 run would train entirely in high precision.
    """
    import dataclasses

    from torchao.float8 import (
        Float8LinearConfig,
        Float8LinearRecipeName,
        convert_to_float8_training,
    )

    config = dataclasses.replace(
        Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.TENSORWISE),
        enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
    )

    unaligned: dict[str, int] = {}
    counts = {"converted": 0, "linear": 0}

    def _filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        """Skip modules that shouldn't use Float8.

        Excluded:
        - Routed expert Linears: use grouped GEMM (torch._grouped_mm), not Linear.forward()
        - Shared expert: a plain Linear call, excluded with the routed experts by convention
        - Router gate: small output dim (num_experts) often not divisible by 16,
          which torch._scaled_mm requires. Also not compute-bound.
        - Any other Linear with a dimension not divisible by 16, which torch._scaled_mm
          rejects in its first call. Each is recorded by name, with layer indices
          collapsed to ``*``, and by shape, for the warning.
        """
        if not isinstance(module, torch.nn.Linear):
            return False
        counts["linear"] += 1
        if "experts" in fqn or "shared_expert" in fqn or "router" in fqn:
            return False
        if module.in_features % FP8_DIM_MULTIPLE or module.out_features % FP8_DIM_MULTIPLE:
            pattern = ".".join("*" if part.isdigit() else part for part in fqn.split("."))
            key = f"{pattern} ({module.out_features}x{module.in_features})"
            unaligned[key] = unaligned.get(key, 0) + 1
            return False
        counts["converted"] += 1
        return True

    convert_to_float8_training(model, config=config, module_filter_fn=_filter_fn)

    kept = ", ".join(f"{name} x{count}" for name, count in unaligned.items())
    if counts["linear"] and not counts["converted"]:
        raise ValueError(
            f"Float8 converted 0 of {counts['linear']} Linear layers, so "
            "train.mixed_precision='fp8' would train entirely in high precision. Experts, "
            "shared experts and routers are excluded by name, and torch._scaled_mm needs both "
            f"dimensions of a weight divisible by {FP8_DIM_MULTIPLE}, which these are not: "
            f"{kept or 'none'}. Make those dimensions multiples of {FP8_DIM_MULTIPLE}, or "
            "train in bf16."
        )

    logger.info(
        f"Applied Float8 training: recipe=TENSORWISE, "
        f"fsdp_float8_all_gather={enable_fsdp_float8_all_gather}, "
        f"converted {counts['converted']} of {counts['linear']} Linear layers"
    )
    if unaligned:
        logger.warning(
            f"Float8 kept {sum(unaligned.values())} Linear layer(s) in high precision because "
            f"torch._scaled_mm needs both dimensions divisible by {FP8_DIM_MULTIPLE}: {kept}"
        )


def _has_ep_moe(module: torch.nn.Module) -> bool:
    """Check if a module contains MoE with expert parallelism active."""
    from kempnerforge.model.moe import MoEMLP

    return any(isinstance(m, MoEMLP) and m.ep_world_size > 1 for m in module.modules())


def _fsdp_wrap_transformer_blocks(
    transformer: Transformer | PipelineStageModule,
    dp_mesh: DeviceMesh,
    policy: MixedPrecisionPolicy,
    reshard_after_forward: bool | int,
) -> int:
    """Per-block FSDP2 wrap over ``transformer.layers`` and
    ``transformer.cross_attention_layers``, EP-MoE aware.

    Dense blocks are wrapped once per ``TransformerBlock``. EP-MoE blocks
    get per-sub-module wrapping (``layer.attention`` and ``layer.mlp``
    separately) because wrapping the whole block would cause FSDP2's
    reduce-scatter to fire between EP's backward all-to-all calls and
    deadlock. Per-sub-module wrapping keeps the two EP all-to-alls
    inside a single FSDP unit (``layer.mlp``), so reduce-scatter only
    fires after both complete.

    Cross-Attention blocks (when present) are wrapped once each, like
    dense ``TransformerBlock``s. ``cross_attention_layers`` has three
    states, all handled: absent, on a ``PipelineStageModule``, which
    defines no such attribute; empty, on the JD / text-only paths; and
    populated, on Cross-Attention configs. Only the third wraps anything.

    Shared by ``apply_fsdp2`` (for text Transformers) and
    ``_apply_fsdp_vlm`` (for the inner Transformer of a VLMWrapper) so
    the VLM path inherits the EP-MoE safety automatically if a future
    config combines MoE with VLM.

    Returns the number of EP-MoE blocks that got the per-sub-module
    wrap; 0 for dense models.
    """
    ep_sub_wrapped = 0
    for layer in transformer.layers.values():
        if _has_ep_moe(layer):
            fully_shard(  # type: ignore[reportCallIssue]
                layer.attention,  # type: ignore[reportArgumentType]
                mesh=dp_mesh,
                mp_policy=policy,
                reshard_after_forward=reshard_after_forward,
            )
            fully_shard(  # type: ignore[reportCallIssue]
                layer.mlp,  # type: ignore[reportArgumentType]
                mesh=dp_mesh,
                mp_policy=policy,
                reshard_after_forward=reshard_after_forward,
            )
            ep_sub_wrapped += 1
            continue
        fully_shard(
            layer,
            mesh=dp_mesh,
            mp_policy=policy,
            reshard_after_forward=reshard_after_forward,
        )

    for ca_block in getattr(transformer, "cross_attention_layers", {}).values():
        fully_shard(
            ca_block,
            mesh=dp_mesh,
            mp_policy=policy,
            reshard_after_forward=reshard_after_forward,
        )

    return ep_sub_wrapped


def apply_fsdp2(
    model: Transformer | PipelineStageModule,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None = None,
    reshard_after_forward: bool | int = True,
) -> None:
    """Apply FSDP2 (fully_shard) to a Transformer model.

    Shards each TransformerBlock independently (via
    ``_fsdp_wrap_transformer_blocks`` so the EP-MoE per-sub-module wrap
    is shared with the VLM path), then wraps the top-level model for
    remaining parameters (embeddings, final norm, output head).

    Must be called AFTER apply_ac and apply_tensor_parallel.

    **EP interaction**: Blocks with expert parallelism get per-sub-module
    wrapping (attention and MoE individually) instead of per-block wrapping.
    Per-block wrapping would cause FSDP2's reduce-scatter to fire between
    EP's backward all-to-all calls (deadlock). Per-sub-module wrapping avoids
    this: the MoE reduce-scatter fires after the entire MoE backward (both
    EP all-to-alls complete), while attention reduce-scatter is EP-free.

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

    ep_sub_wrapped = _fsdp_wrap_transformer_blocks(model, dp_mesh, policy, reshard_after_forward)

    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=policy,
        reshard_after_forward=reshard_after_forward,
    )

    logger.info(
        f"Applied FSDP2: dp_mesh={dp_mesh.mesh_dim_names}, blocks={len(model.layers)}, "
        f"mp={policy}" + (f", ep_moe_blocks_sub_wrapped={ep_sub_wrapped}" if ep_sub_wrapped else "")
    )


def _apply_fsdp_vlm(
    wrapper: torch.nn.Module,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    *,
    encoder_frozen: bool,
    encoder_mp_policy: MixedPrecisionPolicy | None = None,
    reshard_after_forward: bool | int = True,
) -> None:
    """FSDP2 wrap policy for a ``VLMWrapper``.

    Shards per-component:
      - Each ``TransformerBlock`` (via ``_fsdp_wrap_transformer_blocks``
        so the EP-MoE per-sub-module wrap is shared with the text path
        and VLM+MoE cannot deadlock).
      - Transformer root (embedding + final norm + output head): own unit.
      - Adapter: own unit (small, but keeps grad-sync scheduling symmetric).
      - Vision encoder: wrapped only when not fully frozen, with
        ``encoder_mp_policy`` when given and the shared policy otherwise.
        A frozen encoder stays as a full replica in eval mode; replication
        is fine because requires_grad=False means no grad-reduce participation.
      - The ``VLMWrapper`` itself is not wrapped (no direct parameters).
    """
    if not has_dp_mesh(device_mesh):
        return

    from kempnerforge.model.vlm import VLMWrapper

    assert isinstance(wrapper, VLMWrapper), "_apply_fsdp_vlm expects a VLMWrapper"

    dp_mesh = get_dp_mesh(device_mesh)
    policy = mp_policy or default_mp_policy()

    _fsdp_wrap_transformer_blocks(wrapper.transformer, dp_mesh, policy, reshard_after_forward)
    fully_shard(
        wrapper.transformer,
        mesh=dp_mesh,
        mp_policy=policy,
        reshard_after_forward=reshard_after_forward,
    )
    fully_shard(
        wrapper.adapter,
        mesh=dp_mesh,
        mp_policy=policy,
        reshard_after_forward=reshard_after_forward,
    )
    if not encoder_frozen:
        fully_shard(
            wrapper.vision_encoder,
            mesh=dp_mesh,
            mp_policy=encoder_mp_policy if encoder_mp_policy is not None else policy,
            reshard_after_forward=reshard_after_forward,
        )
    logger.info(
        f"Applied VLM FSDP2: dp_mesh={dp_mesh.mesh_dim_names}, "
        f"blocks={len(wrapper.transformer.layers)}, encoder_frozen={encoder_frozen}"
    )


def _cast_floating(value: object, dtype: torch.dtype) -> object:
    """``value`` cast to ``dtype`` if it is a floating-point tensor, else ``value`` unchanged."""
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(dtype)
    return value


class _ComputeInDtype:
    """A module's forward that computes in ``dtype`` while its parameters keep their dtype.

    Installed as the module's ``forward``. Each call puts a ``dtype`` copy in place of
    every floating-point parameter of another dtype, one copy per parameter however many
    names it has, casts the floating-point arguments, and runs the module's own forward
    with autocast off on ``device_type``, so a surrounding ``torch.autocast`` does not
    change the dtype it computes in. The parameters go back when the call returns or
    raises. A nested call finds the copies in place and casts no parameter. Autograd
    returns each copy's gradient to its parameter in the parameter's dtype, as with an
    FSDP2 ``MixedPrecisionPolicy`` whose ``param_dtype`` is ``dtype``.

    Under ``torch.compile`` the call itself runs eagerly, at a graph break, and the
    module's forward compiles as its own graph, entered with autocast off. A compiled
    graph's backward is traced under the autocast its forward was entered with, so this
    keeps the backward in ``dtype`` too.
    """

    def __init__(self, module: torch.nn.Module, dtype: torch.dtype, device_type: str) -> None:
        self._module = module
        self._forward = module.forward
        self.dtype = dtype
        self.device_type = device_type

    def __call__(self, *args: object, **kwargs: object) -> object:
        return _run_eagerly(self, *args, **kwargs)

    def _run(self, *args: object, **kwargs: object) -> object:
        swapped: list[tuple[torch.nn.Module, str, torch.nn.Parameter]] = []
        try:
            copies: dict[int, torch.Tensor] = {}
            for owner in self._module.modules():
                for name, param in list(owner._parameters.items()):
                    if param is None or not param.is_floating_point() or param.dtype == self.dtype:
                        continue
                    if id(param) not in copies:
                        copies[id(param)] = param.to(self.dtype)
                    swapped.append((owner, name, param))
                    owner._parameters[name] = copies[id(param)]  # type: ignore[assignment]
            with torch.autocast(self.device_type, enabled=False):
                return self._forward(
                    *(_cast_floating(a, self.dtype) for a in args),
                    **{k: _cast_floating(v, self.dtype) for k, v in kwargs.items()},
                )
        finally:
            for owner, name, param in reversed(swapped):
                owner._parameters[name] = param


_run_eagerly = torch.compiler.disable(_ComputeInDtype._run, recursive=False)


def _set_encoder_precision(
    encoder: torch.nn.Module, dtype: torch.dtype, *, fsdp_wrapped: bool, device_type: str
) -> None:
    """Make the vision encoder compute in ``dtype`` while its parameters keep their dtype.

    Its floating-point buffers are cast to ``dtype``, every registered name as
    ``Module.to`` does, so they are also saved in it. A wrapped encoder's policy casts
    its parameters and inputs for each forward. An unwrapped one gets
    ``_ComputeInDtype`` as its ``forward``, which does the same and keeps the forward
    out of any ``torch.autocast`` on ``device_type``. Either way its parameters,
    gradients and optimizer state keep the dtype the encoder was loaded in.
    """
    for module in encoder.modules():
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            if buffer.is_floating_point():
                setattr(module, name, buffer.to(dtype))
    if fsdp_wrapped:
        return
    encoder.forward = _ComputeInDtype(encoder, dtype, device_type)


def _build_vlm(
    model_config,
    vision_config,
    adapter_config,
    vlm_config,
    device: torch.device,
    device_mesh: DeviceMesh | None,
    *,
    ac_mode: ActivationCheckpointing,
    mp_policy: MixedPrecisionPolicy | None,
    param_dtype: torch.dtype,
    compile_model: bool,
    fp8: bool,
    frames_per_clip: int = 1,
) -> torch.nn.Module:
    """Build a VLM wrapper with parallelism applied in the correct order.

    The VLM pipeline sequences component-by-component:

      1. Vision encoder is built on CPU via HF (real weights). Never on meta.
      2. Transformer + Adapter are built under ``torch.device("meta")`` when
         TP is active (to avoid OOM before sharding), or directly on CPU
         otherwise.
      3. They are composed into a ``VLMWrapper`` and the image-prefix length
         is cross-checked against ``model_config.max_seq_len`` now that
         ``num_tokens`` is known from the encoder.
      4. TP / EP / Float8 / AC are applied to the transformer only.
      5. FSDP2 is applied component-by-component (transformer blocks,
         transformer root, adapter, and vision encoder iff not frozen).
         With ``vision_config.mixed_precision`` set, the encoder's unit
         gets its own policy that computes in that dtype.
      6. Meta subtrees are materialized (transformer / adapter), the
         vision encoder is moved to ``device``, and the transformer and
         adapter are cast to ``param_dtype``. The vision encoder's
         parameters stay in its HF dtype (D16) to avoid ViT LayerNorm
         numerical drift; with ``vision_config.mixed_precision`` set,
         ``_set_encoder_precision`` makes it compute in that dtype.
      7. Freeze specs are applied via ``apply_freeze_specs`` and a fully
         frozen encoder is switched to ``eval()``.
    """
    from kempnerforge.distributed.expert_parallel import apply_expert_parallel
    from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel
    from kempnerforge.model.adapter import build_adapter
    from kempnerforge.model.transformer import Transformer
    from kempnerforge.model.vlm import (
        VLMWrapper,
        _is_encoder_frozen,
        build_modality_strategy,
    )
    from kempnerforge.training.freeze import apply_freeze_specs

    assert vlm_config is not None, "_build_vlm requires vlm_config to be set"
    assert vision_config is not None, "_build_vlm requires vision_config to be set"
    assert adapter_config is not None, "_build_vlm requires adapter_config to be set"
    tp_enabled = device_mesh is not None and "tp" in device_mesh.mesh_dim_names  # type: ignore[reportOperatorIssue]

    encoder_builder = registry.get_vision_encoder(vision_config.type)
    encoder = encoder_builder(
        vision_config.path,
        num_tokens=vision_config.num_tokens if vision_config.num_tokens > 0 else None,
        feature_dim=vision_config.feature_dim if vision_config.feature_dim > 0 else None,
    )
    in_dim = vision_config.feature_dim or encoder.feature_dim

    ctx = torch.device("meta") if tp_enabled else contextlib.nullcontext()
    with ctx:
        adapter = build_adapter(
            adapter_config, in_dim=in_dim, out_dim=model_config.dim, norm_eps=model_config.norm_eps
        )
        visual_tokens = frames_per_clip * adapter.output_num_tokens(encoder.num_tokens)
        transformer = Transformer(
            model_config, vlm_config=vlm_config, num_image_tokens=visual_tokens
        )

    strategy = build_modality_strategy(vlm_config)
    wrapper = VLMWrapper(encoder, adapter, transformer, strategy, frames_per_clip=frames_per_clip)

    required = wrapper.num_image_tokens + vlm_config.max_text_len
    if model_config.max_seq_len < required:
        raise ValueError(
            f"max_seq_len ({model_config.max_seq_len}) insufficient for VLM: "
            f"num_image_tokens ({wrapper.num_image_tokens}) + vlm.max_text_len "
            f"({vlm_config.max_text_len}) = {required}"
        )

    if tp_enabled:
        apply_tensor_parallel(transformer, device_mesh)  # type: ignore[reportArgumentType]
    apply_expert_parallel(transformer, device_mesh)
    if fp8:
        apply_float8(transformer)
    apply_ac(transformer, ac_mode)

    encoder_frozen = _is_encoder_frozen(vlm_config.freeze)
    encoder_dtype = vision_config.compute_dtype
    if device_mesh is not None:
        _apply_fsdp_vlm(
            wrapper,
            device_mesh,
            mp_policy,
            encoder_frozen=encoder_frozen,
            encoder_mp_policy=(
                default_mp_policy(encoder_dtype) if encoder_dtype is not None else None
            ),
        )

    if tp_enabled:
        transformer.to_empty(device=device)
        transformer.init_weights_and_freqs()
        adapter.to_empty(device=device)
        adapter.reset_parameters()  # type: ignore[reportCallIssue,reportAttributeAccessIssue]
    else:
        transformer.to(device=device)
        adapter.to(device=device)
    encoder.to(device)
    if encoder_dtype is not None:
        _set_encoder_precision(
            encoder,
            encoder_dtype,
            fsdp_wrapped=isinstance(wrapper.vision_encoder, FSDPModule),
            device_type=device.type,
        )
    transformer.to(dtype=param_dtype)
    adapter.to(dtype=param_dtype)

    apply_freeze_specs(wrapper, vlm_config.freeze, vlm_config.module_patterns)
    if encoder_frozen:
        encoder.eval()

    compiled: torch.nn.Module = wrapper
    if compile_model:
        logger.info("Compiling VLM wrapper with torch.compile...")
        compiled = torch.compile(wrapper)  # type: ignore[assignment]

    n_params = sum(p.numel() for p in wrapper.parameters())
    n_trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    logger.info(
        f"VLM model: {n_params:,} total, {n_trainable:,} trainable, "
        f"num_image_tokens={wrapper.num_image_tokens}, encoder_frozen={encoder_frozen}"
    )
    return compiled


def build_parallel_model(
    model_config,
    device: torch.device,
    device_mesh: DeviceMesh | None,
    *,
    vision_config=None,
    adapter_config=None,
    vlm_config=None,
    ac_mode: ActivationCheckpointing = ActivationCheckpointing.none,
    mp_policy: MixedPrecisionPolicy | None = None,
    param_dtype: torch.dtype = torch.bfloat16,
    compile_model: bool = False,
    fp8: bool = False,
    frames_per_clip: int = 1,
) -> torch.nn.Module:
    """Build a Transformer (or a VLMWrapper) with parallelism applied.

    Dispatches on ``vlm_config`` (``None`` -> text-only path). Non-VLM
    configurations follow the original order:

      - TP enabled:  meta-device init -> TP -> EP -> [Float8] -> AC -> FSDP -> materialize
      - TP disabled: create on device -> EP -> [Float8] -> AC -> FSDP

    VLM configurations follow the order documented on ``_build_vlm``.

    This is the non-PP model building path. For pipeline parallelism,
    use ``build_stage_module`` + apply parallelism directly.

    Args:
        model_config: ModelConfig for the Transformer.
        device: Target device for the model.
        device_mesh: Full DeviceMesh (may contain tp, dp_shard, dp_replicate dims).
        vision_config: VisionEncoderConfig (required iff vlm_config is set).
        adapter_config: AdapterConfig (required iff vlm_config is set).
        vlm_config: VLMConfig. None for a pure text-only run.
        ac_mode: Activation checkpointing mode.
        mp_policy: FSDP2 mixed-precision policy. Defaults to bf16 params + fp32 reduce.
        param_dtype: Dtype for model parameters.
        compile_model: Whether to torch.compile the model.
        fp8: Whether to enable Float8 mixed precision (torchao).

    Returns:
        The parallelized model, ready for training.
    """
    if vlm_config is not None:
        return _build_vlm(
            model_config,
            vision_config,
            adapter_config,
            vlm_config,
            device,
            device_mesh,
            ac_mode=ac_mode,
            mp_policy=mp_policy,
            param_dtype=param_dtype,
            compile_model=compile_model,
            fp8=fp8,
            frames_per_clip=frames_per_clip,
        )

    from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel

    tp_enabled = device_mesh is not None and "tp" in device_mesh.mesh_dim_names  # type: ignore[reportOperatorIssue]
    model_builder = registry.get_model(model_config.model_type)

    from kempnerforge.distributed.expert_parallel import apply_expert_parallel

    if tp_enabled:
        with torch.device("meta"):
            model = model_builder(model_config)
        apply_tensor_parallel(model, device_mesh)  # type: ignore[reportArgumentType]
        apply_expert_parallel(model, device_mesh)
        if fp8:
            apply_float8(model)
        apply_ac(model, ac_mode)
        if device_mesh is not None:
            apply_fsdp2(model, device_mesh, mp_policy=mp_policy)
        model.to_empty(device=device)
        model.init_weights_and_freqs()
        model.to(dtype=param_dtype)
    else:
        model = model_builder(model_config).to(device=device, dtype=param_dtype)
        apply_expert_parallel(model, device_mesh)
        if fp8:
            apply_float8(model)
        apply_ac(model, ac_mode)
        if device_mesh is not None:
            apply_fsdp2(model, device_mesh, mp_policy=mp_policy)

    if compile_model:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    return model  # type: ignore[reportReturnType]
