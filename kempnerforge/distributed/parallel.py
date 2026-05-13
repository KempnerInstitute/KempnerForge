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

import torch
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh

from kempnerforge.config.registry import registry
from kempnerforge.config.schema import ActivationCheckpointing
from kempnerforge.model.transformer import Transformer, TransformerBlock

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


def apply_float8(model: Transformer, enable_fsdp_float8_all_gather: bool = True) -> None:
    """Apply Float8 training (torchao) to the model.

    Converts nn.Linear modules to Float8Linear for E4M3 forward / E5M2 backward
    with dynamic tensorwise scaling. Master weights remain in bf16.

    Must be called AFTER apply_tensor_parallel / apply_expert_parallel
    and BEFORE apply_ac / apply_fsdp2.

    MoE expert modules (experts and shared_expert) are excluded because they use
    grouped GEMM (torch._grouped_mm) which bypasses Float8Linear.forward().

    Args:
        model: Transformer model.
        enable_fsdp_float8_all_gather: If True, FSDP2 all-gathers use float8
            (halves communication volume). Requires FSDP2 to be applied after.
            Must be False when TP is active — the float8 weight wrapper calls
            aten.is_pinned on DTensors, which has no sharding strategy yet.
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

    def _filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        """Skip modules that shouldn't use Float8.

        Excluded:
        - Expert Linears: use grouped GEMM (torch._grouped_mm), not Linear.forward()
        - Router gate: small output dim (num_experts) often not divisible by 16,
          which torch._scaled_mm requires. Also not compute-bound.
        """
        if "experts" in fqn or "shared_expert" in fqn:
            return False
        return "router" not in fqn

    convert_to_float8_training(model, config=config, module_filter_fn=_filter_fn)

    logger.info(
        f"Applied Float8 training: recipe=TENSORWISE, "
        f"fsdp_float8_all_gather={enable_fsdp_float8_all_gather}"
    )


def _has_ep_moe(module: torch.nn.Module) -> bool:
    """Check if a module contains MoE with expert parallelism active."""
    from kempnerforge.model.moe import MoEMLP

    return any(isinstance(m, MoEMLP) and m.ep_world_size > 1 for m in module.modules())


def _fsdp_wrap_transformer_blocks(
    transformer: Transformer,
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
    dense ``TransformerBlock``s. The dict is empty for non-CA configs,
    so iteration is a no-op on the JD / text-only paths.

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

    # Cross-Attention blocks (Cross-Attention arch only). Empty dict
    # for non-CA configs.
    for ca_block in transformer.cross_attention_layers.values():
        fully_shard(
            ca_block,
            mesh=dp_mesh,
            mp_policy=policy,
            reshard_after_forward=reshard_after_forward,
        )

    return ep_sub_wrapped


def apply_fsdp2(
    model: Transformer,
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

    # Top-level shard covers remaining params (embeddings, final norm, output
    # head, and layer norms from EP-MoE blocks).
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
    reshard_after_forward: bool | int = True,
) -> None:
    """FSDP2 wrap policy for a ``VLMWrapper``.

    Shards per-component:
      - Each ``TransformerBlock`` (via ``_fsdp_wrap_transformer_blocks``
        so the EP-MoE per-sub-module wrap is shared with the text path
        and VLM+MoE cannot deadlock).
      - Transformer root (embedding + final norm + output head): own unit.
      - Adapter: own unit (small, but keeps grad-sync scheduling symmetric).
      - Vision encoder: wrapped only when not fully frozen. A frozen encoder
        stays as a full replica in eval mode; replication is fine because
        requires_grad=False means no grad-reduce participation.
      - The ``VLMWrapper`` itself is not wrapped (no direct parameters).
    """
    if not has_dp_mesh(device_mesh):
        return

    # Local import to avoid a hard dependency at module load time.
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
            mp_policy=policy,
            reshard_after_forward=reshard_after_forward,
        )
    logger.info(
        f"Applied VLM FSDP2: dp_mesh={dp_mesh.mesh_dim_names}, "
        f"blocks={len(wrapper.transformer.layers)}, encoder_frozen={encoder_frozen}"
    )


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
      6. Meta subtrees are materialized (transformer / adapter), the
         vision encoder is moved to ``device``, and the transformer and
         adapter are cast to ``param_dtype``. The vision encoder stays in
         its HF dtype (D16) to avoid ViT LayerNorm numerical drift.
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

    # 1. Vision encoder on CPU (real HF weights).
    encoder_builder = registry.get_vision_encoder(vision_config.type)
    encoder = encoder_builder(
        vision_config.path,
        num_tokens=vision_config.num_tokens if vision_config.num_tokens > 0 else None,
        feature_dim=vision_config.feature_dim if vision_config.feature_dim > 0 else None,
    )
    in_dim = vision_config.feature_dim or encoder.feature_dim

    # 2. Transformer + Adapter on meta (when TP is active) or CPU.
    # VLM path always constructs Transformer directly (the registry's
    # text-only builder takes a single ModelConfig and cannot accept the
    # extra vlm_config / num_image_tokens kwargs the VLM path needs).
    ctx = torch.device("meta") if tp_enabled else contextlib.nullcontext()
    with ctx:
        transformer = Transformer(
            model_config, vlm_config=vlm_config, num_image_tokens=encoder.num_tokens
        )
        adapter = build_adapter(adapter_config, in_dim=in_dim, out_dim=model_config.dim)

    strategy = build_modality_strategy(vlm_config)
    wrapper = VLMWrapper(encoder, adapter, transformer, strategy)

    # 3. Length cross-check now that num_tokens is resolved.
    required = wrapper.num_image_tokens + vlm_config.max_text_len
    if model_config.max_seq_len < required:
        raise ValueError(
            f"max_seq_len ({model_config.max_seq_len}) insufficient for VLM: "
            f"num_image_tokens ({wrapper.num_image_tokens}) + vlm.max_text_len "
            f"({vlm_config.max_text_len}) = {required}"
        )

    # 4. TP / EP / Float8 / AC on the transformer only.
    if tp_enabled:
        apply_tensor_parallel(transformer, device_mesh)  # type: ignore[reportArgumentType]
    apply_expert_parallel(transformer, device_mesh)
    if fp8:
        apply_float8(transformer)
    apply_ac(transformer, ac_mode)

    # 5. FSDP2 (before materialization when meta-device path is used).
    encoder_frozen = _is_encoder_frozen(vlm_config.freeze)
    if device_mesh is not None:
        _apply_fsdp_vlm(wrapper, device_mesh, mp_policy, encoder_frozen=encoder_frozen)

    # 6. Materialize and move to device.
    if tp_enabled:
        transformer.to_empty(device=device)
        transformer.init_weights_and_freqs()
        adapter.to_empty(device=device)
        # Adapter contract (model/adapter.py): every registered adapter
        # exposes reset_parameters() so meta-device builds can re-init the
        # weights after to_empty. nn.Module itself does not declare the
        # method, so pyright sees an unknown attr; suppress the report.
        adapter.reset_parameters()  # type: ignore[reportCallIssue,reportAttributeAccessIssue]
    else:
        transformer.to(device=device)
        adapter.to(device=device)
    encoder.to(device)  # Keep HF dtype per D16.
    transformer.to(dtype=param_dtype)
    adapter.to(dtype=param_dtype)

    # 7. Freeze specs + eval() for fully frozen encoder.
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
        )

    from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel

    tp_enabled = device_mesh is not None and "tp" in device_mesh.mesh_dim_names  # type: ignore[reportOperatorIssue]
    model_builder = registry.get_model(model_config.model_type)

    from kempnerforge.distributed.expert_parallel import apply_expert_parallel

    if tp_enabled:
        # Meta-device init: create model with zero memory, apply parallelisms,
        # then materialize only the local shards on GPU.
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
