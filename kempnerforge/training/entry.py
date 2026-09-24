"""Library entry point for a KempnerForge training job.

``scripts/train.py`` is a thin CLI wrapper over :func:`run_training`: it
parses argv, loads the config, and calls in. Each phase below is a
separate function so an experiment can reuse the scaffold — or swap one
phase — without editing the shared script.
"""

from __future__ import annotations

from typing import cast

import torch

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.job import JobConfig
from kempnerforge.config.vlm import MoTConfig
from kempnerforge.distributed.parallel import (
    apply_ac,
    apply_float8,
    apply_fsdp2,
    build_parallel_model,
    default_mp_policy,
)
from kempnerforge.distributed.setup import destroy_distributed
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel
from kempnerforge.metrics.logger import get_logger
from kempnerforge.metrics.tracker import MetricsTracker
from kempnerforge.model.mot import mot_warm_start_from_text_stack
from kempnerforge.model.transformer import Transformer
from kempnerforge.model.vlm import inner_transformer
from kempnerforge.profiling.profiler import build_profiler
from kempnerforge.resilience.elastic import resolve_resume_path
from kempnerforge.resilience.health import NaNDetector
from kempnerforge.resilience.signal_handler import ShutdownHandler
from kempnerforge.training.data_pipeline import (
    build_data_pipeline,
    build_eval_dataloader,
    build_phase_state,
)
from kempnerforge.training.freeze import (
    apply_freeze_specs,
    effective_freeze,
    freeze_meta_at_step,
)
from kempnerforge.training.hooks import HookRunner
from kempnerforge.training.loop import (
    LossFn,
    StepFn,
    TrainingSession,
    run_training_loop,
    select_step_fn,
)
from kempnerforge.training.loss import build_loss_fn
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.runtime import PipelineBundle, RuntimeContext, setup_distributed
from kempnerforge.training.scheduler import build_scheduler

logger = get_logger(__name__)


def build_model(
    config: JobConfig, runtime: RuntimeContext, loss_fn: LossFn
) -> tuple[torch.nn.Module, PipelineBundle | None]:
    """Build the model with the full parallelism stack applied.

    With ``distributed.pp > 1`` this builds this rank's pipeline stage plus
    its schedule (which is why ``loss_fn`` is needed here); otherwise it
    delegates to ``build_parallel_model`` and returns no pipeline.
    """
    tc = config.train
    device = runtime.device
    device_mesh = runtime.device_mesh
    mp_policy = default_mp_policy(tc.param_dtype)

    if config.distributed.pp <= 1:
        model = build_parallel_model(
            config.model,
            device,
            device_mesh,
            vision_config=config.vision_encoder,
            adapter_config=config.adapter,
            vlm_config=config.vlm,
            frames_per_clip=(config.video.max_frames if config.video is not None else 1),
            ac_mode=tc.activation_checkpointing,
            mp_policy=mp_policy,
            param_dtype=tc.param_dtype,
            compile_model=tc.compile_model,
            fp8=tc.is_fp8,
        )
        return model, None

    from kempnerforge.distributed.pipeline_parallel import (
        build_pipeline_schedule,
        build_pipeline_stage,
        build_stage_module,
        get_pp_rank,
        get_pp_size,
    )

    if device_mesh is None:
        # Unreachable: JobConfig.validate rejects pp > 1 at world_size 1, the
        # only case init_distributed returns no mesh. Explicit so it survives -O.
        raise RuntimeError("pipeline parallelism requires a device mesh")
    pp_rank = get_pp_rank(device_mesh)
    pp_size = get_pp_size(device_mesh)

    tp_enabled_pp = "tp" in device_mesh.mesh_dim_names  # type: ignore[reportOperatorIssue]

    # apply_{float8,ac,fsdp2} are annotated for Transformer; a stage module
    # exposes the same block structure, hence the cast.
    if tp_enabled_pp:
        # Meta-device init: same pattern as non-PP TP path.
        # Avoids OOM for large PP stages that don't fit on one GPU before TP shards them.
        with torch.device("meta"):
            stage_mod = cast("Transformer", build_stage_module(config.model, pp_rank, pp_size))
        apply_tensor_parallel(stage_mod, device_mesh)
        if tc.is_fp8:
            apply_float8(stage_mod)
        apply_ac(stage_mod, tc.activation_checkpointing)
        apply_fsdp2(stage_mod, device_mesh, mp_policy=mp_policy)
        stage_mod.to_empty(device=device)
        stage_mod.init_weights_and_freqs()
        stage_mod.to(dtype=tc.param_dtype)
    else:
        stage_mod = cast(
            "Transformer",
            build_stage_module(config.model, pp_rank, pp_size).to(
                device=device, dtype=tc.param_dtype
            ),
        )
        if tc.is_fp8:
            apply_float8(stage_mod)
        apply_ac(stage_mod, tc.activation_checkpointing)
        apply_fsdp2(stage_mod, device_mesh, mp_policy=mp_policy)

    model: torch.nn.Module = stage_mod
    if tc.compile_model:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)  # type: ignore[assignment]

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model (PP stage {pp_rank}/{pp_size}): {n_params:,} parameters")

    pp_stage = build_pipeline_stage(
        model,  # type: ignore[arg-type]
        device_mesh,
        device,
        batch_size=tc.batch_size,
        seq_len=tc.seq_len,
        param_dtype=tc.param_dtype,
    )
    pp_schedule = build_pipeline_schedule(
        stage=pp_stage,
        n_microbatches=tc.grad_accum_steps,
        loss_fn=loss_fn,
        schedule=config.distributed.pp_schedule.value,
    )
    return model, PipelineBundle(rank=pp_rank, size=pp_size, schedule=pp_schedule)


def build_checkpoint_manager(
    config: JobConfig,
    runtime: RuntimeContext,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    pipeline: PipelineBundle | None,
) -> CheckpointManager:
    """Create the checkpoint manager, scoping DCP to the PP stage when needed.

    With PP, each stage has different parameters — DCP needs a group scoped
    to ranks within the same PP stage (all non-PP mesh dimensions), and each
    stage saves DCP shards to its own subdirectory to avoid file collisions.
    """
    ckpt_pg = None
    ckpt_pp_rank = None
    device_mesh = runtime.device_mesh
    if pipeline is not None and device_mesh is not None:
        ckpt_pp_rank = pipeline.rank
        non_pp_dims = [d for d in device_mesh.mesh_dim_names if d != "pp"]  # type: ignore[reportOptionalIterable]
        if len(non_pp_dims) == 1:
            ckpt_pg = device_mesh[non_pp_dims[0]].get_group()
        elif len(non_pp_dims) > 1:
            ckpt_pg = device_mesh[tuple(non_pp_dims)].get_group()
    return CheckpointManager(
        config.checkpoint,
        model,
        optimizer,
        process_group=ckpt_pg,
        pp_rank=ckpt_pp_rank,
    )


def restore_checkpoint(
    config: JobConfig,
    model: torch.nn.Module,
    scheduler: object,
    ckpt_mgr: CheckpointManager,
) -> tuple[int, int]:
    """Auto-resume from ``latest`` (or ``checkpoint.load_path``).

    Returns ``(step, tokens_seen)`` — ``(0, 0)`` when there is nothing to
    resume from. Also restores the metrics run ids, runs the MoT warm-start
    hook, and re-applies the effective freeze at the resumed step.
    """
    resume_path = resolve_resume_path(config.checkpoint.dir)
    if not resume_path and not config.checkpoint.load_path:
        return 0, 0

    vlm_cfg = config.vlm
    # On resume the expected freeze metadata reflects the post-transition
    # state at the saved step (effective_freeze handles step-boundary
    # transitions). Peek at the saved step via metadata.json before invoking
    # load() so the comparison uses the same step the checkpoint was written at.
    vlm_freeze_expected = None
    if config.is_vlm:
        assert vlm_cfg is not None  # narrowed by is_vlm
        probe_step = ckpt_mgr.peek_saved_step(str(resume_path) if resume_path else None) or 0
        vlm_freeze_expected = freeze_meta_at_step(probe_step, vlm_cfg)

    step, tokens_seen, ckpt_extra_loaded = ckpt_mgr.load(
        path=str(resume_path) if resume_path else None,
        scheduler=scheduler,
        vlm_freeze_expected=vlm_freeze_expected,
    )
    if ckpt_extra_loaded.get("wandb_run_id"):
        config.metrics.wandb_run_id = ckpt_extra_loaded["wandb_run_id"]
    if ckpt_extra_loaded.get("mlflow_run_id"):
        config.metrics.mlflow_run_id = ckpt_extra_loaded["mlflow_run_id"]

    # MoT warm-start: translate dense TransformerBlock weights from a
    # JD/text-only checkpoint into per-modality copies inside every
    # MoTBlock. Runs once at the start of training (resume_path is
    # None or step == 0); a real resume of an in-flight MoT run
    # already has the MoT-shaped state in the checkpoint and skips
    # this hook.
    if isinstance(vlm_cfg, MoTConfig) and vlm_cfg.mot_warm_start_from_text and step == 0:
        source = torch.load(vlm_cfg.mot_warm_start_path, map_location="cpu", weights_only=True)
        if isinstance(source, dict) and "model" in source:
            source = source["model"]
        mot_warm_start_from_text_stack(inner_transformer(model), source)
        logger.info(
            f"MoT warm-start: copied dense block weights from {vlm_cfg.mot_warm_start_path}"
        )

    # Apply effective freeze at the resumed step so requires_grad reflects
    # the post-transition state of any stages with start_step <= loaded_step.
    # Build-time apply only handles the base freeze list.
    if vlm_cfg is not None and vlm_cfg.freeze_schedule:
        valid_modules = set(vlm_cfg.module_patterns.keys())
        specs = effective_freeze(step, vlm_cfg.freeze, vlm_cfg.freeze_schedule, valid_modules)
        apply_freeze_specs(model, specs, vlm_cfg.module_patterns)
        logger.info(f"Resumed at step={step}; applied effective freeze ({len(specs)} specs)")

    return step, tokens_seen


def run_training(
    config: JobConfig,
    *,
    step_fn: StepFn | None = None,
    hooks: HookRunner | None = None,
) -> None:
    """Run a full training job: build every phase, run the loop, tear down.

    ``step_fn`` and ``hooks`` let an experiment own the step body or register
    hooks without copying the build phases. Both default to what
    ``scripts/train.py`` uses.
    """
    runtime = setup_distributed(config)
    # Bound before the try so the teardown can tell "never built" from "built".
    tracker: MetricsTracker | None = None
    finished = False
    try:
        shutdown_handler = ShutdownHandler(timeout_sec=config.train.shutdown_timeout_sec)
        shutdown_handler.register()
        nan_detector = NaNDetector(action="warn", max_consecutive=10)

        loss_fn = build_loss_fn(config.train)
        model, pipeline = build_model(config, runtime, loss_fn)

        optimizer = build_optimizer(model, config.optimizer)
        scheduler = build_scheduler(optimizer, config.scheduler, max_steps=config.train.max_steps)

        ckpt_mgr = build_checkpoint_manager(config, runtime, model, optimizer, pipeline)
        step, tokens_seen = restore_checkpoint(config, model, scheduler, ckpt_mgr)

        tracker = MetricsTracker(config, num_gpus=runtime.world_size)
        tracker.init_backends(config)

        prof = build_profiler(config.profiling, rank=runtime.rank)

        data = build_data_pipeline(config, runtime)
        # Apply any dataloader state stashed during load(). Runs after dataloader
        # construction because the loader's identity depends on phase scheduling
        # that load() restores. No-op when resuming without a prior dataloader
        # state or when the loader is not stateful (plain TorchDataLoader).
        if data.dataloader is not None:
            ckpt_mgr.apply_dataloader_state(data.dataloader)

        eval_dataloader = build_eval_dataloader(config, runtime)
        phases = build_phase_state(config, data, step)

        session = TrainingSession(
            config=config,
            runtime=runtime,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            step_fn=step_fn or select_step_fn(config),
            data=data,
            phases=phases,
            checkpointer=ckpt_mgr,
            tracker=tracker,
            hooks=hooks or HookRunner(),
            nan_detector=nan_detector,
            shutdown_handler=shutdown_handler,
            pipeline=pipeline,
            eval_dataloader=eval_dataloader,
            profiler=prof,
        )

        run_training_loop(session, step=step, tokens_seen=tokens_seen)
        finished = True
    finally:
        # As a library call this can raise and the caller can keep going, so
        # the metrics run must not outlive it. close() is rank-local: backends
        # are built on rank 0 only, so it is a no-op elsewhere.
        if tracker is not None:
            tracker.close()
        # destroy_process_group() is the cooperative teardown -- it blocks in
        # _wait_for_pending_works() and shuts backends down in a deliberate
        # order because ncclCommAbort has been collective in some NCCL
        # versions; torch ships _abort_process_group() as the separate
        # error-path API. Peers are still mid-collective while we unwind, so
        # on the failure path leave the group to the launcher, as main did.
        if finished:
            destroy_distributed()
