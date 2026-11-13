"""Library entry point for a KempnerForge training job.

``scripts/train.py`` is a thin CLI wrapper over :func:`run_training`: it
parses argv, loads the config, and calls in. Each phase below is a
separate function so an experiment can reuse the scaffold — or swap one
phase — without editing the shared script.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.checkpoint import is_weights_only
from kempnerforge.config.job import JobConfig
from kempnerforge.config.vlm import MoTConfig, VLMConfig
from kempnerforge.distributed.parallel import (
    apply_ac,
    apply_float8,
    apply_fsdp2,
    autocast_dtype,
    build_parallel_model,
    default_mp_policy,
)
from kempnerforge.distributed.setup import destroy_distributed
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel
from kempnerforge.distributed.utils import get_dp_info
from kempnerforge.metrics.logger import get_logger
from kempnerforge.metrics.tracker import MetricsTracker
from kempnerforge.model.masking import use_dynamic_flex_shapes
from kempnerforge.model.mot import mot_warm_start_from_text_stack
from kempnerforge.model.transformer import Transformer
from kempnerforge.model.vlm import inner_transformer, packed_tokens_per_image
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
            frames_per_clip=config.views_per_sample,
            ac_mode=tc.activation_checkpointing,
            mp_policy=mp_policy,
            param_dtype=tc.master_dtype,
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
        raise RuntimeError("pipeline parallelism requires a device mesh")
    pp_rank = get_pp_rank(device_mesh)
    pp_size = get_pp_size(device_mesh)

    tp_enabled_pp = "tp" in device_mesh.mesh_dim_names  # type: ignore[reportOperatorIssue]
    if tp_enabled_pp:
        with torch.device("meta"):
            stage_mod = cast("Transformer", build_stage_module(config.model, pp_rank, pp_size))
        apply_tensor_parallel(stage_mod, device_mesh)
        if tc.is_fp8:
            apply_float8(stage_mod)
        apply_ac(stage_mod, tc.activation_checkpointing)
        apply_fsdp2(stage_mod, device_mesh, mp_policy=mp_policy)
        stage_mod.to_empty(device=device)
        stage_mod.init_weights_and_freqs()
        stage_mod.to(dtype=tc.master_dtype)
    else:
        stage_mod = cast(
            "Transformer",
            build_stage_module(config.model, pp_rank, pp_size).to(
                device=device, dtype=tc.master_dtype
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

    amp_dtype = autocast_dtype(stage_mod, tc.master_dtype, tc.param_dtype)
    shape_inference_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else contextlib.nullcontext()
    )
    with shape_inference_ctx:
        pp_stage = build_pipeline_stage(
            model,  # type: ignore[arg-type]
            device_mesh,
            device,
            batch_size=tc.batch_size,
            seq_len=tc.seq_len,
            param_dtype=tc.param_dtype if amp_dtype is None else tc.master_dtype,
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
    The manager records each rank's data-parallel coordinates, the partition
    ``build_data_pipeline`` reads, with the per-rank loader/RNG state.
    """
    ckpt_pg = None
    ckpt_pp_rank = None
    device_mesh = runtime.device_mesh
    if pipeline is not None and device_mesh is not None:
        from kempnerforge.distributed.pipeline_parallel import build_stage_group

        ckpt_pp_rank = pipeline.rank
        ckpt_pg = build_stage_group(device_mesh)
    return CheckpointManager(
        config.checkpoint,
        model,
        optimizer,
        process_group=ckpt_pg,
        pp_rank=ckpt_pp_rank,
        dp_info=get_dp_info(device_mesh),
        weight_decay_scope=config.optimizer.weight_decay_scope,
        lr_multipliers=config.optimizer.lr_multipliers,
        module_patterns=config.vlm.module_patterns if config.vlm is not None else None,
    )


def _mot_text_warm_start(
    vlm_cfg: VLMConfig | None,
    model: torch.nn.Module,
    agree: Callable[[BaseException | None, str], None],
) -> None:
    """Copy the dense block weights of ``mot_warm_start_path`` into a MoT model.

    ``agree`` is the checkpoint manager's rank agreement: every rank reports
    whether it read the source before any rank enters the copy's DTensor
    collectives, so a rank that cannot read the file fails the run on every rank
    instead of leaving the others waiting until the process-group timeout.
    """
    if not isinstance(vlm_cfg, MoTConfig) or not vlm_cfg.mot_warm_start_from_text:
        return
    source = {}
    error = None
    try:
        source = torch.load(vlm_cfg.mot_warm_start_path, map_location="cpu", weights_only=True)
        if isinstance(source, dict) and "model" in source:
            source = source["model"]
    except Exception as exc:
        error = exc
    agree(error, f"MoT warm-start: reading {vlm_cfg.mot_warm_start_path}")
    mot_warm_start_from_text_stack(inner_transformer(model), source)
    logger.info(f"MoT warm-start: copied dense block weights from {vlm_cfg.mot_warm_start_path}")


def _unused_load_path_message(config: JobConfig, resume_path: Path) -> str:
    """The warning that a resume from ``checkpoint.dir`` wins over ``checkpoint.load_path``."""
    ckpt = config.checkpoint
    message = (
        f"checkpoint.load_path={ckpt.load_path} is not used: checkpoint.dir={ckpt.dir} already "
        f"has a checkpoint, and resuming {resume_path} takes precedence so that a requeued run "
        f"continues where it stopped."
    )
    if ckpt.exclude_from_loading:
        message += (
            f" checkpoint.exclude_from_loading={ckpt.exclude_from_loading} is not applied "
            f"either: a resume restores full state."
        )
    return message + (
        " To start from load_path instead, for example to roll back to an earlier step, "
        "set checkpoint.dir to a new, empty directory."
    )


def _load_path_is_the_resume(config: JobConfig, resume_path: Path) -> bool:
    """Whether ``checkpoint.load_path`` names the resumed checkpoint and excludes nothing.

    The resume then loads what ``load_path`` asks for, so nothing in it goes unused. Both paths
    go through ``os.path.realpath``, so a relative path or one through ``latest`` compares by
    its target, and through ``str`` first: the config loader keeps a bare-number ``load_path``
    such as ``--checkpoint.load_path=5000`` as an int. Unlike ``Path.resolve``,
    ``os.path.realpath`` does not raise on a symlink loop: only rank 0 runs this, and a raise
    here would leave the other ranks waiting in the load's broadcast.
    """
    ckpt = config.checkpoint
    if not ckpt.load_path or ckpt.exclude_from_loading:
        return False
    return os.path.realpath(str(ckpt.load_path)) == os.path.realpath(str(resume_path))


def restore_checkpoint(
    config: JobConfig,
    model: torch.nn.Module,
    scheduler: object,
    ckpt_mgr: CheckpointManager,
) -> tuple[int, int]:
    """Auto-resume from ``latest`` (or ``checkpoint.load_path``).

    Returns ``(step, tokens_seen)`` — ``(0, 0)`` when there is nothing to
    resume from. Also restores the metrics run ids, runs the MoT text warm
    start when no checkpoint supplied the model weights and the run starts at
    step 0, and re-applies the effective freeze at the resumed step.

    A checkpoint in ``checkpoint.dir`` takes precedence over ``checkpoint.load_path``, so a
    requeued warm start resumes rather than starting over; rank 0 then warns that
    ``load_path`` is not used, naming both paths, unless ``load_path`` names the resumed
    checkpoint and excludes nothing.
    """
    num_stages = config.distributed.pp if config.distributed.pp > 1 else None
    if dist.is_initialized():
        chosen = [
            str(resolve_resume_path(config.checkpoint.dir, num_stages) or "")
            if dist.get_rank() == 0
            else ""
        ]
        dist.broadcast_object_list(chosen, src=0)
        resume_path = Path(chosen[0]) if chosen[0] else None
    else:
        resume_path = resolve_resume_path(config.checkpoint.dir, num_stages)
    vlm_cfg = config.vlm
    if not resume_path and not config.checkpoint.load_path:
        _mot_text_warm_start(vlm_cfg, model, ckpt_mgr.agree)
        return 0, 0
    rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    if (
        resume_path
        and config.checkpoint.load_path
        and rank_zero
        and not _load_path_is_the_resume(config, resume_path)
    ):
        logger.warning(_unused_load_path_message(config, resume_path))

    warm_start_exclude = (
        (config.checkpoint.exclude_from_loading or None) if resume_path is None else None
    )
    weights_only = is_weights_only(warm_start_exclude)
    vlm_freeze_expected = None
    if config.is_vlm and not weights_only:
        assert vlm_cfg is not None
        probe_step = ckpt_mgr.peek_saved_step(str(resume_path) if resume_path else None) or 0
        vlm_freeze_expected = freeze_meta_at_step(probe_step, vlm_cfg)

    step, tokens_seen, ckpt_extra_loaded = ckpt_mgr.load(
        path=str(resume_path) if resume_path else None,
        scheduler=scheduler,
        exclude_keys=warm_start_exclude,
        vlm_freeze_expected=vlm_freeze_expected,
        exact_resume=resume_path is not None,
    )
    if ckpt_extra_loaded.get("wandb_run_id"):
        config.metrics.wandb_run_id = ckpt_extra_loaded["wandb_run_id"]
    if ckpt_extra_loaded.get("mlflow_run_id"):
        config.metrics.mlflow_run_id = ckpt_extra_loaded["mlflow_run_id"]

    if warm_start_exclude is not None and "model" in warm_start_exclude:
        _mot_text_warm_start(vlm_cfg, model, ckpt_mgr.agree)

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
    ``scripts/train.py`` uses. A custom ``step_fn`` must run its forward and
    loss under ``session.autocast()``, which casts fp32 master weights to the
    compute dtype when no FSDP2 wrap does (one GPU, a pp-only mesh).
    """
    runtime = setup_distributed(config)
    tracker: MetricsTracker | None = None
    finished = False
    try:
        shutdown_handler = ShutdownHandler(timeout_sec=config.train.shutdown_timeout_sec)
        shutdown_handler.register()
        nan_detector = NaNDetector(action="warn", max_consecutive=10)

        loss_fn = build_loss_fn(config.train)
        use_dynamic_flex_shapes(config.is_vlm and config.data.pack_sequences)
        model, pipeline = build_model(config, runtime, loss_fn)
        amp_dtype = autocast_dtype(model, config.train.master_dtype, config.train.param_dtype)
        if amp_dtype is not None:
            logger.info(
                f"No FSDP2 wrap casts the {config.train.master_dtype} master weights: "
                f"forward passes run under torch.autocast({amp_dtype})"
            )

        optimizer = build_optimizer(
            model,
            config.optimizer,
            module_patterns=config.vlm.module_patterns if config.vlm is not None else None,
        )
        scheduler = build_scheduler(optimizer, config.scheduler, max_steps=config.train.max_steps)

        ckpt_mgr = build_checkpoint_manager(config, runtime, model, optimizer, pipeline)
        step, tokens_seen = restore_checkpoint(config, model, scheduler, ckpt_mgr)

        tracker = MetricsTracker(config, num_gpus=runtime.world_size)
        tracker.init_backends(config)
        ckpt_mgr.set_tracking_source(tracker.tracking_entries)
        ckpt_mgr.set_checkpoint_registrar(tracker)

        prof = build_profiler(config.profiling, rank=runtime.rank)

        tokens_per_image = (
            packed_tokens_per_image(model) if config.is_vlm and config.data.pack_sequences else None
        )
        data = build_data_pipeline(config, runtime, tokens_per_image=tokens_per_image)
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
            autocast_dtype=amp_dtype,
        )

        run_training_loop(session, step=step, tokens_seen=tokens_seen)
        finished = True
    finally:
        use_dynamic_flex_shapes(False)
        if tracker is not None:
            tracker.close()
        if finished:
            destroy_distributed()
