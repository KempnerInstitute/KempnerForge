"""The training step loop and the three step bodies it dispatches to.

``run_training_loop`` owns everything that repeats: the step body, NaN
handling, optimizer/scheduler advance, freeze and phase transitions,
metrics, eval, checkpointing, and shutdown. The step body itself is a
plain callable (``StepFn``) chosen once by :func:`select_step_fn`, so a
model family with a different forward contract is a new function rather
than another branch inside the loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.job import JobConfig
from kempnerforge.distributed.utils import clip_grad_norm_
from kempnerforge.metrics.logger import get_logger
from kempnerforge.metrics.tracker import MetricsTracker
from kempnerforge.model.vlm import inner_transformer
from kempnerforge.profiling.profiler import print_profiler_summary
from kempnerforge.resilience.health import NaNDetector, check_nccl_health
from kempnerforge.resilience.signal_handler import ShutdownHandler
from kempnerforge.training.data_pipeline import DataPipeline, PhaseState, advance_phases
from kempnerforge.training.eval import run_eval
from kempnerforge.training.freeze import apply_freeze_specs, freeze_meta_at_step
from kempnerforge.training.grad import maybe_no_sync
from kempnerforge.training.hooks import HookRunner, StepContext
from kempnerforge.training.runtime import PipelineBundle, RuntimeContext, pp_group

logger = get_logger(__name__)

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class BatchStream:
    """The training dataloader plus its live iterator.

    Wraps the epoch-boundary restart and the "no dataloader configured"
    case so the step bodies don't each re-implement them. The loader is read
    off the pipeline on every access, so swapping ``pipeline.dataloader``
    takes effect on the next step instead of leaving a stale snapshot.
    """

    def __init__(self, pipeline: DataPipeline) -> None:
        self.pipeline = pipeline
        self._iter: Any = None
        self._source: Any = None

    @property
    def dataloader(self) -> Any:
        return self.pipeline.dataloader

    @property
    def has_data(self) -> bool:
        return self.dataloader is not None

    def ensure_started(self) -> None:
        """Materialize the iterator, if it isn't already, for the loader in use.

        ``next_batch`` does this itself; the loop calls it up front so building
        an iterator (which spawns dataloader workers) stays outside the region
        ``MetricsTracker`` times.
        """
        loader = self.dataloader
        if loader is not None and (self._iter is None or self._source is not loader):
            self._iter = iter(loader)
            self._source = loader

    def reset(self) -> None:
        """Drop the iterator so the next step takes a fresh one from the loader.

        A ``StatefulDataLoader`` re-applies its recorded skip, so this picks up
        inside the current epoch rather than restarting it.
        """
        self._iter = None

    def next_batch(self) -> dict[str, torch.Tensor]:
        if self.dataloader is None:
            raise RuntimeError("BatchStream has no dataloader; check has_data first")
        self.ensure_started()
        try:
            return next(self._iter)
        except StopIteration:
            pass
        # Epoch boundary: take a fresh iterator. A second StopIteration means
        # this rank's shard is empty, which would otherwise escape as a bare
        # StopIteration and leave the other ranks blocked in the next
        # collective until the process-group timeout.
        self._iter = iter(self.dataloader)
        self._source = self.dataloader
        try:
            return next(self._iter)
        except StopIteration:
            raise RuntimeError(
                "dataloader yielded no batches for this rank: the dataset is "
                "smaller than batch_size x dp_size, a streaming shard is "
                "exhausted, or a mixture phase weighted this rank's only "
                "source to zero"
            ) from None


@dataclass
class StepResult:
    """What one training step reports back to the loop.

    The per-dataset fields stay empty for step bodies that don't produce
    them; ``text_tokens`` stays None, which a VLM run treats as an error
    rather than logging a zero.
    """

    loss: float
    grad_norm: float
    text_tokens: int | None = None
    dataset_token_counts: dict[str, int] = field(default_factory=dict)
    dataset_loss_sums: dict[str, float] = field(default_factory=dict)
    dataset_loss_counts: dict[str, int] = field(default_factory=dict)


StepFn = Callable[["TrainingSession", int], StepResult]


@dataclass
class TrainingSession:
    """Everything one training run needs after the build phase."""

    config: JobConfig
    runtime: RuntimeContext
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    loss_fn: LossFn
    step_fn: StepFn
    data: DataPipeline
    phases: PhaseState
    checkpointer: CheckpointManager
    tracker: MetricsTracker
    hooks: HookRunner
    nan_detector: NaNDetector
    shutdown_handler: ShutdownHandler
    pipeline: PipelineBundle | None = None
    eval_dataloader: Any | None = None
    profiler: Any | None = None
    _batch_stream: BatchStream | None = field(default=None, init=False, repr=False)

    @property
    def batches(self) -> BatchStream:
        """The live batch iterator. Rebuilt if ``data`` is replaced wholesale."""
        if self._batch_stream is None or self._batch_stream.pipeline is not self.data:
            self._batch_stream = BatchStream(self.data)
        return self._batch_stream


# ---------------------------------------------------------------------------
# Step bodies
# ---------------------------------------------------------------------------


def _clip_grads(model: torch.nn.Module, max_norm: float) -> float:
    grad_norm = clip_grad_norm_(model, max_norm)
    return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm


def _add_moe_losses(loss: torch.Tensor, model: torch.nn.Module, mc: Any) -> torch.Tensor:
    """MoE auxiliary + router z-loss terms (only called when ``mc.is_moe``)."""
    aux_loss = inner_transformer(model).get_moe_aux_loss()  # type: ignore[attr-defined]
    loss = loss + mc.moe_aux_loss_weight * aux_loss
    if mc.moe_router_z_loss_weight > 0:
        z = inner_transformer(model).get_moe_router_z_loss()  # type: ignore[attr-defined]
        loss = loss + mc.moe_router_z_loss_weight * z
    return loss


def pipeline_step(session: TrainingSession, step: int) -> StepResult:
    """PP step: collect microbatches, hand them to the schedule, broadcast loss.

    ``step`` is unused — MoE (the only step-dependent branch) is rejected
    with PP in ``JobConfig.validate``.
    """
    config = session.config
    tc, mc = config.train, config.model
    device = session.runtime.device
    model = session.model
    batches = session.batches
    pipeline = session.pipeline
    if pipeline is None:  # select_step_fn only picks this body when pp > 1
        raise RuntimeError("pipeline_step requires TrainingSession.pipeline")

    # Collect microbatches into a full batch for the schedule.
    # schedule.step() splits along dim 0 into n_microbatches.
    input_ids_list, labels_list = [], []
    for _ in range(tc.grad_accum_steps):
        if batches.has_data:
            batch = batches.next_batch()
            input_ids_list.append(batch["input_ids"].to(device))
            labels_list.append(batch["labels"].to(device))
        else:
            input_ids_list.append(
                torch.randint(0, mc.vocab_size, (tc.batch_size, tc.seq_len), device=device)
            )
            labels_list.append(
                torch.randint(0, mc.vocab_size, (tc.batch_size, tc.seq_len), device=device)
            )

    full_input = torch.cat(input_ids_list, dim=0)
    full_labels = torch.cat(labels_list, dim=0)

    # The schedule handles forward/backward for all microbatches.
    # First stage needs input; last stage needs target for loss.
    # schedule.step() returns model output; losses are collected via the
    # losses= output parameter (list populated by the schedule).
    is_first = pipeline.rank == 0
    is_last = pipeline.rank == pipeline.size - 1
    pp_losses: list[torch.Tensor] = []

    if is_first:
        pipeline.schedule.step(full_input, target=full_labels, losses=pp_losses)
    elif is_last:
        pipeline.schedule.step(target=full_labels, losses=pp_losses)
    else:
        pipeline.schedule.step()

    # Loss is only meaningful on the last stage
    if is_last and pp_losses:
        avg_loss = sum(loss.item() for loss in pp_losses) / len(pp_losses)
    else:
        avg_loss = 0.0

    grad_norm_val = _clip_grads(model, tc.grad_clip_norm)

    # Broadcast loss and grad_norm from last PP stage to all PP stages
    loss_tensor = torch.tensor([avg_loss, grad_norm_val], device=device)
    dist.broadcast(loss_tensor, group_src=pipeline.size - 1, group=pp_group(session.runtime))
    return StepResult(loss=loss_tensor[0].item(), grad_norm=loss_tensor[1].item())


def vlm_step(session: TrainingSession, step: int) -> StepResult:
    """VLM step (no PP): pixel_values + input_ids forward, with a text-token count."""
    config = session.config
    tc, mc = config.train, config.model
    device = session.runtime.device
    model = session.model
    batches = session.batches

    total_loss = 0.0
    total_text_tokens = 0

    for micro_step in range(tc.grad_accum_steps):
        if not batches.has_data:
            raise RuntimeError(
                "VLM training requires a real dataloader; synthetic fallback "
                "(randint) does not produce pixel_values"
            )
        batch = batches.next_batch()
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        # Video batches carry a per-frame validity mask; image batches do not.
        frame_mask = batch["frame_mask"].to(device) if "frame_mask" in batch else None

        with maybe_no_sync(model, micro_step, tc.grad_accum_steps):
            if mc.is_moe:
                inner_transformer(model).set_moe_step(step, tc.max_steps)  # type: ignore[attr-defined]
            logits, labels_out = model(pixel_values, input_ids, labels, frame_mask=frame_mask)
            loss = session.loss_fn(logits, labels_out)

            total_text_tokens += int((labels_out != -100).sum().item())

            if mc.is_moe:
                loss = _add_moe_losses(loss, model, mc)

            scaled_loss = loss / tc.grad_accum_steps
            scaled_loss.backward()
            total_loss += loss.item()

    return StepResult(
        loss=total_loss / tc.grad_accum_steps,
        grad_norm=_clip_grads(model, tc.grad_clip_norm),
        text_tokens=total_text_tokens,
    )


def text_step(session: TrainingSession, step: int) -> StepResult:
    """Standard step (no PP, text-only), with optional per-dataset mixture metrics."""
    config = session.config
    tc, mc = config.train, config.model
    device = session.runtime.device
    model = session.model
    batches = session.batches
    mixture_dataset = session.data.mixture_dataset

    total_loss = 0.0
    ds_token_counts: dict[str, int] = {}
    ds_loss_sums: dict[str, float] = {}
    ds_loss_counts: dict[str, int] = {}

    for micro_step in range(tc.grad_accum_steps):
        batch: dict[str, torch.Tensor] = {}
        if batches.has_data:
            batch = batches.next_batch()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            doc_ids = batch["doc_ids"].to(device) if "doc_ids" in batch else None
        else:
            input_ids = torch.randint(0, mc.vocab_size, (tc.batch_size, tc.seq_len), device=device)
            labels = torch.randint(0, mc.vocab_size, (tc.batch_size, tc.seq_len), device=device)
            doc_ids = None

        with maybe_no_sync(model, micro_step, tc.grad_accum_steps):
            if mc.is_moe:
                inner_transformer(model).set_moe_step(step, tc.max_steps)  # type: ignore[attr-defined]
            logits = model(input_ids, doc_ids=doc_ids)
            loss = session.loss_fn(logits, labels)

            # Per-dataset metrics (before backward, while logits are fresh)
            if mixture_dataset is not None and "dataset_idx" in batch:
                ds_idx = batch["dataset_idx"]
                with torch.no_grad():
                    for i, name in enumerate(mixture_dataset.dataset_names):
                        mask = ds_idx == i
                        count = int(mask.sum().item())
                        if count > 0:
                            ds_token_counts[name] = (
                                ds_token_counts.get(name, 0) + count * tc.seq_len
                            )
                            ds_l = torch.nn.functional.cross_entropy(
                                logits[mask].reshape(-1, logits.size(-1)),
                                labels[mask].reshape(-1),
                                ignore_index=-100,
                            ).item()
                            ds_loss_sums[name] = ds_loss_sums.get(name, 0) + ds_l
                            ds_loss_counts[name] = ds_loss_counts.get(name, 0) + 1

            if mc.is_moe:
                loss = _add_moe_losses(loss, model, mc)

            scaled_loss = loss / tc.grad_accum_steps
            scaled_loss.backward()
            total_loss += loss.item()

    return StepResult(
        loss=total_loss / tc.grad_accum_steps,
        grad_norm=_clip_grads(model, tc.grad_clip_norm),
        dataset_token_counts=ds_token_counts,
        dataset_loss_sums=ds_loss_sums,
        dataset_loss_counts=ds_loss_counts,
    )


def select_step_fn(config: JobConfig) -> StepFn:
    """Pick the step body for this job: PP wins, then VLM, then text-only."""
    if config.distributed.pp > 1:
        return pipeline_step
    if config.is_vlm:
        return vlm_step
    return text_step


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def checkpoint_extra(config: JobConfig, step: int, phases: PhaseState) -> dict:
    """Metadata saved alongside every checkpoint so a resume is exact.

    ``vlm_freeze`` reflects the post-transition state when a FreezeStage
    has fired (see ``freeze_meta_at_step``).
    """
    extra: dict = {"phase_idx": phases.next_idx} if phases.phases else {}
    if config.metrics.wandb_run_id:
        extra["wandb_run_id"] = config.metrics.wandb_run_id
    if config.metrics.mlflow_run_id:
        extra["mlflow_run_id"] = config.metrics.mlflow_run_id
    if config.is_vlm:
        assert config.vlm is not None  # narrowed by is_vlm
        extra["vlm_freeze"] = freeze_meta_at_step(step, config.vlm)
    return extra


def _log_periodic_metrics(session: TrainingSession, result: StepResult, step: int) -> None:
    """MoE / VLM-text-token / per-dataset series, on the main metrics interval."""
    config = session.config
    mc = config.model
    tracker = session.tracker
    model = session.model

    if mc.is_moe:
        inner = inner_transformer(model)
        moe_metrics = {"moe/aux_loss": inner.get_moe_aux_loss().item()}  # type: ignore[attr-defined]
        moe_metrics["moe/router_z_loss"] = inner.get_moe_router_z_loss().item()  # type: ignore[attr-defined]
        expert_counts = inner.get_expert_counts()  # type: ignore[attr-defined]
        if expert_counts:
            all_counts = torch.stack(list(expert_counts.values())).float()
            moe_metrics["moe/expert_balance"] = (all_counts.min() / all_counts.max()).item()
        tracker.log_eval(moe_metrics, step)

    # VLM per-step text-token count (excludes image prefix, -100 pad, and
    # masked prompt tokens). Logged separately from tokens_in_step which
    # still reports sequence positions processed. The counter is DP-local
    # on each rank, so all-reduce it to the global text-token count
    # before logging.
    if config.is_vlm:
        if result.text_tokens is None:
            raise RuntimeError(
                "VLM run: the step body reported no text-token count, so "
                "data/text_tokens_trained would be logged as zero"
            )
        global_text_tokens = result.text_tokens
        if dist.is_initialized():
            t = torch.tensor([result.text_tokens], device=session.runtime.device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            global_text_tokens = int(t.item())
        tracker.log_eval({"data/text_tokens_trained": float(global_text_tokens)}, step)

    if session.data.mixture_dataset is not None and result.dataset_loss_sums:
        ds_metrics: dict[str, float] = {}
        for name in result.dataset_loss_sums:
            ds_metrics[f"loss/{name}"] = (
                result.dataset_loss_sums[name] / result.dataset_loss_counts[name]
            )
        for name, count in result.dataset_token_counts.items():
            ds_metrics[f"data/{name}/tokens"] = float(count)
        tracker.log_eval(ds_metrics, step)


def _apply_freeze_stages(session: TrainingSession, step: int) -> None:
    """FreezeStage hook: apply any stage whose start_step matches this boundary.

    AdamW + set_to_none=True default skips frozen params entirely (no SGD
    step, no weight decay), so mutating requires_grad mid-training is a
    clean no-op for newly-frozen params and re-enables gradient flow for
    newly-unfrozen ones.

    Async-save fence: drain any in-flight save FIRST so that its
    metadata.json — which records the pre-transition spec — lands before we
    flip requires_grad. Otherwise a save started at step S-1 could write
    metadata after the transition, attaching the post-transition spec to
    pre-transition shards.
    """
    vlm_cfg = session.config.vlm
    if not session.config.is_vlm or vlm_cfg is None or not vlm_cfg.freeze_schedule:
        return
    pending_stages = [s for s in vlm_cfg.freeze_schedule if s.start_step == step]
    if not pending_stages:
        return
    session.checkpointer.flush_pending_save()
    for stage in pending_stages:
        flipped = apply_freeze_specs(session.model, stage.specs, vlm_cfg.module_patterns)
        logger.info(f"FreezeStage at step={step}: applied {flipped}")


def run_training_loop(
    session: TrainingSession, *, step: int = 0, tokens_seen: int = 0
) -> tuple[int, int]:
    """Run the step loop from ``step`` to ``train.max_steps``.

    Returns the final ``(step, tokens_seen)``. Drains any pending async
    checkpoint before returning so the caller can tear down the process
    group safely.
    """
    config = session.config
    tc = config.train
    eval_config = config.eval
    runtime = session.runtime
    model = session.model
    optimizer = session.optimizer
    scheduler = session.scheduler
    ckpt_mgr = session.checkpointer
    tracker = session.tracker
    hooks = session.hooks
    data = session.data
    phases = session.phases
    batches = session.batches
    prof = session.profiler

    logger.info(
        f"Starting training: step={step}, max_steps={tc.max_steps}, "
        f"batch_size={tc.batch_size}, grad_accum={tc.grad_accum_steps}, "
        f"world_size={runtime.world_size}"
    )
    if phases.phases:
        logger.info(f"Phase scheduling: {len(phases.phases)} phase(s) configured")

    model.train()
    hooks.on_train_begin(config)

    if prof is not None:
        prof.start()

    # Capture the initial weights (step 0) on fresh start when the
    # dyn_ckpt_window covers step 0 -- the per-step save gate only runs
    # after a training step completes, so without this the random init
    # is never persisted. Skipped on resume (step > 0).
    if step == 0 and config.checkpoint.is_dynamic_milestone(0):
        ckpt_mgr.save(
            step=0,
            tokens_seen=0,
            scheduler=scheduler,
            dataloader=data.dataloader,
            extra=checkpoint_extra(config, 0, phases),
        )
        hooks.on_checkpoint_save(0, config.checkpoint.dir)

    completed_normally = False
    # Only read when completed_normally, i.e. after at least one full
    # iteration has assigned it.
    ckpt_extra: dict = {}

    try:
        while step < tc.max_steps:
            # Refresh data iterator at start / epoch boundary
            batches.ensure_started()

            tracker.start_step()
            result = session.step_fn(session, step)

            # NaN check
            if not session.nan_detector.check_loss(result.loss, step):
                optimizer.zero_grad()
                if session.nan_detector.should_rollback:
                    logger.error("Too many consecutive NaNs — stopping")
                    break
                step += 1
                continue

            # Optimizer step
            optimizer.step()
            scheduler.step()

            # Phase LR scaling (applied after scheduler computes base LR)
            if phases.lr_scale != 1.0:
                for pg in optimizer.param_groups:
                    pg["lr"] *= phases.lr_scale

            optimizer.zero_grad()

            step += 1
            tokens_in_step = tc.batch_size * tc.seq_len * tc.grad_accum_steps * data.dp_size
            tokens_seen += tokens_in_step

            _apply_freeze_stages(session, step)

            # Phase transition check — a fired phase forces a data iterator
            # refresh so the new sampler weights take effect.
            if advance_phases(phases, data, step):
                batches.reset()

            # Metrics (report LR after phase scaling)
            current_lr = optimizer.param_groups[0]["lr"]
            step_metrics = tracker.end_step(
                step=step,
                loss=result.loss,
                grad_norm=result.grad_norm,
                lr=current_lr,
                tokens_in_step=tokens_in_step,
            )

            hooks.on_step_end(
                StepContext(
                    step=step,
                    loss=result.loss,
                    grad_norm=result.grad_norm,
                    lr=current_lr,
                    tokens_seen=tokens_seen,
                    model=model,
                    optimizer=optimizer,
                )
            )

            if step_metrics is not None:
                _log_periodic_metrics(session, result, step)

            # Periodic NCCL health check
            if (
                tc.nccl_health_check_interval > 0
                and step % tc.nccl_health_check_interval == 0
                and not check_nccl_health()
            ):
                logger.error(f"NCCL health check failed at step {step} — stopping")
                break

            # Eval
            if (
                eval_config.enabled
                and session.eval_dataloader is not None
                and step % eval_config.interval == 0
            ):
                pipeline = session.pipeline
                eval_metrics = run_eval(
                    model,
                    session.eval_dataloader,
                    session.loss_fn,
                    runtime.device,
                    eval_config.steps,
                    pp_schedule=pipeline.schedule if pipeline is not None else None,
                    pp_rank=pipeline.rank if pipeline is not None else None,
                    pp_size=pipeline.size if pipeline is not None else None,
                    pp_group=pp_group(runtime) if pipeline is not None else None,
                )
                tracker.log_eval(eval_metrics, step)
                hooks.on_eval_end(eval_metrics, step)

            # Advance profiler schedule
            if prof is not None:
                prof.step()

            # Checkpoint
            ckpt_extra = checkpoint_extra(config, step, phases)
            if config.checkpoint.should_save(step):
                ckpt_mgr.save(
                    step=step,
                    tokens_seen=tokens_seen,
                    scheduler=scheduler,
                    dataloader=data.dataloader,
                    extra=ckpt_extra,
                )
                hooks.on_checkpoint_save(step, config.checkpoint.dir)

            # Graceful shutdown
            if session.shutdown_handler.should_shutdown():
                logger.warning(f"Shutdown requested at step {step} — saving emergency checkpoint")
                ckpt_mgr.save(
                    step=step,
                    tokens_seen=tokens_seen,
                    scheduler=scheduler,
                    dataloader=data.dataloader,
                    extra=ckpt_extra,
                )
                session.shutdown_handler.finish()
                break

            # Clean-completion marker for the unconditional final save after the
            # loop. Only reached when training completes without any errors, e.g.,
            # no NaN/NCCL/shutdown breaks. If a run encounters a NaN, the last step
            # is intentionally *not* saved because the actual model state would be
            # `max_steps - 1`, not `max_steps`.
            if step >= tc.max_steps:
                completed_normally = True

    finally:
        if prof is not None:
            prof.stop()
            if runtime.rank == 0:
                print_profiler_summary(prof, trace_dir=config.profiling.trace_dir)

        if completed_normally and not config.checkpoint.should_save(step):
            ckpt_mgr.save(
                step=step,
                tokens_seen=tokens_seen,
                scheduler=scheduler,
                dataloader=data.dataloader,
                extra=ckpt_extra,
            )
            hooks.on_checkpoint_save(step, config.checkpoint.dir)

        # Flush any pending async checkpoint before tearing down process group.
        # In `finally` so a raise mid-loop cannot leave a half-written save.
        ckpt_mgr.wait()

    logger.info(f"Training complete: {step} steps, {tokens_seen:,} tokens")
    hooks.on_train_end(step, tokens_seen)
    return step, tokens_seen
