#!/usr/bin/env python3
"""KempnerForge training entry point.

Usage:
    # Single GPU
    uv run python scripts/train.py configs/train/debug.toml

    # Multi-GPU (single node, via torchrun)
    uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/7b.toml

    # Multi-node (via SLURM srun — see scripts/slurm/multinode.sh)
    # srun launches one process per GPU; MASTER_ADDR/MASTER_PORT are resolved
    # automatically from SLURM env vars by init_distributed().
    srun uv run python scripts/train.py configs/train/7b.toml

    # With overrides
    uv run python scripts/train.py configs/train/7b.toml \
        --train.max_steps=1000 --optimizer.lr=1e-4
"""

from __future__ import annotations

import math
import sys

import torch
import torch.distributed as dist

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.loader import load_config
from kempnerforge.config.registry import registry
from kempnerforge.data.dataloader import StatefulDataLoader
from kempnerforge.data.dataset import MemoryMappedDataset
from kempnerforge.data.sampler import DistributedSampler
from kempnerforge.distributed.parallel import (
    apply_ac,
    apply_fsdp2,
    build_parallel_model,
    default_mp_policy,
)
from kempnerforge.distributed.setup import destroy_distributed, get_world_info, init_distributed
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel
from kempnerforge.distributed.utils import clip_grad_norm_, get_dp_info
from kempnerforge.metrics.logger import get_logger
from kempnerforge.metrics.tracker import MetricsTracker
from kempnerforge.profiling.profiler import build_profiler, print_profiler_summary
from kempnerforge.resilience.elastic import log_job_info, resolve_resume_path
from kempnerforge.resilience.health import NaNDetector
from kempnerforge.resilience.signal_handler import ShutdownHandler
from kempnerforge.training.grad import maybe_no_sync
from kempnerforge.training.loss import cross_entropy_loss as _  # noqa: F401 — triggers registration
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.scheduler import build_scheduler

logger = get_logger(__name__)


@torch.no_grad()
def eval_step(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    loss_fn: callable,
    device: torch.device,
    eval_steps: int,
    *,
    pp_schedule=None,
    pp_rank: int | None = None,
    pp_size: int | None = None,
    pp_group=None,
) -> dict[str, float]:
    """Run evaluation and return metrics."""
    model.eval()

    if pp_schedule is not None:
        # --- PP eval path ---
        input_ids_list, labels_list = [], []
        eval_iter = iter(eval_dataloader)
        for _ in range(eval_steps):
            try:
                batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_dataloader)
                batch = next(eval_iter)
            input_ids_list.append(batch["input_ids"].to(device))
            labels_list.append(batch["labels"].to(device))

        full_input = torch.cat(input_ids_list, dim=0)
        full_labels = torch.cat(labels_list, dim=0)

        is_first = pp_rank == 0
        is_last = pp_rank == pp_size - 1
        pp_losses: list[torch.Tensor] = []

        if is_first:
            pp_schedule.step(full_input, target=full_labels, losses=pp_losses)
        elif is_last:
            pp_schedule.step(target=full_labels, losses=pp_losses)
        else:
            pp_schedule.step()

        if is_last and pp_losses:
            avg_loss = sum(loss.item() for loss in pp_losses) / len(pp_losses)
        else:
            avg_loss = 0.0

        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.broadcast(loss_tensor, group_src=pp_size - 1, group=pp_group)
        avg_loss = loss_tensor[0].item()
    else:
        # --- Standard eval path ---
        total_loss = 0.0
        eval_iter = iter(eval_dataloader)
        for _ in range(eval_steps):
            try:
                batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_dataloader)
                batch = next(eval_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

        avg_loss = total_loss / eval_steps

    model.train()
    return {"eval/loss": avg_loss, "eval/perplexity": math.exp(min(avg_loss, 20.0))}


def main() -> None:
    # --- Config ---
    if len(sys.argv) < 2:
        print("Usage: train.py <config.toml> [--section.key=value ...]")
        sys.exit(1)

    config_path = sys.argv[1]
    cli_args = sys.argv[2:]
    config = load_config(config_path, cli_args=cli_args)

    # --- Distributed setup ---
    rank, local_rank, world_size = get_world_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    device_mesh = init_distributed(config.distributed, seed=config.train.seed)
    config.validate(world_size)

    log_job_info()
    logger.info(f"Training config: {config}")

    # --- Resilience ---
    shutdown_handler = ShutdownHandler(timeout_sec=120)
    shutdown_handler.register()

    nan_detector = NaNDetector(action="warn", max_consecutive=10)

    tc = config.train
    mc = config.model
    pp_enabled = config.distributed.pp > 1
    mp_policy = default_mp_policy(tc.param_dtype)

    # --- Loss function ---
    loss_fn = registry.get_loss(tc.loss_fn)

    # --- Model ---
    if pp_enabled:
        from kempnerforge.distributed.pipeline_parallel import (
            build_pipeline_schedule,
            build_pipeline_stage,
            build_stage_module,
            get_pp_rank,
            get_pp_size,
        )

        pp_rank = get_pp_rank(device_mesh)
        pp_size = get_pp_size(device_mesh)

        tp_enabled_pp = device_mesh is not None and "tp" in device_mesh.mesh_dim_names

        if tp_enabled_pp:
            # Meta-device init: same pattern as non-PP TP path.
            # Avoids OOM for large PP stages that don't fit on one GPU before TP shards them.
            with torch.device("meta"):
                stage_mod = build_stage_module(config.model, pp_rank, pp_size)
            model = stage_mod
            apply_tensor_parallel(model, device_mesh)
            apply_ac(model, tc.activation_checkpointing)
            if device_mesh is not None:
                apply_fsdp2(model, device_mesh, mp_policy=mp_policy)
            model.to_empty(device=device)
            model.init_weights_and_freqs()
            model.to(dtype=tc.param_dtype)
        else:
            stage_mod = build_stage_module(config.model, pp_rank, pp_size)
            model = stage_mod.to(device=device, dtype=tc.param_dtype)
            apply_ac(model, tc.activation_checkpointing)
            if device_mesh is not None:
                apply_fsdp2(model, device_mesh, mp_policy=mp_policy)

        if tc.compile_model:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model (PP stage {pp_rank}/{pp_size}): {n_params:,} parameters")

        # Build pipeline stage and schedule
        pp_stage = build_pipeline_stage(
            model,
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
    else:
        model = build_parallel_model(
            config.model,
            device,
            device_mesh,
            ac_mode=tc.activation_checkpointing,
            mp_policy=mp_policy,
            param_dtype=tc.param_dtype,
            compile_model=tc.compile_model,
        )

    # --- Optimizer + Scheduler ---
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler, max_steps=tc.max_steps)

    # --- Checkpoint ---
    ckpt_mgr = CheckpointManager(config.checkpoint, model, optimizer)

    # Auto-resume
    resume_path = resolve_resume_path(config.checkpoint.dir)
    step, tokens_seen = 0, 0
    if resume_path or config.checkpoint.load_path:
        step, tokens_seen = ckpt_mgr.load(
            path=str(resume_path) if resume_path else None,
            scheduler=scheduler,
        )

    # --- Metrics ---
    tracker = MetricsTracker(config, num_gpus=world_size)
    tracker.init_backends(config)

    # --- Profiler ---
    prof = build_profiler(config.profiling, rank=rank)

    # --- Data ---
    # With PP, sampler should use DP rank/size (not total world size) since
    # all PP stages in the same DP group process the same batch.
    dp_rank, dp_size = get_dp_info(device_mesh)

    dataset = None
    dataloader = None
    data_iter = None
    if config.data.dataset_path:
        # Pre-tokenized data on disk (fastest path)
        dataset = MemoryMappedDataset(
            data_dir=config.data.dataset_path,
            seq_len=tc.seq_len + 1,
            file_pattern=config.data.file_pattern,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            seed=tc.seed,
        )
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=tc.batch_size,
            sampler=sampler,
            config=config.data,
        )
        logger.info(f"Dataset: {len(dataset):,} samples from {config.data.dataset_path}")
    elif config.data.hf_dataset_name:
        if not config.data.tokenizer_path:
            raise ValueError("data.tokenizer_path is required for HuggingFace datasets")

        if config.data.hf_streaming:
            # Streaming: on-the-fly tokenization, no full download needed
            from torch.utils.data import DataLoader as TorchDataLoader

            from kempnerforge.data.dataset import StreamingHuggingFaceDataset

            dataset = StreamingHuggingFaceDataset(
                dataset_name=config.data.hf_dataset_name,
                split=config.data.hf_dataset_split,
                text_field=config.data.hf_dataset_text_field,
                seq_len=tc.seq_len,
                tokenizer_path=config.data.tokenizer_path,
                dataset_config=config.data.hf_dataset_config,
                rank=dp_rank,
                world_size=dp_size,
                seed=tc.seed,
            )
            dataloader = TorchDataLoader(
                dataset,
                batch_size=tc.batch_size,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                prefetch_factor=(
                    config.data.prefetch_factor if config.data.num_workers > 0 else None
                ),
            )
            logger.info(
                f"Dataset: streaming from {config.data.hf_dataset_name} "
                f"({config.data.hf_dataset_split}), rank={dp_rank}/{dp_size}"
            )
        else:
            # Eager: download, tokenize, and pack all sequences into memory
            from kempnerforge.data.dataset import HuggingFaceDataset

            dataset = HuggingFaceDataset(
                dataset_name=config.data.hf_dataset_name,
                split=config.data.hf_dataset_split,
                text_field=config.data.hf_dataset_text_field,
                seq_len=tc.seq_len,
                tokenizer_path=config.data.tokenizer_path,
                dataset_config=config.data.hf_dataset_config,
            )
            sampler = DistributedSampler(
                dataset,
                num_replicas=dp_size,
                rank=dp_rank,
                shuffle=True,
                seed=tc.seed,
            )
            dataloader = StatefulDataLoader(
                dataset,
                batch_size=tc.batch_size,
                sampler=sampler,
                config=config.data,
            )
            logger.info(
                f"Dataset: {len(dataset):,} packed sequences from "
                f"{config.data.hf_dataset_name} ({config.data.hf_dataset_split})"
            )

    # --- Eval data ---
    eval_config = config.eval
    eval_dataloader = None
    if eval_config.enabled:
        from torch.utils.data import DataLoader as TorchDataLoader

        if eval_config.dataset_path:
            eval_dataset = MemoryMappedDataset(
                data_dir=eval_config.dataset_path,
                seq_len=tc.seq_len + 1,
                file_pattern=eval_config.file_pattern,
            )
            eval_sampler = DistributedSampler(
                eval_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=False, seed=tc.seed
            )
            eval_dataloader = TorchDataLoader(
                eval_dataset, batch_size=tc.batch_size, sampler=eval_sampler
            )
            logger.info(
                f"Eval dataset: {len(eval_dataset):,} samples from {eval_config.dataset_path}"
            )
        elif eval_config.hf_dataset_name:
            import numpy as np

            from kempnerforge.data.dataset import HuggingFaceDataset

            # Rank 0 loads/tokenizes the HF eval dataset, then broadcasts the
            # packed token tensor to all ranks via torch.distributed.broadcast.
            # This avoids file-lock failures (flock) on cluster filesystems
            # (Lustre, VAST) where load_dataset() would crash on all ranks.
            if rank == 0:
                eval_ds = HuggingFaceDataset(
                    dataset_name=eval_config.hf_dataset_name,
                    split=eval_config.hf_dataset_split,
                    text_field=config.data.hf_dataset_text_field,
                    seq_len=tc.seq_len,
                    tokenizer_path=config.data.tokenizer_path,
                    dataset_config=eval_config.hf_dataset_config,
                )
                packed = torch.from_numpy(np.stack(eval_ds._packed_sequences))
                n_seqs = torch.tensor([packed.shape[0]], device=device)
            else:
                n_seqs = torch.tensor([0], device=device)

            dist.broadcast(n_seqs, src=0)
            if rank != 0:
                packed = torch.empty(n_seqs.item(), tc.seq_len + 1, dtype=torch.long)
            packed_gpu = packed.to(device)
            dist.broadcast(packed_gpu, src=0)
            packed = packed_gpu.cpu()
            del packed_gpu

            # Wrap broadcast data as a simple map-style dataset
            class _EvalTensorDataset(torch.utils.data.Dataset):
                def __init__(self, data: torch.Tensor) -> None:
                    self._data = data

                def __len__(self) -> int:
                    return self._data.shape[0]

                def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                    tokens = self._data[idx]
                    return {"input_ids": tokens[:-1], "labels": tokens[1:]}

            eval_dataset = _EvalTensorDataset(packed)
            eval_sampler = DistributedSampler(
                eval_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=False, seed=tc.seed
            )
            eval_dataloader = TorchDataLoader(
                eval_dataset, batch_size=tc.batch_size, sampler=eval_sampler
            )
            logger.info(
                f"Eval dataset: {len(eval_dataset):,} packed sequences from "
                f"{eval_config.hf_dataset_name} ({eval_config.hf_dataset_split})"
            )

    logger.info(
        f"Starting training: step={step}, max_steps={tc.max_steps}, "
        f"batch_size={tc.batch_size}, grad_accum={tc.grad_accum_steps}, "
        f"world_size={world_size}"
    )

    model.train()

    if prof is not None:
        prof.start()

    while step < tc.max_steps:
        # Refresh data iterator at start / epoch boundary
        if dataloader is not None and data_iter is None:
            data_iter = iter(dataloader)

        tracker.start_step()

        if pp_enabled:
            # --- PP training step ---
            # Collect microbatches into a full batch for the schedule.
            # schedule.step() splits along dim 0 into n_microbatches.
            input_ids_list, labels_list = [], []
            for _ in range(tc.grad_accum_steps):
                if dataloader is not None:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
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
            is_first = pp_rank == 0
            is_last = pp_rank == pp_size - 1
            pp_losses: list[torch.Tensor] = []

            if is_first:
                pp_schedule.step(full_input, target=full_labels, losses=pp_losses)
            elif is_last:
                pp_schedule.step(target=full_labels, losses=pp_losses)
            else:
                pp_schedule.step()

            # Loss is only meaningful on the last stage
            if is_last and pp_losses:
                avg_loss = sum(loss.item() for loss in pp_losses) / len(pp_losses)
            else:
                avg_loss = 0.0

            # Gradient clipping
            grad_norm = clip_grad_norm_(model, tc.grad_clip_norm)
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            # Broadcast loss and grad_norm from last PP stage to all PP stages
            pp_mesh = device_mesh["pp"]
            pp_group = pp_mesh.get_group()
            loss_tensor = torch.tensor([avg_loss, grad_norm_val], device=device)
            dist.broadcast(loss_tensor, group_src=pp_size - 1, group=pp_group)
            avg_loss = loss_tensor[0].item()
            grad_norm_val = loss_tensor[1].item()

        else:
            # --- Standard training step (no PP) ---
            total_loss = 0.0

            for micro_step in range(tc.grad_accum_steps):
                if dataloader is not None:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                else:
                    input_ids = torch.randint(
                        0, mc.vocab_size, (tc.batch_size, tc.seq_len), device=device
                    )
                    labels = torch.randint(
                        0, mc.vocab_size, (tc.batch_size, tc.seq_len), device=device
                    )

                with maybe_no_sync(model, micro_step, tc.grad_accum_steps):
                    logits = model(input_ids)
                    loss = loss_fn(logits, labels)
                    scaled_loss = loss / tc.grad_accum_steps
                    scaled_loss.backward()
                    total_loss += loss.item()

            avg_loss = total_loss / tc.grad_accum_steps

            # Gradient clipping
            grad_norm = clip_grad_norm_(model, tc.grad_clip_norm)
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        # NaN check
        if not nan_detector.check_loss(avg_loss, step):
            optimizer.zero_grad()
            if nan_detector.should_rollback:
                logger.error("Too many consecutive NaNs — stopping")
                break
            step += 1
            continue

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step += 1
        tokens_in_step = tc.batch_size * tc.seq_len * tc.grad_accum_steps * dp_size
        tokens_seen += tokens_in_step

        # Metrics
        tracker.end_step(
            step=step,
            loss=avg_loss,
            grad_norm=grad_norm_val,
            lr=scheduler.get_last_lr()[0],
            tokens_in_step=tokens_in_step,
        )

        # Eval
        if eval_config.enabled and eval_dataloader is not None and step % eval_config.interval == 0:
            pp_group = None
            if pp_enabled:
                pp_mesh = device_mesh["pp"]
                pp_group = pp_mesh.get_group()
            eval_metrics = eval_step(
                model,
                eval_dataloader,
                loss_fn,
                device,
                eval_config.steps,
                pp_schedule=pp_schedule if pp_enabled else None,
                pp_rank=pp_rank if pp_enabled else None,
                pp_size=pp_size if pp_enabled else None,
                pp_group=pp_group,
            )
            tracker.log_eval(eval_metrics, step)

        # Advance profiler schedule
        if prof is not None:
            prof.step()

        # Checkpoint
        if step % config.checkpoint.interval == 0:
            ckpt_mgr.save(step=step, tokens_seen=tokens_seen, scheduler=scheduler)

        # Graceful shutdown
        if shutdown_handler.should_shutdown():
            logger.warning(f"Shutdown requested at step {step} — saving emergency checkpoint")
            ckpt_mgr.save(step=step, tokens_seen=tokens_seen, scheduler=scheduler)
            shutdown_handler.finish()
            break

    if prof is not None:
        prof.stop()
        if rank == 0:
            print_profiler_summary(prof, trace_dir=config.profiling.trace_dir)

    # Flush any pending async checkpoint before tearing down process group
    ckpt_mgr.wait()

    logger.info(f"Training complete: {step} steps, {tokens_seen:,} tokens")
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
