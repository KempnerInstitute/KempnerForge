#!/usr/bin/env python3
"""KempnerForge training entry point.

Usage:
    # Single GPU
    uv run python scripts/train.py configs/train/debug.toml

    # Multi-GPU (single node, via torchrun)
    uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/default.toml

    # Multi-node (via SLURM srun — see scripts/slurm/multinode.sh)
    # srun launches one process per GPU; MASTER_ADDR/MASTER_PORT are resolved
    # automatically from SLURM env vars by init_distributed().
    srun uv run python scripts/train.py configs/train/default.toml

    # With overrides
    uv run python scripts/train.py configs/train/default.toml \
        --train.max_steps=1000 --optimizer.lr=1e-4
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.loader import load_config
from kempnerforge.data.dataloader import StatefulDataLoader
from kempnerforge.data.dataset import MemoryMappedDataset
from kempnerforge.data.sampler import DistributedSampler
from kempnerforge.distributed.parallel import apply_ac, apply_fsdp2
from kempnerforge.distributed.setup import destroy_distributed, get_world_info, init_distributed
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel
from kempnerforge.distributed.utils import clip_grad_norm_
from kempnerforge.metrics.logger import get_logger
from kempnerforge.metrics.tracker import MetricsTracker
from kempnerforge.model.transformer import Transformer
from kempnerforge.resilience.elastic import log_job_info, resolve_resume_path
from kempnerforge.resilience.health import NaNDetector
from kempnerforge.resilience.signal_handler import ShutdownHandler
from kempnerforge.training.grad import maybe_no_sync
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.scheduler import build_scheduler

logger = get_logger(__name__)


def _get_dp_info(device_mesh) -> tuple[int, int]:
    """Get (dp_rank, dp_size) from the device mesh, accounting for PP/TP."""
    if device_mesh is None:
        return 0, 1
    dim_names = device_mesh.mesh_dim_names
    if "dp_shard" in dim_names and "dp_replicate" in dim_names:
        dp_mesh = device_mesh["dp_replicate", "dp_shard"]
        return dp_mesh.get_local_rank(), dp_mesh.size()
    elif "dp_shard" in dim_names:
        dp_mesh = device_mesh["dp_shard"]
        return dp_mesh.get_local_rank(), dp_mesh.size()
    elif "dp_replicate" in dim_names:
        dp_mesh = device_mesh["dp_replicate"]
        return dp_mesh.get_local_rank(), dp_mesh.size()
    return 0, 1


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
                apply_fsdp2(model, device_mesh)
            model.to_empty(device=device)
            model.init_weights_and_freqs()
            model.to(dtype=torch.bfloat16)
        else:
            stage_mod = build_stage_module(config.model, pp_rank, pp_size)
            model = stage_mod.to(device=device, dtype=torch.bfloat16)
            apply_ac(model, tc.activation_checkpointing)
            if device_mesh is not None:
                apply_fsdp2(model, device_mesh)

        if tc.compile_model:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model (PP stage {pp_rank}/{pp_size}): {n_params:,} parameters")

        # Build pipeline stage and schedule
        pp_stage = build_pipeline_stage(
            model, device_mesh, device,
            batch_size=tc.batch_size,
            seq_len=tc.seq_len,
        )

        def pp_loss_fn(logits, labels):
            return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        pp_schedule = build_pipeline_schedule(
            stage=pp_stage,
            n_microbatches=tc.grad_accum_steps,
            loss_fn=pp_loss_fn,
            schedule=config.distributed.pp_schedule.value,
        )
    else:
        # Order matters: TP → AC → FSDP. AC wraps blocks in CheckpointWrapper,
        # which changes submodule paths, so TP must see the raw blocks first.
        tp_enabled = device_mesh is not None and "tp" in device_mesh.mesh_dim_names

        if tp_enabled:
            # Meta-device init (torchtitan pattern): create model with zero memory,
            # apply parallelisms, then materialize only the local shards on GPU.
            # This avoids both CPU OOM (4 procs × 280 GB for 70B fp32) and GPU OOM
            # (140 GB bf16 on 141 GB H200).
            with torch.device("meta"):
                model = Transformer(config.model)
            apply_tensor_parallel(model, device_mesh)
            apply_ac(model, tc.activation_checkpointing)
            if device_mesh is not None:
                apply_fsdp2(model, device_mesh)
            # Materialize local shards and initialize weights
            model.to_empty(device=device)
            model.init_weights_and_freqs()
            model.to(dtype=torch.bfloat16)
        else:
            model = Transformer(config.model).to(device=device, dtype=torch.bfloat16)
            apply_ac(model, tc.activation_checkpointing)
            if device_mesh is not None:
                apply_fsdp2(model, device_mesh)

        if tc.compile_model:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {n_params:,} parameters")

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

    # --- Data ---
    # With PP, sampler should use DP rank/size (not total world size) since
    # all PP stages in the same DP group process the same batch.
    dp_rank, dp_size = _get_dp_info(device_mesh)

    dataset = None
    dataloader = None
    data_iter = None
    if config.data.dataset_path:
        dataset = MemoryMappedDataset(
            data_dir=config.data.dataset_path,
            seq_len=tc.seq_len + 1,
            file_pattern=config.data.file_pattern,
        )
        sampler = DistributedSampler(
            dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True, seed=tc.seed,
        )
        dataloader = StatefulDataLoader(
            dataset, batch_size=tc.batch_size, sampler=sampler, config=config.data,
        )
        logger.info(f"Dataset: {len(dataset):,} samples from {config.data.dataset_path}")

    logger.info(
        f"Starting training: step={step}, max_steps={tc.max_steps}, "
        f"batch_size={tc.batch_size}, grad_accum={tc.grad_accum_steps}, "
        f"world_size={world_size}"
    )

    model.train()

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
            is_first = pp_rank == 0
            is_last = pp_rank == pp_size - 1

            if is_first:
                losses = pp_schedule.step(full_input, target=full_labels)
            elif is_last:
                losses = pp_schedule.step(target=full_labels)
            else:
                losses = pp_schedule.step()

            # Loss is only meaningful on the last stage
            if is_last and losses is not None:
                if isinstance(losses, (list, tuple)):
                    avg_loss = sum(loss.item() for loss in losses) / len(losses)
                else:
                    # schedule.step() may return a single tensor
                    avg_loss = losses.item() if losses.dim() == 0 else losses.mean().item()
            else:
                avg_loss = 0.0

            # Gradient clipping + optimizer step
            grad_norm = clip_grad_norm_(model, tc.grad_clip_norm)
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

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
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
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

        # Checkpoint
        if step % config.checkpoint.interval == 0:
            ckpt_mgr.save(step=step, tokens_seen=tokens_seen, scheduler=scheduler)

        # Graceful shutdown
        if shutdown_handler.should_shutdown():
            logger.warning(f"Shutdown requested at step {step} — saving emergency checkpoint")
            ckpt_mgr.save(step=step, tokens_seen=tokens_seen, scheduler=scheduler)
            shutdown_handler.finish()
            break

    logger.info(f"Training complete: {step} steps, {tokens_seen:,} tokens")
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
