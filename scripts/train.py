#!/usr/bin/env python3
"""KempnerForge training entry point.

Usage:
    # Single GPU
    uv run python scripts/train.py configs/train/debug.toml

    # Multi-GPU (single node)
    uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/default.toml

    # With overrides
    uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/default.toml \
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

    # --- Model ---
    model = Transformer(config.model).to(device)
    apply_ac(model, config.train.activation_checkpointing)

    if device_mesh is not None and "tp" in device_mesh.mesh_dim_names:
        apply_tensor_parallel(model, device_mesh)
    if device_mesh is not None:
        apply_fsdp2(model, device_mesh)

    if config.train.compile_model:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    # --- Optimizer + Scheduler ---
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler, max_steps=config.train.max_steps)

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
    tc = config.train
    mc = config.model

    dataset = None
    dataloader = None
    data_iter = None
    if config.data.dataset_path:
        # seq_len + 1 because dataset returns input_ids[:-1] and labels[1:]
        dataset = MemoryMappedDataset(
            data_dir=config.data.dataset_path,
            seq_len=tc.seq_len + 1,
            file_pattern="train_*.npy",
        )
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=tc.seed,
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
                # Fallback: random data if no dataset configured
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

        # Gradient clipping
        grad_norm = clip_grad_norm_(model, tc.grad_clip_norm)
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        # NaN check
        avg_loss = total_loss / tc.grad_accum_steps
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
        tokens_in_step = tc.batch_size * tc.seq_len * tc.grad_accum_steps * world_size
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
