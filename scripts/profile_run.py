#!/usr/bin/env python3
"""Profile a short training run and print detailed GPU kernel stats.

Usage:
    uv run torchrun --nproc_per_node=4 scripts/profile_run.py configs/train/wikitext_test.toml
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, schedule

from kempnerforge.config.loader import load_config
from kempnerforge.data.dataloader import StatefulDataLoader
from kempnerforge.data.dataset import MemoryMappedDataset
from kempnerforge.data.sampler import DistributedSampler
from kempnerforge.distributed.parallel import apply_ac, apply_fsdp2
from kempnerforge.distributed.setup import get_world_info, init_distributed
from kempnerforge.distributed.utils import clip_grad_norm_
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.grad import maybe_no_sync
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.scheduler import build_scheduler


def main() -> None:
    config_path = sys.argv[1]
    cli_args = sys.argv[2:]
    config = load_config(config_path, cli_args=cli_args)

    rank, local_rank, world_size = get_world_info()
    device = torch.device(f"cuda:{local_rank}")
    device_mesh = init_distributed(config.distributed, seed=config.train.seed)
    config.validate(world_size)

    tc = config.train
    mc = config.model

    # Build model
    model = Transformer(mc).to(device)
    apply_ac(model, tc.activation_checkpointing)
    if device_mesh is not None:
        apply_fsdp2(model, device_mesh)
    if tc.compile_model:
        model = torch.compile(model)

    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler, max_steps=tc.max_steps)

    # Data
    dataset = MemoryMappedDataset(
        data_dir=config.data.dataset_path, seq_len=tc.seq_len + 1, file_pattern="train_*.npy"
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = StatefulDataLoader(
        dataset, batch_size=tc.batch_size, sampler=sampler, config=config.data
    )
    data_iter = iter(dataloader)

    model.train()

    # Warmup 5 steps (includes torch.compile warmup)
    if rank == 0:
        print("Warming up (5 steps)...")
    for _ in range(5):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        clip_grad_norm_(model, tc.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Profile 10 steps
    if rank == 0:
        print("\nProfiling 10 steps...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=2, active=8, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for _step in range(10):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            for micro_step in range(tc.grad_accum_steps):
                with maybe_no_sync(model, micro_step, tc.grad_accum_steps):
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    )
                    loss.backward()

            clip_grad_norm_(model, tc.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            prof.step()

    if rank == 0:
        # Print kernel-level summary
        print("\n" + "=" * 100)
        print("TOP CUDA KERNELS (by total CUDA time)")
        print("=" * 100)
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=30, top_level_events_only=False
            )
        )

        print("\n" + "=" * 100)
        print("TOP CUDA KERNELS (by FLOPS)")
        print("=" * 100)
        print(
            prof.key_averages().table(
                sort_by="flops", row_limit=20, top_level_events_only=False
            )
        )

        # Compute aggregate stats
        total_cuda_time = 0
        matmul_time = 0
        comm_time = 0
        memory_time = 0
        other_time = 0
        total_flops = 0

        for evt in prof.key_averages():
            cuda_us = evt.self_device_time_total
            name = evt.key.lower()

            # Skip profiler overhead entries
            if "profilerstep" in name:
                continue

            total_cuda_time += cuda_us
            total_flops += evt.flops if evt.flops else 0

            if any(k in name for k in [
                "gemm", "mm", "matmul", "dot", "bmm", "cublas", "nvjet", "cutlass",
            ]):
                matmul_time += cuda_us
            elif any(k in name for k in ["nccl", "allreduce", "allgather", "reduce_scatter"]):
                comm_time += cuda_us
            elif any(k in name for k in ["memcpy", "memset"]):
                memory_time += cuda_us
            else:
                other_time += cuda_us

        print("\n" + "=" * 100)
        print("AGGREGATE GPU TIME BREAKDOWN")
        print("=" * 100)
        print(f"  Total CUDA time:    {total_cuda_time / 1e6:.3f} s")
        mm_pct = 100 * matmul_time / max(total_cuda_time, 1)
        comm_pct = 100 * comm_time / max(total_cuda_time, 1)
        mem_pct = 100 * memory_time / max(total_cuda_time, 1)
        other_pct = 100 * other_time / max(total_cuda_time, 1)
        print(f"  MatMul/GEMM:        {matmul_time / 1e6:.3f} s ({mm_pct:.1f}%)")
        print(f"  Communication:      {comm_time / 1e6:.3f} s ({comm_pct:.1f}%)")
        print(f"  Memory ops:         {memory_time / 1e6:.3f} s ({mem_pct:.1f}%)")
        print(f"  Other kernels:      {other_time / 1e6:.3f} s ({other_pct:.1f}%)")
        print(f"  Total FLOPS:        {total_flops / 1e12:.2f} TFLOP")
        if total_cuda_time > 0:
            achieved_tflops = total_flops / (total_cuda_time / 1e6) / 1e12
            print(f"  Achieved TFLOPS:    {achieved_tflops:.1f}")
            print("  H200 peak (bf16):   989 TFLOPS")
            print(f"  Kernel efficiency:  {100*achieved_tflops/989:.1f}%")

        # Export chrome trace for rank 0
        trace_path = "profiler_traces/trace_rank0.json"
        import os
        os.makedirs("profiler_traces", exist_ok=True)
        prof.export_chrome_trace(trace_path)
        print(f"\n  Chrome trace saved to: {trace_path}")
        print("  View at: chrome://tracing or https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
