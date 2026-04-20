---
name: explain-architecture
description: Walk through KempnerForge's subsystems in the order a forward pass encounters them. Starting point for anyone new to the codebase.
---

## When to use
- New contributor asking "how does training work" or "where does X live".
- Before a non-trivial change, to confirm which subsystem owns the behavior in question.
- When reviewing a PR that touches multiple subsystems and the user wants to understand the interaction.

This skill is read-only. It does not run anything.

## Preflight
Run:

    uv run python scripts/check_env.py

Baseline only (`uv`, repo layout). If the exit code is non-zero, fix the baseline first, otherwise file paths below may not resolve.

## Context (auto-generated, do not edit)
<!-- context-begin -->
Top-level package: kempnerforge/
Subsystems: config, model, distributed, data, training, checkpoint, resilience, metrics, profiling
Entry point: scripts/train.py (does NOT use torchrun internally; caller launches it)
Config system: kempnerforge/config/{schema,job,data,eval,optimizer}.py with registry.py for pluggable components
Model stack: embedding -> transformer.py (blocks of attention + mlp/moe) -> norm -> output projection
Parallelism layers: FSDP2 (distributed/parallel.py), TP (distributed/tensor_parallel.py), PP (distributed/pipeline_parallel.py), EP (distributed/expert_parallel.py)
Data path: dataset.py (MemoryMapped | HF | Mixture) -> sampler.py (DistributedSampler) -> dataloader.py (StatefulDataLoader)
Checkpoint: checkpoint/manager.py orchestrates DCP save/load, state.py owns state dict layout, async_save.py handles non-blocking writes
Resilience: resilience/ (signal handlers, NaN detection, NCCL health)
Metrics: metrics/tracker.py dispatches to backends (wandb, tensorboard), mfu.py computes MFU, memory.py tracks peak memory
<!-- context-end -->

## Procedure
Walk the user through the path a single forward+backward pass takes. Use the outline below and follow up only on the subsystems they ask about.

### 1. Config is loaded
- `scripts/train.py` calls `load_config(toml_path, cli_args)` from `kempnerforge/config/`.
- Layered resolution: dataclass defaults, then TOML overrides, then `--section.key=value` CLI overrides.
- `JobConfig` is the top-level container. Inspect the sections: `ModelConfig`, `TrainConfig`, `OptimizerConfig`, `SchedulerConfig`, `DistributedConfig`, `DataConfig`, `CheckpointConfig`, `MetricsConfig`.
- Pluggable components (optimizers, schedulers, routers, losses) come from `kempnerforge/config/registry.py` via `@register_*` decorators. Read the registry to see what names are valid for each component.

### 2. Distributed process group is initialized
- `kempnerforge/distributed/setup.py::init_distributed()` reads `MASTER_ADDR` / `MASTER_PORT` (or SLURM env vars) and calls `torch.distributed.init_process_group`.
- `DeviceMesh` from `distributed/parallel.py` gives the training loop a multi-dimensional view (DP x TP x PP) that each subsystem queries.
- On multi-node: the shell script (`scripts/slurm/multinode.sh`) sets env vars; this function only consumes them.

### 3. Model is built and sharded
- `kempnerforge/model/transformer.py::Transformer` composes: embedding -> N blocks (`TransformerBlock`) -> final norm -> output projection.
- Each `TransformerBlock` has an `Attention` (`model/attention.py`, SDPA-based GQA with RoPE from `position.py`) and an MLP (`model/mlp.py` SwiGLU, or `model/moe.py` for mixture of experts with a router from `router.py`).
- After construction, `distributed/parallel.py` applies `fully_shard()` (FSDP2) per block. If TP or PP is enabled, they compose via `parallelize_module` and DeviceMesh.

### 4. Data pipeline is constructed
- `kempnerforge/data/dataset.py` holds three dataset types: `MemoryMappedDataset` (pre-tokenized .npy, fastest), `HuggingFaceDataset` / `StreamingHuggingFaceDataset` (HF Hub), `MixtureDataset` (weighted combination of sub-datasets with phase scheduling).
- `data/sampler.py::DistributedSampler` partitions indices across data-parallel ranks. `set_skip(n)` supports exact resume from mid-epoch.
- `data/dataloader.py::StatefulDataLoader` wraps the torch DataLoader and tracks enough state (epoch, batch index, sampler skip) to restore the exact batch after checkpoint load.

### 5. Optimizer and scheduler
- `kempnerforge/training/optimizer.py` builds the optimizer (AdamW fused, Lion, Muon, schedule-free) from `[optimizer]` config via the registry.
- `training/scheduler.py` builds the LR scheduler. Registered names: `cosine`, `linear`, `wsd`, `constant`, `rex`, `none`.
- `training/grad.py` owns gradient utilities: clipping, accumulation (`maybe_no_sync` context manager for FSDP), NaN detection hooks.

### 6. Training loop
- `scripts/train.py` runs the loop directly (no framework wrapper). Each step: forward, loss (`training/loss.py`), backward, optional gradient accumulation, optimizer step, scheduler step, metrics log, checkpoint if interval hit.
- MoE runs apply an aux loss term inside the model (see `model/transformer.py::get_moe_aux_loss`), which `train.py` adds to the main loss before `backward()`.

### 7. Checkpointing
- `kempnerforge/checkpoint/manager.py::CheckpointManager` orchestrates save and load. Uses `torch.distributed.checkpoint` (DCP), which automatically handles FSDP sharded state and reshards on load.
- `checkpoint/state.py` defines what goes in the state dict: model, optimizer, scheduler, dataloader position, RNG state.
- `checkpoint/async_save.py` wraps `dcp.async_save` so training continues while writes are in flight.
- Resume follows the `latest` symlink in `checkpoint.dir`, falling back to the highest `step_N` directory.

### 8. Metrics and resilience
- `kempnerforge/metrics/tracker.py::MetricsTracker` collects per-step scalars with optional EMA smoothing, then dispatches to enabled backends (wandb, tensorboard).
- `metrics/mfu.py` computes model FLOPs utilization from the model config and measured step time.
- `metrics/memory.py` tracks peak GPU memory, with an optional snapshot export for pytorch.org/memory_viz.
- `resilience/` owns signal handlers (SIGTERM/SIGUSR1 for SLURM preemption), NaN action policies, and GPU/NCCL liveness probes that run every N steps.

## Verification
None. This skill is informational. Ask the user which subsystem they want to dig into, then point them at the specific files and classes listed above.

## Gotchas
- The training loop is in `scripts/train.py`, not inside a `Trainer` class. Deliberately flat, easy to read top to bottom. Do not try to find a `class Trainer`.
- FSDP2 uses `fully_shard()` (composable), NOT `FullyShardedDataParallel` (the v1 class). Old tutorials do not apply.
- Context parallelism (CP) is stubbed but not wired up yet (PyTorch 2.11 has an experimental ring-attention API). If the user asks "how do I enable CP", explain it is pending, not a docs/user error.
- `scripts/train.py` takes the TOML path as a positional arg, not via `--config`. CLI overrides are `--section.key=value`, double dash.

## Related skills
- `/kempnerforge:component-gaps` — ask-it-what-is-built-vs-planned for each subsystem.
- `/kempnerforge:add-optimizer` — concrete walkthrough of the training/optimizer.py + registry pattern.
