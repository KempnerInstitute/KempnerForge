---
name: component-gaps
description: Per-subsystem status report. What is implemented, what is tested, what is planned but unwired, and known limitations.
---

## When to use
- Before starting a feature, to confirm whether the groundwork already exists.
- When a config flag looks supported but does not behave as expected (stub versus real).
- Planning roadmap work: which subsystems have the most gap to close for a given goal.

Complements `/kempnerforge:explain-architecture` (which says what each subsystem does) by adding what each subsystem does not yet do.

## Preflight
Run:

    uv run python scripts/check_env.py

Baseline only. If non-zero, fix the baseline before the file references below will resolve.

## Context (auto-generated, do not edit)
<!-- context-begin -->
Subsystems surveyed: model, distributed (FSDP/TP/PP/EP/CP), data, training, checkpoint, resilience, metrics, profiling
Unit test coverage (per suite): tests/unit/test_{model, moe, router, optimizer, additional_optimizers, scheduler_extensions, data, packing, mixing, annealing, loss, checkpoint, resilience, distributed, pipeline_parallel, fp8, hooks, training, training_hooks, eval, generate, observability, performance, config}.py
Distributed tests: tests/distributed/ (multi-GPU via torchrun)
E2E tests: tests/e2e/ (opt-in via --e2e flag)
Integration tests: tests/integration/ (single GPU)
Known stubs: context parallelism (kempnerforge/distributed/* — no CP module yet)
<!-- context-end -->

## Procedure
Assume preflight has passed. Walk the user through the status of the subsystems they care about. Use this matrix as the reference truth; verify any specific claim by reading the code before recommending an action.

### Model (kempnerforge/model/)
- **Implemented**: Decoder-only Transformer (`transformer.py`), GQA attention with SDPA + RoPE, SwiGLU MLP, RMSNorm, token embedding with tied or separate output, KV cache path via `generate.py`, MoE MLP with sigmoid/softmax top-k routing (`moe.py`, `router.py`), FP8 mixed precision hooks (`hooks.py`).
- **Tested**: Unit tests for model forward/backward shape contracts, RoPE correctness, attention masks, MoE routing balance, FP8 scaling. MoE auxiliary loss flow through to `train.py` is covered by `test_moe.py`.
- **Gaps**: No encoder-decoder. No cross-attention. No vision tower. No multimodal adapter. Tokenizer is delegated to HF; no native BPE trainer.

### Distributed (kempnerforge/distributed/)
- **Implemented**: FSDP2 via `fully_shard()` (`parallel.py`), tensor parallelism (`tensor_parallel.py`), pipeline parallelism (`pipeline_parallel.py`), expert parallelism for MoE (`expert_parallel.py`), DeviceMesh composition.
- **Tested**: `tests/distributed/` covers FSDP e2e, TP sharding, PP schedule, EP dispatch. `tests/e2e/` run real short training on 4 GPUs.
- **Gaps**: **Context parallelism is not implemented**. No `context_parallel.py` exists. PyTorch 2.11 exposes an experimental ring-attention API; wiring it in is tracked work, not a docs oversight. If a user asks to enable CP, the honest answer is "pending".

### Data (kempnerforge/data/)
- **Implemented**: `MemoryMappedDataset` (pre-tokenized `.npy`), `HuggingFaceDataset`, `StreamingHuggingFaceDataset`, `MixtureDataset` with weighted sampling and phase scheduling (annealing), `DistributedSampler` with `set_skip()` for mid-epoch resume, `StatefulDataLoader`.
- **Tested**: `test_data.py`, `test_packing.py` (document-aware packing), `test_mixing.py` (mixture weights), `test_annealing.py` (phase transitions).
- **Gaps**: No streaming for `MixtureDataset` sub-datasets that are themselves HF streaming (each sub-source must materialize a dataset instance). No shuffling buffer for streaming HF (relies on HF's internal shuffle).

### Training (kempnerforge/training/)
- **Implemented**: AdamW, Lion, Muon, Schedule-Free AdamW optimizers (`optimizer.py`); cosine, linear warmup, WSD, none schedulers (`scheduler.py`); grad clipping, accumulation via `maybe_no_sync`, NaN detection hooks (`grad.py`, `hooks.py`); loss registry (`loss.py`); eval loop (`eval.py`).
- **Tested**: `test_optimizer.py`, `test_additional_optimizers.py`, `test_scheduler_extensions.py`, `test_loss.py`, `test_training.py`, `test_training_hooks.py`, `test_eval.py`.
- **Gaps**: No support for gradient compression. No ZeRO-Offload (CPU offload of optimizer state). Loss scaler for fp16 not wired (bf16-only mixed precision).

### Checkpoint (kempnerforge/checkpoint/)
- **Implemented**: DCP sharded save/load (`manager.py`), state dict layout including dataloader position (`state.py`), non-blocking async save (`async_save.py`), auto-resume via `latest` symlink.
- **Tested**: `test_checkpoint.py` (unit + round-trip), `tests/integration/` covers resumption.
- **Gaps**: No per-step-range retention policy beyond `keep_last_n`. No cross-cloud checkpoint (S3/GCS). No partial-shape-compatible resume (e.g., resume a bigger model from a smaller-model checkpoint).

### Resilience (kempnerforge/resilience/)
- **Implemented**: SIGTERM/SIGUSR1 handlers for SLURM preemption (`signal_handler.py`), GPU compute and memory health probes (`health.py`), NCCL liveness via all-reduce (`health.py`), elastic rendezvous helpers (`elastic.py`).
- **Tested**: `test_resilience.py`.
- **Gaps**: No automatic rollback to a specific earlier checkpoint on persistent NaN. Recommendation today is manual: the NaN handler warns and exits with guidance. No integration with external HPC monitoring (Prometheus / DCGM alerts).

### Metrics (kempnerforge/metrics/)
- **Implemented**: `MetricsTracker` with EMA smoothing, WandB and TensorBoard backends (`tracker.py`), MFU computation (`mfu.py`), peak memory tracking with optional snapshot export (`memory.py`), rank-aware logger (`logger.py`).
- **Tested**: `test_observability.py`, `test_performance.py`.
- **Gaps**: No Datadog or Prometheus backend. No structured event log (all logs are unstructured text).

### Profiling (kempnerforge/profiling/)
- **Implemented**: `torch.profiler` integration (`profiler.py`), lightweight CUDA timing (`cuda_timer.py`).
- **Tested**: Smoke covered indirectly by training tests; no dedicated unit tests for the profiler wrapper.
- **Gaps**: No flamegraph export. No Nsight Systems automation.

## Verification
Not applicable — this skill is informational. When the user is about to act on a "gap" above, verify it is still a gap before recommending workarounds:

    # Confirm CP is still absent
    ls kempnerforge/distributed/*context* 2>/dev/null || echo "no CP module yet"

    # Confirm an optimizer is still missing
    uv run python -c "from kempnerforge.config.registry import registry; import kempnerforge.training.optimizer; print(registry.list('optimizer'))"

## Gotchas
- This document lags the code between commits. Re-run the Context auto-gen (Phase 2) or spot-check before acting on a gap claim.
- "Tested" means there is a unit or distributed test. Heavy behaviors (MoE expert balancing over long runs, FSDP2 with >64 GPUs, FP8 stability) are validated by e2e / benchmark runs, not unit tests. Absence of a unit test is not the same as absence of validation.
- "Implemented" does not mean "production-validated at all scales". The `benchmarks/` package holds the actual bench scripts (`bench_forward.py`, `bench_moe.py`, `bench_optimizer.py`, `mfu_scaling/`, `moe_expert_parallel/`, `moe_packed/`) — run them directly for scale validation. The MFU benchmarks cover dense up through 70B and MoE up through 32 GPUs with EP.
- If you are about to tell the user a feature is missing, grep for it first. Claims decay fast.

## Related skills
- `/kempnerforge:explain-architecture` — the companion "what it does" walkthrough.
- `/kempnerforge:add-optimizer` — concrete example of extending a subsystem.
