# Turn on FP8 training

FP8 trades some loss-curve noise for a throughput win on Hopper-class
GPUs. On this codebase, enabling it is a one-line config change — the
interesting questions are whether you *should*, how to verify it
actually ran, and how to read the failure modes. The subsystem
reference at
[Distributed § FP8](../distributed/fp8.md) covers the internals; this
page is the researcher-facing decision and workflow.

## Decide whether FP8 is worth it

| If… | FP8 is likely to help |
|-----|:---------------------:|
| You're on H100 / H200 | yes |
| You're on A100 / V100 or older | no (falls back, no speedup) |
| Model is dense (no MoE) or MoE with heavy attention | yes |
| Model is tiny (dim < ~1024) | marginal |
| You need TP (`distributed.tp > 1`) | **blocked** — see below |
| You need the absolute-best eval loss at small step counts | maybe not — see [Accuracy tradeoff](#accuracy-tradeoff) |

Expected gain on Hopper: the FSDP2 all-gather moves half the bytes,
and matmuls run on fourth-gen tensor cores instead of bf16 tensor
cores. Exact MFU numbers are workload-dependent; consult
[Reference § Benchmarks](../reference/benchmarks.md#mfu-scaling-dense)
for measured values.

## Turn it on

One field in `[train]`:

```toml
[train]
mixed_precision = "fp8"
```

That's the whole surface. `scripts/train.py` picks it up via
[`TrainConfig.is_fp8`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/training.py),
`build_parallel_model` calls `apply_float8(model)` after TP/EP and
before AC/FSDP2, and FSDP2 picks bf16 master weights (FP8 is a
compute dtype only). No per-layer knobs.

### Reference config

[`configs/train/7b_16gpu_fp8.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_fp8.toml)
is a production-ready 7B Llama-3 recipe on 16 H100s:

```toml
[train]
mixed_precision = "fp8"
batch_size = 8                      # larger M-dim → better TC saturation
grad_accum_steps = 2                # keep effective batch matched
activation_checkpointing = "full"   # full AC; selective OOMs at bs=8
compile_model = true                # torch.compile pairs well with FP8

[distributed]
dp_shard = -1                       # FSDP over all 16 ranks
# tp is absent → defaults to 1 (TP is not allowed with FP8)
```

The config header has the tuning rationale: doubling `batch_size` to
saturate the M-dimension of the Float8 GEMM is what actually extracts
the FP8 speedup, and full AC is needed at that batch size or MLP
activations OOM.

## Compatibility matrix

| Pairing | Works? | Notes |
|---------|:------:|-------|
| FP8 + FSDP2 | yes | Primary path. `enable_fsdp_float8_all_gather=True` is on by default. |
| FP8 + TP | **no** | Config validation rejects this — see [`JobConfig.__post_init__`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/job.py). |
| FP8 + EP | yes | Experts and the router are excluded from FP8 conversion; attention and dense blocks still use FP8. |
| FP8 + `torch.compile` | yes | Recommended combination. |
| FP8 + PP | yes | PP stages compose with FP8 the same way as bf16. |

The TP block is explicit and hard:

```
FP8 + Tensor Parallelism is not yet supported (torchao Float8Linear
does not compose with DTensor sharding). Use FP8 with FSDP only, or
TP without FP8.
```

## Verify it's actually running

FP8 failures are silent by default — a misconfigured run still trains
in bf16 and produces sensible loss curves. Check the rank-0 log at
startup:

```
Applied Float8 training: recipe=TENSORWISE, fsdp_float8_all_gather=True
Optimizer groups: ...
Model: ...
```

The `Applied Float8 training` line (from
[`apply_float8`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py))
is the confirmation. If it's missing, `mixed_precision` didn't resolve
to `"fp8"`.

For a deeper check, profile one step:

```toml
[profiling]
enable     = true
start_step = 10
end_step   = 12
trace_dir  = "profiler_traces/fp8_check"
```

The summary table should list kernels whose names contain `scaled_mm`
(the FP8 GEMM) rather than only `gemm` / `cutlass_bf16`. See
[Debug § Shape 4](debug-training-regressions.md#shape-4-throughput-silently-halves)
for how to read the summary.

## What gets converted, what doesn't

Pulled from the filter in
[`apply_float8`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py):

- **Converted** — attention (`q/k/v/o_proj`), dense MLP
  (`gate/up/down_proj`), final `output_head.proj`.
- **Excluded** — MoE experts (`experts.*`, `shared_expert.*`) because
  they use `torch._grouped_mm` and bypass `Float8Linear.forward`;
  the MoE router because its output dim is small and not compute-bound;
  embeddings and norms because they aren't `nn.Linear`.

For an MoE model this means attention is FP8, dense-block MLPs are FP8,
but the expert blocks run in bf16 — typically fine, because attention
and shared experts dominate FLOPs.

## Accuracy tradeoff

Tensorwise FP8 has a measurable early-step loss-curve divergence from
bf16 — usually within noise by step ~1000 but visible before. Three
operating patterns:

1. **All-FP8**. Cheapest. Accept the early-step noise. Fine for most
   long runs.
2. **bf16 warmup → FP8 body**. Start with `mixed_precision = "bf16"`
   for the LR warmup phase, resume with `mixed_precision = "fp8"` for
   the bulk of training. Not automated — you checkpoint and relaunch
   with the new flag.
3. **FP8 body → bf16 cooldown**. Only worth it when you need the last
   fraction of loss improvement for a final eval.

The codebase doesn't automate any of these — they're all
checkpoint-and-relaunch patterns.

## Troubleshooting

**"Applied Float8 training" missing from log, training still runs.**
`mixed_precision` is wrong. Check
`config.train.mixed_precision == "fp8"` (not `"FP8"` or `"float8"`).
The `Literal[...]` type in the dataclass lets only four values through.

**`ValueError: FP8 + Tensor Parallelism is not yet supported`.** The
config validator ran. Either set `distributed.tp = 1` (and let FSDP
handle sharding) or drop `mixed_precision = "fp8"`.

**`torch._scaled_mm` warning about unsupported hardware.** You're on
a pre-Hopper GPU. FP8 falls back to bf16 matmul and you get no
speedup — disable FP8 for this cluster.

**`aten.is_pinned` DTensor error at model build.** A TP + FP8
combination slipped past the config check. The guard in
[`apply_float8`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py)
is explicit about this.

**Loss diverges in the first ~50 steps.** Expected early-step noise
is subtle; an actual divergence usually means the LR is set for
bf16-master-weight assumptions but FP8 is amplifying gradient noise.
Start with a smaller `lr` or a longer `scheduler.warmup_steps` and
check `train/grad_norm` — see
[Debug § Shape 1](debug-training-regressions.md#shape-1-loss-goes-nan).

**OOM on the first FP8 step but not on bf16.** FP8 all-gather saves
bandwidth but not memory; matmul temporaries for `_scaled_mm` are
slightly larger than bf16 matmul. Drop `batch_size` or switch to
`activation_checkpointing = "full"`.

## See also

- [Distributed § FP8](../distributed/fp8.md) — full reference on what
  `apply_float8` does, the E4M3/E5M2 math, and master weight handling.
- [Distributed § FSDP2 § Mixed precision policy](../distributed/fsdp2.md#mixed-precision-policy)
  — how FSDP2 sees FP8 (as bf16 with attached scale tensors).
- [Architecture § Parallelism order](../architecture/parallelism-order.md)
  — why FP8 runs between EP and AC.
- [Configuration § Validation rules](../configuration/validation-rules.md)
  — the `FP8 + TP` check.
- [Debug training regressions](debug-training-regressions.md) — the
  profiler workflow to verify FP8 kernels are firing.
- [Reference § Benchmarks](../reference/benchmarks.md) — measured MFU
  for FP8 runs at several model sizes.
- [`configs/train/7b_16gpu_fp8.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_fp8.toml)
  — the reference recipe this page points at.
