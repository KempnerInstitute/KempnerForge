# FP8

FP8 training converts `nn.Linear` modules to `torchao.float8.Float8Linear`,
which runs E4M3 forward / E5M2 backward matmuls with dynamic
tensorwise scaling. Master weights stay in bf16 — FP8 is a compute
mode, not a storage dtype.

Entry point:
[`apply_float8(model, enable_fsdp_float8_all_gather=True)`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py)
from `kempnerforge/distributed/parallel.py`. Gated by
`TrainConfig.mixed_precision = "fp8"`.

## What gets converted

`torchao.float8.convert_to_float8_training` walks the model and
replaces `nn.Linear` with `Float8Linear`. A filter function selects
which modules to convert:

```python
def _filter_fn(module, fqn):
    if "experts" in fqn or "shared_expert" in fqn:
        return False
    return "router" not in fqn
```

So the converted set is:

| Module | Converted | Reason |
|--------|:---------:|--------|
| Attention `q_proj`, `k_proj`, `v_proj`, `o_proj` | yes | Standard `Linear` call path |
| Dense MLP `gate_proj`, `up_proj`, `down_proj` | yes | Standard `Linear` call path |
| Final `output_head.proj` | yes | Standard `Linear` call path |
| MoE `experts.*.{gate,up,down}_proj` | **no** | Bypass — uses `torch._grouped_mm` |
| `shared_expert.*` | **no** | Same expert GEMM codepath |
| `router.gate` | **no** | Small output dim, not compute-bound |
| `token_embedding` | no | Not a `Linear` — `nn.Embedding` |
| Norms (RMSNorm) | no | Not a `Linear` |

## Exclusion rules

**`"experts"` / `"shared_expert"` in fqn**: the MoE forward uses
`torch._grouped_mm` (grouped GEMM over stacked expert weights) when
available. That call bypasses `Float8Linear.forward`, so wrapping
the expert Linears does nothing useful and would surprise debugging.
See [Expert parallelism § Local compute](expert-parallelism.md#dispatch--combine-flow).

**`"router"` in fqn**: the router gate's output dim is
`num_experts`, typically 8-64. `torch._scaled_mm` (used inside
`Float8Linear`) requires output dim divisible by 16 and gives no
speedup on such small matmuls. The router is also routing-decision
logic, not a compute bottleneck.

**Embeddings & norms**: not `Linear` — `convert_to_float8_training`
skips them automatically (no filter entry needed).

## `enable_fsdp_float8_all_gather`

```python
apply_float8(model, enable_fsdp_float8_all_gather=True)  # default
```

With this flag, FSDP2's all-gather communicates **FP8** weights
instead of bf16 — halves the all-gather bandwidth per layer. The
scale factors travel separately in a second tensor.

Default is `True` when FP8 is enabled. `build_parallel_model`
currently hard-codes the default (see
[`parallel.py` § apply_float8](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py)).

### TP incompatibility

```python
# Must be False when TP is active — the float8 weight wrapper calls
# aten.is_pinned on DTensors, which has no sharding strategy yet.
```

(Comment from `apply_float8`.) When TP is on, the float8 all-gather
path triggers `aten.is_pinned` on DTensor, which PyTorch doesn't
have a sharding rule for and which errors out. Workaround: set
`enable_fsdp_float8_all_gather=False` manually if combining TP + FP8
— the conversion still applies, only the FP8 all-gather optimization
is disabled.

See [Validation rules](../configuration/validation-rules.md) (the
**FP8 + TP** section) for the config-level check.

## Application order

FP8 conversion runs **after** TP / EP (so it sees the already-TP-wrapped
DTensors) and **before** AC / FSDP2:

```
apply_tensor_parallel(model, mesh)
apply_expert_parallel(model, mesh)
apply_float8(model)                 # ← here
apply_ac(model, ac_mode)
apply_fsdp2(model, mesh, ...)
```

Rationale: AC needs to recompute through `Float8Linear.forward` (not
the plain `Linear` it replaced), and FSDP2 needs to know each
parameter's true dtype (FP8 for converted Linears, bf16 for the
rest). See [Parallelism order](../architecture/parallelism-order.md)
(the **Float8 before AC and FSDP** section).

## Master weights vs compute dtype

The confusing part of FP8 training is that parameters are still
stored in bf16 — FP8 quantization happens at the matmul boundary:

- Parameter storage: `bfloat16` (matches `mixed_precision: "bf16"`
  master weights).
- Forward matmul: quantize to `E4M3` → `torch._scaled_mm` →
  dequantize output.
- Backward matmul: quantize to `E5M2` → `torch._scaled_mm` →
  dequantize output.
- Gradient: accumulate in `float32`, reduce-scatter in `float32`
  (same as standard FSDP2 mixed precision).

So from FSDP2's perspective, FP8 params look the same as bf16 params
— except when `enable_fsdp_float8_all_gather=True`, in which case
FSDP2's all-gather collective receives an FP8 tensor + a scale
tensor (torchao handles the interop).

`TrainConfig.param_dtype` returns `bfloat16` for `"fp8"`:

```python
_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp8":  torch.bfloat16,  # FP8 compute with bf16 master weights
}
```

## Config

```toml
[train]
mixed_precision = "fp8"           # bf16 master weights, E4M3/E5M2 matmul
```

That's the entire surface. The recipe is always `TENSORWISE`
(per-tensor scaling), and the filter rules are hard-coded. Custom
recipes / filters require editing `apply_float8` directly.

## Hardware

FP8 requires H100 / H200 or newer (Hopper+ with fourth-gen Tensor
Cores). On Ampere (A100), `torch._scaled_mm` falls back and the
speedup disappears — usually with an explicit PyTorch warning.

## Example: `7b_16gpu_fp8.toml`

```toml
[train]
mixed_precision = "fp8"
batch_size = 8
compile_model = true

[distributed]
dp_shard = -1                     # auto — fills the mesh
tp = 1
```

With 16× H200 and TP off, `enable_fsdp_float8_all_gather=True` is the
full FP8 path — FSDP2 all-gathers ride FP8, matmuls run on Hopper
tensor cores, master weights stay bf16. Benchmark in
[Benchmarks § MFU scaling](../reference/benchmarks.md#mfu-scaling-dense).

## Accuracy notes

Tensorwise FP8 has measurable loss-curve divergence from bf16 for
the first few hundred steps — usually catches up to within noise
by step 1000. For production runs, the standard pattern is:

1. Warmup in bf16 for `warmup_steps` (LR ramp-up).
2. Hand off to FP8 for the bulk of training.
3. (Optional) last-mile bf16 for the final cooldown if the min
   loss matters.

KempnerForge doesn't automate this switch — you'd restart with a
different `mixed_precision` to transition phases.

## See also

- [FSDP2 § Mixed precision policy](fsdp2.md#mixed-precision-policy) —
  how `MixedPrecisionPolicy` interacts with FP8 (it doesn't — FSDP2
  sees FP8 as bf16 with an attached scale).
- [Tensor parallelism](tensor-parallelism.md) — the TP + FP8 +
  `enable_fsdp_float8_all_gather=False` interaction.
- [Parallelism order](../architecture/parallelism-order.md) (see
  **Float8 before AC and FSDP**) — why FP8 runs between EP and AC.
- [Validation rules](../configuration/validation-rules.md) (see
  **FP8 + TP**) — the config-level `tp > 1 + fp8` check.
- [Benchmarks](../reference/benchmarks.md) — measured FP8 MFU
  numbers.
