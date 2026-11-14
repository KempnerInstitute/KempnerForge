# FP8

FP8 training converts `nn.Linear` modules to `torchao.float8.Float8Linear`,
which runs E4M3 forward / E5M2 backward matmuls with dynamic
tensorwise scaling. FP8 is a compute mode, not a storage dtype: master
weights stay in bf16, or fp32 with `train.fp32_master_weights = true`.

Entry point:
[`apply_float8(model, enable_fsdp_float8_all_gather=True)`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py)
from `kempnerforge/distributed/parallel.py`. Gated by
`TrainConfig.mixed_precision = "fp8"`.

## What gets converted

`torchao.float8.convert_to_float8_training` walks the model and
replaces `nn.Linear` with `Float8Linear`. A filter function selects
which modules to convert (simplified here; the real one also counts the
`Linear` layers and records each one the shape check keeps out, for a
startup warning):

```python
def _filter_fn(module, fqn):
    if "experts" in fqn or "shared_expert" in fqn or "router" in fqn:
        return False
    return module.in_features % 16 == 0 and module.out_features % 16 == 0
```

So the converted set is:

| Module | Converted | Reason |
|--------|:---------:|--------|
| Attention `q_proj`, `k_proj`, `v_proj`, `o_proj` | yes | Standard `Linear` call path |
| Dense MLP `gate_proj`, `up_proj`, `down_proj` | yes | Standard `Linear` call path |
| Final `output_head.proj` | yes | Standard `Linear` call path |
| A `Linear` from the three rows above with `in_features` or `out_features` not a multiple of 16 | **no** | `torch._scaled_mm` rejects it; kept in bf16 and named in a startup warning |
| MoE `experts.*.{gate,up,down}_proj` | **no** | Bypass — uses `torch._grouped_mm` |
| `shared_expert.*` | **no** | By convention, to match the routed experts (a plain `Linear` call path) |
| `router.gate` | **no** | Small output dim, not compute-bound |
| `token_embedding` | no | Not a `Linear` — `nn.Embedding` |
| Norms (RMSNorm) | no | Not a `Linear` |

## Exclusion rules

**`"experts"` in fqn**: the routed experts' forward uses
`torch._grouped_mm` (grouped GEMM over stacked expert weights) when
available. That call bypasses `Float8Linear.forward`, so wrapping
the expert Linears does nothing useful and would surprise debugging.
See [Expert parallelism § Local compute](expert-parallelism.md#dispatch--combine-flow).

**`"shared_expert"` in fqn**: the shared expert never takes the
grouped-GEMM path; `MoEMLP.forward` calls it as a plain module, so its
projections are ordinary `Linear` calls. It is excluded by convention,
to match the routed experts (see
[MoE FP8 § Why the shared expert is excluded](../moe/moe-fp8.md#why-the-shared-expert-is-excluded)).

**`"router"` in fqn**: the router gate's output dim is
`num_experts`, typically 8-64. `torch._scaled_mm` (used inside
`Float8Linear`) requires output dim divisible by 16 and gives no
speedup on such small matmuls. The router is also routing-decision
logic, not a compute bottleneck.

**A dimension not divisible by 16**: `torch._scaled_mm` needs both
dimensions of an FP8 weight to be multiples of 16 and raises on CUDA
otherwise. Common cases are `k_proj` and `v_proj` when
`n_kv_heads * head_dim` is small (8 for `dim = 64, n_heads = 8,
n_kv_heads = 1`), an output head with the GPT-2 vocabulary
(`vocab_size = 50257`, as in `configs/train/hf_wikitext.toml`), and an
explicit `ffn_hidden_dim` that is not a multiple of 16. The filter
keeps such a `Linear` in bf16, and `apply_float8` logs one warning that
names each kept layer with its shape and count, for example
`layers.*.attention.k_proj (8x64) x32`.

Config validation rejects the two cases the filter cannot fix (see
[Validation rules § FP8 geometry](../configuration/validation-rules.md#fp8-geometry)):
`model.dim` not a multiple of 16, where no layer could convert, and, in
a text run, `train.batch_size * train.seq_len` not a multiple of 16,
because the FP8 weight gradient multiplies over the micro-batch's
tokens. If a model passes both checks and still converts no `Linear`,
for example `dim = 64` with `n_heads = 4`, `head_dim_override = 10`,
`ffn_hidden_dim = 200` and `vocab_size = 50257`, `apply_float8` raises
a `ValueError` rather than let an FP8 run train entirely in bf16. A
packed VLM run (`data.pack_sequences = true`) is refused with FP8 when
`JobConfig` is built, because its head's token count changes from
batch to batch.

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

Config validation rejects `train.mixed_precision = "fp8"` with
`distributed.tp > 1` (`JobConfig.validate`), so a TP + FP8 run stops at
startup. The reason in the error is that torchao's `Float8Linear` does
not compose with the DTensor sharding the TP plan applies. The comment
above, from the `apply_float8` docstring, records a second problem, in
the float8 all-gather. Turning off `enable_fsdp_float8_all_gather` is
not a workaround: the config check rejects the combination either way.

See [Validation rules](../configuration/validation-rules.md) (the
**FP8 + TP** section) for the config-level check.

## Application order

FP8 conversion runs **after** TP / EP and **before** AC / FSDP2. In
practice TP is off whenever FP8 is on (see
[TP incompatibility](#tp-incompatibility)):

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
  master weights), or `float32` with `fp32_master_weights = true`
  (see [FSDP2 § fp32 master weights](fsdp2.md#fp32-master-weights)).
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
    "fp32": torch.float32,
    "fp8":  torch.bfloat16,  # FP8 compute; bf16 masters unless fp32_master_weights
}
```

## Config

```toml
[train]
mixed_precision = "fp8"           # E4M3/E5M2 matmul, bf16 master weights
fp32_master_weights = false      # true keeps the master weights in fp32
```

That's the entire surface. The recipe is always `TENSORWISE`
(per-tensor scaling), and the filter rules (the names above and the
divisible-by-16 rule) are hard-coded. Custom
recipes / filters require editing `apply_float8` directly.

## Hardware

FP8 needs `torch._scaled_mm`, which runs on GPUs of compute capability
8.9 (Ada) or 9.0 and newer (Hopper, such as H100 and H200). On older
GPUs such as the A100 it raises in the first forward,
`torch._scaled_mm is only supported on CUDA devices with compute
capability >= 9.0 or 8.9`, so an FP8 run stops there instead of falling
back to bf16. Train in bf16 on those GPUs.

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

With 16 H200s and TP off, `enable_fsdp_float8_all_gather=True` is the
full FP8 path: FSDP2 all-gathers ride FP8, matmuls run on Hopper
tensor cores, and master weights stay bf16 (fp32 with
`fp32_master_weights`).

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
- [Tensor parallelism](tensor-parallelism.md): the TP plan, which FP8
  does not run with yet.
- [Parallelism order](../architecture/parallelism-order.md) (see
  **Float8 before AC and FSDP**) — why FP8 runs between EP and AC.
- [Validation rules](../configuration/validation-rules.md) (see
  **FP8 + TP**) — the config-level `tp > 1 + fp8` check.
