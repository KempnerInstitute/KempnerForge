# MFU

Model FLOPs Utilization — achieved compute throughput divided by the
hardware peak. All the math lives in
[`kempnerforge/metrics/mfu.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/mfu.py).

```
MFU = (flops_per_token * tokens_per_sec) / (num_gpus * gpu_peak_tflops * 1e12)
```

## Formula

KempnerForge uses the PaLM paper approximation:

```
flops_per_token = 6 * active_params  +  12 * n_layers * dim * seq_len
```

`6*P` covers forward + backward matmuls (2 FLOPs per MAC, 3 passes — one
forward and two in backward for weight + activation gradients). `12*L*D*S`
covers the attention-score computation, which scales with sequence length.

## Dense active params

```python
# kempnerforge/metrics/mfu.py — _dense_flops_per_token
attn_params = (
    dim * (n_heads * head_dim)         # Q
    + 2 * dim * (n_kv_heads * head_dim) # K + V
    + (n_heads * head_dim) * dim       # O
)
mlp_params = 3 * dim * computed_ffn_hidden_dim   # SwiGLU (up, gate, down)
per_layer = attn_params + mlp_params
output_params = vocab_size * dim
active_params = n_layers * per_layer + output_params
```

Three things this does and doesn't count:

- **Counts output projection.** The LM head matmul is genuinely on the
  critical path.
- **Omits the input embedding table.** It's a gather, not a matmul —
  doesn't consume FLOPs.
- **Omits bias and norm params.** They're free compared to the matmuls.

## GQA is not discounted

```python
# 12 * n_layers * dim * seq_len — no n_kv_heads factor
```

The intuition would be "GQA reads fewer KV heads so attention is cheaper"
— but with FlashAttention (the default in this repo), GQA is expanded at
the kernel boundary: the same hardware multiplies `seq_len × seq_len ×
n_heads` scores. The wall-clock savings come from memory bandwidth, not
FLOPs. So `12*L*D*S` stays correct.

## MoE

```python
# kempnerforge/metrics/mfu.py — _moe_flops_per_token
n_moe_layers = sum(1 for i in range(n_layers) if (i + 1) % moe_frequency == 0)
n_dense_layers = n_layers - n_moe_layers

dense_active = n_dense_layers * (attn_params + mlp_params)
shared_mlp   = moe_shared_experts * mlp_params
moe_active   = n_moe_layers * (attn_params + moe_top_k * mlp_params + shared_mlp)

active_params = dense_active + moe_active + output_params
```

Key moves:

- **Active, not total, params.** Only `top_k` experts run per token, so
  the MoE MLP contribution is `top_k * mlp_params`, not
  `num_experts * mlp_params`.
- **Shared experts are always on.** `shared_mlp` is added unconditionally.
- **MoE frequency.** `(i + 1) % moe_frequency == 0` matches whichever
  layers the config promotes to MoE; the rest stay dense.
- **Router FLOPs ignored.** `dim × num_experts` is tiny compared to
  `mlp_params`.

The attention term (`12 * n_layers * dim * seq_len`) uses the full
`n_layers` regardless of MoE vs dense — attention runs in every layer.

## FP8 caveat

`gpu_peak_tflops` is auto-detected as **bf16** peak, not fp8. For H100:

- bf16 dense peak: 989 TFLOPS (reported in the table)
- fp8 dense peak: ~1979 TFLOPS (2× bf16)

If you train with [FP8](../distributed/fp8.md), the reported MFU is
understated by a factor of ~2. Override it explicitly if that matters:

```python
tracker = MetricsTracker(
    config, num_gpus=world_size,
    gpu_peak_tflops=1979.0,     # fp8 peak for H100
)
```

The opposite problem — reporting fp8 peak while the kernel actually runs
in bf16 — gives an overstated MFU. If you're mixing precision, match the
peak to the dominant matmul format.

## GPU detection

```python
# kempnerforge/metrics/mfu.py — get_gpu_peak_tflops
for gpu_name, tflops in _GPU_PEAK_TFLOPS.items():
    if gpu_name in torch.cuda.get_device_name(device):
        return tflops
# Fallback by compute capability
major, minor = torch.cuda.get_device_capability(device)
if major >= 9:  return 989.0    # Hopper (H100, H200)
elif major >= 8: return 312.0   # Ampere (A100)
else:           return 100.0
```

Known GPUs in the table:

| GPU | bf16 TFLOPS |
|-----|-------------|
| H100 / H100 SXM / H200 / H800 | 989 |
| H100 PCIe | 756 |
| A100 (any variant) | 312 |
| L40S | 362 |
| RTX 4090 | 330 |
| A10G | 125 |
| RTX 3090 | 142 |

Unknown GPUs get a compute-capability-based fallback with a warning log.
Add your GPU to `_GPU_PEAK_TFLOPS` for accurate MFU if the fallback is off.

## Checking whether MFU is reasonable

Rough bands on an H100 cluster (your mileage varies with batch size, seq
length, parallelism strategy):

- **< 25%** — something is wrong. Inspect the profiler for communication
  overhead or CPU stalls.
- **25–40%** — typical range for large models at modest batch sizes.
- **40–55%** — well-tuned. FlashAttention working, FSDP overlap, no
  pipeline bubbles.
- **> 55%** — check for double-counted FLOPs (e.g. top_k forgotten in the
  MoE denominator).

## See also

- [Metrics tracker](metrics-tracker.md) — where `compute_mfu` is called.
- [Configuration § `[model]`](../configuration/config-sections.md) —
  `n_layers`, `dim`, `n_heads`, `n_kv_heads`, `computed_ffn_hidden_dim`,
  `moe_frequency`, `moe_top_k`, `moe_shared_experts` all feed the formula.
- [FP8](../distributed/fp8.md) — when to override `gpu_peak_tflops`.
