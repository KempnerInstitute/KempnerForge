# Tensor parallelism

`apply_tensor_parallel(model, device_mesh)` in
[`kempnerforge/distributed/tensor_parallel.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/tensor_parallel.py)
shards attention and MLP weights across the `tp` mesh dimension using
PyTorch's `parallelize_module` + DTensor.

## What gets sharded

Per `TransformerBlock`:

| Module | Style | Reason |
|--------|-------|--------|
| `attention.q_proj`, `k_proj`, `v_proj` | `ColwiseParallel` | Split attention heads across TP ranks |
| `attention.o_proj` | `RowwiseParallel` | Gather heads, reduce-scatter back to sequence-sharded |
| `mlp.gate_proj`, `mlp.up_proj` (SwiGLU) | `ColwiseParallel` | Split hidden dim |
| `mlp.down_proj` | `RowwiseParallel` | Gather hidden, reduce-scatter |
| `attention_norm`, `mlp_norm` | `SequenceParallel` | Norm computed on sequence-sharded input (see below) |

MoE blocks omit the MLP entries: experts are replicated across TP
(TP doesn't shard expert weights), EP handles that dimension
separately.

At the model level:

| Module | Style | When |
|--------|-------|------|
| `token_embedding` | Forward hook wraps output as `DTensor(Replicate())` | When sequence parallel is on |
| `norm` (final) | `SequenceParallel` | When sequence parallel is on |
| `output_head.proj` | `ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate())` | When sequence parallel is on and embeddings not tied |

## SequenceParallel

When enabled, norms operate on the **sequence-sharded** tensor (each
TP rank holds `seq_len / tp` positions) rather than the
replicated one. Activations flow like this through a block:

```
residual:          Shard(1)                       (seq-sharded)
  attention_norm:  Shard(1) → Shard(1)            (SequenceParallel)
  q/k/v_proj:      Shard(1) → Shard(2)            (Colwise all-gathers seq, shards heads)
  attention:       Shard(2) compute
  o_proj:          Shard(2) → Shard(1)            (Rowwise gathers heads, reduce-scatters seq)
  + residual:      Shard(1)
  mlp_norm:        Shard(1) → Shard(1)
  gate/up_proj:    Shard(1) → Shard(2)
  down_proj:       Shard(2) → Shard(1)
  + residual:      Shard(1)
```

Net effect: activations are kept sequence-sharded everywhere *except*
inside attention and MLP compute. Memory for activations drops by
`tp`× at block boundaries, and the extra all-gathers are overlapped
with compute.

### When SequenceParallel is disabled

Three cases, all handled automatically by `apply_tensor_parallel`:

| Condition | Why SP is off |
|-----------|---------------|
| `is_pp_stage` (model has `stage_id` attribute) | DTensor state at PP stage boundaries causes type-coercion issues |
| `config.tie_embeddings` | ColwiseParallel on the tied output head needs a different sharding than the embedding forward hook can provide |
| `config.is_moe` | Boolean indexing in MoE dispatch (`indices == expert_i`) breaks `Shard(1)` DTensors, and SP-on/SP-off block mixing creates DTensor transition errors |

In these cases the block gets "basic TP" — column/row parallel on the
Linears, no sharding of the sequence dimension. Activations are fully
replicated across the TP group, which costs more memory but works.

## Token embedding hook

With SequenceParallel on, the final norm expects its input to be
`Replicate()`-placed (so it can redistribute to `Shard(1)` internally).
The token embedding output is a plain tensor by default — without a
hook, `SequenceParallel` would reinterpret it as `Shard(1)` without
actually scattering, inflating the global sequence dim by `tp`.

`apply_tensor_parallel` installs a forward hook on `token_embedding`:

```python
def _embed_hook(module, input, output):
    return DTensor.from_local(output, mesh, placements=[Replicate()])
```

Effect: the embedding output is labeled as already-replicated, so
`SequenceParallel` redistributes to `Shard(1)` correctly.

## Post-hooks on attention and MLP

Operations inside attention (SDPA, `view`, `contiguous`) and MLP
strip DTensor metadata on their outputs — you get a plain `Tensor`
where the next `+ residual` expects a `Shard(1)` DTensor.

`apply_tensor_parallel` wraps the outputs back into DTensors via
forward post-hooks:

```python
def _wrap_shard1(module, input, output):
    return DTensor.from_local(output, mesh, placements=[Shard(1)])
```

Without this, you get "mixed Tensor and DTensor" errors at the
residual add. With it, DTensor metadata is restored and the rest of
the block flows correctly.

## Meta-device init

When TP is active, the model is built on `torch.device("meta")`
first — no storage is allocated yet. `apply_tensor_parallel` wraps
the (still-meta) parameters as DTensors, then `apply_fsdp2` further
shards them, and only `model.to_empty(device=device)` allocates real
storage afterward.

The sequence is in [FSDP2 § Application order](fsdp2.md#application-order-inside-build_parallel_model).

## Head divisibility

`n_heads % tp == 0` and `n_kv_heads % tp == 0` are required —
`ColwiseParallel` splits heads evenly across TP ranks, and uneven
splits are rejected.

For Llama-3 70B (`n_heads=64, n_kv_heads=8`), `tp` must be in `{1, 2,
4, 8}` — and `tp=8` means 1 KV head per TP rank, which is usually
where the GQA sharing stops paying off.

See
[Validation rules § Tensor-parallel head divisibility](../configuration/validation-rules.md#tensor-parallel-head-divisibility).

## TP across vs within nodes

- **Within a node** (NVLink): TP's all-gathers and reduce-scatters
  are on 900 GB/s links — fast enough to overlap with compute.
- **Across nodes** (InfiniBand): TP over IB is slow; you'll see
  it in the MFU numbers as soon as `tp > GPUs_per_node`.

Shipped recipes keep `tp ≤ 4` (single-node TP on 4×H200 per node) —
see [Parallelism recipes](../reference/parallelism-recipes.md).

## When to use TP

| Situation | Use TP? |
|-----------|---------|
| FSDP alone fits | No — at 7B on 4 GPUs, `tp=4` loses ~18 MFU points vs pure FSDP |
| `n_layers` blocks FSDP (e.g. 70B on 32 GPUs) | Yes — memory forces it |
| Need to shard attention weights (huge `dim`, few layers) | Yes |
| MoE | Yes, for attention only — TP on experts is replaced by EP |

The rule of thumb and supporting benchmarks are in
[Parallelism recipes § Choosing a parallelism combination](../reference/parallelism-recipes.md#choosing-a-parallelism-combination).

## See also

- [Device mesh](device-mesh.md) — how the `tp` sub-mesh is built.
- [FSDP2](fsdp2.md) — runs after TP and shards the remaining params.
- [Expert parallelism](expert-parallelism.md) — what happens to MoE
  experts (TP replicates, EP shards).
- [Parallelism order](../architecture/parallelism-order.md) — the
  full sequence with reasoning.
- [Configuration § Validation rules](../configuration/validation-rules.md#tensor-parallel-head-divisibility) —
  the head-divisibility check.
