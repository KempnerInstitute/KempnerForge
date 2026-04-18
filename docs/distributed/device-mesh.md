# Device mesh

Every parallelism family in KempnerForge composes through a single
`torch.distributed.device_mesh.DeviceMesh`. `init_distributed` builds
it once, then the `apply_*` functions pull sub-meshes from it. The
shape and dimension order of that mesh are load-bearing.

## Dimension order

```python
# scripts/train.py
device_mesh = init_distributed(config.distributed, seed=config.train.seed)
```

Inside
[`init_distributed`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/setup.py)
the order is fixed:

```python
dim_map = [
    ("pp",           resolved.pp),
    ("dp_replicate", resolved.dp_replicate),
    ("dp_shard",     resolved.dp_shard),
    ("ep",           resolved.ep),
    ("tp",           resolved.tp),
]
```

Only dimensions with size > 1 go into the mesh — a `pp=1, dp_replicate=1,
dp_shard=32, ep=1, tp=4` config produces a 2D mesh with dim names
`("dp_shard", "tp")` and sizes `(8, 4)`.

**Special case — TP or EP active:** when `tp > 1` or `ep > 1` and
`dp_shard` would otherwise be dropped (size 1), it is inserted anyway,
before `ep` and `tp` in the dimension list. FSDP2 needs every
parameter as a DTensor for the fused optimizer path to work; if TP has
already made some params DTensors, FSDP2 has to upgrade the rest —
which requires an (even trivial) `dp_shard` dimension.

If every dimension is `1`, the mesh degenerates to a flat
`("dp_shard",)` mesh of size `world_size` — still present, still what
`get_dp_mesh()` returns.

## Parallelism arithmetic

```
dp_replicate × dp_shard × tp × pp × cp × ep == world_size
```

[`DistributedConfig.validate_world_size`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/distributed.py)
enforces this. The default `dp_shard = -1` is a sentinel meaning
"fill the remaining mesh slots", resolved by `config.resolve(world_size)`
at setup time:

```python
dp_shard = world_size // (dp_replicate * tp * pp * cp * ep)
```

So most configs leave `dp_shard = -1` and let it auto-size — the
arithmetic above then holds by construction. See
[Validation rules § Parallelism arithmetic](../configuration/validation-rules.md#parallelism-arithmetic).

## Extracting sub-meshes

Every apply function reads the sub-mesh it needs:

| Getter | Returns | Used by |
|--------|---------|---------|
| `get_dp_mesh(mesh)` | `mesh["dp_replicate", "dp_shard"]` (HSDP) or whichever single DP dim is present | [`apply_fsdp2`](fsdp2.md) |
| `get_tp_mesh(mesh)` | `mesh["tp"]` or `None` | [`apply_tensor_parallel`](tensor-parallelism.md) |
| `get_pp_mesh(mesh)` | `mesh["pp"]` or `None` | [`build_pipeline_schedule`](pipeline-parallelism.md) |
| `get_pp_rank(mesh)` / `get_pp_size(mesh)` | `pp_mesh.get_local_rank()` / `.size()` or `0` / `1` | `build_stage_module`, training loop |
| `mesh["ep"]` directly | 1D EP mesh | [`apply_expert_parallel`](expert-parallelism.md) |

`get_dp_info(mesh)` (in
[`kempnerforge.distributed.utils`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/utils.py))
collapses the DP axes into `(dp_rank, dp_size)` — the numbers the
metrics tracker uses for "tokens seen across DP ranks".

## Example meshes

### 7B on 32 GPUs, pure FSDP (`7b_32gpu_fsdp.toml`)

```
dp_replicate=1, dp_shard=32, tp=1, pp=1, cp=1, ep=1

mesh: ("dp_shard",) — size (32,)
```

Only FSDP fires. `get_tp_mesh` / `get_pp_mesh` return `None`.

### 70B on 32 GPUs, TP+FSDP (`70b_32gpu_tp4.toml`)

```
dp_replicate=1, dp_shard=8, tp=4, pp=1, cp=1, ep=1

mesh: ("dp_shard", "tp") — size (8, 4)
```

`apply_tensor_parallel` uses the `tp` dim (size 4);
`apply_fsdp2` uses `dp_shard` (size 8).

### 70B on 32 GPUs, TP+PP+FSDP (`70b_32gpu_tp4_pp4.toml`)

```
dp_replicate=1, dp_shard=2, tp=4, pp=4, cp=1, ep=1

mesh: ("pp", "dp_shard", "tp") — size (4, 2, 4)
```

`pp` rank selects the stage, `tp` shards within the stage, `dp_shard`
replicates the remaining across 2 ranks.

### MoE on 32 GPUs, TP+EP+FSDP (`moe_ep_32gpu.toml`)

```
dp_replicate=1, dp_shard=4, tp=4, pp=1, cp=1, ep=2

mesh: ("dp_shard", "ep", "tp") — size (4, 2, 4)
```

`ep` (size 2) splits 8 experts across two groups of 4 ranks each; TP
shards attention within each EP group; FSDP shards the rest across
the `dp_shard` axis. See
[Parallelism recipes § MoE](../reference/parallelism-recipes.md#moe-mixtral-style-8-experts-top-2).

## Context parallelism

`cp` appears in the parallelism arithmetic and the config schema but
there is no `apply_context_parallel` function yet — CP is declared
for validation but not wired. Attempting `cp > 1` passes the
arithmetic check but produces no actual sequence parallelism. See the
[planning issue](https://github.com/KempnerInstitute/KempnerForge/issues)
for status.

## One mesh, one barrier

`init_distributed` calls `dist.barrier()` after `init_device_mesh`
returns, so all ranks complete mesh construction before the training
loop starts. This is the only synchronization point guaranteed during
setup — subsequent `apply_*` calls assume the mesh is already
populated.

## See also

- [FSDP2](fsdp2.md) — how `apply_fsdp2` consumes `get_dp_mesh(mesh)`.
- [Tensor parallelism](tensor-parallelism.md) — how `apply_tensor_parallel`
  consumes the `tp` sub-mesh.
- [Pipeline parallelism](pipeline-parallelism.md) — how the `pp`
  sub-mesh selects each rank's stage.
- [Configuration § DistributedConfig](../configuration/config-sections.md) —
  the dataclass with `dp_replicate`, `dp_shard`, `tp`, `pp`, `cp`, `ep`.
- [Environment variables](../reference/environment-variables.md) — how
  `RANK` / `LOCAL_RANK` / `WORLD_SIZE` feed into `get_world_info()`.
