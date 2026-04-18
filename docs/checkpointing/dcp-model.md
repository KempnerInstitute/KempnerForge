# DCP model + optimizer

KempnerForge uses
[`torch.distributed.checkpoint`](https://pytorch.org/docs/stable/distributed.checkpoint.html)
(DCP) to save and load model and optimizer state. DCP is designed
for sharded state — each rank writes only its local slice, the
reader automatically reshards into whatever parallelism the loader
has. No "rank 0 gathers everything" step.

Entry point:
[`CheckpointManager.save()`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/checkpoint/manager.py)
in `kempnerforge/checkpoint/manager.py`.

## What goes into the DCP shard

```python
dcp_state = {
    "model":     self.model.state_dict(),
    "optimizer": self.optimizer.state_dict(),
}
self._async_ckpt.save(
    dcp_state, checkpoint_id=str(dcp_dir), process_group=self._process_group
)
```

Two top-level keys — `"model"` and `"optimizer"`. DCP introspects the
state dicts, finds `DTensor` / `ShardedTensor` parameters, and
writes each shard to disk with enough metadata to reassemble.

What's in each:

- **`model.state_dict()`** — every parameter and buffer: weights,
  RMSNorm scales, learned RoPE frequencies (if present), and any
  registered buffer. Under FSDP2 these are `DTensor`s; under TP
  they're `DTensor`s on a 2D mesh. DCP handles both.
- **`optimizer.state_dict()`** — AdamW's `exp_avg`, `exp_avg_sq`,
  `step` counters; Lion's `exp_avg`; Muon's internal state. All
  per-parameter tensors live on the same device and parallelism
  shape as the parameter, so DCP saves them symmetrically.

Not in the DCP shard: scheduler, dataloader, RNG, and training
metadata. Those go in `train_state.pt` alongside —
[see Train state](train-state.md).

## On-disk layout

```
checkpoints/
├── step_1000/
│   ├── .metadata                         ← DCP global manifest
│   ├── __0_0.distcp                      ← rank-0 model shard
│   ├── __1_0.distcp                      ← rank-1 model shard
│   ├── ...                               ← one `.distcp` per rank
│   ├── train_state.pt                    ← non-distributed state
│   └── metadata.json                     ← human-readable step info
├── step_2000/                            ← same layout
└── latest -> step_2000                   ← symlink
```

`.distcp` files are the actual tensor storage (one per rank, written
in parallel). `.metadata` is the global index that lets the reader
figure out which `.distcp` file contains what.

### With pipeline parallelism

PP makes each stage hold a different set of parameters — DCP needs
disjoint shards per stage, so `save()` writes to a per-stage
subdirectory:

```python
# manager.py
dcp_dir = ckpt_dir / f"pp{self._pp_rank}" if self._pp_rank is not None else ckpt_dir
```

```
checkpoints/step_1000/
├── pp0/                           ← stage 0 shards (embedding + first layers)
│   ├── .metadata
│   └── __*_0.distcp
├── pp1/                           ← stage 1 shards
│   ├── .metadata
│   └── __*_0.distcp
├── pp2/                           ← ...
├── pp3/                           ← last stage (final norm + output head)
└── train_state.pt                 ← one per checkpoint, written by global rank 0
```

Each stage also gets a process group scoped to that stage's DP + TP
ranks:

```python
# scripts/train.py
non_pp_dims = [d for d in device_mesh.mesh_dim_names if d != "pp"]
if len(non_pp_dims) == 1:
    ckpt_pg = device_mesh[non_pp_dims[0]].get_group()
elif len(non_pp_dims) > 1:
    ckpt_pg = device_mesh[tuple(non_pp_dims)].get_group()
ckpt_mgr = CheckpointManager(config.checkpoint, model, optimizer,
                             process_group=ckpt_pg, pp_rank=pp_rank)
```

A 1-D mesh slice has to be indexed by the dim name directly;
`tuple(...)` wrapping is only valid for ≥2 dims. Both branches land
on the same thing semantically — every rank at the same PP position
coordinating together.

Without the scoped group, DCP would try to coordinate across all
world ranks (including other PP stages), and the stage-0 ranks
would hang waiting for stage-1's shards.

## Save modes

[`AsyncCheckpointer`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/checkpoint/async_save.py)
wraps DCP's `save` / `async_save` behind a mode flag:

| Config | Behavior | Use |
|--------|----------|-----|
| `async_mode = "disabled"` | `dcp.save()` — blocking | Simple, debugging, small models |
| `async_mode = "async"` | `dcp.async_save()` — snapshots to CPU, writes in background | Default for production |
| `async_mode = "async_with_pinned_mem"` | Async with pinned-memory staging (faster GPU→CPU copy) | Very large models where GPU→CPU throughput bottlenecks the snapshot |

Every `save()` call first does `self.wait()` — the previous async
save must fully complete before starting a new one. This avoids
holding two CPU snapshots at once and avoids racing on the same
directory.

The returned future is an `AsyncCheckpointerFuture`; `wait()` calls
`.result()` which blocks until the background writer has flushed to
disk. Training calls `ckpt_mgr.wait()` once before shutdown
(`scripts/train.py` line ~788) to flush any pending save.

## Process groups

The `process_group=` kwarg on `dcp.save` and `dcp.load` scopes the
all-gather / all-reduce calls that DCP uses internally. Rules:

- Non-PP: use the default global group (`process_group=None`). Every
  rank holds a slice of the same state.
- PP: use a per-stage group. Stage `i`'s ranks (all DP × TP ranks at
  PP position `i`) coordinate alone — they produce one DCP shard set
  under `pp{i}/`.

`CheckpointManager` stores the group at construction time and passes
it through on every save/load.

## Loading

Load is the mirror image of save:

```python
dcp_state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
dcp.load(dcp_state, checkpoint_id=str(dcp_dir), process_group=self._process_group)
self.model.load_state_dict(dcp_state["model"])
self.optimizer.load_state_dict(dcp_state["optimizer"])
```

The first `state_dict()` call gives DCP the **shape** to fill — it
doesn't contain the saved data, just the tensor metadata DCP needs
to know what to load where. `dcp.load` mutates the tensors in place
with the loaded values. Then `load_state_dict` consumes them.

Loading with a different GPU count triggers DCP's automatic
resharding — see [Resharding](resharding.md).

To skip loading the optimizer (e.g. for fine-tuning), pass
`exclude_keys=["optimizer"]`:

```python
ckpt_mgr.load(path=..., exclude_keys=["optimizer"])   # scripts/eval.py does this
```

## See also

- [Resharding](resharding.md) — the save-at-N, load-at-M mechanics
  that make DCP worth the extra files.
- [Train state](train-state.md) — what else is in each checkpoint
  directory.
- [Auto-resume](auto-resume.md) — how KempnerForge finds the right
  `step_N` on restart.
- [HF conversion](hf-conversion.md) — exporting DCP shards to
  HuggingFace safetensors.
- [Configuration § CheckpointConfig](../configuration/config-sections.md) —
  `async_mode`, `interval`, `keep_last_n`, `dir`.
