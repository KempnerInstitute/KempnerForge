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
optim_state = get_optimizer_state_dict(self.model, self.checkpointed_optimizers)
dcp_state = {
    "model":                   get_model_state_dict(self.model),
    "optimizer":               optim_state,
    "optimizer_never_stepped": _never_stepped_record(optim_state),
}
self._async_ckpt.save(
    dcp_state, checkpoint_id=str(dcp_dir), process_group=self._process_group
)
```

`checkpointed_optimizers` is the optimizer plus any inner optimizers it
hands parameters to — see [Inner optimizers](#inner-optimizers).

`get_model_state_dict` / `get_optimizer_state_dict` are the DCP-aware
helpers from `torch.distributed.checkpoint.state_dict` — **not** raw
`model.state_dict()` / `optimizer.state_dict()`. They key the
optimizer state by parameter fully-qualified name (not positional
index) and keep the FSDP/DTensor sharding intact, which is what makes
load (and resharding) line up by name. See [Loading](#loading) for why
the raw calls break resume.

Three top-level keys. DCP introspects the state dicts, finds
`DTensor` / `ShardedTensor` parameters, and writes each shard to disk
with enough metadata to reassemble. The third is empty whenever every
parameter has optimizer state, and an empty entry writes no key at all,
so an ordinary run's checkpoints keep the two-key shape any reader
already expects. A save before the first optimizer step writes the model
alone (see below).

What's in each:

- **model state** — every parameter and buffer: weights,
  RMSNorm scales, learned RoPE frequencies (if present), and any
  registered buffer. Under FSDP2 these are `DTensor`s; under TP
  they're `DTensor`s on a 2D mesh. DCP handles both.
- **optimizer state** — AdamW's `exp_avg`, `exp_avg_sq`,
  `step` counters; Lion's `exp_avg`; Muon's `momentum_buffer`s plus
  its inner AdamW's moments. All per-parameter tensors live on the
  same device and parallelism shape as the parameter, so DCP saves
  them symmetrically.
- **never-stepped record** — one entry per parameter the optimizer
  holds without any state, i.e. one that had not yet received a
  gradient. An optimizer creates per-parameter state lazily, so there
  is nothing to save for those, and once the checkpoint is written
  that is indistinguishable from state that went missing. The record
  keeps the distinction for the load — see [Loading](#loading).

Before the first optimizer step (the step-0 checkpoint of a
`dyn_ckpt_window` from step 0) none of the `checkpointed_optimizers`
holds state, so the save writes neither the optimizer nor the record.
There is nothing to save then, and reading the optimizer would create
state: `get_optimizer_state_dict` first gives an optimizer that holds
none a dummy `lr=0` step, which leaves AdamW one step ahead in its bias
correction and makes `ScheduleFreeAdamW` divide by zero. A load that
finds no optimizer in a checkpoint — this one, or one holding model
weights alone — loads it like a pretrained model: the weights and train
state, with every checkpointed optimizer reset to its freshly built
state. It reads that from the DCP metadata before building any optimizer
template, so no optimizer takes the dummy step. Past step 0 it warns,
because the run's optimizer moments are then missing and the resume
does not continue it exactly.

Not in the DCP shard: scheduler, dataloader, RNG, and training
metadata. Those go in `train_state.pt` alongside —
[see Train state](train-state.md).

### Inner optimizers

`get_optimizer_state_dict` keeps only an optimizer's own `state` and
`param_groups`; any other key its `state_dict()` returns is dropped.
Muon hands the parameters Newton-Schulz cannot take — norms and
biases, plus the embedding and output head whenever
`vocab_size / dim > 10` — to an inner `torch.optim.AdamW` that its own
`param_groups` do not list, so passing Muon alone would leave that
AdamW's moments out of the checkpoint and restart them from zero on
every resume.

`Muon.inner_optimizers` names the inner AdamW, and the manager's
`checkpointed_optimizers` — `[optimizer, *optimizer.inner_optimizers]` —
go to both the getter and the setter. Their state shares the `"optimizer"` entry, keyed by parameter
name like the rest, with the inner AdamW's param groups after Muon's —
so a reader that passes Muon alone still finds Muon's state where it
expects it.

A Muon checkpoint saved before the inner AdamW was checkpointed holds
only Muon's own param groups. The load detects that from the DCP
metadata, restores Muon alone, leaves the inner AdamW fresh — as every
resume from such a checkpoint did before — and logs a warning that the
resume is not bit-exact. The next checkpoint it writes is complete.

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
# training/entry.py::build_checkpoint_manager
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
(`training/loop.py::run_training_loop`) to flush any pending save.

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
dcp_state = {
    "model":     get_model_state_dict(self.model),
    "optimizer": get_optimizer_state_dict(self.model, self.checkpointed_optimizers),
}
dcp.load(dcp_state, checkpoint_id=str(dcp_dir), process_group=self._process_group)
set_model_state_dict(self.model, dcp_state["model"])
set_optimizer_state_dict(
    self.model, self.checkpointed_optimizers, optim_state_dict=dcp_state["optimizer"]
)
```

A Muon checkpoint saved before the inner AdamW was checkpointed is the
one exception: the load passes Muon alone — see
[Inner optimizers](#inner-optimizers).

The getter call gives DCP the **shape** to fill — it doesn't contain
the saved data, just the tensor metadata (and, crucially, the
*allocated* optimizer moment tensors) DCP needs to know what to load
where. `dcp.load` mutates those tensors in place with the loaded
values; the setters then write them back into the live model and
optimizer.

> **Why the DCP-aware helpers, not `optimizer.state_dict()`?** On
> resume the optimizer is freshly built, so `optimizer.state_dict()`
> has *no* `exp_avg` / `exp_avg_sq` tensors yet — AdamW creates the
> per-parameter state lazily on the first `.step()`. Passing that
> empty dict as the load template gives `dcp.load` nothing to fill, so
> the saved moments are silently dropped and Adam momentum resets to
> zero at every resume (a non-bit-exact resume). `get_optimizer_state_dict`
> allocates the moment tensors up front in the right sharded layout, so
> `dcp.load` repopulates them. The model side would work with either
> call — its parameters are always allocated — but we use the matching
> getter/setter for symmetry.

The template allocates state for every trainable parameter, while the
checkpoint has none for the parameters recorded as never stepped. A
load planner drops exactly those from the template and gives them
empty state afterwards, so a run in which part of the model has not
yet been reached resumes. Every other state the template asks for
stays required: a parameter with missing or partial state, or one the
saved optimizer did not hold, fails the load.

A checkpoint written before the record existed carries no such evidence,
and an absent state cannot be told apart from one that went missing, so
a never-stepped parameter in one of those still fails the load. Start
from it with `[checkpoint].load_path` and
`exclude_from_loading = ["optimizer"]`, which restores the weights and
begins with a fresh optimizer.

The template also holds every param-group key the optimizer has now,
and DCP requires each one. An optimizer whose param groups gained a key
after its checkpoints were first written lists it in
`LEGACY_GROUP_DEFAULTS`, with the value those checkpoints were trained
with: Muon lists `orthogonalize` as `"local"`, `nesterov` as `False` and
`adjust_lr` as `"original"`, the update it made before those options
existed. Where a checkpoint lacks a listed key, the load drops it
from the template, the optimizer fills in the listed value as it loads
the groups, and the manager logs it. A checkpoint that lacks any other
key still fails the load.

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
