# Checkpointing

Distributed checkpoints via `torch.distributed.checkpoint` (DCP):
what's saved, how resharding works, auto-resume rules, and
HuggingFace interchange.

```{toctree}
:maxdepth: 1

dcp-model
resharding
train-state
auto-resume
hf-conversion
```

## At a glance

Every checkpoint lands in `{config.checkpoint.dir}/step_{N}/` and
contains two kinds of state:

| File(s) | Contents | Format |
|---------|----------|--------|
| DCP shards (`.distcp` + `.metadata`) | Model + optimizer state, one shard per rank | [DCP § Model + optimizer](dcp-model.md) |
| `train_state.pt` | step, tokens_seen, scheduler, RNG, extras (e.g. `phase_idx`, `wandb_run_id`) | [Train state](train-state.md) |
| `metadata.json` | Human-readable `{"step": N, "tokens_seen": M}` | Plain JSON |
| `latest` symlink | Points at the most recent `step_N` (updated atomically) | [Auto-resume](auto-resume.md) |

## Key modules

- [`kempnerforge/checkpoint/manager.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/checkpoint/manager.py)
  — `CheckpointManager.save()`/`load()`/`wait()`, `latest` symlink
  maintenance, retention cleanup.
- [`kempnerforge/checkpoint/async_save.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/checkpoint/async_save.py)
  — `AsyncCheckpointer`: sync / async / pinned-memory modes.
- [`kempnerforge/checkpoint/state.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/checkpoint/state.py)
  — `build_train_state` / `restore_train_state`, RNG capture.
- [`kempnerforge/resilience/elastic.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/elastic.py)
  — `resolve_resume_path()` (checks the `latest` symlink and falls
  back to the highest `step_N`).
- [`scripts/convert_checkpoint.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/convert_checkpoint.py)
  — `dcp-to-hf` and `hf-to-dcp` CLI.

## Read next

- **New reader**: [DCP model + optimizer](dcp-model.md) →
  [Train state](train-state.md).
- **Resuming a job**: [Auto-resume](auto-resume.md) first, then
  [Resharding](resharding.md) if you're changing GPU count.
- **Exporting for inference or HF checkpoints**:
  [HF conversion](hf-conversion.md).
- **Config knobs**:
  [Configuration § CheckpointConfig](../configuration/config-sections.md)
  (search for `interval`, `async_mode`, `keep_last_n`, `load_path`).
