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
| `metadata.json` | Human-readable `{"step": N, "tokens_seen": M}`, plus the logging runs live at save time | Plain JSON, see below |
| `latest` symlink | Points at the most recent `step_N` (updated atomically) | [Auto-resume](auto-resume.md) |

## `metadata.json`

Written by rank 0 beside `train_state.pt`, and readable with a plain `json.load` (no torch).
It holds `step` and `tokens_seen` (VLM runs add `vlm_freeze`). When a logging backend has a
live run at save time it also holds a `tracking` list, one entry per backend:

```json
{
  "step": 1000,
  "tokens_seen": 4096000,
  "tracking": [
    {"backend": "wandb", "run_id": "<run id>", "experiment": "<project>",
     "entity": "<entity>", "url": "<run url>"},
    {"backend": "tensorboard", "log_dir": "tb_logs"},
    {"backend": "mlflow", "run_id": "<run id>", "experiment": "<experiment>",
     "tracking_uri": "<tracking uri>"}
  ]
}
```

- Each backend describes its own run, as it knows it: the resolved tracking URI and the
  experiment name for MLflow, the project for wandb. `entity` and `url` appear when the wandb
  run has them (an offline run has no URL). The MLflow entry always has all four fields.
- A backend contributes only while its run is live, and runs start on the first log. A
  checkpoint saved before that (step 0, or soon after a resume when `checkpoint.interval` is
  shorter than `metrics.log_interval`) lacks the entry, and the key is omitted when no backend
  is live.
- Values are recorded as resolved, the tracking URI included: keep credentials out of it
  (the environment supplies them), and drop the `tracking` key before sharing a checkpoint
  outside your team.
- The key is additive. Older checkpoints lack it, nothing on the resume path reads it, and run
  ids still resume from `train_state.pt`.

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
  (search for `interval`, `dyn_ckpt_window`, `async_mode`, `keep_last_n`, `load_path`).
