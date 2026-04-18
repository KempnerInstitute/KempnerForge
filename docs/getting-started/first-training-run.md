# Your First Training Run

{doc}`quickstart` gave you the commands. This page slows down the single-GPU
debug run and explains what the log line means, what ends up in the
checkpoint directory, and how to resume.

## Run it

```bash
uv run python scripts/train.py configs/train/debug.toml \
  --checkpoint.dir=/tmp/kf_first_run
```

This loads
[`configs/train/debug.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/debug.toml):
a 256-dim, 4-layer model (~20M params), synthetic data, batch size 4,
sequence length 512, 100 steps, cosine schedule with 10 warmup steps,
AdamW at `lr=3e-4`. It runs in under a minute on a single H100.

## What the log line means

You will see one metrics line every 5 steps (controlled by
`metrics.log_interval`):

```
[step 10] loss=9.1800 | lr=3.00e-04 | grad_norm=1.340 | tok/s=42,000 | mfu=18.2% | mem=2.1/80GB | step_time=0.15s
```

| Field | What it is |
|-------|------------|
| `step` | Optimizer step count. Matches checkpoint filenames. |
| `loss` | Training cross-entropy for this step. |
| `lr` | Current learning rate after the scheduler applies. Ramps up for 10 steps, then cosine-decays. |
| `grad_norm` | L2 norm of gradients before clipping (returned by `clip_grad_norm_`). Clipping to `train.grad_clip_norm` (default 1.0) is applied in-place before the optimizer step. |
| `tok/s` | Tokens per second throughput, this step. |
| `mfu` | Model FLOPs Utilization — achieved FLOPs / theoretical peak for the GPU. |
| `mem` | Peak GPU memory this step / total GPU memory. |
| `step_time` | Wall-clock time spent on this step, in seconds. |

Synthetic data means loss starts near `log(vocab_size) ≈ 10.4` and falls
slowly because the "dataset" is random tokens — there is no real signal to
fit. The point of this run is to exercise the pipeline, not to learn
anything.

## What's in the checkpoint directory

With `checkpoint.interval=50` and `checkpoint.keep_last_n=2` from `debug.toml`,
after the run you'll have:

```
/tmp/kf_first_run/
├── step_50/               # Full DCP checkpoint at step 50
├── step_100/              # Full DCP checkpoint at step 100
└── latest → step_100/     # Symlink to the most recent checkpoint
```

Each `step_N/` is a directory (not a single file) because DCP shards
parameters across ranks. On a single-GPU run there is one shard; on
multi-GPU FSDP there is one shard per rank.

`latest` is a symlink updated atomically after each save. Auto-resume reads
this symlink; if it's missing and `checkpoint.load_path` isn't set, the run
starts from scratch.

## Resume

Kill the run partway (`Ctrl-C`) and re-launch the same command:

```bash
uv run python scripts/train.py configs/train/debug.toml \
  --checkpoint.dir=/tmp/kf_first_run
```

`train.py` detects the `latest` symlink, loads model + optimizer + scheduler
+ RNG + dataloader position, and continues from the next step. No flags
needed — auto-resume is the default when `checkpoint.dir` contains a valid
checkpoint.

To explicitly point at a different checkpoint:

```bash
uv run python scripts/train.py configs/train/debug.toml \
  --checkpoint.load_path=/tmp/kf_first_run/step_50
```

## Change something and re-run

Pick one and re-run with a **fresh `--checkpoint.dir`** (shape changes break
auto-resume). Start with:

- **More steps, same model**: `--train.max_steps=500`. Watch MFU stabilize
  past the warmup region.
- **Bigger model**: `--model.dim=512 --model.n_layers=8`. You'll see loss
  curves change and memory go up.
- **Longer context**: `--train.seq_len=2048 --model.max_seq_len=2048`. tok/s
  drops (attention is quadratic) but MFU typically rises.
- **Real data**: point `--data.dataset_path` at pre-tokenized `.npy` shards.
  See {doc}`quickstart` step 4.
- **Different optimizer**: `--optimizer.name=muon`. Note the LR is not
  transferable between optimizers — Muon expects different LRs than AdamW.

## What's next

- {doc}`quickstart` covers multi-GPU, MoE, and hooks if you skipped them.
- {doc}`notebooks` has interactive examples for model inspection,
  activation extraction, and MoE routing diagnostics.
- Production configs (7B, 13B, 70B) live in
  [`configs/train/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/configs/train);
  scale them up with
  [`scripts/slurm/multinode.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/slurm/multinode.sh).
