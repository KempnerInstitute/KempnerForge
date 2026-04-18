# Run evaluation

Three entry points, three use cases:

| Entry point | When to use |
|-------------|-------------|
| `[eval]` section in training config | Periodic eval loss during a training run |
| `scripts/eval.py` | Standalone eval on a saved checkpoint — quick loss/perplexity on new data |
| `scripts/eval_harness.py` | Downstream benchmarks (HellaSwag, ARC, MMLU, …) via lm-eval-harness |

All three compute through the same underlying
[`run_eval`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/eval.py)
function when they need loss; the harness path additionally converts
the DCP checkpoint to HuggingFace format and hands off to `lm_eval`.

## Path 1: eval inside the training loop

Flip on `[eval]` in your TOML:

```toml
[eval]
enabled              = true
interval             = 500          # steps between evals
steps                = 50           # batches per eval
dataset_path         = "/data/eval_set"
file_pattern         = "*.bin"
# OR — HuggingFace eval data:
# hf_dataset_name    = "wikitext"
# hf_dataset_config  = "wikitext-103-raw-v1"
# hf_dataset_split   = "validation"
```

Every `interval` steps, the training loop pauses the optimizer,
switches the model to eval mode, iterates `steps` batches of held-out
data, and logs `eval/loss` and `eval/perplexity` through
[`MetricsTracker.log_eval`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/tracker.py).
The stdout backend renders one compact line per call:

```
[step 500] eval/loss=3.8200 | eval/perplexity=45.6000
```

The eval dataloader is a plain `DataLoader` with a `DistributedSampler`
(not `StatefulDataLoader`). It's deterministic because it doesn't
shuffle, and it's allowed to reset on epoch boundary — eval iteration
uses `StopIteration` → re-init to wrap around if `steps` exceeds the
dataset.

### Perplexity is capped

```python
{"eval/loss": avg_loss, "eval/perplexity": math.exp(min(avg_loss, 20.0))}
```

`exp(20) ≈ 4.85e8`. When the model hasn't converged yet and loss is
huge, `exp` would overflow — so the clamp keeps the number finite.
Treat `perplexity ≈ 5e8` as "loss is still blowing up," not a real
perplexity.

### PP integration

`run_eval` auto-detects pipeline parallelism via its `pp_schedule`
argument. On PP stages, eval runs through the same
`schedule.step(input, target, losses)` machinery as training; the
last stage broadcasts the loss back to rank 0 so logging stays
consistent.

## Path 2: standalone eval on a checkpoint

```bash
# Single GPU
uv run python scripts/eval.py configs/train/7b.toml \
    --checkpoint.load_path=checkpoints/7b/step_10000 \
    --eval.dataset_path=/data/eval_set \
    --eval.steps=100

# Multi-GPU (FSDP for large models)
uv run torchrun --nproc_per_node=4 scripts/eval.py configs/train/7b.toml \
    --checkpoint.load_path=checkpoints/7b/step_10000 \
    --eval.dataset_path=/data/eval_set
```

[`scripts/eval.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/eval.py)
is the training script minus the optimizer:

- Same `load_config` + `init_distributed` + `build_parallel_model`
  path, so FSDP / TP configurations work unchanged
- Loads the checkpoint via `CheckpointManager(...)` with
  `exclude_keys=["optimizer"]` — model + RNG only
- Runs `run_eval(model, dataloader, loss_fn, device, eval_steps)` and
  prints results to stdout + JSON

Output:

```
==================================================
Evaluation Results (step 10000)
==================================================
  eval/loss: 2.4321
  eval/perplexity: 11.3856
==================================================

{"step": 10000, "tokens_seen": 4194304000, "eval/loss": 2.4321, ...}
```

The JSON dump at the end is useful for scripting — redirect to a file
and diff across checkpoints.

### HuggingFace eval data path

```bash
uv run python scripts/eval.py configs/train/7b.toml \
    --checkpoint.load_path=checkpoints/7b/step_10000 \
    --eval.hf_dataset_name=wikitext \
    --eval.hf_dataset_config=wikitext-103-raw-v1
```

Rank 0 tokenizes the full eval split (via
[`HuggingFaceDataset`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/dataset.py))
and broadcasts packed sequences to all other ranks — fine for
benchmark-sized eval sets, avoid for anything multi-GB.

### Pipeline parallelism is not supported here

The standalone script's model build goes through `build_parallel_model`,
not the PP stage builder. If you need PP eval, drive it from the
training loop (Path 1).

## Path 3: lm-eval-harness for downstream tasks

The training-loss eval is useful for training signal; for downstream
benchmarks you want lm-eval-harness.

```bash
# Install the extra (lm-eval is optional — not default dep)
uv add lm-eval

# Run default task suite: hellaswag, arc_easy, arc_challenge,
# winogrande, piqa, boolq
uv run python scripts/eval_harness.py \
    --checkpoint checkpoints/7b/step_10000 \
    --config    configs/train/7b.toml

# Specific tasks
uv run python scripts/eval_harness.py \
    --checkpoint checkpoints/7b/step_10000 \
    --config    configs/train/7b.toml \
    --tasks     hellaswag,mmlu,arc_easy

# Pre-converted HF model — skip DCP conversion
uv run python scripts/eval_harness.py \
    --hf-model ./exports/my_model \
    --tasks    hellaswag
```

[`scripts/eval_harness.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/eval_harness.py)
does three things:

1. Converts the DCP checkpoint to HuggingFace format via
   `dcp_to_hf()` from `scripts/convert_checkpoint.py`, into a tempdir
2. Calls `lm_eval.simple_evaluate(model="hf", model_args=...)` with
   the task list
3. Prints results and optionally writes JSON via `--output`

### Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--checkpoint` | — | DCP checkpoint dir (will be converted) |
| `--config` | — | TOML (required with `--checkpoint` to resolve model architecture) |
| `--hf-model` | — | Pre-converted HF dir, skip conversion |
| `--tasks` | `hellaswag,arc_easy,arc_challenge,winogrande,piqa,boolq` | comma-sep tasks |
| `--batch-size` | `8` | eval batch size |
| `--num-fewshot` | `None` (task default) | override few-shot count |
| `--output` | `None` | save full JSON results |

### Conversion caveat

DCP → HF conversion is a one-time cost per checkpoint — the full
model is materialized on one device, keys are remapped, and weights
are written as safetensors. Time scales with model size and filesystem
throughput, so larger checkpoints on networked storage take longer. If
you're scanning many checkpoints with the harness, convert once
manually and re-use the output with `--hf-model`:

```bash
uv run python scripts/convert_checkpoint.py dcp-to-hf \
    --dcp-dir checkpoints/7b/step_10000 \
    --hf-dir  exports/7b_step_10000 \
    --config  configs/model/llama_7b.toml

uv run python scripts/eval_harness.py \
    --hf-model exports/7b_step_10000 \
    --tasks hellaswag,mmlu
```

## Picking a path

| Goal | Path |
|------|------|
| Track training signal as loss curve | Path 1 (in-training) |
| Loss on a specific held-out set for a specific checkpoint | Path 2 (`scripts/eval.py`) |
| Downstream accuracy — HellaSwag / MMLU / ARC | Path 3 (`scripts/eval_harness.py`) |
| Quick sanity on small datasets | Path 2 |
| Scanning 20 checkpoints for best MMLU | Path 3 with pre-conversion |

The three paths don't overlap in outputs: Path 1/2 report `eval/loss`
+ perplexity; Path 3 reports task-specific metrics (accuracy, acc-norm,
pass@k). You'll often run both — loss for the monitoring signal,
harness for final reporting.

## See also

- [Training § Evaluation](../training/evaluation.md) — `run_eval`
  implementation notes.
- [Configuration § `[eval]`](../configuration/config-sections.md) —
  every `EvalConfig` field with its default.
- [Checkpointing § DCP model format](../checkpointing/dcp-model.md) —
  what the DCP → HF converter reads.
- [End-to-end training run](end-to-end-training-run.md) — the source
  of the checkpoints the three paths consume.
- [`scripts/eval.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/eval.py)
  and
  [`scripts/eval_harness.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/eval_harness.py)
  — the scripts this page documents.
