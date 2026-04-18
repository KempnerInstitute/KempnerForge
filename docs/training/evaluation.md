# Evaluation

KempnerForge runs evaluation two ways: **in-loop** (every
`eval_config.interval` training steps) and **standalone** (via
[`scripts/eval.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/eval.py)).
Both call the same `run_eval` from
[`kempnerforge/training/eval.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/eval.py).

## `EvalConfig`

```toml
[eval]
enabled = true
interval = 1000                         # run eval every N training steps
steps = 50                              # batches per evaluation

# Pre-tokenized data:
dataset_path = "/path/to/tokenized/eval"
file_pattern = "*.npy"

# Or HuggingFace streaming:
hf_dataset_name = "wikitext"
hf_dataset_config = "wikitext-103-raw-v1"
hf_dataset_split = "validation"
```

Validation runs in `__post_init__`: `interval` and `steps` must be
positive. The `dataset_path` / `hf_dataset_name` choice is made by the
loader — if both are set, pre-tokenized takes precedence.

## `run_eval`

```python
@torch.no_grad()
def run_eval(
    model, eval_dataloader, loss_fn, device, eval_steps,
    *, pp_schedule=None, pp_rank=None, pp_size=None, pp_group=None,
) -> dict[str, float]:
```

Returns

```python
{"eval/loss": avg_loss, "eval/perplexity": math.exp(min(avg_loss, 20.0))}
```

The `min(avg_loss, 20.0)` caps perplexity at `e^20 ≈ 5e8` so an early
NaN-adjacent loss doesn't overflow the WandB / TensorBoard logger.

`run_eval` toggles `model.eval()` on entry and `model.train()` on
exit, so the training loop can call it mid-training without leaving the
model in eval mode.

### Standard path (no PP)

Loops over `eval_steps` batches:

```python
for _ in range(eval_steps):
    batch = next(eval_iter)
    logits = model(input_ids)
    loss   = loss_fn(logits, labels)
    total_loss += loss.item()
avg_loss = total_loss / eval_steps
```

If the eval dataloader runs out before `eval_steps`, it cycles — the
`try/except StopIteration` refreshes `eval_iter`. So `eval_steps` can
exceed the eval set's length without error, but the extra steps will
repeat already-seen batches.

### PP path (`pp_schedule` provided)

Mirrors the [PP training step](training-loop.md#pp-step-pp_enabled-is-true):
collects `eval_steps` batches into full tensors, feeds through
`pp_schedule.step(..., losses=pp_losses)`, averages the per-microbatch
losses on the last stage, then broadcasts a single scalar across the
PP group:

```python
loss_tensor = torch.tensor([avg_loss], device=device)
dist.broadcast(loss_tensor, group_src=pp_size - 1, group=pp_group)
```

All PP stages see the same `avg_loss` after the broadcast, which is
what the metrics tracker expects.

## In-loop evaluation

The training loop fires eval at the top of the periodic-work section:

```python
if eval_config.enabled and eval_dataloader is not None \
        and step % eval_config.interval == 0:
    eval_metrics = run_eval(
        model, eval_dataloader, loss_fn, device, eval_config.steps,
        pp_schedule=pp_schedule, pp_rank=pp_rank, pp_size=pp_size,
        pp_group=pp_group,
    )
    tracker.log_eval(eval_metrics, step)
    hook_runner.on_eval_end(eval_metrics, step)
```

Cadence tuning:

- `interval = 1000` with `steps = 50` means eval runs ~50 batches
  every 1000 training steps. Keep `steps × batch_size` below 1% of your
  training step cost or you'll pay it in visible wall time.
- `steps` is independent of the eval set size — the loader cycles, so
  small eval sets with large `steps` will repeat batches (biases the
  estimate toward the loop position).

## Standalone evaluation

`scripts/eval.py` runs `run_eval` without training. It:

1. Loads the same `JobConfig` you trained with (so model
   architecture, tokenizer, parallelism mesh match).
2. Forces `config.eval.enabled = True`.
3. Builds the model **without pipeline parallel** (PP-eval in
   `eval.py` is not implemented — standalone eval assumes single
   forward).
4. Creates a dummy `torch.optim.SGD(lr=0.0)` — unused, but required by
   `CheckpointManager.load()`'s signature.
5. Loads the checkpoint with `exclude_keys=["optimizer"]` so the
   dummy SGD state isn't overwritten.
6. Builds the eval dataloader (pre-tokenized or HF-streaming).
7. Calls `run_eval` and prints the results + a JSON line for
   programmatic consumption.

Usage:

```bash
uv run torchrun --nproc_per_node=4 scripts/eval.py configs/train/7b.toml \
    --checkpoint.load_path /path/to/checkpoint \
    --eval.dataset_path /path/to/eval_tokens \
    --eval.steps 200
```

The stdout-final JSON line (`{"step": ..., "tokens_seen": ...,
"eval/loss": ..., "eval/perplexity": ...}`) is designed to be parsed —
pipe it through `jq` for batch-evaluation sweeps.

## See also

- [Training loop § Periodic work](training-loop.md#periodic-work) —
  where in-loop eval fires and in what order.
- [Hooks § `on_eval_end`](hooks.md#lifecycle-events) — how to append
  custom logic to every eval round.
- [Configuration § EvalConfig](../configuration/config-sections.md) —
  every field with its default.
