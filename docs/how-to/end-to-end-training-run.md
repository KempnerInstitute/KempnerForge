# End-to-end training run

This is the flagship how-to — the one that takes you from a clean
checkout to a trained checkpoint and a sampled completion, using
only what's in the repo. If you can finish this page, every other
how-to is an expansion.

Runnable plan:

1. Install the environment.
2. Cache the tokenizer.
3. Launch a 1-GPU run to confirm the loop works.
4. Scale to 4 GPUs on one node via `torchrun`.
5. Kill the job; auto-resume from the last checkpoint.
6. Generate text from the checkpoint.

The reference config we use throughout is
[`configs/train/hf_wikitext.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/hf_wikitext.toml)
— a small (~40M-param) model that streams Wikitext-103 from
HuggingFace. No dataset setup required.

## 1. Install

```bash
git clone https://github.com/KempnerInstitute/KempnerForge.git
cd KempnerForge
uv sync           # creates .venv and installs all deps
```

`uv sync` installs PyTorch, transformers, datasets, and the rest. If
`uv` isn't on the machine, install it:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

## 2. Cache the tokenizer

The reference config uses the GPT-2 tokenizer. Compute nodes typically have restricted or much slower 
internet access (~1 Gbps vs. ~100 Gbps on login nodes), so it’s best to pre-cache it on the login node:

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"
```

Cached under `~/.cache/huggingface/`. See
[Prepare tokenized data § Cache the tokenizer first](prepare-tokenized-data.md#cache-the-tokenizer-first).

## 3. Single-GPU sanity check

```bash
uv run python scripts/train.py configs/train/hf_wikitext.toml
```

What this does:

- `load_config` reads the TOML into a `JobConfig`.
- `init_distributed` initializes process group (single-rank group on
  1 GPU — still uses the distributed path).
- `build_parallel_model` constructs the `Transformer` and applies
  FSDP2 (`dp_shard = -1` auto-resolves to 1 on one rank).
- The training loop streams Wikitext-103, computes cross-entropy,
  steps AdamW with a cosine schedule, and checkpoints to
  `checkpoints/hf_wikitext/step_N` every 100 steps.

Expected output (first few lines):

```
[rank 0] step 10   loss 10.42  lr 6.0e-05  tok/s 8,420  mfu 2.1%
[rank 0] step 20   loss 9.84   lr 1.2e-04  tok/s 8,510  mfu 2.1%
...
```

Let it run ~50 steps (loss should drop below 10), then `Ctrl+C`.

```{note}
MFU is low (~2%) because this is a 40M-param model on a single H100
— most of the runtime is framework + data loading, not matmul. MFU
becomes meaningful at 7B+ scale. See
[Scaling guide](scaling-guide.md).
```

## 4. Four-GPU run on one node

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/hf_wikitext.toml
```

`torchrun` spawns 4 processes, each binding one GPU. The config's
`dp_shard = -1` resolves to 4 — FSDP2 shards parameters + gradients
+ optimizer state across the 4 GPUs. Same loss curve, ~3.5× the
tokens/sec (assumes 4× H100, bf16, Wikitext streaming fast enough to
not bottleneck).

Watch the log: every rank reports per-step metrics, but only rank 0
writes checkpoints. By default, data appears in `checkpoints/hf_wikitext/`
under the current working directory.

## 5. Kill and auto-resume

KempnerForge catches **SIGTERM** and **SIGUSR1** (the signals SLURM
sends on preemption / timeout) and writes an emergency checkpoint
before exiting. `Ctrl+C` sends SIGINT, which is **not** intercepted
— the process dies immediately and the last durable state is
whatever the interval-100 checkpoint saved.

To exercise the emergency path manually:

```bash
# In another shell, find the rank-0 pid:
kill -TERM <pid>
```

Expected rank-0 log:

```
Shutdown requested at step 247 — saving emergency checkpoint
Emergency checkpoint written to checkpoints/hf_wikitext/step_247
```

If you just hit `Ctrl+C`, you'll resume from the last periodic save
instead (e.g., `step_200`), which is usually fine for dev work.

Relaunch the same command:

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/hf_wikitext.toml
```

[`CheckpointManager`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/checkpoint/manager.py)
follows the `checkpoints/hf_wikitext/latest` symlink (updated on
every save) and falls back to the highest `step_N` directory if the
symlink is missing. Training picks up at the next step with model,
optimizer, scheduler, dataloader, and RNG state restored.

The dataloader resumes from the exact sample via
`StatefulDataLoader.load_state_dict` + `DistributedSampler.set_skip`
(pre-tokenized path) or `_skip_rank_docs` (HF-streaming path), so no
sample is replayed and none is skipped. See
[Checkpointing § Auto-resume](../checkpointing/auto-resume.md) and
[Resilience § SLURM preemption](../resilience/slurm-preemption.md).

## 6. Generate from the checkpoint

Once the loss is reasonable (`< 7` on Wikitext), try generation:

```bash
uv run python scripts/generate.py configs/train/hf_wikitext.toml \
    --checkpoint.load_path=checkpoints/hf_wikitext/latest \
    --data.tokenizer_path=gpt2 \
    --prompt "The Kempner Institute" \
    --max_tokens 64 \
    --temperature 0.8 \
    --top_p 0.9
```

[`scripts/generate.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/generate.py)
is single-GPU, loads the DCP checkpoint into an un-sharded model,
tokenizes the prompt, calls
[`generate()`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/generate.py)
with KV-cache, and prints the decoded output.

Arguments:

| Flag | Default | Purpose |
|------|---------|---------|
| `config` (positional) | — | TOML path |
| `--checkpoint.load_path` | — | Path to a `step_N` directory or `latest` symlink |
| `--data.tokenizer_path` | from config | HF tokenizer ID or local path |
| `--prompt` | `""` | Input text |
| `--max_tokens` | `128` | Max new tokens |
| `--temperature` | `1.0` | Sampling temperature (0 = greedy) |
| `--top_k` | `0` | Top-k filtering (0 = off) |
| `--top_p` | `1.0` | Nucleus threshold |
| `--interactive` | `false` | REPL mode |

For interactive exploration:

```bash
uv run python scripts/generate.py configs/train/hf_wikitext.toml \
    --checkpoint.load_path=checkpoints/hf_wikitext/latest \
    --data.tokenizer_path=gpt2 \
    --interactive
```

## What you learned

You ran the full pipeline: config-driven build, FSDP2 sharding,
stateful resumption, and KV-cache generation — all on data streamed
from the hub without a pre-tokenization step.

Extensions from here:

- Swap Wikitext for pre-tokenized shards → [Prepare tokenized data](prepare-tokenized-data.md).
- Move to a bigger model and add TP / EP / PP → [Scaling guide](scaling-guide.md).
- Launch from SLURM (single- or multi-node) → [SLURM distributed setup](slurm-distributed-setup.md).
- Handle SLURM preemption → [Resilience § SLURM preemption](../resilience/slurm-preemption.md).
- Go deeper on generation (KV cache, batching, samplers) → [Generate from a checkpoint](generate-from-checkpoint.md).
- Debug NaN / OOM / hangs / slowdowns → [Debug training regressions](debug-training-regressions.md).
- Run downstream benchmarks → [Run evaluation](run-evaluation.md).
- Try FP8 → [Distributed § FP8](../distributed/fp8.md).

## See also

- [Getting started](../getting-started/index.md) — shorter install +
  quickstart for someone who just wants `uv sync && uv run …`.
- [Configuration overview](../configuration/index.md) — what the
  TOML schema looks like and how CLI overrides compose.
- [Checkpointing § Auto-resume](../checkpointing/auto-resume.md) —
  the resumption mechanics in detail.
- [Generation](../training/generation.md) — `generate()` internals
  and KV-cache API.
- [Training loop](../training/training-loop.md) — what
  `scripts/train.py` actually does at each step.
