---
name: smoke-test
description: End-to-end one-GPU sanity check. Runs a short training loop to confirm torch, CUDA, NCCL, uv, and the dataloader all work before committing to longer runs.
---

## When to use
- First thing after `/kempnerforge:cluster-config`, before any real training job.
- After a library upgrade (`uv sync`, CUDA driver bump, new kernel image).
- After pulling new code to confirm nothing regressed on this machine.
- Before launching a multi-node job, to rule out single-GPU issues first.

A green smoke test is the precondition for `/kempnerforge:slurm-launch` on anything bigger.

## Preflight
Run:

    uv run python scripts/check_env.py --requires gpu

If the exit code is non-zero, print stdout to the user verbatim and stop. Common outcomes:
- `gpu` MISS: no visible CUDA device. Tell the user to `srun --gres=gpu:1 --pty bash` into a GPU node. (`scripts/slurm/interactive.sh` is NOT an sbatch script — it is an srun launcher that runs inside an existing allocation: `./scripts/slurm/interactive.sh <JOBID> <CONFIG>`.)
- `uv` or `repo_layout` MISS: the user is running from the wrong directory or `uv` is not installed. Follow the fix line in the MISS output.

## Context (auto-generated, do not edit)
<!-- context-begin -->
Debug config: configs/train/debug.toml (4-layer, 256-dim, no dataset_path)
HF streaming demo: configs/train/hf_wikitext.toml (works out of the box with gpt2 tokenizer)
Train entry point: scripts/train.py
Default duration: debug.toml max_steps=100, batch_size=4, seq_len=512
Dataset options (scripts/train.py): config.data.datasets (mixture), config.data.dataset_path (pre-tokenized .npy), config.data.hf_dataset_name (HuggingFace)
Smoke-test artifact: checkpoints/debug/ (from debug.toml)
<!-- context-end -->

## Procedure
Assume preflight has passed.

1. Pick a config:
    - **No local data, no internet**: use `configs/train/debug.toml`. `scripts/train.py` falls back to random-token batches when no dataset is configured, which is enough to exercise the training loop end-to-end.
    - **Fast iteration with HF streaming (recommended when internet is available)**: use `configs/train/hf_wikitext.toml`. Tokenizer is `gpt2` (small, cached on first use). Streams wikitext-103. Real data means the loss curve is interpretable.
    - **User has pre-tokenized data**: use `configs/train/debug.toml` with `--data.dataset_path=<path>`.

2. Cache the `gpt2` tokenizer once (HF path only):

        uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"

    Skip if the user already has HF cache or is using pre-tokenized data.

3. Run on one GPU:

        uv run python scripts/train.py configs/train/hf_wikitext.toml \
          --train.max_steps=20 \
          --checkpoint.interval=0 \
          --metrics.log_interval=5

    Or with pre-tokenized data:

        uv run python scripts/train.py configs/train/debug.toml \
          --data.dataset_path=<abs-path-to-npy-dir> \
          --train.max_steps=20 \
          --checkpoint.interval=0

    Expected: first step within ~60 s (cold compile), then ~1 step/sec on H100. Loss should decrease from step 0 to step 20. `nvidia-smi` should show the process consuming GPU memory.

4. Check for anomalies:
    - Loss NaN on step 0: model init or dtype mismatch. Check `model.vocab_size` matches the tokenizer.
    - Hang before step 0: distributed init problem. Confirm the run used `python`, not `torchrun`, for the one-GPU case.
    - OOM: reduce `--train.batch_size` or `--train.seq_len`.

## Verification
- Training log shows at least 20 steps. With a real dataset (HF or pre-tokenized), loss should trend down (noise is fine). With the synthetic random-token fallback, loss is not interpretable — the check is that the process runs without errors.
- Process exits with code 0.
- `ls checkpoints/debug/` is empty if `--checkpoint.interval=0` was used (the test is intentionally non-destructive).
- `nvidia-smi` returns to idle after the process exits (no zombie CUDA contexts).

## Gotchas
- `debug.toml` has no dataset configured by default. `scripts/train.py` detects this and feeds random-token batches (see the `dataloader is None` branch around line 508); the smoke test still exercises forward/backward/optimizer. Loss will not be meaningful, so for a true "is this converging" signal use `hf_wikitext.toml` or supply `--data.dataset_path`.
- HF streaming writes to `~/.cache/huggingface/` on first use. On read-only home directories, set `HF_HOME=/tmp/hf_cache` before the run.
- Do not use `torchrun` for a one-GPU smoke test. `python scripts/train.py ...` is correct because FSDP2 with world_size=1 initializes a single-rank process group internally.
- If the user is on an A100, `compile_model=true` may add 2 to 3 minutes to the first step. Pass `--train.compile_model=false` for a faster first iteration during smoke testing.
- `hf_wikitext.toml` enables `compile_model = true`. Pass `--train.compile_model=false` for faster iteration if the first step is the bottleneck.

## Related skills
- `/kempnerforge:cluster-config` — prerequisite for anything beyond single-GPU debug.
- `/kempnerforge:slurm-launch` — next step once smoke test is green, for multi-GPU and multi-node runs.
