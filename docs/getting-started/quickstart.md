# Quickstart

A 5-minute walkthrough that trains a tiny model so you can
verify your install and see the training loop end-to-end.

```{tip}
When you change `model.vocab_size`, `model.dim`, or any other shape-affecting
field between runs, use a fresh `--checkpoint.dir` or delete the old one
first. `train.py` auto-resumes from the latest checkpoint in the directory,
which will fail with a shape mismatch if the architecture changed. Examples
below use `/tmp/` paths so runs don't collide.
```

## 1. Install

```bash
git clone git@github.com:KempnerInstitute/KempnerForge.git
cd KempnerForge
uv sync
```

If you want more detail on this step, see {doc}`install`.

## 2. Run a 20M-parameter debug model on a single GPU

```bash
uv run python scripts/train.py configs/train/debug.toml \
  --checkpoint.dir=/tmp/kf_quickstart/step2
```

You should see per-step loss / MFU / step_time logs. The run takes under a
minute. It uses synthetic data (no dataset download) — useful for
sanity-checking the install before pointing at real data.

For a slower walkthrough of what this run does, what the log line means, and
what ends up in the checkpoint directory, see {doc}`first-training-run`.

## 3. Multi-GPU on a single node (FSDP2)

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/debug.toml \
  --distributed.dp_shard=4 \
  --checkpoint.dir=/tmp/kf_quickstart/step3
```

## 4. Point at your own tokenized data

Pre-tokenized `.bin` or `.npy` shards work directly:

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/debug.toml \
  --data.dataset_path=/path/to/your/shards \
  --data.file_pattern='tokenized_*.bin' \
  --model.vocab_size=128256 \
  --checkpoint.dir=/tmp/kf_quickstart/step4
```

Or stream from HuggingFace:

```bash
uv run python scripts/train.py configs/train/hf_wikitext.toml \
  --checkpoint.dir=/tmp/kf_quickstart/step4_hf
```

## 5. Try a different optimizer

Swap AdamW for Muon without touching code:

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/debug.toml \
  --optimizer.name=muon \
  --checkpoint.dir=/tmp/kf_quickstart/step5
```

Available: `adamw`, `muon`, `lion`, `schedule_free_adamw`.

## 6. Enable MoE

```bash
uv run python scripts/train.py configs/train/debug_moe.toml \
  --checkpoint.dir=/tmp/kf_quickstart/step6
```

Or turn on MoE via CLI on the dense debug config:

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/debug.toml \
  --model.num_experts=8 --model.moe_top_k=2 --model.moe_router=sigmoid_topk \
  --checkpoint.dir=/tmp/kf_quickstart/step6_cli
```

## 7. Extend the training loop without forking `train.py`

See [`examples/custom_hook.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/examples/custom_hook.py)
for four example hooks:

- `GradNormHistogramHook` — per-layer gradient norms to WandB
- `LearningDynamicsHook` — weight norms and gradient SNR
- `EarlyStoppingHook` — stop if eval loss plateaus
- `ExpertLoadBalanceHook` — MoE expert utilization metrics

Register them in your own script by subclassing `TrainingHook`.

## Next steps

- **Understand the run**: {doc}`first-training-run` explains the log line,
  the checkpoint directory layout, and auto-resume.
- **Interactive exploration**: {doc}`notebooks` lists six Jupyter notebooks
  (model inspection, attention visualization, activation extraction,
  checkpoint analysis, optimizer comparison, MoE routing).
- **Scale up**: see
  [README § Training Configurations](https://github.com/KempnerInstitute/KempnerForge#training-configurations)
  for 7B / 13B / 70B configs.
- **Run on SLURM**: see
  [README § Quick Start](https://github.com/KempnerInstitute/KempnerForge#quick-start)
  for single- and multi-node launch scripts.
- **Measured performance**: see
  [`benchmarks/mfu_scaling/mfu_scaling.md`](https://github.com/KempnerInstitute/KempnerForge/blob/main/benchmarks/mfu_scaling/mfu_scaling.md)
  for MFU / throughput numbers across 1–32 GPUs.
- **Contribute**:
  [CONTRIBUTING.md](https://github.com/KempnerInstitute/KempnerForge/blob/main/CONTRIBUTING.md)
  walks through the issue → branch → PR flow.
