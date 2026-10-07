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

Subclass `TrainingHook`, override only the events you need, and pass the
hooks to `run_training`. Save this as `my_train.py`:

```python
import sys

from kempnerforge.config.loader import load_config
from kempnerforge.training import run_training
from kempnerforge.training.hooks import HookRunner, StepContext, TrainingHook


class LossPrinter(TrainingHook):
    """Print the loss and learning rate every ``interval`` steps."""

    def __init__(self, interval: int = 10) -> None:
        self.interval = interval

    def on_step_end(self, ctx: StepContext) -> None:
        if ctx.step % self.interval == 0:
            print(f"step {ctx.step}: loss {ctx.loss:.4f}, lr {ctx.lr:.2e}")


if __name__ == "__main__":
    config = load_config(sys.argv[1], cli_args=sys.argv[2:])
    run_training(config, hooks=HookRunner([LossPrinter()]))
```

```bash
uv run python my_train.py configs/train/debug.toml \
  --checkpoint.dir=/tmp/kf_quickstart/step7
```

{doc}`../training/hooks` lists every event and what it receives. Gradients
are already zeroed when `on_step_end` fires.

## Next steps

- **Understand the run**: {doc}`first-training-run` explains the log line,
  the checkpoint directory layout, and auto-resume.
- **Scale up**: see
  [README § Training Configurations](https://github.com/KempnerInstitute/KempnerForge#training-configurations)
  for 7B / 13B / 70B configs.
- **Run on SLURM**: see
  [README § Quick Start](https://github.com/KempnerInstitute/KempnerForge#quick-start)
  for single- and multi-node launch scripts.
- **Measured performance**: see
  [`benchmarks/mfu_scaling/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/benchmarks/mfu_scaling)
  for MFU / throughput numbers across 1–32 GPUs.
- **Contribute**:
  [CONTRIBUTING.md](https://github.com/KempnerInstitute/KempnerForge/blob/main/CONTRIBUTING.md)
  walks through the issue → branch → PR flow.
