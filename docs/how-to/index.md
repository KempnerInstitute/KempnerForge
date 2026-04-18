# How-to Guides

End-to-end researcher workflows. Each guide is a single coherent narrative
with runnable code (or a link to a notebook, config, or script that runs it)
— the reason most researchers come to the docs in the first place.

## Core workflow

- [Build a model](build-a-model.md) — compose a `Transformer` from
  components; swap MLP for MoE; toggle QK-norm; register a new block
  variant.
- [Prepare tokenized data](prepare-tokenized-data.md) — two supported
  paths: pre-tokenize offline (e.g. with `tatm`) and validate with
  `scripts/prepare_data.py`, or stream from HuggingFace.
- [End-to-end training run](end-to-end-training-run.md) — the flagship
  walkthrough: tokenize → write a config → launch 1 GPU → launch 4 GPUs →
  resume → generate.

## Scale up

- [Scaling guide](scaling-guide.md) — 1 → 32 GPU journey, when to add
  TP / FSDP / EP / PP, batch-size scaling, MFU goals, common pitfalls.
- [SLURM distributed setup](slurm-distributed-setup.md) — single-node →
  multi-node, InfiniBand, NCCL env, preemption, auto-resume.

## Operations

- [Run evaluation](run-evaluation.md) — `run_eval()` during training,
  standalone `scripts/eval.py`, `scripts/eval_harness.py` for
  lm-eval-harness.
- [Generate from checkpoint](generate-from-checkpoint.md) — load a DCP
  checkpoint, call `generate()` with temperature / top-k / top-p,
  interact with `KVCache`.
- [Debug training regressions](debug-training-regressions.md) — NaN
  detector, profiler, memory monitor, health checks, five failure
  shapes and how to read them.

## Research knobs

- [Compare optimizers](compare-optimizers.md) — AdamW vs Muon vs Lion
  vs Schedule-Free AdamW, LR conventions, fair-comparison protocol.
- [Mix datasets and anneal data weights](data-mixing-annealing.md) —
  weighted mixtures, temperature, phase transitions, LR scale on phase
  boundaries.
- [Turn on FP8 training](fp8-training.md) — E4M3 / E5M2, bf16 master
  weights, FSDP2 float8 all-gather, exclusion rules, when FP8 doesn't
  help.
- [MoE experiments](moe-experiments.md) — router choice, aux loss
  tuning, hot/cold expert diagnosis, when to turn on EP, shared
  experts.
- [Extract activations for interpretability](mechanistic-interpretability.md)
  — `ActivationStore`, `extract_representations()`, save to `.npz`,
  feed to probing / CKA / SVCCA.

```{toctree}
:maxdepth: 1
:hidden:

build-a-model
prepare-tokenized-data
end-to-end-training-run
scaling-guide
slurm-distributed-setup
run-evaluation
generate-from-checkpoint
debug-training-regressions
compare-optimizers
data-mixing-annealing
fp8-training
moe-experiments
mechanistic-interpretability
```
