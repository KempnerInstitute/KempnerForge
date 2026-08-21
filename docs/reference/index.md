# Reference

Curated tables and exhaustive lists that don't fit a narrative but are
useful: every config preset, every proven parallelism combination at
each GPU count, every env var the framework reads.

```{toctree}
:maxdepth: 1

available-configs
parallelism-recipes
benchmarks
environment-variables
```

- **[Available configs](available-configs.md)** — the full
  [`configs/train/*.toml`](https://github.com/KempnerInstitute/KempnerForge/tree/main/configs/train)
  and
  [`configs/model/*.toml`](https://github.com/KempnerInstitute/KempnerForge/tree/main/configs/model)
  tables, with "what this config exists to prove" per row.
- **[Parallelism recipes](parallelism-recipes.md)** — (model, GPU count,
  parallelism) combinations that we've actually run end-to-end, indexed
  by model rather than by filename.
- **[Benchmarks](benchmarks.md)** — summaries and reproduction commands for
  the measured campaigns: dense 7B/13B/70B MFU scaling on 1–32 GPUs, weak
  scaling of 13B/70B out to 160 GPUs, a 175B dense run on 360 GPUs at 47.7%
  MFU, and MoE Expert Parallelism with per-sub-module FSDP wrapping. The
  full index lives in
  [`benchmarks/README.md`](https://github.com/KempnerInstitute/KempnerForge/tree/main/benchmarks).
- **[Environment variables](environment-variables.md)** — every env var
  the framework reads, grouped by source (torchrun / SLURM / NCCL /
  logging) with who-sets-what.

## See also

- [README § Training Configurations](https://github.com/KempnerInstitute/KempnerForge#training-configurations)
  — the current configs table at the repo root.
- [README § Benchmarks](https://github.com/KempnerInstitute/KempnerForge#benchmarks)
  — the MFU table at the repo root.
