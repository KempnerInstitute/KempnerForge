# Architecture

How the pieces fit together. This section covers the forward pass, the
parallelism application order, and the path a batch of tokens takes from
the dataloader back to a gradient update.

## One-slide overview

```{tip}
The README renders this as a mermaid diagram:
[README § Architecture](https://github.com/KempnerInstitute/KempnerForge#architecture).
```

```
TOML preset ─┐
             ├──► JobConfig ──► scripts/train.py ──┬──► Model
CLI override ┘    (dataclasses)   (training loop)  │     Token Embedding → Blocks (RoPE · GQA · SwiGLU or MoE) → Output Head
                                                   │
                                                   ├──► Parallelism (strict order)
                                                   │     1 · TP → 2 · EP → 3 · FP8 → 4 · AC → 5 · FSDP2
                                                   │
                                                   ├──► Data
                                                   │     MemoryMapped / HF → DistributedSampler / MixtureSampler → StatefulDataLoader
                                                   │
                                                   ├──► Resilience
                                                   │     SIGTERM handler · NaN detector · NCCL health
                                                   │
                                                   └──► Outputs
                                                         DCP checkpoints · MetricsTracker (WandB · TB) · torch.profiler
```

## Component responsibilities

| Package | Responsibility |
|---------|----------------|
| `kempnerforge.config` | Typed dataclass configs, TOML loading, CLI overrides, component registry |
| `kempnerforge.model` | `Transformer`, `TransformerBlock`, `Attention`, `SwiGLUMLP`, `MoEMLP`, `RMSNorm`, RoPE, embeddings, weight init |
| `kempnerforge.distributed` | `DeviceMesh` construction, FSDP2, TP, EP, Float8, activation checkpointing, `build_parallel_model` |
| `kempnerforge.data` | `MemoryMappedDataset`, `MixtureDataset`, `DistributedSampler`, `MixtureSampler`, `StatefulDataLoader` |
| `kempnerforge.training` | Training step, optimizers (AdamW / Lion / Muon / schedule-free), LR schedulers, loss functions |
| `kempnerforge.checkpoint` | DCP-based sharded checkpoints, async save, auto-resume |
| `kempnerforge.resilience` | SIGTERM/SIGUSR1 handler, NaN detector, GPU + NCCL health checks |
| `kempnerforge.metrics` | `MetricsTracker`, MFU, WandB / TensorBoard backends, memory monitor |
| `kempnerforge.profiling` | `torch.profiler` integration, CUDA event timing |

## Design principles

Copied from
[README § Design Principles](https://github.com/KempnerInstitute/KempnerForge#design-principles):

- **PyTorch-native**: FSDP2, DTensor, DeviceMesh, DCP, SDPA, `torch.compile`.
- **Distributed-first**: multi-GPU is the default, not an afterthought.
- **Composition over inheritance**: components are composed via config, not
  a class hierarchy.
- **Minimal abstraction**: readable code over framework magic.
- **Stateful everything**: dataloader, sampler, and training state all
  support checkpoint and resume.
- **Configuration-driven**: all behavior controlled by typed dataclass
  configs, validated at startup.

## Where to go next

```{toctree}
:maxdepth: 1

model
parallelism-order
data-flow
```

- **[Model](model.md)** — the forward pass block by block: embeddings →
  RoPE → attention → SwiGLU or MoE → RMSNorm → output head, plus weight
  init.
- **[Parallelism order](parallelism-order.md)** — the 5-step order
  (TP → EP → FP8 → AC → FSDP2) and what goes wrong when you violate it.
- **[Data flow](data-flow.md)** — the path of a batch from
  `StatefulDataLoader` through forward, loss, backward, optimizer step,
  and checkpoint tick.
