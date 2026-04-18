# Data

KempnerForge's data pipeline: the dataset types, the sampler that
partitions work across data-parallel ranks, the stateful dataloader
that makes checkpoint resume possible, and the mixing / annealing
hooks used for curriculum training.

Entry point: `scripts/train.py` picks one of four paths based on
config — pre-tokenized mmap, HuggingFace eager, HuggingFace streaming,
or multi-source mixture.

## At a glance

| Config path | Dataset class | Sampler | Notes |
|-------------|---------------|---------|-------|
| `data.dataset_path = "..."` | `MemoryMappedDataset` | `DistributedSampler` | Pre-tokenized `.npy` / `.bin` shards. Fastest path. |
| `data.hf_dataset_name = "..."` (+ `hf_streaming=false`) | `HuggingFaceDataset` | `DistributedSampler` | Eager download, tokenize once, pack into RAM. |
| `data.hf_dataset_name = "..."` + `hf_streaming=true` | `StreamingHuggingFaceDataset` | (none — `IterableDataset`) | On-the-fly tokenization from a streaming Hub dataset. |
| `data.datasets = [...]` (non-empty) | `MixtureDataset` over sub-datasets | `MixtureSampler` | Weighted mixing across multiple sources; phase transitions rebalance weights. |

All of the map-style datasets and samplers implement `state_dict()` /
`load_state_dict()`. `StatefulDataLoader` wraps them and tracks the
batch position inside an epoch so training can resume exactly after a
checkpoint — see [Stateful dataloader](stateful-dataloader.md).

## Config

Everything here is driven by `DataConfig` (`kempnerforge/config/data.py`):

```toml
[data]
dataset_path    = ""              # pre-tokenized shard directory
file_pattern    = "*.npy"         # glob for shard files (also "*.bin")
tokenizer_path  = ""              # HF tokenizer name or local path
num_workers     = 4
pin_memory      = true
prefetch_factor = 2
pack_sequences  = false           # cross-doc label masking via EOS

# HuggingFace path
hf_dataset_name      = ""           # default: None (str | None)
hf_dataset_config    = ""           # default: None (str | None)
hf_dataset_split     = "train"
hf_dataset_text_field = "text"
hf_streaming         = false

# Multi-dataset mixing
datasets        = []              # list of DatasetSource tables
mix_temperature = 1.0             # weight scaling (1.0 = as-is, >1 = more uniform)

# Phase scheduling (step-triggered weight + LR transitions)
phases            = []
anneal_start_step = 0             # shortcut for a common 2-phase pattern
anneal_weights    = {}
```

See [Configuration § `[data]` — DataConfig](../configuration/config-sections.md)
for the per-field reference.

## Pages

```{toctree}
:maxdepth: 1

memory-mapped
huggingface
mixing-and-annealing
stateful-dataloader
sampler
```

## See also

- [Checkpointing § Train state](../checkpointing/train-state.md) —
  dataloader state *infrastructure* exists but the shipped training
  loop doesn't wire it into the save; resume restarts the epoch.
- [Training § Training loop](../training/training-loop.md) — where
  the dataloader is consumed step by step.
- [Configuration § `[data]` — DataConfig](../configuration/config-sections.md) —
  every knob on this page.
