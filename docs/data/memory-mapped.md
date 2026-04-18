# Memory-mapped dataset

[`MemoryMappedDataset`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/dataset.py)
is the fastest training path — tokens live on disk as flat `uint16`
or `uint32` arrays, and training reads them via `numpy.memmap` without
copying into user memory.

Use this whenever your dataset is pre-tokenized.

## Expected layout

A directory of shard files, one flat 1-D array per file:

```
/path/to/tokens/
├── shard_00000.npy     # or .bin
├── shard_00001.npy
├── ...
└── metadata.yaml       # optional — produced by tatm
```

Supported formats:

- **`.npy`** — standard numpy array. Dtype is read from the header.
- **`.bin`** — raw binary, flat array of tokens. Dtype is inferred:
  `uint32` if the file size is a multiple of 4, else `uint16`.

The shipped tokenization workflow is external (`tatm`); KempnerForge
ships a validator, not a tokenizer:

```bash
# After tatm produces a directory
uv run python scripts/prepare_data.py /path/to/tokens

# Prints the exact --data flags for scripts/train.py
```

## How sequences are carved out

On construction the dataset:

1. Globs `data_dir/<file_pattern>` and sorts the result — deterministic
   ordering across nodes.
2. Memory-maps every file.
3. Computes a cumulative sample count: each file contributes
   `len(file) // seq_len` samples.

At training time, `__getitem__(idx)`:

1. Binary-searches the cumulative offsets to find which shard the
   global `idx` falls into.
2. Slices `seq_len` tokens starting at `local_idx * seq_len`.
3. Returns `{"input_ids": tokens[:-1], "labels": tokens[1:]}`.

Note: `scripts/train.py` passes `seq_len + 1` to the constructor so
the sliced window contains one extra token, and the `[:-1]` / `[1:]`
split produces inputs and next-token labels of length `seq_len` each.

Partial remainders at the end of a file are simply unused — no
padding, no cross-file concatenation at the sample level.

## Config

```toml
[data]
dataset_path = "/path/to/tokens"
file_pattern = "*.npy"            # or "*.bin"
pack_sequences = false            # see below
tokenizer_path = ""               # only required when pack_sequences=true
```

`scripts/train.py` picks this path when `data.dataset_path` is set
and `data.datasets` is empty.

## Sequence packing

Set `pack_sequences = true` to enable document-aware label masking.
This requires:

- `tokenizer_path` — to look up the EOS token ID.
- Shards that actually contain EOS tokens between documents (typical
  for tatm output).

With packing on, `__getitem__` returns an extra field and masks
cross-document label positions:

```python
{
    "input_ids": LongTensor[seq_len],
    "labels":    LongTensor[seq_len],   # -100 at cross-doc boundaries
    "doc_ids":   LongTensor[seq_len],   # per-token doc index for attention masking
}
```

The `-100` values are the standard PyTorch "ignore" index for
`CrossEntropyLoss` — the loss function simply skips those positions.
The `doc_ids` tensor is what the attention layer uses to prevent
attending across document boundaries (see
[Model § Attention](../architecture/index.md)).

Implementation: [`_compute_packed_output`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/dataset.py)
detects boundaries by incrementing a running `doc_id` whenever it
sees `eos_token_id`; then the cross-boundary mask is
`input_doc_ids != label_doc_ids`.

## Resumption

`state_dict()` captures `{"epoch", "total_samples"}`; `load_state_dict`
restores the epoch. The actual intra-epoch position is tracked by the
sampler and dataloader, not the dataset itself (see
[Sampler](sampler.md) and [Stateful dataloader](stateful-dataloader.md)).

## Performance notes

- mmap is read-only and shared across workers — every DataLoader
  worker process sees the same kernel page cache.
- The first pass through a shard pays for page faults; subsequent
  passes are RAM-speed as long as the working set fits.
- Since sampling is shuffled across the full dataset every epoch,
  expect cold-cache reads for the first epoch on corpora larger than
  system RAM. A 1-2% warmup overhead is typical.
- `uint16` halves the on-disk footprint vs `uint32` and works for any
  vocabulary ≤ 65535. Llama-3 / Mixtral-class tokenizers (vocab
  ~128k) need `uint32`.

## See also

- [Stateful dataloader](stateful-dataloader.md) — resumption logic.
- [Mixing and annealing](mixing-and-annealing.md) — how multiple mmap
  datasets compose via `MixtureDataset`.
- [Sampler](sampler.md) — rank partitioning for this dataset type.
- [`scripts/prepare_data.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/prepare_data.py) —
  validator that prints the right `--data.*` flags.
