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
└── metadata.yaml       # written by tatm; records the .bin dtype
```

Each matched file is read according to its own suffix, so one
`file_pattern` may match both formats. A matched file with any other
suffix (a `metadata.yaml` or `tokenizer.json` caught by `"*"`, say) is
an error that names the file.

- **`.npy`** — standard numpy array. Dtype is read from the header.
- **`.bin`** — raw binary, flat array of tokens, with no header. Its
  dtype is, in order:
  1. `data.token_dtype` (`"uint16"` or `"uint32"`), or the source's own
     `token_dtype` in a mixture, or `eval.token_dtype` for eval data,
     which does not fall back to `data.token_dtype`.
  2. `tokenized_info.dtype` in the `metadata.yaml` next to the shard,
     which tatm writes. When both this and a `token_dtype` are set they
     must agree. The file is looked up in the directory of each matched
     path: a shard in a subdirectory (matched by `**/*.bin`, say) needs
     a `metadata.yaml` in that subdirectory, and a symlinked shard uses
     the one beside the link, not beside its target.
  3. Neither: the shards are read as `uint32`, with a warning that names
     both ways to declare the dtype, provided every shard is a whole
     number of `uint32` tokens. Any other size fits a `uint16` shard and
     a truncated `uint32` shard alike, so the dataset raises instead.

  Once the dtype is known, a shard whose size is not a whole number of
  its tokens is rejected when the dataset is built. So is a
  `metadata.yaml` that cannot be parsed or that records a dtype other
  than `uint16` or `uint32`.

A `uint16` corpus has to declare its dtype. Read as `uint32`, each pair
of tokens fuses into one id far outside the vocabulary, and the run fails
at the first embedding lookup.

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
2. Resolves the dtype of every `.bin` shard (above), then memory-maps
   every file.
3. Computes a cumulative sample count: each file contributes
   `len(file) // seq_len` samples.

At training time, `__getitem__(idx)`:

1. Binary-searches the cumulative offsets to find which shard the
   global `idx` falls into.
2. Slices `seq_len` tokens starting at `local_idx * seq_len`.
3. Returns `{"input_ids": tokens[:-1], "labels": tokens[1:]}`.

Note: the data pipeline passes `seq_len + 1` to the constructor so
the sliced window contains one extra token, and the `[:-1]` / `[1:]`
split produces inputs and next-token labels of length `seq_len` each.

Partial remainders at the end of a file are simply unused — no
padding, no cross-file concatenation at the sample level.

## Config

```toml
[data]
dataset_path = "/path/to/tokens"
file_pattern = "*.npy"            # or "*.bin"
token_dtype = ""                  # .bin only: "uint16" or "uint32"; "" reads metadata.yaml
pack_sequences = false            # see below
tokenizer_path = ""               # only required when pack_sequences=true
```

`build_data_pipeline` picks this path when `data.dataset_path` is set
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
  vocabulary ≤ 65535. Llama-3-class tokenizers (vocab 128,256) need
  `uint32`. A small vocabulary may still be stored as `uint32`, so the
  vocabulary size does not tell you the dtype; `metadata.yaml` does.

## See also

- [Stateful dataloader](stateful-dataloader.md) — resumption logic.
- [Mixing and annealing](mixing-and-annealing.md) — how multiple mmap
  datasets compose via `MixtureDataset`.
- [Sampler](sampler.md) — rank partitioning for this dataset type.
- [`scripts/prepare_data.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/prepare_data.py) —
  validator that resolves the `.bin` dtype the way training does and
  prints the right `--data.*` flags.
