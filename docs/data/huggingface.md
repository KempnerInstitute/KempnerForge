# HuggingFace datasets

Two classes, one selector flag:

- **`HuggingFaceDataset`** (eager) — loads the entire split into
  memory, tokenizes once, packs into fixed-length sequences.
- **`StreamingHuggingFaceDataset`** (streaming) — an `IterableDataset`
  that pulls documents from the Hub on the fly, tokenizes per
  iteration, and yields packed sequences as they fill.

Config flag: `data.hf_streaming` (default `false`).

Both classes live in
[`kempnerforge/data/dataset.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/dataset.py).

## Eager — `HuggingFaceDataset`

```toml
[data]
hf_dataset_name       = "wikitext"
hf_dataset_config     = "wikitext-103-raw-v1"   # optional
hf_dataset_split      = "train"
hf_dataset_text_field = "text"
hf_streaming          = false
tokenizer_path        = "gpt2"
```

Flow on construction:

1. `AutoTokenizer.from_pretrained(tokenizer_path)` — loads the
   tokenizer. The EOS token ID falls back to `0` if the tokenizer
   doesn't declare one.
2. `load_dataset(name, config, split=split)` — the full HF split is
   materialized.
3. Each example's `text_field` is encoded with `add_special_tokens=False`,
   an EOS token is appended, and the combined token stream is sliced
   into fixed `(seq_len + 1)` chunks. The extra `+1` is for the
   next-token label offset.
4. Partial remainder at the end is discarded — clean, full sequences
   only.

At training time `__getitem__(idx)` returns
`{"input_ids": tokens[:-1], "labels": tokens[1:]}` (plus `doc_ids` if
`pack_sequences=true`).

Good for: small-to-medium corpora that fit in memory; repeatable
experiments where you want a fixed sequence order.

Bad for: anything where `load_dataset(...)` itself would OOM.

## Streaming — `StreamingHuggingFaceDataset`

```toml
[data]
hf_dataset_name = "allenai/c4"
hf_dataset_split = "train"
hf_streaming = true
tokenizer_path = "meta-llama/Llama-3-8B"
```

Stream mode is an `IterableDataset`:

- No `__len__`, no sampler (the dataset is its own iterator).
- `scripts/train.py` wraps it in a plain `torch.utils.data.DataLoader`,
  not `StatefulDataLoader`.
- Distributed sharding happens inside the iterator:
  `if doc_idx % world_size != rank: continue`. Each rank takes every
  `world_size`-th document.

Buffered shuffling is applied at the HF level:

```python
ds = load_dataset(name, config, split=split, streaming=True)
ds = ds.shuffle(seed=self.seed + self._epoch,
                buffer_size=self.shuffle_buffer_size)   # default 10k
```

The default buffer holds 10,000 examples. Larger buffers give better
shuffle quality at the cost of startup latency and RAM.

### Tokenizer is lazy

The tokenizer is loaded on first `__iter__` call, not at construction.
This sidesteps a common `fork()` issue with multiprocessing DataLoader
workers: a tokenizer loaded in the parent can deadlock if the worker
processes fork after HF has already spawned its own threads.

### Resumption in streaming mode

State is `{"epoch", "rank_docs_consumed"}`. On resume,
`load_state_dict` sets `_skip_rank_docs = rank_docs_consumed`; the
next `__iter__` fast-forwards past that many documents on this rank
before yielding anything.

Note: this only works if the same number of DP ranks is used at
save and load time. Change the rank count and each rank's document
stream looks different.

## Packing and document boundaries

Same as [Memory-mapped § Sequence packing](memory-mapped.md#sequence-packing):
setting `pack_sequences = true` adds `doc_ids` to each sample and
masks cross-document label positions with `-100`. Both HF paths
support it.

## Eval datasets

`[eval]` can mirror either path independently of training:

```toml
[eval]
enabled          = true
hf_dataset_name  = "wikitext"
hf_dataset_split = "validation"
```

For HF eval sets, rank 0 tokenizes and broadcasts the packed tensor
to all ranks via `torch.distributed.broadcast`. This is a deliberate
choice to avoid `load_dataset` file-lock (`flock`) failures on
cluster filesystems (Lustre, VAST) where parallel opens crash.

## Limitations

- Eager mode re-tokenizes on every process start; no tokenized cache
  beyond the HF `~/.cache/huggingface/` layer.
- Streaming mode with shuffling + resumption isn't bit-exact: after
  resume the same rank will see a different *shuffled order* because
  buffered shuffle state isn't captured.
- Neither class supports on-disk tokenization caching. If you want
  that, pre-tokenize with `tatm` and switch to
  [Memory-mapped](memory-mapped.md).

## See also

- [Memory-mapped](memory-mapped.md) — the faster path once you have
  pre-tokenized shards.
- [Mixing and annealing](mixing-and-annealing.md) — compose HF sources
  alongside mmap ones in a `MixtureDataset`.
- [`configs/train/hf_wikitext.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/hf_wikitext.toml) —
  a working streaming-mode config.
