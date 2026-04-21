# Prepare tokenized data

KempnerForge reads two kinds of data: pre-tokenized binary shards on
disk, or text streamed from HuggingFace. Pick the first for production
runs (faster, deterministic, works offline); pick the second for
small experiments and prototyping.

## Path A: pre-tokenized shards

### On-disk format

[`MemoryMappedDataset`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/dataset.py)
reads a directory of shard files, each a flat 1-D array of token IDs:

| File extension | Encoding | Dtype rule |
|----------------|----------|------------|
| `*.npy` | NumPy mmap | dtype stored in header |
| `*.bin` | Raw packed integers | inferred from size (4-byte multiple → uint32, else uint16), or from `metadata.yaml` |

One file is one flat array. The dataset concatenates shards logically
and splits the concatenated stream into non-overlapping
`seq_len`-token samples. `uint16` works for vocab ≤ 65,535 (GPT-2 at
50,257 and Llama-2 / Mistral at 32,000 fit; Llama-3 at 128,256 and
Gemma at 256,000 need `uint32`).

### Tokenize with your tool of choice

KempnerForge does **not ship a tokenizer** and does **not depend on
`tatm`**. Any tokenizer that outputs `.bin` or `.npy` shards works.
The workflow `scripts/prepare_data.py` documents is the one used
internally at the Kempner Institute:

```bash
# Step 1 — tokenize (tatm is one of many tools that writes this format)
tatm tokenize --tokenizer meta-llama/Llama-3.2-1B \
              --output-dir /path/to/tokenized/dataset \
              <dataset-spec>

# Step 2 — validate
uv run python scripts/prepare_data.py /path/to/tokenized/dataset
```

Rolling your own: write flat arrays of uint16 / uint32 into `.bin` or
`.npy` files, optionally drop a `metadata.yaml` alongside them, and
point training at the directory.

### Validate with `prepare_data.py`

```bash
uv run python scripts/prepare_data.py /path/to/tokenized/dataset
```

[`scripts/prepare_data.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/prepare_data.py)
walks the directory, reports the shard count, size in GB, total token
count, dtype, and (if `metadata.yaml` is present) the tokenizer name
and vocab size. If the dtype is uint16 / uint32 the script prints the
exact train-time flags:

```
Compatible with KempnerForge MemoryMappedDataset.

Train with:
  --data.dataset_path=/path/to/tokenized/dataset
  --data.file_pattern='*.bin'
  --model.vocab_size=128256
```

Wire those into your TOML:

```toml
[data]
dataset_path = "/path/to/tokenized/dataset"
file_pattern = "*.bin"

[model]
vocab_size = 128256   # must match what you tokenized with
```

### Optional: `metadata.yaml` (tatm format)

The validator reads `metadata.yaml` if present. Schema used by
`tatm`:

```yaml
tokenized_info:
  dtype: uint32                              # or uint16
  tokenizer: meta-llama/Llama-3.2-1B
  vocab_size: 128256
```

If you generate shards yourself, dropping this file in the directory
lets the validator print the right `vocab_size` flag and lets
`.bin`-format shards bypass the heuristic dtype inference. It's
optional — without it the validator still runs.

## Path B: HuggingFace streaming

For experiments on standard benchmark datasets, skip the
pre-tokenize step and stream directly:

```toml
[data]
hf_dataset_name       = "wikitext"
hf_dataset_config     = "wikitext-103-raw-v1"
hf_dataset_split      = "train"
hf_dataset_text_field = "text"
hf_streaming          = true
tokenizer_path        = "gpt2"
```

With `hf_streaming = true`,
[`StreamingHuggingFaceDataset`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/dataset.py)
downloads and tokenizes on the fly during iteration. Each rank
processes every `world_size`-th document, so no coordination is
needed for distributed runs.

The tokenizer path is loaded through
`transformers.AutoTokenizer.from_pretrained`, so it can be a
HuggingFace hub ID (`"gpt2"`, `"meta-llama/Llama-3.2-1B"`) or a
local directory.

### Cache the tokenizer first

Hugging Face cannot access the hub from compute nodes with no or limited internet connectivity. 
Since compute nodes also have much slower bandwidth (~1 Gbps vs. ~100 Gbps on login nodes), 
cache the tokenizer once on the login node:

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"
```

The tokenizer lands in `~/.cache/huggingface/` and the compute-node
run picks it up transparently. See
[`configs/train/hf_wikitext.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/hf_wikitext.toml)
for the full HF-streaming reference config.

### Non-streaming (eager) HF loading

`hf_streaming = false` switches to `HuggingFaceDataset`, which
downloads the full dataset and tokenizes in one pass before training
starts. Faster per-step once loaded, but costs memory and startup
time for large datasets. Use only for small benchmarks where the
whole dataset fits.

## Which path to pick

| Constraint | Path |
|------------|------|
| Deterministic, reproducible runs | Pre-tokenized (Path A) |
| Multi-TB datasets | Pre-tokenized (Path A) |
| Air-gapped compute nodes | Pre-tokenized (Path A) |
| Quick experiment on a benchmark dataset | HF streaming (Path B) |
| Dataset bigger than disk | HF streaming (Path B) |
| Same data across reruns, offline | Either — but pre-tokenize once |

Pre-tokenizing is strictly a one-time cost for strictly better
runtime behavior (zero-copy mmap, exact resumption, deterministic
sample order). HF streaming is optimized for exploration.

## Mixed datasets

Both paths support weighted mixing via `data.datasets`:

```toml
[[data.datasets]]
name   = "code"
path   = "/data/tokenized/code"
weight = 0.3

[[data.datasets]]
name      = "web"
hf_name   = "HuggingFaceFW/fineweb"
hf_config = "sample-10BT"
weight    = 0.7
```

See [Data § Mixing and annealing](../data/mixing-and-annealing.md)
for the mechanics and phase-transition recipes.

## Sequence packing

`pack_sequences = true` packs multiple shorter documents into one
`seq_len` window, with a `doc_ids` tensor that prevents cross-document
attention. Useful when document lengths are much smaller than
`seq_len`; otherwise the default (one document per sample, truncated
or padded) is fine.

## Resumption

Both paths work with
[checkpoint auto-resume](../checkpointing/auto-resume.md), but via
different mechanisms:

- **Pre-tokenized (Path A)**: wrapped in `StatefulDataLoader`, which
  saves and restores the sampler's position via
  `DistributedSampler.set_skip()`. A job that preempts at step 3,500
  picks up at sample `3500 × batch_size × world_size` without replay.
- **HF streaming (Path B)**: wrapped in a plain `DataLoader`;
  `StreamingHuggingFaceDataset` tracks its own position and exposes
  `load_state_dict` / `state_dict`, which `scripts/train.py` drives
  directly. Same guarantee — no replay, no skip — via a different
  code path.

## See also

- [Data § MemoryMappedDataset](../data/memory-mapped.md) — the full
  `MemoryMappedDataset` API and the packing implementation.
- [Data § StatefulDataLoader](../data/stateful-dataloader.md) —
  `StatefulDataLoader` and resumption.
- [Configuration § `[data]`](../configuration/config-sections.md) —
  every `DataConfig` field with its default.
- [End-to-end training run](end-to-end-training-run.md) — uses
  Path B (HF wikitext) as the runnable example.
- [`configs/train/hf_wikitext.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/hf_wikitext.toml)
  — reference HF-streaming config.
