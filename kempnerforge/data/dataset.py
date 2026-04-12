"""Dataset implementations for KempnerForge.

Three dataset types:
  - MemoryMappedDataset: Pre-tokenized numpy files with zero-copy mmap access.
  - HuggingFaceDataset: HuggingFace datasets with eager loading and sequence packing.
  - StreamingHuggingFaceDataset: Streaming HuggingFace datasets for very large corpora
    that don't fit in memory. On-the-fly tokenization with sequence packing.

All implement a stateful interface (state_dict / load_state_dict) for
resumption after checkpoint loads.
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _compute_packed_output(tokens: np.ndarray, eos_token_id: int) -> dict[str, torch.Tensor]:
    """Compute input_ids, labels, and doc_ids for a packed token sequence.

    Detects document boundaries from EOS tokens. Each document includes its
    trailing EOS. Cross-document label positions are masked with -100 so the
    loss function can ignore them (predictions at the boundary between two
    documents are meaningless).

    Args:
        tokens: Token array of shape ``(seq_len + 1,)`` (extra token for label offset).
        eos_token_id: Token ID that marks document boundaries.

    Returns:
        Dict with ``input_ids`` (seq_len,), ``labels`` (seq_len, with -100 at
        cross-document boundaries), and ``doc_ids`` (seq_len, integer document
        assignment per input token for attention masking).
    """
    # Assign a document ID to each token: increment after every EOS
    doc_ids = np.zeros(len(tokens), dtype=np.int64)
    doc_id = 0
    for i in range(len(tokens)):
        doc_ids[i] = doc_id
        if tokens[i] == eos_token_id:
            doc_id += 1

    token_tensor = torch.from_numpy(tokens.copy()).long()
    doc_id_tensor = torch.from_numpy(doc_ids.copy())

    input_ids = token_tensor[:-1]
    labels = token_tensor[1:].clone()
    input_doc_ids = doc_id_tensor[:-1]
    label_doc_ids = doc_id_tensor[1:]

    # Mask labels at cross-document boundaries (first token of a new document
    # should not be predicted from the last token of the previous document)
    cross_boundary = input_doc_ids != label_doc_ids
    labels[cross_boundary] = -100

    return {"input_ids": input_ids, "labels": labels, "doc_ids": input_doc_ids}


class MemoryMappedDataset(Dataset):
    """Pre-tokenized dataset backed by memory-mapped numpy files.

    Expects .npy files containing 1D arrays of uint16/uint32 token IDs
    that have been pre-packed into fixed-length sequences.

    File layout: each file stores a flat array of tokens. The dataset
    splits them into non-overlapping chunks of ``seq_len`` tokens.
    Multiple files are concatenated logically.

    Args:
        data_dir: Directory containing .npy token files.
        seq_len: Sequence length (number of tokens per sample).
        file_pattern: Glob pattern for data files.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        file_pattern: str = "*.npy",
        pack_sequences: bool = False,
        eos_token_id: int | None = None,
    ) -> None:
        self.seq_len = seq_len
        self._pack_sequences = pack_sequences
        self._eos_token_id = eos_token_id
        if pack_sequences and eos_token_id is None:
            raise ValueError("eos_token_id is required when pack_sequences=True")

        # Discover and sort data files for deterministic ordering
        data_path = Path(data_dir)
        self._files = sorted(data_path.glob(file_pattern))
        if not self._files:
            raise FileNotFoundError(f"No files matching {file_pattern!r} in {data_dir}")

        # Detect file format from extension
        self._is_bin = self._files[0].suffix == ".bin"

        # Memory-map all files and compute cumulative offsets
        self._mmaps: list[np.ndarray] = []
        self._cumulative_samples: list[int] = [0]
        total_tokens = 0

        for f in self._files:
            if self._is_bin:
                # Raw binary: flat array of tokens. Infer dtype from file size
                # or use uint32 (most common for modern tokenizers with vocab > 65535)
                file_size = f.stat().st_size
                dtype = np.uint32 if file_size % 4 == 0 else np.uint16
                n_tokens = file_size // np.dtype(dtype).itemsize
                mmap = np.memmap(str(f), dtype=dtype, mode="r", shape=(n_tokens,))
            else:
                mmap = np.load(str(f), mmap_mode="r")
            n_samples = len(mmap) // seq_len
            self._mmaps.append(mmap)
            total_tokens += len(mmap)
            self._cumulative_samples.append(self._cumulative_samples[-1] + n_samples)

        self._total_samples = self._cumulative_samples[-1]
        logger.info(
            f"MemoryMappedDataset: {len(self._files)} files, "
            f"{total_tokens:,} tokens, {self._total_samples:,} samples (seq_len={seq_len})"
        )

        # State for resumption
        self._epoch = 0

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._total_samples})")

        # Binary search for the file containing this sample
        file_idx = self._find_file(idx)
        local_idx = idx - self._cumulative_samples[file_idx]
        start = local_idx * self.seq_len
        end = start + self.seq_len

        tokens = self._mmaps[file_idx][start:end].astype(np.int64)

        if self._pack_sequences:
            return _compute_packed_output(tokens, self._eos_token_id)

        token_tensor = torch.from_numpy(tokens.copy())

        # Input: tokens[:-1], Target: tokens[1:] (standard causal LM)
        return {
            "input_ids": token_tensor[:-1],
            "labels": token_tensor[1:],
        }

    def _find_file(self, idx: int) -> int:
        """Binary search for the file index containing global sample idx."""
        lo, hi = 0, len(self._files) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cumulative_samples[mid + 1] <= idx:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def state_dict(self) -> dict:
        """Return checkpoint state. Keys: ``epoch``, ``total_samples``."""
        return {"epoch": self._epoch, "total_samples": self._total_samples}

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint. Only ``epoch`` is restored; sample count is derived."""
        self._epoch = state.get("epoch", 0)


class HuggingFaceDataset(Dataset):
    """HuggingFace dataset with on-the-fly tokenization and sequence packing.

    Loads a HuggingFace dataset, tokenizes text on the fly, and packs
    multiple documents into fixed-length sequences (separated by EOS tokens).

    Args:
        dataset_name: HuggingFace dataset name (e.g., "allenai/c4").
        dataset_config: Optional config name (e.g., "wikitext-2-raw-v1").
        split: Dataset split ("train", "validation", etc.).
        text_field: Name of the text column.
        seq_len: Sequence length for packing.
        tokenizer_path: Path or name for HuggingFace tokenizer.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        text_field: str,
        seq_len: int,
        tokenizer_path: str,
        dataset_config: str | None = None,
        pack_sequences: bool = False,
    ) -> None:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        self.text_field = text_field
        self._packing_enabled = pack_sequences

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._eos_id = self._tokenizer.eos_token_id or 0

        # Load dataset and pack into fixed-length sequences
        logger.info(f"Loading HuggingFace dataset: {dataset_name} ({split})")
        raw_dataset = load_dataset(dataset_name, dataset_config, split=split)
        self._packed_sequences = self._pack_sequences(raw_dataset)
        logger.info(
            f"HuggingFaceDataset: {len(self._packed_sequences)} packed sequences "
            f"(seq_len={seq_len}) from {len(raw_dataset)} documents"
        )

        # State for resumption
        self._epoch = 0
        self._sample_idx = 0

    def _pack_sequences(self, raw_dataset) -> list[np.ndarray]:
        """Tokenize and pack documents into fixed-length sequences.

        Documents are concatenated with EOS separators, then sliced into
        chunks of exactly (seq_len + 1) tokens. The +1 provides the target
        for the last input position.
        """
        chunk_size = self.seq_len + 1  # +1 for the target offset
        buffer: list[int] = []
        packed: list[np.ndarray] = []

        for example in raw_dataset:
            text = example[self.text_field]
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                continue
            buffer.extend(tokens)
            buffer.append(self._eos_id)

            # Flush full chunks from buffer
            while len(buffer) >= chunk_size:
                packed.append(np.array(buffer[:chunk_size], dtype=np.int64))
                buffer = buffer[chunk_size:]

        # Discard partial remainder (no padding — clean sequences only)
        return packed

    def __len__(self) -> int:
        return len(self._packed_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self._packed_sequences[idx]

        if self._packing_enabled:
            return _compute_packed_output(tokens, self._eos_id)

        token_tensor = torch.from_numpy(tokens.copy())
        return {
            "input_ids": token_tensor[:-1],
            "labels": token_tensor[1:],
        }

    def state_dict(self) -> dict:
        """Return checkpoint state. Keys: ``epoch``, ``sample_idx``, ``total_sequences``."""
        return {
            "epoch": self._epoch,
            "sample_idx": self._sample_idx,
            "total_sequences": len(self._packed_sequences),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint. Restores ``epoch`` and ``sample_idx``."""
        self._epoch = state.get("epoch", 0)
        self._sample_idx = state.get("sample_idx", 0)


class StreamingHuggingFaceDataset(torch.utils.data.IterableDataset):
    """Streaming HuggingFace dataset with on-the-fly tokenization and packing.

    For very large datasets that don't fit in memory. Streams documents,
    tokenizes on the fly, and packs into fixed-length sequences.

    Handles distributed training by sharding the document stream across ranks
    (each rank processes every world_size-th document).

    Use directly with ``torch.utils.data.DataLoader`` (no sampler needed —
    IterableDataset handles its own distribution).

    Args:
        dataset_name: HuggingFace dataset name (e.g., "allenai/c4").
        split: Dataset split ("train", "validation", etc.).
        text_field: Name of the text column.
        seq_len: Sequence length for packing.
        tokenizer_path: Path or name for HuggingFace tokenizer.
        dataset_config: Optional config name (e.g., "wikitext-2-raw-v1").
        rank: Current distributed rank (for document sharding).
        world_size: Total number of ranks.
        seed: Random seed for shuffling.
        shuffle_buffer_size: Number of examples to buffer for shuffling.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        text_field: str,
        seq_len: int,
        tokenizer_path: str,
        dataset_config: str | None = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        shuffle_buffer_size: int = 10000,
        pack_sequences: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_field = text_field
        self.seq_len = seq_len
        self.tokenizer_path = tokenizer_path
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self._packing_enabled = pack_sequences

        # Lazy-init tokenizer (avoid loading before fork in multiprocessing workers)
        self._tokenizer = None
        self._eos_id = None

        # State for resumption
        self._epoch = 0
        self._rank_docs_consumed = 0
        self._skip_rank_docs = 0

        logger.info(
            f"StreamingHuggingFaceDataset: {dataset_name} ({split}), "
            f"rank={rank}/{world_size}, seq_len={seq_len}"
        )

    def _ensure_tokenizer(self):
        """Lazy-load tokenizer on first use."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self._eos_id = self._tokenizer.eos_token_id or 0

    def _load_stream(self):
        """Load a shuffled HuggingFace streaming dataset."""
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=True,
        )
        # Shuffle with seed + epoch for different order each epoch
        ds = ds.shuffle(seed=self.seed + self._epoch, buffer_size=self.shuffle_buffer_size)
        return ds

    def __iter__(self):
        self._ensure_tokenizer()
        chunk_size = self.seq_len + 1  # +1 for target offset
        buffer: list[int] = []

        stream = self._load_stream()

        doc_idx = 0  # Global document counter (all ranks)
        rank_docs = 0  # Documents processed by this rank

        for example in stream:
            # Distributed sharding: each rank takes every world_size-th doc
            if doc_idx % self.world_size != self.rank:
                doc_idx += 1
                continue

            # Skip documents for resumption (fast-forward to saved position)
            if rank_docs < self._skip_rank_docs:
                rank_docs += 1
                doc_idx += 1
                continue

            text = example[self.text_field]
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                rank_docs += 1
                doc_idx += 1
                continue

            buffer.extend(tokens)
            buffer.append(self._eos_id)
            rank_docs += 1
            doc_idx += 1
            self._rank_docs_consumed = rank_docs

            # Yield full chunks from buffer
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]

                if self._packing_enabled:
                    yield _compute_packed_output(np.array(chunk, dtype=np.int64), self._eos_id)
                else:
                    token_tensor = torch.tensor(chunk, dtype=torch.long)
                    yield {
                        "input_ids": token_tensor[:-1],
                        "labels": token_tensor[1:],
                    }

        # Epoch complete — reset for next iteration
        self._epoch += 1
        self._skip_rank_docs = 0
        self._rank_docs_consumed = 0

    def state_dict(self) -> dict:
        """Return checkpoint state. Keys: ``epoch``, ``rank_docs_consumed``."""
        return {
            "epoch": self._epoch,
            "rank_docs_consumed": self._rank_docs_consumed,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint. Sets skip count to fast-forward on next iteration."""
        self._epoch = state.get("epoch", 0)
        self._skip_rank_docs = state.get("rank_docs_consumed", 0)
        self._rank_docs_consumed = 0


class MixtureDataset(Dataset):
    """Concatenates multiple datasets for weighted mixing.

    Global index space maps to sub-datasets via cumulative offsets.
    Each sample includes ``dataset_idx`` (integer) so the training loop
    can compute per-dataset metrics.

    Args:
        datasets: List of map-style datasets to mix.
        names: Human-readable name per dataset (for metrics logging).
    """

    def __init__(self, datasets: list[Dataset], names: list[str]) -> None:
        if len(datasets) != len(names):
            raise ValueError("datasets and names must have the same length")
        if not datasets:
            raise ValueError("At least one dataset is required")

        self._datasets = datasets
        self._names = names
        self._cumulative: list[int] = [0]
        for ds in datasets:
            self._cumulative.append(self._cumulative[-1] + len(ds))

        total = self._cumulative[-1]
        logger.info(
            f"MixtureDataset: {len(datasets)} sources, {total:,} total samples "
            f"({', '.join(f'{n}={len(d):,}' for n, d in zip(names, datasets, strict=True))})"
        )

    @property
    def cumulative_sizes(self) -> list[int]:
        """Cumulative dataset sizes: ``[0, len(ds0), len(ds0)+len(ds1), ...]``."""
        return list(self._cumulative)

    @property
    def dataset_names(self) -> list[str]:
        return list(self._names)

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        ds_idx = bisect.bisect_right(self._cumulative, idx) - 1
        local_idx = idx - self._cumulative[ds_idx]
        sample = self._datasets[ds_idx][local_idx]
        sample["dataset_idx"] = ds_idx
        return sample

    def state_dict(self) -> dict:
        """Return per-sub-dataset checkpoint state."""
        return {
            f"dataset_{i}": ds.state_dict()
            for i, ds in enumerate(self._datasets)
            if hasattr(ds, "state_dict")
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore per-sub-dataset state."""
        for i, ds in enumerate(self._datasets):
            key = f"dataset_{i}"
            if key in state and hasattr(ds, "load_state_dict"):
                ds.load_state_dict(state[key])
