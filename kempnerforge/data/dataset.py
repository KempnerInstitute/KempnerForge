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

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.seq_len = seq_len

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
    ) -> None:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        self.text_field = text_field

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
        tokens = torch.from_numpy(self._packed_sequences[idx].copy())
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
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
