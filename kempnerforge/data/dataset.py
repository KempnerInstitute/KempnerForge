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
import contextlib
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from kempnerforge.config.data import TOKEN_DTYPES, check_token_dtype

logger = logging.getLogger(__name__)

TOKEN_FILE_SUFFIXES = (".bin", ".npy")
TOKEN_METADATA_FILE = "metadata.yaml"


def recorded_token_dtype(directory: Path) -> str:
    """Return the ``tokenized_info.dtype`` that ``directory/metadata.yaml`` records.

    tatm writes this file next to the ``.bin`` shards it produces. Returns
    ``""`` when nothing is recorded: the file is absent, or it has no
    ``tokenized_info.dtype``. A file that cannot be read or parsed, or that
    records a dtype other than ``uint16`` or ``uint32``, raises instead of
    falling through, since the shards beside it would then be read with a
    dtype that nothing declares.
    """
    path = directory / TOKEN_METADATA_FILE
    if not path.exists():
        return ""
    try:
        meta = yaml.safe_load(path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        raise ValueError(
            f"Cannot read the token dtype of the .bin shards in {directory}: {path} could not "
            f"be read as YAML. Fix the file, or remove it if it does not describe these shards. "
            f"({type(e).__name__}: {e})"
        ) from e
    info = meta.get("tokenized_info") if isinstance(meta, dict) else None
    dtype = info.get("dtype") if isinstance(info, dict) else None
    if dtype is None or dtype == "":
        return ""
    if dtype not in TOKEN_DTYPES:
        raise ValueError(
            f"{path} records tokenized_info.dtype {dtype!r}, but raw .bin token shards "
            f"are read as 'uint16' or 'uint32'. Correct the file if the shards hold one of "
            f"those, or rewrite them as uint16, uint32 or .npy."
        )
    return dtype


def _declared_token_dtype(directory: Path, token_dtype: str) -> str:
    """Return the dtype declared for the ``.bin`` shards in ``directory``, or ``""``.

    ``token_dtype`` and the dtype that ``metadata.yaml`` records must agree when
    both are set: one of them is wrong, and reading with the wrong one misreads
    every token.
    """
    recorded = recorded_token_dtype(directory)
    if token_dtype and recorded and token_dtype != recorded:
        raise ValueError(
            f"token_dtype is {token_dtype!r}, but {directory / TOKEN_METADATA_FILE} records "
            f"tokenized_info.dtype {recorded!r} for the .bin shards beside it. One of the two "
            f"is wrong; leave token_dtype empty to use the recorded dtype, or correct the file."
        )
    return token_dtype or recorded


def _token_dtype_remedy(directory: Path) -> str:
    """Return the two ways to declare the dtype of the ``.bin`` shards in ``directory``."""
    return (
        f"set token_dtype to 'uint16' or 'uint32' (data.token_dtype, a [[data.datasets]] "
        f"entry's token_dtype, or eval.token_dtype in a config), or record it as "
        f"tokenized_info.dtype in {directory / TOKEN_METADATA_FILE}"
    )


def _bin_shard_dtype(path: Path, declared: str) -> np.dtype:
    """Return the dtype to read the raw ``.bin`` shard ``path`` with.

    A shard whose size is not a whole number of tokens of its declared dtype
    is rejected: it is truncated, or it was written with another dtype. With
    no declared dtype the shard is read as uint32 if its size is a whole
    number of uint32 tokens, as an intact uint32 shard always is. Any other
    size fits a uint16 shard and a truncated uint32 shard alike, so it is
    rejected rather than guessed.
    """
    size = path.stat().st_size
    dtype = np.dtype(declared or "uint32")
    if size % dtype.itemsize == 0:
        return dtype
    if declared:
        raise ValueError(
            f"{path} is {size:,} bytes, which is not a whole number of {declared} tokens "
            f"({dtype.itemsize} bytes each), so the shard is truncated or was not written as "
            f"{declared}. Replace the shard, or correct the dtype declared for it (token_dtype "
            f"or {path.parent / TOKEN_METADATA_FILE})."
        )
    raise ValueError(
        f"Cannot tell whether {path} holds uint16 or uint32 tokens: no dtype is declared for "
        f"it, and its {size:,} bytes are not a whole number of uint32 tokens, so it is either "
        f"uint16 data or a truncated uint32 shard. To read it, "
        f"{_token_dtype_remedy(path.parent)}."
    )


def _bin_shard_dtypes(files: list[Path], token_dtype: str) -> dict[Path, np.dtype]:
    """Resolve the dtype of every ``.bin`` file in ``files``, one directory at a time."""
    declared: dict[Path, str] = {}
    dtypes: dict[Path, np.dtype] = {}
    for f in files:
        if f.suffix != ".bin":
            continue
        if f.parent not in declared:
            declared[f.parent] = _declared_token_dtype(f.parent, token_dtype)
        dtypes[f] = _bin_shard_dtype(f, declared[f.parent])
    for directory in sorted(d for d, dtype in declared.items() if not dtype):
        logger.warning(
            f"No token dtype is declared for the .bin shards in {directory}, so they are read "
            f"as uint32. If they hold uint16 tokens, every token id is misread; to declare it, "
            f"{_token_dtype_remedy(directory)}."
        )
    return dtypes


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
    # Assign a document ID to each token: increment after every EOS, i.e.
    # doc_ids[i] is the number of EOS tokens strictly before i. Vectorized
    # because this runs per __getitem__, once per token -- a Python loop here
    # costs seq_len interpreter iterations on every sample the loader yields.
    doc_ids = np.zeros(len(tokens), dtype=np.int64)
    doc_ids[1:] = np.cumsum(tokens[:-1] == eos_token_id)

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
    """Pre-tokenized dataset backed by memory-mapped token files.

    Expects 1D arrays of uint16/uint32 token IDs that have been pre-packed
    into fixed-length sequences, as ``.npy`` files or raw ``.bin`` shards.
    Each matched file is read by its own suffix, so a pattern may match
    both formats; a matched file with any other suffix is rejected.

    A ``.npy`` file carries its dtype in its header. A ``.bin`` file does not:
    its dtype is ``token_dtype``, else the ``tokenized_info.dtype`` recorded in
    the ``metadata.yaml`` in its directory (tatm writes one). The two must
    agree when both are given, and a shard whose size is not a whole number
    of tokens of that dtype is rejected. When neither declares a dtype, the
    shards are read as uint32 with a warning, provided each is a whole number
    of uint32 tokens; otherwise the dataset refuses to guess and raises.

    File layout: each file stores a flat array of tokens. The dataset
    splits them into non-overlapping chunks of ``seq_len`` tokens.
    Multiple files are concatenated logically.

    Args:
        data_dir: Directory containing .npy or .bin token files.
        seq_len: Sequence length (number of tokens per sample).
        file_pattern: Glob pattern for data files.
        token_dtype: ``"uint16"`` or ``"uint32"`` for the ``.bin`` files, or
            ``""`` to take it from their ``metadata.yaml`` as described above.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        file_pattern: str = "*.npy",
        pack_sequences: bool = False,
        eos_token_id: int | None = None,
        token_dtype: str = "",
    ) -> None:
        self.seq_len = seq_len
        self._pack_sequences = pack_sequences
        self._eos_token_id = eos_token_id
        if pack_sequences and eos_token_id is None:
            raise ValueError("eos_token_id is required when pack_sequences=True")
        check_token_dtype(token_dtype, "token_dtype")

        # Discover and sort data files for deterministic ordering
        data_path = Path(data_dir)
        self._files = sorted(data_path.glob(file_pattern))
        if not self._files:
            raise FileNotFoundError(f"No files matching {file_pattern!r} in {data_dir}")
        unsupported = [f for f in self._files if f.suffix not in TOKEN_FILE_SUFFIXES]
        if unsupported:
            raise ValueError(
                f"file_pattern {file_pattern!r} matches {unsupported[0]}, which is neither "
                f"a .bin nor a .npy token file ({len(unsupported)} such file(s) in "
                f"{data_dir}). MemoryMappedDataset reads raw .bin token shards and .npy "
                f"arrays only; narrow file_pattern so that it matches token files only."
            )
        bin_dtypes = _bin_shard_dtypes(self._files, token_dtype)

        # Memory-map all files and compute cumulative offsets
        self._mmaps: list[np.ndarray] = []
        self._cumulative_samples: list[int] = [0]
        total_tokens = 0

        # If any open fails partway, close the ones we already opened so they
        # don't leak via the exception traceback (pytest, logger.exception,
        # post-mortem debuggers all pin the partial `self` and its mmaps).
        try:
            for f in self._files:
                if f.suffix == ".bin":
                    dtype = bin_dtypes[f]
                    n_tokens = f.stat().st_size // dtype.itemsize
                    mmap = np.memmap(str(f), dtype=dtype, mode="r", shape=(n_tokens,))
                else:
                    mmap = np.load(str(f), mmap_mode="r")
                n_samples = len(mmap) // seq_len
                self._mmaps.append(mmap)
                total_tokens += len(mmap)
                self._cumulative_samples.append(self._cumulative_samples[-1] + n_samples)
        except Exception:
            self._close_mmaps()
            raise

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
            return _compute_packed_output(tokens, self._eos_token_id)  # type: ignore[reportArgumentType]

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

    def _close_mmaps(self) -> None:
        """Release the underlying mmap objects. Idempotent."""
        for mm in self._mmaps:
            inner = getattr(mm, "_mmap", None)
            if inner is not None and not inner.closed:
                # BufferError: live views into the mapping still exist — can't
                # force-close safely; drop the ref and let GC finish it.
                # ValueError: already closed by another code path.
                with contextlib.suppress(BufferError, ValueError):
                    inner.close()
        self._mmaps.clear()

    def close(self) -> None:
        """Release the underlying mmaps. Preferred path; do not rely on ``__del__``."""
        self._close_mmaps()

    def __del__(self) -> None:
        """GC safety net only. Prefer explicit :meth:`close`."""
        with contextlib.suppress(Exception):
            self._close_mmaps()


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
            f"(seq_len={seq_len}) from {len(raw_dataset)} documents"  # type: ignore[reportArgumentType]
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
        ds = ds.shuffle(seed=self.seed + self._epoch, buffer_size=self.shuffle_buffer_size)  # type: ignore[reportCallIssue]
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
            tokens = self._tokenizer.encode(text, add_special_tokens=False)  # type: ignore[reportOptionalMemberAccess]
            if not tokens:
                rank_docs += 1
                doc_idx += 1
                continue

            buffer.extend(tokens)
            buffer.append(self._eos_id)  # type: ignore[reportArgumentType]
            rank_docs += 1
            doc_idx += 1
            self._rank_docs_consumed = rank_docs

            # Yield full chunks from buffer
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]

                if self._packing_enabled:
                    yield _compute_packed_output(np.array(chunk, dtype=np.int64), self._eos_id)  # type: ignore[reportArgumentType]
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
            self._cumulative.append(self._cumulative[-1] + len(ds))  # type: ignore[reportArgumentType]

        total = self._cumulative[-1]
        logger.info(
            f"MixtureDataset: {len(datasets)} sources, {total:,} total samples "
            f"({', '.join(f'{n}={len(d):,}' for n, d in zip(names, datasets, strict=True))})"  # type: ignore[reportArgumentType]
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
            f"dataset_{i}": ds.state_dict()  # type: ignore[reportAttributeAccessIssue]
            for i, ds in enumerate(self._datasets)
            if hasattr(ds, "state_dict")
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore per-sub-dataset state."""
        for i, ds in enumerate(self._datasets):
            key = f"dataset_{i}"
            if key in state and hasattr(ds, "load_state_dict"):
                ds.load_state_dict(state[key])  # type: ignore[reportAttributeAccessIssue]
