#!/usr/bin/env python3
"""Validate tokenized datasets for KempnerForge training.

Checks that pre-tokenized data (produced by tatm or other tools) is
compatible with KempnerForge's MemoryMappedDataset, and prints the
exact train command to use.

Tokenization workflow:
    1. Tokenize with tatm:  tatm tokenize --tokenizer <name> --output-dir <path> <dataset>
    2. (Optional) Transfer to fast storage via Globus
    3. Validate:  uv run python scripts/prepare_data.py <path>
    4. Train:     use the printed --data flags

Usage:
    # Validate testbed data
    uv run python scripts/prepare_data.py \
        /n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/fineweb-edu/tokenized/meta-llama-3/default

    # Validate custom tokenized data
    uv run python scripts/prepare_data.py /n/netscratch/kempner_dev/mmsh/datasets/my-dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def validate(data_path: Path) -> None:
    """Validate tokenized data for KempnerForge compatibility."""
    if not data_path.is_dir():
        print(f"Error: not a directory: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Find shards
    bin_shards = sorted(data_path.glob("*.bin"))
    npy_shards = sorted(data_path.glob("*.npy"))
    shards = bin_shards or npy_shards
    fmt = "bin" if bin_shards else "npy"

    if not shards:
        print(f"Error: no .bin or .npy files in {data_path}", file=sys.stderr)
        sys.exit(1)

    # Read tatm metadata if present
    metadata = _read_metadata(data_path)

    # Analyze shards
    total_size = sum(s.stat().st_size for s in shards)

    if fmt == "npy":
        first = np.load(str(shards[0]), mmap_mode="r", allow_pickle=False)
        sample_dtype = first.dtype
        total_tokens = sum(len(np.load(str(s), mmap_mode="r", allow_pickle=False)) for s in shards)
    else:
        dtype_str = "uint32"
        if metadata and "tokenized_info" in metadata:
            dtype_str = metadata["tokenized_info"].get("dtype", "uint32")
        sample_dtype = np.dtype(dtype_str)
        total_tokens = total_size // sample_dtype.itemsize

    vocab_size = None
    tokenizer_name = None
    if metadata and "tokenized_info" in metadata:
        vocab_size = metadata["tokenized_info"].get("vocab_size")
        tokenizer_name = metadata["tokenized_info"].get("tokenizer")

    # Print report
    print(f"\n{'=' * 60}")
    print(f"  {data_path}")
    print(f"{'=' * 60}")
    print(f"  Shards:     {len(shards)} .{fmt} files")
    print(f"  Total size: {total_size / (1024**3):.1f} GB")
    print(f"  Tokens:     {total_tokens:,}")
    print(f"  Dtype:      {sample_dtype}")
    if tokenizer_name:
        print(f"  Tokenizer:  {tokenizer_name}")
    if vocab_size:
        print(f"  Vocab size: {vocab_size:,}")

    # Compatibility check
    if sample_dtype not in (np.dtype("uint16"), np.dtype("uint32")):
        print(f"\n  WARNING: dtype {sample_dtype} — expected uint16 or uint32")
    else:
        print("\n  Compatible with KempnerForge MemoryMappedDataset.")
        print("\n  Train with:")
        print(f"    --data.dataset_path={data_path}")
        print(f"    --data.file_pattern='*.{fmt}'")
        if vocab_size:
            print(f"    --model.vocab_size={vocab_size}")
    print()


def _read_metadata(data_path: Path) -> dict | None:
    """Read metadata.yaml if present (tatm output format)."""
    meta_path = data_path / "metadata.yaml"
    if not meta_path.exists():
        return None
    import yaml

    with open(meta_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate tokenized data for KempnerForge training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tokenization workflow:\n"
            "  1. Tokenize:  tatm tokenize --tokenizer <name> --output-dir <dir> <dataset>\n"
            "  2. Transfer:  use Globus to move to fast storage (VAST) if needed\n"
            "  3. Validate:  uv run python scripts/prepare_data.py <path>\n"
            "  4. Train:     use the printed --data flags with train.py\n"
        ),
    )
    parser.add_argument("path", type=Path, help="Path to tokenized data directory")
    args = parser.parse_args()
    validate(args.path)


if __name__ == "__main__":
    main()
