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
    # Validate tokenized data
    uv run python scripts/prepare_data.py /path/to/tokenized/dataset

    # Validate .bin shards that no metadata.yaml describes
    uv run python scripts/prepare_data.py /path/to/shards --token-dtype uint16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from kempnerforge.config.data import TOKEN_DTYPES
from kempnerforge.data.dataset import recorded_token_dtype


def validate(data_path: Path, token_dtype: str = "") -> None:
    """Validate tokenized data for KempnerForge compatibility.

    ``token_dtype`` declares the dtype of ``.bin`` shards whose
    ``metadata.yaml`` records none, as ``--data.token_dtype`` does in training.
    """
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

    bin_dtype = _declared_bin_dtype(data_path, token_dtype) if fmt == "bin" else ""

    # Read tatm metadata if present
    metadata = _read_metadata(data_path)

    # Analyze shards
    total_size = sum(s.stat().st_size for s in shards)

    partial: list[Path] = []
    if fmt == "npy":
        first = np.load(str(shards[0]), mmap_mode="r", allow_pickle=False)
        sample_dtype = first.dtype
        total_tokens = sum(len(np.load(str(s), mmap_mode="r", allow_pickle=False)) for s in shards)
    else:
        sample_dtype = np.dtype(bin_dtype or "uint32")
        partial = [s for s in shards if s.stat().st_size % sample_dtype.itemsize]
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
    if fmt == "bin" and not bin_dtype:
        print(f"  Tokens:     {total_tokens:,} if uint32, {total_size // 2:,} if uint16")
        print("  Dtype:      not declared (no tokenized_info.dtype in metadata.yaml)")
    else:
        print(f"  Tokens:     {total_tokens:,}")
        print(f"  Dtype:      {sample_dtype}")
    if tokenizer_name:
        print(f"  Tokenizer:  {tokenizer_name}")
    if vocab_size:
        print(f"  Vocab size: {vocab_size:,}")

    # Compatibility check
    if partial:
        print(
            f"\n  NOT compatible: {len(partial)} shard(s) are not a whole number of "
            f"{sample_dtype} tokens, starting with {partial[0].name}."
        )
        if bin_dtype:
            print(f"  They are truncated or were not written as {bin_dtype}.")
        else:
            print("  With no declared dtype they could be uint16 or truncated uint32,")
            print("  so training refuses them. Rerun with --token-dtype uint16 or uint32.")
        print()
        sys.exit(1)
    if sample_dtype not in (np.dtype("uint16"), np.dtype("uint32")):
        print(f"\n  WARNING: dtype {sample_dtype} — expected uint16 or uint32")
    else:
        if fmt == "bin" and not bin_dtype:
            print("\n  The .bin dtype is not declared, so training reads these shards as")
            print("  uint32 and warns; if they hold uint16 tokens, every token is misread.")
            print("  Rerun with --token-dtype uint16 or uint32 to check them as that dtype.")
        else:
            print("\n  Compatible with KempnerForge MemoryMappedDataset.")
        print("\n  Train with:")
        print(f"    --data.dataset_path={data_path}")
        print(f"    --data.file_pattern='*.{fmt}'")
        if fmt == "bin":
            print(f"    --data.token_dtype={bin_dtype or '<uint16 or uint32>'}")
        if vocab_size:
            print(f"    --model.vocab_size={vocab_size}")
    print()


def _declared_bin_dtype(data_path: Path, token_dtype: str) -> str:
    """Return the dtype declared for the .bin shards, as training resolves it.

    That is ``token_dtype`` or the ``tokenized_info.dtype`` that ``metadata.yaml``
    records, which must agree; ``""`` when neither declares one. Exits with an
    error when the two disagree or the file cannot be read.
    """
    try:
        recorded = recorded_token_dtype(data_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if token_dtype and recorded and token_dtype != recorded:
        print(
            f"Error: --token-dtype is {token_dtype}, but {data_path / 'metadata.yaml'} "
            f"records {recorded}. One of them is wrong; drop --token-dtype or fix the file.",
            file=sys.stderr,
        )
        sys.exit(1)
    return token_dtype or recorded


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
    parser.add_argument(
        "--token-dtype",
        choices=TOKEN_DTYPES,
        default="",
        help="dtype of .bin shards whose metadata.yaml records none (as --data.token_dtype)",
    )
    args = parser.parse_args()
    validate(args.path, args.token_dtype)


if __name__ == "__main__":
    main()
