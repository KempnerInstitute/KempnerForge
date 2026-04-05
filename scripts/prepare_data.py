#!/usr/bin/env python3
"""Download and tokenize a dataset into .npy files for KempnerForge training.

Downloads wikitext-103-raw-v1 from HuggingFace, tokenizes with GPT-2 tokenizer,
and saves as memory-mapped .npy files ready for MemoryMappedDataset.

Usage:
    uv run python scripts/prepare_data.py [--output-dir data/wikitext103] [--tokenizer gpt2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenized .npy data")
    parser.add_argument("--output-dir", type=str, default="data/wikitext103")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument(
        "--max-shard-tokens", type=int, default=50_000_000, help="Max tokens per .npy shard"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  vocab_size: {tokenizer.vocab_size}")

    print("Downloading wikitext-103-raw-v1...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    for split in ["train", "validation", "test"]:
        print(f"\nProcessing {split} split...")
        texts = ds[split]["text"]

        # Filter empty lines and tokenize
        all_tokens = []
        for i, text in enumerate(texts):
            if text.strip():
                tokens = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)
            if (i + 1) % 100_000 == 0:
                print(f"  tokenized {i + 1:,} / {len(texts):,} lines ({len(all_tokens):,} tokens)")

        tokens_array = np.array(all_tokens, dtype=np.uint16)
        print(f"  total: {len(tokens_array):,} tokens")

        # Save as shards
        shard_idx = 0
        offset = 0
        while offset < len(tokens_array):
            end = min(offset + args.max_shard_tokens, len(tokens_array))
            shard = tokens_array[offset:end]
            shard_path = out_dir / f"{split}_{shard_idx:04d}.npy"
            np.save(str(shard_path), shard)
            print(f"  saved {shard_path} ({len(shard):,} tokens)")
            offset = end
            shard_idx += 1

    print(f"\nDone! Data saved to {out_dir}/")
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
