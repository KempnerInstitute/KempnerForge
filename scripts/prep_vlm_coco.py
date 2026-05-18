#!/usr/bin/env python3
"""Prepare a COCO-captions dataset for VLM training, eval, or smoke testing.

Reads the Karpathy-split COCO caption JSON(s) and writes an HF ``Dataset``
(single split) or ``DatasetDict`` (``--all-splits``) via ``save_to_disk``.
The output directory is consumed by ``HuggingFaceVLMDataset`` through its
``load_from_disk`` path — point ``data.hf_dataset_name`` at it in your
training TOML.

Output columns:
- ``image`` — HF ``Image()`` feature, returns ``PIL.Image`` at read.
- ``caption`` — single ``str``. Karpathy train rows store one caption
  per row already; val/test rows store a 5-reference list, in which
  case the first reference is used here so the column is always a str
  (matches what ``HuggingFaceVLMDataset`` expects).
- ``references`` — list of ``str`` with *all* captions for the image
  (1 element for train rows, 5 for val/test). Lets eval loops do
  multi-reference scoring without re-reading the JSON.
- ``image_id`` — ``str``. Present in the train JSON; for val/test the
  source JSON omits it, so it's derived from the image filename stem
  (e.g. ``COCO_val2014_000000391895``).

Modes:
- Default — single split. Reads one JSON (``--caption-json``) and writes
  a bare ``Dataset``. ``--num-samples`` defaults to ``500`` so a no-flag
  invocation is a small, fast smoke prep.
- ``--all-splits`` — also reads the sibling ``coco_karpathy_val.json``
  and ``coco_karpathy_test.json`` next to ``--caption-json`` and writes a
  ``DatasetDict`` keyed by ``train`` / ``val`` / ``test``.

Memory: images stream through ``Dataset.from_generator`` with the
``Image()`` feature receiving file paths, so HF reads + encodes each
image as Arrow shards are written. Decoded pixels are never held in a
Python list — peak RAM stays flat regardless of dataset size.

Examples:
    # 1. Smoke-test prep (small slice, single split — default flags):
    uv run python scripts/prep_vlm_coco.py \\
        --out /tmp/vlm_coco_smoke \\
        --num-samples 500

    # 2. Full dataset, all 3 splits, ready for training + eval:
    uv run python scripts/prep_vlm_coco.py \\
        --out /path/to/datasets/coco-karpathy \\
        --num-samples 0 \\
        --all-splits

Defaults point at the Kempner shared testbed COCO paths; override with
``--caption-json`` / ``--image-root`` for other dataset layouts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_DEFAULT_CAPTION_JSON = (
    "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/annotations/"
    "coco-captions/raw/coco_karpathy_train.json"
)
_DEFAULT_IMAGE_ROOT = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/coco/raw"


def _build_split(caption_json: str, image_root: str, num_samples: int, split_label: str):
    """Build a single HF ``Dataset`` for one Karpathy JSON.

    Uses ``Dataset.from_generator`` + the ``Image()`` feature so peak RAM
    stays flat: paths are yielded one at a time and HF reads + encodes
    each image as Arrow shards are built. Decoded pixels are never held
    in a Python list.
    """
    from datasets import Dataset, Features, Image, Sequence, Value
    from tqdm.auto import tqdm

    with open(caption_json) as fh:
        rows = json.load(fh)
    if num_samples > 0:
        rows = rows[:num_samples]

    skipped = {"missing": 0}

    def gen():
        for row in tqdm(rows, desc=f"encoding {split_label}", unit="img"):
            img_rel = row["image"]
            img_path = Path(image_root) / img_rel
            if not img_path.exists():
                skipped["missing"] += 1
                continue
            # Karpathy train rows: caption is str. Val/test rows: caption is
            # a 5-element list. Normalize so the column is always str and
            # surface all references separately for multi-ref eval.
            raw_caption = row["caption"]
            if isinstance(raw_caption, list):
                references = [str(c) for c in raw_caption]
            else:
                references = [str(raw_caption)]
            # Val/test JSONs omit image_id; fall back to the image filename
            # stem (e.g. COCO_val2014_000000391895), which is unique per image.
            image_id = row.get("image_id")
            if image_id is None:
                image_id = Path(img_rel).stem
            yield {
                "image": str(img_path),
                "caption": references[0],
                "references": references,
                "image_id": str(image_id),
            }

    features = Features(
        {
            "image": Image(),
            "caption": Value("string"),
            "references": Sequence(Value("string")),
            "image_id": Value("string"),
        }
    )
    ds = Dataset.from_generator(gen, features=features)
    print(f"split={split_label}: {len(ds):,} samples (skipped {skipped['missing']} missing)")
    return ds


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare a COCO-captions HF dataset for VLM training/eval/smoke. "
            "Writes a bare Dataset by default, or a DatasetDict with --all-splits."
        ),
        epilog=(
            "Examples:\n"
            "  # Smoke prep (default 500 samples, single split):\n"
            "  uv run python scripts/prep_vlm_coco.py --out /tmp/vlm_coco_smoke\n"
            "\n"
            "  # Full dataset, all 3 splits, for training + eval:\n"
            "  uv run python scripts/prep_vlm_coco.py \\\n"
            "      --out /path/to/datasets/coco-karpathy \\\n"
            "      --num-samples 0 --all-splits"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--out",
        required=True,
        help=(
            "Output directory passed to save_to_disk. Must not already exist. "
            "Point data.hf_dataset_name at this path in your training TOML."
        ),
    )
    ap.add_argument(
        "--caption-json",
        default=_DEFAULT_CAPTION_JSON,
        help=(
            "Karpathy-split COCO caption JSON to read. With --all-splits this "
            "selects the directory; sibling val/test JSONs are picked up "
            "automatically. Default: shared testbed train split."
        ),
    )
    ap.add_argument(
        "--image-root",
        default=_DEFAULT_IMAGE_ROOT,
        help="Root directory the JSON's relative image paths resolve against.",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help=(
            "Per-split sample cap (0 = all rows). Default 500 keeps the "
            "no-flag invocation a fast smoke prep; pass 0 for a real run."
        ),
    )
    ap.add_argument(
        "--split",
        default="train",
        help="Split label for single-split mode (ignored when --all-splits).",
    )
    ap.add_argument(
        "--all-splits",
        action="store_true",
        help=(
            "Read sibling coco_karpathy_{train,val,test}.json next to "
            "--caption-json and save a DatasetDict keyed by train/val/test."
        ),
    )
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise FileExistsError(f"{args.out} already exists; pick a new path or remove it first")

    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if args.all_splits:
        from datasets import DatasetDict

        caption_dir = Path(args.caption_json).parent
        split_files = {
            "train": caption_dir / "coco_karpathy_train.json",
            "val": caption_dir / "coco_karpathy_val.json",
            "test": caption_dir / "coco_karpathy_test.json",
        }
        missing = [k for k, p in split_files.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"--all-splits expected sibling JSONs in {caption_dir}; missing: {missing}"
            )

        splits = {
            label: _build_split(str(path), args.image_root, args.num_samples, label)
            for label, path in split_files.items()
        }
        ddict = DatasetDict(splits)
        ddict.save_to_disk(args.out)
        total = sum(len(d) for d in splits.values())
        print(f"Saved DatasetDict ({total:,} samples across {list(splits)}) to {args.out}")
    else:
        ds = _build_split(args.caption_json, args.image_root, args.num_samples, args.split)
        ds.save_to_disk(args.out)
        # save_to_disk writes the Dataset (not a DatasetDict). HuggingFaceVLMDataset
        # handles both shapes via its is_local path; for this prep we store a
        # single split so load_from_disk returns a bare Dataset that matches
        # split="train" by convention.
        print(f"Saved {len(ds):,} samples to {args.out}")


if __name__ == "__main__":
    main()
