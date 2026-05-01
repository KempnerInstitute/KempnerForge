#!/usr/bin/env python3
"""Prepare a small COCO-captions dataset for VLM smoke testing.

Reads the Karpathy-split COCO caption JSON from the shared testbed and
writes an HF ``Dataset`` (saved with ``save_to_disk``) containing
``image`` (PIL) and ``caption`` columns. The saved directory is consumed
by ``HuggingFaceVLMDataset`` via its ``load_from_disk`` fallback.

Usage:
    uv run python scripts/prep_vlm_coco_smoke.py \\
        --out /tmp/vlm_coco_smoke \\
        --num-samples 500

Defaults point at the Kempner shared testbed COCO paths; override with
``--caption-json`` / ``--image-root`` for other dataset layouts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image

_DEFAULT_CAPTION_JSON = (
    "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/annotations/"
    "coco-captions/raw/coco_karpathy_train.json"
)
_DEFAULT_IMAGE_ROOT = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/coco/raw"


def build_dataset(caption_json: str, image_root: str, num_samples: int, split_name: str) -> None:
    from datasets import Dataset

    with open(caption_json) as fh:
        rows = json.load(fh)
    if num_samples > 0:
        rows = rows[:num_samples]

    images: list[Image.Image] = []
    captions: list[str] = []
    skipped = 0
    for row in rows:
        img_path = Path(image_root) / row["image"]
        if not img_path.exists():
            skipped += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"skip {img_path}: {e}")
            skipped += 1
            continue
        images.append(img)
        captions.append(row["caption"])

    print(
        f"Loaded {len(images)} samples, skipped {skipped}; writing to disk as split={split_name!r}"
    )
    ds = Dataset.from_dict({"image": images, "caption": captions})
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory (save_to_disk target)")
    ap.add_argument("--caption-json", default=_DEFAULT_CAPTION_JSON)
    ap.add_argument("--image-root", default=_DEFAULT_IMAGE_ROOT)
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise FileExistsError(f"{args.out} already exists; pick a new path or remove it first")

    ds = build_dataset(args.caption_json, args.image_root, args.num_samples, args.split)
    ds.save_to_disk(args.out)
    # save_to_disk writes the Dataset (not a DatasetDict). HuggingFaceVLMDataset
    # handles both shapes via its is_local path; for this prep we store a
    # single split so load_from_disk returns a bare Dataset that matches
    # split="train" by convention.
    print(f"Saved {len(ds)} samples to {args.out}")


if __name__ == "__main__":
    main()
