"""VLM dataset and collator (Joint-Decoder).

``HuggingFaceVLMDataset`` wraps a HuggingFace image-text dataset and
produces the ``VLMSample`` contract:

- ``pixel_values``: ``(3, H, W)`` float tensor, resized to ``image_size``
  and normalized with the provided mean/std.
- ``input_ids``: ``(T,)`` int64 tensor, right-padded to ``max_text_len``.
- ``labels``: ``(T,)`` int64 tensor matching ``input_ids`` with ``-100``
  on padding positions and (optionally) on prompt positions when
  ``prompt_field`` is set.

``VLMCollator`` stacks a list of samples into a batch. All batches are
padded to the same fixed ``max_text_len`` regardless of batch content so
different ranks see identical tensor shapes (no NCCL desync under
FSDP2). The collator also emits ``image_positions: (B,)`` zeros; this
slot is reserved for a future multi-image extension and is unused by
the Joint-Decoder wrapper today.

The HF ``datasets`` and ``transformers`` packages are imported lazily
so this module is safe to import without them (e.g. in unit tests that
don't exercise the dataset path).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# SigLIP-style normalization defaults; pass explicit mean/std if your
# vision encoder was trained with CLIP normalization.
DEFAULT_IMAGE_MEAN = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD = (0.5, 0.5, 0.5)


def _pil_to_tensor(
    img: Any,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    """Resize, convert to (3, H, W) float tensor, and normalize.

    Accepts a PIL ``Image``. Converts to RGB if needed so grayscale /
    RGBA inputs do not drop into the encoder with the wrong channel
    count.
    """
    from PIL import Image

    if not isinstance(img, Image.Image):
        raise TypeError(
            f"Expected a PIL.Image, got {type(img).__name__}. "
            "If the dataset returns raw bytes, decode with PIL first."
        )
    img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    import numpy as np

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)
    mean_t = torch.tensor(mean, dtype=t.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=t.dtype).view(3, 1, 1)
    return (t - mean_t) / std_t


def _tokenize_and_mask(
    tokenizer: Any,
    text: str,
    max_text_len: int,
    prompt: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize and build right-padded input_ids + labels.

    When ``prompt`` is provided, the prompt portion of ``labels`` is
    masked with ``-100`` (loss does not backpropagate through prompt
    tokens). Padding positions in both ``input_ids`` and ``labels`` are
    handled via ``ignore_index=-100`` on the loss.

    BPE and SentencePiece tokenizers are NOT prefix-preserving: in
    general ``tokenize(prompt) + tokenize(text)`` differs from
    ``tokenize(prompt + text)`` at the boundary (tokens can merge or
    split). To guarantee the mask lines up with the prompt boundary we
    tokenize prompt and text independently, then concatenate the id
    lists. The mask length is ``len(prompt_ids)``, so masking
    ``labels[:prompt_len]`` cannot leak a prompt token into supervision
    or erase the first target token.
    """
    if prompt is not None:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        full_ids = list(prompt_ids) + list(text_ids)
        prompt_len = min(len(prompt_ids), max_text_len)
    else:
        full_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        prompt_len = 0

    full_ids = full_ids[:max_text_len]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    n = len(full_ids)
    input_ids = torch.full((max_text_len,), pad_id, dtype=torch.long)
    labels = torch.full((max_text_len,), -100, dtype=torch.long)
    if n > 0:
        ids_tensor = torch.tensor(full_ids, dtype=torch.long)
        input_ids[:n] = ids_tensor
        labels[:n] = ids_tensor
        if prompt_len > 0:
            labels[:prompt_len] = -100
    return input_ids, labels


class HuggingFaceVLMDataset(Dataset):
    """Map-style HF image-text dataset for Joint-Decoder training.

    Args:
        dataset_name: HF dataset name (e.g. ``"sayakpaul/coco-30-val-2014"``)
            or a local directory written by ``datasets.save_to_disk``.
        split: Dataset split.
        image_field: Column name for the PIL image.
        text_field: Column name for the caption / target text.
        tokenizer_path: HF tokenizer id or local path.
        max_text_len: Fixed-length pad target; passed to the collator.
        prompt_field: Optional column name for a prompt that should NOT
            receive loss (e.g. the instruction in an instruction-tuned
            dataset). Prompt tokens get ``labels=-100``.
        image_size: Target square image size. Default 224.
        image_mean / image_std: Normalization stats. Defaults match
            SigLIP's ``(0.5, 0.5, 0.5)``.
        dataset_config: HF dataset config name, if required.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        image_field: str,
        text_field: str,
        tokenizer_path: str,
        max_text_len: int,
        prompt_field: str | None = None,
        image_size: int = 224,
        image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN,
        image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD,
        dataset_config: str | None = None,
    ) -> None:
        import os

        from datasets import Dataset as HFDataset
        from datasets import load_dataset, load_from_disk
        from transformers import AutoTokenizer

        # When dataset_name points at an existing directory on disk (absolute
        # or relative path), prefer load_from_disk. Otherwise treat it as an
        # HF Hub id. This keeps the TOML contract unchanged while letting
        # users preprocess a dataset once and load it without network access.
        is_local = os.path.isdir(dataset_name)
        if is_local:
            loaded = load_from_disk(dataset_name)
            # load_from_disk returns a Dataset if the dir holds a single
            # split, or a DatasetDict of splits; select by split name in
            # the latter case.
            ds = loaded[split] if not isinstance(loaded, HFDataset) else loaded
        else:
            ds = load_dataset(dataset_name, dataset_config, split=split)

        if not isinstance(ds, HFDataset):
            raise TypeError(
                f"Expected a map-style datasets.Dataset, got {type(ds).__name__}. "
                "Streaming or split-dict outputs are not supported by "
                "HuggingFaceVLMDataset (use `split=` to resolve a single split)."
            )
        self._ds: HFDataset = ds
        self._image_field = image_field
        self._text_field = text_field
        self._prompt_field = prompt_field
        self._image_size = image_size
        self._image_mean = image_mean
        self._image_std = image_std
        self._max_text_len = max_text_len
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(
            f"HuggingFaceVLMDataset: {dataset_name} [{split}], {len(self._ds):,} samples, "
            f"image_size={image_size}, max_text_len={max_text_len}"
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._ds[idx]
        pixels = _pil_to_tensor(
            row[self._image_field], self._image_size, self._image_mean, self._image_std
        )
        prompt_val = row[self._prompt_field] if self._prompt_field is not None else None
        text_val = row[self._text_field]
        if not isinstance(text_val, str):
            raise TypeError(
                f"Dataset field {self._text_field!r} must be str, got {type(text_val).__name__}"
            )
        prompt_str = prompt_val if isinstance(prompt_val, str) else None
        input_ids, labels = _tokenize_and_mask(
            self._tokenizer, text_val, self._max_text_len, prompt_str
        )
        return {"pixel_values": pixels, "input_ids": input_ids, "labels": labels}


class VLMCollator:
    """Stack VLM samples into a fixed-length batch.

    Output keys:
      - ``pixel_values``: ``(B, 3, H, W)``.
      - ``input_ids``: ``(B, max_text_len)`` int64.
      - ``labels``: ``(B, max_text_len)`` int64 with ``-100`` on pad.
      - ``image_positions``: ``(B,)`` long tensor. Reserved slot for
        multi-image extensions; currently all zeros (single image per
        example placed at sequence position 0).

    Padding is always to ``max_text_len``, never batch-max, so ranks
    always see identical tensor shapes under FSDP2.
    """

    def __init__(self, pad_id: int, max_text_len: int) -> None:
        if max_text_len <= 0:
            raise ValueError("max_text_len must be positive")
        self.pad_id = pad_id
        self.max_text_len = max_text_len

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("VLMCollator received an empty batch")
        B = len(samples)
        pixels = torch.stack([s["pixel_values"] for s in samples], dim=0)
        input_ids = torch.full((B, self.max_text_len), self.pad_id, dtype=torch.long)
        labels = torch.full((B, self.max_text_len), -100, dtype=torch.long)
        for i, s in enumerate(samples):
            ids = s["input_ids"]
            lbl = s["labels"]
            n = min(ids.shape[0], self.max_text_len)
            input_ids[i, :n] = ids[:n]
            labels[i, :n] = lbl[:n]
        image_positions = torch.zeros(B, dtype=torch.long)
        return {
            "pixel_values": pixels,
            "input_ids": input_ids,
            "labels": labels,
            "image_positions": image_positions,
        }
