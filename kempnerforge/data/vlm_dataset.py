"""VLM dataset and collator (Joint-Decoder).

``HuggingFaceVLMDataset`` wraps a HuggingFace image-text dataset and
produces the ``VLMSample`` contract:

- ``pixel_values``: ``(3, H, W)`` float tensor, resized to ``image_size``
  and normalized with the provided mean/std; ``(V, 3, H, W)`` when
  ``tile_grid`` cuts the image into ``V`` views (``tile_image``).
- ``input_ids``: ``(T,)`` int64 tensor, right-padded to ``max_text_len``.
- ``labels``: ``(T,)`` int64 next-token targets (``labels[i]`` = input
  token ``i+1``), right-padded to ``max_text_len``; pad, trailing, and
  prompt-predicting positions are ``-100``.
- ``text_len``: 0-d int64, how many leading positions of ``input_ids`` are
  real tokens (the rest is padding).

``VLMCollator`` stacks a list of samples into a batch. All batches are
padded to the same fixed ``max_text_len`` regardless of batch content so
different ranks see identical tensor shapes (no NCCL desync under
FSDP2). The collator also emits ``image_positions: (B,)`` zeros; this
slot is reserved for a future multi-image extension and is unused by
the Joint-Decoder wrapper today.

``PackedVLMCollator`` packs several documents into each row instead
(``pack_vlm_samples``): a document is its image's placeholder slots followed
by its real text, rows are filled first-fit to ``seq_len``, and the batch
carries ``doc_ids``, per-document ``position_ids`` and the image placement the
Joint-Decoder wrapper uses to write each image's tokens over its slots.

The HF ``datasets`` and ``transformers`` packages are imported lazily
so this module is safe to import without them (e.g. in unit tests that
don't exercise the dataset path).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_samples_masked_for_prompt = itertools.count(1)

DEFAULT_IMAGE_MEAN = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD = (0.5, 0.5, 0.5)


def pil_to_tensor(
    img: Any,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    """Resize, convert to (3, H, W) float tensor, and normalize.

    Accepts a PIL ``Image``. Converts to RGB if needed so grayscale /
    RGBA inputs do not drop into the encoder with the wrong channel
    count.

    Public so that out-of-tree callers (e.g. the VLM evaluation adapter)
    reuse the exact training-time preprocessing as the single source of
    truth, rather than re-implementing resize/normalize.
    """
    from PIL import Image

    _require_pil(img)
    img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    import numpy as np

    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean_t = torch.tensor(mean, dtype=t.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=t.dtype).view(3, 1, 1)
    return (t - mean_t) / std_t


def _require_pil(img: Any) -> None:
    from PIL import Image

    if not isinstance(img, Image.Image):
        raise TypeError(
            f"Expected a PIL.Image, got {type(img).__name__}. "
            "If the dataset returns raw bytes, decode with PIL first."
        )


def tile_image(
    img: Any,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    *,
    tile_grid: int,
    tile_global: bool = True,
) -> torch.Tensor:
    """Cut an image into views: the whole image, then an ``n × n`` raster of crops.

    The whole image, resized to ``image_size`` square exactly as ``pil_to_tensor``
    does, comes first when ``tile_global``. The crops are cut from the image
    resized to ``tile_grid * image_size`` square, so each is ``image_size`` px and
    together they cover that canvas exactly; they follow in raster order (left
    to right, then top to bottom). Returns ``(views, 3, image_size,
    image_size)``, every view normalized like ``pil_to_tensor``.

    Public for the same reason as ``pil_to_tensor``: an evaluation adapter must
    tile a request's image exactly as training did.
    """
    if tile_grid < 2:
        raise ValueError(f"tile_grid must be >= 2 (an n × n grid of crops), got {tile_grid}")
    _require_pil(img)
    views = []
    if tile_global:
        views.append(pil_to_tensor(img, image_size, mean, std).unsqueeze(0))
    canvas = pil_to_tensor(img, tile_grid * image_size, mean, std)
    crops = canvas.reshape(3, tile_grid, image_size, tile_grid, image_size)
    views.append(
        crops.permute(1, 3, 0, 2, 4).reshape(tile_grid * tile_grid, 3, image_size, image_size)
    )
    return torch.cat(views, dim=0)


def image_to_views(
    img: Any,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    *,
    tile_grid: int = 0,
    tile_global: bool = True,
) -> torch.Tensor:
    """One ``(3, H, W)`` view, or the ``(V, 3, H, W)`` views of ``tile_image`` when tiling."""
    if tile_grid:
        return tile_image(img, image_size, mean, std, tile_grid=tile_grid, tile_global=tile_global)
    return pil_to_tensor(img, image_size, mean, std)


def frames_to_clip_tensor(
    frames: list[Any],
    *,
    max_frames: int,
    frame_size: int,
    image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN,
    image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack frames into a fixed ``(max_frames, 3, H, W)`` zero-padded clip + mask.

    Fills the first ``min(len(frames), max_frames)`` slots with per-frame
    ``pil_to_tensor`` output (temporal order preserved) and leaves the remaining
    slots zero; also returns a ``(max_frames,)`` bool ``frame_mask`` (``True`` for
    real frames). A single image is the length-1 case (a 1-frame clip).

    Args:
        frames: Ordered list of PIL ``Image`` objects (e.g. decoded clip frames).
        max_frames: Fixed per-clip frame budget (the output's frame dimension).
        frame_size: Square pixel size each frame is resized to.
        image_mean / image_std: Per-channel normalization (SigLIP defaults).
        dtype: Output ``pixel_values`` dtype.

    Returns:
        ``(pixel_values, frame_mask)`` of shapes ``(max_frames, 3, frame_size,
        frame_size)`` and ``(max_frames,)``.
    """
    pixel_values = torch.zeros(max_frames, 3, frame_size, frame_size, dtype=dtype)
    frame_mask = torch.zeros(max_frames, dtype=torch.bool)
    n_real = min(len(frames), max_frames)
    for i in range(n_real):
        pixel_values[i] = pil_to_tensor(frames[i], frame_size, image_mean, image_std)
        frame_mask[i] = True
    return pixel_values, frame_mask


def build_tokenizer(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


def resolve_pad_id(tokenizer: Any) -> int:
    """Resolve a pad id, falling back to EOS then 0.

    Mirrors what the dataset uses to right-pad ``input_ids``; tokenizers
    without a dedicated pad token fall back to
    the EOS id, and finally to ``0`` if neither is set.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    return pad_id


def _min_text_len_for_prompt(num_prompt_tokens: int, tokenizer: Any) -> int:
    """Smallest ``max_text_len`` that still supervises a caption token after a prompt.

    Room for the prompt's tokens, one caption token (which the last prompt token
    predicts), and the EOS that caption token predicts when the tokenizer has one.
    """
    return num_prompt_tokens + 1 + int(tokenizer.eos_token_id is not None)


def check_prompt_fits(tokenizer: Any, prompt: str, max_text_len: int) -> None:
    """Reject a prompt that leaves no room in ``max_text_len`` for a caption token.

    Args:
        tokenizer: The tokenizer the samples are built with.
        prompt: The prompt, fixed for the whole run.
        max_text_len: The fixed text length of every sample.

    Raises:
        ValueError: ``max_text_len`` is below the prompt's token count plus one
            caption token, plus one more when the tokenizer defines an EOS.
    """
    num_prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    min_text_len = _min_text_len_for_prompt(num_prompt_tokens, tokenizer)
    if min_text_len > max_text_len:
        raise ValueError(
            f"The prompt is {num_prompt_tokens} tokens, so max_text_len={max_text_len} "
            f"leaves no room after it for a caption token (needs max_text_len >= "
            f"{min_text_len}); raise vlm.max_text_len or shorten the prompt"
        )


def _prompt_or_none(value: Any, prompt_field: str) -> str | None:
    """The value of ``prompt_field`` in a row, which must be a ``str`` or ``None``.

    Raises:
        TypeError: ``value`` is neither a ``str`` nor ``None``.
    """
    if value is not None and not isinstance(value, str):
        raise TypeError(
            f"Dataset field {prompt_field!r} must be str or None, got {type(value).__name__}"
        )
    return value


def _check_prompt_column_is_string(rows: Any, prompt_field: str) -> None:
    """Reject a prompt column whose type admits a value that is neither a string nor missing.

    Args:
        rows: A ``datasets.Dataset`` with a ``prompt_field`` column.
        prompt_field: The prompt column's name.

    Raises:
        TypeError: The column is not a ``datasets.Value`` of dtype ``string`` or
            ``large_string``, or ``null`` when no row has a prompt.
        ValueError: ``prompt_field`` is not a column of ``rows``.
    """
    from datasets import Value

    feature = rows.select_columns([prompt_field]).features[prompt_field]
    if not (isinstance(feature, Value) and feature.dtype in ("string", "large_string", "null")):
        raise TypeError(
            f"Dataset field {prompt_field!r} must be a string column (a Value of dtype "
            f"'string' or 'large_string', or 'null' when no row has a prompt), got {feature}"
        )


def _check_prompt_column_has_room(
    rows: Any, prompt_field: str, tokenizer: Any, max_text_len: int
) -> None:
    """Reject a prompt column in which no row leaves room for a caption token.

    Args:
        rows: A ``datasets.Dataset`` with a ``prompt_field`` column.
        prompt_field: The prompt column's name.
        tokenizer: The tokenizer the samples are built with.
        max_text_len: The fixed text length of every sample.

    Raises:
        TypeError: A prompt read is neither a ``str`` nor ``None``.
        ValueError: The column has rows and none of them leaves room for a caption
            token; the message names the shortest prompt's token count and the
            ``max_text_len`` it needs.
    """
    shortest: int | None = None
    for row in rows.select_columns([prompt_field]):
        prompt = _prompt_or_none(row[prompt_field], prompt_field)
        if prompt is None:
            return
        num_prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if _min_text_len_for_prompt(num_prompt_tokens, tokenizer) <= max_text_len:
            return
        shortest = num_prompt_tokens if shortest is None else min(shortest, num_prompt_tokens)
    if shortest is not None:
        raise ValueError(
            f"No prompt in column {prompt_field!r} leaves room in max_text_len={max_text_len} "
            f"for a caption token: the shortest is {shortest} tokens (needs max_text_len >= "
            f"{_min_text_len_for_prompt(shortest, tokenizer)}); raise vlm.max_text_len or "
            "shorten the prompts"
        )


def _warn_sample_masked_for_prompt(min_text_len: int, max_text_len: int) -> None:
    masked = next(_samples_masked_for_prompt)
    if masked & (masked - 1) == 0:
        logger.warning(
            "A prompt needs max_text_len >= %d to leave room for a caption token, but "
            "max_text_len=%d: masked every label of that sample so it adds no loss "
            "(%d such samples so far in this process, logged at powers of two)",
            min_text_len,
            max_text_len,
            masked,
            extra={"every_rank": True},
        )


def _tokenize_and_mask(
    tokenizer: Any,
    text: str,
    max_text_len: int,
    prompt: str | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Tokenize into right-padded input_ids + next-token labels, plus the real token count.

    ``labels[i]`` is token ``i+1`` (matching text pretraining and the no-shift
    loss); pad, trailing, and prompt-predicting positions are ``-100``. An EOS
    token is appended (when the tokenizer defines one) so the last caption token
    is trained to stop. The first caption token is supervised only when a
    ``prompt`` precedes it (the last prompt token predicts it); a plain,
    unprompted caption leaves the first token unsupervised.

    BPE and SentencePiece tokenizers are NOT prefix-preserving: in
    general ``tokenize(prompt) + tokenize(text)`` differs from
    ``tokenize(prompt + text)`` at the boundary (tokens can merge or
    split). To guarantee the mask lines up with the prompt boundary we
    tokenize prompt and text independently, then concatenate the id
    lists, so the prompt/target boundary is exact and only positions
    predicting prompt tokens are masked.

    A prompt that leaves no room for a caption token (the bound
    ``check_prompt_fits`` enforces) masks every label, so the sample adds no
    loss; each process, on every rank, logs a warning at the 1st, 2nd, 4th, ...
    such sample. Truncating a long caption is unaffected.
    """
    if prompt is not None:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        full_ids = list(prompt_ids) + list(text_ids)
        prompt_len = min(len(prompt_ids), max_text_len)
        min_text_len = _min_text_len_for_prompt(len(prompt_ids), tokenizer)
    else:
        full_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        prompt_len = 0
        min_text_len = 0

    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        full_ids = full_ids[: max_text_len - 1] + [eos_id]
    else:
        full_ids = full_ids[:max_text_len]
    pad_id = resolve_pad_id(tokenizer)

    n = len(full_ids)
    input_ids = torch.full((max_text_len,), pad_id, dtype=torch.long)
    labels = torch.full((max_text_len,), -100, dtype=torch.long)
    if n > 0:
        ids_tensor = torch.tensor(full_ids, dtype=torch.long)
        input_ids[:n] = ids_tensor
        labels[: n - 1] = ids_tensor[1:]
        if prompt_len > 1:
            labels[: prompt_len - 1] = -100
    if min_text_len > max_text_len:
        labels.fill_(-100)
        _warn_sample_masked_for_prompt(min_text_len, max_text_len)
    return input_ids, labels, n


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
            be supervised (e.g. the instruction in an instruction-tuned
            dataset). Positions predicting prompt tokens are ``-100``; the
            last prompt token still predicts the first caption token. A row
            whose prompt leaves no room in ``max_text_len`` for a caption
            token has every label masked, and building the dataset raises
            ``ValueError`` when no row leaves that room. A ``None`` value
            means the row has no prompt. Building the dataset raises
            ``TypeError`` unless the column holds strings (dtype ``string``
            or ``large_string``, or ``null`` when no row has a prompt).
        image_size: Target square image size. Default 224.
        image_mean / image_std: Normalization stats. Defaults match
            SigLIP's ``(0.5, 0.5, 0.5)``.
        dataset_config: HF dataset config name, if required.
        tile_grid: ``0`` keeps one view per image; ``n`` cuts each image into
            an ``n × n`` grid of ``image_size`` crops (``tile_image``), so
            ``pixel_values`` is ``(views, 3, H, W)``.
        tile_global: Whether a tiled image's views start with the whole image.
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
        tile_grid: int = 0,
        tile_global: bool = True,
    ) -> None:
        import os

        from datasets import Dataset as HFDataset
        from datasets import load_dataset, load_from_disk

        is_local = os.path.isdir(dataset_name)
        if is_local:
            loaded = load_from_disk(dataset_name)
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
        self._tile_grid = tile_grid
        self._tile_global = tile_global
        self._max_text_len = max_text_len
        self._tokenizer = build_tokenizer(tokenizer_path)
        if prompt_field is not None:
            _check_prompt_column_is_string(ds, prompt_field)
            _check_prompt_column_has_room(ds, prompt_field, self._tokenizer, max_text_len)
        logger.info(
            f"HuggingFaceVLMDataset: {dataset_name} [{split}], {len(self._ds):,} samples, "
            f"image_size={image_size}, tile_grid={tile_grid}, max_text_len={max_text_len}"
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._ds[idx]
        pixels = image_to_views(
            row[self._image_field],
            self._image_size,
            self._image_mean,
            self._image_std,
            tile_grid=self._tile_grid,
            tile_global=self._tile_global,
        )
        text_val = row[self._text_field]
        if not isinstance(text_val, str):
            raise TypeError(
                f"Dataset field {self._text_field!r} must be str, got {type(text_val).__name__}"
            )
        prompt = None
        if self._prompt_field is not None:
            prompt = _prompt_or_none(row[self._prompt_field], self._prompt_field)
        input_ids, labels, text_len = _tokenize_and_mask(
            self._tokenizer, text_val, self._max_text_len, prompt
        )
        return {
            "pixel_values": pixels,
            "input_ids": input_ids,
            "labels": labels,
            "text_len": torch.tensor(text_len, dtype=torch.long),
        }


class VLMCollator:
    """Stack VLM samples into a fixed-length batch.

    Output keys:
      - ``pixel_values``: ``(B, 3, H, W)``, or ``(B, V, 3, H, W)`` when every
        sample carries ``V`` views of its image.
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


def pack_vlm_samples(
    samples: list[dict[str, torch.Tensor]],
    *,
    seq_len: int,
    tokens_per_image: int,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Pack image-text samples into rows of ``seq_len``, each document's image before its text.

    A sample is one document: ``pixel_values`` ``(3, H, W)`` or ``(V, 3, H, W)``
    (``V`` views, each an encoder input), ``input_ids`` and ``labels`` ``(T,)``
    whose first ``text_len`` positions are real (``0`` packs the image alone
    with nothing to supervise, as the unpacked path's all-masked row does). Its footprint is ``V *
    tokens_per_image`` placeholder slots followed by ``text_len`` tokens. The
    last packed label is forced to ``-100`` as a guard for samples built outside
    ``HuggingFaceVLMDataset`` (which already leaves it so): the token after a
    trimmed ``text_len`` is not in the row.
    Documents go first-fit into rows in order: into the first open row with
    room, else a new row. A row's tail is padding that forms its own document.

    Returns:
        ``pixel_values`` ``(N, 3, H, W)``, every image or view across the batch in
        document order; ``input_ids``, ``labels`` and ``doc_ids`` ``(rows,
        seq_len)``, with ``pad_id`` and ``-100`` on the image slots and the
        padding tail, and ``doc_ids`` numbering a row's documents from ``0`` with
        the tail one past the last; ``position_ids`` ``(rows, seq_len)``, each
        document's positions (and the tail's) restarting at ``0``;
        ``image_rows`` and ``image_starts`` ``(N,)``,
        the row and first slot of each image; ``image_tokens``, 0-d, the
        ``tokens_per_image`` the slots were laid out with; ``text_rows`` and
        ``text_positions`` ``(M,)``, the row and position of every real text
        token in document order, where the wrapper runs the LM head.

    Raises:
        ValueError: ``samples`` is empty, a sample lacks ``text_len``, a document
            has no image or a footprint over ``seq_len``, or ``seq_len`` or
            ``tokens_per_image`` is not positive.
    """
    if seq_len <= 0 or tokens_per_image <= 0:
        raise ValueError("seq_len and tokens_per_image must be positive")
    if not samples:
        raise ValueError("pack_vlm_samples received no samples")
    rows: list[dict[str, Any]] = []
    images: list[torch.Tensor] = []
    image_rows: list[int] = []
    image_starts: list[int] = []
    text_rows: list[int] = []
    text_positions: list[int] = []
    for sample in samples:
        views = sample["pixel_values"]
        if views.dim() == 3:
            views = views.unsqueeze(0)
        if views.shape[0] == 0:
            raise ValueError("pack_vlm_samples: every document needs at least one image")
        n_ids = sample["input_ids"].shape[0]
        if "text_len" not in sample:
            raise ValueError(
                "pack_vlm_samples: every sample needs text_len, its count of real tokens; "
                "a right-padded sample without it would pack its padding as text"
            )
        text_len = int(sample["text_len"])
        if not 0 <= text_len <= n_ids:
            raise ValueError(f"text_len ({text_len}) must lie within the {n_ids} input_ids")
        footprint = views.shape[0] * tokens_per_image + text_len
        if footprint > seq_len:
            raise ValueError(
                f"a document of {views.shape[0]} image(s) x {tokens_per_image} tokens + "
                f"{text_len} text tokens does not fit a row of seq_len={seq_len}"
            )
        row_index = next((i for i, r in enumerate(rows) if r["fill"] + footprint <= seq_len), None)
        if row_index is None:
            row_index = len(rows)
            rows.append(
                {
                    "fill": 0,
                    "docs": 0,
                    "input_ids": torch.full((seq_len,), pad_id, dtype=torch.long),
                    "labels": torch.full((seq_len,), -100, dtype=torch.long),
                    "doc_ids": torch.zeros(seq_len, dtype=torch.long),
                    "position_ids": torch.zeros(seq_len, dtype=torch.long),
                }
            )
        row = rows[row_index]
        cursor = row["fill"]
        for view in views:
            images.append(view)
            image_rows.append(row_index)
            image_starts.append(cursor)
            cursor += tokens_per_image
        text_rows.extend([row_index] * text_len)
        text_positions.extend(range(cursor, cursor + text_len))
        row["input_ids"][cursor : cursor + text_len] = sample["input_ids"][:text_len]
        row["labels"][cursor : cursor + text_len] = sample["labels"][:text_len]
        if text_len:
            row["labels"][cursor + text_len - 1] = -100
        row["doc_ids"][row["fill"] : cursor + text_len] = row["docs"]
        row["position_ids"][row["fill"] : cursor + text_len] = torch.arange(footprint)
        row["fill"] = cursor + text_len
        row["docs"] += 1
    for row in rows:
        row["doc_ids"][row["fill"] :] = row["docs"]
        row["position_ids"][row["fill"] :] = torch.arange(seq_len - row["fill"])
    return {
        "pixel_values": torch.stack(images, dim=0),
        "input_ids": torch.stack([r["input_ids"] for r in rows]),
        "labels": torch.stack([r["labels"] for r in rows]),
        "doc_ids": torch.stack([r["doc_ids"] for r in rows]),
        "position_ids": torch.stack([r["position_ids"] for r in rows]),
        "image_rows": torch.tensor(image_rows, dtype=torch.long),
        "image_starts": torch.tensor(image_starts, dtype=torch.long),
        "image_tokens": torch.tensor(tokens_per_image, dtype=torch.long),
        "text_rows": torch.tensor(text_rows, dtype=torch.long),
        "text_positions": torch.tensor(text_positions, dtype=torch.long),
    }


class PackedVLMCollator:
    """Pack the documents of a batch into rows of ``seq_len`` (``pack_vlm_samples``).

    The number of rows follows the documents' footprints, so it varies from
    batch to batch; ``train.batch_size`` is the number of documents per
    micro-batch, not the number of rows.
    """

    def __init__(self, pad_id: int, seq_len: int, tokens_per_image: int) -> None:
        self.pad_id = pad_id
        self.seq_len = seq_len
        self.tokens_per_image = tokens_per_image

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return pack_vlm_samples(
            samples,
            seq_len=self.seq_len,
            tokens_per_image=self.tokens_per_image,
            pad_id=self.pad_id,
        )
