"""Training/eval data pipeline construction and phase (annealing) scheduling.

The builders here own every ``[data]`` / ``[eval]`` branch that used to sit
inline in the training entry point: pre-tokenized mmap, HuggingFace (eager or
streaming), multi-dataset mixtures, and the VLM image/video paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader as TorchDataLoader

from kempnerforge.config.data import TrainingPhase
from kempnerforge.config.job import JobConfig
from kempnerforge.data.dataloader import StatefulDataLoader
from kempnerforge.data.dataset import (
    HuggingFaceDataset,
    MemoryMappedDataset,
    MixtureDataset,
    StreamingHuggingFaceDataset,
)
from kempnerforge.data.sampler import DistributedSampler, MixtureSampler
from kempnerforge.distributed.utils import get_dp_info
from kempnerforge.metrics.logger import get_logger
from kempnerforge.training.eval import should_build_eval_dataloader
from kempnerforge.training.runtime import RuntimeContext

logger = get_logger(__name__)


@dataclass
class DataPipeline:
    """The training dataset/loader plus the mixture state phases need.

    ``dataloader is None`` means no data source was configured; the text and
    PP step bodies then fall back to random tokens. ``dp_rank`` / ``dp_size``
    record the data-parallel partition the samplers were built for.
    """

    dataloader: Any | None = None
    dp_rank: int = 0
    dp_size: int = 1
    mixture_dataset: MixtureDataset | None = None
    mixture_sampler: MixtureSampler | None = None
    mixture_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Phase scheduling needs both mixture handles and a weight for every
        # dataset name; either gap would surface only at the first transition,
        # as an AttributeError or a KeyError.
        if (self.mixture_dataset is None) != (self.mixture_sampler is None):
            raise ValueError("mixture_dataset and mixture_sampler must be set together")
        if self.mixture_dataset is not None and set(self.mixture_weights) != set(
            self.mixture_dataset.dataset_names
        ):
            raise ValueError(
                f"mixture_weights keys {sorted(self.mixture_weights)} must cover every "
                f"mixture dataset name {sorted(set(self.mixture_dataset.dataset_names))}"
            )


@dataclass
class PhaseState:
    """Data-annealing phases and the currently active scaling."""

    phases: list[TrainingPhase] = field(default_factory=list)
    original_weights: dict[str, float] = field(default_factory=dict)
    temperature: float = 1.0
    next_idx: int = 0
    lr_scale: float = 1.0


def _resolve_eos_token_id(config: JobConfig) -> int | None:
    """EOS id for sequence packing in ``MemoryMappedDataset`` (None when unused)."""
    if not config.data.pack_sequences:
        return None
    has_mmap = bool(config.data.dataset_path) or any(s.path for s in config.data.datasets)
    if not has_mmap:
        return None
    if not config.data.tokenizer_path:
        raise ValueError("data.tokenizer_path is required when pack_sequences=True")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.data.tokenizer_path).eos_token_id


def _resolve_pad_id(tokenizer_path: str) -> int:
    """Pad id for the VLM collators — the same resolution the datasets use."""
    from kempnerforge.data.vlm_dataset import build_tokenizer, resolve_pad_id

    return int(resolve_pad_id(build_tokenizer(tokenizer_path)))


def _build_vlm_pipeline(config: JobConfig, dp_rank: int, dp_size: int) -> DataPipeline:
    """Image (Joint-Decoder) or video VLM loader, selected by ``config.is_video``."""
    tc = config.train
    vlm_cfg = config.vlm
    assert vlm_cfg is not None  # narrowed by is_vlm

    if config.is_video:
        # --- Video data path (a clip = ordered frames; same VLM wrapper) ---
        assert config.video is not None  # narrowed by is_video
        if not config.data.tokenizer_path:
            raise ValueError("Video training requires data.tokenizer_path")

        from kempnerforge.data.video_dataset import VideoCollator, build_video_dataset

        vcfg = config.video
        # Dataset style is registry-selected via [video].dataset_type; the
        # builder reads the rest of the knobs off vcfg.
        dataset: Any = build_video_dataset(vcfg, config.data.tokenizer_path, vlm_cfg.max_text_len)
        pad_id = _resolve_pad_id(config.data.tokenizer_path)
        collator: Any = VideoCollator(pad_id=pad_id, max_text_len=vlm_cfg.max_text_len)
        logger.info(f"Video dataset: {len(dataset):,} clips from {vcfg.data_root}")
    else:
        # --- Image VLM (Joint-Decoder) data path ---
        # Mixing VLM + text-only datasets in one run is out of scope on this
        # branch. DatasetSource doesn't describe image sources yet; follow-up.
        if not config.data.hf_dataset_name or not config.data.tokenizer_path:
            raise ValueError("VLM training requires data.hf_dataset_name and data.tokenizer_path")

        from kempnerforge.data.vlm_dataset import HuggingFaceVLMDataset, VLMCollator

        dataset = HuggingFaceVLMDataset(
            dataset_name=config.data.hf_dataset_name,
            split=config.data.hf_dataset_split,
            image_field=config.data.hf_dataset_image_field,
            text_field=config.data.hf_dataset_text_field,
            tokenizer_path=config.data.tokenizer_path,
            max_text_len=vlm_cfg.max_text_len,
            prompt_field=config.data.hf_dataset_prompt_field or None,
            image_size=config.data.hf_image_size,
            dataset_config=config.data.hf_dataset_config,
        )
        # Collator enforces fixed-length padding so all DP ranks see identical
        # tensor shapes, and emits the image_positions slot (D18) for
        # downstream multi-image work.
        collator = VLMCollator(
            pad_id=_resolve_pad_id(config.data.tokenizer_path),
            max_text_len=vlm_cfg.max_text_len,
        )
        logger.info(f"VLM dataset: {len(dataset):,} samples from {config.data.hf_dataset_name}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=True,
        seed=tc.effective_data_seed,
    )
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=tc.batch_size,
        sampler=sampler,
        config=config.data,
        collate_fn=collator,
    )
    return DataPipeline(dataloader=dataloader, dp_rank=dp_rank, dp_size=dp_size)


def _build_mixture_pipeline(
    config: JobConfig, dp_rank: int, dp_size: int, eos_token_id: int | None
) -> DataPipeline:
    """Weighted mixture over ``[[data.datasets]]`` sources."""
    tc = config.train

    sub_datasets = []
    names = []
    weights = []
    for src in config.data.datasets:
        if src.path:
            ds = MemoryMappedDataset(
                data_dir=src.path,
                seq_len=tc.seq_len + 1,
                file_pattern=config.data.file_pattern,
                pack_sequences=config.data.pack_sequences,
                eos_token_id=eos_token_id,
            )
        elif src.hf_name:
            if not config.data.tokenizer_path:
                raise ValueError(f"data.tokenizer_path required for HF dataset '{src.hf_name}'")
            ds = HuggingFaceDataset(
                dataset_name=src.hf_name,
                split=config.data.hf_dataset_split,
                text_field=config.data.hf_dataset_text_field,
                seq_len=tc.seq_len,
                tokenizer_path=config.data.tokenizer_path,
                dataset_config=src.hf_config or None,
                pack_sequences=config.data.pack_sequences,
            )
        else:
            continue
        sub_datasets.append(ds)
        names.append(src.name or src.path or src.hf_name)
        weights.append(src.weight)

    mixture_dataset = MixtureDataset(sub_datasets, names)
    sampler = MixtureSampler(
        cumulative_sizes=mixture_dataset.cumulative_sizes,
        weights=weights,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=True,
        seed=tc.effective_data_seed,
        temperature=config.data.mix_temperature,
    )
    dataloader = StatefulDataLoader(
        mixture_dataset,
        batch_size=tc.batch_size,
        sampler=sampler,
        config=config.data,
    )
    logger.info(
        f"Dataset: mixture of {len(sub_datasets)} sources, {len(mixture_dataset):,} total samples"
    )
    return DataPipeline(
        dataloader=dataloader,
        dp_rank=dp_rank,
        dp_size=dp_size,
        mixture_dataset=mixture_dataset,
        mixture_sampler=sampler,
        mixture_weights=dict(zip(names, weights, strict=True)),
    )


def _build_mmap_pipeline(
    config: JobConfig, dp_rank: int, dp_size: int, eos_token_id: int | None
) -> DataPipeline:
    """Pre-tokenized data on disk (fastest path)."""
    tc = config.train
    dataset = MemoryMappedDataset(
        data_dir=config.data.dataset_path,
        seq_len=tc.seq_len + 1,
        file_pattern=config.data.file_pattern,
        pack_sequences=config.data.pack_sequences,
        eos_token_id=eos_token_id,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=True,
        seed=tc.effective_data_seed,
    )
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=tc.batch_size,
        sampler=sampler,
        config=config.data,
    )
    logger.info(f"Dataset: {len(dataset):,} samples from {config.data.dataset_path}")
    return DataPipeline(dataloader=dataloader, dp_rank=dp_rank, dp_size=dp_size)


def _build_hf_pipeline(config: JobConfig, dp_rank: int, dp_size: int) -> DataPipeline:
    """HuggingFace text dataset, streamed or tokenized eagerly into memory."""
    tc = config.train
    if not config.data.tokenizer_path:
        raise ValueError("data.tokenizer_path is required for HuggingFace datasets")
    hf_dataset_name = config.data.hf_dataset_name
    if not hf_dataset_name:  # build_data_pipeline dispatches on this
        raise ValueError("data.hf_dataset_name is required for HuggingFace datasets")

    if config.data.hf_streaming:
        # Streaming: on-the-fly tokenization, no full download needed
        dataset = StreamingHuggingFaceDataset(
            dataset_name=hf_dataset_name,
            split=config.data.hf_dataset_split,
            text_field=config.data.hf_dataset_text_field,
            seq_len=tc.seq_len,
            tokenizer_path=config.data.tokenizer_path,
            dataset_config=config.data.hf_dataset_config,
            rank=dp_rank,
            world_size=dp_size,
            seed=tc.effective_data_seed,
            pack_sequences=config.data.pack_sequences,
        )
        dataloader: Any = TorchDataLoader(
            dataset,
            batch_size=tc.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            prefetch_factor=(config.data.prefetch_factor if config.data.num_workers > 0 else None),
        )
        logger.info(
            f"Dataset: streaming from {config.data.hf_dataset_name} "
            f"({config.data.hf_dataset_split}), rank={dp_rank}/{dp_size}"
        )
    else:
        # Eager: download, tokenize, and pack all sequences into memory
        dataset = HuggingFaceDataset(
            dataset_name=hf_dataset_name,
            split=config.data.hf_dataset_split,
            text_field=config.data.hf_dataset_text_field,
            seq_len=tc.seq_len,
            tokenizer_path=config.data.tokenizer_path,
            dataset_config=config.data.hf_dataset_config,
            pack_sequences=config.data.pack_sequences,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            seed=tc.effective_data_seed,
        )
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=tc.batch_size,
            sampler=sampler,
            config=config.data,
        )
        logger.info(
            f"Dataset: {len(dataset):,} packed sequences from "
            f"{config.data.hf_dataset_name} ({config.data.hf_dataset_split})"
        )
    return DataPipeline(dataloader=dataloader, dp_rank=dp_rank, dp_size=dp_size)


def build_data_pipeline(config: JobConfig, runtime: RuntimeContext) -> DataPipeline:
    """Build the training data pipeline for whichever ``[data]`` source is set.

    Returns an empty pipeline when no source is configured — the text and PP
    step bodies then run on random tokens.
    """
    eos_token_id = _resolve_eos_token_id(config)
    # With PP, samplers use DP rank/size (not total world size) since all PP
    # stages in the same DP group process the same batch.
    dp_rank, dp_size = get_dp_info(runtime.device_mesh)

    if config.is_vlm:
        return _build_vlm_pipeline(config, dp_rank, dp_size)
    if config.data.datasets:
        return _build_mixture_pipeline(config, dp_rank, dp_size, eos_token_id)
    if config.data.dataset_path:
        return _build_mmap_pipeline(config, dp_rank, dp_size, eos_token_id)
    if config.data.hf_dataset_name:
        return _build_hf_pipeline(config, dp_rank, dp_size)
    return DataPipeline(dp_rank=dp_rank, dp_size=dp_size)


def build_eval_dataloader(config: JobConfig, runtime: RuntimeContext) -> Any | None:
    """Build the eval dataloader, or None when eval is off or unsupported.

    VLM + eval is out of scope on this branch: ``run_eval`` calls
    ``model(input_ids)``, which does not match ``VLMWrapper.forward``.
    """
    tc = config.train
    eval_config = config.eval
    device = runtime.device
    dp_rank, dp_size = get_dp_info(runtime.device_mesh)

    build_eval, warn_vlm_eval = should_build_eval_dataloader(eval_config.enabled, config.is_vlm)
    if warn_vlm_eval:
        logger.warning(
            "eval.enabled=true is ignored for VLM configs on this branch. "
            "run_eval does not support VLMWrapper.forward yet; disabling "
            "eval for the duration of this run."
        )
    if not build_eval:
        return None

    if eval_config.dataset_path:
        eval_dataset: Any = MemoryMappedDataset(
            data_dir=eval_config.dataset_path,
            seq_len=tc.seq_len + 1,
            file_pattern=eval_config.file_pattern,
        )
        logger.info(f"Eval dataset: {len(eval_dataset):,} samples from {eval_config.dataset_path}")
    elif eval_config.hf_dataset_name:
        # Rank 0 loads/tokenizes the HF eval dataset, then broadcasts the
        # packed token tensor to all ranks via torch.distributed.broadcast.
        # This avoids file-lock failures (flock) on cluster filesystems
        # (Lustre, VAST) where load_dataset() would crash on all ranks.
        if runtime.rank == 0:
            eval_ds = HuggingFaceDataset(
                dataset_name=eval_config.hf_dataset_name,
                split=eval_config.hf_dataset_split,
                text_field=config.data.hf_dataset_text_field,
                seq_len=tc.seq_len,
                tokenizer_path=config.data.tokenizer_path,
                dataset_config=eval_config.hf_dataset_config,
            )
            packed = torch.from_numpy(np.stack(eval_ds._packed_sequences))
            n_seqs = torch.tensor([packed.shape[0]], device=device)
        else:
            packed = torch.empty(0, dtype=torch.long)
            n_seqs = torch.tensor([0], device=device)

        # Single-process callers (and unit tests) have no group to broadcast on.
        if dist.is_initialized():
            dist.broadcast(n_seqs, src=0)
            if runtime.rank != 0:
                packed = torch.empty(int(n_seqs.item()), tc.seq_len + 1, dtype=torch.long)
            packed_gpu = packed.to(device)
            dist.broadcast(packed_gpu, src=0)
            packed = packed_gpu.cpu()
            del packed_gpu

        eval_dataset = _EvalTensorDataset(packed)
        logger.info(
            f"Eval dataset: {len(eval_dataset):,} packed sequences from "
            f"{eval_config.hf_dataset_name} ({eval_config.hf_dataset_split})"
        )
    else:
        return None

    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=False,
        seed=tc.seed,
    )
    return TorchDataLoader(eval_dataset, batch_size=tc.batch_size, sampler=eval_sampler)


class _EvalTensorDataset(torch.utils.data.Dataset):
    """Map-style view over broadcast eval tokens."""

    def __init__(self, data: torch.Tensor) -> None:
        self._data = data

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self._data[idx]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def build_phase_state(config: JobConfig, data: DataPipeline, step: int) -> PhaseState:
    """Resolve data-annealing phases and re-derive the active one on resume."""
    phases: list[TrainingPhase] = []
    if config.data.phases:
        phases = sorted(config.data.phases, key=lambda p: p.start_step)
    elif config.data.anneal_start_step > 0 and config.data.anneal_weights:
        phases = [
            TrainingPhase(
                start_step=config.data.anneal_start_step,
                dataset_weights=dict(config.data.anneal_weights),
            )
        ]

    state = PhaseState(
        phases=phases,
        original_weights=dict(data.mixture_weights),
        temperature=config.data.mix_temperature,
    )

    mixture, sampler = data.mixture_dataset, data.mixture_sampler
    if step > 0 and phases and mixture is not None and sampler is not None:
        for i, phase in enumerate(phases):
            if step >= phase.start_step:
                _apply_phase(phase, state, mixture, sampler)
                state.next_idx = i + 1
        if state.next_idx > 0:
            logger.info(f"Resumed into phase {state.next_idx - 1}, lr_scale={state.lr_scale}")

    return state


def _apply_phase(
    phase: TrainingPhase,
    state: PhaseState,
    mixture: MixtureDataset,
    sampler: MixtureSampler,
) -> None:
    # A phase need not name every dataset; the rest keep their original weight.
    new_weights = [
        phase.dataset_weights.get(name, state.original_weights[name])
        for name in mixture.dataset_names
    ]
    sampler.update_weights(new_weights, temperature=state.temperature)
    state.lr_scale = phase.lr_scale


def advance_phases(state: PhaseState, data: DataPipeline, step: int) -> bool:
    """Activate every phase whose ``start_step`` has been reached.

    Returns True when at least one phase fired, so the caller can drop the
    materialized data iterator and pick up the new sampler weights.
    """
    mixture, sampler = data.mixture_dataset, data.mixture_sampler
    if not state.phases or mixture is None or sampler is None:
        return False

    fired = False
    while state.next_idx < len(state.phases) and step >= state.phases[state.next_idx].start_step:
        phase = state.phases[state.next_idx]
        _apply_phase(phase, state, mixture, sampler)
        logger.info(
            f"Phase transition at step {step}: phase={state.next_idx}, lr_scale={state.lr_scale}"
        )
        state.next_idx += 1
        fired = True
    return fired
