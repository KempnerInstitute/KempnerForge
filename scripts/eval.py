#!/usr/bin/env python3
"""Standalone evaluation: load a checkpoint and compute loss/perplexity.

No training, no optimizer — just model + data + eval. Works on single GPU
or multi-GPU via torchrun (uses FSDP for large models).

Usage:
    # Single GPU — eval on pre-tokenized data
    uv run python scripts/eval.py configs/train/7b.toml \
        --checkpoint.load_path=checkpoints/7b/step_10000 \
        --eval.dataset_path=data/eval_set \
        --eval.steps=100

    # Single GPU — eval on HuggingFace dataset
    uv run python scripts/eval.py configs/train/7b.toml \
        --eval.hf_dataset_name=wikitext \
        --eval.hf_dataset_config=wikitext-103-raw-v1 \
        --eval.steps=100

    # Multi-GPU
    uv run torchrun --nproc_per_node=4 scripts/eval.py configs/train/7b.toml \
        --checkpoint.load_path=checkpoints/7b/step_10000 \
        --eval.dataset_path=data/eval_set
"""

from __future__ import annotations

import json
import sys

import torch
import torch.distributed as dist

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.loader import load_config
from kempnerforge.data.dataset import MemoryMappedDataset
from kempnerforge.data.sampler import DistributedSampler
from kempnerforge.distributed.parallel import build_parallel_model, default_mp_policy
from kempnerforge.distributed.setup import destroy_distributed, get_world_info, init_distributed
from kempnerforge.distributed.utils import get_dp_info
from kempnerforge.metrics.logger import get_logger
from kempnerforge.resilience.elastic import resolve_resume_path
from kempnerforge.training import build_loss_fn, run_eval

logger = get_logger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: eval.py <config.toml> [--section.key=value ...]")
        sys.exit(1)

    config_path = sys.argv[1]
    cli_args = sys.argv[2:]
    config = load_config(config_path, cli_args=cli_args)

    # Force eval enabled
    config.eval.enabled = True

    # --- Distributed setup ---
    rank, local_rank, world_size = get_world_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    device_mesh = init_distributed(config.distributed, seed=config.train.seed)
    config.validate(world_size)

    tc = config.train
    mp_policy = default_mp_policy(tc.param_dtype)

    # --- Loss function ---
    loss_fn = build_loss_fn(tc)

    # --- Model (non-PP only for standalone eval) ---
    model = build_parallel_model(
        config.model,
        device,
        device_mesh,
        ac_mode="none",
        mp_policy=mp_policy,
        param_dtype=tc.param_dtype,
        compile_model=False,
        fp8=False,
    )

    # --- Load checkpoint ---
    # Use a dummy optimizer for CheckpointManager (required by DCP but not used)
    dummy_optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    ckpt_mgr = CheckpointManager(config.checkpoint, model, dummy_optimizer)

    resume_path = resolve_resume_path(config.checkpoint.dir)
    load_path = config.checkpoint.load_path or (str(resume_path) if resume_path else None)

    if load_path is None:
        logger.error("No checkpoint found. Specify --checkpoint.load_path or --checkpoint.dir")
        sys.exit(1)

    step, tokens_seen = ckpt_mgr.load(
        path=load_path, exclude_keys=["optimizer"]
    )
    logger.info(f"Loaded checkpoint: step={step}, tokens_seen={tokens_seen:,}")

    # --- Eval data ---
    eval_config = config.eval
    dp_rank, dp_size = get_dp_info(device_mesh)

    if eval_config.dataset_path:
        eval_dataset = MemoryMappedDataset(
            data_dir=eval_config.dataset_path,
            seq_len=tc.seq_len + 1,
            file_pattern=eval_config.file_pattern,
        )
    elif eval_config.hf_dataset_name:
        import numpy as np

        from kempnerforge.data.dataset import HuggingFaceDataset

        if rank == 0:
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
            n_seqs = torch.tensor([0], device=device)

        if dist.is_initialized():
            dist.broadcast(n_seqs, src=0)
            if rank != 0:
                packed = torch.empty(n_seqs.item(), tc.seq_len + 1, dtype=torch.long)
            packed_gpu = packed.to(device)
            dist.broadcast(packed_gpu, src=0)
            packed = packed_gpu.cpu()
            del packed_gpu

        class _EvalTensorDataset(torch.utils.data.Dataset):
            def __init__(self, data: torch.Tensor) -> None:
                self._data = data

            def __len__(self) -> int:
                return self._data.shape[0]

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                tokens = self._data[idx]
                return {"input_ids": tokens[:-1], "labels": tokens[1:]}

        eval_dataset = _EvalTensorDataset(packed)
    else:
        logger.error(
            "No eval data configured. Set --eval.dataset_path or --eval.hf_dataset_name"
        )
        sys.exit(1)

    eval_sampler = DistributedSampler(
        eval_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=False, seed=tc.seed
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=tc.batch_size, sampler=eval_sampler
    )
    logger.info(f"Eval dataset: {len(eval_dataset):,} samples, {eval_config.steps} eval steps")

    # --- Run eval ---
    metrics = run_eval(model, eval_dataloader, loss_fn, device, eval_config.steps)

    if rank == 0:
        print(f"\n{'=' * 50}")
        print(f"Evaluation Results (step {step})")
        print(f"{'=' * 50}")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}")
        print(f"{'=' * 50}\n")

        # Write results to JSON
        results = {"step": step, "tokens_seen": tokens_seen, **metrics}
        print(json.dumps(results, indent=2))

    destroy_distributed()


if __name__ == "__main__":
    main()
