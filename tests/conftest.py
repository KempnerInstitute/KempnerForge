"""Shared test fixtures for KempnerForge."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kempnerforge.config.schema import (
    CheckpointConfig,
    JobConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
)

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

TINY_MODEL_CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)

SMALL_MODEL_CONFIG = ModelConfig(dim=128, n_layers=4, n_heads=4, vocab_size=512, max_seq_len=128)


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)


@pytest.fixture
def small_model_config() -> ModelConfig:
    return ModelConfig(dim=128, n_layers=4, n_heads=4, vocab_size=512, max_seq_len=128)


# ---------------------------------------------------------------------------
# Job configs
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_job_config(tmp_path) -> JobConfig:
    """Minimal job config for fast testing."""
    return JobConfig(
        model=ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64),
        train=TrainConfig(
            batch_size=2,
            seq_len=32,
            max_steps=10,
            grad_accum_steps=1,
            compile_model=False,
        ),
        optimizer=OptimizerConfig(lr=1e-3, fused=False),
        scheduler=SchedulerConfig(warmup_steps=2),
        checkpoint=CheckpointConfig(dir=str(tmp_path / "checkpoints"), interval=5),
    )


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def random_batch(tiny_model_config, device) -> dict[str, torch.Tensor]:
    """A random batch of input_ids and labels."""
    batch_size, seq_len = 2, 32
    return {
        "input_ids": torch.randint(
            0, tiny_model_config.vocab_size, (batch_size, seq_len), device=device
        ),
        "labels": torch.randint(
            0, tiny_model_config.vocab_size, (batch_size, seq_len), device=device
        ),
    }


@pytest.fixture
def mmap_data_dir(tmp_path) -> str:
    """Create a temp directory with small .npy token files."""
    tokens = np.arange(1000, dtype=np.uint16)
    for i in range(2):
        np.save(tmp_path / f"shard_{i}.npy", tokens[i * 500 : (i + 1) * 500])
    return str(tmp_path)
