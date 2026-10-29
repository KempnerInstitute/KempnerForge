"""Evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass

from kempnerforge.config.data import check_token_dtype


@dataclass
class EvalConfig:
    """Evaluation pipeline settings (disabled by default).

    ``token_dtype`` is the integer type of the raw ``.bin`` eval shards, as
    ``DataConfig.token_dtype`` is for the training shards.
    """

    enabled: bool = False
    interval: int = 1000  # Eval every N training steps
    steps: int = 50  # Number of eval batches per evaluation
    # Pre-tokenized eval data (same format as training)
    dataset_path: str = ""
    file_pattern: str = "*.npy"
    token_dtype: str = ""
    # HuggingFace eval data
    hf_dataset_name: str | None = None
    hf_dataset_config: str | None = None
    hf_dataset_split: str = "validation"

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("eval interval must be positive")
        if self.steps <= 0:
            raise ValueError("eval steps must be positive")
        check_token_dtype(self.token_dtype, "eval.token_dtype")
