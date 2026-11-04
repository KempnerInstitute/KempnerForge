"""Metrics configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass
class MetricsConfig:
    """Logging and metrics settings."""

    log_interval: int = 10
    enable_wandb: bool = False
    enable_tensorboard: bool = False
    enable_mlflow: bool = False
    wandb_project: str = "kempnerforge"
    wandb_run_name: str | None = None
    wandb_run_id: str = ""
    tensorboard_dir: str = "tb_logs"

    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str | None = None
    mlflow_run_name: str | None = None
    mlflow_description: str | None = None
    mlflow_tags: dict[str, str] = field(default_factory=dict)
    mlflow_run_id: str = ""
    mlflow_log_system_metrics: bool = True
    mlflow_system_metrics_interval: float = 10.0
    mlflow_register_checkpoints: bool = False

    def resolve_mlflow_tracking_uri(self) -> str:
        """The tracking URI in force: the field, else $MLFLOW_TRACKING_URI, else Databricks."""
        if self.mlflow_tracking_uri is not None:
            return self.mlflow_tracking_uri
        return os.environ.get("MLFLOW_TRACKING_URI") or "databricks"

    def __post_init__(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.mlflow_description is not None and not _is_text(self.mlflow_description):
            raise ValueError(
                "mlflow_description must be a non-empty string or unset; "
                f"got {self.mlflow_description!r}"
            )
        if not isinstance(self.mlflow_tags, dict):
            raise ValueError(
                "mlflow_tags must be a table of tag names to string values "
                f'(e.g. mlflow_tags = {{ team = "vlm" }}); got {self.mlflow_tags!r}'
            )
        for key, value in self.mlflow_tags.items():
            if not _is_text(key):
                raise ValueError(f"mlflow_tags keys must be non-empty strings; got {key!r}")
            if not _is_text(value):
                raise ValueError(f"mlflow_tags[{key!r}] must be a non-empty string; got {value!r}")
        if (
            self.enable_mlflow
            and self.mlflow_experiment is not None
            and not self.mlflow_experiment.startswith("/")
            and self.resolve_mlflow_tracking_uri().startswith("databricks")
        ):
            raise ValueError(
                "mlflow_experiment must be an absolute workspace path on Databricks "
                f"(e.g. '/Users/you@example.com/proj'); got {self.mlflow_experiment!r}"
            )
        if self.mlflow_register_checkpoints and not self.enable_mlflow:
            raise ValueError(
                "mlflow_register_checkpoints requires enable_mlflow = true "
                "(enable MLflow, or leave checkpoint registration off)"
            )
