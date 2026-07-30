"""Metrics configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricsConfig:
    """Logging and metrics settings."""

    log_interval: int = 10  # Log every N steps
    enable_wandb: bool = False
    enable_tensorboard: bool = False
    enable_mlflow: bool = False
    wandb_project: str = "kempnerforge"
    wandb_run_name: str | None = None  # None -> auto-generated
    wandb_run_id: str = ""  # Restored from checkpoint on resume; empty = new run
    tensorboard_dir: str = "tb_logs"

    # MLflow (Databricks-hosted or local); credentials come from the env, not config.
    mlflow_tracking_uri: str = "databricks"  # or an http(s):// MLflow server
    mlflow_experiment: str | None = None  # absolute workspace path on Databricks; None -> env/auto
    mlflow_run_name: str | None = None  # None -> auto-generated
    mlflow_run_id: str = ""  # Restored from checkpoint on resume; empty = new run
    mlflow_log_system_metrics: bool = True  # CPU/GPU/memory sampled on a background thread
    mlflow_system_metrics_interval: float = 10.0  # seconds between samples

    def __post_init__(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        # Databricks requires an absolute workspace path; fail fast at config load.
        if (
            self.enable_mlflow
            and self.mlflow_tracking_uri.startswith("databricks")
            and self.mlflow_experiment is not None
            and not self.mlflow_experiment.startswith("/")
        ):
            raise ValueError(
                "mlflow_experiment must be an absolute workspace path on Databricks "
                f"(e.g. '/Users/you@example.com/proj'); got {self.mlflow_experiment!r}"
            )
