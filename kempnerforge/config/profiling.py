"""Profiling configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProfilingConfig:
    """Performance profiling settings."""

    enable: bool = False
    start_step: int = 5
    end_step: int = 8
    trace_dir: str = "profiler_traces"

    def __post_init__(self) -> None:
        if self.end_step <= self.start_step:
            raise ValueError("end_step must be greater than start_step")
