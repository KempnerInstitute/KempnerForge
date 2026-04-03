"""Rank-aware logging utilities for KempnerForge.

Provides a simple logging setup that:
- Only emits from rank 0 by default (avoids duplicate log lines)
- Supports color-coded output for key metrics
- Integrates with Python's standard logging module
"""

from __future__ import annotations

import logging
import os
import sys

# ANSI color codes
_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def _supports_color() -> bool:
    """Check if the terminal supports color output."""
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


class _RankFormatter(logging.Formatter):
    """Formatter that includes rank info and optional color."""

    def __init__(self, use_color: bool = True) -> None:
        self.use_color = use_color and _supports_color()
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        rank = os.environ.get("RANK", "0")
        level = record.levelname

        if self.use_color:
            c = _COLORS
            level_colors = {
                "DEBUG": c["dim"],
                "INFO": c["green"],
                "WARNING": c["yellow"],
                "ERROR": c["red"],
                "CRITICAL": c["bold"] + c["red"],
            }
            color = level_colors.get(level, "")
            prefix = f"{c['dim']}[rank {rank}]{c['reset']} {color}{level:<8}{c['reset']}"
        else:
            prefix = f"[rank {rank}] {level:<8}"

        return f"{prefix} {record.getMessage()}"


class _RankFilter(logging.Filter):
    """Only allow log records from the specified rank."""

    def __init__(self, rank: int = 0) -> None:
        super().__init__()
        self.allowed_rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        current_rank = int(os.environ.get("RANK", "0"))
        return current_rank == self.allowed_rank


_configured = False


def _configure_root(rank_zero_only: bool = True) -> None:
    """Configure the root kempnerforge logger (called once)."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("kempnerforge")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_RankFormatter())

    if rank_zero_only:
        handler.addFilter(_RankFilter(rank=0))

    root.addHandler(handler)


def get_logger(name: str, rank_zero_only: bool = True) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: Logger name (typically __name__).
        rank_zero_only: If True, only rank 0 emits logs.

    Returns:
        A configured logging.Logger instance.
    """
    _configure_root(rank_zero_only)
    return logging.getLogger(f"kempnerforge.{name}")


def _format_number(val: float | int) -> str:
    """Format a number for compact display.

    - Large ints: 125000 → '125k', 1500000 → '1.5M'
    - Small floats / scientific: 3e-4 → '3.00e-04'
    - Percentages (0-100 range with fractional part): '52.3'
    - Regular floats: '2.3400'
    """
    if isinstance(val, int) or (isinstance(val, float) and val == int(val) and abs(val) >= 1000):
        ival = int(val)
        abs_val = abs(ival)
        if abs_val >= 1_000_000_000:
            return f"{ival / 1e9:.1f}B"
        if abs_val >= 1_000_000:
            return f"{ival / 1e6:.1f}M"
        if abs_val >= 1_000:
            return f"{ival / 1e3:.0f}k" if abs_val % 1000 == 0 else f"{ival / 1e3:.1f}k"
        return str(ival)
    if isinstance(val, float):
        if abs(val) < 0.01 or abs(val) >= 1e6:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def format_metrics(step: int, metrics: dict[str, float | int | str]) -> str:
    """Format a metrics dict into a compact, color-coded log line.

    Example output:
        [step 1000] loss=2.34 | lr=3.00e-04 | grad_norm=1.2 | tok/s=125k | mfu=52.3% | mem=71.2/80GB
    """
    use_color = _supports_color()
    c = _COLORS if use_color else {k: "" for k in _COLORS}

    parts = []
    for key, val in metrics.items():
        formatted = _format_number(val) if isinstance(val, (int, float)) else str(val)
        parts.append(f"{c['cyan']}{key}{c['reset']}={c['bold']}{formatted}{c['reset']}")

    step_str = f"{c['dim']}[step {step}]{c['reset']}"
    return f"{step_str} {' | '.join(parts)}"
