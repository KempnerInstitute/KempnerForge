"""Config loading: TOML files → dataclass configs with CLI overrides.

Loading pipeline:
  1. Start with default JobConfig
  2. Load TOML file (if provided) and overlay
  3. Apply CLI overrides (--model.dim=512 style)
  4. Return JobConfig instance (call .validate(world_size) at distributed setup time)
"""

from __future__ import annotations

import ast
import sys
import tomllib
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from kempnerforge.config.schema import JobConfig


def _coerce_value(field_type: Any, value: Any) -> Any:
    """Coerce a raw TOML/CLI value to the expected field type."""
    origin = get_origin(field_type)

    # Handle Union types: int | None (UnionType) and Optional[int] (typing.Union)
    if origin is types.UnionType or origin is Union:
        args = get_args(field_type)
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            return None
        if len(non_none) == 1:
            return _coerce_value(non_none[0], value)
        return value

    # Handle Literal["float32", "bfloat16"] — accept value as-is
    if origin is Literal:
        allowed = get_args(field_type)
        if value not in allowed:
            raise ValueError(f"Value {value!r} not in allowed Literal values: {allowed}")
        return value

    # Handle Enum
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        if isinstance(value, field_type):
            return value
        return field_type(value)

    # Handle tuple (e.g., betas: tuple[float, float])
    if origin is tuple:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return value

    # Handle list
    if origin is list:
        if isinstance(value, list):
            args = get_args(field_type)
            if args and is_dataclass(args[0]):
                # list[SomeDataclass] — convert each dict element to a dataclass instance
                dc_type = args[0]
                return [
                    _apply_dict_to_dataclass(dc_type(), item) if isinstance(item, dict) else item  # type: ignore[reportCallIssue]
                    for item in value
                ]
            return value
        return [value]

    # Primitive coercion
    if isinstance(field_type, type):
        if field_type is bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        if field_type is int and isinstance(value, (str, float)):
            return int(value)
        if field_type is float and isinstance(value, (str, int)):
            return float(value)

    return value


# Cache resolved type hints per dataclass type to avoid repeated evaluation.
_type_hints_cache: dict[type, dict[str, Any]] = {}


def _get_type_hints(dc_type: type) -> dict[str, Any]:
    """Get resolved type hints for a dataclass, with caching."""
    if dc_type not in _type_hints_cache:
        _type_hints_cache[dc_type] = get_type_hints(dc_type)
    return _type_hints_cache[dc_type]


def _apply_dict_to_dataclass(dc: Any, data: dict[str, Any]) -> Any:
    """Recursively apply a dict to a dataclass, creating a new instance.

    Raises ValueError on unknown keys to catch typos early.
    """
    if not is_dataclass(dc):
        return dc

    dc_type = type(dc)
    type_hints = _get_type_hints(dc_type)
    field_names = {f.name for f in fields(dc)}

    # Reject unknown keys (catches typos like 'dimm' or 'modell')
    unknown = set(data.keys()) - field_names
    if unknown:
        raise ValueError(
            f"Unknown config keys in {dc_type.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(field_names)}"
        )

    kwargs = {}
    for f in fields(dc):
        current_val = getattr(dc, f.name)
        if f.name in data:
            raw = data[f.name]
            if is_dataclass(current_val) and isinstance(raw, dict):
                kwargs[f.name] = _apply_dict_to_dataclass(current_val, raw)
            else:
                kwargs[f.name] = _coerce_value(type_hints[f.name], raw)
        else:
            kwargs[f.name] = current_val

    return dc_type(**kwargs)


def _parse_cli_overrides(args: list[str]) -> dict[str, Any]:
    """Parse --section.key=value CLI arguments into a nested dict.

    Supports:
      --model.dim=512
      --train.compile_model=true
      --optimizer.betas=[0.9,0.95]
    """
    overrides: dict[str, Any] = {}

    for arg in args:
        if not arg.startswith("--"):
            continue
        if "=" not in arg:
            # Boolean flag: --train.compile_model means True
            key = arg[2:]
            value_str = "true"
        else:
            key, value_str = arg[2:].split("=", 1)

        # Parse the value
        value: Any
        # Try to parse as Python literal (handles ints, floats, bools, lists)
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str  # Keep as string

        # Build nested dict from dotted key
        parts = key.split(".")
        d = overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    return overrides


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(
    config_path: str | Path | None = None,
    cli_args: list[str] | None = None,
) -> JobConfig:
    """Load a JobConfig from optional TOML file + CLI overrides.

    The returned config has all sub-config __post_init__ validations applied.
    Cross-config validation (e.g., parallelism vs world_size) requires calling
    config.validate(world_size=...) separately at distributed setup time.

    Args:
        config_path: Path to a TOML config file (or None for defaults).
        cli_args: CLI arguments to parse (defaults to sys.argv[1:]).

    Returns:
        A JobConfig with layered defaults → TOML → CLI overrides.
    """
    config = JobConfig()

    # Layer 1: TOML file
    if config_path is not None:
        toml_data = load_toml(config_path)
        config = _apply_dict_to_dataclass(config, toml_data)

    # Layer 2: CLI overrides
    if cli_args is None:
        cli_args = sys.argv[1:]
    overrides = _parse_cli_overrides(cli_args)
    if overrides:
        config = _apply_dict_to_dataclass(config, overrides)

    return config
