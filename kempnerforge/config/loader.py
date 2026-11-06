"""Config loading: TOML files → dataclass configs with CLI overrides.

Loading pipeline:
  1. Start with default JobConfig
  2. Import the modules listed in the `plugins` key (registry side effects)
  3. Load TOML file (if provided) and overlay
  4. Apply CLI overrides (--model.dim=512 style)
  5. Return JobConfig instance (call .validate(world_size) at distributed setup time)
"""

from __future__ import annotations

import ast
import importlib
import sys
import tomllib
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from kempnerforge.config.model import CORE_MLP_KEY_ACTIVATIONS, Activation, NormType
from kempnerforge.config.registry import registry
from kempnerforge.config.scheduler import SchedulerType
from kempnerforge.config.schema import JobConfig
from kempnerforge.config.vlm import VLMConfig

_REGISTRY_ENUMS: dict[type[Enum], str] = {
    SchedulerType: "scheduler",
    NormType: "norm",
    Activation: "mlp",
}


def _coerce_registry_enum(enum_type: type[Enum], category: str, value: Any) -> Any:
    """Coerce ``value`` to an ``enum_type`` member, or keep a name ``category`` registers.

    These enums select a registry builder, so a name a plugin registered is valid
    though the enum does not list it, and it stays a plain ``str``. ``Activation``
    reads ``mlp``: ``build_mlp`` maps the built-in activations to ``mlp`` keys and
    looks any other name up there directly. Core's own ``mlp`` keys pass through
    whatever was imported, so ``ModelConfig`` rejects each with the activation
    that selects it, and they are left out of the names the error lists.

    Raises:
        ValueError: ``value`` is neither a member nor a registered name; the
            message lists both.
    """
    try:
        return enum_type(value)
    except ValueError:
        core_keys = frozenset(CORE_MLP_KEY_ACTIVATIONS) if enum_type is Activation else frozenset()
        if isinstance(value, str) and value in core_keys:
            return value
        registered = [name for name in registry.list(category) if name not in core_keys]
        if isinstance(value, str) and value in registered:
            return value
        members = [member.value for member in enum_type]
        raise ValueError(
            f"{value!r} is neither in {enum_type.__name__} nor registered under "
            f"{category!r}. {enum_type.__name__} members: {members}. Registered "
            f"{category!r} names: {sorted(registered)}. Modules listed in 'plugins' "
            "register their names before this check."
        ) from None


def _coerce_value(field_type: Any, value: Any) -> Any:
    """Coerce a raw TOML/CLI value to the expected field type."""
    origin = get_origin(field_type)

    if origin is types.UnionType or origin is Union:
        args = get_args(field_type)
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            return None
        if len(non_none) == 1:
            return _coerce_value(non_none[0], value)
        return value

    if origin is Literal:
        allowed = get_args(field_type)
        if value not in allowed:
            raise ValueError(f"Value {value!r} not in allowed Literal values: {allowed}")
        return value

    if isinstance(field_type, type) and issubclass(field_type, Enum):
        if isinstance(value, field_type):
            return value
        if field_type in _REGISTRY_ENUMS:
            return _coerce_registry_enum(field_type, _REGISTRY_ENUMS[field_type], value)
        return field_type(value)

    if origin is tuple:
        if isinstance(value, (list, tuple)):
            args = get_args(field_type)
            if (
                len(args) == 2
                and args[1] is Ellipsis
                and isinstance(args[0], type)
                and is_dataclass(args[0])
            ):
                dc_type = args[0]
                return tuple(
                    _instantiate_from_dict(dc_type, item) if isinstance(item, dict) else item
                    for item in value
                )
            if all(isinstance(a, type) and is_dataclass(a) for a in args) and len(args) == len(
                value
            ):
                return tuple(
                    _instantiate_from_dict(dc_type, item) if isinstance(item, dict) else item
                    for dc_type, item in zip(args, value, strict=True)
                )
            return tuple(value)
        return value

    if origin is list:
        if isinstance(value, list):
            args = get_args(field_type)
            if args and isinstance(args[0], type) and is_dataclass(args[0]):
                dc_type = args[0]
                return [
                    _instantiate_from_dict(dc_type, item) if isinstance(item, dict) else item
                    for item in value
                ]
            return value
        return [value]

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


_type_hints_cache: dict[type, dict[str, Any]] = {}


def _get_type_hints(dc_type: type) -> dict[str, Any]:
    """Get resolved type hints for a dataclass, with caching."""
    if dc_type not in _type_hints_cache:
        _type_hints_cache[dc_type] = get_type_hints(dc_type)
    return _type_hints_cache[dc_type]


def _dataclass_type_from_hint(hint: Any) -> type | None:
    """Return the dataclass type inside ``hint`` if the hint is a dataclass
    or ``dataclass | None`` (Optional[dataclass]). Returns ``None`` otherwise.
    """
    origin = get_origin(hint)
    if origin is types.UnionType or origin is Union:
        for arg in get_args(hint):
            if arg is type(None):
                continue
            if isinstance(arg, type) and is_dataclass(arg):
                return arg
        return None
    if isinstance(hint, type) and is_dataclass(hint):
        return hint
    return None


def _instantiate_from_dict(dc_type: type, data: dict[str, Any]) -> Any:
    """Construct a dataclass directly from a dict, without first building a
    default instance.

    Used for nested Optional[dataclass] fields (e.g. ``JobConfig.vlm``)
    where the parent's default is ``None`` and the nested dataclass's
    ``__post_init__`` may reject an empty-default instance.

    When ``dc_type`` is ``VLMConfig`` (the discriminated-union base),
    dispatch on the ``arch`` field to the registered subclass before
    populating fields, so subclass-specific fields (e.g.
    ``cross_attention_every_n_layers``) are recognized.
    """
    if dc_type is VLMConfig:
        dc_type = _resolve_vlm_subclass(data)
    type_hints = _get_type_hints(dc_type)
    field_names = {f.name for f in fields(dc_type)}
    unknown = set(data.keys()) - field_names
    if unknown:
        raise ValueError(
            f"Unknown config keys in {dc_type.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(field_names)}"
        )
    kwargs: dict[str, Any] = {}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        raw = data[f.name]
        hint = type_hints[f.name]
        inner = _dataclass_type_from_hint(hint)
        if inner is not None and isinstance(raw, dict):
            kwargs[f.name] = _instantiate_from_dict(inner, raw)
        else:
            kwargs[f.name] = _coerce_value(hint, raw)
    return dc_type(**kwargs)


def _resolve_vlm_subclass(data: dict[str, Any]) -> type:
    """Resolve a TOML ``[vlm]`` table to the right ``VLMConfig`` subclass
    via the registered-arch lookup.

    An unregistered arch raises the same ``ValueError`` as
    ``VLMConfig.for_arch``, whichever site constructs the config.
    """
    arch = data.get("arch", "joint_decoder")
    try:
        return registry.get_vlm_config(arch)
    except KeyError as e:
        raise ValueError(
            f"Unknown vlm.arch: {arch!r}. Registered: {sorted(registry.list_vlm_configs())}."
        ) from e


def _check_vlm_arch_override(current: VLMConfig, arch: Any) -> None:
    """Reject an ``arch`` override whose registered config class is not ``current``'s.

    Applying a dict keeps the class the ``[vlm]`` section was built with, so such
    an override would build one arch's blocks behind another arch's strategy.
    """
    target = _resolve_vlm_subclass({"arch": arch})
    if target is not type(current):
        raise ValueError(
            f"vlm.arch={arch!r} cannot be applied as an override: the [vlm] section was "
            f"built as {type(current).__name__} (arch {type(current).declared_arch()!r}), and an "
            f'override keeps that class. Set arch = "{arch}" in the TOML\'s [vlm] '
            "section instead."
        )


def _apply_dict_to_dataclass(dc: Any, data: dict[str, Any]) -> Any:
    """Recursively apply a dict to a dataclass, creating a new instance.

    Raises ValueError on unknown keys to catch typos early, and on an ``arch``
    for a ``VLMConfig`` that needs a different config class than ``dc``'s.
    """
    if not is_dataclass(dc):
        return dc

    if isinstance(dc, VLMConfig) and "arch" in data:
        _check_vlm_arch_override(dc, data["arch"])

    dc_type = type(dc)
    type_hints = _get_type_hints(dc_type)
    field_names = {f.name for f in fields(dc)}

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
            elif current_val is None and isinstance(raw, dict):
                inner = _dataclass_type_from_hint(type_hints[f.name])
                if inner is not None:
                    kwargs[f.name] = _instantiate_from_dict(inner, raw)
                else:
                    kwargs[f.name] = _coerce_value(type_hints[f.name], raw)
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
            key = arg[2:]
            value_str = "true"
        else:
            key, value_str = arg[2:].split("=", 1)

        value: Any
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str

        parts = key.split(".")
        d = overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    return overrides


def _plugin_module_names(raw: Any) -> list[str]:
    """Normalize a raw ``plugins`` value to a list of module paths."""
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError(
            "Config key 'plugins' must be a list of module paths "
            f'(e.g. plugins = ["my_experiment.components"]); got {raw!r}'
        )
    return names


def _plugin_not_found(error: Exception, name: str) -> bool:
    """Whether ``error`` says the plugin module ``name``, or a package above it, was not found."""
    if not isinstance(error, ModuleNotFoundError) or error.name is None:
        return False
    parts = name.split(".")
    return error.name in {".".join(parts[: i + 1]) for i in range(len(parts))}


def _import_plugins(raw: Any) -> None:
    """Import config-declared modules for their registry side effects.

    ``import_module`` is idempotent, so repeated ``load_config`` calls in one
    process never re-run a module body. A module that fails to import raises a
    ``ValueError`` naming it. The ``sys.path`` hint is added only when the module
    itself, or a package above it, was not found; any other failure, a missing
    dependency of the module included, means the module was found.
    """
    for name in _plugin_module_names(raw):
        try:
            importlib.import_module(name)
        except Exception as e:
            hint = (
                " Plugin modules must be on sys.path (the launch directory is not): add "
                "their directory to PYTHONPATH, or install them."
                if _plugin_not_found(e, name)
                else ""
            )
            raise ValueError(
                f"Failed to import plugin module {name!r} from config key 'plugins': "
                f"{type(e).__name__}: {e}.{hint}"
            ) from e


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

    Modules named by the top-level ``plugins`` key are imported before the
    overlay, so components they register are visible to those validations.

    Args:
        config_path: Path to a TOML config file (or None for defaults).
        cli_args: CLI arguments to parse (defaults to sys.argv[1:]).

    Returns:
        A JobConfig with layered defaults → TOML → CLI overrides.
    """
    config = JobConfig()

    toml_data: dict[str, Any] = load_toml(config_path) if config_path is not None else {}

    if cli_args is None:
        cli_args = sys.argv[1:]
    overrides = _parse_cli_overrides(cli_args)

    plugins = overrides.get("plugins", toml_data.get("plugins"))
    if plugins is not None:
        _import_plugins(plugins)

    if config_path is not None:
        config = _apply_dict_to_dataclass(config, toml_data)

    if overrides:
        config = _apply_dict_to_dataclass(config, overrides)

    return config
