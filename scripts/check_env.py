#!/usr/bin/env python3
"""Centralized environment preflight for KempnerForge Claude Code skills.

Every skill delegates here for system-dependency and cluster-config checks.
Baseline checks always run; additional checks are requested via ``--requires``.

Exit codes:
    0: all checks OK or WARN.
    1: any check reported MISS (baseline or tag).
    2: script error (invalid args, unreadable config, etc.).

Usage:
    uv run python scripts/check_env.py
    uv run python scripts/check_env.py --requires gpu,slurm
    uv run python scripts/check_env.py --requires gpu,slurm,multi-node --json
    uv run python scripts/check_env.py --init
"""

from __future__ import annotations

import argparse
import configparser
import io
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


OK = "OK"
WARN = "WARN"
MISS = "MISS"

KNOWN_TAGS = ("gpu", "slurm", "multi-node", "wandb", "mlflow", "hf", "gh")
BASELINE_ALIASES = ("uv", "repo-layout")

DEFAULT_CLUSTER_CONFIG = "configs/cluster/local.toml"
_DISTRIBUTED_FS_TYPES = {"nfs", "nfs4", "lustre", "gpfs", "beegfs"}


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "status": self.status, "message": self.message}
        if self.fix:
            d["fix"] = self.fix
        return d


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (cwd by default) to find the KempnerForge root."""
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists() and (p / "kempnerforge").exists():
            return p
    return cur


def load_cluster_config(repo_root: Path, override: str | None = None) -> dict | None:
    """Load configs/cluster/local.toml (or ``override``). None if missing."""
    path = Path(override) if override else repo_root / DEFAULT_CLUSTER_CONFIG
    if not path.exists():
        return None
    if tomllib is None:
        raise RuntimeError("Python 3.11+ required (tomllib unavailable)")
    with path.open("rb") as f:
        return tomllib.load(f)


def check_uv() -> CheckResult:
    """Check that the uv package manager is on PATH and runnable."""
    if shutil.which("uv") is None:
        return CheckResult(
            "uv",
            MISS,
            "uv not on PATH",
            "Install uv: https://docs.astral.sh/uv/getting-started/installation/",
        )
    try:
        out = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, timeout=5, check=False
        )
    except (subprocess.SubprocessError, OSError) as e:
        return CheckResult("uv", MISS, f"uv --version failed: {e}", "Reinstall uv")
    if out.returncode != 0:
        return CheckResult("uv", MISS, f"uv --version exit {out.returncode}", "Reinstall uv")
    return CheckResult("uv", OK, f"{out.stdout.strip() or 'uv'} on PATH")


def check_repo_layout(repo_root: Path) -> CheckResult:
    """Check expected top-level files and directories exist."""
    missing = [
        n for n in ("pyproject.toml", "configs", "kempnerforge") if not (repo_root / n).exists()
    ]
    if missing:
        return CheckResult(
            "repo-layout",
            MISS,
            f"missing: {', '.join(missing)} (cwd={repo_root})",
            "Run check_env from the KempnerForge repo root",
        )
    return CheckResult(
        "repo-layout",
        OK,
        "pyproject.toml + configs/ + kempnerforge/ present",
    )


def check_gpu() -> CheckResult:
    """Check for an NVIDIA GPU. nvidia-smi preferred; /dev/nvidia* as fallback."""
    if shutil.which("nvidia-smi") is not None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (subprocess.SubprocessError, OSError):
            out = None
        if out is not None and out.returncode == 0 and out.stdout.strip():
            lines = out.stdout.strip().splitlines()
            count = len(lines)
            parts = [p.strip() for p in lines[0].split(",")]
            name = parts[0] if parts else "?"
            driver = parts[1] if len(parts) > 1 else "?"
            return CheckResult("gpu", OK, f"{count}x {name}, driver {driver}")
    if Path("/dev/nvidia0").exists() or Path("/proc/driver/nvidia/gpus").exists():
        return CheckResult("gpu", OK, "NVIDIA device nodes present (nvidia-smi unavailable)")
    return CheckResult(
        "gpu",
        MISS,
        "no NVIDIA GPU detected",
        "This skill requires CUDA. On CPU-only machines, try /kempnerforge:explain-architecture.",
    )


def check_slurm(config: dict | None) -> CheckResult:
    """Check sbatch is available and optional account/QoS are recorded.

    Whether ``account`` and ``qos`` are strictly required varies by cluster.
    Strictness is controlled by ``[slurm].require_account`` (default ``true``)
    and ``[slurm].require_qos`` (default ``false``) in local.toml. When a
    required field is missing, the status is MISS; otherwise missing fields
    are reported as WARN so the skill can still proceed.
    """
    if shutil.which("sbatch") is None:
        return CheckResult(
            "slurm",
            MISS,
            "sbatch not on PATH",
            "SLURM not installed on this node. On a login node, check your modules.",
        )
    slurm_cfg = (config or {}).get("slurm", {}) or {}
    account = (
        slurm_cfg.get("account")
        or os.environ.get("SLURM_ACCOUNT")
        or os.environ.get("SBATCH_ACCOUNT")
    )
    qos = slurm_cfg.get("qos") or os.environ.get("SLURM_QOS")
    require_account = bool(slurm_cfg.get("require_account", True))
    require_qos = bool(slurm_cfg.get("require_qos", False))

    parts = ["sbatch on PATH"]
    if account:
        parts.append(f"account={account}")
    if qos:
        parts.append(f"qos={qos}")
    message = ", ".join(parts)

    if require_account and not account:
        return CheckResult(
            "slurm",
            MISS,
            f"{message} (no account configured; cluster requires one)",
            "Run /kempnerforge:cluster-config to record your SLURM account "
            "(or set [slurm].require_account=false for clusters that don't need --account)",
        )
    if require_qos and not qos:
        return CheckResult(
            "slurm",
            MISS,
            f"{message} (no qos configured; cluster requires one)",
            "Set [slurm].qos in local.toml "
            "(or set [slurm].require_qos=false for clusters that don't need --qos)",
        )
    if not account and not qos:
        return CheckResult(
            "slurm",
            WARN,
            f"{message} (no account or qos configured; some clusters need one or both)",
        )
    return CheckResult("slurm", OK, message)


def _fs_type(path: Path) -> str | None:
    """Return filesystem type for ``path`` via ``stat -f`` (Linux). None on failure."""
    try:
        out = subprocess.run(
            ["stat", "-f", "--format=%T", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode == 0 and out.stdout.strip():
        return out.stdout.strip()
    return None


def check_multi_node(config: dict | None, repo_root: Path) -> CheckResult:
    """Verify repo lives on a distributed FS and InfiniBand is detected."""
    if not config or "network" not in config:
        return CheckResult(
            "multi-node",
            MISS,
            "no [network] section in local.toml",
            "Run /kempnerforge:cluster-config to set shared_fs_root",
        )
    shared_raw = config["network"].get("shared_fs_root")
    if not shared_raw:
        return CheckResult(
            "multi-node",
            MISS,
            "network.shared_fs_root missing in local.toml",
            "Set shared_fs_root to the distributed FS mount in local.toml",
        )
    shared = Path(shared_raw).resolve()
    if not shared.exists():
        return CheckResult(
            "multi-node",
            MISS,
            f"shared_fs_root {shared} does not exist",
            "Correct shared_fs_root in local.toml",
        )
    try:
        repo_root.resolve().relative_to(shared)
    except ValueError:
        return CheckResult(
            "multi-node",
            MISS,
            f"repo {repo_root} is not under shared_fs_root {shared}",
            "Clone the repo onto the shared filesystem",
        )
    ckpt_raw = (config.get("paths") or {}).get("checkpoints_root")
    if ckpt_raw:
        ckpt = Path(ckpt_raw).resolve()
        try:
            ckpt.relative_to(shared)
        except ValueError:
            return CheckResult(
                "multi-node",
                WARN,
                f"checkpoints_root {ckpt} not under shared_fs_root {shared}",
            )
    fs = _fs_type(shared)
    fs_ok = bool(fs and fs.lower() in _DISTRIBUTED_FS_TYPES)
    ib_dir = Path("/sys/class/infiniband")
    try:
        ib_ok = ib_dir.exists() and any(ib_dir.iterdir())
    except PermissionError:
        ib_ok = False
    if fs_ok and ib_ok:
        return CheckResult("multi-node", OK, f"shared FS ({fs}) + IB detected")
    if fs_ok:
        return CheckResult("multi-node", WARN, f"shared FS ({fs}); IB not detected (TCP fallback)")
    if ib_ok:
        return CheckResult(
            "multi-node",
            WARN,
            f"IB detected; shared FS type unverified ({fs or 'unknown'})",
        )
    return CheckResult("multi-node", WARN, f"IB and FS type unverified ({fs or 'unknown'})")


def check_wandb(check_credentials: bool = False) -> CheckResult:
    """Check WANDB_API_KEY env var; optionally validate via API."""
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        return CheckResult(
            "wandb",
            MISS,
            "WANDB_API_KEY not set",
            "export WANDB_API_KEY=... (needed for metrics.backend=wandb)",
        )
    if check_credentials:
        try:
            import wandb  # type: ignore

            api = wandb.Api()
            user = api.viewer.username
        except Exception as e:
            return CheckResult("wandb", WARN, f"WANDB_API_KEY set but API probe failed: {e}")
        return CheckResult("wandb", OK, f"WANDB_API_KEY valid, user={user}")
    return CheckResult("wandb", OK, "WANDB_API_KEY set (not validated)")


_MLFLOW_AUTH_ROUTES = (
    "routes: (1) env DATABRICKS_HOST + DATABRICKS_TOKEN; (2) a profile in $DATABRICKS_CONFIG_FILE, "
    "else ~/.databrickscfg, picked by a databricks://<profile> $MLFLOW_TRACKING_URI or "
    "$DATABRICKS_CONFIG_PROFILE, else the SDK default profile ([__settings__] default_profile, "
    "else DEFAULT) and read only when no credential variable is set; (3) SDK OAuth, which a "
    "preflight cannot verify offline"
)
_SETTINGS_SECTION = "__settings__"

_DATABRICKS_CREDENTIAL_VAR_AUTH: dict[str, str] = {
    "DATABRICKS_TOKEN": "pat",
    "DATABRICKS_API_TOKEN": "pat",
    "DATABRICKS_USERNAME": "basic",
    "DATABRICKS_PASSWORD": "basic",
    "DATABRICKS_CLIENT_ID": "oauth",
    "DATABRICKS_CLIENT_SECRET": "oauth",
    "DATABRICKS_OIDC_TOKEN_ENV": "env-oidc",
    "DATABRICKS_OIDC_TOKEN_FILEPATH": "file-oidc",
    "DATABRICKS_OIDC_TOKEN_FILE": "file-oidc",
    "DATABRICKS_GOOGLE_SERVICE_ACCOUNT": "google",
    "GOOGLE_CREDENTIALS": "google",
    "DATABRICKS_AZURE_RESOURCE_ID": "azure",
    "ARM_USE_MSI": "azure",
    "ARM_CLIENT_ID": "azure",
    "ARM_CLIENT_SECRET": "azure",
    "ARM_TENANT_ID": "azure",
    "DATABRICKS_METADATA_SERVICE_URL": "metadata-service",
}

_DATABRICKS_ENV_AUTH: dict[str, tuple[str, tuple[tuple[str, ...], ...]]] = {
    "pat": ("token", (("DATABRICKS_TOKEN", "DATABRICKS_API_TOKEN"),)),
    "basic": ("username and password", (("DATABRICKS_USERNAME",), ("DATABRICKS_PASSWORD",))),
    "oauth-m2m": (
        "client id and secret",
        (("DATABRICKS_CLIENT_ID",), ("DATABRICKS_CLIENT_SECRET",)),
    ),
}


def _selected_databricks_profile() -> str | None:
    """The Databricks profile explicitly selected, or None when none is.

    A databricks://<profile> $MLFLOW_TRACKING_URI, then $DATABRICKS_CONFIG_PROFILE.
    """
    try:
        uri = urlparse(os.environ.get("MLFLOW_TRACKING_URI", ""))
    except ValueError:
        uri = urlparse("")
    if uri.scheme == "databricks" and uri.netloc and not {":", "@"} & set(uri.netloc):
        return uri.netloc
    return os.environ.get("DATABRICKS_CONFIG_PROFILE") or None


def _databricks_config_file() -> str:
    """The profile file the SDK reads, as written and as a message names it.

    $DATABRICKS_CONFIG_FILE when it is set and not empty, else ~/.databrickscfg.
    """
    return os.environ.get("DATABRICKS_CONFIG_FILE") or "~/.databrickscfg"


@dataclass(frozen=True)
class _ProfileFile:
    """What a Databricks profile file offers, as far as a message may say.

    Attributes:
        profiles: The profile names it offers.
        default_profile: The profile the SDK falls back to, None when the stripped
            ``[__settings__] default_profile`` still spans more than one line.
        default_profile_quotable: False when that value began on a continuation line, whose
            text can be a token, so no message names it.
        problem: Why the file offers no profile, or None.
        missing: The file does not exist, so the SDK reads nothing from it.
    """

    profiles: frozenset[str]
    default_profile: str | None
    default_profile_quotable: bool
    problem: str | None
    missing: bool = False


def _read_databricks_config(name: str) -> _ProfileFile:
    """Read the Databricks profile file ``name`` the way the SDK reads it.

    The path is expanded and tested for existence as the SDK does it, so a broken or looping
    symlink, or a path through a regular file, is a missing file. It is parsed with ConfigParser
    in the encoding ``ConfigParser.read`` uses, as the SDK parses it, so an indented line
    continues the value above it rather than opening a section, a repeated ``[DEFAULT]`` merges,
    and a file the SDK rejects is reported rather than passed. A failure is named by exception
    class alone, because its text can quote the file, which can hold a token. ``[DEFAULT]`` is a
    profile only when it carries keys, and the reserved ``[__settings__]`` section is never one.
    The default profile is stripped before it is resolved, as the SDK strips it.
    """
    try:
        path = Path(name).expanduser()
        exists = path.exists()
    except (OSError, RuntimeError) as e:
        return _ProfileFile(
            frozenset(), "DEFAULT", True, f"{name} cannot be resolved ({type(e).__name__})"
        )
    if not exists:
        return _ProfileFile(frozenset(), "DEFAULT", True, f"{name} not found", missing=True)
    parser = configparser.ConfigParser()
    try:
        with path.open(encoding=io.text_encoding(None)) as f:
            parser.read_file(f)
    except OSError as e:
        return _ProfileFile(
            frozenset(), "DEFAULT", True, f"{name} unreadable: {e.strerror or type(e).__name__}"
        )
    except (configparser.Error, UnicodeDecodeError) as e:
        return _ProfileFile(
            frozenset(), "DEFAULT", True, f"{name} is malformed ({type(e).__name__})"
        )
    profiles = {section for section in parser.sections() if section != _SETTINGS_SECTION}
    if parser.defaults():
        profiles.add("DEFAULT")
    parser.remove_option(parser.default_section, "default_profile")
    raw = parser.get(_SETTINGS_SECTION, "default_profile", raw=True, fallback="")
    default = raw.strip()
    if "\n" in default:
        return _ProfileFile(frozenset(profiles), None, False, None)
    return _ProfileFile(frozenset(profiles), default or "DEFAULT", "\n" not in raw, None)


def _databricks_env_credentials() -> tuple[str, ...]:
    """The Databricks credential variables the environment carries.

    Any one of them makes the SDK skip the profile file when no profile is selected.
    """
    names = ("DATABRICKS_HOST", *_DATABRICKS_CREDENTIAL_VAR_AUTH)
    return tuple(name for name in names if os.environ.get(name))


def _databricks_env_auth() -> tuple[str | None, str]:
    """What the SDK authenticates with from the environment alone, else None and why not.

    Only the SDK strategies whose requirements are all environment variables are settled here
    (pat, basic and oauth-m2m); a CLI login, a metadata service and cloud credentials need state
    a preflight cannot see. Credentials from more than one auth family are the SDK's own error
    unless $DATABRICKS_AUTH_TYPE picks one.
    """
    if not os.environ.get("DATABRICKS_HOST"):
        return None, "DATABRICKS_HOST not set"
    complete = {
        auth_type: label
        for auth_type, (label, groups) in _DATABRICKS_ENV_AUTH.items()
        if all(any(os.environ.get(name) for name in group) for group in groups)
    }
    if pinned := os.environ.get("DATABRICKS_AUTH_TYPE"):
        if label := complete.get(pinned):
            return label, ""
        return None, f"$DATABRICKS_AUTH_TYPE={pinned} is not completed by the variables set"
    families = {
        family for name, family in _DATABRICKS_CREDENTIAL_VAR_AUTH.items() if os.environ.get(name)
    }
    if len(families) > 1:
        return None, (
            f"credentials for more than one auth method are set ({', '.join(sorted(families))}), "
            "which the SDK rejects unless $DATABRICKS_AUTH_TYPE picks one"
        )
    if not complete:
        return None, "none of a token, a username and password, or a client id and secret is set"
    return next(iter(complete.values())), ""


def check_mlflow(check_credentials: bool = False) -> CheckResult:
    """Check the Databricks auth route for MLflow, resolved the way the SDK resolves it.

    The profile file is the one $DATABRICKS_CONFIG_FILE names, else ~/.databrickscfg. With a
    profile selected by a databricks://<profile> $MLFLOW_TRACKING_URI or
    $DATABRICKS_CONFIG_PROFILE, the profile route applies and environment variables only fill in
    or override what it leaves; when the file does not exist the SDK reads nothing from it, so
    the environment route applies instead. With none selected, any Databricks credential
    variable makes the SDK skip the file, so the environment route applies; otherwise the file's
    default profile does. A route no offline evidence settles, SDK OAuth among them, is a WARN
    naming all three routes rather than a pass.

    ``check_credentials`` validates the resolved route through the SDK, naming a profile only
    when one is selected. It first exports DATABRICKS_TOKEN from DATABRICKS_API_TOKEN as the
    backend does, so the probe authenticates the way the run will. It is also the only way to
    settle an environment route no offline evidence settles, so it runs even then.
    """
    selected = _selected_databricks_profile()
    config_name = _databricks_config_file()
    env_credentials = () if selected else _databricks_env_credentials()
    config = None if env_credentials else _read_databricks_config(config_name)
    if config is None or (selected and config.missing):
        if config is None:
            skipped = f"{', '.join(env_credentials)} set, so {config_name} is not read"
            found_prefix = ""
        else:
            skipped = f"{config_name} not found, so profile {selected} is not read"
            found_prefix = f"{skipped}; "
        auth, reason = _databricks_env_auth()
        unsettled = f"{skipped}, and the environment route is incomplete ({reason})"
        if auth is None and not check_credentials:
            return CheckResult("mlflow", WARN, f"{unsettled}; {_MLFLOW_AUTH_ROUTES}")
        found = unsettled if auth is None else f"{found_prefix}DATABRICKS_HOST + {auth} set"
    else:
        profile = selected or config.default_profile
        quotable = selected is not None or config.default_profile_quotable
        problem = config.problem
        if profile is None:
            problem = "the [__settings__] default_profile spans more than one line"
        elif profile == _SETTINGS_SECTION:
            problem = f"[{_SETTINGS_SECTION}] is a reserved section name and cannot be a profile"
        elif problem is None and profile not in config.profiles:
            if quotable:
                problem = f"{config_name} has no [{profile}] section"
            else:
                problem = f"the [__settings__] default_profile names no section of {config_name}"
        if problem:
            return CheckResult(
                "mlflow",
                WARN,
                f"no Databricks credentials found ({problem}); {_MLFLOW_AUTH_ROUTES}",
            )
        found = (
            f"profile {profile} found in {config_name}"
            if quotable
            else f"profile named by [__settings__] default_profile found in {config_name}"
        )
    if check_credentials:
        if not os.environ.get("DATABRICKS_TOKEN") and os.environ.get("DATABRICKS_API_TOKEN"):
            os.environ["DATABRICKS_TOKEN"] = os.environ["DATABRICKS_API_TOKEN"]
        try:
            from databricks.sdk import WorkspaceClient  # type: ignore

            client = WorkspaceClient(profile=selected) if selected else WorkspaceClient()
            user = client.current_user.me().user_name
        except Exception as e:
            return CheckResult("mlflow", WARN, f"{found} but SDK probe failed ({type(e).__name__})")
        return CheckResult("mlflow", OK, f"Databricks reachable, user={user}")
    return CheckResult("mlflow", OK, f"{found} (not validated)")


def check_hf(check_credentials: bool = False) -> CheckResult:
    """Check HF_TOKEN env var; optionally validate via API."""
    key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not key:
        return CheckResult(
            "hf",
            MISS,
            "HF_TOKEN / HUGGING_FACE_HUB_TOKEN not set",
            "export HF_TOKEN=... (needed for gated models or datasets)",
        )
    if check_credentials:
        try:
            from huggingface_hub import HfApi  # type: ignore

            api = HfApi(token=key)
            info = api.whoami()
        except Exception as e:
            return CheckResult("hf", WARN, f"HF_TOKEN set but API probe failed: {e}")
        return CheckResult("hf", OK, f"HF_TOKEN valid, user={info.get('name', '?')}")
    return CheckResult("hf", OK, "HF_TOKEN set (not validated)")


def check_gh() -> CheckResult:
    """Check gh CLI is available and authenticated."""
    if shutil.which("gh") is None:
        return CheckResult(
            "gh",
            MISS,
            "gh not on PATH",
            "Install GitHub CLI: https://cli.github.com/",
        )
    try:
        out = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as e:
        return CheckResult("gh", MISS, f"gh auth status failed: {e}")
    if out.returncode != 0:
        return CheckResult(
            "gh",
            MISS,
            "gh not authenticated",
            "Run: gh auth login",
        )
    combined = (out.stdout + out.stderr).strip().splitlines()
    summary = next(
        (line.strip() for line in combined if "Logged in" in line or "account" in line.lower()),
        "authenticated",
    )
    return CheckResult("gh", OK, f"gh on PATH, {summary}")


LOCAL_TOML_TEMPLATE = """[cluster]
name = "{cluster_name}"
scheduler = "{scheduler}"

[slurm]
account = "{slurm_account}"
partition = "{slurm_partition}"
qos = "{slurm_qos}"
default_time = "{slurm_default_time}"
default_nodes = {slurm_default_nodes}
default_gpus_per_node = {slurm_default_gpus_per_node}
require_account = {require_account}   # set false if your cluster does not use --account
require_qos = {require_qos}            # set true if your cluster requires --qos

[network]
ib_interface = "{ib_interface}"
shared_fs_root = "{shared_fs_root}"
# Required only for multi-node runs. Must be a distributed filesystem
# (NFS / Lustre / GPFS) that all ranks can read and write. check_env
# --requires multi-node verifies that:
#   1. the repo checkout is under shared_fs_root,
#   2. shared_fs_root is a distributed FS type,
#   3. checkpoints_root (below) is also under it.

[paths]
checkpoints_root = "{checkpoints_root}"

[credentials]
wandb_entity = "{wandb_entity}"
hf_user = "{hf_user}"
"""


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{label}{suffix}: ").strip()
    return raw or default


def _init_defaults(args: argparse.Namespace, repo_root: Path) -> dict[str, str]:
    """Resolve initial field values from CLI flags, then env vars, then static defaults."""
    env = os.environ
    return {
        "cluster_name": args.cluster_name or env.get("SLURM_CLUSTER_NAME", ""),
        "scheduler": args.scheduler or ("slurm" if shutil.which("sbatch") else "none"),
        "slurm_account": args.slurm_account or env.get("SLURM_ACCOUNT", ""),
        "slurm_partition": args.slurm_partition or env.get("SLURM_PARTITION", ""),
        "slurm_qos": args.slurm_qos or env.get("SLURM_QOS", ""),
        "slurm_default_time": args.slurm_time or "01:00:00",
        "slurm_default_nodes": args.slurm_nodes or "1",
        "slurm_default_gpus_per_node": args.slurm_gpus_per_node or "4",
        "require_account": args.require_account or "true",
        "require_qos": args.require_qos or "false",
        "ib_interface": args.ib_interface or "auto",
        "shared_fs_root": args.shared_fs_root or "",
        "checkpoints_root": args.checkpoints_root or str(repo_root / "checkpoints"),
        "wandb_entity": args.wandb_entity or "",
        "hf_user": args.hf_user or "",
    }


def _write_local_toml(target: Path, values: dict[str, str], repo_root: Path) -> None:
    """Atomically render the template, then warn if the target is git-tracked."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(LOCAL_TOML_TEMPLATE.format(**values))
    tmp.replace(target)
    try:
        rel = target.relative_to(repo_root)
    except ValueError:
        rel = None
    if rel is not None:
        try:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(rel)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if tracked.returncode == 0:
                print(
                    f"WARNING: {rel} is tracked by git. "
                    "Add configs/cluster/local.toml to .gitignore to avoid committing secrets.",
                    file=sys.stderr,
                )
        except (subprocess.SubprocessError, OSError):
            pass
    print(f"Wrote {target}")


def do_init(repo_root: Path, args: argparse.Namespace) -> int:
    """Write configs/cluster/local.toml, interactively or via --non-interactive flags."""
    target = (
        Path(args.cluster_config) if args.cluster_config else repo_root / DEFAULT_CLUSTER_CONFIG
    )
    defaults = _init_defaults(args, repo_root)

    if args.non_interactive:
        if target.exists() and not args.force:
            print(
                f"ERROR: {target} exists. Pass --force to overwrite, "
                "or omit --non-interactive for an interactive prompt.",
                file=sys.stderr,
            )
            return 2
        _write_local_toml(target, defaults, repo_root)
        return 0

    if not sys.stdin.isatty():
        print(
            "ERROR: --init requires an interactive terminal (stdin is not a TTY). "
            "Use --non-interactive with explicit flags for agent-driven writes.",
            file=sys.stderr,
        )
        return 2
    if target.exists():
        resp = input(f"{target} exists. Overwrite? [y/N]: ").strip().lower()
        if resp != "y":
            print(f"Left existing {target} untouched.")
            return 0
    values = {
        "cluster_name": _prompt("Cluster name", defaults["cluster_name"]),
        "scheduler": _prompt("Scheduler (slurm|none)", defaults["scheduler"]),
        "slurm_account": _prompt("SLURM account", defaults["slurm_account"]),
        "slurm_partition": _prompt("SLURM partition", defaults["slurm_partition"]),
        "slurm_qos": _prompt("SLURM QoS", defaults["slurm_qos"]),
        "slurm_default_time": _prompt("Default wall time", defaults["slurm_default_time"]),
        "slurm_default_nodes": _prompt("Default nodes", defaults["slurm_default_nodes"]),
        "slurm_default_gpus_per_node": _prompt(
            "Default GPUs per node", defaults["slurm_default_gpus_per_node"]
        ),
        "require_account": _prompt(
            "Require --account for this cluster? (true|false)", defaults["require_account"]
        ),
        "require_qos": _prompt(
            "Require --qos for this cluster? (true|false)", defaults["require_qos"]
        ),
        "ib_interface": _prompt(
            "InfiniBand interface (auto or e.g. ib0)", defaults["ib_interface"]
        ),
        "shared_fs_root": _prompt("Shared FS root (for multi-node)", defaults["shared_fs_root"]),
        "checkpoints_root": _prompt("Checkpoints root", defaults["checkpoints_root"]),
        "wandb_entity": _prompt("WandB entity (no API key)", defaults["wandb_entity"]),
        "hf_user": _prompt("HuggingFace username", defaults["hf_user"]),
    }
    _write_local_toml(target, values, repo_root)
    return 0


def parse_requires(raw: str | None) -> list[str]:
    """Comma-separated list; empty string or None means baseline only."""
    if raw is None or raw == "":
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def run_checks(
    requires: list[str],
    config: dict | None,
    repo_root: Path,
    check_credentials: bool = False,
) -> list[CheckResult]:
    results: list[CheckResult] = [check_uv(), check_repo_layout(repo_root)]
    done: set[str] = set()
    for tag in requires:
        if tag in done or tag in BASELINE_ALIASES:
            continue
        done.add(tag)
        if tag == "gpu":
            results.append(check_gpu())
        elif tag == "slurm":
            results.append(check_slurm(config))
        elif tag == "multi-node":
            results.append(check_multi_node(config, repo_root))
        elif tag == "wandb":
            results.append(check_wandb(check_credentials))
        elif tag == "mlflow":
            results.append(check_mlflow(check_credentials))
        elif tag == "hf":
            results.append(check_hf(check_credentials))
        elif tag == "gh":
            results.append(check_gh())
        else:
            results.append(
                CheckResult(
                    tag,
                    MISS,
                    f"unknown requires tag: {tag}",
                    f"Known tags: {', '.join(KNOWN_TAGS)}",
                )
            )
    return results


def format_text(results: list[CheckResult]) -> str:
    width = max((len(r.name) for r in results), default=4)
    lines: list[str] = []
    for r in results:
        lines.append(f"{r.name.ljust(width)}  {r.status.ljust(4)}  {r.message}")
        if r.fix and r.status == MISS:
            lines.append(f"{' ' * (width + 8)}fix: {r.fix}")
    return "\n".join(lines)


def exit_code_for(results: list[CheckResult]) -> int:
    return 1 if any(r.status == MISS for r in results) else 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="check_env",
        description="KempnerForge environment preflight for Claude Code skills.",
    )
    ap.add_argument(
        "--requires",
        type=str,
        default="",
        help="Comma-separated tags (gpu,slurm,multi-node,wandb,mlflow,hf,gh). "
        "Empty or omitted runs baseline only.",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON report.")
    ap.add_argument(
        "--init",
        action="store_true",
        help="Write configs/cluster/local.toml. Interactive unless --non-interactive is given.",
    )
    ap.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts; use --<field> flags (with env-var fallbacks). Requires --init.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing local.toml when used with --init --non-interactive.",
    )
    for flag, default_help in (
        ("--cluster-name", "Cluster label. Defaults to $SLURM_CLUSTER_NAME."),
        ("--scheduler", "slurm|none. Auto-detected from sbatch if omitted."),
        ("--slurm-account", "SLURM account. Defaults to $SLURM_ACCOUNT."),
        ("--slurm-partition", "SLURM partition. Defaults to $SLURM_PARTITION."),
        ("--slurm-qos", "SLURM QoS. Defaults to $SLURM_QOS."),
        ("--slurm-time", "Default wall time (HH:MM:SS). Defaults to 01:00:00."),
        ("--slurm-nodes", "Default node count. Defaults to 1."),
        ("--slurm-gpus-per-node", "Default GPUs per node. Defaults to 4."),
        ("--require-account", "true|false. Defaults to true."),
        ("--require-qos", "true|false. Defaults to false."),
        ("--ib-interface", "IB interface or 'auto'. Defaults to auto."),
        ("--shared-fs-root", "Distributed FS root (multi-node)."),
        ("--checkpoints-root", "Checkpoints directory. Defaults to <repo>/checkpoints."),
        ("--wandb-entity", "WandB entity name (no API key)."),
        ("--hf-user", "HuggingFace username."),
    ):
        ap.add_argument(flag, type=str, default="", help=default_help)
    ap.add_argument(
        "--cluster-config",
        type=str,
        default=None,
        help="Path to an alternate cluster config TOML.",
    )
    ap.add_argument(
        "--check-credentials",
        action="store_true",
        help="Validate WandB/HF/Databricks tokens via API (slow; opt-in).",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = find_repo_root()
    if args.non_interactive and not args.init:
        print("ERROR: --non-interactive requires --init.", file=sys.stderr)
        return 2
    if args.init:
        return do_init(repo_root, args)
    try:
        config = load_cluster_config(repo_root, args.cluster_config)
    except Exception as e:
        print(f"ERROR: failed to read cluster config: {e}", file=sys.stderr)
        return 2
    requires = parse_requires(args.requires)
    try:
        results = run_checks(requires, config, repo_root, args.check_credentials)
    except Exception as e:
        print(f"ERROR: check_env internal failure: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"results": [r.to_dict() for r in results]}, indent=2))
    else:
        print(format_text(results))
    return exit_code_for(results)


if __name__ == "__main__":
    sys.exit(main())
