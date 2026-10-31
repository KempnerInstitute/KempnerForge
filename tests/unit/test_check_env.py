"""Unit tests for scripts/check_env.py."""

from __future__ import annotations

import configparser
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import check_env  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_which(available: set[str]):
    def _which(binary):
        return f"/usr/bin/{binary}" if binary in available else None

    return _which


def _run_map(cases):
    """Build a subprocess.run stub driven by a list of (prefix, result) pairs.

    ``prefix`` is a tuple matched against the first N elements of the command.
    """

    def _run(cmd, *_args, **_kwargs):
        key = tuple(cmd) if isinstance(cmd, list | tuple) else (cmd,)
        for prefix, result in cases:
            if key[: len(prefix)] == prefix:
                return result
        return MagicMock(returncode=127, stdout="", stderr="not found")

    return _run


def _fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "kempnerforge").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


class TestCheckUv:
    def test_ok(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"uv"}))
        monkeypatch.setattr(
            "subprocess.run",
            _run_map(
                [
                    (
                        ("uv", "--version"),
                        MagicMock(returncode=0, stdout="uv 0.4.28\n", stderr=""),
                    )
                ]
            ),
        )
        r = check_env.check_uv()
        assert r.status == check_env.OK
        assert "uv 0.4.28" in r.message

    def test_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        r = check_env.check_uv()
        assert r.status == check_env.MISS
        assert "PATH" in r.message
        assert r.fix

    def test_uv_crashes(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"uv"}))

        def _boom(*_a, **_kw):
            raise OSError("broken")

        monkeypatch.setattr("subprocess.run", _boom)
        r = check_env.check_uv()
        assert r.status == check_env.MISS


class TestCheckRepoLayout:
    def test_ok(self, tmp_path):
        repo = _fake_repo(tmp_path)
        r = check_env.check_repo_layout(repo)
        assert r.status == check_env.OK

    def test_missing_dir(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("")
        # configs/ and kempnerforge/ missing
        r = check_env.check_repo_layout(tmp_path)
        assert r.status == check_env.MISS
        assert "configs" in r.message
        assert "kempnerforge" in r.message


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


class TestCheckGpu:
    def test_ok_via_nvidia_smi(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"nvidia-smi"}))
        monkeypatch.setattr(
            "subprocess.run",
            _run_map(
                [
                    (
                        ("nvidia-smi",),
                        MagicMock(
                            returncode=0,
                            stdout="H100 80GB HBM3, 535.129.03\nH100 80GB HBM3, 535.129.03\n",
                            stderr="",
                        ),
                    )
                ]
            ),
        )
        r = check_env.check_gpu()
        assert r.status == check_env.OK
        assert "H100" in r.message
        assert "2x" in r.message

    def test_fallback_via_dev_nvidia(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        real_exists = Path.exists

        def fake_exists(self):
            if str(self) == "/dev/nvidia0":
                return True
            return real_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)
        r = check_env.check_gpu()
        assert r.status == check_env.OK
        assert "device nodes" in r.message.lower()

    def test_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        real_exists = Path.exists

        def fake_exists(self):
            s = str(self)
            if s in ("/dev/nvidia0", "/proc/driver/nvidia/gpus"):
                return False
            return real_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)
        r = check_env.check_gpu()
        assert r.status == check_env.MISS


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------


class TestCheckSlurm:
    def test_no_sbatch(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        r = check_env.check_slurm(None)
        assert r.status == check_env.MISS
        assert "sbatch" in r.message

    def test_sbatch_no_account(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"sbatch"}))
        monkeypatch.delenv("SLURM_ACCOUNT", raising=False)
        monkeypatch.delenv("SBATCH_ACCOUNT", raising=False)
        r = check_env.check_slurm(None)
        assert r.status == check_env.MISS
        assert "account" in r.message

    def test_account_from_config(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"sbatch"}))
        monkeypatch.delenv("SLURM_ACCOUNT", raising=False)
        r = check_env.check_slurm({"slurm": {"account": "lab_x"}})
        assert r.status == check_env.OK
        assert "account=lab_x" in r.message

    def test_account_from_env(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"sbatch"}))
        monkeypatch.setenv("SLURM_ACCOUNT", "lab_y")
        r = check_env.check_slurm(None)
        assert r.status == check_env.OK
        assert "account=lab_y" in r.message

    def test_config_overrides_env(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"sbatch"}))
        monkeypatch.setenv("SLURM_ACCOUNT", "env_val")
        r = check_env.check_slurm({"slurm": {"account": "config_val"}})
        assert r.status == check_env.OK
        assert "config_val" in r.message
        assert "env_val" not in r.message

    def test_require_account_false_allows_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"sbatch"}))
        monkeypatch.delenv("SLURM_ACCOUNT", raising=False)
        monkeypatch.delenv("SBATCH_ACCOUNT", raising=False)
        monkeypatch.delenv("SLURM_QOS", raising=False)
        r = check_env.check_slurm({"slurm": {"require_account": False}})
        assert r.status == check_env.WARN
        assert "sbatch on PATH" in r.message

    def test_require_qos_true_enforces_qos(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"sbatch"}))
        monkeypatch.delenv("SLURM_QOS", raising=False)
        cfg = {"slurm": {"account": "lab_x", "require_qos": True}}
        r = check_env.check_slurm(cfg)
        assert r.status == check_env.MISS
        assert "qos" in r.message

    def test_qos_included_in_message(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"sbatch"}))
        cfg = {"slurm": {"account": "lab_x", "qos": "high"}}
        r = check_env.check_slurm(cfg)
        assert r.status == check_env.OK
        assert "qos=high" in r.message

    def test_qos_from_env(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"sbatch"}))
        monkeypatch.setenv("SLURM_QOS", "env_qos")
        r = check_env.check_slurm({"slurm": {"account": "lab_x"}})
        assert r.status == check_env.OK
        assert "qos=env_qos" in r.message


# ---------------------------------------------------------------------------
# Multi-node
# ---------------------------------------------------------------------------


class TestCheckMultiNode:
    def test_no_config(self, tmp_path):
        r = check_env.check_multi_node(None, tmp_path)
        assert r.status == check_env.MISS
        assert "[network]" in r.message

    def test_no_shared_root(self, tmp_path):
        r = check_env.check_multi_node({"network": {}}, tmp_path)
        assert r.status == check_env.MISS
        assert "shared_fs_root" in r.message

    def test_shared_root_missing_on_disk(self, tmp_path):
        cfg = {"network": {"shared_fs_root": str(tmp_path / "does_not_exist")}}
        r = check_env.check_multi_node(cfg, tmp_path)
        assert r.status == check_env.MISS
        assert "does not exist" in r.message

    def test_repo_not_under_shared(self, tmp_path):
        shared = tmp_path / "shared"
        shared.mkdir()
        repo = tmp_path / "elsewhere"
        repo.mkdir()
        cfg = {"network": {"shared_fs_root": str(shared)}}
        r = check_env.check_multi_node(cfg, repo)
        assert r.status == check_env.MISS
        assert "not under" in r.message

    def test_ok_path_but_unverified_fs_and_ib(self, tmp_path, monkeypatch):
        shared = tmp_path / "shared"
        shared.mkdir()
        repo = shared / "repo"
        repo.mkdir()
        cfg = {"network": {"shared_fs_root": str(shared)}}
        # Force _fs_type to return something non-distributed
        monkeypatch.setattr(check_env, "_fs_type", lambda p: "ext4")
        # Force IB path not to exist
        real_exists = Path.exists

        def fake_exists(self):
            if str(self) == "/sys/class/infiniband":
                return False
            return real_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)
        r = check_env.check_multi_node(cfg, repo)
        assert r.status == check_env.WARN

    def test_checkpoints_not_under_shared(self, tmp_path, monkeypatch):
        shared = tmp_path / "shared"
        shared.mkdir()
        repo = shared / "repo"
        repo.mkdir()
        ckpt = tmp_path / "elsewhere"
        ckpt.mkdir()
        cfg = {
            "network": {"shared_fs_root": str(shared)},
            "paths": {"checkpoints_root": str(ckpt)},
        }
        monkeypatch.setattr(check_env, "_fs_type", lambda p: "nfs")
        r = check_env.check_multi_node(cfg, repo)
        assert r.status == check_env.WARN
        assert "checkpoints_root" in r.message


# ---------------------------------------------------------------------------
# WandB, HF, gh
# ---------------------------------------------------------------------------


class TestCheckWandb:
    def test_missing(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        r = check_env.check_wandb()
        assert r.status == check_env.MISS

    def test_set_unvalidated(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "xxx")
        r = check_env.check_wandb()
        assert r.status == check_env.OK
        assert "not validated" in r.message


class TestCheckMlflow:
    _CLEARED_VARS = (
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DATABRICKS_API_TOKEN",
        "DATABRICKS_USERNAME",
        "DATABRICKS_PASSWORD",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_OIDC_TOKEN_ENV",
        "DATABRICKS_OIDC_TOKEN_FILEPATH",
        "DATABRICKS_OIDC_TOKEN_FILE",
        "DATABRICKS_GOOGLE_SERVICE_ACCOUNT",
        "GOOGLE_CREDENTIALS",
        "DATABRICKS_AZURE_RESOURCE_ID",
        "ARM_USE_MSI",
        "ARM_CLIENT_ID",
        "ARM_CLIENT_SECRET",
        "ARM_TENANT_ID",
        "DATABRICKS_METADATA_SERVICE_URL",
        "DATABRICKS_AUTH_TYPE",
        "DATABRICKS_CONFIG_PROFILE",
        "MLFLOW_TRACKING_URI",
    )

    @pytest.fixture(autouse=True)
    def home(self, monkeypatch, tmp_path):
        """An empty home, with no ambient credentials or profile selection in the environment."""
        for name in self._CLEARED_VARS:
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    @staticmethod
    def _fake_sdk(monkeypatch, client_cls):
        import types as _types

        fake_sdk = _types.ModuleType("databricks.sdk")
        fake_sdk.WorkspaceClient = client_cls
        monkeypatch.setitem(sys.modules, "databricks", _types.ModuleType("databricks"))
        monkeypatch.setitem(sys.modules, "databricks.sdk", fake_sdk)

    @staticmethod
    def _recording_client(seen: list, user_name: str = "someone"):
        import types as _types

        class _WC:
            def __init__(self, **kwargs):
                seen.append(kwargs)
                self.current_user = _types.SimpleNamespace(
                    me=lambda: _types.SimpleNamespace(user_name=user_name)
                )

        return _WC

    def test_nothing_configured_names_the_routes(self):
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "~/.databrickscfg not found" in r.message
        for route in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "~/.databrickscfg", "OAuth"):
            assert route in r.message
        assert not r.fix

    def test_host_without_token_is_informational(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        assert check_env.check_mlflow().status == check_env.WARN

    def test_token_without_host_is_informational(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_API_TOKEN", "dapi-xxx")
        assert check_env.check_mlflow().status == check_env.WARN

    @pytest.mark.parametrize("token_var", ["DATABRICKS_TOKEN", "DATABRICKS_API_TOKEN"])
    def test_env_pair_ok(self, monkeypatch, token_var):
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv(token_var, "dapi-xxx")
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "not validated" in r.message

    def test_informational_result_does_not_fail_the_preflight(self):
        assert check_env.exit_code_for([check_env.check_mlflow()]) == 0

    def test_default_profile_ok(self, home):
        (home / ".databrickscfg").write_text("[DEFAULT]\nhost = h\ntoken = secret-value\n")
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "profile DEFAULT" in r.message
        assert "not validated" in r.message

    def test_profile_from_tracking_uri(self, home, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks://my-profile")
        (home / ".databrickscfg").write_text("[other]\nhost = h\n\n[my-profile]\nhost = h\n")
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "profile my-profile" in r.message

    def test_profile_from_config_profile_env(self, home, monkeypatch):
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "my-profile")
        (home / ".databrickscfg").write_text("[my-profile]\nhost = h\n")
        assert check_env.check_mlflow().status == check_env.OK

    def test_tracking_uri_profile_wins_over_config_profile_env(self, home, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks://from-uri")
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "from-env")
        (home / ".databrickscfg").write_text("[from-env]\nhost = h\n")
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "no [from-uri] section" in r.message

    def test_profile_not_in_file_is_informational(self, home):
        (home / ".databrickscfg").write_text("[other]\nhost = h\n")
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "no [DEFAULT] section" in r.message
        assert "OAuth" in r.message

    def test_commented_header_is_not_a_profile(self, home):
        (home / ".databrickscfg").write_text("# [DEFAULT]\n; [DEFAULT]\nhost = [DEFAULT]\n")
        assert check_env.check_mlflow().status == check_env.WARN

    def test_header_may_carry_a_trailing_comment(self, home):
        (home / ".databrickscfg").write_text("[DEFAULT]  # personal\nhost = h\n")
        assert check_env.check_mlflow().status == check_env.OK

    def test_unreadable_file_is_informational(self, home):
        (home / ".databrickscfg").mkdir()
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "unreadable" in r.message

    @pytest.mark.parametrize(
        "uri",
        [
            "",
            "databricks",
            "databricks://scope:prefix",
            "databricks://id@host",
            "http://h:1",
            "databricks://[",
        ],
    )
    def test_uri_naming_no_profile_selects_none(self, monkeypatch, uri):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        assert check_env._selected_databricks_profile() is None

    def test_malformed_tracking_uri_does_not_crash_the_preflight(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks://[")
        assert check_env.check_mlflow().status == check_env.WARN

    def test_values_are_never_printed(self, home, monkeypatch):
        cfg = home / ".databrickscfg"
        cfg.write_text("[work]\nhost = https://private.example\ntoken = secret-value\n")
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "work")
        found = check_env.check_mlflow()
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "absent")
        missing = check_env.check_mlflow()
        assert (found.status, missing.status) == (check_env.OK, check_env.WARN)
        for r in (found, missing):
            shown = json.dumps(r.to_dict()) + check_env.format_text([r])
            assert "secret-value" not in shown and "private.example" not in shown

    def test_check_credentials_applies_api_token_alias(self, monkeypatch):
        """--check-credentials mirrors DATABRICKS_API_TOKEN->DATABRICKS_TOKEN before the SDK
        probe (the same alias the backend applies), so it doesn't falsely WARN. (Finding #6)"""
        import types as _types

        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_API_TOKEN", "dapi-x")
        monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

        class _WC:  # authenticates only when the SDK-native DATABRICKS_TOKEN is present
            def __init__(self):
                import os as _os

                if not _os.environ.get("DATABRICKS_TOKEN"):
                    raise RuntimeError("no DATABRICKS_TOKEN")
                self.current_user = _types.SimpleNamespace(
                    me=lambda: _types.SimpleNamespace(user_name="abbas")
                )

        self._fake_sdk(monkeypatch, _WC)

        r = check_env.check_mlflow(check_credentials=True)
        assert r.status == check_env.OK
        assert "abbas" in r.message

    def test_check_credentials_probes_the_selected_profile(self, home, monkeypatch):
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "my-profile")
        (home / ".databrickscfg").write_text("[my-profile]\nhost = h\n")
        seen: list[dict] = []
        self._fake_sdk(monkeypatch, self._recording_client(seen, "someone"))
        r = check_env.check_mlflow(check_credentials=True)
        assert r.status == check_env.OK
        assert "someone" in r.message
        assert seen == [{"profile": "my-profile"}]

    def test_check_credentials_probes_the_env_pair_without_a_profile(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")
        seen: list[dict] = []
        self._fake_sdk(monkeypatch, self._recording_client(seen))
        assert check_env.check_mlflow(check_credentials=True).status == check_env.OK
        assert seen == [{}]

    def test_failed_probe_is_a_warn_naming_the_route_and_error_class(self, home, monkeypatch):
        (home / ".databrickscfg").write_text("[DEFAULT]\nhost = h\n")

        class _Failing:
            def __init__(self, **kwargs):
                raise RuntimeError("401 unauthorized")

        self._fake_sdk(monkeypatch, _Failing)
        r = check_env.check_mlflow(check_credentials=True)
        assert r.status == check_env.WARN
        assert "profile DEFAULT" in r.message and "RuntimeError" in r.message
        assert "401 unauthorized" not in r.message

    @pytest.mark.parametrize(
        "error",
        [
            configparser.MissingSectionHeaderError("profile", 3, "token secret-value"),
            ValueError("Invalid header value b'Bearer secret-value\\ncontinued'"),
        ],
        ids=["parser-line", "header-value"],
    )
    @pytest.mark.parametrize("route", ["profile", "env"])
    def test_failed_probe_never_quotes_a_secret(self, home, monkeypatch, route, error):
        if route == "profile":
            (home / ".databrickscfg").write_text("[DEFAULT]\nhost = h\ntoken = secret-value\n")
        else:
            monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
            monkeypatch.setenv("DATABRICKS_TOKEN", "secret-value")

        class _Raising:
            def __init__(self, **kwargs):
                raise error

        self._fake_sdk(monkeypatch, _Raising)
        r = check_env.check_mlflow(check_credentials=True)
        assert r.status == check_env.WARN
        assert type(error).__name__ in r.message
        assert "secret-value" not in json.dumps(r.to_dict()) + check_env.format_text([r])

    def test_check_credentials_does_not_probe_when_nothing_is_found(self, monkeypatch):
        seen: list[dict] = []
        self._fake_sdk(monkeypatch, self._recording_client(seen))
        assert check_env.check_mlflow(check_credentials=True).status == check_env.WARN
        assert seen == []

    def test_lone_token_beside_a_default_section_is_not_the_profile_route(self, home, monkeypatch):
        """A credential variable with no profile selected makes the SDK skip ~/.databrickscfg."""
        (home / ".databrickscfg").write_text("[DEFAULT]\nhost = h\ntoken = secret-value\n")
        monkeypatch.setenv("DATABRICKS_TOKEN", "exported-elsewhere")
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "profile DEFAULT" not in r.message
        assert "so ~/.databrickscfg is not read" in r.message
        for route in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "~/.databrickscfg", "OAuth"):
            assert route in r.message

    def test_lone_host_beside_a_default_section_is_not_the_profile_route(self, home, monkeypatch):
        (home / ".databrickscfg").write_text("[DEFAULT]\nhost = h\ntoken = secret-value\n")
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "profile DEFAULT" not in r.message
        assert "so ~/.databrickscfg is not read" in r.message

    def test_env_pair_beside_a_default_section_is_the_environment_route(self, home, monkeypatch):
        (home / ".databrickscfg").write_text("[DEFAULT]\nhost = h\ntoken = secret-value\n")
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert r.message == "DATABRICKS_HOST + token set (not validated)"

    def test_oauth_client_credentials_are_a_complete_environment_route(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "an-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "a-secret")
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "client id and secret" in r.message

    def test_username_and_password_are_a_complete_environment_route(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_USERNAME", "someone")
        monkeypatch.setenv("DATABRICKS_PASSWORD", "a-secret")
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "username and password" in r.message

    def test_half_an_oauth_pair_is_an_incomplete_environment_route(self, home, monkeypatch):
        (home / ".databrickscfg").write_text("[DEFAULT]\nhost = h\ntoken = secret-value\n")
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "an-id")
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "profile DEFAULT" not in r.message
        assert "client id and secret" in r.message

    @pytest.mark.parametrize(
        "var",
        [
            "ARM_CLIENT_ID",
            "GOOGLE_CREDENTIALS",
            "DATABRICKS_METADATA_SERVICE_URL",
            "DATABRICKS_OIDC_TOKEN_FILE",
            "DATABRICKS_AZURE_RESOURCE_ID",
        ],
    )
    def test_any_credential_variable_makes_the_sdk_skip_the_file(self, home, monkeypatch, var):
        (home / ".databrickscfg").write_text("[DEFAULT]\nhost = h\ntoken = secret-value\n")
        monkeypatch.setenv(var, "set-elsewhere")
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "profile DEFAULT" not in r.message
        assert var in r.message

    @pytest.mark.parametrize(
        "selector",
        [
            ("MLFLOW_TRACKING_URI", "databricks://my-profile"),
            ("DATABRICKS_CONFIG_PROFILE", "my-profile"),
        ],
        ids=["tracking-uri", "config-profile"],
    )
    def test_a_selected_profile_route_survives_environment_variables(
        self, home, monkeypatch, selector
    ):
        """A selected profile is read whatever the environment carries; it only fills in."""
        (home / ".databrickscfg").write_text("[my-profile]\nhost = h\n")
        monkeypatch.setenv(*selector)
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "profile my-profile found in ~/.databrickscfg" in r.message

    def test_a_selected_profile_missing_from_the_file_is_a_warn_despite_env_credentials(
        self, home, monkeypatch
    ):
        """The SDK raises on a selected profile the file lacks, whatever the environment has."""
        (home / ".databrickscfg").write_text("[other]\nhost = h\n")
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "my-profile")
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "no [my-profile] section" in r.message

    def test_settings_default_profile_beats_the_default_section(self, home):
        home.joinpath(".databrickscfg").write_text(
            "[__settings__]\ndefault_profile = work\n\n[DEFAULT]\nhost = h\n\n[work]\nhost = h\n"
        )
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "profile work found in ~/.databrickscfg" in r.message

    def test_settings_default_profile_must_name_a_section_that_exists(self, home):
        home.joinpath(".databrickscfg").write_text(
            "[__settings__]\ndefault_profile = work\n\n[DEFAULT]\nhost = h\n"
        )
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "no [work] section" in r.message

    def test_settings_default_profile_loses_to_a_selected_profile(self, home, monkeypatch):
        home.joinpath(".databrickscfg").write_text(
            "[__settings__]\ndefault_profile = work\n\n[work]\nhost = h\n\n[mine]\nhost = h\n"
        )
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "mine")
        assert "profile mine" in check_env.check_mlflow().message

    @pytest.mark.parametrize(
        "text",
        [
            "[DEFAULT]\ndefault_profile = work\n",
            "[__settings__]\ndefault_profile =\n",
            "[__settings__]\n\n[other]\ndefault_profile = work\n",
        ],
        ids=["wrong-section", "empty-value", "left-the-section"],
    )
    def test_default_profile_outside_the_settings_section_is_ignored(self, home, text):
        home.joinpath(".databrickscfg").write_text(text)
        assert check_env._read_databricks_config()[1] == "DEFAULT"

    def test_settings_default_profile_accepts_a_colon_delimiter(self, home):
        home.joinpath(".databrickscfg").write_text("[__settings__]\nDefault_Profile: work\n")
        assert check_env._read_databricks_config()[1] == "work"

    def test_default_profile_falls_back_when_the_file_is_unreadable(self, home):
        (home / ".databrickscfg").mkdir()
        profiles, default, problem = check_env._read_databricks_config()
        assert (profiles, default) == (frozenset(), "DEFAULT")
        assert problem is not None and "unreadable" in problem

    def test_two_auth_methods_without_a_pin_are_a_warn(self, monkeypatch):
        """The SDK refuses to build a Config with more than one auth method configured."""
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "an-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "a-secret")
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "DATABRICKS_AUTH_TYPE" in r.message

    def test_auth_type_picks_one_of_two_auth_methods(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "an-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "a-secret")
        monkeypatch.setenv("DATABRICKS_AUTH_TYPE", "oauth-m2m")
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "client id and secret" in r.message

    @pytest.mark.parametrize("pinned", ["oauth-m2m", "databricks-cli"])
    def test_auth_type_the_environment_cannot_complete_is_a_warn(self, monkeypatch, pinned):
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")
        monkeypatch.setenv("DATABRICKS_AUTH_TYPE", pinned)
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert f"DATABRICKS_AUTH_TYPE={pinned}" in r.message

    @pytest.mark.parametrize(
        "text",
        [
            "[DEFAULT]\nhost = h\ntoken = {s}\n",
            "[DEFAULT]\nhost = h\ntoken = head\n  [__settings__]\n  default_profile = {s}\n",
            "[__settings__]\ndefault_profile = work\n  token = {s}\n[work]\nhost = h\n",
            "[__settings__]\ndefault_profile =\n  token = {s}\n[DEFAULT]\nhost = h\n",
            "[__settings__]\ndefault_profile = a\n[__settings__]\ndefault_profile = b\n"
            "[a]\ntoken = {s}\n",
            "[DEFAULT]\nhost = h\ntoken = {s}\n[DEFAULT]\ntoken = {s}\n",
            "[a]\ntoken = {s}\n[a]\ntoken = {s}\n",
            "{s}\n",
            "[DEFAULT]\n{s}\n",
            "[{s}]\nhost = h\n",
        ],
        ids=[
            "plain-token",
            "continued-into-a-settings-section",
            "continued-under-default-profile",
            "continued-under-an-empty-default-profile",
            "duplicate-settings-section",
            "duplicate-default-section",
            "duplicate-profile-section",
            "no-section-header",
            "key-without-a-delimiter",
            "section-name",
        ],
    )
    def test_no_value_read_from_the_profile_file_reaches_the_message(self, home, text):
        """Every line of the file can hold a token, so none of it is ever quoted back."""
        secret = "dapi-never-quote-me"
        home.joinpath(".databrickscfg").write_text(text.format(s=secret))
        assert secret not in check_env.check_mlflow().message

    def test_an_indented_section_inside_a_token_is_not_a_settings_section(self, home):
        """ConfigParser continues the value above an indented line, and so does the check."""
        home.joinpath(".databrickscfg").write_text(
            "[DEFAULT]\nhost = h\ntoken = dapi-first-line\n"
            "  [__settings__]\n  default_profile = dapi-secret-tail\n"
        )
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "profile DEFAULT" in r.message
        assert "dapi" not in r.message

    def test_a_file_the_sdk_cannot_parse_is_a_warn_quoting_nothing_from_it(self, home):
        home.joinpath(".databrickscfg").write_text(
            "[__settings__]\ndefault_profile = alpha\n[__settings__]\ndefault_profile = beta\n"
            "[alpha]\nhost = h\ntoken = dapi-secret\n"
        )
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "malformed (DuplicateSectionError)" in r.message
        assert "dapi-secret" not in r.message
        assert "alpha" not in r.message

    def test_the_reserved_settings_section_is_not_a_default_profile(self, home):
        home.joinpath(".databrickscfg").write_text(
            "[__settings__]\ndefault_profile = __settings__\n[DEFAULT]\nhost = h\ntoken = t\n"
        )
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "reserved section name" in r.message

    def test_the_reserved_settings_section_is_not_a_selected_profile(self, home, monkeypatch):
        home.joinpath(".databrickscfg").write_text(
            "[__settings__]\ndefault_profile = work\n[work]\nhost = h\ntoken = t\n"
        )
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "__settings__")
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "reserved section name" in r.message

    def test_a_default_section_with_no_keys_is_not_a_profile(self, home):
        """The SDK falls back to DEFAULT only when that section carries keys."""
        home.joinpath(".databrickscfg").write_text("[DEFAULT]\n")
        assert check_env.check_mlflow().status == check_env.WARN

    def test_a_percent_in_a_token_is_not_read_as_an_interpolation(self, home):
        home.joinpath(".databrickscfg").write_text("[DEFAULT]\nhost = h\ntoken = dapi-100%-of-it\n")
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "profile DEFAULT" in r.message

    def test_an_environment_route_no_offline_evidence_settles_is_a_warn(self, home, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("ARM_CLIENT_ID", "an-id")
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "~/.databrickscfg is not read" in r.message

    def test_a_default_profile_spanning_more_than_one_line_is_not_a_section_name(self, home):
        """Lines indented under default_profile continue its value and can be a token."""
        home.joinpath(".databrickscfg").write_text(
            "[__settings__]\ndefault_profile = work\n  token = dapi-secret\n[work]\nhost = h\n"
        )
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "spans more than one line" in r.message
        assert "dapi-secret" not in r.message

    def test_an_empty_default_profile_continued_on_the_next_line_is_not_a_section_name(self, home):
        """Stripping first would hide the continuation, leaving a token as the profile name."""
        home.joinpath(".databrickscfg").write_text(
            "[__settings__]\ndefault_profile =\n  token = dapi-secret\n[DEFAULT]\nhost = h\n"
        )
        r = check_env.check_mlflow()
        assert r.status == check_env.WARN
        assert "spans more than one line" in r.message
        assert "dapi-secret" not in r.message

    def test_a_default_section_whose_only_key_is_default_profile_is_still_a_profile(self, home):
        """The SDK counts DEFAULT by its keys, default_profile among them."""
        home.joinpath(".databrickscfg").write_text(
            "[DEFAULT]\ndefault_profile = work\n[work]\nhost = h\n"
        )
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "profile DEFAULT" in r.message

    def test_a_repeated_default_section_merges_as_the_sdk_merges_it(self, home):
        """ConfigParser takes a second [DEFAULT] as more keys, not a duplicate section."""
        home.joinpath(".databrickscfg").write_text("[DEFAULT]\nhost = h\n[DEFAULT]\ntoken = t\n")
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "profile DEFAULT" in r.message

    def test_a_percent_in_a_default_profile_is_not_an_interpolation(self, home):
        home.joinpath(".databrickscfg").write_text(
            "[__settings__]\ndefault_profile = my%profile\n[my%profile]\nhost = h\n"
        )
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "profile my%profile" in r.message

    def test_a_default_profile_in_the_default_section_does_not_reach_the_settings_section(
        self, home
    ):
        """ConfigParser propagates [DEFAULT] keys into every section; the SDK reads neither."""
        home.joinpath(".databrickscfg").write_text(
            "[DEFAULT]\nhost = h\ndefault_profile = work\n[__settings__]\n[work]\nhost = h\n"
        )
        r = check_env.check_mlflow()
        assert r.status == check_env.OK
        assert "profile DEFAULT" in r.message

    def test_check_credentials_probes_an_environment_route_it_cannot_settle(
        self, home, monkeypatch
    ):
        """Azure and the other routes a preflight cannot read are settled by the probe itself."""
        (home / ".databrickscfg").write_text("[DEFAULT]\nhost = h\ntoken = t\n")
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("ARM_CLIENT_ID", "an-id")
        monkeypatch.setenv("ARM_CLIENT_SECRET", "a-secret")
        monkeypatch.setenv("ARM_TENANT_ID", "a-tenant")
        seen: list[dict] = []
        self._fake_sdk(monkeypatch, self._recording_client(seen))
        assert check_env.check_mlflow(check_credentials=True).status == check_env.OK
        assert seen == [{}]

    def test_check_credentials_does_not_force_a_profile_when_none_is_selected(
        self, home, monkeypatch
    ):
        """The backend never passes profile=, so neither does the probe."""
        (home / ".databrickscfg").write_text("[DEFAULT]\nhost = h\n")
        seen: list[dict] = []
        self._fake_sdk(monkeypatch, self._recording_client(seen))
        assert check_env.check_mlflow(check_credentials=True).status == check_env.OK
        assert seen == [{}]

    def test_the_fixture_clears_every_credential_variable(self):
        assert set(check_env._DATABRICKS_CREDENTIAL_VAR_AUTH) <= set(self._CLEARED_VARS)


class TestCheckHf:
    def test_missing(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        r = check_env.check_hf()
        assert r.status == check_env.MISS

    def test_set_from_hf_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "xxx")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        r = check_env.check_hf()
        assert r.status == check_env.OK

    def test_set_from_hub_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "xxx")
        r = check_env.check_hf()
        assert r.status == check_env.OK


class TestCheckGh:
    def test_missing_binary(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        r = check_env.check_gh()
        assert r.status == check_env.MISS

    def test_not_authenticated(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"gh"}))
        monkeypatch.setattr(
            "subprocess.run",
            _run_map(
                [
                    (
                        ("gh", "auth", "status"),
                        MagicMock(returncode=1, stdout="", stderr="not logged in"),
                    )
                ]
            ),
        )
        r = check_env.check_gh()
        assert r.status == check_env.MISS
        assert "auth login" in r.fix

    def test_authenticated(self, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"gh"}))
        monkeypatch.setattr(
            "subprocess.run",
            _run_map(
                [
                    (
                        ("gh", "auth", "status"),
                        MagicMock(
                            returncode=0,
                            stdout="",
                            stderr="Logged in to github.com as tester\n",
                        ),
                    )
                ]
            ),
        )
        r = check_env.check_gh()
        assert r.status == check_env.OK


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class TestParseRequires:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, []),
            ("", []),
            ("  ", []),
            ("gpu", ["gpu"]),
            ("gpu,slurm", ["gpu", "slurm"]),
            ("gpu, slurm ,multi-node", ["gpu", "slurm", "multi-node"]),
            ("gpu,,slurm", ["gpu", "slurm"]),
        ],
    )
    def test_parse(self, raw, expected):
        assert check_env.parse_requires(raw) == expected


class TestRunChecks:
    def test_baseline_only(self, tmp_path, monkeypatch):
        repo = _fake_repo(tmp_path)
        monkeypatch.setattr("shutil.which", _fake_which({"uv"}))
        monkeypatch.setattr(
            "subprocess.run",
            _run_map(
                [
                    (
                        ("uv", "--version"),
                        MagicMock(returncode=0, stdout="uv 0.4.28\n", stderr=""),
                    )
                ]
            ),
        )
        results = check_env.run_checks([], None, repo)
        assert [r.name for r in results] == ["uv", "repo-layout"]
        assert all(r.status == check_env.OK for r in results)

    def test_unknown_tag_is_miss(self, tmp_path, monkeypatch):
        repo = _fake_repo(tmp_path)
        monkeypatch.setattr("shutil.which", _fake_which({"uv"}))
        monkeypatch.setattr(
            "subprocess.run",
            _run_map(
                [
                    (
                        ("uv", "--version"),
                        MagicMock(returncode=0, stdout="uv 1\n", stderr=""),
                    )
                ]
            ),
        )
        results = check_env.run_checks(["bogus"], None, repo)
        assert results[-1].name == "bogus"
        assert results[-1].status == check_env.MISS

    def test_baseline_aliases_are_ignored(self, tmp_path, monkeypatch):
        repo = _fake_repo(tmp_path)
        monkeypatch.setattr("shutil.which", _fake_which({"uv"}))
        monkeypatch.setattr(
            "subprocess.run",
            _run_map(
                [
                    (
                        ("uv", "--version"),
                        MagicMock(returncode=0, stdout="uv 1\n", stderr=""),
                    )
                ]
            ),
        )
        results = check_env.run_checks(["uv", "repo-layout"], None, repo)
        assert [r.name for r in results] == ["uv", "repo-layout"]


class TestExitCode:
    def test_all_ok(self):
        results = [check_env.CheckResult("a", check_env.OK, "")]
        assert check_env.exit_code_for(results) == 0

    def test_warn_is_zero(self):
        results = [check_env.CheckResult("a", check_env.WARN, "")]
        assert check_env.exit_code_for(results) == 0

    def test_miss_is_one(self):
        results = [
            check_env.CheckResult("a", check_env.OK, ""),
            check_env.CheckResult("b", check_env.MISS, "broken"),
        ]
        assert check_env.exit_code_for(results) == 1


class TestFormatText:
    def test_alignment(self):
        results = [
            check_env.CheckResult("uv", check_env.OK, "ok"),
            check_env.CheckResult("multi-node", check_env.WARN, "partial"),
        ]
        out = check_env.format_text(results)
        lines = out.splitlines()
        assert lines[0].startswith("uv          ")
        assert lines[1].startswith("multi-node  ")

    def test_fix_indented_under_miss(self):
        results = [
            check_env.CheckResult("gpu", check_env.MISS, "absent", fix="install cuda"),
        ]
        out = check_env.format_text(results)
        assert "fix: install cuda" in out


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------


class TestMain:
    def test_baseline_json(self, tmp_path, monkeypatch, capsys):
        repo = _fake_repo(tmp_path)
        monkeypatch.setattr(check_env, "find_repo_root", lambda start=None: repo)
        monkeypatch.setattr("shutil.which", _fake_which({"uv"}))
        monkeypatch.setattr(
            "subprocess.run",
            _run_map(
                [
                    (
                        ("uv", "--version"),
                        MagicMock(returncode=0, stdout="uv 1\n", stderr=""),
                    )
                ]
            ),
        )
        rc = check_env.main(["--json"])
        out = capsys.readouterr().out
        payload = json.loads(out)
        names = [r["name"] for r in payload["results"]]
        assert names == ["uv", "repo-layout"]
        assert rc == 0

    def test_miss_returns_one(self, tmp_path, monkeypatch):
        repo = _fake_repo(tmp_path)
        monkeypatch.setattr(check_env, "find_repo_root", lambda start=None: repo)
        monkeypatch.setattr("shutil.which", _fake_which(set()))  # no uv
        rc = check_env.main([])
        assert rc == 1

    def test_empty_requires_is_baseline(self, tmp_path, monkeypatch):
        repo = _fake_repo(tmp_path)
        monkeypatch.setattr(check_env, "find_repo_root", lambda start=None: repo)
        monkeypatch.setattr("shutil.which", _fake_which({"uv"}))
        monkeypatch.setattr(
            "subprocess.run",
            _run_map(
                [
                    (
                        ("uv", "--version"),
                        MagicMock(returncode=0, stdout="uv 1\n", stderr=""),
                    )
                ]
            ),
        )
        rc = check_env.main(["--requires", ""])
        assert rc == 0


# ---------------------------------------------------------------------------
# do_init non-interactive path
# ---------------------------------------------------------------------------


class TestDoInitNonInteractive:
    def _parse(self, *argv) -> Any:
        return check_env.build_argparser().parse_args(list(argv))

    def test_writes_file_with_flag_values(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        target = tmp_path / "local.toml"
        args = self._parse(
            "--init",
            "--non-interactive",
            f"--cluster-config={target}",
            "--cluster-name=cx",
            "--slurm-account=acct",
            "--slurm-partition=part",
            "--slurm-qos=q",
            "--shared-fs-root=/tmp",
        )
        rc = check_env.do_init(tmp_path, args)
        assert rc == 0
        assert target.exists()
        content = target.read_text()
        assert 'name = "cx"' in content
        assert 'account = "acct"' in content
        assert 'partition = "part"' in content
        assert 'qos = "q"' in content
        assert 'shared_fs_root = "/tmp"' in content

    def test_uses_env_defaults_when_flags_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "env_cluster")
        monkeypatch.setenv("SLURM_ACCOUNT", "env_acct")
        monkeypatch.setenv("SLURM_PARTITION", "env_part")
        monkeypatch.setenv("SLURM_QOS", "env_qos")
        target = tmp_path / "local.toml"
        args = self._parse("--init", "--non-interactive", f"--cluster-config={target}")
        assert check_env.do_init(tmp_path, args) == 0
        content = target.read_text()
        assert 'name = "env_cluster"' in content
        assert 'account = "env_acct"' in content
        assert 'partition = "env_part"' in content
        assert 'qos = "env_qos"' in content

    def test_cli_flag_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        monkeypatch.setenv("SLURM_ACCOUNT", "env_acct")
        target = tmp_path / "local.toml"
        args = self._parse(
            "--init",
            "--non-interactive",
            f"--cluster-config={target}",
            "--slurm-account=cli_acct",
        )
        assert check_env.do_init(tmp_path, args) == 0
        content = target.read_text()
        assert 'account = "cli_acct"' in content
        assert "env_acct" not in content

    def test_works_without_tty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        target = tmp_path / "local.toml"
        args = self._parse("--init", "--non-interactive", f"--cluster-config={target}")
        assert check_env.do_init(tmp_path, args) == 0
        assert target.exists()

    def test_refuses_overwrite_without_force(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        target = tmp_path / "local.toml"
        target.write_text("existing contents\n")
        args = self._parse("--init", "--non-interactive", f"--cluster-config={target}")
        rc = check_env.do_init(tmp_path, args)
        assert rc == 2
        assert target.read_text() == "existing contents\n"
        err = capsys.readouterr().err
        assert "--force" in err

    def test_force_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        target = tmp_path / "local.toml"
        target.write_text("existing\n")
        args = self._parse(
            "--init",
            "--non-interactive",
            "--force",
            f"--cluster-config={target}",
            "--cluster-name=new_name",
        )
        assert check_env.do_init(tmp_path, args) == 0
        assert 'name = "new_name"' in target.read_text()

    def test_non_interactive_without_init_errors(self, capsys):
        rc = check_env.main(["--non-interactive"])
        assert rc == 2
        assert "requires --init" in capsys.readouterr().err

    def test_scheduler_auto_detects_slurm(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which({"sbatch"}))
        target = tmp_path / "local.toml"
        args = self._parse("--init", "--non-interactive", f"--cluster-config={target}")
        assert check_env.do_init(tmp_path, args) == 0
        assert 'scheduler = "slurm"' in target.read_text()

    def test_scheduler_falls_back_to_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", _fake_which(set()))
        target = tmp_path / "local.toml"
        args = self._parse("--init", "--non-interactive", f"--cluster-config={target}")
        assert check_env.do_init(tmp_path, args) == 0
        assert 'scheduler = "none"' in target.read_text()
