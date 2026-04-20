"""Unit tests for scripts/check_env.py."""

from __future__ import annotations

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
