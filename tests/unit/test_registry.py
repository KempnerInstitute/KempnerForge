"""Tests for the registry's duplicate-registration error, which names the modules involved,
and for the loader's sys.path hint when a plugin module fails to import."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import kempnerforge.training.scheduler  # noqa: F401
from kempnerforge.config.loader import load_config
from kempnerforge.config.registry import Registry

REPO_ROOT = Path(__file__).resolve().parents[2]

COSINE_PLUGIN = """\
from kempnerforge.config.registry import registry


@registry.register_scheduler("cosine")
def build(optimizer, config, max_steps):
    raise NotImplementedError
"""


def _register_from(module: str, source: str, reg: Registry) -> None:
    """Run ``source`` as the body of a module named ``module``, with ``reg`` in scope."""
    exec(compile(source, f"<{module}>", "exec"), {"__name__": module, "reg": reg})


def _module(module: str, source: str, **names: object) -> dict[str, object]:
    """Run ``source`` as the body of a module named ``module``; return its namespace."""
    namespace: dict[str, object] = {"__name__": module, **names}
    exec(compile(source, f"<{module}>", "exec"), namespace)
    return namespace


@pytest.fixture
def write_plugin(tmp_path, monkeypatch, isolated_registry):
    """Write importable plugin modules, or a package's ``__init__`` with ``package=True``;
    their ``sys.modules`` entries, and the registry entries they add, go afterward."""
    written: list[str] = []
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(module: str, source: str, package: bool = False) -> str:
        path = tmp_path / module / "__init__.py" if package else tmp_path / f"{module}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(source)
        importlib.invalidate_caches()
        written.append(module)
        return module

    yield write
    for module in [name for name in sys.modules if name.split(".")[0] in written]:
        del sys.modules[module]


REGISTRATIONS = {
    "register": "reg.register('scheduler', 'dup', len)\n",
    "decorator": "@reg.register_scheduler('dup')\ndef build(optimizer, config, steps):\n    pass\n",
    "inside-a-function": "def setup():\n    reg.register('scheduler', 'dup', len)\n\n\nsetup()\n",
}


class TestDuplicateRegistration:
    @pytest.mark.parametrize("source", REGISTRATIONS.values(), ids=REGISTRATIONS.keys())
    def test_error_names_the_owner_and_the_module_registering_again(self, source):
        reg = Registry()
        _register_from("kf_owner_first", source, reg)
        with pytest.raises(
            ValueError,
            match="^scheduler/dup is already registered by kf_owner_first; "
            "kf_owner_second tried to register it again$",
        ):
            _register_from("kf_owner_second", source, reg)

    def test_owner_is_the_registering_module_not_the_objects_module(self):
        """``json.dumps`` lives in ``json``; the module that registered it owns it."""
        reg = Registry()
        source = "import json\n\nreg.register('mlp', 'dup', json.dumps)\n"
        _register_from("kf_owner_reexport", source, reg)
        with pytest.raises(ValueError, match=r"already registered by kf_owner_reexport\b"):
            _register_from("kf_owner_other", "reg.register('mlp', 'dup', len)\n", reg)

    def test_a_rejected_registration_changes_neither_the_entry_nor_its_owner(self):
        reg = Registry()
        _register_from("kf_owner_first", "reg.register('loss', 'dup', len)\n", reg)
        for module in ("kf_owner_second", "kf_owner_third"):
            with pytest.raises(ValueError, match=f"by kf_owner_first; {module} tried"):
                _register_from(module, "reg.register('loss', 'dup', abs)\n", reg)
        assert reg.get("loss", "dup") is len

    def test_the_same_name_in_another_category_is_not_a_duplicate(self):
        reg = Registry()
        _register_from("kf_owner_first", "reg.register('norm', 'dup', len)\n", reg)
        _register_from("kf_owner_second", "reg.register('mlp', 'dup', abs)\n", reg)
        assert reg.get("norm", "dup") is len
        assert reg.get("mlp", "dup") is abs


class TestDefiningModule:
    """A function or class defined outside the registering module is named with its module."""

    def test_a_shared_helper_names_each_plugins_function(self):
        reg = Registry()
        helper = "def register_all(builders):\n    for name, fn in builders.items():\n"
        helper += "        reg.register('loss', name, fn)\n"
        register_all = _module("kf_helper", helper, reg=reg)["register_all"]
        plugin = "def build():\n    pass\n\n\nregister_all({'dup': build})\n"
        _module("kf_user_a", plugin, register_all=register_all)
        with pytest.raises(
            ValueError,
            match=r"^loss/dup is already registered by kf_helper \(for kf_user_a\.build\); "
            r"kf_helper \(for kf_user_b\.build\) tried to register it again$",
        ):
            _module("kf_user_b", plugin, register_all=register_all)

    def test_an_init_subclass_hook_names_each_plugins_class(self):
        reg = Registry()
        hook = "class Base:\n    def __init_subclass__(cls, name, **kwargs):\n"
        hook += "        super().__init_subclass__(**kwargs)\n"
        hook += "        reg.register('loss', name, cls)\n"
        base = _module("kf_hook_base", hook, reg=reg)["Base"]
        _module("kf_hook_a", "class A(Base, name='dup'):\n    pass\n", Base=base)
        with pytest.raises(
            ValueError,
            match=r"^loss/dup is already registered by kf_hook_base \(for kf_hook_a\.A\); "
            r"kf_hook_base \(for kf_hook_b\.B\) tried to register it again$",
        ):
            _module("kf_hook_b", "class B(Base, name='dup'):\n    pass\n", Base=base)

    def test_a_reexported_function_names_where_it_is_defined(self):
        reg = Registry()
        _register_from(
            "kf_owner_reexport", "import json\n\nreg.register('mlp', 'dup', json.dumps)\n", reg
        )
        with pytest.raises(
            ValueError,
            match=r"^mlp/dup is already registered by kf_owner_reexport \(for json\.dumps\); "
            r"kf_owner_other tried to register it again$",
        ):
            _register_from("kf_owner_other", "reg.register('mlp', 'dup', len)\n", reg)

    def test_an_object_that_is_not_a_function_or_class_adds_no_module(self):
        """A partial reports ``functools`` as its module, which says nothing about the plugin."""
        reg = Registry()
        source = "import functools\n\nreg.register('loss', 'dup', functools.partial(len))\n"
        _register_from("kf_partial_a", source, reg)
        with pytest.raises(
            ValueError,
            match="^loss/dup is already registered by kf_partial_a; "
            "kf_partial_b tried to register it again$",
        ):
            _register_from("kf_partial_b", source, reg)


class TestPluginCollisions:
    def test_plugin_claiming_a_core_name_is_named_with_the_core_module(self, write_plugin):
        """Core registered ``cosine`` first, so importing the plugin fails, naming both."""
        write_plugin("kf176_core_first", COSINE_PLUGIN)
        with pytest.raises(ValueError) as excinfo:
            load_config(cli_args=['--plugins=["kf176_core_first"]'])
        message = str(excinfo.value)
        assert message.startswith(
            "Failed to import plugin module 'kf176_core_first' from config key 'plugins': "
            "ValueError: scheduler/cosine is already registered by "
            "kempnerforge.training.scheduler; kf176_core_first tried to register it again"
        )
        assert "sys.path" not in message

    def test_core_registering_after_the_plugin_names_the_plugin(self, tmp_path):
        """With only the loader imported the scheduler category is still empty, so the
        plugin takes ``cosine`` and the collision fires when core's module registers it.
        Runs in a fresh interpreter, since this one imported core already."""
        (tmp_path / "kf176_plugin_first.py").write_text(COSINE_PLUGIN)
        code = (
            "from kempnerforge.config.loader import load_config\n"
            "load_config(cli_args=['--plugins=[\"kf176_plugin_first\"]'])\n"
            "print('plugin imported')\n"
            "import kempnerforge.training.scheduler\n"
        )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(tmp_path)])}
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=150
        )
        assert result.returncode != 0
        assert result.stdout == "plugin imported\n"
        assert (
            "ValueError: scheduler/cosine is already registered by kf176_plugin_first; "
            "kempnerforge.training.scheduler tried to register it again"
        ) in result.stderr


class TestPluginImportHint:
    def test_a_missing_plugin_keeps_the_sys_path_hint(self):
        with pytest.raises(ValueError, match=r"kf176_absent.*ModuleNotFoundError.*sys\.path"):
            load_config(cli_args=['--plugins=["kf176_absent"]'])

    def test_a_plugin_that_raises_gets_no_sys_path_hint(self, write_plugin):
        write_plugin("kf176_raises", "raise RuntimeError('kaboom')\n")
        with pytest.raises(ValueError, match="kf176_raises.*RuntimeError: kaboom") as excinfo:
            load_config(cli_args=['--plugins=["kf176_raises"]'])
        assert "sys.path" not in str(excinfo.value)

    def test_a_missing_dependency_of_a_found_plugin_gets_no_sys_path_hint(self, write_plugin):
        write_plugin("kf176_needs_dep", "import kf176_missing_dependency\n")
        with pytest.raises(
            ValueError, match="No module named 'kf176_missing_dependency'"
        ) as excinfo:
            load_config(cli_args=['--plugins=["kf176_needs_dep"]'])
        assert "sys.path" not in str(excinfo.value)

    def test_a_missing_package_above_the_plugin_keeps_the_sys_path_hint(self):
        with pytest.raises(ValueError, match=r"No module named 'kf176_absent_pkg'\..*sys\.path"):
            load_config(cli_args=['--plugins=["kf176_absent_pkg.sub.mod"]'])

    def test_a_missing_module_in_a_found_package_keeps_the_sys_path_hint(self, write_plugin):
        write_plugin("kf176_pkg", "", package=True)
        with pytest.raises(ValueError, match=r"No module named 'kf176_pkg\.absent'\..*sys\.path"):
            load_config(cli_args=['--plugins=["kf176_pkg.absent"]'])
