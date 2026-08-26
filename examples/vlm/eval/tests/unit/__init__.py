"""Package marker — load-bearing, do not remove.

Making ``unit/`` a package gives its ``conftest.py`` the qualified module name
``unit.conftest``: pytest's prepend import mode names a non-package conftest
after its bare stem, and a second module named ``conftest`` beside
``../conftest.py`` would raise ``ImportPathMismatchError``. The marker also
makes the conftest's ``from . import _fake_lmms_eval`` resolvable.
"""
