"""Import isolation: ``import kempnerforge`` must not require lmms-eval.

The eval subpackage depends on the optional, undeclared ``lmms-eval`` package,
so it must never be imported on the default import path. This test runs in a
fresh subprocess (so prior imports in the pytest session do not pollute
``sys.modules``) and asserts that importing the main package — and even the
eval namespace packages — does not pull in ``lmms_eval``. It intentionally does
NOT skip when lmms-eval is installed: the property must hold either way.
"""

from __future__ import annotations

import subprocess
import sys


def test_import_kempnerforge_does_not_import_lmms_eval():
    code = (
        "import sys, kempnerforge, kempnerforge.eval, kempnerforge.eval.vlm; "
        "assert 'lmms_eval' not in sys.modules, 'importing kempnerforge pulled in lmms_eval'; "
        "print('ISOLATED')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ISOLATED" in result.stdout
