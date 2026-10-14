"""Import isolation: ``import kempnerforge`` must not require lmms-eval.

This example's adapter depends on the optional, undeclared ``lmms-eval``
package. Keeping the adapter here — outside ``kempnerforge/`` — is what keeps
that dependency off the core package's import path; this test pins it. It runs
in a fresh subprocess (so prior imports in the pytest session, including the
fake ``lmms_eval`` the unit conftest injects, do not pollute ``sys.modules``)
and intentionally does NOT skip when lmms-eval is installed: the property must
hold either way.
"""

from __future__ import annotations

import subprocess
import sys


def test_import_kempnerforge_does_not_import_lmms_eval():
    code = (
        "import sys, kempnerforge; "
        "assert 'lmms_eval' not in sys.modules, 'importing kempnerforge pulled in lmms_eval'; "
        "print('ISOLATED')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ISOLATED" in result.stdout
