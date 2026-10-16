"""Path bootstrap for the whole example test tree.

Puts the example root on ``sys.path`` so config ``plugins = ["recipe"]``
imports resolve in test sessions the same way they do for train-script
launches, where ``sys.path[0]`` is the script directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLE_ROOT = str(Path(__file__).resolve().parents[1])
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)
