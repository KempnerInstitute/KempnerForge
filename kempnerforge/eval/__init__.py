"""Evaluation subsystems for KempnerForge.

This namespace hosts model-type-specific evaluation subsystems that integrate
external harnesses. ``eval/vlm/`` (VLM evaluation via lmms-eval) is the only
subsystem today; future siblings (``eval/lm/``, ``eval/audio/``, ...) slot in
without restructuring.

Import isolation: this package and its submodules are deliberately NOT imported
from ``kempnerforge/__init__.py`` or any default import path, because
``eval/vlm`` depends on the optional, undeclared ``lmms-eval`` package.
``import kempnerforge`` must keep working with ``lmms-eval`` absent, so do not
add eager imports of the eval subpackages here.
"""
