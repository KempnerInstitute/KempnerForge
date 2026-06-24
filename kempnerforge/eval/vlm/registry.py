# pyright: reportMissingImports=false
# ^ lmms-eval is an optional, undeclared dependency; see adapter.py for why this
#   file-level directive is used instead of an inline ignore.
"""lmms-eval registration manifest for the KempnerForge VLM adapter.

``MANIFEST`` is discovered by lmms-eval through the ``lmms_eval.models`` entry
point declared in ``pyproject.toml`` (``kempnerforge_vlm =
"kempnerforge.eval.vlm.registry:MANIFEST"``). Only ``chat_class_path`` is set:
the adapter is chat-only (``is_simple = False``), so resolution stays correct
even under ``force_simple``. The entry point is metadata only and does not make
lmms-eval a runtime dependency of KempnerForge.
"""

from lmms_eval.models.registry_v2 import ModelManifest

MANIFEST = ModelManifest(
    model_id="kempnerforge_vlm",
    chat_class_path="kempnerforge.eval.vlm.adapter.KempnerForgeVLM",
)
