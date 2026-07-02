"""VLM evaluation subsystem: an lmms-eval chat-model adapter over ``VLMWrapper``.

The adapter (``adapter.py``) and its registration manifest (``manifest.py``)
both import the optional ``lmms-eval`` package, so they are intentionally NOT
imported here. They are loaded only by ``scripts/vlm_eval_harness.py`` and by
lmms-eval's entry-point loader (at which point ``lmms-eval`` is necessarily
installed). Keeping this ``__init__`` import-free preserves
``import kempnerforge`` with ``lmms-eval`` absent.
"""
