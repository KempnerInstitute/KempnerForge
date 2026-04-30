---
name: install-and-verify
description: Run `uv sync` and the four CI gate checks (ruff check, ruff format, pyright, pytest unit). First command after cloning. Auto-handles non-CUDA hosts via `--no-sources`.
---

## When to use
- Right after `git clone`, before any other skill.
- Optionally after `git pull` if the user asks for it.
- When CI fails and you want to reproduce the four PR-gate checks locally.

Run every step in the background so the user isn't blocked — in Claude Code, set `run_in_background=true` on each Bash call and report status as each completes.

## Preflight

1. Detect the host. The torch wheel pin in `pyproject.toml` is CUDA-only:

       command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && echo CUDA || echo NO_CUDA

   On `NO_CUDA`, **print this warning to the user verbatim** before proceeding, and append `--no-sources` to every `uv run` and `uv sync` invocation in this skill (the flag is per-command, not sticky):

   > No CUDA GPU detected. Using `--no-sources` to fetch torch from PyPI instead of the CUDA wheel index. Sufficient for the dev loop (lint, type-check, CPU unit tests); training and GPU tests still need a Linux GPU box. Note: `--no-sources` also bypasses any other `[tool.uv.sources]` entries the project may add later.

2. Baseline check (use the form matching your host):

       uv run python scripts/check_env.py                  # CUDA
       uv run --no-sources python scripts/check_env.py     # non-CUDA

   Non-zero exit: print stdout verbatim and stop. Common: install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`) or `cd` into the KempnerForge checkout.

## Context (auto-generated, do not edit)
<!-- context-begin -->
Lint scope: kempnerforge/ tests/ scripts/
Type-check scope: kempnerforge/
Dev group: ruff, pyright[nodejs], pytest, pytest-timeout, vulture
GPU test paths (NOT here): tests/integration/, tests/distributed/, tests/e2e/
Non-CUDA fallback: uv sync --no-sources (bypasses [tool.uv.sources] torch CUDA pin)
<!-- context-end -->

## Procedure
On non-CUDA hosts, append `--no-sources` to every `uv` command below.

1. Sync:

       uv sync

   Expected: ~30 s cached, 2–5 min cold.

2. Confirm Python ≥ 3.12:

       uv run python -c "import sys; assert sys.version_info >= (3, 12), sys.version_info; print(sys.version_info[:3])"

3. Lint:

       uv run ruff check kempnerforge/ tests/ scripts/

4. Format check:

       uv run ruff format --check kempnerforge/ tests/ scripts/

5. Type check:

       uv run pyright kempnerforge/

6. Unit tests (CPU-only):

       uv run pytest tests/unit/ -v --timeout=60

   Expected: ~890 tests in 10–60 s.

7. Report each step OK / FAIL. Next:
   - SLURM login node → `/kempnerforge:cluster-config`
   - GPU box → `/kempnerforge:smoke-test`
   - Non-CUDA dev machine → re-state the warning so the user remembers this venv is dev-only.

## Gotchas
- `--no-sources` is per-command. Forgetting it on one call → that call fails with the torch wheel error.
- Stale `.venv` from a different Python or torch flavor → `rm -rf .venv && uv sync`. Symptom: tests run on the wrong Python or pyright diverges from CI.
- `tests/integration/` / `tests/distributed/` / `tests/e2e/` are GPU-only; skipped here. Use `/kempnerforge:smoke-test` for the GPU path.
- The verify quartet mirrors `.github/workflows/ci.yml`. Local-vs-CI divergence: check `uv --version` against `astral-sh/setup-uv@v6`.

## Related skills
- `/kempnerforge:cluster-config` — first cluster onboarding step.
- `/kempnerforge:smoke-test` — short single-GPU training to validate end-to-end.
