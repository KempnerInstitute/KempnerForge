## Summary
<!-- Bullet points of what this PR does -->

## Testing
- [ ] `uv run ruff check kempnerforge/ tests/` passes
- [ ] `uv run ruff format --check kempnerforge/ tests/ scripts/` passes
- [ ] `uv run pyright kempnerforge/` passes (0 errors)
- [ ] `uv run pytest tests/unit/ -v --timeout=60` passes
- [ ] If distributed code changed: `uv run torchrun --nproc_per_node=4 -m pytest tests/distributed/ -v`
- [ ] If training loop / parallelism / optimizers changed: `uv run pytest tests/e2e/ --e2e -v`

Closes #
