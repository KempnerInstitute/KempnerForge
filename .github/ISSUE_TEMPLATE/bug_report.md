---
name: Bug report
about: Report a reproducible issue
title: ''
labels: bug
---

## What happened
<!-- Describe what went wrong vs what you expected -->

## Reproduction
<!-- Config file and exact command -->
```bash
uv run python scripts/train.py configs/train/<file>.toml --<overrides>
```

## Error / traceback
<!-- Full error output, not a screenshot -->
```
```

## Environment
- GPU type and count:
- Nodes:
- PyTorch version: `python -c "import torch; print(torch.__version__)"`
- KempnerForge commit: `git rev-parse HEAD`
