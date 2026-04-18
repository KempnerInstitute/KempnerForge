# Configuration

Every run is driven by a typed `JobConfig` built from three layers:
dataclass defaults, a TOML preset, and `--section.key=value` CLI
overrides. All three flow through `load_config` in
[`kempnerforge/config/loader.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/loader.py).

```python
from kempnerforge.config import load_config

config = load_config("configs/train/debug.toml",
                     cli_args=["--model.dim=512", "--train.compile_model=false"])
config.validate(world_size=1)
```

`validate(world_size)` enforces cross-section rules that a single
dataclass can't check on its own (e.g. parallelism factors × world size,
`tie_embeddings` vs PP, FP8 vs TP, MoE vs PP).

## Layering

1. **Defaults** — every dataclass in
   [`kempnerforge/config/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/kempnerforge/config)
   declares field defaults that reproduce a 7B Llama-like run.
2. **TOML preset** — `configs/train/*.toml` overlays those defaults.
   Unknown keys raise `ValueError` (catches typos like `modell.dim`
   immediately, not at step 1000).
3. **CLI overrides** — each `--section.key=value` flag rewrites the
   layered dict. `ast.literal_eval` parses ints, floats, lists, tuples,
   and booleans; anything unparseable stays as a string.

## Pages

```{toctree}
:maxdepth: 1

config-sections
cli-overrides
validation-rules
registry
```

- **[Config sections](config-sections.md)** — `JobConfig` and its 10
  sub-configs, field by field.
- **[CLI overrides](cli-overrides.md)** — dotted-path syntax, booleans,
  lists, nested dataclasses.
- **[Validation rules](validation-rules.md)** — per-dataclass
  `__post_init__` checks plus `JobConfig.validate(world_size)`
  cross-section rules.
- **[Registry](registry.md)** — the seven component categories
  (`model`, `optimizer`, `scheduler`, `loss`, `norm`, `router`,
  `mlp`) and how to plug a new builder in.

## See also

- [Architecture § Design principles](../architecture/index.md#design-principles)
- [Parallelism order](../architecture/parallelism-order.md) — the
  invariants the cross-section validator enforces.
