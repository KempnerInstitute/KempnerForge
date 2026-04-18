# CLI overrides

Any `JobConfig` field reachable through dotted attributes can be
overridden from the command line with `--section.key=value`. Parsing
happens in `_parse_cli_overrides` in
[`kempnerforge/config/loader.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/loader.py);
type coercion happens in `_coerce_value` in the same file.

## Syntax

```bash
uv run python scripts/train.py configs/train/debug.toml \
    --model.dim=512 \
    --train.compile_model=false \
    --optimizer.lr=1e-4 \
    --optimizer.betas="[0.9,0.95]" \
    --distributed.tp=2
```

Each flag must begin with `--`. Parsing splits once on `=`:

- Left of `=` is the dotted path. It builds a nested dict:
  `--model.dim=512` becomes `{"model": {"dim": 512}}`.
- Right of `=` is the literal value. The loader calls
  `ast.literal_eval` on it. If that raises, the raw string is kept.

## Value parsing

`ast.literal_eval` accepts Python literals:

| CLI snippet | Parsed as |
|-------------|-----------|
| `--model.dim=512` | `int(512)` |
| `--optimizer.lr=3e-4` | `float(0.0003)` |
| `--train.compile_model=true` | `bool(True)` (via coercion — see below) |
| `--optimizer.betas=[0.9,0.95]` | `list → tuple(0.9, 0.95)` (coerced to tuple by field type) |
| `--checkpoint.load_path=/scratch/ckpt` | `str("/scratch/ckpt")` (literal_eval fails, keep as string) |
| `--data.anneal_weights={"c4":0.7,"books":0.3}` | `dict` (parsed into `DataConfig.anneal_weights`) |

Quote any value the shell would otherwise eat:

```bash
--optimizer.betas="[0.9,0.95]"   # list literal with spaces/brackets
--data.dataset_path="/n/holylfs06/.../shards"
```

Lists of dataclasses (`DataConfig.datasets`, `DataConfig.phases`) are
not ergonomic on the CLI — pass them in the TOML preset instead. The
loader will accept them if you really want to, by expanding each entry
to a dict literal, but the TOML version is readable:

```toml
[[data.datasets]]
path = "/scratch/c4"
weight = 0.7
name = "c4"

[[data.datasets]]
path = "/scratch/books"
weight = 0.3
name = "books"
```

## Boolean shorthand

A flag with no `=` becomes `True`:

```bash
--train.compile_model     # equivalent to --train.compile_model=true
```

For `False`, write it explicitly: `--train.compile_model=false`.

The bool coercion also accepts the strings `"true"`, `"1"`, `"yes"`
(case-insensitive) as `True`; everything else falls back to
`bool(value)`.

## Type coercion

After `ast.literal_eval` produces a raw value, `_coerce_value` walks
the dataclass type hint and coerces:

| Field type | Input | Result |
|------------|-------|--------|
| `int` | `"512"` or `3.0` | `int(…)` |
| `float` | `"3e-4"` or `1` | `float(…)` |
| `bool` | `"true"`, `"1"`, `"yes"` | `True`; else `bool(value)` |
| `tuple[float, float]` | `[0.9, 0.95]` | `(0.9, 0.95)` |
| `list[str]` | `"a"` | `["a"]` (wraps bare scalar) |
| `list[DatasetSource]` | `[{"path": …}]` | `[DatasetSource(path=…)]` (recursive) |
| `Literal["bf16", "fp16", "fp32", "fp8"]` | anything else | `ValueError` |
| `StrEnum` (`NormType`, `SchedulerType`, …) | `"rmsnorm"` | `NormType.rmsnorm` |
| `X \| None` (Optional) | `None` → `None`; else coerce to `X` |

`Literal` fields reject unknown strings at coercion time — a typo like
`--train.mixed_precision=bfloat16` (should be `"bf16"`) fails before
training starts.

## Unknown keys

`_apply_dict_to_dataclass` rejects keys not declared on the dataclass.
A typo surfaces immediately:

```bash
$ uv run python scripts/train.py configs/train/debug.toml --model.dimm=512
ValueError: Unknown config keys in ModelConfig: ['dimm'].
Valid keys: ['activation', 'dim', ...]
```

## Inspecting the final config

Print the resolved config before the training loop begins by adding
the flag to `scripts/train.py` or dumping it from a REPL:

```python
from kempnerforge.config import load_config
from dataclasses import asdict
import json

config = load_config("configs/train/debug.toml",
                     cli_args=["--model.dim=512"])
print(json.dumps(asdict(config), indent=2, default=str))
```

## See also

- [Config sections](config-sections.md) — every field you can override.
- [Validation rules](validation-rules.md) — what the loader checks
  after overrides land.
