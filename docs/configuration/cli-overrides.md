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
| `SchedulerType`, `NormType`, `Activation` | a registered name | the name, as a `str` |
| `X \| None` (Optional) | `None` → `None`; else coerce to `X` |

`Literal` fields reject unknown strings at coercion time — a typo like
`--train.mixed_precision=bfloat16` (should be `"bf16"`) fails before
training starts. `"fp16"` passes this check on purpose and is then
rejected by `TrainConfig.__post_init__`, whose error explains that
fp16 would train with no loss scaling.

Three enums select a registry builder, so they also accept a name
registered in their category: `scheduler.name` (`SchedulerType`,
category `scheduler`), `model.norm_type` (`NormType`, `norm`) and
`model.activation` (`Activation`, `mlp`). A plugin module in `plugins`
registers its names before the overrides apply, so
`--scheduler.name=my_sched` selects a scheduler it registered. A name
that is neither a member nor registered raises a `ValueError` that
lists both. The other enums (`train.activation_checkpointing`,
`distributed.pp_schedule`, `checkpoint.async_mode`) accept only their
members. `model.activation` also rejects core's own `mlp` keys
(`swiglu`, `standard_gelu`, `standard_relu`) in favor of `silu`, `gelu`
and `relu`, and rejects a plugin's MLP when `num_experts > 0`, since the
MoE expert kernels compute core's MLPs from the expert weights.

## Unknown keys

`_apply_dict_to_dataclass` rejects keys not declared on the dataclass.
A typo surfaces immediately:

```bash
$ uv run python scripts/train.py configs/train/debug.toml --model.dimm=512
ValueError: Unknown config keys in ModelConfig: ['dimm'].
Valid keys: ['activation', 'dim', ...]
```

`--vlm.arch` is the one field an override cannot always change. Each
arch has its own `[vlm]` config class, picked when the TOML's `[vlm]`
section is built, and an override keeps that class; so when the TOML
has a `[vlm]` section, `--vlm.arch` naming another arch is rejected with
a `ValueError` that says to set `arch` in the TOML instead.

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
