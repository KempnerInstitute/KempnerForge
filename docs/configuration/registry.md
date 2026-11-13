# Registry

The component registry is a two-level map
`category → name → builder`, lives in
[`kempnerforge/config/registry.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/registry.py),
and is the single indirection that lets configs pick components by
string. Seven categories ship populated; you can add your own builder
under any of them (or create a new category) without touching the core
training loop.

```python
from kempnerforge.config.registry import registry

registry.register("mlp", "my_mlp", builder_fn)
model = registry.get("mlp", "my_mlp")(dim=4096, hidden_dim=11008)
```

`register(category, name, obj)` fails fast on duplicate names, with a
`ValueError` naming the module that registered the name and the module
registering it again, each with the module that defines the registered
function or class when that is another one (a shared helper or an
`__init_subclass__` hook registers for every plugin that uses it);
`get` raises `KeyError` with an "Available: […]" hint when a name is
missing; `list(category)` returns registered names; `unregister(category, name)`
removes an entry and returns it, raising the same `KeyError` when the name
is missing (an adapter's `accepts` declaration goes with it).

## The seven categories

| Category | Registered keys | Selected by config field |
|----------|-----------------|--------------------------|
| `model` | `transformer` | `model.model_type` |
| `optimizer` | `adamw`, `lion`, `muon`, `schedule_free_adamw` | `optimizer.name` |
| `scheduler` | `cosine`, `linear`, `wsd`, `constant`, `rex`, `none` | `scheduler.name` |
| `loss` | `cross_entropy`, `chunked_cross_entropy` | `train.loss_fn` |
| `norm` | `rmsnorm`, `layernorm` | `model.norm_type`, `adapter.pre_norm` |
| `router` | `softmax_topk`, `sigmoid_topk` | `model.moe_router` |
| `mlp` | `swiglu`, `standard_gelu`, `standard_relu` | `model.activation` (mapped) |

The `mlp` case is the only one that isn't a direct string match:
`build_mlp` in
[`kempnerforge/model/mlp.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/mlp.py)
maps `activation="silu"` → `swiglu`, `"gelu"` → `standard_gelu`,
`"relu"` → `standard_relu` before calling `registry.get("mlp", …)`,
and looks any other `activation` up as an `mlp` key directly.

`scheduler.name`, `model.norm_type` and `model.activation` are typed as
enums (`SchedulerType`, `NormType`, `Activation`), and a built-in name
loads as the enum member. The loader also accepts any other name
registered in the field's category (`scheduler`, `norm`, `mlp`) and
keeps it as a plain string; a name that is neither raises a
`ValueError` listing the members and the registered names. The
registration has to run before `load_config`, which is what the
`plugins` key below is for.

`model.activation` has two limits. Core's own `mlp` keys (`swiglu`,
`standard_gelu`, `standard_relu`) are rejected, with a pointer to the
activation that selects each (`silu`, `gelu`, `relu`), whatever was
imported first. And a plugin's MLP is rejected when `num_experts > 0`:
the MoE expert kernels (grouped GEMM, packed experts, and expert
parallelism, which uses both), MoT's included, compute core's SwiGLU or
standard MLP from the expert weights rather than calling the expert
module. MoMa's experts call each expert module, so a plugin's MLP runs
there as written.

## Consumers

Where each category is looked up in the codebase:

| Category | Call site |
|----------|-----------|
| `model` | `registry.get_model(model_config.model_type)` in [`kempnerforge/distributed/parallel.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py) |
| `optimizer` | `registry.get_optimizer(config.name)` in [`kempnerforge/training/optimizer.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/optimizer.py) |
| `scheduler` | `registry.get_scheduler(name)` in [`kempnerforge/training/scheduler.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/scheduler.py) |
| `loss` | `registry.get_loss(config.loss_fn)` in [`kempnerforge/training/loss.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/loss.py) |
| `norm` | `registry.get("norm", norm_type)` in [`kempnerforge/model/norm.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/norm.py) |
| `router` | `registry.get("router", router_type)` in [`kempnerforge/model/moe.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/moe.py) |
| `mlp` | `registry.get("mlp", key)` in [`kempnerforge/model/mlp.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/mlp.py) |

## Registering a new builder

Four of the seven categories provide a decorator (`register_model`,
`register_optimizer`, `register_scheduler`, `register_loss`). The other
three are registered with `register("category", "name", fn)` at
module-import time. Both routes land in the same `_stores` dict.

### With a decorator

```python
from kempnerforge.config.registry import registry

@registry.register_optimizer("my_adam")
def build_my_adam(param_groups, config, model):
    return MyAdam(param_groups, lr=config.lr, betas=config.betas)
```

Configs then pick it up via `[optimizer] name = "my_adam"`. `build_optimizer`
passes every optimizer builder the decay/no-decay `param_groups`, the
`OptimizerConfig` and the model, so a builder can route parameters by the
module that owns them.

### With `register(...)`

```python
from kempnerforge.config.registry import registry
from kempnerforge.model.norm import RMSNorm

def _build_my_norm(dim, eps):
    return MyNorm(dim, eps=eps)

registry.register("norm", "my_norm", _build_my_norm)
```

Configs then pick it up via `[model] norm_type = "my_norm"`.

### Adapters declare the fields they use

`build_adapter` passes an adapter builder every `AdapterConfig` field but
`type`, plus `norm_eps`. The builder names the fields it uses as parameters
and swallows the rest with `**_`, and that signature is its declaration:
`AdapterConfig` rejects any other field set away from its default, so a
config cannot ask for a setting the adapter would drop.

```python
from kempnerforge.config.registry import registry

@registry.register_adapter("my_pool")
def build_my_pool(in_dim, out_dim, pool_window=2, **_):
    return MyPool(in_dim, out_dim, pool_window)
```

`my_pool` uses `pool_window`, so `type = "my_pool"` with `pool_heads = 8`
fails to load. A parameter counts when it is named after an `AdapterConfig`
field and can be passed by keyword; a positional-only one never receives the
field. A builder with no `**` parameter at all is read the same way. The
builder's own signature counts: a wrapper made with `functools.wraps` is read
as the wrapper, not as the function it wraps, since the wrapper is what
receives the fields.

Pass `accepts` to `register_adapter` when the signature does not tell the
whole story. When the fields in use depend on other fields, `accepts` is a
callable that takes the `AdapterConfig` and returns them as a collection of
names, never a bare string (that fails when the config loads). The built-in
`avgpool` works this way: it uses `activation` and `pre_norm` only once
`hidden_dim > 0` builds its MLP head.

```python
def _my_head_accepts(config):
    if config.hidden_dim > 0:
        return ("hidden_dim", "activation")
    return ("hidden_dim",)

@registry.register_adapter("my_head", accepts=_my_head_accepts)
def build_my_head(in_dim, out_dim, hidden_dim=None, activation="gelu", **_):
    return MyHead(in_dim, out_dim, hidden_dim, activation)
```

`accepts` can also be a plain collection of field names, such as
`accepts=("pool_window",)`. A builder whose `**` parameter has any other name
(`**kwargs`) may read fields from it, so its signature says nothing: without
`accepts` that adapter accepts every field, and its builder receives them
all. A declaration that names a field `AdapterConfig` does not have fails
when the config loads.

### Making sure the module runs

Registration happens at import. If nothing imports the module that
calls `registry.register`, the entry never lands. Typical approach:

- Put the new builder in a module under
  `kempnerforge/<subsystem>/`, and add an import in that package's
  `__init__.py` (or the existing module the training loop already
  pulls in). Existing example:
  [`kempnerforge/model/norm.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/norm.py)
  registers at module scope, and
  [`kempnerforge/model/__init__.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/__init__.py)
  imports it so the registration runs.
- Third-party code (not landed in `kempnerforge/`) can import its
  registration module explicitly before calling `load_config` or
  `build_parallel_model`, or declare it in the `plugins` config key
  below.

### The `plugins` config key

`plugins` is a top-level list of module paths that `load_config` imports
*before* the TOML is applied to the dataclasses:

```toml
plugins = ["my_project.components"]

[video]
dataset_type = "my_corpus"   # registered by the module above
```

That ordering is the point: the `__post_init__` validators that resolve
a name against the registry (`video.dataset_type`, `adapter.type`,
`vision_encoder.type`, …) and the loader's check of `scheduler.name`,
`model.norm_type` and `model.activation` run during the overlay, so a
later import would be too late. It keeps experiments standalone: a run
registers its own components without any core module importing them.

Modules have to be on `sys.path`. The launch directory is *not* — a
module that sits next to the run needs `PYTHONPATH=<its dir>`. A module
inside the repository resolves because a `uv sync` editable install puts
the repo root on `sys.path`; anything outside the repo must be installed
or on `PYTHONPATH`. A missing or failing module raises a
`ValueError` naming it, and `--plugins=["pkg.mod"]` overrides the TOML
list like any other field.

A plugin that registers a name another module already registered fails
the same way, and the error names both modules. Some categories are
filled only when core first imports the module that registers them, so
a plugin imported earlier takes the name, and the error is raised at
that later import instead, still naming the plugin as the owner.

Four rules for plugin authors, all from the distributed contract:

- **Register the same thing on every rank.** `load_config` runs on every
  rank before `init_distributed`, so all ranks import the same list in
  the same order. Never branch registration on rank, host, or cwd —
  ranks would build different modules and the DCP load would mismatch.
- **Do not consume global RNG at import time.** Seeding happens just
  after the config load, so today it is harmless, but it is not a
  guarantee to lean on.
- **No blocking I/O or collectives at import time.** The import runs
  before `init_distributed`, so a rank stalled reading a manifest never
  reaches the rendezvous — the other ranks then fail there, minutes
  later and far from the cause.
- **Keep `plugins` in the config you resume with.** The list is not
  recorded in the checkpoint. Resuming with a list that builds a
  different component fails the load with `Missing key in checkpoint
  state_dict` rather than training a mismatched model — but a component
  whose parameters are a strict subset of the saved ones would load
  silently.

## Adding a new category

There's nothing special about the seven — `register(category, name,
obj)` creates the store on first use:

```python
registry.register("reward_model", "pair_rm", build_pair_rm)
builder = registry.get("reward_model", "pair_rm")
```

The convenience `register_*` / `get_*` methods on `Registry` are just
sugar over the generic `register` / `get`. Add your own if it reads
better.

## Signatures

The registry is untyped — `register(category, name, obj)` accepts any
callable. The de-facto contract is "the call site expects a specific
signature, and your builder has to match it." Existing examples:

| Category | Expected signature |
|----------|--------------------|
| `model` | `fn(config: ModelConfig) → nn.Module` |
| `optimizer` | `fn(param_groups, config: OptimizerConfig, model: nn.Module) → torch.optim.Optimizer` |
| `scheduler` | `fn(optimizer, config: SchedulerConfig, max_steps: int) → LRScheduler` |
| `loss` | `fn(logits, labels, …) → Tensor` (see existing `cross_entropy` for the exact kwargs) |
| `norm` | `fn(dim: int, eps: float) → nn.Module` |
| `router` | `fn(dim, num_experts, top_k, **kwargs) → nn.Module` |
| `mlp` | `fn(dim: int, hidden_dim: int) → nn.Module` |
| `adapter` | `fn(in_dim: int, out_dim: int, norm_eps: float, **fields) → VisionAdapter`, where `fields` is `AdapterConfig.extra_kwargs()`, every field but `type` |

Check the closest existing builder (e.g.
[`_build_swiglu`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/mlp.py))
before writing a new one — matching its signature exactly is the
fastest path to correctness.

## See also

- [Config sections](config-sections.md) — which config fields resolve
  to which registry category.
- [Architecture § Model](../architecture/model.md#registry-keys) — the
  per-component registry lookup table from the model side.
