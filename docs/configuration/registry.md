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

`register(category, name, obj)` fails fast on duplicate names; `get`
raises `KeyError` with an "Available: […]" hint when a name is missing;
`list(category)` returns registered names.

## The seven categories

| Category | Registered keys | Selected by config field |
|----------|-----------------|--------------------------|
| `model` | `transformer` | `model.model_type` |
| `optimizer` | `adamw`, `lion`, `muon`, `schedule_free_adamw` | `optimizer.name` |
| `scheduler` | `cosine`, `linear`, `wsd`, `constant`, `rex`, `none` | `scheduler.name` |
| `loss` | `cross_entropy`, `chunked_cross_entropy` | `train.loss_fn` |
| `norm` | `rmsnorm`, `layernorm` | `model.norm_type` |
| `router` | `softmax_topk`, `sigmoid_topk` | `model.moe_router` |
| `mlp` | `swiglu`, `standard_gelu`, `standard_relu` | `model.activation` (mapped) |

The `mlp` case is the only one that isn't a direct string match:
`build_mlp` in
[`kempnerforge/model/mlp.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/mlp.py)
maps `activation="silu"` → `swiglu`, `"gelu"` → `standard_gelu`,
`"relu"` → `standard_relu` before calling `registry.get("mlp", …)`.

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
def build_my_adam(params, config):
    return MyAdam(params, lr=config.lr, betas=config.betas)
```

Configs then pick it up via `[optimizer] name = "my_adam"`.

### With `register(...)`

```python
from kempnerforge.config.registry import registry
from kempnerforge.model.norm import RMSNorm

def _build_my_norm(dim, eps):
    return MyNorm(dim, eps=eps)

registry.register("norm", "my_norm", _build_my_norm)
```

Configs then pick it up via `[model] norm_type = "my_norm"`.

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
  `build_parallel_model`.

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
| `optimizer` | `fn(params, config: OptimizerConfig) → torch.optim.Optimizer` |
| `scheduler` | `fn(optimizer, config: SchedulerConfig, max_steps: int) → LRScheduler` |
| `loss` | `fn(logits, labels, …) → Tensor` (see existing `cross_entropy` for the exact kwargs) |
| `norm` | `fn(dim: int, eps: float) → nn.Module` |
| `router` | `fn(dim, num_experts, top_k, **kwargs) → nn.Module` |
| `mlp` | `fn(dim: int, hidden_dim: int) → nn.Module` |

Check the closest existing builder (e.g.
[`_build_swiglu`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/mlp.py))
before writing a new one — matching its signature exactly is the
fastest path to correctness.

## See also

- [Config sections](config-sections.md) — which config fields resolve
  to which registry category.
- [Architecture § Model](../architecture/model.md#registry-keys) — the
  per-component registry lookup table from the model side.
