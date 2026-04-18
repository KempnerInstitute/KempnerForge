# Build a model

KempnerForge builds a `Transformer` from a `ModelConfig` via a
component registry. Seven sub-component categories are swappable by
name — `mlp`, `router`, `norm`, `model`, `optimizer`, `scheduler`,
`loss`. This page walks through the common cases: choose the
activation, flip on MoE, toggle QK-norm, and register your own MLP
variant.

## The minimal model

```python
from kempnerforge.config import load_config
from kempnerforge.distributed import init_distributed
from kempnerforge.distributed.parallel import build_parallel_model

config  = load_config("configs/train/debug.toml")
mesh    = init_distributed(config.distributed)
model   = build_parallel_model(config.model, device="cuda", device_mesh=mesh)
```

[`build_parallel_model`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py)
looks up `config.model.model_type` (default `"transformer"`) in the
`model` registry, constructs a `Transformer`, and applies parallelism
in the canonical
[5-step order](../architecture/parallelism-order.md). The
`Transformer` itself is composition:

```
TokenEmbedding
→ [TransformerBlock × n_layers]          # attention + MLP (or MoE)
→ RMSNorm
→ OutputHead (tied or separate)
```

Every sub-component it uses is pulled from a registry at import time.

## Registry categories

```python
# Every category follows the same register(cat, name, fn) pattern
# kempnerforge/config/registry.py
registry.register("norm",      "rmsnorm",    _build_rmsnorm)
registry.register("norm",      "layernorm",  _build_layernorm)
registry.register("mlp",       "swiglu",     _build_swiglu)
registry.register("mlp",       "standard_gelu", _build_standard_gelu)
registry.register("mlp",       "standard_relu", _build_standard_relu)
registry.register("router",    "softmax_topk",  _build_softmax_topk)
registry.register("router",    "sigmoid_topk",  _build_sigmoid_topk)
registry.register_model("transformer")(Transformer)
# + optimizer, scheduler, loss (see Registry reference)
```

Full list and the registration mechanics:
[Registry](../configuration/registry.md).

## Swap the activation / MLP flavor

```toml
[model]
activation = "silu"    # "silu" (SwiGLU), "gelu", or "relu"
```

The activation string maps to an MLP registry key:

| `activation` | MLP class | Projections |
|--------------|-----------|:-----------:|
| `silu` | `SwiGLUMLP` | 3 (gate + up + down) |
| `gelu` | `StandardMLP` | 2 (up + down) |
| `relu` | `StandardMLP` | 2 (up + down) |

```python
# kempnerforge/model/mlp.py — build_mlp
_ACTIVATION_TO_MLP = {"silu": "swiglu", "gelu": "standard_gelu", "relu": "standard_relu"}

def build_mlp(dim, hidden_dim, activation="silu"):
    key = _ACTIVATION_TO_MLP.get(activation, activation)
    return registry.get("mlp", key)(dim, hidden_dim)
```

No other config knob decides the MLP — change `activation` and the
right class gets built.

## Flip a model to MoE

```toml
[model]
num_experts   = 8     # >0 enables MoE layers
moe_top_k     = 2
moe_router    = "sigmoid_topk"   # or "softmax_topk"
moe_frequency = 1                # every layer; 2 = alternating dense/MoE
```

`TransformerBlock.__init__` picks MoE per-layer based on
`config.is_moe` and `moe_frequency`:

```python
# kempnerforge/model/transformer.py — TransformerBlock.__init__
use_moe = config.is_moe and ((layer_idx + 1) % config.moe_frequency == 0)
if use_moe:
    self.mlp = build_moe(...)
else:
    self.mlp = build_mlp(...)
```

With `moe_frequency = 2`, odd-indexed layers (1, 3, 5…) are MoE and
the rest are dense. See [MoE § overview](../moe/index.md).

## Toggle QK-norm

```toml
[model]
qk_norm = true    # Gemma / DeepSeek-V3 style per-head RMSNorm on Q/K before RoPE
```

When set, `Attention.__init__` adds `self.q_norm` and `self.k_norm`
(RMSNorm on `head_dim`) and applies them per-head before RoPE:

```python
# kempnerforge/model/attention.py — Attention.__init__
if qk_norm:
    self.q_norm = RMSNorm(head_dim)
    self.k_norm = RMSNorm(head_dim)
```

Default is `false` (plain Llama / Mixtral). Flip it on for
reproductions of Gemma or DeepSeek-V3.

## Swap the norm

```toml
[model]
norm_type = "rmsnorm"    # or "layernorm"
norm_eps  = 1e-5
```

Used by both the attention pre-norm and the MLP pre-norm on every
block, plus the final norm before the output head. Same two keys
(`rmsnorm`, `layernorm`) — flip via config, no code change.

## Register a new MLP variant

All four component categories pulled from the registry at block
construction (`mlp`, `router`, `norm`, `model`) accept new entries at
import time. Example — a gated MLP with a custom gate function:

```python
# my_project/mlp_variants.py
import torch.nn as nn
from kempnerforge.config import registry

class MyCustomMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.up   = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        # ... whatever you want ...

    def forward(self, x):
        return self.down(self.up(x).tanh())   # or your gate

def _build_my_mlp(dim: int, hidden_dim: int) -> MyCustomMLP:
    return MyCustomMLP(dim, hidden_dim)

registry.register("mlp", "my_mlp", _build_my_mlp)
```

Import it before training starts (in `scripts/train.py` or a plugin
module), then:

```toml
[model]
activation = "my_mlp"   # matches the registry key
```

The activation-name → MLP-key mapping covers `silu`/`gelu`/`relu`; any
other string is looked up directly in the `"mlp"` category, so your
key just works.

```{note}
The registry is a global singleton; `registry.get("mlp", "my_mlp")`
raises `ValueError` if the module containing the `register()` call
wasn't imported. Import it at the top of `scripts/train.py` or in your
package's `__init__.py`.
```

## What's *not* registered

`TransformerBlock` itself is **hardcoded**, not registered. There's no
`"block"` category. If you want a new block variant (e.g., parallel
attention+MLP à la PaLM, or a different residual pattern), you fork
`kempnerforge/model/transformer.py` and edit `TransformerBlock`.

Similarly, attention is hardcoded — the QK-norm toggle is the only
attention variant reachable from config.

This is deliberate: block-level topology changes are rare and touch
the forward signature (e.g., what gets passed between attention and
MLP), so a config toggle would leak into every block. If you find
yourself wanting a new block registered, open an issue — it's a
design question, not a quick registry addition.

## Full model-construction path

```
load_config()  → JobConfig with a ModelConfig
    ↓
build_parallel_model(model_config, device, mesh, ...)
    ├─ registry.get_model(config.model_type)       → Transformer
    ├─ Transformer(config)
    │     └─ TransformerBlock(config, layer_idx)   × n_layers
    │           ├─ build_norm(norm_type, dim)      ← registry["norm"]
    │           ├─ Attention(..., qk_norm=...)     ← hardcoded class
    │           └─ MLP per block (choice by config.is_moe and moe_frequency):
    │                 build_mlp(activation, ...)   ← registry["mlp"]
    │              OR build_moe(router_type, ...)  ← registry["router"] for the
    │                                                router + registry["mlp"] for
    │                                                each expert
    ├─ apply_tensor_parallel(...)   [if tp > 1]
    ├─ apply_expert_parallel(...)   [if ep > 1]
    ├─ apply_float8(...)            [if mixed_precision == "fp8"]
    ├─ apply_ac(...)                [if activation_checkpointing]
    └─ fully_shard(...)             [FSDP2]
```

Every box that isn't "hardcoded class" goes through a registry lookup.
Pipeline parallelism (not shown) is applied outside this flow — it
splits the model into stages in `scripts/train.py` *before*
`build_parallel_model` runs on each stage.

## See also

- [Registry](../configuration/registry.md) — the singleton itself,
  registration semantics, `get_model` / `get_optimizer` shortcuts.
- [Configuration § `[model]`](../configuration/config-sections.md) —
  every `ModelConfig` field with its default.
- [Architecture § Model](../architecture/model.md) — what the
  Transformer actually does at forward time.
- [Parallelism order](../architecture/parallelism-order.md) — what
  `build_parallel_model` does after it constructs the `Transformer`.
- [MoE § overview](../moe/index.md) — the full MoE wiring when
  `num_experts > 0`.
