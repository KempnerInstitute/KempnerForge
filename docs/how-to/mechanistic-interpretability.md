# Extract activations for interpretability

Capture intermediate tensors from a trained checkpoint, save them to
`.npz`, and feed them into a downstream analysis (linear probes, CKA,
SVCCA, dictionary learning). KempnerForge ships the extraction plumbing
in [`kempnerforge/model/hooks.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/hooks.py);
the analysis itself is left to whatever external tooling you prefer.

This page covers the three entry points — `ActivationStore`,
`extract_representations`, and `save_activations` — and the workflow
around them.

## The three APIs

| API | When to use |
|-----|-------------|
| [`ActivationStore`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/hooks.py) | You drive the forward passes yourself (custom prompts, interactive exploration, per-sample inspection). |
| [`extract_representations()`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/hooks.py) | You have a `Dataset` and want one big tensor per layer across N samples. |
| [`save_activations()`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/hooks.py) | Persist the result to `.npz` so analysis runs out-of-process. |

`extract_representations` is built on top of `ActivationStore`. If the
built-in loop doesn't match your needs (e.g., you want per-sample
activations keyed by prompt ID, or you need to batch across different
model states), use `ActivationStore` directly.

## Pick the layers

Module FQNs use `nn.ModuleDict` keys — index into
[`Transformer.layers`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/transformer.py)
with a string key, then into submodules:

```
layers.0.attention          # attention output (pre-residual)
layers.0.mlp                # MLP output (pre-residual)
layers.0.attention_norm     # pre-attention RMSNorm output
layers.0                    # full block output (post-residual)
layers.5.mlp                # e.g. layer 5's MLP output
norm                        # final RMSNorm before output head
output_head                 # lm head logits (batch, seq, vocab)
token_embedding             # input embedding (batch, seq, dim)
```

Confirm what's actually there for your checkpoint by dumping the
named modules:

```python
for name, _ in model.named_modules():
    print(name)
```

A 7B Llama-3 has `n_layers = 32`, so you get `layers.0` through
`layers.31`. The block's substructure is `attention_norm`, `attention`,
`mlp_norm`, `mlp`.

## Quick path: dataset in, `.npz` out

```python
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.utils.data import Subset

from kempnerforge.config.loader import load_config
from kempnerforge.config.registry import registry
from kempnerforge.data.dataset import MemoryMappedDataset
from kempnerforge.model.hooks import extract_representations, save_activations

device = torch.device("cuda")
dtype = torch.bfloat16
config = load_config("configs/train/7b.toml")

# Build model and load a DCP checkpoint — same pattern scripts/generate.py uses.
model_builder = registry.get_model(config.model.model_type)
model = model_builder(config.model).to(device=device, dtype=dtype)

ckpt_path = Path("checkpoints/7b/step_50000")
state_dict = {"model": model.state_dict()}
dcp.load(state_dict, checkpoint_id=str(ckpt_path))
model.load_state_dict(state_dict["model"])

dataset = MemoryMappedDataset(
    data_dir=config.data.dataset_path,
    file_pattern=config.data.file_pattern,
    seq_len=config.train.seq_len,
)
subset = Subset(dataset, list(range(2048)))   # bound the sample count

acts = extract_representations(
    model=model,
    dataset=subset,
    layers=["layers.0.mlp", "layers.15.mlp", "layers.31.mlp"],
    device=device,
    batch_size=16,
    max_samples=2048,
)

# acts["layers.15.mlp"] has shape (2048, seq_len, dim) on CPU.
# Cast to float32 first if the model ran in bf16 — see dtype caveat below.
save_activations({k: v.float() for k, v in acts.items()}, "acts/7b_step50k_mlp.npz")
```

`extract_representations` does the right thing automatically: sets
`eval()` mode, runs under `torch.no_grad()`, captures to CPU per batch,
and restores the original training flag on exit. The `.npz` is a single
file indexed by layer name, loadable from any numpy-aware downstream
tool.

Replace `MemoryMappedDataset` with a `HuggingFaceDataset` if you want
streaming — any map-style `Dataset` that returns `{"input_ids": ...}`
works.

## Fine-grained: `ActivationStore` directly

When you need interactive control — custom prompts, inspection between
passes, or state that doesn't fit the dataset model — drive it yourself:

```python
from kempnerforge.model.hooks import ActivationStore

store = ActivationStore(
    model,
    layers=["layers.0.attention", "layers.15.mlp", "norm"],
)
store.enable()

with torch.no_grad():
    prompt_a = tokenizer.encode("The quick brown fox", return_tensors="pt").to(device)
    model(prompt_a)
    act_a = {name: store.get(name).clone() for name in store.layer_names}

    store.clear()
    prompt_b = tokenizer.encode("The lazy dog", return_tensors="pt").to(device)
    model(prompt_b)
    act_b = {name: store.get(name).clone() for name in store.layer_names}

store.disable()
```

Four rules to remember:

- `enable()` / `disable()` register and remove the forward hooks. The
  class also removes hooks from `__del__`, but relying on GC is fragile
  — always disable explicitly.
- `get(name)` returns the **last** captured activation for that layer.
  Every new forward pass overwrites it. Call `clone()` if you need to
  keep it across passes.
- `clear()` empties the captured dict but keeps hooks registered.
- Captured tensors are already on CPU (the hook calls
  `output.detach().cpu()`). You can move them back with `.to(device)`
  for in-place analysis without needing `.clone()`.

### Tuple outputs

A handful of modules return `(output, aux)` tuples instead of a single
tensor. The hook handles this — it captures `output[0]`. If a module
you're targeting returns something odder (a `dataclass`, a `dict`, a
nested tuple), the store silently captures nothing. Verify with:

```python
store.enable()
model(prompt)
assert all(store.get(n) is not None for n in store.layer_names), \
    "Some layer's forward returns a non-tensor — wrap it or target a child module."
```

## Performance and memory

Activation extraction is expensive. A 7B model at `seq_len=4096`,
`dim=4096`, batch 16, float32, one layer's output = `16 × 4096 × 4096 ×
4 B = 1 GB`. Across 32 layers for 2 k samples, you're at
256 GB — well beyond host RAM. Keep the capture budget bounded:

- **Pick ≤ 3–5 layers.** Full-network sweeps are rarely the right first
  experiment. Start with early / middle / late, expand from there.
- **Cap `max_samples`.** A few thousand samples is enough for CKA or
  probe training; tens of thousands is research-grade.
- **Store as bf16 or float16 after capture** if you can — the hooks
  capture whatever dtype the forward produces, which is
  `bf16` during a normal `eval()` on a bf16-compiled model. See
  dtype caveat below.
- **Subsample the sequence dimension.** If you only care about the
  final token, slice `acts[name][:, -1]` before saving.

The CPU transfer is synchronous per batch inside the hook, so there's
some overhead vs a hookless eval run — budget for ~1.5× eval wallclock.

### Dtype caveat for `.npz`

`save_activations` calls `v.numpy()`, which fails on `torch.bfloat16`
(numpy has no bf16). If your model runs in bf16, cast before saving:

```python
acts_fp32 = {k: v.float() for k, v in acts.items()}
save_activations(acts_fp32, "acts/…npz")
```

Or save to `.pt` directly with `torch.save(acts, "acts/…pt")` if you
want to preserve the original dtype.

## Load activations for analysis

```python
import numpy as np

loaded = np.load("acts/7b_step50k_mlp.npz")
print(list(loaded.keys()))            # ['layers.0.mlp', 'layers.15.mlp', ...]
x = loaded["layers.15.mlp"]           # ndarray, shape (N, seq_len, dim)
```

From here, standard analysis is out-of-scope for this repo — but
common next steps:

- **Linear probing.** Flatten `(N, seq, dim) → (N*seq, dim)`, fit an
  `sklearn` logistic regression against labels to measure how much of
  a property is decodable from that layer.
- **CKA / SVCCA.** Compare two sets of activations (e.g., same inputs
  across a fine-tune vs base model) to quantify representational
  similarity per layer. External libraries like
  [`anatome`](https://github.com/moskomule/anatome) consume `.npz`
  directly.
- **Sparse autoencoders / dictionary learning.** Feed MLP or residual
  stream activations into an SAE training run; the `.npz` is a
  drop-in substitute for forwarding the model during SAE training.

## Limitations

**Output only.** The hook uses `register_forward_hook`, which captures
a module's *output*. You cannot:

- Capture attention weights (they're internal to SDPA; not exposed as
  a module output). You'd need to rewrite
  [`Attention.forward`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/attention.py)
  to return them, or swap SDPA for an eager implementation.
- Capture gradients or activations during the backward pass — no
  `register_full_backward_hook` wrapper is provided. If you need
  gradient-based analysis (integrated gradients, attribution methods),
  extend `ActivationStore` or use PyTorch's hooks directly.

**Compiled models.** If the model was compiled with `torch.compile`,
forward hooks still fire — PyTorch preserves module-level hook
semantics through compilation — but the hook itself is not compiled, so
captures add eager-mode overhead.

**FSDP2 / TP.** Extraction is meant for a single-process inference-time
load (as shown in the quick path: build the model, `dcp.load` the
weights, run on one GPU). Running it on a fully-sharded model works
but captures per-rank shards of each tensor, which is almost never
what you want for analysis.

## Troubleshooting

**`ValueError: Module 'layers.0.attention' not found in model.`**
The FQN is wrong. The error message prints the first 20 available
names — check capitalization and the `layers.{idx}.` prefix (integer
keys are stringified by `ModuleDict`).

**`store.get(name)` returns `None`.** The hook never fired. Either the
forward pass didn't reach that layer (PP rank?), or the module returned
a non-tensor/non-tuple. See [Tuple outputs](#tuple-outputs).

**`RuntimeError: can't convert bfloat16 numpy()`.** `save_activations`
was called on a bf16 tensor. Cast with `.float()` first — see
[Dtype caveat](#dtype-caveat-for-npz).

**Host RAM OOM mid-run.** You're capturing too many layers × samples.
Cut `max_samples`, reduce `layers`, or slice the seq dim inside a
custom extraction loop before appending to `results`.

**Memory leak across repeated `extract_representations` calls.** Make
sure `ActivationStore.disable()` is being called. The helper does this
in a `finally` block, but if you're using the class directly and an
exception fires before `disable()`, the hooks stick around on the
model and every future forward captures into the stale store. Always
wrap manual use in `try / finally`.

## See also

- [Generate from checkpoint](generate-from-checkpoint.md) — the
  `registry.get_model` + `dcp.load` pattern used above to load a
  checkpoint into a single-GPU model.
- [Architecture § Model](../architecture/model.md) — the layer FQN
  layout (`layers.{i}.attention`, `layers.{i}.mlp`, etc.).
- [Data § Datasets](../data/memory-mapped.md) — the dataset class used
  in the quick-path example; any map-style dataset yielding
  `{"input_ids": ...}` works.
- [Training § Hooks](../training/hooks.md) — the unrelated
  *training-loop* hook interface (`TrainingHook`). Activation hooks
  observe the model's forward pass; `TrainingHook` observes the
  training loop.
- [`kempnerforge/model/hooks.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/hooks.py)
  — the ~180-line module that defines `ActivationStore`,
  `extract_representations`, and `save_activations`.
