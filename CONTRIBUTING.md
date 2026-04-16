# Contributing to KempnerForge

## Environment Setup

### Prerequisites

- Python >= 3.12
- CUDA-capable GPU (for integration/distributed/e2e tests)
- [uv](https://docs.astral.sh/uv/) package manager

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and Install

```bash
git clone git@github.com:KempnerInstitute/KempnerForge.git
cd KempnerForge

# Install all dependencies (creates .venv automatically)
uv sync
```

### Verify Your Setup

Run all three checks that CI will run on your PR:

```bash
# Lint + format + type check
uv run ruff check kempnerforge/ tests/
uv run ruff format --check kempnerforge/ tests/ scripts/
uv run pyright kempnerforge/

# Unit tests (no GPU needed)
uv run pytest tests/unit/ -v --timeout=60
```

If all four pass, your environment is ready.

### GPU Access on SLURM Clusters

Unit tests run on CPU. Integration, distributed, and e2e tests require GPUs via SLURM.

```bash
# Interactive allocation — 1 node, 4 GPUs (for integration + distributed tests)
salloc --partition=<partition-name> --account=<account-name> \
  --nodes=1 --gpus-per-node=4 --cpus-per-task=16 --mem=256G --time=2:00:00

# Once allocated, run GPU tests inside the allocation:

# Integration tests (1 GPU)
srun --ntasks=1 --gpus-per-node=1 uv run pytest tests/integration/ -v

# Distributed tests (4 GPUs, via torchrun)
uv run torchrun --nproc_per_node=4 -m pytest tests/distributed/ -v

# E2E tests (4 GPUs, opt-in)
uv run pytest tests/e2e/ --e2e -v
```

## Contribution Workflow

### Step 1: Open an Issue

Every change starts with an issue. Open one on GitHub before writing code.

**Bug report — include:**
- What happened vs. what you expected
- Steps to reproduce (config file, command, error traceback)
- Environment: GPU type, node count, PyTorch version (`python -c "import torch; print(torch.__version__)"`)

**Feature request — include:**
- What the feature does and why it's needed
- Which config sections or modules it touches
- Whether it's backward compatible (existing configs should keep working)

**Example issue body (feature):**

```
Add WSD sqrt cooldown variant.

Currently `wsd_decay_type` supports cosine and linear. Sqrt cooldown
(lr * sqrt(1 - progress)) gives a gentler ramp-down useful for
long-context fine-tuning.

Touches: `kempnerforge/training/scheduler.py`, `kempnerforge/config/scheduler.py`.
Backward compatible — new option, existing configs unchanged.
```

### Step 2: Create a Branch

```bash
git checkout -b <category>/<short-description> main
```

Branch naming convention:

| Prefix | Use |
|--------|-----|
| `feat/` | New feature |
| `fix/` | Bug fix |
| `refactor/` | Code cleanup, no behavior change |
| `test/` | Adding or fixing tests |
| `docs/` | Documentation only |

Examples: `feat/context-parallelism`, `fix/checkpoint-resume-rank0`, `refactor/cleanup-eval-registry`.

### Step 3: Make Changes and Test

Write your code. Write tests. Then run the pre-push checklist:

```bash
# 1. Format (auto-fix)
uv run ruff format kempnerforge/ tests/ scripts/

# 2. Lint (auto-fix what it can)
uv run ruff check --fix kempnerforge/ tests/

# 3. Type check
uv run pyright kempnerforge/

# 4. Unit tests
uv run pytest tests/unit/ -v --timeout=60

# 5. If you changed distributed code, also run:
uv run torchrun --nproc_per_node=4 -m pytest tests/distributed/ -v

# 6. If you changed the training loop, optimizers, or parallelism, also run:
uv run pytest tests/e2e/ --e2e -v
```

### Step 4: Commit and Push

```bash
git add <files>
git commit -m "Add WSD sqrt cooldown scheduler variant"
git push -u origin feat/wsd-sqrt-cooldown
```

Commit message style:
- Imperative mood: "Add", "Fix", "Remove", "Update" (not "Added", "Fixes")
- Short first line (under 72 characters)
- Body for context if needed, but keep it brief

### Step 5: Open a Pull Request

Open a PR on GitHub targeting `main`. Use this structure:

```markdown
## Summary
- Add `sqrt` option to `scheduler.wsd_decay_type`
- Implement sqrt cooldown curve in `build_wsd_scheduler()`
- Add unit tests for the new decay curve

## Testing
- [ ] `uv run ruff check` passes
- [ ] `uv run ruff format --check` passes
- [ ] `uv run pyright kempnerforge/` passes
- [ ] `uv run pytest tests/unit/ -v` passes (N tests, 0 failures)
- [ ] `uv run pytest tests/e2e/ --e2e -v` passes (if applicable)
- [ ] Tested on N GPUs with config: `configs/train/debug.toml --scheduler.wsd_decay_type=sqrt`

Closes #42
```

Include `Closes #N` to auto-close the issue on merge.

**PR guidelines:**
- Keep PRs focused. One feature or fix per PR.
- If your change is large, break it into smaller PRs that each leave the codebase in a working state.
- Respond to review comments. Push follow-up commits (don't force-push during review).

## Code Style

- **Formatter/linter:** ruff, 100-character line length, Python 3.12 target.
- **Naming:** `snake_case` everywhere. Module names match their primary class/function.
- **Imports:** sorted by ruff (`isort` rules). `kempnerforge` is first-party.
- **Type annotations:** used throughout. Pyright runs in CI with zero errors — keep it that way.
- **Comments:** only where the logic isn't self-evident. No docstrings on obvious methods.

```bash
# Auto-fix lint issues
uv run ruff check --fix kempnerforge/ tests/

# Auto-format
uv run ruff format kempnerforge/ tests/ scripts/

# Type check (must be zero errors)
uv run pyright kempnerforge/
```

## CI Pipeline

CI runs on every push to `main` and every PR. All jobs must pass before merge.

| Job | What it checks | Runs on |
|-----|----------------|---------|
| `lint` | `ruff check` + `ruff format --check` + `pyright` | Every push/PR |
| `unit-tests` | `pytest tests/unit/ -v --timeout=60` | Every push/PR |
| `gpu-tests` | `pytest tests/integration/` | Manual dispatch |

The most common CI failure is `ruff format --check`. Run `uv run ruff format --check kempnerforge/ tests/ scripts/` locally before pushing.

## Testing

### Test Organization

| Directory | GPU? | When to run | What it covers |
|-----------|------|-------------|----------------|
| `tests/unit/` | No | Always | Config validation, model shapes, data pipeline logic, scheduler curves |
| `tests/integration/` | 1 GPU | GPU changes | Checkpoint round-trips, compiled model, single train step |
| `tests/distributed/` | 4 GPUs | Parallelism changes | FSDP, TP, EP, multi-GPU correctness |
| `tests/e2e/` | 4 GPUs | Training loop changes | Full training runs as subprocesses |
| `tests/smoke/` | 4 GPUs | Major changes | Parallelism config matrix |

### Writing Tests

**Unit tests** must run on CPU without a GPU. Use shared fixtures from `tests/conftest.py`:

```python
import pytest
import torch
from kempnerforge.config.schema import ModelConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Available fixtures from conftest.py:
#   tiny_model_config  — ModelConfig(dim=64, n_layers=2, n_heads=2, ...)
#   small_model_config — ModelConfig(dim=128, n_layers=4, n_heads=4, ...)
#   tiny_job_config    — Full JobConfig with tiny model, 10 steps
#   device             — cuda if available, else cpu
#   random_batch       — dict with input_ids and labels tensors
#   mmap_data_dir      — temp dir with small .npy token files


class TestMyFeature:
    def test_output_shape(self, tiny_model_config):
        model = build_something(tiny_model_config)
        out = model(torch.randn(2, 32, 64))
        assert out.shape == (2, 32, 64)

    def test_default_config_value(self):
        m = ModelConfig()
        assert m.my_new_field == 0  # disabled by default

    def test_rejects_invalid_config(self):
        with pytest.raises(ValueError, match="must be positive"):
            ModelConfig(my_new_field=-1)
```

**Test patterns:**
- Group related tests in a class (e.g., `TestRMSNorm`, `TestSigmoidRouter`).
- Test the happy path, edge cases, and invalid inputs.
- Every config field with constraints needs both a default-value test and a rejection test.
- Use `pytest.raises(ValueError, match="...")` with a match pattern for validation tests.

### Running Specific Tests

```bash
# By keyword
uv run pytest tests/unit/ -k "test_output_shape"

# By file
uv run pytest tests/unit/test_model.py -v

# By class
uv run pytest tests/unit/test_config.py::TestModelConfig -v

# By specific test
uv run pytest tests/unit/test_router.py::TestSigmoidTopKRouter::test_bias_adjustment -v
```

## Project Structure

```
kempnerforge/
  config/        — One dataclass per domain (model.py, training.py, distributed.py, ...)
                   schema.py re-exports all config classes for backward compat
                   registry.py — component registry for models, optimizers, schedulers, losses
                   loader.py — TOML parsing + CLI override merging
                   job.py — top-level JobConfig with cross-section validation
  model/         — Transformer blocks, attention, MLP, MoE, routers, norms, RoPE, embeddings
  distributed/   — DeviceMesh, FSDP2, tensor/expert/pipeline parallelism, FP8
  data/          — MemoryMappedDataset, MixtureDataset, StatefulDataLoader, samplers
  training/      — Optimizers, loss functions, LR schedulers, gradient utils, training hooks
  checkpoint/    — DCP-based distributed checkpointing with sync/async save
  resilience/    — SLURM signal handling, NaN detection, GPU/NCCL health checks
  metrics/       — MetricsTracker, MFU, WandB/TensorBoard backends, rank-aware logger
  profiling/     — torch.profiler integration
configs/
  model/         — Model size presets (llama_7b.toml, ...)
  train/         — Training configs (debug.toml, 7b.toml, moe_24gpu.toml, ...)
scripts/
  train.py       — Main training entry point
  slurm/         — SLURM launch scripts (singlenode.sh, multinode.sh, interactive.sh)
tests/
  unit/          — No GPU required
  integration/   — Single GPU
  distributed/   — Multi-GPU via torchrun
  e2e/           — Opt-in full training runs (--e2e)
  smoke/         — Parallelism config matrix (--smoke)
  conftest.py    — Shared fixtures (tiny configs, data helpers)
```

## Adding a New Feature

### New config field

1. Add the field to the appropriate dataclass in `kempnerforge/config/` (e.g., `model.py`, `training.py`).
2. **Default must preserve existing behavior.** Use `0`, `False`, `"none"`, or equivalent so existing configs keep working without changes.
3. Add validation in `__post_init__` if the field has constraints.
4. If it needs to be in `schema.py` re-exports, add it there.
5. Add unit tests in `tests/unit/test_config.py`: default value, valid values, rejection of invalid values.
6. Cross-section validation (e.g., "MoE + PP is not supported") goes in `JobConfig.__post_init__` in `kempnerforge/config/job.py`.

### New model component

1. Create a module in `kempnerforge/model/` or add to an existing one.
2. If it's a swappable component (like a router or norm variant), register it:
   ```python
   from kempnerforge.config.registry import registry
   registry.register("router", "my_router", _build_my_router)
   ```
3. Wire it up via config — a string field selects the registered builder (e.g., `moe_router = "sigmoid_topk"`).
4. Add unit tests for shapes, dtypes, edge cases, and backward/gradient flow.
5. If it changes distributed behavior, add distributed tests.

### New optimizer or scheduler

1. Implement in `kempnerforge/training/optimizer.py` or `scheduler.py`.
2. Register via the registry.
3. Add the name to the config validation.
4. Unit test the optimizer step and scheduler curve shape.
5. Add an E2E test that trains for a few steps and verifies loss descent.

### New parallelism mode

1. Implement in `kempnerforge/distributed/`.
2. Add config fields in `kempnerforge/config/distributed.py`.
3. Update `validate_world_size()` if it adds a new mesh dimension.
4. Respect the parallelism application order — wrong order causes silent correctness bugs:
   1. **Tensor Parallelism** — must see raw `nn.Linear` modules
   2. **Expert Parallelism** — partitions MoE experts across EP group
   3. **Float8 Training** — converts `nn.Linear` to `Float8Linear` (excludes experts/router)
   4. **Activation Checkpointing** — wraps blocks in `CheckpointWrapper`
   5. **FSDP2** — shards everything (uses float8 all-gather when FP8 is enabled)
5. Add distributed tests with `torchrun --nproc_per_node=4`.
6. Add an E2E test with the new parallelism configuration.

### New TOML config preset

If your feature needs a new training configuration (e.g., a new parallelism combination at a specific GPU count):

1. Add the TOML file in `configs/train/`.
2. Name it descriptively: `<model>_<gpus>_<parallelism>.toml` (e.g., `7b_16gpu_tp4.toml`).
3. Add it to the "Available configs" table in `README.md`.

### New training hook

Hooks extend the training loop without modifying `scripts/train.py`:

```python
from kempnerforge.training.hooks import TrainingHook, StepContext

class MyHook(TrainingHook):
    def on_step_end(self, ctx: StepContext) -> None:
        # ctx has: step, loss, grad_norm, lr, tokens_seen, model, optimizer
        if ctx.step % 100 == 0:
            do_something(ctx.model)
```

Available hook points: `on_train_begin`, `on_step_end`, `on_eval_end`, `on_checkpoint_save`, `on_train_end`.

## Configuration System

All behavior is controlled by typed dataclasses. Configs layer: **defaults -> TOML file -> CLI overrides**.

```bash
# CLI overrides use --section.key=value
uv run python scripts/train.py configs/train/debug.toml \
  --model.dim=512 --train.max_steps=100 --optimizer.lr=1e-4
```

Config rules:
- New fields must default to disabled/off so existing configs keep working.
- Validate in `__post_init__` — fail fast with a clear `ValueError`.
- Cross-section validation (e.g., "EP requires MoE") goes in `JobConfig.__post_init__`.

## Logging

Use the rank-aware logger in all library code:

```python
from kempnerforge.metrics.logger import get_logger
logger = get_logger(__name__)

logger.info("Training started")  # Only prints on rank 0
```

Never use `print()`. The logger suppresses output on non-zero ranks to avoid duplicated lines in distributed runs.

## Dependencies

```bash
# Add a runtime dependency
uv add <package>

# Add a dev-only dependency
uv add --group dev <package>
```

Always use `uv` — never pip, conda, or venv. PyTorch is pinned to the CUDA 12.8 index in `pyproject.toml`.

## Quick Reference

```bash
# Setup
uv sync

# Pre-push checklist (run all before every push)
uv run ruff format kempnerforge/ tests/ scripts/
uv run ruff check kempnerforge/ tests/
uv run pyright kempnerforge/
uv run pytest tests/unit/ -v --timeout=60

# GPU tests (inside a SLURM allocation)
uv run pytest tests/integration/ -v
uv run torchrun --nproc_per_node=4 -m pytest tests/distributed/ -v
uv run pytest tests/e2e/ --e2e -v

# Run a single test
uv run pytest tests/unit/test_model.py::TestRMSNorm::test_output_shape -v

# Debug training run
uv run python scripts/train.py configs/train/debug.toml

# Multi-GPU training
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/7b.toml
```

## Common Pitfalls

| Mistake | Why it breaks | Fix |
|---------|---------------|-----|
| Skip `ruff format --check` | CI format check fails even when lint passes | Run `uv run ruff format` before pushing |
| GPU-dependent unit test | CI unit tests run on CPU-only runners | Use `DEVICE = torch.device("cuda" if ... else "cpu")` |
| Wrong parallelism order | Silent numerical correctness bugs | Follow the 5-step order: TP -> EP -> FP8 -> AC -> FSDP |
| `print()` in library code | Duplicated output on every rank | Use `get_logger(__name__)` |
| New config without validation | Invalid values accepted silently | Add `__post_init__` check + rejection test |
| New config with breaking default | Existing configs break | Default to disabled (`0`, `False`, `"none"`) |
| Modifying `train.py` for extensibility | Couples experiment code to the training loop | Use `TrainingHook` subclass instead |
