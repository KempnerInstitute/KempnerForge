---
name: add-optimizer
description: Add a new optimizer to KempnerForge. Covers implementation, registry hook, config fields, tests, and a preset TOML.
---

## When to use
- Researcher wants to try an optimizer that is not in the registry.
- Adapting a published algorithm (e.g. a new Muon variant, a sign-based method) to KempnerForge's param-group + FSDP2 flow.

## Preflight
Run:

    uv run python scripts/check_env.py

Baseline only: `uv`, repo layout. Optional follow-up for the GPU smoke step at the end: `uv run python scripts/check_env.py --requires gpu`.

## Context (auto-generated, do not edit)
<!-- context-begin -->
Registered optimizers: adamw, lion, schedule_free_adamw, muon
Registry: kempnerforge/config/registry.py (registry.register_optimizer("<name>"))
Optimizer builders live in: kempnerforge/training/optimizer.py
Top-level builder: kempnerforge/training/optimizer.py::build_optimizer(model, config) — creates decay/no-decay param groups, looks up builder from registry, calls it
Builder signature: fn(param_groups: list[dict], config: OptimizerConfig) -> torch.optim.Optimizer
Config dataclass: kempnerforge/config/optimizer.py::OptimizerConfig
Existing fields: name, lr, weight_decay, betas, eps, fused, muon_momentum, muon_ns_steps, muon_adam_lr, schedule_free_warmup_steps
Existing tests: tests/unit/test_optimizer.py, tests/unit/test_additional_optimizers.py
<!-- context-end -->

## Procedure
Assume preflight has passed.

1. Implement the optimizer class in `kempnerforge/training/optimizer.py`. Follow the Lion example starting around line 47 for a full-from-scratch class. If you are wrapping an existing torch optimizer (like AdamW), the builder alone is enough — no class needed.

    For a from-scratch optimizer:

        class MyOpt(torch.optim.Optimizer):
            def __init__(self, params, lr=1e-3, ...):
                defaults = dict(lr=lr, ...)
                super().__init__(params, defaults)

            @torch.no_grad()
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        # update p.data using p.grad and self.state[p]
                return loss

2. Register the builder. Pattern from `optimizer.py`:

        @registry.register_optimizer("my_opt")
        def _build_my_opt(
            param_groups: list[dict],
            config: OptimizerConfig,
        ) -> torch.optim.Optimizer:
            return MyOpt(
                param_groups,
                lr=config.lr,
                ...,
            )

    The builder MUST accept `param_groups` (a list of dicts built by `build_optimizer`) rather than a flat `model.parameters()`, because `build_optimizer` pre-splits decay and no-decay groups.

3. Add optimizer-specific fields to `OptimizerConfig` in `kempnerforge/config/optimizer.py` if needed. Validate them in `__post_init__`. Keep defaults backward-compatible so existing configs do not break.

4. Add unit tests. Put narrow unit tests under `tests/unit/test_optimizer.py` or `tests/unit/test_additional_optimizers.py` (either is fine; group with similar optimizers). Cover at minimum:
    - Constructor accepts the documented args and sets up state correctly.
    - `step()` updates parameters.
    - State dict round-trip (`opt.state_dict()` then `opt.load_state_dict(...)`) preserves behavior.
    - Param-group separation is respected (different `weight_decay` per group does not leak).

5. Add a preset TOML under `configs/train/` to demonstrate the optimizer end-to-end. Use `configs/train/7b_16gpu_muon.toml` as a template. The preset should override `[optimizer].name` and any new fields you added.

6. Optional: update `docs/training/optimizers.md` if it exists and covers registered optimizers (verify by `ls docs/training/` and `grep` the doc).

## Verification
- `uv run pytest tests/unit/test_optimizer.py tests/unit/test_additional_optimizers.py -v` passes.
- `uv run ruff check kempnerforge/ tests/` is clean.
- `uv run ruff format --check kempnerforge/ tests/` passes (CI runs this too).
- Smoke test with the new optimizer: `uv run python scripts/train.py configs/train/debug.toml --data.dataset_path=<path> --optimizer.name=my_opt --train.max_steps=20`. Loss must decrease.

## Gotchas
- `build_optimizer` passes a `param_groups` list, not a parameter generator. The builder signature must be `(param_groups, config)`.
- `fused=True` paths require CUDA tensors. Guard with `config.fused and torch.cuda.is_available()` when passing `fused=` to a torch optimizer.
- Schedule-free optimizers should be paired with `scheduler.name="none"`. Adding a scheduler on top of one that manages its own warmup inside `step()` will double-count.
- Do not add the optimizer to `__init__.py` re-exports. Discovery is via the registry only — re-exports create a second source of truth that drifts.
- FSDP2 sharded params: optimizers see local shards via DTensor. Most torch optimizers handle this transparently; custom ops (like Muon's Newton-Schulz) need to be careful about `.to_local()` vs DTensor ops.
- Registry entries are global. A second module registering the same name raises `ValueError` at import time — choose a unique name.

## Related skills
- `/kempnerforge:explain-architecture` — see the "Optimizer and scheduler" section for where this fits in the training loop.
- `/kempnerforge:smoke-test` — run after adding, to validate end to end on 1 GPU.
