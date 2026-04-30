# KempnerForge plugin

KempnerForge ships a plugin that lets an AI coding assistant drive first-run setup, smoke tests, SLURM launches, and common extension tasks through a small set of well-scoped skills. Every skill starts with the same preflight contract (a call to `scripts/check_env.py`), so failures come back with actionable fix lines instead of silent misbehavior.

## Install

From a fresh clone:

```
/plugin marketplace add /abs/path/to/KempnerForge
/plugin marketplace list
/plugin install kempnerforge@kempnerforge
/reload-plugins
```

The first command points the plugin system at this repo's `.claude-plugin/marketplace.json`. Use the absolute path to the checkout; relative paths like `.` are accepted silently but do not actually register the marketplace.

Before installing, verify the marketplace registered: `/plugin marketplace list` should show `kempnerforge` under "Configured marketplaces". If it does not appear, the add silently failed and you need to re-check the absolute path.

The third command installs the `kempnerforge` plugin from the `kempnerforge` marketplace, exposing skills under the `kempnerforge:` namespace. The fourth reloads the session so skills are immediately available.

Optional schema validation:

```
/plugin validate .
```

## First-run flow

Once installed, run these in order:

```
/kempnerforge:install-and-verify
/kempnerforge:cluster-config
```

`install-and-verify` runs `uv sync`, confirms the venv is on Python ≥ 3.12 (per `.python-version`), then runs the same four checks CI runs on every PR (`ruff check`, `ruff format --check`, `pyright`, `pytest tests/unit/`). A green pass means your environment matches what CI gates against. Skip it only if you have already run those four locally on this checkout.

`cluster-config` calls `scripts/check_env.py --init`, which reads `SLURM_*` env vars as defaults, prompts for each field, and writes `configs/cluster/local.toml` atomically. The file is gitignored. Every other skill reads from it. Run this whenever you onboard onto a new SLURM cluster (skip it on CPU-only or non-SLURM dev boxes).

Subsequent sessions on the same checkout skip both prompts. Reconfiguration is a re-run of the same command.

## v0.1 skill catalog

These skills cover the end-to-end path from clone to real runs.

| Skill | Category | Preflight tags | What it does |
|-------|----------|----------------|--------------|
| `install-and-verify` | onboarding | baseline | Run `uv sync`, confirm Python ≥ 3.12, then run the four CI checks (ruff check / format, pyright, pytest unit). First step after cloning. |
| `cluster-config` | onboarding | baseline | Write `configs/cluster/local.toml`. First step on any new cluster. |
| `smoke-test` | run-training | `gpu` | Short one-GPU training loop. Confirms torch, CUDA, NCCL, uv, dataloader. |
| `slurm-launch` | run-training | `slurm` (+ `multi-node`) | Submit `sbatch` for single- or multi-node jobs. Inject account, partition, QoS, time from `local.toml`. |
| `explain-architecture` | orient | baseline | Walk through subsystems in forward-pass order. Read-only. |
| `add-optimizer` | extend-component | baseline | Register a new optimizer. Covers implementation, config fields, tests, preset TOML. |
| `component-gaps` | orient | baseline | Per-subsystem status: implemented, tested, stubbed, known limitations. |

"Preflight tags" above are the arguments each skill passes to `check_env.py --requires` in its Preflight section. Baseline (empty tag list) just verifies `uv` and repo layout.

Invoke any skill with `/kempnerforge:<name>` in a session.

## Preflight contract

Every skill's Preflight section calls `check_env.py` with the tag list it needs. Before running the procedure, the agent runs:

```
uv run python scripts/check_env.py --requires <tags>
```

Tags: `gpu`, `slurm`, `multi-node`, `wandb`, `hf`, `gh`. Empty list runs baseline only (`uv`, repo layout).

Exit codes:
- `0`: OK or WARN. Proceed.
- `1`: MISS. Print stdout to the user and stop. The output includes a fix line for each missing dependency.
- `2`: Script error. Print stderr and stop.

JSON output is available via `--json` for scripting. Credential probes (wandb, HF) are opt-in via `--check-credentials`; the default baseline treats env-var presence as sufficient.

## How skills stay current

Each skill file (`plugins/kempnerforge/skills/<name>/SKILL.md`) has three blocks:

1. **Frontmatter** (`name`, `description`): the two fields Claude Code reads when deciding whether a skill matches a user request.
2. **Context block** (bounded by `<!-- context-begin -->` / `<!-- context-end -->`): registered component names, key file paths, config fields — the facts that drift as the codebase evolves.
3. **Procedure block** (hand-written): steps, verification, gotchas. Stable across releases.

In v0.1, all three blocks are hand-maintained, and the preflight tag list lives in the procedure's Preflight section (not in frontmatter). Phase 2 introduces `scripts/regenerate_skills.py` to rewrite block 2 from the codebase, and a CI check that fails PRs if any skill drifts from the code it references.

## Extending the plugin

To add a new skill:

1. Create `plugins/kempnerforge/skills/<name>/SKILL.md` with `name` / `description` frontmatter and the Preflight / Context / Procedure / Verification / Gotchas sections.
2. In the Preflight section, call `uv run python scripts/check_env.py --requires <tags>` with the minimal tag list your skill needs.
3. If your skill runs a code path not yet covered by `check_env`, add a new tag to `scripts/check_env.py` and a unit test in `tests/unit/test_check_env.py`.
4. Bump `plugins[0].version` in `.claude-plugin/marketplace.json` and the README badge.

See the existing skills under `plugins/kempnerforge/skills/` for the reference layout.

## Design choices worth knowing

- **Single cluster-config writer.** Only `cluster-config` runs `check_env --init`. Every other skill is a read-only consumer. Stops drift between multiple code paths trying to reconcile env state.
- **Fail fast on MISS.** Skills do not try to auto-install or auto-fix dependencies. They print the fix line and stop. Keeps behavior predictable and side-effect-free.
- **Credentials never in `local.toml`.** The file stores usernames and entity names only. API keys stay in env vars (`WANDB_API_KEY`, `HF_TOKEN`).
- **No hidden state.** The plugin's whole configuration is two files: `configs/cluster/local.toml` (per user, gitignored) and the committed `plugins/kempnerforge/skills/`. Agents cannot accumulate state between sessions beyond what those files carry.

## Versioning

Plugin version is declared in `.claude-plugin/marketplace.json` under `plugins[0].version`. Bumps follow the repo's semver track. Skills are developed in lockstep with the plugin version; there is no independent per-skill version in v0.1.

## Manual fallback

For users who prefer not to run a plugin:

- Copy `configs/cluster/local.toml.example` to `configs/cluster/local.toml` and edit it by hand.
- Run `scripts/check_env.py --requires <tags>` directly to preflight.
- Read the individual `SKILL.md` files under `plugins/kempnerforge/skills/` as reference procedures; each one is a standalone runbook.

The plugin is a convenience layer, not a dependency.
