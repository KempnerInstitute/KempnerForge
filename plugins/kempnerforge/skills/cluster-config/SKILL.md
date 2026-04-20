---
name: cluster-config
description: First-run setup. Detect SLURM account/partition/QoS from env and write configs/cluster/local.toml so every other skill can preflight.
---

## When to use
- First time running any KempnerForge skill on a new machine or cluster.
- Any other skill reports a MISS that points at `configs/cluster/local.toml` (account, partition, shared_fs_root, etc.).
- Switching to a different cluster or account on the same checkout.

This is the only skill that writes `configs/cluster/local.toml`. Every other skill is a read-only consumer.

## Preflight
Run:

    uv run python scripts/check_env.py

If the exit code is non-zero, print stdout to the user verbatim and stop. The baseline checks (`uv`, repo layout) must pass before this skill can proceed.

## Context (auto-generated, do not edit)
<!-- context-begin -->
Config target: configs/cluster/local.toml (gitignored)
Schema reference: configs/cluster/local.toml.example
Writer entry point: scripts/check_env.py do_init()
Writer modes: interactive (default) or --non-interactive with per-field flags for agent-driven writes
Precedence (do_init): CLI flag > env var > static default
Precedence (check_env runtime): explicit --cluster-config override > env var > local.toml
Discovered defaults: SLURM_CLUSTER_NAME, SLURM_ACCOUNT, SLURM_PARTITION, SLURM_QOS
Fields written: [cluster].name, [cluster].scheduler, [slurm].{account, partition, qos, default_time, default_nodes, default_gpus_per_node, require_account, require_qos}, [network].{ib_interface, shared_fs_root}, [paths].checkpoints_root, [credentials].{wandb_entity, hf_user}
<!-- context-end -->

## Procedure

Dispatch on whether `configs/cluster/local.toml` already exists.

### Flow A — config is missing (first run)

Collect values directly in chat, then call the non-interactive writer in one shot. Do **not** hand control back to a terminal prompt loop.

1. Read `$SLURM_CLUSTER_NAME`, `$SLURM_ACCOUNT`, `$SLURM_PARTITION`, `$SLURM_QOS` and echo them as suggested defaults.
2. Ask the user to confirm or override each field in one message. Use the field-by-field guidance below. Leave QoS blank if unused.
3. Run the writer with explicit flags:

        uv run python scripts/check_env.py --init --non-interactive \
            --cluster-name=<name> \
            --slurm-account=<acct> --slurm-partition=<part> --slurm-qos=<qos> \
            --shared-fs-root=<path> --checkpoints-root=<path> \
            --wandb-entity=<entity> --hf-user=<user>

    Omit any flag the user left at its default. The script falls back to env vars, then static defaults.

4. Verify:

        uv run python scripts/check_env.py --requires slurm,multi-node

    Expect OK or WARN for each tag. A MISS means a value needs correcting (usually `shared_fs_root` does not exist on disk or is not a distributed FS). Re-run step 3 with `--force` and the corrected flags.

5. Confirm the file is gitignored:

        git check-ignore -v configs/cluster/local.toml

    Should print `configs/cluster/.gitignore:3:local.toml` and exit 0.

### Flow B — config already exists

Validate the existing file, surface any problems, and only rewrite if the user wants to change something.

1. Run validation:

        uv run python scripts/check_env.py --requires slurm,multi-node

2. If every tag is OK or WARN: summarize what the file says (cluster, account, partition, qos, shared_fs_root, checkpoints_root) and ask the user whether any field needs updating. If they say no, stop. Do not rewrite.

3. If any tag is MISS, or the user wants to change a field: collect the new value(s) in chat, then rewrite with `--force`:

        uv run python scripts/check_env.py --init --non-interactive --force \
            --<field>=<new_value> [...]

    Only pass the flags whose values are changing. The unchanged fields will fall back through env vars and static defaults, so pass flags for **every** field the user wants preserved (read the current file first and forward those values explicitly).

4. Re-verify with `--requires slurm,multi-node`.

### Field-by-field guidance

- **cluster_name**: free-form label for logging. Example: `kempner_h100`.
- **scheduler**: `slurm` if `sbatch` is on PATH, otherwise `none`. The script auto-detects if the flag is omitted.
- **slurm_account / slurm_partition / slurm_qos**: take from `sacctmgr show assoc user=$USER`. Leave QoS blank if the cluster does not require one.
- **require_account** (default `true`): set `false` for clusters where `sbatch` runs without `--account`.
- **require_qos** (default `false`): set `true` only on clusters that reject jobs without `--qos`.
- **ib_interface**: leave as `auto` unless the user knows a specific interface. `check_env --requires multi-node` probes `/sys/class/infiniband` to verify.
- **shared_fs_root**: distributed FS root all ranks share (NFS / Lustre / GPFS). The repo checkout must live under this path for multi-node runs.
- **checkpoints_root**: typically `<shared_fs_root>/checkpoints` for multi-node, or `<repo>/checkpoints` for single-node.
- **wandb_entity / hf_user**: never an API key. Tokens live in env vars (`WANDB_API_KEY`, `HF_TOKEN`).

## Verification
- `cat configs/cluster/local.toml` and sanity-check each section against what the user specified.
- `uv run python scripts/check_env.py --requires slurm` returns OK.
- `uv run python scripts/check_env.py --requires multi-node` returns OK or WARN (WARN is fine for single-node-only users).
- `git status` shows no tracked changes under `configs/cluster/local.toml`.

## Gotchas
- Use `--non-interactive` from an agent session. Without it, `--init` reads from stdin and blocks when stdin is not a TTY.
- `--non-interactive` without `--force` refuses to overwrite an existing file (exit 2). This is intentional: in Flow B, read the file first before rewriting.
- When rewriting, pass a flag for every field the user wants preserved. Unspecified flags fall back to env vars and then static defaults; an unset env var will blank the field.
- Credentials section stores names only, never tokens. If the user tries to paste an API key, stop them and point at the `WANDB_API_KEY` / `HF_TOKEN` env vars.
- On multi-cluster setups with a single checkout, users can keep multiple config files and pass `--cluster-config <path>` to select one.

## Related skills
- `/kempnerforge:smoke-test` — next step after cluster-config on any GPU node.
- `/kempnerforge:slurm-launch` — wraps `sbatch` and depends on `[slurm]` section being populated.
