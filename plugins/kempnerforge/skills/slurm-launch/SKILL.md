---
name: slurm-launch
description: Submit a training job via sbatch. Wraps singlenode.sh for one node and multinode.sh for multiple. Injects account, partition, QoS, and time overrides from local.toml.
---

## When to use
- User wants to queue a real training run, not just a smoke test.
- User wants to scale up from 1 GPU (smoke-test) to a full node or multiple nodes.
- User wants to requeue an interrupted run under SLURM's `--requeue` semantics.

Prerequisites: `/kempnerforge:cluster-config` has been run, and `/kempnerforge:smoke-test` passed on the target cluster.

## Preflight
Run:

    uv run python scripts/check_env.py --requires slurm

For multi-node jobs, also run:

    uv run python scripts/check_env.py --requires slurm,multi-node

If the exit code is non-zero, print stdout to the user verbatim and stop. Common MISS cases:
- `slurm` MISS: `sbatch` not on PATH, or `require_account=true` with no account configured. Tell the user to run `/kempnerforge:cluster-config`.
- `multi-node` MISS: `shared_fs_root` missing, repo not under shared FS, or checkpoints_root on the wrong mount. The fix line points at the specific issue.

## Context (auto-generated, do not edit)
<!-- context-begin -->
Single-node script: scripts/slurm/singlenode.sh (1 task per node, launches torchrun internally)
Multi-node script: scripts/slurm/multinode.sh (ntasks-per-node == gpus-per-node, uses srun direct)
Interactive script: scripts/slurm/interactive.sh (salloc wrapper for debugging)
Requeue-aware template: scripts/slurm/7b_requeue.sh (SIGTERM@120, --requeue, auto-resume via latest symlink)
Example: scripts/slurm/13b_validation.sh (large-model multi-node config reference)
Config fields consumed: [slurm].{account, partition, qos, default_time, default_nodes, default_gpus_per_node}
Env vars set by scripts: MASTER_ADDR, MASTER_PORT, NCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME, NCCL_IB_*, NCCL_TIMEOUT
<!-- context-end -->

## Procedure
Assume preflight has passed. Read `configs/cluster/local.toml` to pull `[slurm].account`, `[slurm].partition`, `[slurm].qos` (skip if empty), and `[slurm].default_time`.

1. Decide single-node vs multi-node from the user's request:
    - 1 to 8 GPUs on one box: single-node (`singlenode.sh`, uses torchrun).
    - 2+ nodes: multi-node (`multinode.sh`, uses srun direct, one task per GPU).

2. Build the `sbatch` command. Pass `[slurm]` fields via CLI flags rather than editing the `#SBATCH` lines in the script (the scripts have placeholders like `<partition-name>` that must be overridden at submit time):

    Single-node:

        sbatch \
          --account=<account> \
          --partition=<partition> \
          [--qos=<qos>] \
          --time=<default_time> \
          --gpus-per-node=<N> \
          scripts/slurm/singlenode.sh <config.toml> [--train.max_steps=...]

    Multi-node:

        sbatch \
          --account=<account> \
          --partition=<partition> \
          [--qos=<qos>] \
          --time=<default_time> \
          --nodes=<N> \
          --ntasks-per-node=<gpus> \
          --gpus-per-node=<gpus> \
          scripts/slurm/multinode.sh <config.toml>

    Omit `--qos=` when `[slurm].qos` is empty. For multi-node, `--ntasks-per-node` MUST equal `--gpus-per-node` (one process per GPU). Mismatch is the most common multi-node misconfiguration.

3. Submit, capture the job ID:

        JOB=$(sbatch --parsable ...)
        echo "Submitted job $JOB"

4. Tail the log (path pattern is `logs/<jobid>_<jobname>.out`):

        tail -f logs/${JOB}_*.out

    First useful output for single-node: `=== KempnerForge Training ===` header. For multi-node: `=== KempnerForge Multi-Node Training ===` with `Total GPUs` and `Master` lines. If those do not appear within ~60 seconds of the job entering `RUNNING` state, something is wrong with the distributed init.

## Verification
- `squeue -u $USER -j $JOB` shows the job in `RUNNING` state.
- Log file appears under `logs/` and contains the header banner.
- Loss decreases over the first ~20 steps (grep `loss=` in the log).
- `sacct -j $JOB --format=JobID,State,ExitCode` shows `COMPLETED` with exit code 0 on normal termination, or `TIMEOUT` / `REQUEUED` on preemption (both expected and recoverable).

## Gotchas
- The `#SBATCH --partition=<partition-name>` placeholders in the shipped scripts will cause submission failures unless overridden on the `sbatch` command line. Do NOT edit those lines in the committed file; override at submit time so the scripts stay user-agnostic.
- `multinode.sh` is NOT a torchrun wrapper. Each `srun` task maps to one GPU, and rank info flows from SLURM env vars via `init_distributed()`. Do not add `torchrun` inside the multi-node script.
- A run killed with `scancel` exits immediately (SIGKILL after 30 s). A run preempted by SLURM gets `SIGTERM` 120 s before the wall-time cutoff (see `--signal=B:SIGTERM@120`), which the training loop catches and saves a checkpoint. Use `--requeue` for automatic restart.
- Interactive sessions: never `scancel` the job allocation. `Ctrl-C` kills only the `srun` process; the allocation stays up for reuse. `scancel $JOB` kills the entire allocation and forces requeue from disk.
- `logs/` must exist before submission. The scripts create it, but if the repo root is read-only (rare), submission fails late.

## Related skills
- `/kempnerforge:cluster-config` — sets the `[slurm]` fields this skill consumes.
- `/kempnerforge:smoke-test` — run this first on any new cluster.
