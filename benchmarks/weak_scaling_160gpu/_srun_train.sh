#!/bin/bash
# Per-task launch wrapper. Sets the torchrun-convention environment variables from
# SLURM so KempnerForge's get_world_info() sees the whole job. Without this,
# WORLD_SIZE falls back to a single node's GPU count and any multi-node run dies
# with "Parallelism dimensions ... do not match world_size".
#
# Invoked by run_one.sbatch as:
#   srun ... _srun_train.sh <train.py overrides...>
set -uo pipefail

# Diagnostic: confirms srun spans every node. Expect ntasks = nodes x 4, distinct hosts.
echo "[srun-env] host=$(hostname) procid=${SLURM_PROCID:-?} localid=${SLURM_LOCALID:-?} ntasks=${SLURM_NTASKS:-?} nnodes=${SLURM_NNODES:-?}" >&2

export RANK="$SLURM_PROCID"
export LOCAL_RANK="$SLURM_LOCALID"
export WORLD_SIZE="$SLURM_NTASKS"

# MASTER_ADDR has to be a node inside this srun step. SLURM_NODELIST here is the
# step's nodelist and is identical on every task, so head -1 is both in-subset and
# consistent across ranks, and rank 0 lives there under block distribution.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
# Pick a free port deterministically per job rather than per task, so all ranks agree.
export MASTER_PORT="${MASTER_PORT:-$((15000 + ${SLURM_JOB_ID:-0} % 5000))}"

cd "${REPO:-$PWD}"
exec uv run python scripts/train.py configs/train/debug.toml "$@"
