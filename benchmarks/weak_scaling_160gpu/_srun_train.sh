#!/bin/bash
set -uo pipefail

echo "[srun-env] host=$(hostname) procid=${SLURM_PROCID:-?} localid=${SLURM_LOCALID:-?} ntasks=${SLURM_NTASKS:-?} nnodes=${SLURM_NNODES:-?}" >&2

export RANK="$SLURM_PROCID"
export LOCAL_RANK="$SLURM_LOCALID"
export WORLD_SIZE="$SLURM_NTASKS"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
export MASTER_PORT="${MASTER_PORT:-$((15000 + ${SLURM_JOB_ID:-0} % 5000))}"

cd "${REPO:-$PWD}"
exec uv run python scripts/train.py configs/train/debug.toml "$@"
