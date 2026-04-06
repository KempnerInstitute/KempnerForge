#!/bin/bash
#SBATCH --job-name=kempnerforge
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --signal=B:SIGTERM@120
#
# Single-node multi-GPU training via torchrun.
#
# Usage:
#   sbatch scripts/slurm/singlenode.sh configs/train/default.toml
#   sbatch scripts/slurm/singlenode.sh configs/train/default.toml --train.max_steps=1000
#
# The first argument is the config TOML file. Any additional arguments are
# passed through as CLI overrides (e.g., --optimizer.lr=1e-4).

set -euo pipefail

# ---- Config ----
CONFIG="${1:?Usage: launch.sh <config.toml> [overrides...]}"
shift
OVERRIDES="$@"

NGPUS="${SLURM_GPUS_PER_NODE:-4}"

# ---- Environment ----
# Prevent NCCL memory stacking
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# Use InfiniBand for GPU-to-GPU communication
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3

# Ensure log directory exists
mkdir -p logs

echo "=== KempnerForge Training ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPUs:       ${NGPUS}"
echo "Config:     ${CONFIG}"
echo "Overrides:  ${OVERRIDES}"
echo "============================"

# ---- Launch ----
uv run torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    scripts/train.py "${CONFIG}" ${OVERRIDES}
