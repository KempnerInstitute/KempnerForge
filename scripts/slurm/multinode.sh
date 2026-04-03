#!/bin/bash
#SBATCH --job-name=kempnerforge
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --signal=B:SIGTERM@120
#SBATCH --requeue
#
# Multi-node multi-GPU training via torchrun with SLURM.
#
# Usage:
#   sbatch --nodes=4 scripts/slurm/multinode.sh configs/train/default.toml
#   sbatch --nodes=8 --partition=kempner_h100 scripts/slurm/multinode.sh \
#       configs/train/default.toml --train.max_steps=50000
#
# Automatically detects master addr/port from SLURM environment and
# supports auto-resume on requeue via latest checkpoint symlink.

set -euo pipefail

# ---- Config ----
CONFIG="${1:?Usage: multinode.sh <config.toml> [overrides...]}"
shift
OVERRIDES="$@"

NNODES="${SLURM_NNODES}"
NGPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"

# ---- Master address/port from SLURM ----
# Get the first node in the allocation as master
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"

# ---- Environment ----
# Prevent NCCL memory stacking
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# InfiniBand / RoCE settings (common on HPC clusters)
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3

# Timeout for NCCL operations (seconds) — increase for large clusters
export NCCL_TIMEOUT=600

# Ensure log directory exists
mkdir -p logs

echo "=== KempnerForge Multi-Node Training ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Nodes:        ${NNODES} (${SLURM_JOB_NODELIST})"
echo "GPUs/node:    ${NGPUS_PER_NODE}"
echo "Total GPUs:   $((NNODES * NGPUS_PER_NODE))"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:       ${CONFIG}"
echo "Overrides:    ${OVERRIDES}"
echo "Restart cnt:  ${SLURM_RESTART_COUNT:-0}"
echo "========================================="

# ---- Launch with srun ----
# srun launches one process per node; torchrun spawns per-GPU workers
srun --kill-on-bad-exit=1 \
    uv run torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NGPUS_PER_NODE}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    scripts/train.py "${CONFIG}" ${OVERRIDES}
