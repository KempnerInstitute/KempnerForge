#!/bin/bash
#SBATCH --job-name=kempnerforge
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --signal=B:SIGTERM@120
#SBATCH --requeue
#
# Multi-node multi-GPU training via srun.
#
# IMPORTANT: --ntasks-per-node MUST equal --gpus-per-node. Each srun task
# maps to one GPU process. SLURM sets RANK, LOCAL_RANK, WORLD_SIZE via
# SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS automatically.
#
# Usage:
#   sbatch --nodes=4 scripts/slurm/multinode.sh configs/train/default.toml
#   sbatch --nodes=8 scripts/slurm/multinode.sh \
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
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
# Pick a random free port (avoids collisions with other jobs on the same node)
export MASTER_PORT=$(comm -23 <(seq 15000 20000 | sort) \
    <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# ---- Environment ----
# Prevent NCCL memory stacking
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# Use InfiniBand for inter-node communication
export NCCL_SOCKET_IFNAME=ib0
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
# srun launches one process per GPU. Each process gets:
#   RANK=SLURM_PROCID, LOCAL_RANK=SLURM_LOCALID, WORLD_SIZE=SLURM_NTASKS
srun --kill-on-bad-exit=1 \
    uv run python scripts/train.py "${CONFIG}" ${OVERRIDES}
