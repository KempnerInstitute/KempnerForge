#!/bin/bash
# Interactive srun launcher for multi-node training.
#
# Usage:
#   ./scripts/srun_launch.sh <JOBID> <CONFIG> [overrides...]
#
# Examples:
#   ./scripts/srun_launch.sh 3565401 configs/train/multinode_70b_tp4_fsdp8.toml
#   ./scripts/srun_launch.sh 3565401 configs/train/multinode_70b_tp4_fsdp8.toml --train.max_steps=50

set -euo pipefail

JOBID="${1:?Usage: srun_launch.sh <JOBID> <CONFIG> [overrides...]}"
CONFIG="${2:?Usage: srun_launch.sh <JOBID> <CONFIG> [overrides...]}"
shift 2
OVERRIDES="$@"

# ---- Resolve nodes from SLURM job ----
NODELIST=$(squeue -j "$JOBID" -h -o "%N")
export MASTER_ADDR=$(scontrol show hostnames "$NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)

# ---- NCCL / InfiniBand ----
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# ---- CUDA ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Info ----
NNODES=$(squeue -j "$JOBID" -h -o "%D")
GPUS_PER_NODE=4
echo "=== KempnerForge srun Launch ==="
echo "Job ID:     ${JOBID}"
echo "Nodes:      ${NNODES} (${NODELIST})"
echo "GPUs/node:  ${GPUS_PER_NODE}"
echo "Total GPUs: $((NNODES * GPUS_PER_NODE))"
echo "Master:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:     ${CONFIG}"
echo "Overrides:  ${OVERRIDES}"
echo "================================"

# ---- Launch ----
srun --jobid="$JOBID" \
    --nodes="$NNODES" \
    --ntasks-per-node="$GPUS_PER_NODE" \
    --gpus-per-node="$GPUS_PER_NODE" \
    --kill-on-bad-exit=1 \
    uv run python scripts/train.py "${CONFIG}" ${OVERRIDES}
