#!/bin/bash

set -euo pipefail

JOBID="${1:?Usage: scripts/slurm/interactive.sh <JOBID> <CONFIG> [overrides...]}"
CONFIG="${2:?Usage: scripts/slurm/interactive.sh <JOBID> <CONFIG> [overrides...]}"
shift 2
OVERRIDES="$@"

NODELIST=$(squeue -j "$JOBID" -h -o "%N")
export MASTER_ADDR=$(scontrol show hostnames "$NODELIST" | head -n 1)
export MASTER_PORT=$(comm -23 <(seq 15000 20000 | sort) \
    <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

IB_IFNAME=$(ip -br addr | awk '/^ib[0-9]+\s+UP\s+[0-9]/ {print $1; exit}')
export NCCL_SOCKET_IFNAME="${IB_IFNAME:-ib0}"
export GLOO_SOCKET_IFNAME="${IB_IFNAME:-ib0}"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NNODES=$(squeue -j "$JOBID" -h -o "%D")
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"
echo "=== KempnerForge srun Launch ==="
echo "Job ID:     ${JOBID}"
echo "Nodes:      ${NNODES} (${NODELIST})"
echo "GPUs/node:  ${GPUS_PER_NODE}"
echo "Total GPUs: $((NNODES * GPUS_PER_NODE))"
echo "Master:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "IB iface:   ${IB_IFNAME:-ib0 (default)}"
echo "Config:     ${CONFIG}"
echo "Overrides:  ${OVERRIDES}"
echo "================================"

srun --jobid="$JOBID" \
    --nodes="$NNODES" \
    --ntasks-per-node="$GPUS_PER_NODE" \
    --gpus-per-node="$GPUS_PER_NODE" \
    --kill-on-bad-exit=1 \
    uv run python scripts/train.py "${CONFIG}" ${OVERRIDES}
