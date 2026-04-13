#!/bin/bash
# Launch wrapper for running scripts/train.py under srun multi-node,
# srun-direct pattern (one task per GPU). Mirrors scripts/slurm/multinode.sh
# but for use inside an existing interactive allocation.
#
# Invoke from the repo root (same convention as multinode.sh):
#   srun --jobid=<ID> --nodes=<N> --ntasks-per-node=<gpus_per_node> \
#        --gpus-per-node=<G> --overlap \
#        scripts/slurm/_run_training.sh <config.toml> [overrides...]
#
# Environment overrides:
#   KEMPNERFORGE_LOG_DIR   where per-rank logs go (default: $SLURM_SUBMIT_DIR/logs/distributed or $PWD/logs/distributed)
#
# Per-rank logs go to $LOG_DIR/train_job<JOB>_rank<N>.log; rank 0 also
# streams to stdout.

set -eo pipefail

CONFIG="${1:?Usage: _run_training.sh <config.toml> [overrides...]}"
shift

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 5000))

IB_IFNAME=$(ip -br addr | awk '/^ib[0-9]+\s+UP\s+[0-9]/ {print $1; exit}')
IB_IFNAME="${IB_IFNAME:-ib0}"
export NCCL_SOCKET_IFNAME="$IB_IFNAME"
export GLOO_SOCKET_IFNAME="$IB_IFNAME"
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_TIMEOUT=1800

LOG_DIR="${KEMPNERFORGE_LOG_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/logs/distributed}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_job${SLURM_JOB_ID}_rank${SLURM_PROCID}.log"

if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "=== Training (srun-direct) ==="
    echo "Nodes:       $SLURM_JOB_NODELIST"
    echo "World size:  $WORLD_SIZE"
    echo "Master:      $MASTER_ADDR:$MASTER_PORT"
    echo "IB:          $IB_IFNAME"
    echo "Config:      $CONFIG"
    echo "Overrides:   $*"
    echo "Log dir:     $LOG_DIR"
    echo "=============================="
    uv run python scripts/train.py "$CONFIG" "$@" 2>&1 | tee "$LOG_FILE"
    exit "${PIPESTATUS[0]}"
else
    exec uv run python scripts/train.py "$CONFIG" "$@" > "$LOG_FILE" 2>&1
fi
