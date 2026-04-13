#!/bin/bash
# Launch wrapper for running distributed tests under srun multi-node.
# One srun task per GPU (srun-direct pattern, matching scripts/slurm/multinode.sh).
#
# Usage (from within a SLURM allocation):
#   srun --jobid=<ID> --nodes=<N> --ntasks-per-node=<gpus_per_node> \
#        --gpus-per-node=<G> --overlap \
#        scripts/slurm/_run_distributed_tests.sh <pytest_target> [pytest args...]
#
# Each task sets RANK/LOCAL_RANK/WORLD_SIZE from SLURM env so pytest's
# `RANK`-gated skip mark in tests/distributed/conftest.py does not fire.

set -eo pipefail

# Pytest discovers tests relative to CWD; run from the repo root (or pass a
# target path as an argument). No hard-coded absolute path — the caller's
# shell cwd is respected.

# --- Distributed env vars (torchrun-compatible) ---
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Deterministic port from job ID; same value on every task so they rendezvous.
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 5000))

# --- NCCL / Gloo: bind to IB ---
IB_IFNAME=$(ip -br addr | awk '/^ib[0-9]+\s+UP\s+[0-9]/ {print $1; exit}')
IB_IFNAME="${IB_IFNAME:-ib0}"
export NCCL_SOCKET_IFNAME="$IB_IFNAME"
export GLOO_SOCKET_IFNAME="$IB_IFNAME"
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_TIMEOUT=1800

# --- Per-rank logging on shared FS ---
# All ranks (0..N-1) write to $LOG_DIR/job<JOBID>_rank<N>.log so failures on any
# rank can be inspected. Rank 0 also streams to stdout so the terminal sees live
# progress. LOG_DIR defaults to <cwd>/logs/distributed; override with
# KEMPNERFORGE_LOG_DIR if you want a fixed absolute location.
LOG_DIR="${KEMPNERFORGE_LOG_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/logs/distributed}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/job${SLURM_JOB_ID}_rank${SLURM_PROCID}.log"

if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "=== Distributed tests ==="
    echo "Nodes:       $SLURM_JOB_NODELIST"
    echo "World size:  $WORLD_SIZE"
    echo "Master:      $MASTER_ADDR:$MASTER_PORT"
    echo "IB:          $IB_IFNAME"
    echo "Pytest args: $*"
    echo "Log dir:     $LOG_DIR"
    echo "Per-rank:    job${SLURM_JOB_ID}_rank<N>.log"
    echo "========================="
    # Rank 0: stream to stdout AND per-rank file. PIPESTATUS preserves pytest exit code.
    uv run pytest "$@" 2>&1 | tee "$LOG_FILE"
    exit "${PIPESTATUS[0]}"
else
    # Other ranks: per-rank file only.
    exec uv run pytest "$@" > "$LOG_FILE" 2>&1
fi
