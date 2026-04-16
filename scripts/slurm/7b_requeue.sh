#!/bin/bash
#SBATCH --job-name=kf-7b-adamw
#SBATCH --partition=<partition-name>     # use a preemptible/requeue partition
#SBATCH --account=<account-name>         # your SLURM account
#SBATCH --constraint=<gpu-type>          # e.g. h200, h100, a100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --signal=B:SIGTERM@120
#SBATCH --requeue
#
# Llama-3 7B training on 16 GPUs via a preemptible SLURM partition.
#
# Preemption-resilient:
#   - SIGTERM@120: SLURM sends SIGTERM 120s before kill
#   - ShutdownHandler saves emergency checkpoint on SIGTERM
#   - Auto-resume from latest checkpoint on requeue
#   - Checkpoints every 500 steps (~1.5h of work)
#
# Usage:
#   sbatch scripts/slurm/7b_requeue.sh
#   sbatch scripts/slurm/7b_requeue.sh --train.max_steps=50000  # override
#
# Monitor:
#   sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
#   tail -f logs/<JOBID>_kf-7b-adamw.out

set -euo pipefail

CONFIG="configs/train/7b_16gpu_adamw.toml"
OVERRIDES="$@"

NNODES="${SLURM_NNODES}"
NGPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"

# ---- Master address/port from SLURM ----
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=$(comm -23 <(seq 15000 20000 | sort) \
    <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# ---- Network interface detection ----
IB_IFNAME=$(ip -br addr | awk '/^ib[0-9]+\s+UP\s+[0-9]/ {print $1; exit}')
IB_IFNAME="${IB_IFNAME:-ib0}"

# ---- NCCL / Gloo environment ----
export NCCL_SOCKET_IFNAME="${IB_IFNAME}"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export NCCL_TIMEOUT=1800
export GLOO_SOCKET_IFNAME="${IB_IFNAME}"

mkdir -p logs

echo "=== KempnerForge 7B Training (preemptible) ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Nodes:        ${NNODES} (${SLURM_JOB_NODELIST})"
echo "GPUs/node:    ${NGPUS_PER_NODE}"
echo "Total GPUs:   $((NNODES * NGPUS_PER_NODE))"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "IB interface: ${IB_IFNAME}"
echo "Config:       ${CONFIG}"
echo "Overrides:    ${OVERRIDES}"
echo "Restart cnt:  ${SLURM_RESTART_COUNT:-0}"
echo "=================================================="

srun --kill-on-bad-exit=1 \
    uv run python scripts/train.py "${CONFIG}" ${OVERRIDES}
