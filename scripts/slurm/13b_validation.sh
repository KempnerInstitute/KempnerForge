#!/bin/bash
#SBATCH --job-name=13b-validation
#SBATCH --partition=<partition-name>     # e.g. gpu, kempner_dev
#SBATCH --account=<account-name>         # your SLURM account
#SBATCH --constraint=<gpu-type>          # e.g. h200, h100, a100
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --signal=B:SIGTERM@120
#
# Full validation run: 13B model on 24 H200 GPUs (6 nodes x 4).
# Tests: training, TP+FSDP, eval pipeline, checkpointing, profiling.

set -euo pipefail

CONFIG="configs/train/13b_24gpu_validation.toml"

NNODES="${SLURM_NNODES}"
NGPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=$(comm -23 <(seq 15000 20000 | sort) \
    <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

IB_IFNAME=$(ip -br addr | awk '/^ib[0-9]+\s+UP\s+[0-9]/ {print $1; exit}')
IB_IFNAME="${IB_IFNAME:-ib0}"

export NCCL_SOCKET_IFNAME="${IB_IFNAME}"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export GLOO_SOCKET_IFNAME="${IB_IFNAME}"

mkdir -p logs

echo "=== 13B Validation Run ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Nodes:        ${NNODES} (${SLURM_JOB_NODELIST})"
echo "GPUs/node:    ${NGPUS_PER_NODE}"
echo "Total GPUs:   $((NNODES * NGPUS_PER_NODE))"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "IB interface: ${IB_IFNAME}"
echo "=========================="

srun --kill-on-bad-exit=1 \
    uv run python scripts/train.py "${CONFIG}"
