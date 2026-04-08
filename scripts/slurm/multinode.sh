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
#   sbatch --nodes=4 scripts/slurm/multinode.sh configs/train/7b.toml
#   sbatch --nodes=8 scripts/slurm/multinode.sh \
#       configs/train/7b.toml --train.max_steps=50000
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

# ---- Network interface detection ----
# Detect the first UP InfiniBand interface with an IP address.
# Verified across all Kempner node types (H200, H100, A100):
#   - H200/H100: 4x CX-7 NDR (ib0-ib3), only ib0 has IP
#   - A100: 1x CX-6 HDR (ib0), has IP
# Without explicit IFNAME, gloo binds to em4 (management Ethernet) which
# can cause timeouts for multi-node collectives (wrong subnet / firewall).
IB_IFNAME=$(ip -br addr | awk '/^ib[0-9]+\s+UP\s+[0-9]/ {print $1; exit}')
IB_IFNAME="${IB_IFNAME:-ib0}"

# ---- Environment ----
# NCCL: use IB for RDMA data transport, IB interface for OOB bootstrap socket.
# NCCL auto-discovers all active HCAs and selects the closest per GPU via
# PCIe topology — no need for NCCL_IB_HCA on multi-HCA nodes.
export NCCL_SOCKET_IFNAME="${IB_IFNAME}"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3

# Gloo: used by DCP async checkpoint for CPU-side coordination.
# Must bind to the same IB interface so peer connections route correctly.
export GLOO_SOCKET_IFNAME="${IB_IFNAME}"

# Timeout for NCCL operations (seconds) — increase for large clusters
export NCCL_TIMEOUT=1800

# Ensure log directory exists
mkdir -p logs

echo "=== KempnerForge Multi-Node Training ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Nodes:        ${NNODES} (${SLURM_JOB_NODELIST})"
echo "GPUs/node:    ${NGPUS_PER_NODE}"
echo "Total GPUs:   $((NNODES * NGPUS_PER_NODE))"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "IB interface: ${IB_IFNAME}"
echo "Config:       ${CONFIG}"
echo "Overrides:    ${OVERRIDES}"
echo "Restart cnt:  ${SLURM_RESTART_COUNT:-0}"
echo "========================================="

# ---- Launch with srun ----
# srun launches one process per GPU. Each process gets:
#   RANK=SLURM_PROCID, LOCAL_RANK=SLURM_LOCALID, WORLD_SIZE=SLURM_NTASKS
srun --kill-on-bad-exit=1 \
    uv run python scripts/train.py "${CONFIG}" ${OVERRIDES}
