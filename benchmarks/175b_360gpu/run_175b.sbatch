#!/bin/bash
# The 175B / 360-GPU benchmark job. Not meant to be submitted by hand:
# 175b_360gpu_bench.sh sets --nodes, --partition, --account, --time, --output and
# the environment below on the sbatch CLI.
#
#   sbatch --nodes=90 ... run_175b.sbatch <name> "<train.py args>" <steps> <ckpt_interval>
#
# Environment (exported by the driver):
#   DATA         tokenized dataset directory
#   REPO         KempnerForge checkout to run from
#   CAMPAIGN_DIR this directory (SLURM spools the batch script, so it cannot
#                locate itself reliably and is told instead)
#   RESULTS_DIR  where the log is written
#
# --requeue and --signal are deliberately NOT set. They belong to a preemption or
# MTTR experiment; on a plain long benchmark they can trigger an unwanted requeue
# at walltime and silently restart an 11-hour run.
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -uo pipefail

NAME="${1:?usage: run_175b.sbatch <name> \"<args>\" <steps> <ckpt_interval>}"
ARGS="${2:?model + parallelism args}"
STEPS="${3:?steps}"
CKPT_INT="${4:-300}"
DATA="${DATA:?export DATA=/path/to/tokenized}"
REPO="${REPO:?export REPO=/path/to/KempnerForge}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:?export CAMPAIGN_DIR=/path/to/this/directory}"
RESULTS_DIR="${RESULTS_DIR:-$CAMPAIGN_DIR/results}"
export REPO

# NCCL over InfiniBand. The errno-97 IPv6 warnings are benign; it falls back to IPv4.
IB=$(ip -br addr | awk '/^ib[0-9]+\s+UP\s+[0-9]/ {print $1; exit}')
export NCCL_SOCKET_IFNAME="${IB:-ib0}" GLOO_SOCKET_IFNAME="${IB:-ib0}"
export NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_IB_GID_INDEX=3
# Load-bearing at this size: without expandable segments, allocator fragmentation
# under bf16 + full activation checkpointing pushes peak memory over the limit.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Checkpoints are large here: bf16 weights ~350 GB plus fp32 optimizer moments
# ~2.1 TB across the job. Point CKPT_DIR at scratch with room for keep_last_n=1.
CKPT="${CKPT_DIR:-${TMPDIR:-/tmp}}/kf_175b_ckpt_${SLURM_JOB_ID}"
COMMON="--metrics.log_interval=1 --metrics.enable_wandb=false --checkpoint.interval=${CKPT_INT} --checkpoint.dir=$CKPT"
TRAIN="--train.seed=42 --train.grad_clip_norm=1.0 --train.activation_checkpointing=full --optimizer.lr=3e-4 --optimizer.fused=true --scheduler.warmup_steps=5"
DATA_ARGS="--data.dataset_path=$DATA --data.file_pattern=tokenized_*.bin"
OUT="$RESULTS_DIR/${NAME}.log"
mkdir -p "$RESULTS_DIR"
cd "$REPO"

echo "=== $NAME | job ${SLURM_JOB_ID:-?} | ${SLURM_NNODES:-?} nodes x4 GPUs | ckpt_int=$CKPT_INT | $(date) ==="
echo "args: $ARGS --train.max_steps=$STEPS"

# --ntasks-per-node=4 with block distribution so ranks 0-3 land on node 0 and
# LOCAL_RANK maps 1:1 to the four local GPUs. Plain --ntasks=N lets SLURM
# distribute cyclically, which collides two ranks on one GPU.
srun --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --distribution=block:block \
    "$CAMPAIGN_DIR/_srun_train.sh" \
    $ARGS --train.max_steps="$STEPS" $TRAIN $COMMON $DATA_ARGS > "$OUT" 2>&1
rc=$?

echo "[$NAME] exit=$rc | $(grep '\[step' "$OUT" | tail -1)"
exit $rc
