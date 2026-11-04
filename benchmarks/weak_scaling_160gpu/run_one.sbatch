#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -uo pipefail

NAME="${1:?usage: run_one.sbatch <name> \"<args>\" <steps>}"
ARGS="${2:?model + parallelism args}"
STEPS="${3:?steps}"
DATA="${DATA:?export DATA=/path/to/tokenized}"
REPO="${REPO:?export REPO=/path/to/KempnerForge}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:?export CAMPAIGN_DIR=/path/to/this/directory}"
RESULTS_DIR="${RESULTS_DIR:-$CAMPAIGN_DIR/results}"
export REPO

IB=$(ip -br addr | awk '/^ib[0-9]+\s+UP\s+[0-9]/ {print $1; exit}')
export NCCL_SOCKET_IFNAME="${IB:-ib0}" GLOO_SOCKET_IFNAME="${IB:-ib0}"
export NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_IB_GID_INDEX=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT_DIR:-${TMPDIR:-/tmp}}/kf_mfu_ckpt_${SLURM_JOB_ID}"
COMMON="--metrics.log_interval=1 --metrics.enable_wandb=false --metrics.enable_tensorboard=false --checkpoint.interval=99999 --checkpoint.dir=$CKPT"
TRAIN="--train.seed=42 --train.grad_clip_norm=1.0 --train.activation_checkpointing=full --optimizer.lr=3e-4 --optimizer.fused=true --scheduler.warmup_steps=5"
DATA_ARGS="--data.dataset_path=$DATA --data.file_pattern=tokenized_*.bin"
OUT="$RESULTS_DIR/${NAME}.log"
mkdir -p "$RESULTS_DIR/pulse"
cd "$REPO"

echo "=== $NAME | job ${SLURM_JOB_ID:-?} | ${SLURM_NNODES:-?} nodes x4 GPUs | $(date) ==="
echo "args: $ARGS --train.max_steps=$STEPS"

srun --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --distribution=block:block \
    "$CAMPAIGN_DIR/_srun_train.sh" \
    $ARGS --train.max_steps="$STEPS" $TRAIN $COMMON $DATA_ARGS > "$OUT" 2>&1 &
TPID=$!

if [ -x "$CAMPAIGN_DIR/pulse_capture.sh" ]; then
    ( sleep 120; bash "$CAMPAIGN_DIR/pulse_capture.sh" "$NAME" "$RESULTS_DIR/pulse" ) &
    PULSE_PID=$!
fi

wait "$TPID"; rc=$?
[ -n "${PULSE_PID:-}" ] && wait "$PULSE_PID" 2>/dev/null || true
echo "[$NAME] exit=$rc | $(grep '\[step' "$OUT" | tail -1)"
exit $rc
