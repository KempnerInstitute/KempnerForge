#!/bin/bash
# ONE benchmark config = ONE SLURM job. The allocation IS the config's node set,
# so the whole allocation runs a single srun and the rendezvous master is rank 0's
# own node. See the comment in weak_scaling_160gpu_bench.sh for why a shared
# allocation with srun subsets hangs instead.
#
# Not meant to be submitted by hand. weak_scaling_160gpu_bench.sh sets --nodes,
# --partition, --account, --output and the environment below on the sbatch CLI.
#
#   sbatch --nodes=N ... run_one.sbatch <name> "<train.py args>" <steps>
#
# Environment (exported by the driver):
#   DATA         tokenized dataset directory
#   REPO         KempnerForge checkout to run from
#   CAMPAIGN_DIR this directory (SLURM spools the batch script, so the script
#                cannot locate itself reliably and is told instead)
#   RESULTS_DIR  where the per-config log is written
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

# NCCL over InfiniBand. The errno-97 IPv6 warnings are benign; it falls back to IPv4.
IB=$(ip -br addr | awk '/^ib[0-9]+\s+UP\s+[0-9]/ {print $1; exit}')
export NCCL_SOCKET_IFNAME="${IB:-ib0}" GLOO_SOCKET_IFNAME="${IB:-ib0}"
export NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_IB_GID_INDEX=3
# Load-bearing at 70B: without expandable segments, allocator fragmentation under
# bf16 + full activation checkpointing pushes peak memory over the limit.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# A fresh, empty checkpoint directory per job, so auto-resume finds nothing and
# every config cold-starts. Without this, train.py resumes whatever debug
# checkpoint it finds and dies on a token_embedding shape mismatch. The interval
# exceeds max_steps, so nothing is ever actually written here.
CKPT="${CKPT_DIR:-${TMPDIR:-/tmp}}/kf_mfu_ckpt_${SLURM_JOB_ID}"
COMMON="--metrics.log_interval=1 --metrics.enable_wandb=false --metrics.enable_tensorboard=false --checkpoint.interval=99999 --checkpoint.dir=$CKPT"
TRAIN="--train.seed=42 --train.grad_clip_norm=1.0 --train.activation_checkpointing=full --optimizer.lr=3e-4 --optimizer.fused=true --scheduler.warmup_steps=5"
DATA_ARGS="--data.dataset_path=$DATA --data.file_pattern=tokenized_*.bin"
OUT="$RESULTS_DIR/${NAME}.log"
mkdir -p "$RESULTS_DIR/pulse"
cd "$REPO"

echo "=== $NAME | job ${SLURM_JOB_ID:-?} | ${SLURM_NNODES:-?} nodes x4 GPUs | $(date) ==="
echo "args: $ARGS --train.max_steps=$STEPS"

# --ntasks-per-node=4 with block distribution so ranks 0-3 land on node 0, 4-7 on
# node 1, and LOCAL_RANK maps 1:1 to the four local GPUs. Plain --ntasks=N lets
# SLURM distribute cyclically, which collides two ranks on one GPU
# ("Duplicate GPU detected").
srun --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --distribution=block:block \
    "$CAMPAIGN_DIR/_srun_train.sh" \
    $ARGS --train.max_steps="$STEPS" $TRAIN $COMMON $DATA_ARGS > "$OUT" 2>&1 &
TPID=$!

# Optional mid-run GPU telemetry. parse_results.py folds results/pulse/*.csv into
# the report if they exist and ignores them if they do not, so this is best-effort
# and its absence is not an error.
if [ -x "$CAMPAIGN_DIR/pulse_capture.sh" ]; then
    ( sleep 120; bash "$CAMPAIGN_DIR/pulse_capture.sh" "$NAME" "$RESULTS_DIR/pulse" ) &
    PULSE_PID=$!
fi

wait "$TPID"; rc=$?
[ -n "${PULSE_PID:-}" ] && wait "$PULSE_PID" 2>/dev/null || true
echo "[$NAME] exit=$rc | $(grep '\[step' "$OUT" | tail -1)"
exit $rc
