#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${KEMPNERFORGE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
SB="$SCRIPT_DIR/run_175b.sbatch"

DATA="${DATA:?Set DATA to your tokenized dataset directory}"
PARTITION="${PARTITION:?Set PARTITION to a GPU partition with 4 GPUs per node}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your SLURM account}"
QOS="${QOS:-}"
NODES="${NODES:-90}"
STEPS="${STEPS:-2000}"
CKPT="${CKPT:-300}"
TIME="${TIME:-13:00:00}"
EXCLUDE="${EXCLUDE:-}"
NAME="175b-360gpu"

GPUS=$((NODES * 4))
DP_SHARD="$NODES"

M175B="--model.dim=12288 --model.n_layers=96 --model.n_heads=96 --model.n_kv_heads=8 --model.vocab_size=128256 --model.ffn_hidden_dim=39680 --model.max_seq_len=4096 --model.rope_theta=500000.0"
B175="--train.batch_size=8 --train.seq_len=4096 --train.grad_accum_steps=1 --train.compile_model=true --distributed.tp=4 --distributed.dp_shard=$DP_SHARD"
FT="--checkpoint.async_mode=async --checkpoint.keep_last_n=1 --train.nccl_health_check_interval=50"
ARGS="$M175B $B175 $FT"

mkdir -p "$RESULTS_DIR"

echo "175B dense (RoPE/GQA/SwiGLU/RMSNorm) | TP=4 x FSDP${DP_SHARD} = ${GPUS} GPUs on ${NODES} nodes"
echo "  steps=$STEPS  async ckpt every $CKPT  health check every 50  walltime=$TIME"
echo "  results -> $RESULTS_DIR/${NAME}.log"

if [ "${1:-}" != "GO" ]; then
  echo
  echo "DRY RUN, nothing submitted. Submit with:  bash $(basename "$0") GO"
  exit 0
fi

jid=$(sbatch --parsable \
  --nodes="$NODES" --job-name="kf-$NAME" --time="$TIME" \
  --partition="$PARTITION" --account="$ACCOUNT" \
  ${QOS:+--qos="$QOS"} ${EXCLUDE:+--exclude="$EXCLUDE"} \
  --output="$RESULTS_DIR/${NAME}.slurm-%j.out" \
  --error="$RESULTS_DIR/${NAME}.slurm-%j.err" \
  --export=ALL,DATA="$DATA",REPO="$REPO_ROOT",CAMPAIGN_DIR="$SCRIPT_DIR",RESULTS_DIR="$RESULTS_DIR" \
  "$SB" "$NAME" "$ARGS" "$STEPS" "$CKPT")
echo "submitted $jid : $NAME ($NODES nodes)"
echo
echo "watch the first ~10 steps and kill early on OOM or hang:"
echo "  tail -f $RESULTS_DIR/${NAME}.log"
echo "then:"
echo "  uv run python $SCRIPT_DIR/parse_results.py $RESULTS_DIR/${NAME}.log"
echo "  uv run python $SCRIPT_DIR/make_figures.py  $RESULTS_DIR/${NAME}.log $SCRIPT_DIR/figures"
