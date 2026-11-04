#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${KEMPNERFORGE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
SB="$SCRIPT_DIR/run_one.sbatch"

DATA="${DATA:?Set DATA to your tokenized dataset directory}"
PARTITION="${PARTITION:?Set PARTITION to a GPU partition with 4 GPUs per node}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your SLURM account}"
QOS="${QOS:-}"
TIME="${TIME:-2:00:00}"
EXCLUDE="${EXCLUDE:-}"
FILTER="${1:-all}"

mkdir -p "$RESULTS_DIR/pulse"

M13B="--model.dim=5120 --model.n_layers=40 --model.n_heads=40 --model.n_kv_heads=8 --model.vocab_size=128256 --model.ffn_dim_multiplier=1.3 --model.max_seq_len=4096 --model.rope_theta=500000.0"
M70B="--model.dim=8192 --model.n_layers=80 --model.n_heads=64 --model.n_kv_heads=8 --model.vocab_size=128256 --model.ffn_hidden_dim=28672 --model.max_seq_len=4096 --model.rope_theta=500000.0"

B13="--train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=true"
B70="--train.batch_size=2 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=true"

CONFIGS=(
  "70b_032gpu_tp4_fsdp8|8|$M70B $B70 --distributed.tp=4 --distributed.dp_shard=8|30|scaling"
  "70b_064gpu_tp4_fsdp16|16|$M70B $B70 --distributed.tp=4 --distributed.dp_shard=16|30|scaling"
  "70b_096gpu_tp4_fsdp24|24|$M70B $B70 --distributed.tp=4 --distributed.dp_shard=24|30|scaling"
  "70b_128gpu_tp4_fsdp32|32|$M70B $B70 --distributed.tp=4 --distributed.dp_shard=32|30|scaling"
  "70b_160gpu_tp4_fsdp40_r1|40|$M70B $B70 --distributed.tp=4 --distributed.dp_shard=40|200|headline"
  "70b_160gpu_tp4_fsdp40_r2|40|$M70B $B70 --distributed.tp=4 --distributed.dp_shard=40|200|headline"
  "13b_032gpu_tp4_fsdp8|8|$M13B $B13 --distributed.tp=4 --distributed.dp_shard=8|30|scaling"
  "13b_064gpu_tp4_fsdp16|16|$M13B $B13 --distributed.tp=4 --distributed.dp_shard=16|30|scaling"
  "13b_128gpu_tp4_fsdp32|32|$M13B $B13 --distributed.tp=4 --distributed.dp_shard=32|30|scaling"
  "13b_160gpu_tp4_fsdp40_r1|40|$M13B $B13 --distributed.tp=4 --distributed.dp_shard=40|200|headline"
  "13b_160gpu_tp4_fsdp40_r2|40|$M13B $B13 --distributed.tp=4 --distributed.dp_shard=40|200|headline"
)
CONFIGS+=(
  "preflight_13b_008gpu_tp4_fsdp2|2|$M13B $B13 --distributed.tp=4 --distributed.dp_shard=2|30|preflight"
  "preflight_70b_008gpu_tp4_fsdp2|2|$M70B $B70 --distributed.tp=4 --distributed.dp_shard=2|30|preflight"
)

n=0
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name nodes args steps tag <<< "$c"
  case "$FILTER" in
    all) ;;
    scaling|headline|preflight) [ "$tag" = "$FILTER" ] || continue ;;
    [0-9]*) [ "$nodes" = "$FILTER" ] || continue ;;
    *) echo "unknown filter '$FILTER' (all|scaling|headline|preflight|<nodes>)" >&2; exit 2 ;;
  esac
  jid=$(sbatch --parsable \
    --nodes="$nodes" --job-name="kf-$name" --time="$TIME" \
    --partition="$PARTITION" --account="$ACCOUNT" \
    ${QOS:+--qos="$QOS"} ${EXCLUDE:+--exclude="$EXCLUDE"} \
    --output="$RESULTS_DIR/${name}.slurm-%j.out" \
    --error="$RESULTS_DIR/${name}.slurm-%j.err" \
    --export=ALL,DATA="$DATA",REPO="$REPO_ROOT",CAMPAIGN_DIR="$SCRIPT_DIR",RESULTS_DIR="$RESULTS_DIR" \
    "$SB" "$name" "$args" "$steps")
  echo "submitted $jid : $name ($nodes nodes, $steps steps, $tag)"
  n=$((n + 1))
done
echo "--- $n job(s) submitted ---"
echo "parse when they finish:  uv run python $SCRIPT_DIR/parse_results.py $RESULTS_DIR"
