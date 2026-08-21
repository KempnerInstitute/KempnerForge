#!/bin/bash
# Weak-scaling sweep for dense 13B / 70B from 8 to 160 H200 GPUs.
#
# Submits every config as its OWN SLURM job: one config = one allocation. That
# matters. A single large allocation running each config over an srun node subset
# hangs, because MASTER_ADDR is the full allocation's first node and a subset may
# not contain it, leaving the rendezvous host unreachable. One job per config
# makes the allocation and the config the same node set, so rank 0 always lives
# on the master node. Configs also queue independently, so one bad config cannot
# block the rest.
#
# Parallelism is held fixed at TP=4 (intra-node, NVLink) x FSDP across nodes, and
# per-GPU batch is held constant, so the total problem size grows with the machine.
# That makes this weak scaling, not strong scaling.
#
# Throughput and MFU only: 30 steps on the scaling points, 200 on the headlines.
# No quality claims.
#
# Required env:
#   DATA       pre-tokenized FineWeb-Edu (Llama-3), matching file_pattern tokenized_*.bin
#   PARTITION  SLURM partition with 4 GPUs per node
#   ACCOUNT    SLURM account to charge
# Optional env:
#   QOS               SLURM QoS, if your site requires one
#   KEMPNERFORGE_ROOT repo checkout to run from (default: two levels up from here)
#   RESULTS_DIR       where logs land (default: ./results beside this script)
#   TIME              walltime per job (default 2:00:00)
#   EXCLUDE           comma-separated nodes to avoid
#
# Usage:
#   DATA=... PARTITION=... ACCOUNT=... bash weak_scaling_160gpu_bench.sh
#   ... bash weak_scaling_160gpu_bench.sh scaling    # only the curve points
#   ... bash weak_scaling_160gpu_bench.sh headline   # only the 160-GPU repeats
#   ... bash weak_scaling_160gpu_bench.sh 8          # only 8-node configs (validate first)
#
# Then:  uv run python parse_results.py results
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

# Model architectures. These match benchmarks/mfu_scaling/mfu_bench.sh so the
# 32-GPU points are comparable with the earlier 1-32 GPU sweep.
M13B="--model.dim=5120 --model.n_layers=40 --model.n_heads=40 --model.n_kv_heads=8 --model.vocab_size=128256 --model.ffn_dim_multiplier=1.3 --model.max_seq_len=4096 --model.rope_theta=500000.0"
M70B="--model.dim=8192 --model.n_layers=80 --model.n_heads=64 --model.n_kv_heads=8 --model.vocab_size=128256 --model.ffn_hidden_dim=28672 --model.max_seq_len=4096 --model.rope_theta=500000.0"

# Per-GPU batch, held constant across every GPU count. This is what makes the
# sweep weak scaling: tokens per step grows linearly with the machine.
B13="--train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=true"
B70="--train.batch_size=2 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=true"

# name | nodes | train.py args | steps | tag
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
# The 8-GPU points in the report came from a 2-node pre-flight of the same two
# configs at dp_shard=2. Run those first to validate the path before spending a
# 40-node window:  bash weak_scaling_160gpu_bench.sh 2
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
