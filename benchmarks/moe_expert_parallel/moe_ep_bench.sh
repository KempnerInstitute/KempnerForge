#!/bin/bash
# MoE Expert Parallelism Benchmark Suite
#
# Tests EP+TP+FSDP2 composition at various batch sizes and configurations.
# Requires 8 nodes with 4 H200 (141 GB) GPUs each (32 GPUs total).
#
# Usage:
#   salloc -p <partition> --account=<account> \
#       --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 \
#       --cpus-per-task=16 --mem=1490G -t 00-04:00:00
#   bash benchmarks/moe_expert_parallel/moe_ep_bench.sh

set -uo pipefail

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
RESULTS_DIR="moe_ep_results"
mkdir -p "$RESULTS_DIR"

DATA="${DATA:?Set DATA to your tokenized dataset path (e.g. /path/to/tokenized/llama3)}"
DATA_ARGS="--data.dataset_path=$DATA --data.file_pattern=tokenized_*.bin"
COMMON="--metrics.log_interval=1 --metrics.enable_wandb=false --metrics.enable_tensorboard=false --checkpoint.interval=99999"

STEPS=20

# ~4B total / ~1.8B active MoE model
MOE_MODEL="--model.dim=2048 --model.n_layers=24 --model.n_heads=16 --model.n_kv_heads=4 --model.vocab_size=128256 --model.ffn_dim_multiplier=1.3 --model.max_seq_len=4096 --model.rope_theta=500000.0 --model.num_experts=8 --model.moe_top_k=2 --model.moe_frequency=1 --model.moe_router=softmax_topk --model.moe_aux_loss_weight=0.01 --model.moe_capacity_factor=0.0"

TRAIN="--train.seed=42 --train.grad_clip_norm=1.0 --optimizer.lr=3e-4 --optimizer.fused=true --scheduler.warmup_steps=5"

run_experiment() {
    local name="$1"
    local mode="$2"     # "srun:NODES:NTASKS"
    local args="$3"
    local outfile="$RESULTS_DIR/${name}.log"

    echo ">>> Running: $name"

    local rest="${mode#srun:}"
    local nodes="${rest%%:*}"
    local ntasks="${rest#*:}"
    export MASTER_PORT=$(comm -23 <(seq 15000 20000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    sleep 3  # let previous NCCL sockets drain
    srun --nodes="$nodes" --ntasks="$ntasks" --gpus-per-node=4 --cpus-per-task=4 \
        --kill-on-bad-exit=1 \
        --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python scripts/train.py configs/train/debug.toml \
        $args $COMMON $DATA_ARGS \
        > "$outfile" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        local last=$(grep '\[step' "$outfile" | tail -1)
        echo "    OK: $last"
    else
        echo "    FAILED (exit $rc)"
        tail -5 "$outfile"
    fi
    echo ""
}

echo "============================================="
echo "  KempnerForge MoE Expert Parallelism Bench"
echo "  Nodes: $SLURM_NNODES x 4 H200 GPUs"
echo "  Data: fineweb-edu (Llama-3 tokenized)"
echo "  Steps: $STEPS per experiment"
echo "============================================="
echo ""

# =====================================================================
# Experiment 1: EP=2, TP=4, FSDP=4, batch_size=8, full AC (best config)
# =====================================================================
run_experiment "01_32gpu_moe_ep2_tp4_fsdp4_bs8_fullac" "srun:8:32" \
    "$MOE_MODEL $TRAIN --train.max_steps=$STEPS --train.batch_size=8 --train.seq_len=4096 --train.grad_accum_steps=1 --train.compile_model=false --train.mixed_precision=bf16 --train.activation_checkpointing=full --distributed.tp=4 --distributed.ep=2 --distributed.dp_shard=4"

# =====================================================================
# Experiment 2: EP=2, batch_size=2, full AC (low batch baseline)
# =====================================================================
run_experiment "02_32gpu_moe_ep2_tp4_fsdp4_bs2_fullac" "srun:8:32" \
    "$MOE_MODEL $TRAIN --train.max_steps=$STEPS --train.batch_size=2 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=false --train.mixed_precision=bf16 --train.activation_checkpointing=full --distributed.tp=4 --distributed.ep=2 --distributed.dp_shard=4"

# =====================================================================
# Experiment 3: EP=2, batch_size=12, full AC (high memory pressure)
# =====================================================================
run_experiment "03_32gpu_moe_ep2_tp4_fsdp4_bs12_fullac" "srun:8:32" \
    "$MOE_MODEL $TRAIN --train.max_steps=$STEPS --train.batch_size=12 --train.seq_len=4096 --train.grad_accum_steps=1 --train.compile_model=false --train.mixed_precision=bf16 --train.activation_checkpointing=full --distributed.tp=4 --distributed.ep=2 --distributed.dp_shard=4"

# =====================================================================
# Experiment 4: EP=2, batch_size=1, selective AC (original config)
# =====================================================================
run_experiment "04_32gpu_moe_ep2_tp4_fsdp4_bs1_selectiveac" "srun:8:32" \
    "$MOE_MODEL $TRAIN --train.max_steps=$STEPS --train.batch_size=1 --train.seq_len=4096 --train.grad_accum_steps=8 --train.compile_model=false --train.mixed_precision=bf16 --train.activation_checkpointing=selective --distributed.tp=4 --distributed.ep=2 --distributed.dp_shard=4"

# =====================================================================
# Experiment 5: NO EP (baseline), TP=4, FSDP=8, batch_size=8, full AC
# =====================================================================
run_experiment "05_32gpu_moe_noep_tp4_fsdp8_bs8_fullac" "srun:8:32" \
    "$MOE_MODEL $TRAIN --train.max_steps=$STEPS --train.batch_size=8 --train.seq_len=4096 --train.grad_accum_steps=1 --train.compile_model=false --train.mixed_precision=bf16 --train.activation_checkpointing=full --distributed.tp=4 --distributed.ep=1 --distributed.dp_shard=8"

# =====================================================================
# Experiment 6: NO EP, TP=4, FSDP=8, batch_size=8, selective AC
# =====================================================================
run_experiment "06_32gpu_moe_noep_tp4_fsdp8_bs8_selectiveac" "srun:8:32" \
    "$MOE_MODEL $TRAIN --train.max_steps=$STEPS --train.batch_size=8 --train.seq_len=4096 --train.grad_accum_steps=1 --train.compile_model=false --train.mixed_precision=bf16 --train.activation_checkpointing=selective --distributed.tp=4 --distributed.ep=1 --distributed.dp_shard=8"

echo "============================================="
echo "  All experiments complete."
echo "  Results in: $RESULTS_DIR/"
echo "============================================="
