#!/bin/bash

set -uo pipefail

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
RESULTS_DIR="moe_packed_results"
mkdir -p "$RESULTS_DIR"

DATA="${DATA:?Set DATA to your tokenized dataset path (e.g. /path/to/tokenized/llama3)}"
DATA_ARGS="--data.dataset_path=$DATA --data.file_pattern=tokenized_*.bin"
COMMON="--metrics.log_interval=1 --metrics.enable_wandb=false --metrics.enable_tensorboard=false --checkpoint.interval=99999"

STEPS=20

MOE_MODEL_BASE="--model.dim=2048 --model.n_layers=24 --model.n_heads=16 --model.n_kv_heads=4 --model.vocab_size=128256 --model.ffn_dim_multiplier=1.3 --model.max_seq_len=4096 --model.rope_theta=500000.0 --model.moe_top_k=2 --model.moe_frequency=1 --model.moe_router=softmax_topk --model.moe_aux_loss_weight=0.01 --model.moe_capacity_factor=0.0"

TRAIN="--train.seed=42 --train.grad_clip_norm=1.0 --optimizer.lr=3e-4 --optimizer.fused=true --scheduler.warmup_steps=5"

run_experiment() {
    local name="$1"
    local mode="$2"
    local args="$3"
    local outfile="$RESULTS_DIR/${name}.log"

    echo ">>> Running: $name"

    local rest="${mode#srun:}"
    local nodes="${rest%%:*}"
    local ntasks="${rest#*:}"
    export MASTER_PORT=$(comm -23 <(seq 15000 20000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    sleep 3
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
echo "  KempnerForge MoE Expert Packing Bench"
echo "  Nodes: $SLURM_NNODES x 4 H200 GPUs"
echo "  Data: fineweb-edu (Llama-3 tokenized)"
echo "  Steps: $STEPS per experiment"
echo "============================================="
echo ""

TRAIN_COMMON="--train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=1 --train.compile_model=false --train.mixed_precision=bf16 --train.activation_checkpointing=full"

run_experiment "01_8gpu_moe_e8_fsdp8_unpacked" "srun:2:8" \
    "$MOE_MODEL_BASE --model.num_experts=8 --model.moe_packed_experts=false $TRAIN $TRAIN_COMMON --distributed.tp=1 --distributed.ep=1 --distributed.dp_shard=8"

run_experiment "02_8gpu_moe_e8_fsdp8_packed" "srun:2:8" \
    "$MOE_MODEL_BASE --model.num_experts=8 --model.moe_packed_experts=true $TRAIN $TRAIN_COMMON --distributed.tp=1 --distributed.ep=1 --distributed.dp_shard=8"

run_experiment "03_8gpu_moe_e16_fsdp8_unpacked" "srun:2:8" \
    "$MOE_MODEL_BASE --model.num_experts=16 --model.moe_packed_experts=false $TRAIN $TRAIN_COMMON --distributed.tp=1 --distributed.ep=1 --distributed.dp_shard=8"

run_experiment "04_8gpu_moe_e16_fsdp8_packed" "srun:2:8" \
    "$MOE_MODEL_BASE --model.num_experts=16 --model.moe_packed_experts=true $TRAIN $TRAIN_COMMON --distributed.tp=1 --distributed.ep=1 --distributed.dp_shard=8"

run_experiment "05_8gpu_moe_e64_fsdp2_ep4_unpacked" "srun:2:8" \
    "$MOE_MODEL_BASE --model.num_experts=64 --model.moe_packed_experts=false $TRAIN $TRAIN_COMMON --distributed.tp=1 --distributed.ep=4 --distributed.dp_shard=2"

run_experiment "06_8gpu_moe_e64_fsdp2_ep4_packed" "srun:2:8" \
    "$MOE_MODEL_BASE --model.num_experts=64 --model.moe_packed_experts=true $TRAIN $TRAIN_COMMON --distributed.tp=1 --distributed.ep=4 --distributed.dp_shard=2"

echo "============================================="
echo "  All experiments complete."
echo "  Results in: $RESULTS_DIR/"
echo "============================================="
