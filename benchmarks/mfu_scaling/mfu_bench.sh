#!/bin/bash
# MFU Benchmark Suite — runs all experiments and saves output.
# Usage: bash scripts/mfu_bench.sh
set -uo pipefail

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
RESULTS_DIR="mfu_results"
mkdir -p "$RESULTS_DIR"

DATA="/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/fineweb-edu/tokenized/meta-llama-3/default"
DATA_ARGS="--data.dataset_path=$DATA --data.file_pattern=tokenized_*.bin"
COMMON="--metrics.log_interval=1 --metrics.enable_wandb=false --metrics.enable_tensorboard=false --checkpoint.interval=99999"

STEPS=20

# 7B base config
M7B="--model.dim=4096 --model.n_layers=32 --model.n_heads=32 --model.n_kv_heads=8 --model.vocab_size=128256 --model.ffn_dim_multiplier=1.3 --model.max_seq_len=4096 --model.rope_theta=500000.0"
# 13B base config
M13B="--model.dim=5120 --model.n_layers=40 --model.n_heads=40 --model.n_kv_heads=8 --model.vocab_size=128256 --model.ffn_dim_multiplier=1.3 --model.max_seq_len=4096 --model.rope_theta=500000.0"
# 70B base config
M70B="--model.dim=8192 --model.n_layers=80 --model.n_heads=64 --model.n_kv_heads=8 --model.vocab_size=128256 --model.ffn_hidden_dim=28672 --model.max_seq_len=4096 --model.rope_theta=500000.0"

TRAIN="--train.seed=42 --train.grad_clip_norm=1.0 --train.activation_checkpointing=full --optimizer.lr=3e-4 --optimizer.fused=true --scheduler.warmup_steps=5"

run_experiment() {
    local name="$1"
    local mode="$2"     # "torchrun:N" or "srun:NODES:NTASKS"
    local args="$3"
    local outfile="$RESULTS_DIR/${name}.log"

    echo ">>> Running: $name"

    if [[ "$mode" == torchrun:* ]]; then
        local nproc="${mode#torchrun:}"
        uv run torchrun --nproc_per_node="$nproc" \
            scripts/train.py configs/train/debug.toml \
            $args $COMMON $DATA_ARGS \
            > "$outfile" 2>&1
    elif [[ "$mode" == srun:* ]]; then
        local rest="${mode#srun:}"
        local nodes="${rest%%:*}"
        local ntasks="${rest#*:}"
        export MASTER_PORT=$(comm -23 <(seq 15000 20000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
        sleep 3  # let previous NCCL sockets drain
        srun --nodes="$nodes" --ntasks="$ntasks" --gpus-per-node=4 --cpus-per-task=4 \
            uv run python scripts/train.py configs/train/debug.toml \
            $args $COMMON $DATA_ARGS \
            > "$outfile" 2>&1
    fi

    local rc=$?
    if [ $rc -eq 0 ]; then
        # Extract last logged step metrics
        local last=$(grep '\[step' "$outfile" | tail -1)
        echo "    OK: $last"
    else
        echo "    FAILED (exit $rc)"
        tail -5 "$outfile"
    fi
    echo ""
}

echo "============================================="
echo "  KempnerForge MFU Benchmark"
echo "  Nodes: $SLURM_NNODES × 4 H200 GPUs"
echo "  Data: fineweb-edu (Llama-3 tokenized)"
echo "  Steps: $STEPS per experiment"
echo "============================================="
echo ""

# =====================================================================
# 1 GPU — 7B baseline
# =====================================================================
run_experiment "01_1gpu_7b" "torchrun:1" \
    "$M7B $TRAIN --train.max_steps=$STEPS --train.batch_size=2 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=true --distributed.dp_shard=1"

# =====================================================================
# 2 GPUs — 7B
# =====================================================================
run_experiment "02_2gpu_7b_fsdp2" "torchrun:2" \
    "$M7B $TRAIN --train.max_steps=$STEPS --train.batch_size=2 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=true --distributed.dp_shard=2"

run_experiment "03_2gpu_7b_tp2" "torchrun:2" \
    "$M7B $TRAIN --train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=2 --train.compile_model=true --distributed.tp=2 --distributed.dp_shard=1"

# =====================================================================
# 4 GPUs (1 node) — 7B
# =====================================================================
run_experiment "04_4gpu_7b_fsdp4" "torchrun:4" \
    "$M7B $TRAIN --train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=2 --train.compile_model=true --distributed.dp_shard=4"

run_experiment "05_4gpu_7b_tp4" "torchrun:4" \
    "$M7B $TRAIN --train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=2 --train.compile_model=true --distributed.tp=4 --distributed.dp_shard=1"

run_experiment "06_4gpu_7b_tp2_fsdp2" "torchrun:4" \
    "$M7B $TRAIN --train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=2 --train.compile_model=true --distributed.tp=2 --distributed.dp_shard=2"

# =====================================================================
# 8 GPUs (2 nodes) — 7B and 13B
# =====================================================================
run_experiment "07_8gpu_7b_tp4_fsdp2" "srun:2:8" \
    "$M7B $TRAIN --train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=2 --train.compile_model=true --distributed.tp=4 --distributed.dp_shard=2"

run_experiment "08_8gpu_13b_fsdp8" "srun:2:8" \
    "$M13B $TRAIN --train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=false --distributed.dp_shard=8"

run_experiment "09_8gpu_13b_tp4_fsdp2" "srun:2:8" \
    "$M13B $TRAIN --train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=true --distributed.tp=4 --distributed.dp_shard=2"

# =====================================================================
# 16 GPUs (4 nodes) — 7B, 13B
# =====================================================================
run_experiment "10_16gpu_7b_tp4_fsdp4" "srun:4:16" \
    "$M7B $TRAIN --train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=2 --train.compile_model=true --distributed.tp=4 --distributed.dp_shard=4"

run_experiment "11_16gpu_13b_tp4_fsdp4" "srun:4:16" \
    "$M13B $TRAIN --train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=true --distributed.tp=4 --distributed.dp_shard=4"

# =====================================================================
# 32 GPUs (8 nodes) — 7B, 13B, 70B
# =====================================================================
run_experiment "12_32gpu_7b_tp4_fsdp8" "srun:8:32" \
    "$M7B $TRAIN --train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=2 --train.compile_model=true --distributed.tp=4 --distributed.dp_shard=8"

run_experiment "13_32gpu_13b_tp4_fsdp8" "srun:8:32" \
    "$M13B $TRAIN --train.max_steps=$STEPS --train.batch_size=4 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=true --distributed.tp=4 --distributed.dp_shard=8"

run_experiment "14_32gpu_70b_tp4_fsdp8" "srun:8:32" \
    "$M70B $TRAIN --train.max_steps=$STEPS --train.batch_size=2 --train.seq_len=4096 --train.grad_accum_steps=4 --train.compile_model=true --distributed.tp=4 --distributed.dp_shard=8"

echo "============================================="
echo "  All experiments complete."
echo "  Results in: $RESULTS_DIR/"
echo "============================================="
