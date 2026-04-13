#!/bin/bash
# Retry the E=64 cells with smaller batch to avoid OOM.
# Both cells in the main suite OOM in backward at batch_size=4;
# drop to batch_size=1 so we can get full 20-step tok/s numbers.

set -uo pipefail

# Resolve job id + nodelist so this works whether called from inside a job
# allocation or from a detached shell that just exports SLURM_JOB_ID.
if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
    SLURM_JOB_NODELIST=$(scontrol show job "$SLURM_JOB_ID" | awk '/^   NodeList=/ {sub("^   NodeList=",""); print; exit}')
    export SLURM_JOB_NODELIST
fi
# Avoid inheriting a too-small CPUs-per-task from an outer srun wrapper.
unset SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
RESULTS_DIR="moe_packed_results"
mkdir -p "$RESULTS_DIR"

DATA="/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/fineweb-edu/tokenized/meta-llama-3/default"
DATA_ARGS="--data.dataset_path=$DATA --data.file_pattern=tokenized_*.bin"
COMMON="--metrics.log_interval=1 --metrics.enable_wandb=false --metrics.enable_tensorboard=false --checkpoint.interval=99999"

STEPS=20
MOE_MODEL_BASE="--model.dim=2048 --model.n_layers=24 --model.n_heads=16 --model.n_kv_heads=4 --model.vocab_size=128256 --model.ffn_dim_multiplier=1.3 --model.max_seq_len=4096 --model.rope_theta=500000.0 --model.moe_top_k=2 --model.moe_frequency=1 --model.moe_router=softmax_topk --model.moe_aux_loss_weight=0.01 --model.moe_capacity_factor=0.0"
TRAIN="--train.seed=42 --train.grad_clip_norm=1.0 --optimizer.lr=3e-4 --optimizer.fused=true --scheduler.warmup_steps=5"

run_experiment() {
    local name="$1"
    local args="$2"
    local outfile="$RESULTS_DIR/${name}.log"

    echo ">>> Running: $name"
    export MASTER_PORT=$(comm -23 <(seq 15000 20000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    sleep 3
    srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=2 --ntasks=8 --gpus-per-node=4 --cpus-per-task=4 \
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

# batch_size=1 to fit on 8 H200. Seq length kept the same so tok/s is comparable.
TRAIN_COMMON_BS1="--train.max_steps=$STEPS --train.batch_size=1 --train.seq_len=4096 --train.grad_accum_steps=1 --train.compile_model=false --train.mixed_precision=bf16 --train.activation_checkpointing=full"

run_experiment "05b_8gpu_moe_e64_fsdp2_ep4_unpacked_bs1" \
    "$MOE_MODEL_BASE --model.num_experts=64 --model.moe_packed_experts=false $TRAIN $TRAIN_COMMON_BS1 --distributed.tp=1 --distributed.ep=4 --distributed.dp_shard=2"

run_experiment "06b_8gpu_moe_e64_fsdp2_ep4_packed_bs1" \
    "$MOE_MODEL_BASE --model.num_experts=64 --model.moe_packed_experts=true $TRAIN $TRAIN_COMMON_BS1 --distributed.tp=1 --distributed.ep=4 --distributed.dp_shard=2"

echo "done"
