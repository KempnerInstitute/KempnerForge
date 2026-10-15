#!/bin/bash
# FlexAttention vs dense-mask SDPA per parallelism strategy, 4 GPUs on one node.
#
#   sbatch benchmarks/flex_packing_parallel/run_bench.sbatch                        # all modes
#   sbatch --export=ALL,BENCH_MODE=pp benchmarks/flex_packing_parallel/run_bench.sbatch
#
# One mode per job is the better shape on a preemptible partition: a preemption
# then costs a single mode rather than the whole sweep, and shorter jobs clear a
# busy queue sooner. Each mode is a separate torchrun regardless, since the
# meshes differ and cannot share a process group.
#SBATCH --job-name=flexpar
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

set -u
REPO="${BENCH_REPO:-$HOME/KempnerForge}"
BENCH="$REPO/benchmarks/flex_packing_parallel"

# Not $SCRATCH: the cluster already exports that as /n/netscratch, which is not
# writable. Override with BENCH_CACHE_DIR if your scratch lives elsewhere.
CACHE_DIR="${BENCH_CACHE_DIR:-/n/netscratch/kempner_dev/Lab/$USER}"

# Keep compile caches off NFS; they are large and rewritten constantly.
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor"
export BENCH_OUT_DIR="$BENCH/results"

mkdir -p "$BENCH_OUT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO" || exit 1

for mode in ${BENCH_MODE:-fsdp tp tp_fsdp pp}; do
  rm -f "$BENCH_OUT_DIR/flex_parallel_${mode}.txt"
  MODE=$mode timeout 3000 uv run torchrun --nproc_per_node=4 \
    "$BENCH/bench_flex_parallel.py" >"$BENCH_OUT_DIR/${mode}_raw.log" 2>&1
  echo "=== $mode (exit $?) ==="
  cat "$BENCH_OUT_DIR/flex_parallel_${mode}.txt" 2>/dev/null
  echo
done
