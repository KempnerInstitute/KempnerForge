#!/bin/bash
#SBATCH --job-name=evaluate-checkpoint
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=<memory>
#SBATCH --time=<time limit>
#SBATCH --output=%x-%j.out

set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
TASKS=$3
OUTPUT=$4
shift 4

cd <repository root>
source <environment activation>

python examples/image_understanding/evaluate_checkpoint.py "$CONFIG" \
    --checkpoint "$CHECKPOINT" --tasks "$TASKS" --output "$OUTPUT" "$@"
