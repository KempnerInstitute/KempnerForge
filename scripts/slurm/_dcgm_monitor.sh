#!/bin/bash

set -eo pipefail

LOG_DIR="${KEMPNERFORGE_LOG_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/logs/distributed}"
mkdir -p "$LOG_DIR"
OUT="$LOG_DIR/dcgm_job${SLURM_JOB_ID}_$(hostname -s).log"

echo "[$(hostname -s)] dcgmi dmon -> $OUT"

exec dcgmi dmon \
    -e 203,1002,1003,1004,1005,252,155,150 \
    -d 1000 \
    > "$OUT" 2>&1
