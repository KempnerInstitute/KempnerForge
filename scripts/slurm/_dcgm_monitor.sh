#!/bin/bash
# Per-node GPU utilization monitor using dcgmi dmon.
# Run one instance per node (srun --ntasks-per-node=1) alongside training.
#
# Fields (DCGM field IDs, see /usr/include/dcgm_fields.h or `dcgmi dmon -l`):
#   203  = DCGM_FI_DEV_GPU_UTIL              (GPUTL, %)  — GPU active (coarse)
#   1002 = DCGM_FI_PROF_SM_ACTIVE            (SMACT, 0..1) — fraction of SMs active
#   1003 = DCGM_FI_PROF_SM_OCCUPANCY         (SMOCC, 0..1) — warp occupancy per SM
#   1004 = DCGM_FI_PROF_PIPE_TENSOR_ACTIVE   (TENSO, 0..1) — tensor pipe active (MFU signal)
#   1005 = DCGM_FI_PROF_DRAM_ACTIVE          (DRAMA, 0..1) — HBM bandwidth use
#   252  = DCGM_FI_DEV_FB_USED               (FBUSD, MiB)
#   155  = DCGM_FI_DEV_POWER_USAGE           (POWER, W)
#   150  = DCGM_FI_DEV_GPU_TEMP              (TMPTR, C)
#
# Output: ${KEMPNERFORGE_LOG_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/logs/distributed}/dcgm_job${JOB}_${HOSTNAME}.log

set -eo pipefail

LOG_DIR="${KEMPNERFORGE_LOG_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/logs/distributed}"
mkdir -p "$LOG_DIR"
OUT="$LOG_DIR/dcgm_job${SLURM_JOB_ID}_$(hostname -s).log"

echo "[$(hostname -s)] dcgmi dmon -> $OUT"

# -d 1000ms sampling; runs until killed by the caller (training finishes)
exec dcgmi dmon \
    -e 203,1002,1003,1004,1005,252,155,150 \
    -d 1000 \
    > "$OUT" 2>&1
