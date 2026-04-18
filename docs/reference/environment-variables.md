# Environment variables

Every environment variable KempnerForge reads, grouped by source.
Authoritative locations in the tree:
[`kempnerforge/distributed/setup.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/setup.py),
[`kempnerforge/resilience/elastic.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/elastic.py),
[`kempnerforge/metrics/logger.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/logger.py),
and the helper launch scripts under
[`scripts/slurm/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/scripts/slurm).

## Rank / world-size (torchrun or SLURM)

Read by `get_world_info()` in
[`kempnerforge/distributed/setup.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/setup.py).
Each torchrun-style var falls back to the SLURM equivalent, so the
same entry point works under both launchers.

| Variable | Fallback | Purpose |
|----------|----------|---------|
| `RANK` | `SLURM_PROCID` | Global rank (0..world_size-1) |
| `LOCAL_RANK` | `SLURM_LOCALID` | Rank within a node (0..gpus_per_node-1) |
| `WORLD_SIZE` | `SLURM_NTASKS` | Total number of ranks |

`get_world_info()` reads whichever is set and then calls
`os.environ.setdefault(...)` on all three, so downstream code
(PyTorch, WandB, logging) sees a torchrun-shaped environment even on
srun-direct launches.

## Rendezvous (MASTER_ADDR / MASTER_PORT)

| Variable | Set by | Notes |
|----------|--------|-------|
| `MASTER_ADDR` | user / launch script / `init_distributed()` | If unset, `init_distributed()` runs `scontrol show hostnames $SLURM_JOB_NODELIST | head -n1` |
| `MASTER_PORT` | user / launch script / `init_distributed()` | If unset, derived from `SLURM_JOB_ID` via a seeded RNG; otherwise picked by binding an ephemeral socket |

The SLURM launch helpers
([`scripts/slurm/multinode.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/slurm/multinode.sh),
[`_run_training.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/slurm/_run_training.sh))
export both before `srun` so every rank agrees. `init_distributed()`
only fills in missing values — it never overwrites.

## NCCL / Gloo (auto-set by `_set_nccl_env()`)

`init_distributed()` calls `_set_nccl_env()`, which detects the IB
interface (`ls /sys/class/net | grep '^ib'`) and populates defaults:

| Variable | Default | Purpose |
|----------|---------|---------|
| `NCCL_SOCKET_IFNAME` | detected IB interface (e.g. `ib0`) | NCCL bootstrap transport |
| `GLOO_SOCKET_IFNAME` | same as above | Gloo (used by DCP async checkpoint coordination) |
| `NCCL_IB_DISABLE` | `0` | Keep InfiniBand RDMA enabled |
| `NCCL_NET_GDR_LEVEL` | `2` | GPUDirect RDMA level — enables GPU→NIC DMA |

All four use `setdefault`, so anything you export before launch wins.
The launch scripts also set `NCCL_IB_GID_INDEX=3` and
`NCCL_TIMEOUT=1800`; those are shell-level conventions, not read
anywhere in the Python code.

## SLURM metadata (resilience)

Read by `get_slurm_context()` and `running_under_slurm()` in
[`kempnerforge/resilience/elastic.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/elastic.py).
Purely informational — used to populate logs and decide whether a job
is a restart.

| Variable | Purpose |
|----------|---------|
| `SLURM_JOB_ID` | Job identifier; also used to derive `MASTER_PORT` |
| `SLURM_JOB_NAME` | Job name (logged) |
| `SLURM_JOB_NODELIST` | Nodelist; parsed by `scontrol show hostnames` to find `MASTER_ADDR` |
| `SLURM_NNODES` | Node count (logged) |
| `SLURM_NTASKS_PER_NODE` | Tasks per node (logged) |
| `SLURM_RESTART_COUNT` | Non-zero means this is a requeue (used to detect restarts) |
| `SLURM_JOB_PARTITION` | Partition name (logged) |
| `SLURM_ARRAY_TASK_ID` | Array index when applicable (logged) |

## Logging

| Variable | Read in | Purpose |
|----------|---------|---------|
| `NO_COLOR` | [`kempnerforge/metrics/logger.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/logger.py) | Disables ANSI color codes in logs when set to any truthy value |
| `RANK` | same | Used to prefix log lines with the current rank |

## User-facing launch-script variables

Read only by the SLURM helpers under
[`scripts/slurm/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/scripts/slurm),
not by any Python module:

| Variable | Default | Purpose |
|----------|---------|---------|
| `KEMPNERFORGE_LOG_DIR` | `${SLURM_SUBMIT_DIR:-$PWD}/logs/distributed` | Per-rank log destination for `_run_training.sh`, `_run_distributed_tests.sh`, `_dcgm_monitor.sh` |
| `IB_IFNAME` | `ib0` (fallback in `interactive.sh`) | IB interface for the NCCL/Gloo exports in the launch scripts |
| `PYTORCH_CUDA_ALLOC_CONF` | (unset) | Passed through in `scripts/slurm/interactive.sh` and the MoE-EP benchmark scripts as `expandable_segments:True` |

## Who sets what

| Source | Variables it populates |
|--------|------------------------|
| User / job script | `KEMPNERFORGE_LOG_DIR`, `IB_IFNAME`, `PYTORCH_CUDA_ALLOC_CONF`, optionally `NCCL_*` overrides |
| `torchrun` | `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` |
| `srun` (SLURM) | `SLURM_*` (incl. `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`, `SLURM_JOB_*`) |
| `scripts/slurm/_run_training.sh` | Exports `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `NCCL_*`, `GLOO_*` before calling the trainer |
| `init_distributed()` | Fills missing `RANK`/`LOCAL_RANK`/`WORLD_SIZE` from `SLURM_*`, missing `MASTER_ADDR`/`MASTER_PORT`, and all four `NCCL_*`/`GLOO_*` defaults |

## See also

- [SLURM launch scripts](https://github.com/KempnerInstitute/KempnerForge/tree/main/scripts/slurm)
  — how `multinode.sh`, `_run_training.sh`, and friends assemble these
  variables end-to-end.
- [Architecture § One-slide overview](../architecture/index.md) —
  where `RANK`/`LOCAL_RANK`/`WORLD_SIZE` feed into the mesh
  construction.
- [Benchmarks § MoE Expert Parallelism](benchmarks.md#moe-expert-parallelism)
  — where `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is
  load-bearing.
