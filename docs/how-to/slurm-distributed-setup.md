# SLURM distributed setup

KempnerForge ships three SLURM launch scripts under
[`scripts/slurm/`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/slurm):

| Script | Launcher | Scope |
|--------|----------|-------|
| `singlenode.sh` | `torchrun --standalone` | 1 node, N GPUs |
| `multinode.sh`  | `srun` direct | 2+ nodes, N×M GPUs |
| `interactive.sh`| `srun` inside an existing allocation | attach to `salloc` session |

This page explains when to use each, what environment the scripts
assemble, and how preemption + auto-resume compose with them.

## Single node: `singlenode.sh`

```bash
sbatch scripts/slurm/singlenode.sh configs/train/7b.toml
sbatch scripts/slurm/singlenode.sh configs/train/7b.toml --train.max_steps=1000
```

The script:

- Reads `SLURM_GPUS_PER_NODE` (default 4) to build `torchrun
  --nproc_per_node=$NGPUS`
- Detects the first UP InfiniBand interface (`ip -br addr`) and
  exports `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` to it
- Launches `torchrun --standalone` so there's no rendezvous
  coordination needed (single-host)

Edit the header for your cluster:

```bash
#SBATCH --partition=<partition-name>
#SBATCH --account=<account-name>
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
```

## Multi-node: `multinode.sh`

```bash
sbatch --nodes=4 scripts/slurm/multinode.sh configs/train/7b.toml
sbatch --nodes=8 scripts/slurm/multinode.sh \
    configs/train/7b.toml --train.max_steps=50000
```

Two things make multi-node different from single-node on this
codebase:

1. **Launch method is `srun` direct, not `torchrun`.** Each srun task
   is one process bound to one GPU. SLURM already sets
   `SLURM_PROCID` / `SLURM_LOCALID` / `SLURM_NTASKS`, and
   [`get_world_info`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/setup.py)
   maps them to `RANK` / `LOCAL_RANK` / `WORLD_SIZE`. No torchrun in
   the loop.
2. **`--ntasks-per-node` must equal `--gpus-per-node`.** This is the
   load-bearing invariant. Violate it and local rank maps incorrectly,
   processes land on the wrong GPU, and NCCL fails silently or crashes
   at first collective.

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4      # ← must match gpus-per-node
#SBATCH --gpus-per-node=4
#SBATCH --signal=B:SIGTERM@120   # SLURM sends SIGTERM 120s before time-limit
#SBATCH --requeue                # auto-resubmit on preemption
```

The script then:

- Extracts `MASTER_ADDR` via `scontrol show hostnames
  $SLURM_JOB_NODELIST | head -n 1`
- Picks a random free port in `[15000, 20000]` for `MASTER_PORT` (so
  two jobs on the same node don't collide)
- Detects the first UP IB interface and binds both NCCL and Gloo to
  it
- Launches `srun uv run python scripts/train.py $CONFIG`

### Why Gloo needs the same IB interface

Async DCP checkpointing uses a CPU-side Gloo process group for
coordination. Without `GLOO_SOCKET_IFNAME` pointed at the IB
interface, Gloo falls back to the management Ethernet (`em4` on
Kempner nodes), which is on a different subnet — peer connections
time out and the checkpoint never completes.

### NCCL environment the script sets

```bash
NCCL_SOCKET_IFNAME=ib0      # OOB bootstrap + socket transport
NCCL_IB_DISABLE=0           # enable IB verbs transport
NCCL_NET_GDR_LEVEL=2        # GPU-direct RDMA
NCCL_IB_GID_INDEX=3         # H100/H200 fabric config
NCCL_TIMEOUT=1800           # 30 minutes — raise for slow collectives
GLOO_SOCKET_IFNAME=ib0
```

`NCCL_IB_GID_INDEX=3` is specific to the RoCE/IB fabric on Kempner
H100/H200 nodes. On other clusters this may need to be 0 or 1 — check
with your admins or `ibv_devinfo`.

## Interactive / existing allocation: `interactive.sh`

When you already have an allocation (`salloc` or a queued job) and
want to attach training to it without a new `sbatch`:

```bash
# Pass the JOBID of the existing allocation, then the config:
bash scripts/slurm/interactive.sh 3565401 configs/train/debug.toml
bash scripts/slurm/interactive.sh 3565401 configs/train/7b.toml --train.max_steps=50
```

The script resolves `NODELIST` / `MASTER_ADDR` via `squeue -j $JOBID`
and `scontrol show hostnames`, auto-detects the IB interface, then
dispatches `srun --jobid=$JOBID` with `--ntasks-per-node=$GPUS_PER_NODE`.
Useful for debugging on a reserved node without going through the
queue again.

## Preemption, SIGTERM, auto-resume

The full mechanics live in
[Resilience § SLURM preemption](../resilience/slurm-preemption.md);
the short version:

- `#SBATCH --signal=B:SIGTERM@120` tells SLURM: "send SIGTERM 120
  seconds before the wallclock time limit, to rank 0 (`B:` = batch
  script process)."
- The training loop's
  [`ShutdownHandler`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/signal_handler.py)
  catches SIGTERM, flips `should_shutdown()` to `True`, and the step
  loop writes an emergency DCP checkpoint before exiting cleanly.
- `#SBATCH --requeue` puts the job back in the queue. When it starts
  again, `CheckpointManager` follows the `latest` symlink in
  `checkpoint.dir` and resumes at the exact step / sample.

The 120-second window is a design parameter: emergency checkpoints
must finish within it or the job is SIGKILL'd. Async DCP save times
depend on model size, FSDP degree, and filesystem — measure on your
cluster with `checkpoint.async_save = true` and a test run before
trusting the default for 70B+. If your save consistently overruns,
raise `#SBATCH --signal=B:SIGTERM@<seconds>` accordingly.

## Auto-resume in practice

With `--requeue` set, you can monitor a preempted-and-restarted job:

```bash
squeue -j <jobid>                  # see current state
sacct -j <jobid> --format=JobID,State,ExitCode,Start,End
scontrol show job <jobid> | grep RestartCount
```

On first launch, `SLURM_RESTART_COUNT` is `0`; after a requeue it
increments. The script echoes this so the log makes it obvious:

```
Restart cnt:  2
```

If you see rising restart counts but training never crosses step
`eval_interval`, something is preempting the job faster than it can
checkpoint — raise the `#SBATCH --signal` window or drop
`checkpoint.interval`.

## Checking the launch worked

First time you run on a new cluster, verify:

1. **All ranks start** — the script's banner line should print once
   per rank, and rank 0 log should show `world_size = N × M`.
2. **NCCL talks over IB** — set `NCCL_DEBUG=INFO` in the script once
   to see the negotiated transport. Each rank should log `NET/IB`
   (not `NET/Socket`).
3. **First all-reduce succeeds** — rank 0 reports model build and
   step 1 loss within a minute or two. A multi-minute hang at step 1
   usually means Gloo is on the wrong interface (DCP init fails) or
   IB GID is wrong.

If step 1 hangs, kill the job and re-run with:

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

## What the SLURM scripts don't do

- **Tokenizer caching** — you still need to pre-cache HuggingFace
  tokenizers on a login node (compute nodes are usually air-gapped).
  See
  [Prepare tokenized data § Cache the tokenizer first](prepare-tokenized-data.md#cache-the-tokenizer-first).
- **Data staging** — `configs/train/*.toml` expects your dataset at
  `data.dataset_path`. Pre-copy or symlink before `sbatch`.
- **Env mounts** — the scripts assume `uv run python` resolves the
  repo's `.venv`. If your cluster uses different mount points, edit
  the `uv run` lines.

## See also

- [Resilience § SLURM preemption](../resilience/slurm-preemption.md)
  — the full `ShutdownHandler` timeline and SLURM requeue dance.
- [Distributed § DeviceMesh](../distributed/device-mesh.md) — how
  `init_distributed` turns rank info into a mesh.
- [End-to-end training run](end-to-end-training-run.md) — runs
  through the single-node path before this page scales it.
- [Scaling guide](scaling-guide.md) — which parallelism combo to pick
  at each GPU count; this page tells you how to launch it.
- [`scripts/slurm/multinode.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/slurm/multinode.sh)
  — the reference script this page documents.
