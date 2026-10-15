"""Distributed tests for checkpointing.

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_checkpoint.py -v
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from kempnerforge.checkpoint import manager as mgr_mod
from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.schema import AsyncCheckpointMode, CheckpointConfig, ModelConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import build_optimizer

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

SMALL_CONFIG = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=512, max_seq_len=64)

# Shared filesystem temp directory (visible to all nodes)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TEST_TMP = _PROJECT_ROOT / ".test_tmp"


@pytest.fixture
def shared_tmp_dir():
    """Create a temp directory on the shared filesystem, cleaned up after use."""
    rank = dist.get_rank()
    # Rank 0 creates, then broadcasts path
    if rank == 0:
        _TEST_TMP.mkdir(exist_ok=True)
        import tempfile

        d = tempfile.mkdtemp(dir=_TEST_TMP)
    else:
        d = ""
    obj_list = [d]
    dist.broadcast_object_list(obj_list, src=0)
    d = obj_list[0]
    yield d
    dist.barrier()
    if rank == 0:
        shutil.rmtree(d, ignore_errors=True)


class TestCheckpointRoundTrip:
    def test_save_load_fsdp(self, distributed_env, shared_tmp_dir):
        """Save and load under FSDP — model output should be identical."""
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir

        # Build model + FSDP
        torch.manual_seed(42)
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        from kempnerforge.config.schema import OptimizerConfig

        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        # Do a forward + backward to get non-trivial optimizer state
        tokens = torch.randint(0, 512, (2, 32), device="cuda")
        out = model(tokens)
        out.sum().backward()
        opt.step()
        opt.zero_grad()

        # Capture reference output
        with torch.no_grad():
            ref_out = model(tokens).clone()

        # Save
        config = CheckpointConfig(dir=ckpt_dir, keep_last_n=2)
        mgr = CheckpointManager(config, model, opt)
        mgr.save(step=1, tokens_seen=64)

        # Perturb model to verify load actually restores
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Verify output changed
        with torch.no_grad():
            perturbed_out = model(tokens)
        assert not torch.allclose(ref_out, perturbed_out, atol=1e-3)

        # Load
        mgr.load()

        # Verify output matches reference
        with torch.no_grad():
            restored_out = model(tokens)
        assert torch.allclose(ref_out, restored_out, atol=1e-5), (
            f"Restored output differs: max diff={(ref_out - restored_out).abs().max().item()}"
        )

    def test_resume_restores_optimizer_moments(self, distributed_env, shared_tmp_dir):
        """Optimizer moments must be restored into a FRESH optimizer on resume.

        Regression for the bug where the manager used raw ``optimizer.state_dict()``
        with DCP: a freshly-constructed optimizer has empty state, so the load
        template had no moment tensors and ``dcp.load`` silently dropped them ->
        AdamW ``exp_avg``/``exp_avg_sq`` reset to zero on every resume. Loading
        into the *same* already-stepped optimizer (as test_save_load_fsdp does)
        hides this, so here we load into a fresh optimizer like a real resume.
        """
        from kempnerforge.config.schema import OptimizerConfig

        mesh = distributed_env
        ckpt_dir = shared_tmp_dir

        def snapshot_moments(optimizer):
            """Per-rank (sharded) (exp_avg, exp_avg_sq) clones, in param order."""
            out = []
            for group in optimizer.param_groups:
                for p in group["params"]:
                    st = optimizer.state.get(p, {})
                    if "exp_avg" in st:
                        out.append(
                            (st["exp_avg"].detach().clone(), st["exp_avg_sq"].detach().clone())
                        )
            return out

        # Build, step a few times so the moments are non-trivial, then save.
        torch.manual_seed(42)
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        for _ in range(3):
            tokens = torch.randint(0, 512, (2, 32), device="cuda")
            model(tokens).sum().backward()
            opt.step()
            opt.zero_grad()

        ref_moments = snapshot_moments(opt)
        assert ref_moments, "optimizer had no moment state to test"
        assert any(ea.abs().sum().item() > 0 for ea, _ in ref_moments), (
            "reference moments are all zero"
        )

        CheckpointManager(CheckpointConfig(dir=ckpt_dir, keep_last_n=2), model, opt).save(
            step=3, tokens_seen=192
        )

        # FRESH model + FRESH optimizer (never stepped -> empty optimizer state),
        # exactly the resume scenario. Then load.
        torch.manual_seed(42)
        model2 = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model2, mesh)
        opt2 = build_optimizer(model2, OptimizerConfig(lr=1e-3, fused=False))
        assert not snapshot_moments(opt2), "fresh optimizer should have empty state before load"

        CheckpointManager(CheckpointConfig(dir=ckpt_dir, keep_last_n=2), model2, opt2).load()

        loaded_moments = snapshot_moments(opt2)
        assert loaded_moments, (
            "optimizer moments were not restored (state still empty after load) "
            "-- AdamW momentum reset on resume"
        )
        assert len(loaded_moments) == len(ref_moments)
        for i, ((rea, rev), (lea, lev)) in enumerate(zip(ref_moments, loaded_moments, strict=True)):
            assert torch.equal(rea, lea), f"param {i}: exp_avg not restored bit-exactly"
            assert torch.equal(rev, lev), f"param {i}: exp_avg_sq not restored bit-exactly"

    def test_resume_with_a_never_stepped_parameter(self, distributed_env, shared_tmp_dir):
        """A sharded parameter that never received a gradient has no saved optimizer
        state; resume restores every other parameter's state and leaves it stateless."""
        from torch.distributed.tensor import DTensor

        from kempnerforge.config.schema import OptimizerConfig

        def build(seed):
            torch.manual_seed(seed)
            model = Transformer(SMALL_CONFIG).cuda()
            model.unused = torch.nn.Linear(128, 128).cuda()  # never in the forward
            apply_fsdp2(model, distributed_env)
            return model, build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        def local(t):
            return t.to_local() if isinstance(t, DTensor) else t

        model, opt = build(42)
        for _ in range(2):
            tokens = torch.randint(0, 512, (2, 32), device="cuda")
            model(tokens).sum().backward()
            opt.step()
            opt.zero_grad()
        config = CheckpointConfig(dir=shared_tmp_dir, keep_last_n=2)
        CheckpointManager(config, model, opt).save(step=2)

        model2, opt2 = build(7)
        assert CheckpointManager(config, model2, opt2).load()[0] == 2
        for (name, p), p2 in zip(model.named_parameters(), model2.parameters(), strict=True):
            assert torch.equal(local(p), local(p2)), name
            state, state2 = opt.state.get(p, {}), opt2.state.get(p2, {})
            assert state.keys() == state2.keys(), name
            assert bool(state2) is not name.startswith("unused."), name
            for key in state:
                assert torch.equal(local(state[key]), local(state2[key])), f"{name}.{key}"

    def test_latest_symlink(self, distributed_env, shared_tmp_dir):
        """The 'latest' symlink should point to the most recent checkpoint."""
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir

        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        from kempnerforge.config.schema import OptimizerConfig

        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        config = CheckpointConfig(dir=ckpt_dir, keep_last_n=3)
        mgr = CheckpointManager(config, model, opt)

        mgr.save(step=10)
        mgr.save(step=20)

        if dist.get_rank() == 0:
            latest = Path(ckpt_dir) / "latest"
            assert latest.exists()
            assert latest.resolve().name == "step_20"


class TestCheckpointSaveBarrier:
    """save() must synchronize all ranks on the rank-0 metadata writes."""

    def test_save_waits_for_rank0_writes(self, distributed_env, shared_tmp_dir):
        """Non-rank-0 must not return from save() before rank 0 finishes.

        Without the end-of-save barrier, non-rank-0 returns immediately
        after the async DCP dispatch, while rank 0 is still writing
        train_state.pt and the latest symlink. We force the race to
        be measurable by slowing rank-0's torch.save.
        """
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir
        rank = dist.get_rank()

        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        from kempnerforge.config.schema import OptimizerConfig

        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        cfg = CheckpointConfig(dir=ckpt_dir, keep_last_n=2)
        mgr = CheckpointManager(cfg, model, opt)

        real_torch_save = torch.save
        sleep_sec = 0.5

        def slow_on_rank0(*args, **kwargs):
            if rank == 0:
                time.sleep(sleep_sec)
            return real_torch_save(*args, **kwargs)

        # Barrier before timing so all ranks start save() at roughly the
        # same instant; isolates the signal we care about.
        dist.barrier()
        t0 = time.perf_counter()
        with patch.object(mgr_mod.torch, "save", side_effect=slow_on_rank0):
            mgr.save(step=1, tokens_seen=100)
        elapsed = time.perf_counter() - t0

        # Every rank must have waited for rank-0's slow write.
        assert elapsed >= 0.4 * sleep_sec, (
            f"rank {rank}: save() returned after {elapsed:.3f}s — the "
            f"end-of-save barrier is missing. Expected >= {0.4 * sleep_sec:.3f}s."
        )

        # And every rank observes rank-0's writes afterwards.
        step_dir = Path(ckpt_dir) / "step_1"
        assert (step_dir / "train_state.pt").exists(), (
            f"rank {rank}: train_state.pt not visible after save()"
        )
        assert (step_dir / "metadata.json").exists(), (
            f"rank {rank}: metadata.json not visible after save()"
        )


class TestCheckpointLoadDivergentExistence:
    """load() must not hang if ranks disagree about train_state.pt existence.

    Simulates attribute-cache skew (NFS/Lustre) by patching Path.exists
    so non-rank-0 sees a missing file. Rank-0's answer must be authoritative
    via broadcast; otherwise only some ranks enter the torch.load branch
    and the subsequent broadcast_object_list deadlocks.
    """

    def test_load_does_not_hang_on_divergent_exists(self, distributed_env, shared_tmp_dir):
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir
        rank = dist.get_rank()

        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        from kempnerforge.config.schema import OptimizerConfig

        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        cfg = CheckpointConfig(dir=ckpt_dir, keep_last_n=2)
        mgr = CheckpointManager(cfg, model, opt)

        # Save so there's something to load.
        mgr.save(step=1, tokens_seen=100)
        dist.barrier()

        # Patch exists() so non-rank-0 sees the file as missing. Without
        # the authoritative broadcast, non-rank-0 skips torch.load but
        # rank 0 enters and calls broadcast_object_list — deadlock.
        real_exists = Path.exists

        def skewed_exists(self):
            if rank != 0 and self.name == "train_state.pt":
                return False
            return real_exists(self)

        # Wrap in a timeout via a CUDA event + sleep loop would be ideal;
        # simpler: rely on the PG default timeout to surface a deadlock as
        # a RuntimeError rather than blocking the test runner forever.
        # With the fix, load() completes promptly; without, the test hangs
        # until the PG timeout fires.
        with patch.object(Path, "exists", skewed_exists):
            step, tokens_seen, _ = mgr.load()

        assert step == 1, f"rank {rank}: expected step=1 after load, got {step}"
        assert tokens_seen == 100, (
            f"rank {rank}: expected tokens_seen=100 after load, got {tokens_seen}"
        )


class TestAsyncCheckpointLatestSafety:
    """Real 2-rank FSDP coverage for the async `latest`-before-flush bug.

    With ``async_with_pinned_mem`` the DCP shards + ``.metadata`` are
    written by a background thread. ``latest`` must never resolve to a
    checkpoint whose ``.metadata`` is not yet durable, and resume must
    recover from the newest durable step.
    """

    def test_latest_only_resolves_to_durable_checkpoints(self, distributed_env, shared_tmp_dir):
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir
        rank = dist.get_rank()

        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        from kempnerforge.config.schema import OptimizerConfig

        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        cfg = CheckpointConfig(
            dir=ckpt_dir,
            keep_last_n=3,
            async_mode=AsyncCheckpointMode.async_pinned,
        )
        mgr = CheckpointManager(cfg, model, opt)

        latest = Path(ckpt_dir) / "latest"

        # Mirror the real training loop exactly: periodic save() calls, then
        # a single wait() after the loop (training/loop.py::run_training_loop).
        # NO barriers are injected between/after save() here — a mid-flush collective on
        # the default PG races DCP's own async-executor collective thread,
        # an unrelated pre-existing hazard that has nothing to do with the
        # `latest` symlink fix under test.
        for s in (10, 20, 30, 40):
            mgr.save(step=s)
            # Rank-0 local check only (no collective): whenever `latest`
            # resolves, it MUST be a durable checkpoint. Pre-fix it pointed
            # at the in-flight step whose DCP `.metadata` did not exist yet.
            if rank == 0 and latest.exists():
                target = latest.resolve()
                assert (target / ".metadata").exists(), (
                    f"after save({s}): latest -> {target} has no DCP .metadata"
                )

        # Final drain (the training loop calls wait() after the loop); this
        # commits the last deferred checkpoint. wait() ends with its own
        # barrier, so all ranks resync here.
        mgr.wait()
        if rank == 0:
            assert latest.exists(), "latest missing after wait()"
            target = latest.resolve()
            assert target.name == "step_40", f"latest -> {target.name}, want step_40"
            assert (target / ".metadata").exists(), "final latest not durable"

        # No async flush is in flight after wait(), so this barrier cannot
        # race a DCP collective; it just re-syncs before the collective
        # load() (rank 0 did extra local fs asserts above).
        dist.barrier()
        step, tokens_seen, _ = mgr.load()
        assert step == 40, f"rank {rank}: resumed at step {step}, want 40"

    def test_resume_falls_back_when_latest_incomplete(self, distributed_env, shared_tmp_dir):
        """Defense in depth: if `latest` points at an interrupted async
        flush (no DCP `.metadata`), resume falls back to the newest
        complete checkpoint instead of hard-failing in dcp.load."""
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir
        rank = dist.get_rank()

        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        from kempnerforge.config.schema import OptimizerConfig

        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        cfg = CheckpointConfig(dir=ckpt_dir, keep_last_n=5)
        mgr = CheckpointManager(cfg, model, opt)

        # A real, complete checkpoint at step 1 (sync save -> durable).
        mgr.save(step=1, tokens_seen=64)
        dist.barrier()

        # Simulate an interrupted async flush at step 2: rank 0 fabricates
        # a step_2 dir with train_state/metadata but NO DCP `.metadata`,
        # and repoints `latest` at it (exactly the crash-mid-flush state).
        if rank == 0:
            step2 = Path(ckpt_dir) / "step_2"
            step2.mkdir(parents=True, exist_ok=True)
            torch.save({"step": 2, "tokens_seen": 128, "rng": {}}, step2 / "train_state.pt")
            (step2 / "metadata.json").write_text('{"step": 2, "tokens_seen": 128}')
            link = Path(ckpt_dir) / "latest"
            tmp = link.with_suffix(".tmp")
            tmp.unlink(missing_ok=True)
            tmp.symlink_to("step_2")
            tmp.rename(link)
        dist.barrier()

        # Resume must fall back to the durable step_1, not crash on step_2.
        step, tokens_seen, _ = mgr.load()
        assert step == 1, f"rank {rank}: expected fallback to step 1, got {step}"
        assert tokens_seen == 64


class TestAsyncSaveBarrierNoDeadlock:
    """save()'s end barrier must not deadlock against DCP's async-save
    background collective.

    ``dcp.async_save()`` runs its collective planning (all_gather /
    reduce_scatter) on a background thread using the default process
    group. Pre-fix, save()'s end barrier ran on that same group from the
    main thread; two collectives interleaving on one communicator with
    nondeterministic per-rank ordering deadlock under tightly spaced
    async saves (no training collectives in between to separate them).
    The fix moves the barrier onto a dedicated gloo group.
    """

    def test_back_to_back_async_saves_do_not_deadlock(self, distributed_env, shared_tmp_dir):
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir
        from kempnerforge.config.schema import OptimizerConfig

        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        cfg = CheckpointConfig(
            dir=ckpt_dir,
            keep_last_n=3,
            async_mode=AsyncCheckpointMode.async_pinned,
        )
        mgr = CheckpointManager(cfg, model, opt)

        # Tight loop, no training collectives between saves: the exact
        # pattern that deadlocks pre-fix. Reaching the asserts at all means
        # the gloo barrier did not race the in-flight DCP collective.
        for s in (10, 20, 30, 40):
            mgr.save(step=s)
        mgr.wait()

        if dist.get_rank() == 0:
            latest = Path(ckpt_dir) / "latest"
            assert latest.exists()
            assert latest.resolve().name == "step_40"
        dist.barrier()


class _CursorLoader:
    """Minimal stateful loader with a rank-local cursor."""

    def __init__(self, cursor: int) -> None:
        self.cursor = cursor
        self.restored: dict | None = None

    def state_dict(self) -> dict:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        self.restored = dict(state)
        self.cursor = state["cursor"]


class TestPerRankState:
    """Each rank must restore ITS OWN dataloader cursor and RNG streams.

    Regression for the rank-0 broadcast bug: ``train_state.pt`` held only
    rank 0's ``rng``/``dataloader`` and ``load()`` broadcast that single copy
    to every rank, so divergent cursors and random streams collapsed onto
    rank 0's on every resume.
    """

    @staticmethod
    def _seed_all(seed: int) -> None:
        import random as pyrandom

        import numpy as np

        pyrandom.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _draw() -> tuple[float, float, float, float]:
        import random as pyrandom

        import numpy as np

        return (
            pyrandom.random(),
            float(np.random.random()),
            float(torch.rand(())),
            float(torch.rand((), device="cuda")),
        )

    def test_each_rank_restores_its_own_loader_and_rng(self, distributed_env, shared_tmp_dir):
        from kempnerforge.config.schema import OptimizerConfig

        rank = dist.get_rank()
        world = dist.get_world_size()

        torch.manual_seed(42)
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, distributed_env)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        tokens = torch.randint(0, 512, (2, 32), device="cuda")
        model(tokens).sum().backward()
        opt.step()
        opt.zero_grad()

        mgr = CheckpointManager(
            CheckpointConfig(dir=shared_tmp_dir, keep_last_n=3), model, opt, dp_info=(rank, world)
        )

        # Distinct per-rank streams and cursors, as a divergent run has them.
        self._seed_all(1000 + rank)
        loader = _CursorLoader(cursor=100 + rank)
        mgr.save(step=5, tokens_seen=320, dataloader=loader)

        # The next values each generator produces without interruption.
        expected = self._draw()

        saved = torch.load(Path(shared_tmp_dir) / "step_5" / "train_state.pt", weights_only=False)
        assert saved["per_rank"]["data_parallel"] == {r: (r, world) for r in range(world)}

        # Scramble everything identically on every rank; only a genuinely
        # per-rank restore can bring the distinct streams back.
        self._seed_all(7)
        loader2 = _CursorLoader(cursor=-1)
        step, tokens_seen, _ = mgr.load(
            path=str(Path(shared_tmp_dir) / "step_5"), dataloader=loader2, exact_resume=True
        )
        assert step == 5 and tokens_seen == 320
        assert loader2.restored == {"cursor": 100 + rank}, (
            f"rank {rank} restored {loader2.restored}, wanted its own cursor {100 + rank}"
        )
        got = self._draw()
        assert got == expected, f"rank {rank} RNG streams are not its own"

        # Every rank restored a DIFFERENT cursor (the actual regression).
        cursors: list = [None] * world
        dist.all_gather_object(cursors, loader2.cursor)
        assert sorted(cursors) == [100 + r for r in range(world)]
        draws: list = [None] * world
        dist.all_gather_object(draws, got)
        assert len(set(draws)) == world, "rank RNG streams collapsed onto one copy"

        # Second consecutive resume: the resumed run saves a fresh envelope
        # and restores it again.
        loader2.cursor = 200 + rank
        mgr.save(step=6, tokens_seen=384, dataloader=loader2)
        loader3 = _CursorLoader(cursor=-1)
        step, _, _ = mgr.load(
            path=str(Path(shared_tmp_dir) / "step_6"), dataloader=loader3, exact_resume=True
        )
        assert step == 6
        assert loader3.restored == {"cursor": 200 + rank}
        dist.barrier()

    def test_layout_change_refused_or_discarded_on_every_rank(
        self, distributed_env, shared_tmp_dir
    ):
        """Same world size, another data-parallel layout: an exact resume refuses
        on all ranks, or with the override discards on all ranks, never a
        per-rank mix; a load that is not a resume restores rank 0's single
        copy on every rank, as before."""
        from kempnerforge.config.schema import OptimizerConfig

        rank = dist.get_rank()
        world = dist.get_world_size()
        if world % 2:
            pytest.skip("needs an even world size")
        half = world // 2

        torch.manual_seed(42)
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, distributed_env)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        model(torch.randint(0, 512, (2, 32), device="cuda")).sum().backward()
        opt.step()
        opt.zero_grad()

        # Saved as dp x 2 (rank -> rank // 2), resumed as 2 x dp (rank -> rank % half).
        # With 4 ranks, ranks 0 and 3 keep their own coordinates.
        saved_dp = (rank // 2, half)
        current_dp = (rank % half, half)
        path = str(Path(shared_tmp_dir) / "step_5")
        cfg = CheckpointConfig(dir=shared_tmp_dir, keep_last_n=3)
        self._seed_all(1000 + rank)
        CheckpointManager(cfg, model, opt, dp_info=saved_dp).save(
            step=5, tokens_seen=320, dataloader=_CursorLoader(cursor=100 + rank)
        )
        saved_draws: list = [None] * world
        dist.all_gather_object(saved_draws, self._draw())

        loader = _CursorLoader(cursor=-1)
        refusing = CheckpointManager(cfg, model, opt, dp_info=current_dp)
        with pytest.raises(ValueError, match="data-parallel layout"):
            refusing.load(path=path, dataloader=loader, exact_resume=True)
        assert loader.restored is None

        lenient = CheckpointConfig(
            dir=shared_tmp_dir, keep_last_n=3, ignore_rank_state_mismatch=True
        )
        mgr = CheckpointManager(lenient, model, opt, dp_info=current_dp)
        self._seed_all(7)
        fresh = self._draw()
        self._seed_all(7)
        step, tokens_seen, _ = mgr.load(path=path, dataloader=loader, exact_resume=True)
        assert (step, tokens_seen) == (5, 320)
        assert loader.restored is None, f"rank {rank} restored a cursor across a layout change"
        assert self._draw() == fresh, f"rank {rank} restored RNG across a layout change"

        # Evaluation or a load_path warm start: no resume context.
        plain_loader = _CursorLoader(cursor=-1)
        plain = CheckpointManager(cfg, model, opt, dp_info=current_dp)
        self._seed_all(7)
        step, _, _ = plain.load(path=path, dataloader=plain_loader, exclude_keys=["optimizer"])
        assert step == 5
        assert plain_loader.restored == {"cursor": 100}, f"rank {rank} did not get rank 0's cursor"
        assert self._draw() == saved_draws[0], f"rank {rank} did not restore rank 0's RNG"
        dist.barrier()
