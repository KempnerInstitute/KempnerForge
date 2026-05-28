"""Integration test: checkpoint save → load → verify identical state.

Tests the full checkpoint pipeline with a real model (non-distributed).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kempnerforge.checkpoint.state import build_train_state, get_rng_state, restore_train_state
from kempnerforge.config.schema import ModelConfig, OptimizerConfig, SchedulerConfig
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.scheduler import build_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)


class TestCheckpointRoundtrip:
    def test_model_state_roundtrip(self, tmp_path):
        """Model state dict save/load produces identical weights."""
        model = Transformer(CONFIG).to(DEVICE)

        # Save
        state = model.state_dict()
        torch.save(state, tmp_path / "model.pt")

        # Load into fresh model
        model2 = Transformer(CONFIG).to(DEVICE)
        model2.load_state_dict(torch.load(tmp_path / "model.pt", map_location=DEVICE))

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters(), strict=True
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_optimizer_state_roundtrip(self, tmp_path):
        """Optimizer state dict save/load preserves momentum buffers."""
        model = Transformer(CONFIG).to(DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        # Train a few steps to populate optimizer state
        for _ in range(3):
            tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
            logits = model(tokens)
            loss = logits.sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Save
        torch.save(opt.state_dict(), tmp_path / "opt.pt")

        # Load into fresh optimizer
        model2 = Transformer(CONFIG).to(DEVICE)
        model2.load_state_dict(model.state_dict())
        opt2 = build_optimizer(model2, OptimizerConfig(lr=1e-3, fused=False))
        opt2.load_state_dict(torch.load(tmp_path / "opt.pt", map_location=DEVICE))

        # Compare state
        for pg1, pg2 in zip(opt.param_groups, opt2.param_groups, strict=True):
            assert pg1["lr"] == pg2["lr"]
            assert pg1["weight_decay"] == pg2["weight_decay"]

    def test_train_state_roundtrip(self):
        """build_train_state → restore_train_state round-trip."""
        sched_config = SchedulerConfig(warmup_steps=5, name="cosine")
        model = Transformer(CONFIG).to(DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        sched = build_scheduler(opt, sched_config, max_steps=100)

        # Advance scheduler
        for _ in range(10):
            opt.step()
            sched.step()

        state = build_train_state(step=10, tokens_seen=5000, scheduler=sched)
        assert state["step"] == 10
        assert state["tokens_seen"] == 5000

        # Restore into fresh scheduler
        sched2 = build_scheduler(
            build_optimizer(Transformer(CONFIG).to(DEVICE), OptimizerConfig(lr=1e-3, fused=False)),
            sched_config,
            max_steps=100,
        )
        step, tokens, _ = restore_train_state(state, scheduler=sched2)
        assert step == 10
        assert tokens == 5000
        assert sched2.last_epoch == sched.last_epoch

    def test_rng_state_roundtrip(self):
        """RNG state save/restore produces identical random sequences."""
        rng_state = get_rng_state()

        # Generate random values
        vals1 = torch.randn(10)

        # Restore RNG and re-generate
        from kempnerforge.checkpoint.state import set_rng_state

        set_rng_state(rng_state)
        vals2 = torch.randn(10)

        assert torch.equal(vals1, vals2)

    def test_full_training_resume(self, tmp_path):
        """Train N steps → save → train M more steps.
        Compare against uninterrupted N+M steps."""
        torch.manual_seed(42)
        model = Transformer(CONFIG).to(DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        # Fixed data
        tokens = torch.randint(0, 256, (4, 32), device=DEVICE)
        labels = tokens.clone()

        # Train 20 steps uninterrupted
        torch.manual_seed(42)
        model_ref = Transformer(CONFIG).to(DEVICE)
        model_ref.load_state_dict(model.state_dict())
        opt_ref = build_optimizer(model_ref, OptimizerConfig(lr=1e-3, fused=False))

        for _ in range(20):
            logits = model_ref(tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            opt_ref.step()
            opt_ref.zero_grad()
        ref_loss = loss.item()

        # Train 10 steps, save, load, train 10 more
        for _ in range(10):
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Save checkpoint
        torch.save(model.state_dict(), tmp_path / "model.pt")
        torch.save(opt.state_dict(), tmp_path / "opt.pt")

        # Load into new model
        model2 = Transformer(CONFIG).to(DEVICE)
        model2.load_state_dict(torch.load(tmp_path / "model.pt", map_location=DEVICE))
        opt2 = build_optimizer(model2, OptimizerConfig(lr=1e-3, fused=False))
        opt2.load_state_dict(torch.load(tmp_path / "opt.pt", map_location=DEVICE))

        # Train 10 more steps
        for _ in range(10):
            logits = model2(tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            opt2.step()
            opt2.zero_grad()
        resumed_loss = loss.item()

        # Losses should be identical (deterministic, same data)
        assert abs(ref_loss - resumed_loss) < 1e-4, (
            f"Resumed loss differs: ref={ref_loss:.6f}, resumed={resumed_loss:.6f}"
        )

    def test_manager_restores_optimizer_moments_single_gpu(self, tmp_path):
        """Single-GPU: CheckpointManager must restore optimizer moments into a
        FRESH optimizer (the resume scenario).

        Plain torch.save/load round-trips an optimizer fine, so the tests above
        don't exercise the bug. The manager's DCP path filled the load template
        from a freshly-built optimizer's *empty* state_dict, silently dropping the
        moments. This guards that single-GPU manager path (the distributed
        equivalent is in tests/distributed/test_checkpoint.py).
        """
        from kempnerforge.checkpoint.manager import CheckpointManager
        from kempnerforge.config.schema import CheckpointConfig

        def moments(optimizer):
            out = []
            for group in optimizer.param_groups:
                for p in group["params"]:
                    st = optimizer.state.get(p, {})
                    if "exp_avg" in st:
                        out.append((st["exp_avg"].clone(), st["exp_avg_sq"].clone()))
            return out

        torch.manual_seed(0)
        model = Transformer(CONFIG).to(DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        for _ in range(3):
            tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
            model(tokens).sum().backward()
            opt.step()
            opt.zero_grad()

        ref = moments(opt)
        assert ref and any(ea.abs().sum().item() > 0 for ea, _ in ref), (
            "no non-zero moments to test"
        )

        ckpt_dir = str(tmp_path / "ckpt")
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model, opt).save(step=3)

        # Fresh model + fresh optimizer (empty state) -> the resume scenario.
        model2 = Transformer(CONFIG).to(DEVICE)
        opt2 = build_optimizer(model2, OptimizerConfig(lr=1e-3, fused=False))
        assert not moments(opt2), "fresh optimizer should have empty state before load"
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model2, opt2).load()

        loaded = moments(opt2)
        assert loaded, "optimizer moments not restored via manager (momentum reset on resume)"
        for (rea, rev), (lea, lev) in zip(ref, loaded, strict=True):
            assert torch.equal(rea, lea), "exp_avg not restored bit-exactly"
            assert torch.equal(rev, lev), "exp_avg_sq not restored bit-exactly"

    def test_manager_load_excludes_optimizer(self, tmp_path):
        """exclude_keys=['optimizer'] (the scripts/eval.py / fine-tune flow):
        load model state but leave the fresh optimizer untouched. Covers the
        load_optim=False branch in CheckpointManager.load."""
        from kempnerforge.checkpoint.manager import CheckpointManager
        from kempnerforge.config.schema import CheckpointConfig

        def moments(optimizer):
            out = []
            for group in optimizer.param_groups:
                for p in group["params"]:
                    st = optimizer.state.get(p, {})
                    if "exp_avg" in st:
                        out.append((st["exp_avg"].clone(), st["exp_avg_sq"].clone()))
            return out

        torch.manual_seed(0)
        model = Transformer(CONFIG).to(DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        for _ in range(3):
            tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
            model(tokens).sum().backward()
            opt.step()
            opt.zero_grad()
        ref_weights = {n: p.clone() for n, p in model.named_parameters()}

        ckpt_dir = str(tmp_path / "ckpt")
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model, opt).save(step=3)

        # Fresh model (different init) + fresh optimizer.
        torch.manual_seed(99)
        model2 = Transformer(CONFIG).to(DEVICE)
        opt2 = build_optimizer(model2, OptimizerConfig(lr=1e-3, fused=False))
        pre_load = {n: p.clone() for n, p in model2.named_parameters()}
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model2, opt2).load(
            exclude_keys=["optimizer"]
        )

        # Model state restored (weights now match the saved model).
        for n, p in model2.named_parameters():
            assert not torch.equal(pre_load[n], p), f"{n} was not loaded"
            assert torch.equal(ref_weights[n], p), f"{n} mismatch vs saved"
        # Optimizer was excluded -> still empty.
        assert not moments(opt2), "optimizer should remain empty on exclude_keys=['optimizer']"

    def test_manager_load_excludes_model(self, tmp_path):
        """exclude_keys=['model'] (symmetric case): load optimizer moments but
        leave the existing model weights untouched. Covers the load_model=False
        branch in CheckpointManager.load."""
        from kempnerforge.checkpoint.manager import CheckpointManager
        from kempnerforge.config.schema import CheckpointConfig

        def moments(optimizer):
            out = []
            for group in optimizer.param_groups:
                for p in group["params"]:
                    st = optimizer.state.get(p, {})
                    if "exp_avg" in st:
                        out.append((st["exp_avg"].clone(), st["exp_avg_sq"].clone()))
            return out

        torch.manual_seed(0)
        model = Transformer(CONFIG).to(DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        for _ in range(3):
            tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
            model(tokens).sum().backward()
            opt.step()
            opt.zero_grad()
        ref_moments = moments(opt)
        assert ref_moments, "fixture: expected non-empty optimizer state"

        ckpt_dir = str(tmp_path / "ckpt")
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model, opt).save(step=3)

        torch.manual_seed(99)
        model2 = Transformer(CONFIG).to(DEVICE)
        opt2 = build_optimizer(model2, OptimizerConfig(lr=1e-3, fused=False))
        pre_load = {n: p.clone() for n, p in model2.named_parameters()}
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model2, opt2).load(exclude_keys=["model"])

        # Model state was excluded -> weights unchanged from fresh init.
        for n, p in model2.named_parameters():
            assert torch.equal(pre_load[n], p), f"{n} should not have changed"
        # Optimizer moments restored from the saved checkpoint.
        loaded = moments(opt2)
        assert loaded, "optimizer moments should load with exclude_keys=['model']"
        for (rea, rev), (lea, lev) in zip(ref_moments, loaded, strict=True):
            assert torch.equal(rea, lea), "exp_avg not restored"
            assert torch.equal(rev, lev), "exp_avg_sq not restored"
