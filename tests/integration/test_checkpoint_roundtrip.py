"""Integration test: training state round-trips in one process, with a real model.

The model and optimizer state dicts go through plain ``torch.save`` and
``torch.load``, and the train-state and RNG helpers of
``kempnerforge.checkpoint.state`` round-trip in memory. Nothing here uses
``CheckpointManager`` or DCP: ``tests/unit/test_checkpoint.py`` tests the manager
(``TestCheckpointManagerLoad`` saves and loads through DCP, including the optimizer
moments), and ``tests/distributed/test_checkpoint.py::TestCheckpointRoundTrip`` tests
it under FSDP2.
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
        """A model state dict reloaded through ``torch.save`` gives identical weights."""
        model = Transformer(CONFIG).to(DEVICE)

        state = model.state_dict()
        torch.save(state, tmp_path / "model.pt")

        model2 = Transformer(CONFIG).to(DEVICE)
        model2.load_state_dict(torch.load(tmp_path / "model.pt", map_location=DEVICE))

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters(), strict=True
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_optimizer_state_roundtrip(self, tmp_path):
        """An AdamW state dict reloaded through ``torch.save`` keeps each parameter's state.

        Its ``step``, ``exp_avg`` and ``exp_avg_sq`` must come back bit for bit. They are
        compared on the CPU: AdamW keeps ``step`` there unless it is fused or capturable,
        while the reloaded ``step`` lands on ``DEVICE``.
        """
        model = Transformer(CONFIG).to(DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        for _ in range(3):
            tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
            logits = model(tokens)
            loss = logits.sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        torch.save(opt.state_dict(), tmp_path / "opt.pt")

        model2 = Transformer(CONFIG).to(DEVICE)
        model2.load_state_dict(model.state_dict())
        opt2 = build_optimizer(model2, OptimizerConfig(lr=1e-3, fused=False))
        opt2.load_state_dict(torch.load(tmp_path / "opt.pt", map_location=DEVICE))

        for pg1, pg2 in zip(opt.param_groups, opt2.param_groups, strict=True):
            assert pg1["lr"] == pg2["lr"]
            assert pg1["weight_decay"] == pg2["weight_decay"]

        for (name, p1), (_, p2) in zip(
            model.named_parameters(), model2.named_parameters(), strict=True
        ):
            saved, loaded = opt.state[p1], opt2.state[p2]
            for key in ("step", "exp_avg", "exp_avg_sq"):
                assert key in loaded, f"{key} of {name} was not loaded"
                assert torch.equal(saved[key].cpu(), loaded[key].cpu()), f"{key} of {name} differs"

    def test_train_state_roundtrip(self):
        """build_train_state → restore_train_state round-trip."""
        sched_config = SchedulerConfig(warmup_steps=5, name="cosine")
        model = Transformer(CONFIG).to(DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        sched = build_scheduler(opt, sched_config, max_steps=100)

        for _ in range(10):
            opt.step()
            sched.step()

        state = build_train_state(step=10, tokens_seen=5000, scheduler=sched)
        assert state["step"] == 10
        assert state["tokens_seen"] == 5000

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

        vals1 = torch.randn(10)

        from kempnerforge.checkpoint.state import set_rng_state

        set_rng_state(rng_state)
        vals2 = torch.randn(10)

        assert torch.equal(vals1, vals2)

    def test_full_training_resume(self, tmp_path):
        """Train N steps, save the model and optimizer with ``torch.save``, reload them
        and train M more steps. Compare against uninterrupted N+M steps."""
        torch.manual_seed(42)
        model = Transformer(CONFIG).to(DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        tokens = torch.randint(0, 256, (4, 32), device=DEVICE)
        labels = tokens.clone()

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

        for _ in range(10):
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            opt.step()
            opt.zero_grad()

        torch.save(model.state_dict(), tmp_path / "model.pt")
        torch.save(opt.state_dict(), tmp_path / "opt.pt")

        model2 = Transformer(CONFIG).to(DEVICE)
        model2.load_state_dict(torch.load(tmp_path / "model.pt", map_location=DEVICE))
        opt2 = build_optimizer(model2, OptimizerConfig(lr=1e-3, fused=False))
        opt2.load_state_dict(torch.load(tmp_path / "opt.pt", map_location=DEVICE))

        for _ in range(10):
            logits = model2(tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            opt2.step()
            opt2.zero_grad()
        resumed_loss = loss.item()

        assert abs(ref_loss - resumed_loss) < 1e-4, (
            f"Resumed loss differs: ref={ref_loss:.6f}, resumed={resumed_loss:.6f}"
        )
