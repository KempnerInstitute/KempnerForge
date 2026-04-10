# Next Features

Training components to add before continuing the MoE engineering roadmap (Phases 10-16).

---

## Must Have

### 1. Z-Loss (logit magnitude regularizer)

PaLM/Gemini stability technique. Adds a small penalty on the log of the sum of exponentials of logits, preventing logit drift that causes NaN/divergence in long training runs. Cheap insurance — negligible compute cost.

- **Where**: `kempnerforge/training/loss.py` — new registry entry `"z_loss"`
- **Config**: `z_loss_weight: float = 0.0` in `TrainConfig` (0 = disabled, PaLM uses 1e-4)
- **Formula**: `z = z_loss_weight * (logsumexp(logits, dim=-1) ** 2).mean()`
- **Integration**: Added to main loss in training loop, logged as separate metric
- **Tests**: Unit test verifying z-loss is zero when logits are small, positive when logits are large

### 2. Chunked Cross-Entropy

Computes cross-entropy in vocabulary chunks to avoid materializing the full `(batch*seq, vocab_size)` logit tensor. At vocab=128K, batch=8, seq=4096, the naive logit tensor is 128K * 32K * 4 bytes = 16 GB. Chunking computes loss per chunk and reduces memory to `chunk_size / vocab_size` of that.

- **Where**: `kempnerforge/training/loss.py` — new registry entry `"chunked_cross_entropy"`
- **Config**: `loss_fn: str = "cross_entropy"` already exists; add `"chunked_cross_entropy"` option. `ce_chunk_size: int = 0` in `TrainConfig` (0 = auto, e.g. 4096)
- **Approach**: Loop over vocab dimension in chunks, compute `F.cross_entropy` per chunk, accumulate. Linear layer output is also chunked (compute `hidden @ weight[chunk]` per chunk instead of full matmul)
- **Tests**: Unit test verifying chunked output matches standard CE within fp32 tolerance

### 3. Muon Optimizer

Momentum with orthogonalized updates. Uses SVD-based update orthogonalization to maintain update directions independent of parameter scale. Consistently matches or beats AdamW with less hyperparameter sensitivity in recent large-scale training runs.

- **Where**: `kempnerforge/training/optimizer.py` — new registry entry `"muon"`
- **Config**: `name: str = "muon"` in `OptimizerConfig`, reuses `lr`, `weight_decay`, `betas[0]` for momentum
- **Implementation**: Newton-Schulz iteration (5 steps) to approximate orthogonal projection of momentum — no actual SVD needed. Apply Muon to 2D+ weight matrices, fall back to AdamW for embeddings/norms/biases (1D params)
- **Key detail**: The Newton-Schulz iteration is `X = a*X + b*X@X^T@X + c*X@X^T@X@X^T@X` with specific constants — 15 FLOPs per parameter per step, negligible vs forward/backward
- **Tests**: Unit test verifying Muon produces finite updates and decreases loss on a small model. Test that 1D params get AdamW treatment.

---

## Good to Have

### 4. Adafactor Optimizer

Memory-efficient Adam variant from Google (T5, PaLM). Replaces the full second moment matrix with row and column factors — reduces optimizer state from 2x to ~1.3x model size. Useful when memory-constrained at 70B+ scale.

- **Where**: `kempnerforge/training/optimizer.py` — new registry entry `"adafactor"`
- **Config**: Reuses existing `OptimizerConfig` fields; add `adafactor_decay: float = -0.8` for second moment decay schedule
- **Implementation**: Use `torch.optim.Adafactor` if available (PyTorch nightly), otherwise implement factored second moment from the paper
- **Tests**: Unit test verifying optimizer state is smaller than AdamW for a 2D parameter

### 5. Schedule-Free AdamW

Meta's approach that eliminates LR schedule tuning. Maintains two interpolated parameter sequences — no warmup or decay schedule needed. One fewer hyperparameter to sweep.

- **Where**: `kempnerforge/training/optimizer.py` — new registry entry `"schedule_free_adamw"`
- **Config**: `name: str = "schedule_free_adamw"`, uses existing `lr`, `betas`, `weight_decay`. Scheduler is ignored when this optimizer is selected.
- **Implementation**: Use `schedulefree` package (Meta open source) or implement the weight interpolation directly
- **Interaction**: Training loop needs `optimizer.train()` / `optimizer.eval()` calls before forward and before eval/checkpoint
- **Tests**: Unit test verifying loss decreases without any LR scheduler. Test checkpoint save/load of interpolated state.

### 6. GeGLU Activation

GELU-gated GLU variant. Same architecture as SwiGLU but replaces SiLU gate with GELU. Used in some models (Gemma, CodeGen2). Trivial to add since we already have the gated MLP structure.

- **Where**: `kempnerforge/model/mlp.py` — `SwiGLUMLP` already supports activation dispatch
- **Config**: `activation: str = "geglu"` — the existing `Activation` enum gets a new entry, `build_mlp` routes to gated MLP with GELU
- **Tests**: Unit test verifying GeGLU MLP produces same shapes as SwiGLU
