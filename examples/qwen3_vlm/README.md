# Video training on WebVid-10M with a Qwen3 backbone

A worked, self-contained example: build a Joint-Decoder VLM on a **Qwen3-0.6B**
backbone from pretrained components (a HuggingFace LLM + SigLIP2), then train it
to caption video on **WebVid-10M**.

The Qwen3 backbone is expressed **entirely through config** — there is no
Qwen-specific model class. It relies on two *general* KempnerForge features
(reusable by any model, already in core and unit-tested there):

- `ModelConfig.head_dim` — an overridable field, so the attention width can
  decouple from the model dim (Qwen3-0.6B: `dim=1024`, `head_dim=128` →
  2048-wide attention). Default preserves Llama-style `dim // n_heads`.
- `checkpoint.exclude_from_loading` — wired into the warm-start path in
  `scripts/train.py`, so a weights-only checkpoint loads without an optimizer.

Nothing in `kempnerforge/` is modified by this example.

## Contents

```
examples/qwen3_vlm/
├── README.md
├── convert_hf_backbone.py   # HF LLM + configured vision encoder -> one DCP
├── configs/
│   └── vlm_qwen3_0.6b_joint_decoder_webvid.toml
└── tests/
    ├── test_configs.py       # the config loads + builds the right Qwen3 shapes
    └── test_hf_backbone.py    # HF key mapping + tied head (pure)
```

## How it fits together

| piece | choice |
|-------|--------|
| backbone | Qwen3-0.6B, warm-started from HuggingFace (frozen) |
| vision | SigLIP2 so400m patch14 @ 224 px, pretrained (trains) |
| connector | `avgpool`, `pool_window = 2` (fresh; trains) |
| arch | `joint_decoder` — visual tokens are prepended to the text sequence |

**Phase-1 alignment.** The pretrained LLM is frozen (`freeze = transformer`); only
the vision encoder and the fresh adapter learn, so the visual pathway adapts to a
fixed language model.

**Frame budget.** SigLIP2 emits a 16×16 = 256-patch grid per frame. `avgpool` with
`pool_window = 2` averages each 2×2 neighbourhood into one token → 8×8 = **64
tokens/frame**, so 18 frames × 64 = 1152 visual + 96 text = 1248 ≤ `max_seq_len`
(1280). Trading per-frame detail for 4.5× the temporal coverage is what makes
many-frame video fit. Frames are sampled uniformly in time at 2 fps, capped at 18.

**Captions** are trained next-token with an appended EOS (so generation learns to
stop). `[video].prompt` is masked from the loss, and its last token predicts the
first caption token — first-token supervision plus a generation seed.

## 1. Build a starting checkpoint

Qwen3 LLM weights from HuggingFace, the pretrained SigLIP2 tower, and a fresh
adapter, saved as one **complete** DCP the warm-start load accepts (any key the
source does not supply keeps its init). Dense / joint-decoder targets only.

```bash
uv run python examples/qwen3_vlm/convert_hf_backbone.py \
    --hf-dir Qwen/Qwen3-0.6B \
    --config examples/qwen3_vlm/configs/vlm_qwen3_0.6b_joint_decoder_webvid.toml \
    --out   path-to-init-checkpoint
```

The config already points `[checkpoint].load_path` at that output and loads it
weights-only:

```toml
[checkpoint]
load_path = "path-to-init-checkpoint"
exclude_from_loading = ["optimizer", "dataloader"]
```

`tie_embeddings = true` matches HF Qwen3-0.6B, which ties its output head to the
token embedding.

## 2. Train on WebVid-10M

Video decoding needs the `video` extra (PyAV) — run `uv sync --group video` once.

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py \
    examples/qwen3_vlm/configs/vlm_qwen3_0.6b_joint_decoder_webvid.toml
```

Fill in the placeholders first: `[video].data_root` (your WebVid corpus),
`[checkpoint].load_path` and `.dir`, and the `[metrics]` wandb fields. SigLIP2
(`google/siglip2-so400m-patch14-224`) and the `Qwen/Qwen3-0.6B` tokenizer must be
reachable (local path or `HF_HOME`).

For a quick shakedown instead of the full epoch, override on the command line:

```bash
--train.max_steps=20 --video.max_samples=512 --metrics.enable_wandb=false
```

The config as written targets 16 H200s (4 nodes × 4): global batch 256, lr 8e-5,
41,905 steps ≈ one epoch over the ~10.7M-clip manifest.

## 3. Run the tests

Standalone (not part of the main `tests/` suite):

```bash
uv run pytest examples/qwen3_vlm/tests/
```
