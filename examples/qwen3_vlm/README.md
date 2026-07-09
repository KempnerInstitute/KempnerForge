# Vision-Language Model with a Qwen3 backbone

A worked example: build KempnerForge VLMs (Joint-Decoder, Cross-Attention, MoT,
MoMa) on a **Qwen3-0.6B** backbone, optionally **warm-start** them from an
externally-trained (`KempnerInstitute/multimodal`) checkpoint, and **fine-tune
on video** (WebVid-10M).

The Qwen3 backbone is expressed **entirely through config** — there is no
Qwen-specific model class. It relies on a few small, *general* additions to
KempnerForge core (reusable by any model, not just this example):

- `ModelConfig.head_dim` — an overridable field, so the attention width can
  decouple from the model dim (Qwen3-0.6B: `dim=1024`, `head_dim=128` →
  2048-wide attention). Default preserves Llama-style `dim // n_heads`.
- `mlp_2layer` gains an optional `pre_norm` — a pre-projection norm (`ln_q`)
  over the vision features, selected by norm-registry key (`rmsnorm` /
  `layernorm`); off by default, so existing `mlp_2layer` configs are unchanged.
- Cross-attention `head_dim` threading — CA blocks honor the decoupled head_dim.
- `checkpoint.exclude_from_loading` is wired into the warm-start path in
  `scripts/train.py`, so a weights-only checkpoint loads without an optimizer.

Those live in `kempnerforge/`, tested alongside the components they extend
(`tests/unit/test_config.py`, `test_model.py`, `test_adapter.py`,
`test_cross_attention.py`). Everything specific to *this* use case lives here.

## Contents

```
examples/qwen3_vlm/
├── README.md
├── convert_multimodal_checkpoint.py   # multimodal .pt -> KempnerForge DCP
├── configs/
│   ├── vlm_qwen3_0.6b_joint_decoder.toml   # warm-start from converted JD ckpt
│   ├── vlm_qwen3_0.6b_mot.toml             # warm-start from converted MoT ckpt
│   ├── vlm_qwen3_0.6b_cross_attn.toml      # from scratch (see note)
│   └── vlm_qwen3_0.6b_moma.toml            # from scratch (no source ckpt)
└── tests/
    ├── test_configs.py       # configs load + build the right Qwen3-0.6B shapes
    └── test_converter.py     # the key mapping (pure, no GPU/network)
```

## The four arches

| arch | source checkpoint | this example |
|------|-------------------|--------------|
| `joint_decoder` | converts 1:1 | warm-start from a converted JD DCP |
| `mot` | converts 1:1, or init from a dense JD (duplicate per modality) | warm-start from a converted MoT DCP |
| `moma` | none (no multimodal-repo counterpart) | **from scratch** |
| `cross_attention` | exists, but doesn't convert faithfully yet | **from scratch** (see below) |

**Cross-attention note.** The `multimodal` repo's cross-attention is an extra
sub-module at every 4th layer that projects fused K/V straight from the **vision
dim (1152)** and reuses that layer's FFN, with a vision-side norm and QK-norm.
KF's `CrossAttentionBlock` is a separate block that projects K/V from the
**text/adapter dim (1024)** with its own FFN and no vision-side norm. So the
trained cross-attn tensors don't map 1:1. A faithful CA checkpoint needs a small
KF cross-attn alignment (identity adapter + K/V from the vision dim + QK/vision
norms) — tracked as its own PR. Until then, cross-attention trains from scratch.

## 1. Convert a checkpoint (optional; JD / MoT only)

Network-free: builds the KF transformer + adapter on `meta` to validate
keys/shapes and carries the SigLIP tower straight from the source (no download).
Reads only the model `*.pt` (optimizer state is ignored) and refuses to write
inside the source tree.

```bash
# Joint-Decoder (1:1)
uv run python examples/qwen3_vlm/convert_multimodal_checkpoint.py \
    --src   /path/to/stage-1/vlm/<run>/<ts>/model_epoch_0_100000.pt \
    --config examples/qwen3_vlm/configs/vlm_qwen3_0.6b_joint_decoder.toml \
    --out   /your/scratch/converted/jd

# Init a MoT checkpoint from a dense Joint-Decoder checkpoint (duplicate)
uv run python examples/qwen3_vlm/convert_multimodal_checkpoint.py \
    --src /path/to/jd_model.pt --from joint_decoder \
    --config examples/qwen3_vlm/configs/vlm_qwen3_0.6b_mot.toml \
    --out /your/scratch/converted/mot_from_jd

# Native MoT source (1:1) — state which source modality index is the image stream
uv run python examples/qwen3_vlm/convert_multimodal_checkpoint.py \
    --src /path/to/mot_model.pt \
    --config examples/qwen3_vlm/configs/vlm_qwen3_0.6b_mot.toml \
    --out /your/scratch/converted/mot --mot-image-index 0
```

Then point the config at the output:

```toml
[checkpoint]
load_path = "/your/scratch/converted/jd"
exclude_from_loading = ["optimizer", "dataloader"]   # weights-only warm start
```

## 2. Fine-tune on video (WebVid-10M)

Each config wires the `[video]` pipeline (SigLIP2 so400m + a pre-norm
`mlp_2layer`, **4 frames × 256 tokens/frame** = 1024 visual tokens, vision
tower frozen). Video decoding needs the `video` extra (PyAV) — run
`uv sync --group video` before the first finetune.

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py \
    examples/qwen3_vlm/configs/vlm_qwen3_0.6b_joint_decoder.toml
```

Before a real run, edit the placeholders in the config: `[checkpoint].load_path`
(your converted DCP), `[video].data_root` (your WebVid corpus), `[checkpoint].dir`,
and the `[metrics]` wandb fields. SigLIP2 (`google/siglip2-so400m-patch14-224`) and
the `Qwen/Qwen3-0.6B` tokenizer must be reachable (local path or `HF_HOME`).

**Frame budget.** The `mlp_2layer` connector keeps 256 tokens/frame (no pooling),
which is why these configs use 4 frames. For longer clips, switch `[adapter].type`
to a pooling connector (`avgpool` / `attentional_pool`) — that reduces tokens/frame
but re-initializes the adapter (the backbone + vision still warm-start).

## 3. Run the tests

Standalone (not part of the main `tests/` suite):

```bash
uv run pytest examples/qwen3_vlm/tests/
```
