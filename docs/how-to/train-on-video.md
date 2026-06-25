# Train on video

The VLM path ingests **video** through the same wrapper, connectors, and fusion
archs as images — a clip is just an ordered set of frames. This guide covers the
data layout, the `[video]` config, the frame-sampling policy, and how all four
archs consume a clip.

## The model's view of a clip

A clip of `F` frames becomes `F × P′` visual tokens:

1. **Sample** `F` frames from the video by timestamp (target `fps`, uniform,
   first and last frame always kept).
2. **Encode** each frame with the frozen vision tower (e.g. SigLIP2), fold the
   frame axis into the batch so `B×F` frames run through the encoder once.
3. **Pool + project** each frame with the connector — an `avgpool` or
   `attentional_pool` adapter reduces a `grid×grid` patch map to
   `P′ = ceil(grid/window)²` tokens per frame (e.g. SigLIP2 @224/patch16 →
   14×14 → 49 tokens at `pool_window=2`).
4. **Fuse** the resulting `(B, F·P′, dim)` visual tokens into the backbone the
   same way images are fused — so **all four archs work unchanged**:
   - `joint_decoder` / `mot` / `moma`: the `F·P′` tokens prepend the text in the
     residual stream and are trimmed before the LM head.
   - `cross_attention`: the `F·P′` tokens flow as K/V into the cross-attention
     blocks; the residual stays text-only (so it fits more frames per
     `max_seq_len`).

Temporal order is carried by frame order (sequential positions). Per-frame
timestamp tokens and grounding outputs are a separate follow-up (see below).

## Token budget

For the residual-stream archs (JD / MoT / MoMa):

```
max_frames × tokens_per_frame + max_text_len  ≤  model.max_seq_len
```

e.g. 8 frames × 49 + 64 text = 456 ≤ 576. Cross-attention only needs
`max_text_len ≤ max_seq_len` (visual tokens are K/V, not in the residual). The
build- and config-time checks enforce this and fail before any GPU work.

## Configure it

A video run adds a `[video]` section (sibling of `[vision_encoder]` /
`[adapter]` / `[vlm]`) and a token-reducing connector. See
`configs/train/vlm_video_webvid.toml` for a complete example; the key parts:

```toml
[adapter]
type = "avgpool"          # or "attentional_pool"; pools patches per frame
pool_window = 2           # 14×14 grid -> 7×7 = 49 tokens/frame

[vlm]
arch = "joint_decoder"    # also: cross_attention | mot | moma

[video]
data_root = "/path/to/webvid-10m"
split = "train"           # "train" | "validation"
fps = 2.0                 # target sampling rate
max_frames = 8            # per-clip frame budget
min_frames = 4
frame_size = 224
max_samples = 0           # 0 = full manifest; set small for a smoke
```

The `[video]` section is decoded by `WebVidVideoDataset` (a WebVid-style layout:
CSV manifests under `raw/webvid-10M/data/<split>/partitions/` and `.mp4` files
under `raw/videos/<split>/`). Decoding uses PyAV (its wheel bundles FFmpeg, so no
system FFmpeg is required); it is imported lazily, so the package imports without
`av` and only actual decoding needs it.

## Launch

```bash
# 4-GPU video training (Joint-Decoder)
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/vlm_video_webvid.toml

# Quick smoke: no SigLIP download, a few clips, few steps
uv run torchrun --nproc_per_node=2 scripts/train.py configs/train/vlm_video_webvid.toml \
    --vision_encoder.type=random --vision_encoder.num_tokens=196 \
    --vision_encoder.feature_dim=768 --video.max_samples=256 --train.max_steps=20
```

To switch arch, change `[vlm].arch` in the config — everything else (frame
sampling, connector, dataset) is identical. (`arch` is resolved at config-load
time, so it is set in the TOML, not via a `--vlm.arch=` CLI override.)

## Constraints and follow-ups

- **Causal attention; no per-frame timestamps yet** — temporal order is frame
  order. Per-frame timestamp tokens + grounding (`<points>`/`<tracks>` outputs
  with point-F1 / track-J&F eval) are a follow-up.
- **Padded frames are not yet masked from attention** — short clips pad to
  `max_frames` with blank frames; a `frame_mask` is produced but not yet
  consumed by the attention mask.
- **Fixed `F` per batch** keeps tensor shapes static (for `torch.compile` and
  DP-rank consistency); variable-length clips arrive with VLM sequence packing.
- **Long-context** (many frames) is blocked on context-parallel being wired.
