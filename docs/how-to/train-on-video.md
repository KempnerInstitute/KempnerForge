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
   14×14 → 49 tokens at `pool_window=2`). Ragged windows work too — both
   connectors pool partial edge windows over their real patches — so a 3×3
   window on the 14×14 grid gives 5×5 = 25 tokens.
4. **Fuse** the resulting `(B, F·P′, dim)` visual tokens into the backbone the
   same way images are fused — so **all four archs work unchanged**:
   - `joint_decoder` / `mot` / `moma`: the `F·P′` tokens prepend the text in the
     residual stream and are trimmed before the LM head.
   - `cross_attention`: the `F·P′` tokens flow as K/V into the cross-attention
     blocks; the residual stays text-only (so it fits more frames per
     `max_seq_len`).

Temporal order is carried by frame order (sequential positions). **Optionally**,
each frame's **timestamp in seconds** can be embedded and added to that frame's
visual tokens, so the model sees *when* each frame occurs, not just its order.
This is **opt-in**: add a `[time_embedding]` section to enable it (within the
section `type` defaults to `sinusoidal` — sinusoidal features at log-spaced
periods through a zero-initialized projection; `type = "none"` disables it).
With **no `[time_embedding]` section (the default) no embedding is built**, and
the model is identical to one with no timestamps at all. It is fully decoupled —
added as a self-contained post-step in `VLMWrapper.forward`, so the fusion
strategies and the transformer backbone never see `frame_times` — and
registry-driven, so new techniques (learned, Fourier, …) register as small
additions and switch via config. Grounding outputs are a separate follow-up
(see below).

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
dataset_type = "webvid"      # registry key; add styles via @registry.register_video_dataset
dataset_name = "webvid-10M"  # corpus dir under raw/<dataset_name>/data (WebVid style)
sampling_policy = "uniform"  # registry key; the frame-sampling policy
split = "train"              # "train" | "validation"
fps = 2.0                    # target sampling rate
max_frames = 8               # per-clip frame budget
min_frames = 4
frame_size = 224
max_samples = 0              # 0 = full manifest; set small for a smoke
```

The dataset side is **pluggable**: `dataset_type` selects a builder from the
`video_dataset` registry, and `sampling_policy` selects a registered
frame-sampling policy (`"uniform"` is the default). The WebVid corpus directory
is parameterized by `dataset_name`, so any WebVid-style dataset works, not just
`webvid-10M`: CSV manifests under `raw/<dataset_name>/data/<split>/partitions/`
and `.mp4` files under `raw/videos/<split>/`.

## Corpora that ship

| `dataset_type` | Text | Layout under `data_root` |
|---|---|---|
| `webvid` | caption | `raw/<dataset_name>/data/<split>/partitions/*.csv` + prefix-nested `raw/videos/<split>/` |
| `molmo2_videocapqa` | caption from a yt-dlp sidecar | `videos/<subset>/<id>/<id>.{mp4,mkv,webm}` (+ `<id>.json` / `<id>.grover.json`) |
| `perception_test` | 3-way MCQ | `mc_question_<split>.json` + flat `videos/<video_id>.mp4` |
| `nextqa` | 5-way MCQ (`subset="MC"`) or free text (`"OE"`) | `annotations/{MC,OE}/<split>.{csv,parquet}` + flat `videos/<video>.mp4` |
| `cinepile` | 5-way MCQ | `<dataset_name>/<split>-*.parquet` (`v1`/`v2`) + flat `videos/<ytid>.mp4` |

Add another corpus by subclassing `VideoQADataset` — it only needs a
`VideoRecord(video_path, prompt, target)` per index; decode, frame padding,
timestamps, prompt masking and the skip-with-mask path are inherited.

**Question rendering** is a registry knob, not a per-corpus decision.
`qa_format` picks how an MCQ becomes supervised text:

| `qa_format` | Target | Notes |
|---|---|---|
| `mcq_letter` (default) | `" C"` | ~2 supervised tokens; matches the usual MC eval protocol |
| `mcq_letter_text` | `" C. lay on floor"` | still letter-parseable, supervises the wording |
| `mcq_text` | `" lay on floor"` | generative setup |

The question and its options go in the prompt, which `-100` masks out of the
loss. Give MCQ corpora enough `vlm.max_text_len` for the whole option list —
CinePile's five long options need ~192 — or the answer is truncated away and the
sample silently trains on nothing. Every corpus logs the fraction of its first
128 samples that supervise at least one token at startup, and warns below 50%;
watch that line.

## Mix several corpora

List `[[video.datasets]]` entries instead of the flat corpus fields. Frame
geometry (`fps`, `max_frames`, `min_frames`, `frame_size`, `sampling_policy`)
stays global on `[video]` — it sizes the visual-token budget checked against
`model.max_seq_len`, so every source must share it, and setting it on a source
is rejected as an unknown key.

```toml
[video]                      # geometry: global, shared by every source
fps = 2.0
max_frames = 8
min_frames = 2
frame_size = 224

[[video.datasets]]
dataset_type = "webvid"
data_root = "/path/to/webvid-10m"
prompt = "Describe the video:"
weight = 1.0

[[video.datasets]]
dataset_type = "nextqa"
data_root = "/path/to/NExTQA"
subset = "MC"
weight = 2.0
```

Per-source fields: `dataset_type`, `data_root`, `dataset_name`, `subset`,
`split`, `prompt`, `text_source`, `qa_format`, `max_samples`,
`require_video_file`, `weight`, `name`. Anything left empty inherits the
`[video]` value, so shared settings are written once.

This reuses the text path's `MixtureDataset` + `MixtureSampler`, so
`data.mix_temperature` and `[[data.phases]]` apply, and each corpus gets its own
`loss/<name>` and `data/<name>/tokens` metric. Sources are concatenated into one
index space and drawn by weight; `name` disambiguates two sources of the same
type. See `configs/train/vlm_video_stage1_mix.toml` for a complete example.

**Weights control rows, not gradient share.** There is one loss: a batch mixes
rows from every corpus and cross-entropy averages over all supervised *tokens*
in it, so a corpus's real contribution is `rows × supervised-tokens-per-row`.
A caption supervises ~20 tokens and an `mcq_letter` answer ~2, so at equal
weights caption corpora take roughly 87% of the gradient in a caption+MCQ mix.
Raise the MCQ `weight`, or switch those sources to `mcq_letter_text`, to
rebalance. The per-corpus `loss/<name>` values are diagnostics computed under
`no_grad` — they are never combined or backwarded.

**Partially-downloaded corpora**: `require_video_file` drops manifest rows whose
video is missing, so they cost no decode and no wasted step. It defaults on for
the QA corpora and off where an existence scan would be prohibitive (WebVid's
10M-row manifest).

Decoding uses **PyAV**, an **optional** dependency (its wheel bundles FFmpeg, so
no system FFmpeg is required): install it with `uv sync --group video`. It is
imported lazily, so the package imports without `av` and only actual decoding
requires it.

## Query-aware frame selection (optional)

By default a clip is decoded to `max_frames` frames uniformly. Add a
`[frame_selector]` section to instead decode a larger **candidate pool** and keep
the `max_frames` frames most relevant to each sample's query — a training-free,
plug-in selection stage that runs on the data path (no model changes). With no
`[frame_selector]` section the path is bit-identical to uniform sampling.

```toml
[frame_selector]
type = "mdp3"            # "topk" (cosine) | "qframe" (Gumbel-Max top-k) | "mdp3"
scorer = "siglip2"       # frozen dual encoder for frame/query embeddings ("clip" too)
scorer_path = "google/siglip2-base-patch16-224"
candidate_frames = 32    # decoded pool; must be >= [video].max_frames
mdp3_lambda = 0.2        # relevance/diversity trade-off (mdp3)
mdp3_segment_size = 32   # temporal segment length for sequentiality; 0 = plain conditional DPP
```

- **Selectors.** `topk` = cosine top-k of frame/query similarity; `qframe` =
  Q-Frame QFS, a Gumbel-Max top-k over `softmax(sim/τ)` (stochastic, seeded per
  sample); `mdp3` = Markov-DPP list-wise selection balancing query relevance,
  list-wise diversity, and temporal sequentiality.
- **Query.** Selection always conditions on the sample's question/instruction
  prompt — the text the model is given at inference — never on its target
  (caption/answer), which is not available at test time. A captioning corpus,
  whose prompt is a static instruction or empty, therefore has no meaningful
  per-sample query: with an empty prompt the selector logs a one-time warning and
  falls back to uniform sampling. This is deliberate — the expectation is that
  query-aware selection helps VQA but not captioning, which the eval can falsify.
- **Dataset-agnostic.** Selection is a `VideoDataset` base-class capability, so a
  new dataset style adopts it in three lines: take `frame_selector_config` in its
  builder, call `self._init_frame_selector(...)`, and decode via
  `self._decode_clip(path, query, ...)`. A dataset without a decodable file path
  can call `self._frame_selector.select(frames, times, query, k, seed_key=...)`
  directly.
- **Scorer prefetch + cost.** The scorer weights load lazily on first use, once
  per worker; prefetch them on a networked node (set `HF_HOME` to the shared
  cache) so offline compute nodes don't fail mid-run. The default scorer
  (`so400m` at 128 candidates) is **eval-grade** — dataloader workers run at one
  intra-op thread, so for CPU-worker *training* use a base-scale scorer and a
  modest `candidate_frames` (as above); real-scale training belongs in an offline
  precompute step (future work).
- **Default deviation.** The default `scorer_path` is SigLIP2-so400m @ 224 (stack
  consistency, cheaper scoring); the mDP3 paper used SigLIP(v1)-so400m @ 384 — one
  `scorer_path` away.

At eval, the same selection turns on per-checkpoint via the harness
`--override frame_selector.type=...` flag (see the vlm-evaluation README), so an
existing checkpoint can be A/B'd across selectors without editing its config.

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

- **Grounding outputs are a follow-up** — per-frame timestamps are encoded (see
  above), but structured grounding (`<points>`/`<tracks>` outputs with point-F1
  / track-J&F eval) is not yet implemented.
- **Sequence-modifying time encodings are a separate hook** — the
  `[time_embedding]` registry is for *additive* per-frame embeddings (no change
  to sequence length). Interleaved text time-tokens change the
  token sequence and need interleaved/variable-length sequence support KF does
  not have yet; they would hook the sequence-assembly layer, not this registry.
- **Inference must pass `frame_times`** — a video model silently drops the
  learned temporal signal if `frame_times` is `None` (no error is raised).
  Training threads it automatically, and the vlm-evaluation adapter now threads it
  too (per-frame times of the decoded/selected clip); any other custom
  generate path must pass it for video models.
- **Checkpoint compatibility** — because the time embedding is opt-in (off by
  default), a default video model has no `frame_time_embed` parameters and loads
  pre-timestamp checkpoints unchanged. As for any component, config must match
  the checkpoint: a checkpoint trained *with* a `[time_embedding]` section must
  be resumed/evaluated with the same section, and one trained without must not
  add one (a mismatch fails the strict load, exactly like changing the adapter).
- **Padded frames are masked from attention** — short/undecodable clips pad to
  `max_frames` with blank frames, and the `frame_mask` is consumed so real
  tokens never attend to padded-frame visual tokens (MoMa also drops them from
  expert-choice routing); a NaN guard keeps an all-padded clip finite. It is a
  pure mask (no new checkpoint keys); image/text keep the FlashAttention-2 path.
  For the image-prefix arches (Joint-Decoder/MoT/MoMa), video self-attention
  always takes the explicit-mask SDPA path (FA2 disabled, a `(B,1,S,S)` mask
  built) even for fully-decoded clips — a deliberate compile/DP-friendly
  trade-off; recovering FA2 / FlexAttention is a follow-up. (Cross-Attention
  keeps FA2 on its text self-attention; it masks padded image K/V in the
  cross-attention blocks instead.) *Remaining:* MoT
  configured with an MoE FFN still routes padded tokens through the shared MoE
  (a "generic token-validity in MoE" follow-up).
- **Fixed `F` per batch** keeps tensor shapes static (for `torch.compile` and
  DP-rank consistency); variable-length clips arrive with VLM sequence packing.
- **Long-context** (many frames) is blocked on context-parallel being wired.
