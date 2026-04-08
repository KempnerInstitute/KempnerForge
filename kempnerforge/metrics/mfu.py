"""Model FLOPs Utilization (MFU) computation.

Implements the PaLM paper formula for estimating achieved FLOPS relative
to hardware peak, with auto-detection of GPU capabilities.

MFU = achieved_tflops / peak_tflops

Where:
  model_flops_per_token = 6*P + 12*L*D*S  (forward + backward)
  achieved_tflops = model_flops_per_token * tokens_per_sec / 1e12
"""

from __future__ import annotations

import logging

import torch

from kempnerforge.config.schema import ModelConfig

logger = logging.getLogger(__name__)

# Peak bf16 TFLOPS for common GPU types (per GPU)
# Source: NVIDIA specs (dense tensor core throughput)
_GPU_PEAK_TFLOPS: dict[str, float] = {
    # H-series
    "H200": 989.0,
    "H100": 989.0,
    "H100 SXM": 989.0,
    "H100 PCIe": 756.0,
    "H800": 989.0,
    # A-series
    "A100": 312.0,
    "A100 SXM": 312.0,
    "A100 PCIe": 312.0,
    "A100-SXM4-80GB": 312.0,
    "A100-SXM4-40GB": 312.0,
    "A100-PCIE-80GB": 312.0,
    "A100-PCIE-40GB": 312.0,
    # Consumer / other
    "A10G": 125.0,
    "L40S": 362.0,
    "RTX 4090": 330.0,
    "RTX 3090": 142.0,
}


def get_gpu_peak_tflops(device: int = 0) -> float:
    """Auto-detect GPU peak bf16 TFLOPS.

    Tries to match the GPU name against known models. Falls back to a
    conservative estimate based on compute capability.

    Args:
        device: CUDA device index.

    Returns:
        Peak bf16 TFLOPS for this GPU.
    """
    if not torch.cuda.is_available():
        return 1.0  # dummy for CPU-only

    name = torch.cuda.get_device_name(device)

    # Try exact and substring matches
    for gpu_name, tflops in _GPU_PEAK_TFLOPS.items():
        if gpu_name in name:
            logger.info(f"Detected GPU: {name} → {tflops} bf16 TFLOPS")
            return tflops

    # Fallback: estimate from compute capability
    major, minor = torch.cuda.get_device_capability(device)
    if major >= 9:
        # Hopper-class
        tflops = 989.0
    elif major >= 8:
        # Ampere-class
        tflops = 312.0
    else:
        tflops = 100.0

    logger.warning(
        f"Unknown GPU: {name} (cc {major}.{minor}). "
        f"Using estimated {tflops} bf16 TFLOPS. "
        f"Add this GPU to _GPU_PEAK_TFLOPS for accuracy."
    )
    return tflops


def estimate_model_flops_per_token(config: ModelConfig, seq_len: int | None = None) -> int:
    """Estimate FLOPS per token for forward + backward pass.

    Uses the PaLM paper approximation: ``6*P + 12*L*D*S``

    For MoE: uses active params (top_k experts per layer, not all experts).
    Excludes embedding (table lookup, not matmul). Includes output projection.
    The 12*L*D*S attention term does not discount GQA — FlashAttention expands
    GQA internally, so the hardware performs full attention compute.
    Router FLOPS (dim × num_experts) are intentionally omitted — negligible.

    Args:
        config: Model configuration.
        seq_len: Actual training sequence length. Falls back to
            config.max_seq_len if not provided.

    Returns:
        Estimated FLOPS per token.
    """
    s = seq_len if seq_len is not None else config.max_seq_len
    if config.is_moe:
        return _moe_flops_per_token(config, s)
    return _dense_flops_per_token(config, s)


def _dense_flops_per_token(config: ModelConfig, seq_len: int) -> int:
    head_dim = config.head_dim
    attn_params = (
        config.dim * (config.n_heads * head_dim)  # Q
        + 2 * config.dim * (config.n_kv_heads * head_dim)  # K + V
        + (config.n_heads * head_dim) * config.dim  # O
    )
    mlp_params = 3 * config.dim * config.computed_ffn_hidden_dim  # SwiGLU
    per_layer = attn_params + mlp_params
    output_params = config.vocab_size * config.dim
    active_params = config.n_layers * per_layer + output_params
    return 6 * active_params + 12 * config.n_layers * config.dim * seq_len


def _moe_flops_per_token(config: ModelConfig, seq_len: int) -> int:
    head_dim = config.head_dim
    attn_params = (
        config.dim * (config.n_heads * head_dim)
        + 2 * config.dim * (config.n_kv_heads * head_dim)
        + (config.n_heads * head_dim) * config.dim
    )
    mlp_params = 3 * config.dim * config.computed_ffn_hidden_dim

    n_moe_layers = sum(
        1 for i in range(config.n_layers) if (i + 1) % config.moe_frequency == 0
    )
    n_dense_layers = config.n_layers - n_moe_layers

    dense_active = n_dense_layers * (attn_params + mlp_params)
    shared_mlp = config.moe_shared_experts * mlp_params
    moe_active = n_moe_layers * (attn_params + config.moe_top_k * mlp_params + shared_mlp)

    output_params = config.vocab_size * config.dim
    active_params = dense_active + moe_active + output_params
    return 6 * active_params + 12 * config.n_layers * config.dim * seq_len


def compute_mfu(
    config: ModelConfig,
    tokens_per_sec: float,
    num_gpus: int = 1,
    gpu_peak_tflops: float | None = None,
    seq_len: int | None = None,
) -> float:
    """Compute Model FLOPs Utilization.

    Args:
        config: Model configuration.
        tokens_per_sec: Global throughput (tokens/sec across all GPUs).
        num_gpus: Number of GPUs.
        gpu_peak_tflops: Peak bf16 TFLOPS per GPU. Auto-detected if None.
        seq_len: Actual training sequence length for attention FLOPS.
            Falls back to config.max_seq_len if not provided.

    Returns:
        MFU as a fraction (0.0 to 1.0).
    """
    if gpu_peak_tflops is None:
        gpu_peak_tflops = get_gpu_peak_tflops()

    flops_per_token = estimate_model_flops_per_token(config, seq_len=seq_len)
    achieved_tflops = flops_per_token * tokens_per_sec / 1e12
    peak_tflops = gpu_peak_tflops * num_gpus

    if peak_tflops == 0:
        return 0.0

    return achieved_tflops / peak_tflops
