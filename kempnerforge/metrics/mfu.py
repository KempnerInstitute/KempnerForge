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


def estimate_model_flops_per_token(config: ModelConfig) -> int:
    """Estimate FLOPS per token for forward + backward pass.

    Uses the PaLM paper approximation:
      flops_per_token = 6*P + 12*L*D*S

    Where:
      P = number of parameters (non-embedding)
      L = number of layers
      D = model dimension
      S = sequence length

    The 6× accounts for forward (2×) + backward (4×) matmul FLOPS.
    The 12×L×D×S term accounts for attention FLOPS.

    Args:
        config: Model configuration.

    Returns:
        Estimated FLOPS per token.
    """
    num_params = config.num_params_estimate
    return 6 * num_params + 12 * config.n_layers * config.dim * config.max_seq_len


def compute_mfu(
    config: ModelConfig,
    tokens_per_sec: float,
    num_gpus: int = 1,
    gpu_peak_tflops: float | None = None,
) -> float:
    """Compute Model FLOPs Utilization.

    Args:
        config: Model configuration.
        tokens_per_sec: Global throughput (tokens/sec across all GPUs).
        num_gpus: Number of GPUs.
        gpu_peak_tflops: Peak bf16 TFLOPS per GPU. Auto-detected if None.

    Returns:
        MFU as a fraction (0.0 to 1.0).
    """
    if gpu_peak_tflops is None:
        gpu_peak_tflops = get_gpu_peak_tflops()

    flops_per_token = estimate_model_flops_per_token(config)
    achieved_tflops = flops_per_token * tokens_per_sec / 1e12
    peak_tflops = gpu_peak_tflops * num_gpus

    if peak_tflops == 0:
        return 0.0

    return achieved_tflops / peak_tflops
