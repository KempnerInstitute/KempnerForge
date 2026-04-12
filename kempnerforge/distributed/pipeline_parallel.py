"""Pipeline parallelism for KempnerForge.

Splits a Transformer model across pipeline stages, assigning layer ranges
to different ranks. Uses torch.distributed.pipelining for schedule execution.

Stage assignment:
  - Stage 0: token_embedding + first chunk of transformer layers
  - Middle stages: transformer layer chunks only
  - Last stage: last chunk of layers + final norm + output head

Application order when combining parallelisms:
  1. Build per-stage model via build_stage_module()
  2. Tensor parallelism (per stage, via apply_tensor_parallel) — must see raw blocks
  3. Activation checkpointing (per stage, via apply_ac)
  4. FSDP2 (per stage, via apply_fsdp2 with reshard_after_forward=False)

Note on FSDP reshard policy:
  When using PP, set reshard_after_forward=False in apply_fsdp2 to avoid
  per-microbatch all-gathers. PP schedules send multiple microbatches through
  each stage, so keeping gathered params avoids redundant communication.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.embedding import OutputHead, TokenEmbedding
from kempnerforge.model.init import init_weights
from kempnerforge.model.norm import build_norm
from kempnerforge.model.position import precompute_rope_frequencies
from kempnerforge.model.transformer import TransformerBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


def get_pp_mesh(device_mesh: DeviceMesh) -> DeviceMesh | None:
    """Extract the PP sub-mesh from a DeviceMesh.

    Returns None if no 'pp' dimension exists.
    """
    if "pp" not in device_mesh.mesh_dim_names:  # type: ignore[reportOperatorIssue]
        return None
    return device_mesh["pp"]


def get_pp_rank(device_mesh: DeviceMesh) -> int:
    """Get the pipeline parallel rank for this process."""
    pp_mesh = get_pp_mesh(device_mesh)
    if pp_mesh is None:
        return 0
    return pp_mesh.get_local_rank()


def get_pp_size(device_mesh: DeviceMesh) -> int:
    """Get the pipeline parallel world size."""
    pp_mesh = get_pp_mesh(device_mesh)
    if pp_mesh is None:
        return 1
    return pp_mesh.size()


# ---------------------------------------------------------------------------
# Layer assignment
# ---------------------------------------------------------------------------


def compute_layer_assignment(
    n_layers: int,
    pp_size: int,
) -> list[tuple[int, int]]:
    """Compute which layers go to which PP stage.

    Distributes layers as evenly as possible. Earlier stages get one extra
    layer when n_layers is not evenly divisible by pp_size.

    Args:
        n_layers: Total number of transformer layers.
        pp_size: Number of pipeline stages.

    Returns:
        List of (start_layer, end_layer) tuples, one per stage.
        end_layer is exclusive.

    Raises:
        ValueError: If pp_size > n_layers.
    """
    if pp_size > n_layers:
        raise ValueError(f"pp_size ({pp_size}) cannot exceed n_layers ({n_layers})")

    base = n_layers // pp_size
    remainder = n_layers % pp_size

    assignments = []
    start = 0
    for stage_id in range(pp_size):
        count = base + (1 if stage_id < remainder else 0)
        assignments.append((start, start + count))
        start += count

    return assignments


# ---------------------------------------------------------------------------
# Stage model
# ---------------------------------------------------------------------------


class PipelineStageModule(nn.Module):
    """A model chunk for a single pipeline stage.

    Only allocates parameters needed for this stage:
      - Stage 0: token_embedding + assigned transformer layers
      - Middle stages: assigned transformer layers only
      - Last stage: assigned transformer layers + final norm + output head

    Layer keys in self.layers match the full Transformer (e.g. "4", "5", "6")
    to maintain DCP checkpoint compatibility.
    """

    def __init__(
        self,
        config: ModelConfig,
        stage_id: int,
        num_stages: int,
        layer_range: tuple[int, int],
    ) -> None:
        super().__init__()
        self.config = config
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.is_first = stage_id == 0
        self.is_last = stage_id == num_stages - 1

        start, end = layer_range

        # Token embedding — only on first stage
        self.token_embedding: TokenEmbedding | None = (
            TokenEmbedding(config.vocab_size, config.dim) if self.is_first else None
        )

        # Assigned transformer blocks (string keys match full model for DCP compat)
        self.layers = nn.ModuleDict(
            {str(i): TransformerBlock(config, layer_idx=i) for i in range(start, end)}
        )

        # Final norm + output head — only on last stage
        self.norm = (
            build_norm(config.norm_type, config.dim, eps=config.norm_eps) if self.is_last else None
        )
        self.output_head: OutputHead | None = (
            OutputHead(config.dim, config.vocab_size) if self.is_last else None
        )

        # Precompute RoPE cos/sin tables and initialize weights.
        # Skip when on meta device (no data); call init_weights_and_freqs() later.
        self._rope_cos = None
        self._rope_sin = None
        if not any(p.is_meta for p in self.parameters()):
            self._rope_cos, self._rope_sin = precompute_rope_frequencies(
                head_dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
            init_weights(self, config)

        logger.info(
            f"PP stage {stage_id}/{num_stages}: layers [{start}, {end}), "
            f"embedding={'yes' if self.is_first else 'no'}, "
            f"output={'yes' if self.is_last else 'no'}"
        )

    def init_weights_and_freqs(self) -> None:
        """Initialize weights and RoPE frequencies after meta-device materialization."""
        if self._rope_cos is None:
            self._rope_cos, self._rope_sin = precompute_rope_frequencies(
                head_dim=self.config.head_dim,
                max_seq_len=self.config.max_seq_len,
                theta=self.config.rope_theta,
            )
        init_weights(self, self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for this pipeline stage.

        Args:
            x: For stage 0: token IDs of shape (batch, seq_len).
               For other stages: hidden states of shape (batch, seq_len, dim).

        Returns:
            For last stage: logits of shape (batch, seq_len, vocab_size).
            For other stages: hidden states of shape (batch, seq_len, dim).
        """
        # First stage: embed tokens
        if self.is_first and self.token_embedding is not None:
            x = self.token_embedding(x)

        # Slice RoPE for current sequence length (device transfer cached after first call)
        seq_len = x.shape[1]
        if self._rope_cos.device != x.device:  # type: ignore[reportOptionalMemberAccess]
            self._rope_cos = self._rope_cos.to(x.device)  # type: ignore[reportOptionalMemberAccess]
            self._rope_sin = self._rope_sin.to(x.device)  # type: ignore[reportOptionalMemberAccess]
        cos = self._rope_cos[:seq_len]  # type: ignore[reportOptionalSubscript]
        sin = self._rope_sin[:seq_len]  # type: ignore[reportOptionalSubscript]

        # Run through assigned layers
        for layer in self.layers.values():
            x = layer(x, cos, sin)

        # Last stage: norm + output head
        if self.is_last:
            x = self.norm(x)  # type: ignore[reportOptionalCall]
            if self.output_head is not None:
                x = self.output_head(x)

        return x


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def build_stage_module(
    config: ModelConfig,
    pp_rank: int,
    pp_size: int,
) -> PipelineStageModule:
    """Build the model chunk for a specific pipeline stage.

    Args:
        config: Model configuration.
        pp_rank: This process's pipeline rank (0-indexed).
        pp_size: Total number of pipeline stages.

    Returns:
        A PipelineStageModule containing only the parameters for this stage.
    """
    assignments = compute_layer_assignment(config.n_layers, pp_size)
    layer_range = assignments[pp_rank]

    return PipelineStageModule(
        config=config,
        stage_id=pp_rank,
        num_stages=pp_size,
        layer_range=layer_range,
    )


def build_pipeline_stage(
    stage_module: PipelineStageModule,
    device_mesh: DeviceMesh,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    param_dtype: torch.dtype = torch.bfloat16,
) -> torch.distributed.pipelining.PipelineStage:  # type: ignore[reportAttributeAccessIssue]
    """Wrap a stage module in a PipelineStage for schedule execution.

    Args:
        stage_module: The model chunk for this stage.
        device_mesh: Full DeviceMesh with a 'pp' dimension.
        device: Device for this stage.
        batch_size: Micro-batch size (for shape inference).
        seq_len: Sequence length (for shape inference).
        param_dtype: Dtype for intermediate activations (matches mixed precision).

    Returns:
        A PipelineStage ready for use with a pipeline schedule.
    """
    from torch.distributed.pipelining import PipelineStage

    pp_mesh = get_pp_mesh(device_mesh)
    pp_group = pp_mesh.get_group() if pp_mesh is not None else None

    # Example input for shape inference
    if stage_module.is_first:
        input_args = (torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),)
    else:
        input_args = (
            torch.zeros(
                batch_size,
                seq_len,
                stage_module.config.dim,
                dtype=param_dtype,
                device=device,
            ),
        )

    return PipelineStage(
        submodule=stage_module,
        stage_index=stage_module.stage_id,
        num_stages=stage_module.num_stages,
        device=device,
        input_args=input_args,
        group=pp_group,
    )


def build_pipeline_schedule(
    stage: torch.distributed.pipelining.PipelineStage,  # type: ignore[reportAttributeAccessIssue]
    n_microbatches: int,
    loss_fn: callable,  # type: ignore[reportGeneralTypeIssues]
    schedule: str = "1f1b",
):
    """Create a pipeline execution schedule.

    Args:
        stage: The PipelineStage for this rank.
        n_microbatches: Number of microbatches per training step.
            Must be >= pp_size for 1F1B to fill the pipeline.
        loss_fn: Loss function (applied on last stage only).
        schedule: Schedule type — "1f1b", "gpipe", or "interleaved_1f1b".
            Note: "interleaved_1f1b" requires multiple stages per rank
            (virtual pipeline stages). Pass a list of stages instead.

    Returns:
        A pipeline schedule object. Call schedule.step() in the training loop:
          - First stage: schedule.step(input_tensor, target=labels)
          - Other stages: schedule.step(target=labels) or schedule.step()
    """
    from torch.distributed.pipelining.schedules import (
        Schedule1F1B,
        ScheduleGPipe,
        ScheduleInterleaved1F1B,
    )

    if schedule == "interleaved_1f1b":
        # Interleaved 1F1B expects a list of stages (multiple per rank)
        if not isinstance(stage, (list, tuple)):
            raise ValueError(
                "interleaved_1f1b schedule requires a list of PipelineStage objects "
                "(multiple stages per rank for virtual pipeline stages)"
            )
        return ScheduleInterleaved1F1B(
            stages=list(stage),
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
        )

    schedules = {
        "1f1b": Schedule1F1B,
        "gpipe": ScheduleGPipe,
    }

    if schedule not in schedules:
        raise ValueError(
            f"Unknown PP schedule: {schedule!r}. "
            f"Choose from {list(schedules) + ['interleaved_1f1b']}"
        )

    return schedules[schedule](
        stage=stage,
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )
