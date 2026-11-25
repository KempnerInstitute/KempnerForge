"""The training step loop and the three step bodies it dispatches to.

``run_training_loop`` owns everything that repeats: the step body, NaN
handling, optimizer/scheduler advance, freeze and phase transitions,
metrics, eval, checkpointing, and shutdown. The step body itself is a
plain callable (``StepFn``) chosen once by :func:`select_step_fn`, so a
model family with a different forward contract is a new function rather
than another branch inside the loop.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.job import JobConfig
from kempnerforge.config.vlm import FreezeSpec
from kempnerforge.distributed.parallel import refresh_fsdp_dtypes
from kempnerforge.distributed.utils import (
    any_rank,
    clip_grad_norm_,
    data_parallel_groups,
    dp_all_reduce_,
    dp_mean,
    new_control_group,
)
from kempnerforge.metrics.logger import get_logger
from kempnerforge.metrics.mfu import vision_flops_per_image
from kempnerforge.metrics.tracker import MetricsTracker
from kempnerforge.model.transformer import Transformer
from kempnerforge.model.vlm import inner_transformer
from kempnerforge.profiling.profiler import print_profiler_summary
from kempnerforge.resilience.health import NaNDetector, check_nccl_health, params_are_finite
from kempnerforge.resilience.signal_handler import ShutdownHandler
from kempnerforge.training.data_pipeline import DataPipeline, PhaseState, advance_phases
from kempnerforge.training.eval import run_eval
from kempnerforge.training.freeze import apply_freeze_specs, freeze_meta_at_step
from kempnerforge.training.grad import maybe_no_sync
from kempnerforge.training.hooks import HookRunner, StepContext
from kempnerforge.training.optimizer import evaluation_point, module_learning_rates
from kempnerforge.training.runtime import PipelineBundle, RuntimeContext, pp_group

if TYPE_CHECKING:
    from kempnerforge.model.transformer import Transformer

logger = get_logger(__name__)

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


_WORKER_KILL_TIMEOUT_S = 5.0


class BatchStream:
    """The training dataloader plus its live iterator.

    Wraps the epoch-boundary restart and the "no dataloader configured"
    case so the step bodies don't each re-implement them. The loader is read
    off the pipeline on every access, so swapping ``pipeline.dataloader``
    takes effect on the next step instead of leaving a stale snapshot.
    """

    def __init__(self, pipeline: DataPipeline) -> None:
        self.pipeline = pipeline
        self._iter: Any = None
        self._source: Any = None

    @property
    def dataloader(self) -> Any:
        return self.pipeline.dataloader

    @property
    def has_data(self) -> bool:
        return self.dataloader is not None

    def ensure_started(self) -> None:
        """Materialize the iterator, if it isn't already, for the loader in use.

        ``next_batch`` does this itself; the loop calls it up front so building
        an iterator (which spawns dataloader workers) stays outside the region
        ``MetricsTracker`` times.
        """
        loader = self.dataloader
        if loader is not None and (self._iter is None or self._source is not loader):
            self._iter = iter(loader)
            self._source = loader

    def reset(self) -> None:
        """Drop the iterator so the next step takes a fresh one from the loader.

        A ``StatefulDataLoader`` re-applies its recorded skip, so this picks up
        inside the current epoch rather than restarting it.
        """
        self._iter = None

    def close(self) -> None:
        """Shut the loader's worker processes down explicitly.

        Workers ignore SIGTERM (see ``ignore_shutdown_signals_in_worker``) so a
        preemption signal cannot kill them out from under an in-flight step.
        The same shield means ``Process.terminate()``, which is SIGTERM, cannot
        reap them either: left to the interpreter's exit hook, the process would
        block forever joining daemonic children that refuse to die. Draining
        them through the loader's own sentinel path avoids that.

        Rank-local and idempotent — it issues no collectives, so it is safe on
        the exception path in ``train_loop``'s ``finally``.
        """
        self._iter = None
        self._source = None
        loader = self.pipeline.dataloader
        if loader is None:
            return
        inner = getattr(loader, "_dataloader", loader)
        iterator = getattr(inner, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
            for worker in getattr(iterator, "_workers", ()):
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=_WORKER_KILL_TIMEOUT_S)
        if hasattr(inner, "_iterator"):
            inner._iterator = None

    def next_batch(self) -> dict[str, torch.Tensor]:
        if self.dataloader is None:
            raise RuntimeError("BatchStream has no dataloader; check has_data first")
        self.ensure_started()
        try:
            return next(self._iter)
        except StopIteration:
            pass
        self._iter = iter(self.dataloader)
        self._source = self.dataloader
        try:
            return next(self._iter)
        except StopIteration:
            raise RuntimeError(
                "dataloader yielded no batches for this rank: the dataset is "
                "smaller than batch_size x dp_size, a streaming shard is "
                "exhausted, or a mixture phase weighted this rank's only "
                "source to zero"
            ) from None


@dataclass
class StepResult:
    """What one training step reports back to the loop.

    The per-dataset fields stay None for step bodies that don't produce
    them; ``text_tokens`` stays None, which a VLM run treats as an error
    rather than logging a zero. With a data mixture, on a step the tracker
    will log, ``dataset_tokens`` holds the positions this rank drew from each
    source, by source index, and ``dataset_losses`` each source's summed
    cross-entropy (row 0) and labeled-token count (row 1). ``tokens`` is the
    number of positions the model ran on this rank over every micro-step,
    image positions and padding included, and ``seq_len`` the length of each
    of those rows; with ``tokens`` None the loop counts
    ``train.batch_size * train.seq_len`` per micro-step. A VLM step also
    reports the ``documents`` it trained on, the ``image_tokens`` of their
    real images, their ``document_text_tokens`` (prompt included, padding
    excluded; None when the batch does not say) and the ``encoder_images``
    the vision encoder ran, padding frames included.
    """

    loss: float
    grad_norm: float
    text_tokens: int | None = None
    dataset_tokens: torch.Tensor | None = None
    dataset_losses: torch.Tensor | None = None
    tokens: int | None = None
    seq_len: int | None = None
    documents: int | None = None
    image_tokens: int | None = None
    document_text_tokens: int | None = None
    encoder_images: int | None = None


StepFn = Callable[["TrainingSession", int], StepResult]


@dataclass
class TrainingSession:
    """Everything one training run needs after the build phase.

    ``documents_seen`` counts the documents a VLM run trained on over the data-parallel
    ranks, from the checkpoint it resumed; it stays None while the step body reports none.
    """

    config: JobConfig
    runtime: RuntimeContext
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    loss_fn: LossFn
    step_fn: StepFn
    data: DataPipeline
    phases: PhaseState
    checkpointer: CheckpointManager
    tracker: MetricsTracker
    hooks: HookRunner
    nan_detector: NaNDetector
    shutdown_handler: ShutdownHandler
    pipeline: PipelineBundle | None = None
    eval_dataloader: Any | None = None
    profiler: Any | None = None
    autocast_dtype: torch.dtype | None = None
    control_group: Any | None = field(default=None, repr=False)
    documents_seen: int | None = None
    _batch_stream: BatchStream | None = field(default=None, init=False, repr=False)
    _dp_groups: tuple | None = field(default=None, init=False, repr=False)

    @property
    def batches(self) -> BatchStream:
        """The live batch iterator. Rebuilt if ``data`` is replaced wholesale."""
        if self._batch_stream is None or self._batch_stream.pipeline is not self.data:
            self._batch_stream = BatchStream(self.data)
        return self._batch_stream

    @property
    def dp_groups(self) -> tuple:
        """``data_parallel_groups`` of the runtime's mesh, taken from it once.

        The logged losses and token counts are reduced over these groups, so every
        rank logs and hands its hooks the same values for the whole job.
        """
        if self._dp_groups is None:
            self._dp_groups = data_parallel_groups(self.runtime.device_mesh)
        return self._dp_groups

    def autocast(self) -> contextlib.AbstractContextManager:
        """``torch.autocast`` to ``autocast_dtype``, or a no-op when it is None."""
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.runtime.device.type, dtype=self.autocast_dtype)


def clip_grads(model: torch.nn.Module, max_norm: float, pp_group: Any = None) -> float:
    grad_norm = clip_grad_norm_(model, max_norm, pp_group=pp_group)
    return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm


def add_moe_losses(loss: torch.Tensor, model: torch.nn.Module, mc: Any) -> torch.Tensor:
    """MoE auxiliary + router z-loss terms (only called when ``mc.is_moe``).

    The routers' losses are then detached (see ``Transformer.detach_router_losses``).
    """
    aux_loss = inner_transformer(model).get_moe_aux_loss()  # type: ignore[attr-defined]
    loss = loss + mc.moe_aux_loss_weight * aux_loss
    if mc.moe_router_z_loss_weight > 0:
        z = inner_transformer(model).get_moe_router_z_loss()  # type: ignore[attr-defined]
        loss = loss + mc.moe_router_z_loss_weight * z
    transformer = inner_transformer(model)
    if isinstance(transformer, Transformer):
        transformer.detach_router_losses()
    return loss


def _expert_balance(routers: list[tuple[str | None, torch.Tensor]]) -> dict[str, float]:
    """``moe/expert_balance`` over every router, plus one value per MoT modality.

    Each router's expert counts are divided by that router's own total before
    taking min over max, so routers that see different token counts (a MoT
    model's image and text routers) compare as loads. Every flat MoE router sees
    the same tokens, so for a text model this equals min over max of the raw
    counts. A router whose counts sum to zero is skipped.

    Args:
        routers: ``(modality, counts)`` per router; ``modality`` is ``None`` for
            a flat MoE layer.

    Returns:
        ``moe/expert_balance``, and ``moe/expert_balance/<modality>`` for each
        MoT modality; empty when no router saw a token.
    """
    loads: list[tuple[str | None, torch.Tensor]] = []
    for modality, counts in routers:
        total = counts.float().sum()
        if total.item() > 0:
            loads.append((modality, counts.float() / total))
    if not loads:
        return {}
    groups = {"moe/expert_balance": [load for _, load in loads]}
    for modality in sorted({m for m, _ in loads if m is not None}):
        groups[f"moe/expert_balance/{modality}"] = [load for m, load in loads if m == modality]
    balance = {}
    for key, group in groups.items():
        values = torch.cat(group)
        balance[key] = (values.min() / values.max()).item()
    return balance


def pipeline_step(session: TrainingSession, step: int) -> StepResult:
    """PP step: collect microbatches, hand them to the schedule, broadcast loss.

    ``step`` is unused — MoE (the only step-dependent branch) is rejected
    with PP in ``JobConfig.validate``.
    """
    config = session.config
    tc, mc = config.train, config.model
    device = session.runtime.device
    model = session.model
    batches = session.batches
    pipeline = session.pipeline
    if pipeline is None:
        raise RuntimeError("pipeline_step requires TrainingSession.pipeline")

    input_ids_list, labels_list, doc_ids_list = [], [], []
    for _ in range(tc.grad_accum_steps):
        if batches.has_data:
            batch = batches.next_batch()
            input_ids_list.append(batch["input_ids"].to(device))
            labels_list.append(batch["labels"].to(device))
            if "doc_ids" in batch:
                doc_ids_list.append(batch["doc_ids"].to(device))
        else:
            input_ids_list.append(
                torch.randint(0, mc.vocab_size, (tc.batch_size, tc.seq_len), device=device)
            )
            labels_list.append(
                torch.randint(0, mc.vocab_size, (tc.batch_size, tc.seq_len), device=device)
            )

    full_input = torch.cat(input_ids_list, dim=0)
    full_labels = torch.cat(labels_list, dim=0)
    pp_kwargs = {"doc_ids": torch.cat(doc_ids_list, dim=0)} if doc_ids_list else {}

    is_first = pipeline.rank == 0
    is_last = pipeline.rank == pipeline.size - 1
    pp_losses: list[torch.Tensor] = []

    with session.autocast():
        if is_first:
            pipeline.schedule.step(full_input, target=full_labels, losses=pp_losses, **pp_kwargs)
        elif is_last:
            pipeline.schedule.step(target=full_labels, losses=pp_losses, **pp_kwargs)
        else:
            pipeline.schedule.step(**pp_kwargs)

    if is_last and pp_losses:
        avg_loss = sum(loss.item() for loss in pp_losses) / len(pp_losses)
    else:
        avg_loss = 0.0

    group = pp_group(session.runtime)
    grad_norm_val = clip_grads(model, tc.grad_clip_norm, group)

    loss_tensor = torch.tensor([avg_loss], device=device)
    dist.broadcast(loss_tensor, group_src=pipeline.size - 1, group=group)
    return StepResult(
        loss=loss_tensor.item(),
        grad_norm=grad_norm_val,
        tokens=full_input.numel(),
        seq_len=full_input.shape[1],
    )


_PACKED_BATCH_KEYS = (
    "doc_ids",
    "position_ids",
    "attention_group_ids",
    "image_rows",
    "image_starts",
    "text_rows",
    "text_positions",
)

_SOURCE_CE_CHUNK = 1024
"""Tokens the per-source cross-entropy runs at once, about 1 GB of fp32 at a 128k vocabulary."""


def vlm_step(session: TrainingSession, step: int) -> StepResult:
    """VLM step (no PP): pixel_values + input_ids forward, with a text-token count.

    Each row the decoder runs holds the model's ``num_image_tokens`` image
    positions (zero when they do not enter the residual stream) and the text.
    A packed batch (one with ``image_rows``) also carries ``doc_ids``,
    ``position_ids``, the image placement (``image_starts``, ``image_tokens``)
    and the text positions, all passed through to the wrapper, with
    ``attention_group_ids`` when a collator emits them. Its rows hold their
    image positions already, and its row count varies, so the leading dims are
    marked dynamic for ``torch.compile``.
    """
    config = session.config
    tc, mc = config.train, config.model
    device = session.runtime.device
    model = session.model
    batches = session.batches

    total_loss = 0.0
    total_text_tokens = 0
    total_tokens = 0
    seq_len = 0
    inner = getattr(model, "_orig_mod", model)
    image_positions = getattr(inner, "num_image_tokens", 0)
    adapter, encoder = getattr(inner, "adapter", None), getattr(inner, "vision_encoder", None)
    tokens_per_image = (
        adapter.output_num_tokens(encoder.num_tokens)
        if adapter is not None and encoder is not None
        else None
    )
    documents = image_tokens = encoder_images = 0
    document_text_tokens: int | None = 0

    for micro_step in range(tc.grad_accum_steps):
        if not batches.has_data:
            raise RuntimeError(
                "VLM training requires a real dataloader; synthetic fallback "
                "(randint) does not produce pixel_values"
            )
        batch = batches.next_batch()
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        frame_mask = batch["frame_mask"].to(device) if "frame_mask" in batch else None
        extras: dict[str, Any] = {
            key: batch[key].to(device) for key in _PACKED_BATCH_KEYS if key in batch
        }
        if "image_tokens" in batch:
            extras["image_tokens"] = int(batch["image_tokens"])
        if "image_rows" in extras:
            seq_len = input_ids.shape[1]
            if tc.compile_model:
                _mark_rows_dynamic(pixel_values, input_ids, labels, *extras.values())
        else:
            seq_len = input_ids.shape[1] + image_positions
        total_tokens += input_ids.shape[0] * seq_len
        counts = _batch_counts(batch, image_positions, tokens_per_image)
        documents += counts[0]
        image_tokens += counts[1]
        encoder_images += counts[3]
        if document_text_tokens is not None:
            document_text_tokens = None if counts[2] is None else document_text_tokens + counts[2]

        with maybe_no_sync(model, micro_step, tc.grad_accum_steps):
            if mc.is_moe:
                inner_transformer(model).set_moe_step(step, tc.max_steps)  # type: ignore[attr-defined]
            with session.autocast():
                logits, labels_out = model(
                    pixel_values, input_ids, labels, frame_mask=frame_mask, **extras
                )
                loss = session.loss_fn(logits, labels_out)

            total_text_tokens += int((labels_out != -100).sum().item())

            if mc.is_moe:
                loss = add_moe_losses(loss, model, mc)

            scaled_loss = loss / tc.grad_accum_steps
            scaled_loss.backward()
            total_loss += loss.item()

    return StepResult(
        loss=total_loss / tc.grad_accum_steps,
        grad_norm=clip_grads(model, tc.grad_clip_norm),
        tokens=total_tokens,
        seq_len=seq_len,
        text_tokens=total_text_tokens,
        documents=documents,
        image_tokens=image_tokens,
        document_text_tokens=document_text_tokens,
        encoder_images=encoder_images,
    )


def _source_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, sources: torch.Tensor, n_sources: int, chunk: int
) -> torch.Tensor:
    """Each source's summed cross-entropy and labeled-token count, as a ``(2, n_sources)`` tensor.

    ``sources`` holds the source index of each row. The cross-entropy runs in fp32,
    ``chunk`` tokens at a time, so it copies no source's logits and holds one chunk's
    fp32 logits and log-softmax at most. A token labeled -100 counts in neither row.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    token_sources = sources.repeat_interleave(labels.shape[-1])
    out = torch.zeros(2, n_sources, dtype=torch.float64, device=logits.device)
    for start in range(0, flat_labels.numel(), chunk):
        chunk_labels = flat_labels[start : start + chunk]
        losses = F.cross_entropy(
            flat_logits[start : start + chunk].float(),
            chunk_labels,
            ignore_index=-100,
            reduction="none",
        )
        chunk_sources = token_sources[start : start + chunk]
        out[0].index_add_(0, chunk_sources, losses.double())
        out[1].index_add_(0, chunk_sources, (chunk_labels != -100).double())
    return out


def _mark_rows_dynamic(*tensors: Any) -> None:
    """Mark each tensor's leading dim dynamic so a new packed row count does not recompile."""
    from torch._dynamo import maybe_mark_dynamic

    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            maybe_mark_dynamic(tensor, 0)


def _route_balance(session: TrainingSession, *, step_ran: bool) -> None:
    """Balance the sigmoid routers after an optimizer step, or drop the counts of one skipped.

    Every rank calls this at the same point, since balancing all-reduces the routers'
    expert counts (see ``Transformer.balance_routers``).
    """
    transformer = inner_transformer(session.model)
    if not session.config.model.is_moe or not isinstance(transformer, Transformer):
        return
    if step_ran:
        transformer.balance_routers()
    else:
        transformer.discard_router_counts()


def _batch_counts(
    batch: dict[str, Any], image_positions: int, tokens_per_image: int | None = None
) -> tuple[int, int, int | None, int]:
    """A VLM batch's documents, real-image tokens, document text tokens and encoder images.

    A packed batch (one with ``image_rows``) holds several documents per row: they are the
    distinct row and document pairs at its text positions and image starts, its image tokens
    are its images times ``image_tokens``, and its text tokens its ``text_positions``; with no
    image, the encoder runs one blank image. A padded batch holds one document per row, and its
    image tokens are ``tokens_per_image``, the adapter's output tokens, for each real image or
    frame ``frame_mask`` marks (``image_positions`` split evenly over the frames when
    ``tokens_per_image`` is None); its text tokens are what ``text_len`` gives. The text count
    is None when the batch does not give it. The encoder runs every image or frame, blank or
    padding ones included.
    """
    pixel_values = batch["pixel_values"]
    if "image_rows" in batch:
        empty = torch.zeros(0, dtype=torch.long)
        text_positions = batch.get("text_positions")
        rows = torch.cat([batch.get("text_rows", empty), batch["image_rows"]])
        starts = torch.cat(
            [empty if text_positions is None else text_positions, batch["image_starts"]]
        )
        doc_ids = batch["doc_ids"]
        pairs = rows * (int(doc_ids.max()) + 1) + doc_ids[rows, starts]
        n_images = int(pixel_values.shape[0])
        return (
            int(torch.unique(pairs).numel()),
            n_images * int(batch["image_tokens"]),
            None if text_positions is None else int(text_positions.numel()),
            max(n_images, 1),
        )
    rows = int(batch["input_ids"].shape[0])
    frames = int(pixel_values.shape[1]) if pixel_values.dim() == 5 else 1
    frame_mask = batch.get("frame_mask")
    real = int(frame_mask.sum()) if frame_mask is not None else rows * frames
    per_image = image_positions // frames if tokens_per_image is None else tokens_per_image
    text_len = batch.get("text_len")
    text = int(text_len.sum()) if text_len is not None else None
    return rows, real * per_image, text, rows * frames


def _dp_sum(session: TrainingSession, counts: list[int]) -> list[int]:
    """``counts`` summed over the data-parallel ranks, in one reduction.

    Without data-parallel groups, which a run of one data-parallel rank has, each count is
    taken ``data.dp_size`` times, as if every rank had run the same batch.
    """
    if not session.dp_groups:
        return [count * session.data.dp_size for count in counts]
    tensor = torch.tensor(counts, device=session.runtime.device, dtype=torch.long)
    return dp_all_reduce_(tensor, session.dp_groups).tolist()


def _step_counts(
    session: TrainingSession, result: StepResult
) -> tuple[int, int | None, int | None]:
    """Positions, documents and encoder images the step ran over the data-parallel ranks.

    Every rank adds its counts in one reduction, whatever its step body reports, so ranks that
    ran different batches count what each of them ran and every rank takes part. A rank that
    reports no ``tokens`` adds the configured batch, and one that reports no documents or
    encoder images adds zero; those come back None, on every rank, when no rank reported them.
    """
    tc = session.config.train
    tokens = result.tokens
    if tokens is None:
        tokens = tc.batch_size * tc.seq_len * tc.grad_accum_steps
    counts = [
        tokens,
        result.documents or 0,
        result.encoder_images or 0,
        int(result.documents is not None),
        int(result.encoder_images is not None),
    ]
    total, documents, images, with_documents, with_images = _dp_sum(session, counts)
    return total, (documents if with_documents else None), (images if with_images else None)


def text_step(session: TrainingSession, step: int) -> StepResult:
    """Standard step (no PP, text-only), with optional per-dataset mixture metrics.

    Each source's tokens and loss are counted only on a step the tracker will log,
    which is ``step + 1``: ``step`` counts the steps done before this one. The loss is
    the plain cross-entropy of the source's labeled tokens, without the z-loss or MoE
    terms the training loss may add.
    """
    config = session.config
    tc, mc = config.train, config.model
    device = session.runtime.device
    model = session.model
    batches = session.batches
    mixture_dataset = session.data.mixture_dataset
    n_sources = len(mixture_dataset.dataset_names) if mixture_dataset is not None else 0

    total_loss = 0.0
    total_tokens = 0
    seq_len = 0
    ds_tokens = ds_losses = None
    if n_sources and session.tracker.is_log_step(step + 1):
        ds_tokens = torch.zeros(n_sources, dtype=torch.long)
        ds_losses = torch.zeros(2, n_sources, dtype=torch.float64, device=device)
    ce_chunk = min(tc.ce_chunk_size, _SOURCE_CE_CHUNK) if tc.ce_chunk_size > 0 else _SOURCE_CE_CHUNK

    for micro_step in range(tc.grad_accum_steps):
        batch: dict[str, torch.Tensor] = {}
        if batches.has_data:
            batch = batches.next_batch()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            doc_ids = batch["doc_ids"].to(device) if "doc_ids" in batch else None
        else:
            input_ids = torch.randint(0, mc.vocab_size, (tc.batch_size, tc.seq_len), device=device)
            labels = torch.randint(0, mc.vocab_size, (tc.batch_size, tc.seq_len), device=device)
            doc_ids = None
        seq_len = input_ids.shape[1]
        total_tokens += input_ids.numel()

        with maybe_no_sync(model, micro_step, tc.grad_accum_steps):
            if mc.is_moe:
                inner_transformer(model).set_moe_step(step, tc.max_steps)  # type: ignore[attr-defined]
            with session.autocast():
                logits = model(input_ids, doc_ids=doc_ids)
                loss = session.loss_fn(logits, labels)

            if ds_tokens is not None and ds_losses is not None and "dataset_idx" in batch:
                sources = batch["dataset_idx"]
                ds_tokens += torch.bincount(sources, minlength=n_sources) * seq_len
                with torch.no_grad():
                    ds_losses += _source_cross_entropy(
                        logits, labels, sources.to(device), n_sources, ce_chunk
                    )

            if mc.is_moe:
                loss = add_moe_losses(loss, model, mc)

            scaled_loss = loss / tc.grad_accum_steps
            scaled_loss.backward()
            total_loss += loss.item()

    return StepResult(
        loss=total_loss / tc.grad_accum_steps,
        grad_norm=clip_grads(model, tc.grad_clip_norm),
        tokens=total_tokens,
        seq_len=seq_len,
        dataset_tokens=ds_tokens,
        dataset_losses=ds_losses,
    )


def select_step_fn(config: JobConfig) -> StepFn:
    """Pick the step body for this job: PP wins, then VLM, then text-only."""
    if config.distributed.pp > 1:
        return pipeline_step
    if config.is_vlm:
        return vlm_step
    return text_step


def checkpoint_extra(
    config: JobConfig,
    step: int,
    phases: PhaseState,
    documents_seen: int | None = None,
    consecutive_nans: int = 0,
) -> dict:
    """Metadata saved alongside every checkpoint so a resume is exact.

    ``vlm_freeze`` reflects the post-transition state when a FreezeStage
    has fired (see ``freeze_meta_at_step``). ``plugins`` is the run's plugins
    list, which a resume compares with its own (see ``restore_checkpoint``).
    ``documents_seen``, when given, is saved for the resumed run to count on from.
    ``consecutive_nans``, the NaN detector's count of NaN steps in a row, is saved
    when it is not zero, for the resumed run to count on from, so it stops at the
    step the uninterrupted run stops at; a checkpoint without it resumes at zero.
    """
    extra: dict = {"phase_idx": phases.next_idx} if phases.phases else {}
    extra["plugins"] = list(config.plugins)
    if documents_seen is not None:
        extra["documents_seen"] = documents_seen
    if consecutive_nans:
        extra["consecutive_nans"] = consecutive_nans
    if config.metrics.wandb_run_id:
        extra["wandb_run_id"] = config.metrics.wandb_run_id
    if config.metrics.mlflow_run_id:
        extra["mlflow_run_id"] = config.metrics.mlflow_run_id
    if config.is_vlm:
        assert config.vlm is not None
        extra["vlm_freeze"] = freeze_meta_at_step(step, config.vlm)
    return extra


@contextlib.contextmanager
def _logged_time(tracker: MetricsTracker, key: str, step: int) -> Iterator[None]:
    """Log the time the block takes as the metric ``key`` of ``step``.

    Nothing is logged when the block raises.
    """
    start = time.perf_counter()
    yield
    tracker.log_eval({key: time.perf_counter() - start}, step)


def _log_periodic_metrics(session: TrainingSession, result: StepResult, step: int) -> None:
    """MoE / VLM-text-token / per-dataset series, on the main metrics interval."""
    config = session.config
    mc = config.model
    tracker = session.tracker
    model = session.model

    if mc.is_moe:
        inner = cast("Transformer", inner_transformer(model))
        moe_metrics = {"moe/aux_loss": inner.get_moe_aux_loss().item()}
        moe_metrics["moe/router_z_loss"] = inner.get_moe_router_z_loss().item()
        routers: list[tuple[str | None, torch.Tensor]] = [
            (None, counts) for counts in inner.get_expert_counts().values()
        ]
        for per_modality in getattr(inner, "get_mot_expert_counts", dict)().values():
            routers.extend(per_modality.items())
        moe_metrics.update(_expert_balance(routers))
        tracker.log_eval(moe_metrics, step)

    if config.is_vlm:
        if any_rank(result.text_tokens is None, session.control_group):
            raise RuntimeError(
                "VLM run: a step body reported no text-token count, so "
                "data/text_tokens_trained would be logged as zero"
            )
        assert result.text_tokens is not None
        local = [
            result.text_tokens,
            result.image_tokens or 0,
            result.document_text_tokens or 0,
            int(result.image_tokens is not None),
            int(result.document_text_tokens is not None),
        ]
        trained, image_tokens, text_tokens, with_images, with_text = _dp_sum(session, local)
        data_metrics = {"data/text_tokens_trained": float(trained)}
        if with_images:
            data_metrics["data/image_tokens"] = float(image_tokens)
        if with_text:
            data_metrics["data/text_tokens"] = float(text_tokens)
        if session.documents_seen is not None:
            data_metrics["data/documents_seen"] = float(session.documents_seen)
        tracker.log_eval(data_metrics, step)

    mixture = session.data.mixture_dataset
    if mixture is not None:
        _log_dataset_metrics(session, result, mixture.dataset_names, step)


def _log_dataset_metrics(
    session: TrainingSession, result: StepResult, names: list[str], step: int
) -> None:
    """``loss/<name>`` and ``data/<name>/tokens`` for each source, over every DP rank.

    The loss is the cross-entropy of the source's labeled tokens, weighted by token, and
    the tokens are the positions drawn from the source in this step. A source no rank
    drew from in this step is left out. Every rank takes part in the reduction, adding
    nothing when its step body reports no per-source counts.
    """
    stats = torch.zeros(3, len(names), dtype=torch.float64, device=session.runtime.device)
    if result.dataset_tokens is not None:
        stats[0] = result.dataset_tokens.to(stats.device)
    if result.dataset_losses is not None:
        stats[1:] = result.dataset_losses
    totals = dp_all_reduce_(stats, session.dp_groups).tolist()
    ds_metrics: dict[str, float] = {}
    for name, tokens, loss_sum, labeled in zip(names, *totals, strict=True):
        if labeled > 0:
            ds_metrics[f"loss/{name}"] = loss_sum / labeled
        if tokens > 0:
            ds_metrics[f"data/{name}/tokens"] = tokens
    if ds_metrics:
        session.tracker.log_eval(ds_metrics, step)


def _advance_lr_schedule(session: TrainingSession) -> None:
    """Step the scheduler, then scale the new LR by the active phase's ``lr_scale``.

    Runs once per step, a step skipped for a non-finite loss included, so
    the schedule stays aligned with the step counter.
    """
    session.scheduler.step()
    if session.phases.lr_scale != 1.0:
        for pg in session.optimizer.param_groups:
            pg["lr"] *= session.phases.lr_scale


def _apply_stage_specs(
    model: torch.nn.Module, specs: Iterable[FreezeSpec], pattern_map: dict[str, list[str]]
) -> dict[str, int]:
    """Apply one freeze stage's specs to the model in training.

    Parameters the stage unfreezes have their FSDP2 reduction dtypes refreshed. When the
    stage unfreezes parameters of a compiled model, ``torch.compiler.reset`` runs, which
    clears every compiled function in the process.

    Args:
        model: The model in training, compiled or not.
        specs: The stage's freeze specs.
        pattern_map: Module aliases mapped to fnmatch patterns.

    Returns:
        The counts ``apply_freeze_specs`` returns.
    """
    frozen = {id(p) for p in model.parameters() if not p.requires_grad}
    flipped = apply_freeze_specs(model, specs, pattern_map)
    unfrozen = [p for p in model.parameters() if p.requires_grad and id(p) in frozen]
    refresh_fsdp_dtypes(model, unfrozen)
    if unfrozen and hasattr(model, "_orig_mod"):
        torch.compiler.reset()
        logger.info("A freeze stage unfroze parameters of a compiled model: it recompiles")
    return flipped


def _apply_freeze_stages(session: TrainingSession, step: int) -> None:
    """FreezeStage hook: apply any stage whose start_step matches this boundary.

    Each stage goes through ``_apply_stage_specs``, and the model is then put in
    training mode again, which puts a vision encoder the stages froze in eval mode and
    one they unfroze back in training mode (see ``VLMWrapper.train``).

    Async-save fence: drain any in-flight save FIRST so that its
    metadata.json — which records the pre-transition spec — lands before we
    flip requires_grad. Otherwise a save started at step S-1 could write
    metadata after the transition, attaching the post-transition spec to
    pre-transition shards.
    """
    vlm_cfg = session.config.vlm
    if not session.config.is_vlm or vlm_cfg is None or not vlm_cfg.freeze_schedule:
        return
    pending_stages = [s for s in vlm_cfg.freeze_schedule if s.start_step == step]
    if not pending_stages:
        return
    ckpt_mgr = session.checkpointer
    if ckpt_mgr.has_pending_save:
        with _logged_time(session.tracker, "checkpoint/wait_sec", step):
            ckpt_mgr.flush_pending_save()
    else:
        ckpt_mgr.flush_pending_save()
    for stage in pending_stages:
        flipped = _apply_stage_specs(session.model, stage.specs, vlm_cfg.module_patterns)
        logger.info(f"FreezeStage at step={step}: applied {flipped}")
    session.model.train()


def _apply_step_transitions(session: TrainingSession, step: int) -> None:
    """Apply the freeze stages and data phases due at ``step``.

    Runs right after the step counter advances, on a step skipped for a
    non-finite loss too, so a transition lands on its configured step. A
    phase that fires drops the data iterator, so the next batch is drawn
    with the new sampler weights.
    """
    _apply_freeze_stages(session, step)
    if advance_phases(session.phases, session.data, step):
        session.batches.reset()


def _shutdown_if_requested(
    session: TrainingSession,
    step: int,
    tokens_seen: int,
    *,
    saved_this_step: bool,
    nan_skipped: bool = False,
) -> bool:
    """Write the emergency checkpoint when a shutdown is pending on any rank.

    Every rank must call this at the same step. It first reinstalls the
    signal handlers, which native code can replace below Python (Triton's
    LLVM does on its first in-process compile, after which SIGUSR1 does
    nothing). The ranks then agree over ``session.control_group`` whether any
    of them was signaled, so they all stop or all go on, whichever ranks the
    signal reached and whenever. A rank that learns of the shutdown only from
    the vote calls ``request()``, so every rank that stops has its forced-exit
    timer armed. The handler is left armed: the timer still bounds the flush
    that ``run_training_loop`` ends with and the caller's teardown, and the
    caller calls ``finish()`` after both.

    Returns True when the loop must stop. With ``saved_this_step``, the
    scheduled save of ``step`` is the emergency checkpoint, so it is drained
    with ``wait()`` rather than written twice. A ``nan_skipped`` step is
    saved only if every rank's parameters are finite: after a bad update
    (an optimizer step that left non-finite weights although its loss and
    gradient norm were finite) they are not, and their checkpoint would
    become ``latest`` and count toward ``keep_last_n``, which can prune the
    last good one.
    """
    session.shutdown_handler.reinstall()
    requested_here = session.shutdown_handler.should_shutdown()
    if not any_rank(requested_here, session.control_group):
        return False
    if not requested_here:
        session.shutdown_handler.request()
        logger.warning(f"Shutdown requested on another rank at step {step}")
    ckpt_mgr = session.checkpointer
    if saved_this_step:
        logger.warning(
            f"Shutdown requested at step {step}; "
            "its scheduled checkpoint is the emergency checkpoint"
        )
        if ckpt_mgr.has_pending_save:
            with _logged_time(session.tracker, "checkpoint/wait_sec", step):
                ckpt_mgr.wait()
        else:
            ckpt_mgr.wait()
    elif nan_skipped and not params_are_finite(session.model, session.runtime.device):
        logger.error(
            f"Shutdown requested at step {step}, skipped for a NaN/Inf loss, but the "
            "parameters are not finite: saving no emergency checkpoint, so these weights "
            "neither become `latest` nor count toward `checkpoint.keep_last_n`"
        )
    else:
        logger.warning(f"Shutdown requested at step {step}; saving emergency checkpoint")
        with _logged_time(session.tracker, "checkpoint/save_sec", step):
            ckpt_mgr.save(
                step=step,
                tokens_seen=tokens_seen,
                scheduler=session.scheduler,
                dataloader=session.data.dataloader,
                extra=checkpoint_extra(
                    session.config,
                    step,
                    session.phases,
                    session.documents_seen,
                    consecutive_nans=session.nan_detector.state.consecutive_nans,
                ),
            )
    return True


def run_training_loop(
    session: TrainingSession, *, step: int = 0, tokens_seen: int = 0
) -> tuple[int, int]:
    """Run the step loop from ``step`` to ``train.max_steps``.

    Returns the final ``(step, tokens_seen)``. Drains any pending async
    checkpoint before returning so the caller can tear down the process
    group safely. ``session.data`` is read once and must stay fixed for the
    duration; only its ``dataloader`` may be swapped mid-run.

    Every rank must call it: the shutdown and NCCL-health stops are agreed
    over ``session.control_group``. When that is ``None`` the loop makes the
    group, and destroys it when it returns or raises. It never calls
    ``shutdown_handler.finish()``; the handler's owner does, once teardown is
    done. After an NCCL-health stop it saves nothing, drains any pending async
    save and raises ``RuntimeError`` on every rank, so the job fails rather
    than ending as if it completed. The NaN stop, once
    ``nan_detector.rollback_recommended``, saves nothing either and returns:
    nothing rolls back, and the rollback to an earlier checkpoint is manual.
    """
    config = session.config
    tc = config.train
    eval_config = config.eval
    runtime = session.runtime
    model = session.model
    optimizer = session.optimizer
    scheduler = session.scheduler
    ckpt_mgr = session.checkpointer
    tracker = session.tracker
    hooks = session.hooks
    data = session.data
    phases = session.phases
    prof = session.profiler

    logger.info(
        f"Starting training: step={step}, max_steps={tc.max_steps}, "
        f"batch_size={tc.batch_size}, grad_accum={tc.grad_accum_steps}, "
        f"world_size={runtime.world_size}"
    )
    if phases.phases:
        logger.info(f"Phase scheduling: {len(phases.phases)} phase(s) configured")

    model.train()
    hooks.on_train_begin(config)

    if prof is not None:
        prof.start()

    if step == 0 and config.checkpoint.is_dynamic_milestone(0):
        with _logged_time(tracker, "checkpoint/save_sec", 0):
            ckpt_mgr.save(
                step=0,
                tokens_seen=0,
                scheduler=scheduler,
                dataloader=data.dataloader,
                extra=checkpoint_extra(
                    config,
                    0,
                    phases,
                    session.documents_seen,
                    consecutive_nans=session.nan_detector.state.consecutive_nans,
                ),
            )
        hooks.on_checkpoint_save(0, config.checkpoint.dir)

    completed_normally = False
    ckpt_extra: dict = {}
    nccl_failed_at: int | None = None
    owns_control_group = session.control_group is None
    if owns_control_group:
        session.control_group = new_control_group(config.distributed.nccl_timeout_sec)

    try:
        while step < tc.max_steps:
            session.batches.ensure_started()

            tracker.start_step()
            result = session.step_fn(session, step)
            tokens_in_step, documents_in_step, images_in_step = _step_counts(session, result)
            vision_flops = (
                images_in_step * vision_flops_per_image(model)
                if images_in_step is not None
                else None
            )

            if not session.nan_detector.check_loss(result.loss, step, grad_norm=result.grad_norm):
                _route_balance(session, step_ran=False)
                optimizer.zero_grad()
                if session.nan_detector.rollback_recommended:
                    logger.error(
                        f"Too many consecutive NaNs at step {step}: stopping without a "
                        "checkpoint. Nothing rolls back on its own; to roll back, resubmit "
                        "with a new, empty checkpoint.dir and checkpoint.load_path set to "
                        "an earlier step_N"
                    )
                    break
                _advance_lr_schedule(session)
                step += 1
                _apply_step_transitions(session, step)
                if _shutdown_if_requested(
                    session, step, tokens_seen, saved_this_step=False, nan_skipped=True
                ):
                    break
                continue

            applied_lr = optimizer.param_groups[0]["lr"]
            applied_module_lrs = module_learning_rates(optimizer)
            optimizer.step()
            _route_balance(session, step_ran=True)
            _advance_lr_schedule(session)

            optimizer.zero_grad()

            step += 1
            tokens_seen += tokens_in_step
            if documents_in_step is not None:
                session.documents_seen = (session.documents_seen or 0) + documents_in_step

            _apply_step_transitions(session, step)

            loss = dp_mean(result.loss, session.dp_groups, runtime.device)
            step_metrics = tracker.end_step(
                step=step,
                loss=loss,
                grad_norm=result.grad_norm,
                lr=applied_lr,
                tokens_in_step=tokens_in_step,
                seq_len=result.seq_len,
                vision_flops=vision_flops,
            )

            hooks.on_step_end(
                StepContext(
                    step=step,
                    loss=loss,
                    grad_norm=result.grad_norm,
                    lr=applied_lr,
                    tokens_seen=tokens_seen,
                    model=model,
                    optimizer=optimizer,
                )
            )

            if step_metrics is not None:
                if applied_module_lrs:
                    tracker.log_eval(applied_module_lrs, step)
                _log_periodic_metrics(session, result, step)

            if (
                tc.nccl_health_check_interval > 0
                and step % tc.nccl_health_check_interval == 0
                and any_rank(not check_nccl_health(), session.control_group)
            ):
                logger.error(f"NCCL health check failed at step {step}; stopping")
                nccl_failed_at = step
                break

            if (
                eval_config.enabled
                and session.eval_dataloader is not None
                and step % eval_config.interval == 0
            ):
                pipeline = session.pipeline
                with session.autocast(), evaluation_point(optimizer, model):
                    eval_metrics = run_eval(
                        model,
                        session.eval_dataloader,
                        session.loss_fn,
                        runtime.device,
                        eval_config.steps,
                        pp_schedule=pipeline.schedule if pipeline is not None else None,
                        pp_rank=pipeline.rank if pipeline is not None else None,
                        pp_size=pipeline.size if pipeline is not None else None,
                        pp_group=pp_group(runtime) if pipeline is not None else None,
                        pp_n_microbatches=tc.grad_accum_steps if pipeline is not None else None,
                        dp_groups=session.dp_groups,
                    )
                tracker.log_eval(eval_metrics, step)
                hooks.on_eval_end(eval_metrics, step)

            if prof is not None:
                prof.step()

            ckpt_extra = checkpoint_extra(
                config,
                step,
                phases,
                session.documents_seen,
                consecutive_nans=session.nan_detector.state.consecutive_nans,
            )
            saved_this_step = config.checkpoint.should_save(step)
            if saved_this_step:
                with _logged_time(tracker, "checkpoint/save_sec", step):
                    ckpt_mgr.save(
                        step=step,
                        tokens_seen=tokens_seen,
                        scheduler=scheduler,
                        dataloader=data.dataloader,
                        extra=ckpt_extra,
                    )
                hooks.on_checkpoint_save(step, config.checkpoint.dir)

            if _shutdown_if_requested(session, step, tokens_seen, saved_this_step=saved_this_step):
                break

            if step >= tc.max_steps:
                completed_normally = True

    finally:
        if prof is not None:
            prof.stop()
            if runtime.rank == 0:
                print_profiler_summary(prof, trace_dir=config.profiling.trace_dir)
        session.batches.close()
        if owns_control_group and session.control_group is not None:
            dist.destroy_process_group(session.control_group)
            session.control_group = None

    if completed_normally and not config.checkpoint.should_save(step):
        with _logged_time(tracker, "checkpoint/save_sec", step):
            ckpt_mgr.save(
                step=step,
                tokens_seen=tokens_seen,
                scheduler=scheduler,
                dataloader=data.dataloader,
                extra=ckpt_extra,
            )
        hooks.on_checkpoint_save(step, config.checkpoint.dir)

    if ckpt_mgr.has_pending_save:
        with _logged_time(tracker, "checkpoint/wait_sec", step):
            ckpt_mgr.wait()
    else:
        ckpt_mgr.wait()
    if nccl_failed_at is not None:
        raise RuntimeError(
            f"NCCL health check failed at step {nccl_failed_at} on at least one rank; "
            "every rank stopped without a final save"
        )

    logger.info(f"Training complete: {step} steps, {tokens_seen:,} tokens")
    hooks.on_train_end(step, tokens_seen)
    return step, tokens_seen
