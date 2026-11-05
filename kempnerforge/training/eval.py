"""Evaluation utilities for KempnerForge.

Provides `run_eval` for computing eval loss and perplexity on a held-out
dataset. Works with any parallel model (FSDP, TP, PP) — same model reference,
no unwrapping needed.
"""

from __future__ import annotations

import logging
import math
from datetime import timedelta

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_EVAL_BROADCAST_TIMEOUT_SEC = 300.0


def should_build_eval_dataloader(eval_enabled: bool, is_vlm: bool) -> tuple[bool, bool]:
    """Decide whether to build an eval dataloader and whether to warn.

    The training loop calls ``run_eval(model, eval_dataloader, ...)`` which
    invokes ``model(input_ids)`` — this does not match
    ``VLMWrapper.forward(pixel_values, input_ids, labels)``. VLM configs
    with ``eval.enabled=true`` would crash on the first eval interval.
    This helper gates the eval setup: for VLM configs it suppresses eval
    and flags that a warning should be logged so users see their eval
    setting was ignored. VLM eval support is a tracked follow-up.

    Returns ``(should_build, should_warn_vlm_skip)``.
    """
    if eval_enabled and is_vlm:
        return False, True
    return eval_enabled, False


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    loss_fn: callable,  # type: ignore[reportGeneralTypeIssues]
    device: torch.device,
    eval_steps: int,
    *,
    pp_schedule=None,
    pp_rank: int | None = None,
    pp_size: int | None = None,
    pp_group=None,
    pp_n_microbatches: int | None = None,
) -> dict[str, float]:
    """Run evaluation and return metrics.

    Args:
        model: The model (FSDP-wrapped, TP-sharded, or plain).
        eval_dataloader: DataLoader yielding {"input_ids", "labels"} batches.
        loss_fn: Loss function (logits, labels) -> scalar tensor.
        device: Device to move batches to.
        eval_steps: Number of eval batches to process.
        pp_schedule: Pipeline parallel schedule (None for non-PP). It is the training
            schedule, run forward-only through its ``eval()``. Its gradient reduction still
            runs and divides any gradients left on the stage by the microbatch count, so
            call this with the gradients cleared, as the training loop does.
        pp_rank: This rank's PP stage index.
        pp_size: Total number of PP stages.
        pp_group: Process group for PP loss broadcast.
        pp_n_microbatches: Microbatches per call of ``pp_schedule``. The batches go
            through it this many at a time, so each microbatch has the shape the
            stages were built for; ``eval_steps`` must be a multiple of it. None
            sends all ``eval_steps`` batches in one call.

    Returns:
        Dict with "eval/loss" and "eval/perplexity".

    Raises:
        ValueError: Under PP, ``eval_steps`` is not a multiple of
            ``pp_n_microbatches``, or ``eval_dataloader`` yields no batch.
    """
    model.eval()

    if pp_schedule is not None:
        n_microbatches = pp_n_microbatches or eval_steps
        if eval_steps % n_microbatches:
            raise ValueError(
                f"eval_steps ({eval_steps}) must be a multiple of the pipeline schedule's "
                f"{n_microbatches} microbatches"
            )
        is_first = pp_rank == 0
        is_last = pp_rank == pp_size - 1  # type: ignore[reportOptionalOperand]
        pp_losses: list[torch.Tensor] = []
        batches = _cycle(eval_dataloader)

        for _ in range(eval_steps // n_microbatches):
            group = [next(batches) for _ in range(n_microbatches)]
            full_input = torch.cat([b["input_ids"].to(device) for b in group], dim=0)
            full_labels = torch.cat([b["labels"].to(device) for b in group], dim=0)
            call_losses: list[torch.Tensor] = []
            if is_first:
                pp_schedule.eval(full_input, target=full_labels, losses=call_losses)
            elif is_last:
                pp_schedule.eval(target=full_labels, losses=call_losses)
            else:
                pp_schedule.eval()
            pp_losses.extend(call_losses)

        if is_last and pp_losses:
            avg_loss = sum(loss.item() for loss in pp_losses) / len(pp_losses)
        else:
            avg_loss = 0.0

        loss_tensor = torch.tensor([avg_loss], device=device)
        work = dist.broadcast(
            loss_tensor,
            group_src=pp_size - 1,  # type: ignore[reportOptionalOperand]
            group=pp_group,
            async_op=True,
        )
        try:
            done = work.wait(timeout=timedelta(seconds=_EVAL_BROADCAST_TIMEOUT_SEC))  # type: ignore[reportOptionalMemberAccess]
        except RuntimeError as e:
            logger.error(
                f"Eval loss broadcast timed out after {_EVAL_BROADCAST_TIMEOUT_SEC}s; "
                f"a PP stage is likely wedged. Reporting nan loss. Underlying: {e}"
            )
            avg_loss = float("nan")
        else:
            if done is False:
                logger.error(
                    f"Eval loss broadcast timed out after {_EVAL_BROADCAST_TIMEOUT_SEC}s; "
                    "reporting nan loss."
                )
                avg_loss = float("nan")
            else:
                avg_loss = loss_tensor[0].item()
    else:
        total_loss = 0.0
        eval_iter = iter(eval_dataloader)
        for _ in range(eval_steps):
            try:
                batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_dataloader)
                batch = next(eval_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

        avg_loss = total_loss / eval_steps

    model.train()
    return {"eval/loss": avg_loss, "eval/perplexity": math.exp(min(avg_loss, 20.0))}


def _cycle(dataloader):
    """Yield the batches of ``dataloader`` over and over, restarting it at each end.

    Raises:
        ValueError: ``dataloader`` yields no batch, as a ``drop_last`` loader over
            fewer samples than one batch does.
    """
    while True:
        empty = True
        for batch in dataloader:
            empty = False
            yield batch
        if empty:
            raise ValueError("The eval dataloader yields no batch")
