"""GPU regression for packed VLM compilation across varying row and image counts."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import FLEX_BLOCK_SIZE, ModelConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import VLMConfig
from kempnerforge.data.vlm_dataset import pack_vlm_samples
from kempnerforge.model.masking import use_dynamic_flex_shapes
from kempnerforge.model.vlm import VLMWrapper, build_vlm_wrapper
from kempnerforge.training.loop import _mark_rows_dynamic

_IMAGE_TOKENS = 8
_SEQ_LEN = FLEX_BLOCK_SIZE
_TEXT_LEN = _SEQ_LEN // 2 - _IMAGE_TOKENS


def _model(backend: str) -> VLMWrapper:
    """Build the same small VLM with either dense or compiled sparse attention."""
    return (
        build_vlm_wrapper(
            ModelConfig(
                dim=64,
                n_layers=1,
                n_heads=2,
                vocab_size=128,
                max_seq_len=_SEQ_LEN,
                attention_backend=backend,
            ),
            VisionEncoderConfig(type="random", feature_dim=32, num_tokens=_IMAGE_TOKENS),
            AdapterConfig(),
            VLMConfig(max_text_len=_TEXT_LEN),
        )
        .to("cuda")
        .eval()
    )


def _batch(rows: int) -> dict[str, torch.Tensor]:
    """Place two half-row documents per row, with a single document in the last row."""
    samples = []
    for index in range(2 * rows - 1):
        ids = (torch.arange(_TEXT_LEN, dtype=torch.long) + index + 1) % 127 + 1
        labels = torch.full_like(ids, -100)
        labels[:-1] = ids[1:]
        samples.append(
            {
                "pixel_values": torch.full((3, 4, 4), float(index + 1)),
                "input_ids": ids,
                "labels": labels,
                "text_len": torch.tensor(_TEXT_LEN),
            }
        )
    batch = pack_vlm_samples(samples, seq_len=_SEQ_LEN, tokens_per_image=_IMAGE_TOKENS, pad_id=0)
    assert batch["input_ids"].shape == (rows, _SEQ_LEN)
    assert batch["pixel_values"].shape[0] == 2 * rows - 1
    return {name: tensor.to("cuda") for name, tensor in batch.items()}


@pytest.mark.gpu
def test_packed_flex_compile_tracks_varying_rows_and_images() -> None:
    """Keep dense parity and graph reuse while packed rows and image counts change.

    The random encoder uses CPU generators in a Python loop, so only that test
    stub runs outside Dynamo. The adapter, inline placement and transformer
    follow the compiled training route, including its dynamic-dimension marks.
    Singleton dimensions specialize separately. Warm rows one through three,
    then return to one after Dynamo generalizes the larger shapes, before
    checking unseen row counts and repeated returns to the warmed shapes.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch._dynamo.reset()
    counters.clear()
    use_dynamic_flex_shapes(True)
    torch.manual_seed(0)
    dense = _model("sdpa")
    sparse = _model("flex")
    sparse.load_state_dict(dense.state_dict())
    sparse.vision_encoder.forward = torch._dynamo.disable(sparse.vision_encoder.forward)
    compiled = torch.compile(sparse)

    def step(rows: int) -> None:
        batch = _batch(rows)
        image_tokens = int(batch.pop("image_tokens"))
        _mark_rows_dynamic(*batch.values())
        dense.zero_grad(set_to_none=True)
        sparse.zero_grad(set_to_none=True)
        reference, reference_labels = dense(**batch, image_tokens=image_tokens)
        actual, _ = compiled(**batch, image_tokens=image_tokens)
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
        assert reference_labels is not None
        labels = reference_labels.reshape(-1)
        reference_loss = F.cross_entropy(reference.reshape(-1, 128), labels)
        actual_loss = F.cross_entropy(actual.reshape(-1, 128), labels)
        assert torch.isfinite(actual_loss)
        torch.testing.assert_close(actual_loss, reference_loss, rtol=2e-5, atol=2e-5)
        reference_loss.backward()
        actual_loss.backward()
        for (name, expected), (_, parameter) in zip(
            dense.named_parameters(), sparse.named_parameters(), strict=True
        ):
            if not parameter.requires_grad:
                continue
            assert parameter.grad is not None, name
            assert expected.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-4, atol=1e-6)

    try:
        for rows in (1, 2, 3, 1):
            step(rows)
        after_warmup = counters["stats"]["unique_graphs"]
        assert after_warmup > 0, "packed FlexAttention never compiled"
        for rows in (*range(4, 10), 1, 2, 3, 1):
            step(rows)
            assert counters["stats"]["unique_graphs"] == after_warmup, (
                f"packed row count {rows} caused recompilation after warmup: "
                f"{after_warmup} -> {counters['stats']['unique_graphs']} graphs"
            )
    finally:
        use_dynamic_flex_shapes(False)
        torch._dynamo.reset()
