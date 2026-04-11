#!/usr/bin/env python3
"""Generate text from a KempnerForge checkpoint.

Single-GPU script for research and debugging — load a checkpoint, tokenize a
prompt, generate tokens, and print the decoded output.

Usage:
    # Greedy generation from a DCP checkpoint
    uv run python scripts/generate.py configs/train/debug.toml \
        --checkpoint.load_path=checkpoints/step_1000 \
        --data.tokenizer_path=gpt2 \
        --prompt "The Kempner Institute"

    # With sampling parameters
    uv run python scripts/generate.py configs/train/7b.toml \
        --checkpoint.load_path=checkpoints/step_50000 \
        --data.tokenizer_path=meta-llama/Llama-2-7b-hf \
        --prompt "Once upon a time" \
        --max_tokens 256 --temperature 0.8 --top_p 0.9

    # Interactive REPL mode
    uv run python scripts/generate.py configs/train/debug.toml \
        --checkpoint.load_path=checkpoints/step_1000 \
        --data.tokenizer_path=gpt2 \
        --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from kempnerforge.config.loader import load_config
from kempnerforge.config.registry import registry
from kempnerforge.model.generate import generate


def _load_model(config, device, dtype):
    """Build model and load DCP checkpoint weights."""
    model_builder = registry.get_model(config.model.model_type)
    model = model_builder(config.model).to(device=device, dtype=dtype)

    load_path = config.checkpoint.load_path
    if load_path:
        ckpt_path = Path(load_path)
        if not ckpt_path.exists():
            print(f"Error: checkpoint path does not exist: {ckpt_path}", file=sys.stderr)
            sys.exit(1)

        # DCP handles single-process loading without dist init
        state_dict = {"model": model.state_dict()}
        dcp.load(state_dict, checkpoint_id=str(ckpt_path))
        model.load_state_dict(state_dict["model"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("Warning: no checkpoint specified, using random weights", file=sys.stderr)

    return model


def _generate_and_print(model, tokenizer, prompt, args):
    """Tokenize prompt, generate, decode, and print."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the generated portion
    generated_ids = output_ids[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\n--- Prompt ---\n{prompt}")
    print(f"\n--- Generated ({len(generated_ids)} tokens) ---\n{generated_text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a KempnerForge checkpoint")
    parser.add_argument("config", help="Path to config TOML file")
    parser.add_argument("--prompt", type=str, default="", help="Input prompt text")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature (0=greedy)"
    )
    parser.add_argument("--top_k", type=int, default=0, help="Top-k filtering (0=disabled)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling threshold")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument(
        "--device", type=str, default=None, help="Device (default: cuda if available)"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"]
    )

    args, config_overrides = parser.parse_known_args()

    # Load config with CLI overrides
    config = load_config(args.config, cli_args=config_overrides)

    # Resolve device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # Build and load model
    model = _load_model(config, device, dtype)

    # Build tokenizer
    if not config.data.tokenizer_path:
        print("Error: --data.tokenizer_path is required for generation", file=sys.stderr)
        sys.exit(1)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path)

    print(f"Model: {config.model.num_params_estimate / 1e6:.0f}M params on {device} ({dtype})")

    if args.interactive:
        print("Interactive mode — type a prompt and press Enter. Ctrl+C to exit.\n")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not prompt.strip():
                continue
            _generate_and_print(model, tokenizer, prompt, args)
    elif args.prompt:
        _generate_and_print(model, tokenizer, args.prompt, args)
    else:
        print("Error: provide --prompt or use --interactive", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
