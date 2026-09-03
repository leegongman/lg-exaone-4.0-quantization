#!/usr/bin/env python3
"""Measure local vLLM token timing with synthetic prompts.

This utility is for local comparisons on a fixed machine. It is not an
official competition evaluator and does not compute a competition score.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

DEFAULT_TOKENIZER = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DEFAULT_PROMPT_LENGTHS = (64, 128, 256, 512, 1024, 2048)


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("prompt lengths must be positive integers")
    return lengths


def build_prompts(
    tokenizer: Any,
    lengths: tuple[int, ...],
    prompts_per_length: int,
    seed: int,
) -> list[str]:
    random.seed(seed)
    special_ids = set(tokenizer.all_special_ids)
    token_ids = [token_id for token_id in tokenizer.get_vocab().values() if token_id not in special_ids]
    if not token_ids:
        raise RuntimeError("the tokenizer did not expose non-special token ids")

    prompts: list[str] = []
    for length in lengths:
        for _ in range(prompts_per_length):
            text = tokenizer.decode(random.choices(token_ids, k=length))
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (AttributeError, ValueError):
                prompt = text
            prompts.append(prompt)

    random.shuffle(prompts)
    return prompts


def run_benchmark(
    model_name: str,
    tokenizer_name: str,
    prompt_lengths: tuple[int, ...],
    prompts_per_length: int,
    max_tokens: int,
    seed: int,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    prompts = build_prompts(tokenizer, prompt_lengths, prompts_per_length, seed)

    engine = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)

    # Warmup is intentionally excluded from the timing report.
    engine.generate(prompts[:1], sampling_params)

    started_at = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params)
    elapsed_seconds = time.perf_counter() - started_at

    prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    generated_tokens = sum(
        len(candidate.token_ids) for output in outputs for candidate in output.outputs
    )
    total_tokens = prompt_tokens + generated_tokens
    if generated_tokens == 0 or total_tokens == 0:
        raise RuntimeError("no tokens were generated; benchmark result is invalid")

    return {
        "status": "local-benchmark",
        "scope": "Generation wall time after model load and warmup; not an official competition evaluation.",
        "model": model_name,
        "tokenizer": tokenizer_name,
        "seed": seed,
        "prompt_lengths": list(prompt_lengths),
        "prompts_per_length": prompts_per_length,
        "max_tokens": max_tokens,
        "prompt_count": len(prompts),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed_seconds,
        "decode_seconds_per_token": elapsed_seconds / generated_tokens,
        "total_seconds_per_token": elapsed_seconds / total_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed_seconds,
        "gpu_memory_utilization": gpu_memory_utilization,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local path or Hugging Face model identifier")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer identifier or path (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--prompt-lengths",
        default=",".join(str(length) for length in DEFAULT_PROMPT_LENGTHS),
        type=parse_lengths,
        help="Comma-separated synthetic prompt lengths",
    )
    parser.add_argument("--prompts-per-length", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. Use an ignored directory such as outputs/.",
    )
    args = parser.parse_args()

    if args.prompts_per_length <= 0 or args.max_tokens <= 0:
        parser.error("--prompts-per-length and --max-tokens must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    report = run_benchmark(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        prompt_lengths=args.prompt_lengths,
        prompts_per_length=args.prompts_per_length,
        max_tokens=args.max_tokens,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    serialized_report = json.dumps(report, indent=2, sort_keys=True)
    print(serialized_report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{serialized_report}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
