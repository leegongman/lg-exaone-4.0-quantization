#!/usr/bin/env python3

import argparse
import json
import logging
import statistics
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

BASE_MODEL_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"


def load_prompts(calibset_path: str, limit: int, tokenizer) -> list[str]:
    prompts = []
    with open(calibset_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            entry = json.loads(line)
            if "message" in entry:
                prompts.append(
                    tokenizer.apply_chat_template(
                        entry["message"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                continue
            if "input_ids" in entry:
                prompts.append(
                    tokenizer.decode(entry["input_ids"], skip_special_tokens=False)
                )
                continue
            raise KeyError("Expected either 'message' or 'input_ids' in calibset entry.")
    return prompts


def run_single_batch(llm: LLM, prompts: list[str], max_tokens: int) -> dict:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        ignore_eos=True,
    )
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()

    prompt_tokens = sum(len(out.prompt_token_ids) for out in outputs)
    decode_tokens = sum(len(seq.token_ids) for out in outputs for seq in out.outputs)
    elapsed = end - start
    return {
        "elapsed": elapsed,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "total_tokens": prompt_tokens + decode_tokens,
        "tokens_per_sec_total": (prompt_tokens + decode_tokens) / max(elapsed, 1e-9),
        "tokens_per_sec_decode": decode_tokens / max(elapsed, 1e-9),
    }


def run_promptwise(llm: LLM, prompts: list[str], max_tokens: int) -> dict:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        ignore_eos=True,
    )
    latencies = []
    decode_tps = []

    for prompt in prompts:
        start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        end = time.time()
        elapsed = end - start
        decode_tokens = sum(len(seq.token_ids) for seq in outputs[0].outputs)
        latencies.append(elapsed)
        decode_tps.append(decode_tokens / max(elapsed, 1e-9))

    return {
        "latency_p50": statistics.median(latencies),
        "latency_p90": sorted(latencies)[max(int(len(latencies) * 0.9) - 1, 0)],
        "decode_tps_p50": statistics.median(decode_tps),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibset", required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--quantization", default="omni_activation_real")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    prompts = load_prompts(args.calibset, args.limit, tokenizer)

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "dtype": args.dtype,
    }
    if not args.baseline:
        llm_kwargs["quantization"] = args.quantization

    llm = LLM(**llm_kwargs)

    warmup_tokens = min(args.max_tokens, 32)
    warmup_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=warmup_tokens,
        min_tokens=warmup_tokens,
        ignore_eos=True,
    )
    llm.generate(prompts[:1], warmup_params)

    batch_metrics = run_single_batch(llm, prompts, args.max_tokens)
    promptwise_metrics = run_promptwise(llm, prompts[: min(len(prompts), 8)], min(args.max_tokens, 128))

    report = {**batch_metrics, **promptwise_metrics}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
