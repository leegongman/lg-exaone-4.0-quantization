#!/usr/bin/env python3

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

BASE_MODEL_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"

logger = logging.getLogger(__name__)


def seed_every(seed: int = 42):
    np.random.seed(seed)


def generate_prompts(calibset_path: str, tokenizer):
    with open(calibset_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    prompts = []
    for entry in entries:
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
            prompts.append(tokenizer.decode(entry["input_ids"], skip_special_tokens=False))
            continue
        raise KeyError("Expected either 'message' or 'input_ids' in calibset entry.")

    return prompts


def measure_tpt(
    model_name: str,
    prompts: list[str],
    max_gen_toks: int = 4096,
    trust_remote_code: bool = True,
    gpu_memory_utilization: float = 0.85,
    quantization: str | None = None,
    dtype: str | None = None,
    enforce_eager: bool | None = None,
    fixed_decode_tokens: bool = False,
):
    logger.info("Loading model: %s", model_name)

    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    if quantization:
        llm_kwargs["quantization"] = quantization
    if dtype:
        llm_kwargs["dtype"] = dtype
    if enforce_eager is not None:
        llm_kwargs["enforce_eager"] = enforce_eager

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_gen_toks,
        min_tokens=max_gen_toks if fixed_decode_tokens else 0,
        ignore_eos=fixed_decode_tokens,
    )

    logger.info("Warmup...")
    _ = llm.generate(prompts[:1], sampling_params)

    logger.info("Measuring...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()

    total_time = end - start
    total_prefill_tokens = 0
    total_decode_tokens = 0

    for out in outputs:
        total_prefill_tokens += len(out.prompt_token_ids)
        for o in out.outputs:
            total_decode_tokens += len(o.token_ids)

    total_tokens = total_prefill_tokens + total_decode_tokens
    tpt_decode_only = total_time / max(total_decode_tokens, 1)
    tpt_total = total_time / max(total_tokens, 1)

    return {
        "tpt_decode_only": tpt_decode_only,
        "tpt_total": tpt_total,
        "total_time": total_time,
        "prefill_tokens": total_prefill_tokens,
        "decode_tokens": total_decode_tokens,
        "total_tokens": total_tokens,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default=BASE_MODEL_NAME)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--calibset", type=str, required=True)
    parser.add_argument("--max_gen_toks", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--quantization", type=str, default="omni_activation_real")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--fixed_decode_tokens", action="store_true")
    args = parser.parse_args()

    seed_every(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
    )
    prompts = generate_prompts(args.calibset, tokenizer)

    model_results = measure_tpt(
        args.model,
        prompts,
        max_gen_toks=args.max_gen_toks,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        fixed_decode_tokens=args.fixed_decode_tokens,
    )

    base_results = measure_tpt(
        args.baseline,
        prompts,
        max_gen_toks=args.max_gen_toks,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=None,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        fixed_decode_tokens=args.fixed_decode_tokens,
    )

    speed_norm = 1.0 - (model_results["tpt_total"] / base_results["tpt_total"])

    print("\n==============================")
    print("[Speed Report]")
    print("--------------------------------------------------")
    print("[Total]")
    print(f"TPT (model)              : {model_results['tpt_total']:.6f}")
    print(f"TPT (baseline)           : {base_results['tpt_total']:.6f}")
    print(f"Normalized Speed         : {speed_norm:.6f} ({speed_norm*100:+.2f}%)")
    print("--------------------------------------------------")
    print("[Decode Only]")
    print(f"TPT (model)              : {model_results['tpt_decode_only']:.6f}")
    print(f"TPT (baseline)           : {base_results['tpt_decode_only']:.6f}")
    print("--------------------------------------------------")
    print("[Token Statistics]")
    print(f"Prefill Tokens           : {model_results['prefill_tokens']:,}")
    print(f"Decode Tokens            : {model_results['decode_tokens']:,}")
    print(f"Total Tokens             : {model_results['total_tokens']:,}")
    print(f"Total Time (s)           : {model_results['total_time']:.4f}")
    print("==============================")

    model_name_for_file = os.path.basename(args.model.rstrip("/"))
    csv_filename = os.path.join(args.output_path, f"norm_speed_{model_name_for_file}.csv")
    os.makedirs(args.output_path, exist_ok=True)

    row_dict = {
        "model": args.model,
        "baseline": args.baseline,
        "tpt_total_model": model_results["tpt_total"],
        "tpt_total_baseline": base_results["tpt_total"],
        "tpt_decode_model": model_results["tpt_decode_only"],
        "tpt_decode_baseline": base_results["tpt_decode_only"],
        "normalized_speed": speed_norm,
        "normalized_speed_percent": speed_norm * 100,
        "prefill_tokens": model_results["prefill_tokens"],
        "decode_tokens": model_results["decode_tokens"],
        "total_tokens": model_results["total_tokens"],
        "total_time_sec": model_results["total_time"],
        "fixed_decode_tokens": args.fixed_decode_tokens,
        "max_gen_toks": args.max_gen_toks,
    }

    df_new = pd.DataFrame([row_dict])
    if os.path.exists(csv_filename):
        df_existing = pd.read_csv(csv_filename)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    main()
