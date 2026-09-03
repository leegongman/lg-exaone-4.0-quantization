# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tiny request-level generate smoke for EXAONE omni activation real.

This script is intentionally small and correctness-oriented. It validates that
the request-level path can:

1. initialize an LLM/engine on a real checkpoint,
2. enter the request/generate path,
3. produce at least one generated token when possible, and
4. expose decode helper counters from the loaded model via apply_model().
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_CACHE_ROOT", "/tmp/vllm-cache")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm import LLM, SamplingParams


def _inspect_model(model) -> dict[str, Any]:
    attn = model.model.layers[0].self_attn.attn
    return {
        "layer_name": attn.layer_name,
        "engine_status": getattr(attn, "omni_last_attention_engine_status", None),
        "qkt_runtime_calls": int(
            getattr(attn, "omni_last_attention_qkt_runtime_calls", 0)
        ),
        "pv_runtime_calls": int(
            getattr(attn, "omni_last_attention_pv_runtime_calls", 0)
        ),
        "cache_write_calls": int(
            getattr(attn, "omni_last_attention_cache_write_calls", 0)
        ),
        "cache_read_calls": int(
            getattr(attn, "omni_last_attention_cache_read_calls", 0)
        ),
        "cache_backend": getattr(attn, "omni_last_attention_cache_backend", None),
        "metadata_summary": getattr(attn, "omni_last_attention_metadata_summary", None),
        "attention_helper_attached": hasattr(attn, "omni_attention_helper"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=2)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    prompts = [
        "1+1=",
        "The capital of France is",
    ]
    summary: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "repo_root": str(REPO_ROOT),
        "checkpoint_dir": str(checkpoint_dir),
        "request_path_used": True,
        "last_success": None,
        "last_failure": None,
        "status": "started",
    }

    try:
        llm = LLM(
            model=str(checkpoint_dir),
            tokenizer=str(checkpoint_dir),
            quantization="omni_activation_real",
            dtype="float16",
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=64,
            gpu_memory_utilization=0.1,
        )
        summary["engine"] = {
            "engine_class": type(llm.llm_engine).__name__,
            "model_runner_type": llm.runner_type,
        }
        summary["last_success"] = "engine_initialized"

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            ignore_eos=True,
        )

        prompt_summaries: list[dict[str, Any]] = []
        for idx, prompt in enumerate(prompts):
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            req_output = outputs[0]
            completion = req_output.outputs[0]
            model_debug = llm.apply_model(_inspect_model)[0]
            prompt_summaries.append(
                {
                    "prompt_index": idx,
                    "prompt": prompt,
                    "generated_token_count": len(completion.token_ids),
                    "generated_token_ids": list(completion.token_ids),
                    "generated_text": completion.text,
                    "finish_reason": str(getattr(completion, "finish_reason", None)),
                    "stop_reason": str(getattr(completion, "stop_reason", None)),
                    "model_debug": model_debug,
                }
            )
            summary["last_success"] = (
                "first_request_generated" if idx == 0 else "second_request_generated"
            )
        summary["requests"] = prompt_summaries
        summary["status"] = "ok"
    except Exception as exc:
        summary["status"] = "failed"
        summary["last_failure"] = (
            f"{summary['last_success'] or 'startup'}:{type(exc).__name__}"
        )
        summary["error"] = str(exc)
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
