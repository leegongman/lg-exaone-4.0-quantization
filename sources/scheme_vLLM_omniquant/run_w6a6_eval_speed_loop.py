#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE_MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DEFAULT_MODEL_DIR = (
    "/home/ubuntu/vLLM_OmniQuant/artifacts/checkpoints/"
    "w6a6_3epoch_deactive_amp_let1e-2_lwc5e-3"
)
DEFAULT_CALIBSET = "/home/ubuntu/vLLM_OmniQuant/calibset_v2.1.jsonl"


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return process.wait()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def find_latest_results_json(output_root: Path) -> Path | None:
    candidates = sorted(output_root.rglob("results*.json"))
    return candidates[-1] if candidates else None


def extract_gsm8k_metrics(results_json: Path) -> dict[str, float | None]:
    payload = load_json(results_json)
    gsm8k = payload.get("results", {}).get("gsm8k", {})
    if not isinstance(gsm8k, dict):
        return {
            "exact_match_strict": None,
            "exact_match_flexible": None,
        }
    return {
        "exact_match_strict": _safe_float(gsm8k.get("exact_match,strict-match")),
        "exact_match_flexible": _safe_float(gsm8k.get("exact_match,flexible-extract")),
    }


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def read_latest_csv_row(csv_path: Path) -> dict[str, str] | None:
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def print_stage_summary(stage: str, payload: dict[str, Any]) -> None:
    print("\n==============================")
    print(f"[{stage}]")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("==============================\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--repo-root", default="/home/ubuntu/vLLM_OmniQuant")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--baseline-model", default=BASELINE_MODEL)
    parser.add_argument("--calibset", default=DEFAULT_CALIBSET)
    parser.add_argument("--gsm8k-limit", type=int, default=100)
    parser.add_argument("--bench-limit", type=int, default=16)
    parser.add_argument("--smoke-max-tokens", type=int, default=8)
    parser.add_argument("--smoke-max-model-len", type=int, default=256)
    parser.add_argument("--bench-max-tokens", type=int, default=256)
    parser.add_argument("--norm-max-gen-toks", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--quantization", default="omni_activation_real")
    parser.add_argument("--pythonpath", action="append", default=[])
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--fixed-decode-tokens",
        dest="fixed_decode_tokens",
        action="store_true",
    )
    parser.add_argument(
        "--no-fixed-decode-tokens",
        dest="fixed_decode_tokens",
        action="store_false",
    )
    parser.set_defaults(fixed_decode_tokens=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_name = model_dir.name
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else repo_root / "artifacts" / "loop_runs" / f"{model_name}_{run_stamp}"
    )
    logs_dir = output_root / "logs"
    summary_path = output_root / "summary.json"

    env = os.environ.copy()
    pythonpath_entries = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    extra_pythonpath_entries = [str(Path(entry).resolve()) for entry in args.pythonpath]
    if extra_pythonpath_entries:
        env["PYTHONPATH"] = os.pathsep.join(extra_pythonpath_entries + pythonpath_entries)
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    summary: dict[str, Any] = {
        "run_stamp_utc": run_stamp,
        "repo_root": str(repo_root),
        "model_dir": str(model_dir),
        "baseline_model": args.baseline_model,
        "calibset": args.calibset,
        "fixed_decode_tokens": args.fixed_decode_tokens,
        "stages": {},
    }
    write_json(summary_path, summary)

    smoke_cmd = [
        args.python_bin,
        str(
            repo_root
            / "OmniQuant_EXAONE_v4"
            / "scripts"
            / "run_vllm_omni_activation_real_smoke.py"
        ),
        "--model_dir",
        str(model_dir),
        "--max_tokens",
        str(args.smoke_max_tokens),
        "--max_model_len",
        str(args.smoke_max_model_len),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
    ]
    smoke_rc = run_command(
        smoke_cmd,
        cwd=repo_root,
        env=env,
        log_path=logs_dir / "01_smoke.log",
    )
    summary["stages"]["smoke"] = {
        "return_code": smoke_rc,
        "log_path": str(logs_dir / "01_smoke.log"),
    }
    write_json(summary_path, summary)
    print_stage_summary("Smoke", summary["stages"]["smoke"])
    if smoke_rc != 0:
        return smoke_rc

    eval_output = output_root / "eval"
    eval_cmd = [
        args.python_bin,
        str(
            repo_root
            / "OmniQuant_EXAONE_v4"
            / "scripts"
            / "run_vllm_omni_activation_real_lm_eval.py"
        ),
        "--model-dir",
        str(model_dir),
        "--output-root",
        str(eval_output),
        "--tasks",
        "gsm8k",
        "--limit",
        str(args.gsm8k_limit),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--python-bin",
        args.python_bin,
        "--batch-size",
        "auto",
    ]
    eval_rc = run_command(
        eval_cmd,
        cwd=repo_root,
        env=env,
        log_path=logs_dir / "02_gsm8k_eval.log",
    )
    results_json = find_latest_results_json(eval_output)
    gsm8k_metrics = extract_gsm8k_metrics(results_json) if results_json else {}
    summary["stages"]["gsm8k_eval"] = {
        "return_code": eval_rc,
        "limit": args.gsm8k_limit,
        "results_json": str(results_json) if results_json else None,
        "log_path": str(logs_dir / "02_gsm8k_eval.log"),
        **gsm8k_metrics,
    }
    write_json(summary_path, summary)
    print_stage_summary(f"GSM8K {args.gsm8k_limit}", summary["stages"]["gsm8k_eval"])
    if eval_rc != 0:
        return eval_rc

    speed_output = output_root / "speed"
    speed_cmd = [
        args.python_bin,
        str(repo_root / "norm_speed.py"),
        "--model",
        str(model_dir),
        "--baseline",
        args.baseline_model,
        "--calibset",
        args.calibset,
        "--max_gen_toks",
        str(args.norm_max_gen_toks),
        "--output_path",
        str(speed_output),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--quantization",
        args.quantization,
        "--dtype",
        args.dtype,
    ]
    if args.fixed_decode_tokens:
        speed_cmd.append("--fixed_decode_tokens")
    speed_rc = run_command(
        speed_cmd,
        cwd=repo_root,
        env=env,
        log_path=logs_dir / "03_norm_speed.log",
    )
    speed_csv = speed_output / f"norm_speed_{model_name}.csv"
    speed_row = read_latest_csv_row(speed_csv) or {}
    summary["stages"]["norm_speed"] = {
        "return_code": speed_rc,
        "csv_path": str(speed_csv),
        "log_path": str(logs_dir / "03_norm_speed.log"),
        "normalized_speed_percent": _safe_float(speed_row.get("normalized_speed_percent")),
        "tpt_total_model": _safe_float(speed_row.get("tpt_total_model")),
        "tpt_total_baseline": _safe_float(speed_row.get("tpt_total_baseline")),
        "tpt_decode_model": _safe_float(speed_row.get("tpt_decode_model")),
        "tpt_decode_baseline": _safe_float(speed_row.get("tpt_decode_baseline")),
        "fixed_decode_tokens": args.fixed_decode_tokens,
    }
    write_json(summary_path, summary)
    print_stage_summary("Norm Speed", summary["stages"]["norm_speed"])
    if speed_rc != 0:
        return speed_rc

    bench_dir = output_root / "bench"
    bench_model_cmd = [
        args.python_bin,
        str(repo_root / "bench_real_latency.py"),
        "--model",
        str(model_dir),
        "--calibset",
        args.calibset,
        "--limit",
        str(args.bench_limit),
        "--max_tokens",
        str(args.bench_max_tokens),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--quantization",
        args.quantization,
        "--dtype",
        args.dtype,
    ]
    bench_model_rc = run_command(
        bench_model_cmd,
        cwd=repo_root,
        env=env,
        log_path=logs_dir / "04_bench_quant.log",
    )
    bench_base_cmd = [
        args.python_bin,
        str(repo_root / "bench_real_latency.py"),
        "--model",
        args.baseline_model,
        "--calibset",
        args.calibset,
        "--limit",
        str(args.bench_limit),
        "--max_tokens",
        str(args.bench_max_tokens),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
        "--baseline",
    ]
    bench_base_rc = run_command(
        bench_base_cmd,
        cwd=repo_root,
        env=env,
        log_path=logs_dir / "05_bench_baseline.log",
    )

    bench_quant_payload = _extract_last_json(logs_dir / "04_bench_quant.log")
    bench_base_payload = _extract_last_json(logs_dir / "05_bench_baseline.log")
    write_json(bench_dir / "quant.json", bench_quant_payload or {})
    write_json(bench_dir / "baseline.json", bench_base_payload or {})

    summary["stages"]["bench_real_latency"] = {
        "quant_return_code": bench_model_rc,
        "baseline_return_code": bench_base_rc,
        "quant_json_path": str(bench_dir / "quant.json"),
        "baseline_json_path": str(bench_dir / "baseline.json"),
        "quant": bench_quant_payload,
        "baseline": bench_base_payload,
    }
    write_json(summary_path, summary)
    print_stage_summary("Bench Real Latency", summary["stages"]["bench_real_latency"])

    if bench_model_rc != 0:
        return bench_model_rc
    if bench_base_rc != 0:
        return bench_base_rc
    return 0


def _extract_last_json(log_path: Path) -> dict[str, Any] | None:
    text = log_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    for start_idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start_idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            best = payload
    return best


if __name__ == "__main__":
    raise SystemExit(main())
