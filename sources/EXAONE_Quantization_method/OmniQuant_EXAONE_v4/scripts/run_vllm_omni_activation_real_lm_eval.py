#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_model_config_dtype(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    config = _load_json(config_path)
    changed = False

    if config.get("dtype") != "float16":
        config["dtype"] = "float16"
        changed = True
    if config.get("torch_dtype") != "float16":
        config["torch_dtype"] = "float16"
        changed = True

    if changed:
        _write_json(config_path, config)
    return changed


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    pythonpath_entries = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    extra_entries = [str(Path(p).resolve()) for p in args.pythonpath]
    if extra_entries:
        env["PYTHONPATH"] = os.pathsep.join(extra_entries + pythonpath_entries)

    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.gloo_socket_ifname:
        env["GLOO_SOCKET_IFNAME"] = args.gloo_socket_ifname
    if args.vllm_cache_root:
        env["VLLM_CACHE_ROOT"] = args.vllm_cache_root

    return env


def build_lm_eval_env(env: dict[str, str]) -> dict[str, str]:
    clean_env = env.copy()
    pythonpath_entries = [
        entry
        for entry in clean_env.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    filtered_entries = []
    for entry in pythonpath_entries:
        entry_path = Path(entry)
        if (entry_path / "lm_eval").exists():
            continue
        filtered_entries.append(entry)
    if filtered_entries:
        clean_env["PYTHONPATH"] = os.pathsep.join(filtered_entries)
    else:
        clean_env.pop("PYTHONPATH", None)
    return clean_env


def run_command(cmd: list[str], env: dict[str, str], cwd: Path) -> int:
    print("$", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=cwd, env=env)
    return completed.returncode


def run_readiness_check(
    python_bin: str,
    script_path: Path,
    model_dir: Path,
    env: dict[str, str],
    cwd: Path,
) -> int:
    cmd = [
        python_bin,
        str(script_path),
        "--model_dir",
        str(model_dir),
    ]
    return run_command(cmd, env=env, cwd=cwd)


def build_lm_eval_command(
    python_bin: str,
    wrapper_script: Path,
    model_dir: Path,
    output_path: Path,
    tasks: str,
    limit: int,
    batch_size: str,
    gpu_memory_utilization: float,
    extra_model_args: list[str],
    apply_chat_template: bool,
) -> list[str]:
    model_args = [
        f"pretrained={model_dir}",
        f"gpu_memory_utilization={gpu_memory_utilization}",
        "enable_thinking=False",
        "enforce_eager=True",
        "dtype=float16",
        "quantization=omni_activation_real",
    ]
    model_args.extend(extra_model_args)

    cmd = [
        python_bin,
        str(wrapper_script),
        "--model",
        "vllm",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        tasks,
        "--output_path",
        str(output_path),
        "--limit",
        str(limit),
        "--batch_size",
        batch_size,
    ]
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    return cmd


def find_results_json(output_path: Path) -> Path | None:
    candidates = sorted(output_path.rglob("results*.json"))
    if candidates:
        return candidates[-1]
    for candidate in sorted(output_path.rglob("*.json")):
        try:
            data = _load_json(candidate)
        except Exception:
            continue
        if isinstance(data, dict) and "results" in data:
            return candidate
    return None


def extract_gsm8k_score(results_json_path: Path) -> tuple[float | None, dict[str, Any]]:
    payload = _load_json(results_json_path)
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return None, payload

    gsm8k_entries = {
        key: value
        for key, value in results.items()
        if str(key).startswith("gsm8k")
    }
    if not gsm8k_entries:
        return None, payload

    best_score = None
    for metrics in gsm8k_entries.values():
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_value in metrics.items():
            if "exact_match" not in str(metric_name):
                continue
            if not isinstance(metric_value, (int, float)):
                continue
            metric_value = float(metric_value)
            if best_score is None or metric_value > best_score:
                best_score = metric_value
    return best_score, payload


def parse_extra_model_args(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    lm_eval_wrapper = script_dir / "lm_eval_entry_with_omni_quant.py"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tasks", default="aime25,gsm8k,truthfulqa_mc1,ruler")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--pythonpath", action="append", default=[str(repo_root)])
    parser.add_argument("--extra-model-args", default="")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--gloo-socket-ifname", default=None)
    parser.add_argument("--vllm-cache-root", default=None)
    parser.add_argument("--skip-readiness-check", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ensure_model_config_dtype(model_dir)
    env = build_env(args)
    lm_eval_env = build_lm_eval_env(env)
    extra_model_args = parse_extra_model_args(args.extra_model_args)

    if not args.skip_readiness_check:
        readiness_script = repo_root / "scripts" / "check_vllm_omni_activation_real_readiness.py"
        readiness_rc = run_readiness_check(
            python_bin=args.python_bin,
            script_path=readiness_script,
            model_dir=model_dir,
            env=env,
            cwd=repo_root,
        )
        if readiness_rc != 0:
            return readiness_rc

    attempts = [
        ("chat_template", True),
        ("no_chat_template", False),
    ]

    summary: dict[str, Any] = {
        "model_dir": str(model_dir),
        "output_root": str(output_root),
        "attempts": [],
    }

    best_attempt: dict[str, Any] | None = None

    for attempt_name, apply_chat_template in attempts:
        attempt_output = output_root / attempt_name
        cmd = build_lm_eval_command(
            python_bin=args.python_bin,
            wrapper_script=lm_eval_wrapper,
            model_dir=model_dir,
            output_path=attempt_output,
            tasks=args.tasks,
            limit=args.limit,
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            extra_model_args=extra_model_args,
            apply_chat_template=apply_chat_template,
        )
        rc = run_command(cmd, env=lm_eval_env, cwd=output_root)
        results_json = find_results_json(attempt_output)
        score = None
        if results_json is not None:
            score, _ = extract_gsm8k_score(results_json)

        attempt_summary = {
            "name": attempt_name,
            "apply_chat_template": apply_chat_template,
            "return_code": rc,
            "results_json": str(results_json) if results_json else None,
            "gsm8k_exact_match": score,
        }
        summary["attempts"].append(attempt_summary)
        print(json.dumps(attempt_summary, indent=2, ensure_ascii=False), flush=True)

        if rc == 0 and (
            best_attempt is None
            or (attempt_summary["gsm8k_exact_match"] or -1) > (best_attempt["gsm8k_exact_match"] or -1)
        ):
            best_attempt = attempt_summary

        if rc == 0 and score is not None and score > 0:
            summary["final"] = attempt_summary
            _write_json(output_root / "summary.json", summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            return 0

        if rc != 0:
            break

    if best_attempt is not None:
        summary["final"] = best_attempt
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
