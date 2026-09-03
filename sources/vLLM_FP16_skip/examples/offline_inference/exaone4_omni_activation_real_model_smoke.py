# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-level smoke test for EXAONE omni activation real checkpoints.

This script is intended for submission-side validation of a real checkpoint
directory. It does not mutate the checkpoint. Instead it:

1. validates that the checkpoint layout is readable,
2. checks whether the checkpoint is actually compatible with
   ``omni_activation_real``,
3. if compatible, attempts model construction, weight loading, attach
   validation, and a tiny forward pass,
4. otherwise exits with a precise failure stage so the remaining blocker is
   obvious.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_CACHE_ROOT", "/tmp/vllm-cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from safetensors import safe_open

from vllm.config import LoadConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import (
    get_quant_config,
    safetensors_weights_iterator,
)
from vllm.model_executor.models.exaone4 import Exaone4ForCausalLM


def _infer_layout(sample_keys: list[str]) -> str:
    if any(key.endswith(".qweight") for key in sample_keys):
        return "gptq_like"
    if any("weight_packed" in key for key in sample_keys):
        return "packed_weight_like"
    if any("weight_scale" in key for key in sample_keys) and any(
        "input_scale" in key for key in sample_keys
    ):
        return "omni_activation_real_like"
    return "unknown"


def _checkpoint_file_summary(checkpoint_dir: Path) -> dict[str, dict[str, Any]]:
    file_names = [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "omni_act_quant_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    ]
    summary: dict[str, dict[str, Any]] = {}
    for name in file_names:
        path = checkpoint_dir / name
        summary[name] = {
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else None,
        }
    return summary


def _inspect_safetensors(path: Path) -> dict[str, Any]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        sample_keys = keys[:32]
        return {
            "valid": True,
            "num_keys": len(keys),
            "sample_keys": sample_keys,
            "inferred_layout": _infer_layout(sample_keys),
            "has_qweight": any(key.endswith(".qweight") for key in keys),
            "has_weight_scale": any("weight_scale" in key for key in keys),
            "has_input_scale": any("input_scale" in key for key in keys),
        }
    except Exception as exc:
        return {
            "valid": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _build_model_config(
    checkpoint_dir: Path,
    *,
    quantization: str | None,
) -> ModelConfig:
    return ModelConfig(
        model=str(checkpoint_dir),
        tokenizer=str(checkpoint_dir),
        runner="generate",
        dtype="float16",
        enforce_eager=True,
        quantization=quantization,
    )


def _inspect_layer0(model: Exaone4ForCausalLM) -> dict[str, Any]:
    layer0 = model.model.layers[0]
    self_attn = layer0.self_attn
    mlp = layer0.mlp
    qkv_qweight = getattr(self_attn.qkv_proj, "qweight", None)
    gate_up_qweight = getattr(mlp.gate_up_proj, "qweight", None)
    o_proj_qweight = getattr(self_attn.o_proj, "qweight", None)
    down_proj_qweight = getattr(mlp.down_proj, "qweight", None)
    qkv_omni_weight = getattr(self_attn.qkv_proj, "omni_weight", None)
    gate_up_omni_weight = getattr(mlp.gate_up_proj, "omni_weight", None)
    o_proj_omni_weight = getattr(self_attn.o_proj, "omni_weight", None)
    down_proj_omni_weight = getattr(mlp.down_proj, "omni_weight", None)
    return {
        "layer0_type": layer0.__class__.__name__,
        "qkv_quant_method": self_attn.qkv_proj.quant_method.__class__.__name__,
        "o_proj_quant_method": self_attn.o_proj.quant_method.__class__.__name__,
        "gate_up_quant_method": mlp.gate_up_proj.quant_method.__class__.__name__,
        "down_proj_quant_method": mlp.down_proj.quant_method.__class__.__name__,
        "attention_helper_attached": hasattr(self_attn.attn, "omni_attention_helper"),
        "qkv_qweight_shape": list(qkv_qweight.shape) if qkv_qweight is not None else None,
        "gate_up_qweight_shape": (
            list(gate_up_qweight.shape) if gate_up_qweight is not None else None
        ),
        "o_proj_qweight_shape": (
            list(o_proj_qweight.shape) if o_proj_qweight is not None else None
        ),
        "down_proj_qweight_shape": (
            list(down_proj_qweight.shape) if down_proj_qweight is not None else None
        ),
        "qkv_omni_weight_shape": (
            list(qkv_omni_weight.shape) if qkv_omni_weight is not None else None
        ),
        "gate_up_omni_weight_shape": (
            list(gate_up_omni_weight.shape)
            if gate_up_omni_weight is not None
            else None
        ),
        "o_proj_omni_weight_shape": (
            list(o_proj_omni_weight.shape) if o_proj_omni_weight is not None else None
        ),
        "down_proj_omni_weight_shape": (
            list(down_proj_omni_weight.shape)
            if down_proj_omni_weight is not None
            else None
        ),
    }


def _run_forward(model: Exaone4ForCausalLM) -> dict[str, Any]:
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    positions = torch.tensor([0, 1, 2], dtype=torch.long)
    output = model(input_ids=input_ids, positions=positions, intermediate_tensors=None)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(output)}")
    return {
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
    }


def _init_single_process_model_parallel() -> str:
    tmp = tempfile.NamedTemporaryFile(prefix="vllm-omni-act-dist-", delete=False)
    tmp.close()
    init_method = f"file://{tmp.name}"
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=init_method,
        backend="gloo",
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        prefill_context_model_parallel_size=1,
        decode_context_model_parallel_size=1,
        backend="gloo",
    )
    return tmp.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    summary: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "repo_root": str(REPO_ROOT),
        "checkpoint_dir": str(checkpoint_dir),
        "last_success": None,
        "last_failure": None,
        "status": "started",
    }

    quant_cls = get_quantization_config("omni_activation_real")
    summary["registration"] = {
        "quant_config_class": quant_cls.__name__,
    }
    summary["last_success"] = "registration"

    summary["files"] = _checkpoint_file_summary(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        summary["status"] = "failed"
        summary["last_failure"] = "missing_config_json"
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)

    with open(config_path) as f:
        config_dict = json.load(f)
    summary["config"] = {
        "architectures": config_dict.get("architectures"),
        "model_type": config_dict.get("model_type"),
        "quantization_config": config_dict.get("quantization_config"),
        "torch_dtype": config_dict.get("torch_dtype"),
        "num_hidden_layers": config_dict.get("num_hidden_layers"),
        "num_attention_heads": config_dict.get("num_attention_heads"),
        "num_key_value_heads": config_dict.get("num_key_value_heads"),
    }
    summary["last_success"] = "config_json_read"

    weight_path = checkpoint_dir / "model.safetensors"
    if weight_path.exists():
        summary["safetensors"] = _inspect_safetensors(weight_path)
        if summary["safetensors"]["valid"]:
            summary["last_success"] = "safetensors_readable"
    else:
        summary["safetensors"] = {"valid": False, "error": "model.safetensors missing"}

    hf_quant_cfg = config_dict.get("quantization_config")
    if hf_quant_cfg is None:
        detected_quant = None
    else:
        detected_quant = quant_cls.override_quantization_method(
            hf_quant_cfg,
            None,
        )
    summary["detected_quantization"] = detected_quant

    try:
        generic_model_config = _build_model_config(checkpoint_dir, quantization=None)
        summary["generic_model_config"] = {
            "resolved_quantization": generic_model_config.quantization,
            "dtype": str(generic_model_config.dtype),
            "max_model_len": generic_model_config.max_model_len,
        }
        summary["last_success"] = "generic_model_config_created"
    except Exception as exc:
        summary["status"] = "failed"
        summary["last_failure"] = f"generic_model_config_failed:{type(exc).__name__}"
        summary["error"] = str(exc)
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)

    try:
        forced_model_config = _build_model_config(
            checkpoint_dir,
            quantization="omni_activation_real",
        )
        summary["forced_model_config"] = {
            "resolved_quantization": forced_model_config.quantization,
            "dtype": str(forced_model_config.dtype),
        }
        summary["last_success"] = "forced_model_config_created"
    except Exception as exc:
        summary["status"] = "failed"
        summary["last_failure"] = f"forced_model_config_failed:{type(exc).__name__}"
        summary["error"] = str(exc)
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)

    load_config = LoadConfig()
    try:
        quant_config = get_quant_config(forced_model_config, load_config)
        summary["quant_config"] = {
            "name": quant_config.get_name(),
            "config_source": getattr(quant_config, "config_source", None),
            "checkpoint_metadata": getattr(quant_config, "checkpoint_metadata", None),
        }
        summary["last_success"] = "omni_quant_config_parsed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["last_failure"] = f"omni_quant_config_parse_failed:{type(exc).__name__}"
        summary["error"] = str(exc)
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)

    vllm_config = VllmConfig(
        model_config=forced_model_config,
        load_config=load_config,
        quant_config=quant_config,
    )

    dist_init_file = None
    try:
        with set_current_vllm_config(vllm_config):
            dist_init_file = _init_single_process_model_parallel()
            model = Exaone4ForCausalLM(vllm_config=vllm_config)
        summary["last_success"] = "model_initialized"
    except Exception as exc:
        summary["status"] = "failed"
        summary["last_failure"] = f"model_init_failed:{type(exc).__name__}"
        summary["error"] = str(exc)
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)

    try:
        weights_iter = safetensors_weights_iterator(
            [str(weight_path)],
            use_tqdm_on_load=False,
        )
        loaded = model.load_weights(weights_iter)
        summary["weight_load"] = {
            "loaded_param_count": len(loaded),
        }
        process_weights_after_loading(model, forced_model_config, torch.device("cpu"))
        summary["last_success"] = "weights_loaded_and_processed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["last_failure"] = f"weight_load_failed:{type(exc).__name__}"
        summary["error"] = str(exc)
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)

    summary["layer0"] = _inspect_layer0(model)
    summary["last_success"] = "layer0_attach_checked"

    try:
        summary["forward"] = _run_forward(model)
        summary["last_success"] = "forward_succeeded"
        summary["status"] = "ok"
    except Exception as exc:
        summary["status"] = "failed"
        summary["last_failure"] = f"forward_failed:{type(exc).__name__}"
        summary["error"] = str(exc)
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)
    finally:
        cleanup_dist_env_and_memory()
        if dist_init_file is not None:
            try:
                os.unlink(dist_init_file)
            except FileNotFoundError:
                pass

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
