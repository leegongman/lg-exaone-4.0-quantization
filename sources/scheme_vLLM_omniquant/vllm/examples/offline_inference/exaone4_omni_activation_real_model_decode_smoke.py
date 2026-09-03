# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal decode smoke for EXAONE omni activation real checkpoints.

This script reuses the real checkpoint/model loading path and validates the
engine-style attention correctness branch on an actual checkpoint-loaded
layer-0 attention module:

1. model init
2. weight load + process_weights_after_loading
3. prefill with forward_context / attn_metadata
4. first decode step
5. second decode step

The goal is not full generation. It is a correctness-first smoke that proves
the minimal decode path on a real model instance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_CACHE_ROOT", "/tmp/vllm-cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from vllm.config import LoadConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import (
    get_quant_config,
    safetensors_weights_iterator,
)
from vllm.model_executor.models.exaone4 import Exaone4ForCausalLM
from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackend,
    CPUAttentionMetadata,
)


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs - rhs).abs().max().item())


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


def _init_single_process_model_parallel() -> str:
    tmp = tempfile.NamedTemporaryFile(prefix="vllm-omni-act-decode-dist-", delete=False)
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


def _load_model(
    checkpoint_dir: Path,
) -> tuple[Exaone4ForCausalLM, ModelConfig, dict[str, Any], str]:
    quant_cls = get_quantization_config("omni_activation_real")
    forced_model_config = _build_model_config(
        checkpoint_dir,
        quantization="omni_activation_real",
    )
    load_config = LoadConfig()
    quant_config = get_quant_config(forced_model_config, load_config)
    vllm_config = VllmConfig(
        model_config=forced_model_config,
        load_config=load_config,
        quant_config=quant_config,
    )

    with set_current_vllm_config(vllm_config):
        dist_init_file = _init_single_process_model_parallel()
        model = Exaone4ForCausalLM(vllm_config=vllm_config)

    weights_iter = safetensors_weights_iterator(
        [str(checkpoint_dir / "model.safetensors")],
        use_tqdm_on_load=False,
    )
    loaded = model.load_weights(weights_iter)
    process_weights_after_loading(model, forced_model_config, torch.device("cpu"))
    return model, forced_model_config, {
        "quant_config_class": quant_cls.__name__,
        "resolved_quantization": forced_model_config.quantization,
        "loaded_param_count": len(loaded),
        "checkpoint_metadata": getattr(quant_config, "checkpoint_metadata", None),
        "config_source": getattr(quant_config, "config_source", None),
    }, dist_init_file


def _run_self_attn_with_context(
    self_attn,
    *,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    metadata: CPUAttentionMetadata,
) -> torch.Tensor:
    layer_name = self_attn.attn.layer_name
    forward_context = ForwardContext(
        no_compile_layers={layer_name: self_attn.attn},
        attn_metadata={layer_name: metadata},
        slot_mapping={layer_name: metadata.slot_mapping},
    )
    with override_forward_context(forward_context):
        return self_attn(positions=positions, hidden_states=hidden_states)


def _capture_attention_debug(attn) -> dict[str, Any]:
    return {
        "engine_status": getattr(attn, "omni_last_attention_engine_status", None),
        "cache_backend": getattr(attn, "omni_last_attention_cache_backend", None),
        "cache_layout_compatible": bool(
            getattr(attn, "omni_last_attention_cache_layout_compatible", False)
        ),
        "cache_write_calls": int(
            getattr(attn, "omni_last_attention_cache_write_calls", 0)
        ),
        "cache_read_calls": int(
            getattr(attn, "omni_last_attention_cache_read_calls", 0)
        ),
        "qkt_runtime_calls": int(
            getattr(attn, "omni_last_attention_qkt_runtime_calls", 0)
        ),
        "pv_runtime_calls": int(
            getattr(attn, "omni_last_attention_pv_runtime_calls", 0)
        ),
        "metadata_summary": getattr(attn, "omni_last_attention_metadata_summary", None),
        "kv_cache_shape": (
            list(attn.kv_cache.shape)
            if hasattr(attn, "kv_cache") and isinstance(attn.kv_cache, torch.Tensor)
            else None
        ),
    }


def _reference_self_attn(self_attn, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    return self_attn(positions=positions, hidden_states=hidden_states)


def _embed_hidden_states(model: Exaone4ForCausalLM, input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    hidden_states = model.model.embed_input_ids(input_ids)
    return hidden_states.to(dtype)


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

    dist_init_file = None
    try:
        model, forced_model_config, load_summary, dist_init_file = _load_model(
            checkpoint_dir
        )
        summary["load"] = load_summary
        summary["last_success"] = "weights_loaded_and_processed"

        layer0 = model.model.layers[0]
        self_attn = layer0.self_attn
        attn = self_attn.attn
        layer_name = attn.layer_name
        summary["layer0"] = {
            "layer_name": layer_name,
            "layer_type": layer0.__class__.__name__,
            "self_attn_type": self_attn.__class__.__name__,
            "attn_type": attn.__class__.__name__,
            "attention_helper_attached": hasattr(attn, "omni_attention_helper"),
        }
        summary["last_success"] = "layer0_ready"

        block_size = 2
        kv_cache_shape = CPUAttentionBackend.get_kv_cache_shape(
            num_blocks=4,
            block_size=block_size,
            num_kv_heads=attn.num_kv_heads,
            head_size=attn.head_size,
        )
        attn.kv_cache = torch.zeros(kv_cache_shape, dtype=torch.float32)
        summary["kv_cache_shape"] = list(attn.kv_cache.shape)

        hidden_dtype = self_attn.qkv_proj.omni_weight.dtype
        prompt_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        prompt_positions = torch.tensor([0, 1, 2], dtype=torch.long)
        prompt_hidden = _embed_hidden_states(model, prompt_ids, hidden_dtype)

        prompt_metadata = CPUAttentionMetadata(
            isa="decode-prefill-real-model",
            num_actual_tokens=3,
            max_query_len=3,
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            max_seq_len=3,
            seq_lens=torch.tensor([3], dtype=torch.int32),
            block_table=torch.tensor([[0, 1, -1]], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
            scheduler_metadata=None,
        )
        prompt_output = _run_self_attn_with_context(
            self_attn,
            positions=prompt_positions,
            hidden_states=prompt_hidden,
            metadata=prompt_metadata,
        )
        prompt_debug = _capture_attention_debug(attn)
        prompt_reference = _reference_self_attn(self_attn, prompt_positions, prompt_hidden)
        torch.testing.assert_close(prompt_output, prompt_reference)
        summary["prefill"] = {
            **prompt_debug,
            "slot_mapping": prompt_metadata.slot_mapping.tolist(),
            "block_table": prompt_metadata.block_table.tolist(),
            "output_shape": list(prompt_output.shape),
            "output_dtype": str(prompt_output.dtype),
            "max_abs_diff": _max_abs_diff(prompt_output, prompt_reference),
        }
        summary["last_success"] = "prefill_succeeded"

        decode1_ids = torch.tensor([4], dtype=torch.long)
        decode1_positions = torch.tensor([3], dtype=torch.long)
        decode1_hidden = _embed_hidden_states(model, decode1_ids, hidden_dtype)
        decode1_metadata = CPUAttentionMetadata(
            isa="decode-step-1-real-model",
            num_actual_tokens=1,
            max_query_len=1,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            max_seq_len=4,
            seq_lens=torch.tensor([4], dtype=torch.int32),
            block_table=torch.tensor([[0, 1, -1]], dtype=torch.int32),
            slot_mapping=torch.tensor([3], dtype=torch.int32),
            scheduler_metadata=None,
        )
        decode1_output = _run_self_attn_with_context(
            self_attn,
            positions=decode1_positions,
            hidden_states=decode1_hidden,
            metadata=decode1_metadata,
        )
        decode1_debug = _capture_attention_debug(attn)
        full_positions_1 = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        full_hidden_1 = _embed_hidden_states(
            model,
            torch.tensor([1, 2, 3, 4], dtype=torch.long),
            hidden_dtype,
        )
        decode1_reference = _reference_self_attn(self_attn, full_positions_1, full_hidden_1)[-1:]
        torch.testing.assert_close(decode1_output, decode1_reference)
        summary["first_decode"] = {
            **decode1_debug,
            "slot_mapping": decode1_metadata.slot_mapping.tolist(),
            "block_table": decode1_metadata.block_table.tolist(),
            "output_shape": list(decode1_output.shape),
            "output_dtype": str(decode1_output.dtype),
            "max_abs_diff": _max_abs_diff(decode1_output, decode1_reference),
        }
        summary["last_success"] = "first_decode_succeeded"

        decode2_ids = torch.tensor([5], dtype=torch.long)
        decode2_positions = torch.tensor([4], dtype=torch.long)
        decode2_hidden = _embed_hidden_states(model, decode2_ids, hidden_dtype)
        decode2_metadata = CPUAttentionMetadata(
            isa="decode-step-2-real-model",
            num_actual_tokens=1,
            max_query_len=1,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            max_seq_len=5,
            seq_lens=torch.tensor([5], dtype=torch.int32),
            block_table=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            slot_mapping=torch.tensor([4], dtype=torch.int32),
            scheduler_metadata=None,
        )
        decode2_output = _run_self_attn_with_context(
            self_attn,
            positions=decode2_positions,
            hidden_states=decode2_hidden,
            metadata=decode2_metadata,
        )
        decode2_debug = _capture_attention_debug(attn)
        full_positions_2 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        full_hidden_2 = _embed_hidden_states(
            model,
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            hidden_dtype,
        )
        decode2_reference = _reference_self_attn(self_attn, full_positions_2, full_hidden_2)[-1:]
        torch.testing.assert_close(decode2_output, decode2_reference)
        summary["second_decode"] = {
            **decode2_debug,
            "slot_mapping": decode2_metadata.slot_mapping.tolist(),
            "block_table": decode2_metadata.block_table.tolist(),
            "output_shape": list(decode2_output.shape),
            "output_dtype": str(decode2_output.dtype),
            "max_abs_diff": _max_abs_diff(decode2_output, decode2_reference),
        }
        summary["last_success"] = "second_decode_succeeded"
        summary["status"] = "ok"
    except Exception as exc:
        summary["status"] = "failed"
        summary["last_failure"] = f"{summary['last_success'] or 'startup'}:{type(exc).__name__}"
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
