# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Smoke test for EXAONE + omni activation real Python-only integration.

This script validates the pre-build path that is expected to remain useful
for wheel packaging smoke tests:

1. import + auto-registration
2. config parsing from ``config.json`` + ``omni_act_quant_config.json``
3. fused EXAONE-style shard loading for QKV and gate/up projections
4. custom quant attach on vLLM linear layers
5. correctness-first layer forward path
6. attention helper path without a vLLM forward context

Run:
    python3 examples/offline_inference/exaone4_omni_activation_real_smoke.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    LoadConfig,
    set_current_vllm_config,
)
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackend,
    CPUAttentionMetadata,
)

try:
    from transformers import Exaone4Config
except ImportError as exc:  # pragma: no cover - smoke script only
    raise SystemExit(
        "transformers with Exaone4Config support is required to run this smoke test."
    ) from exc


def _load_get_quant_config():
    weight_utils_path = REPO_ROOT / "vllm/model_executor/model_loader/weight_utils.py"
    spec = importlib.util.spec_from_file_location(
        "_exaone4_omni_act_weight_utils_smoke",
        weight_utils_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_quant_config


get_quant_config = _load_get_quant_config()


def _fake_quantize_input(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.round(x / scale) * scale


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs - rhs).abs().max().item())


def _build_model_dir() -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="exaone4-omni-act-"))
    hf_config = Exaone4Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    ).to_dict()
    hf_config["architectures"] = ["Exaone4ForCausalLM"]
    hf_config["quantization_config"] = {
        "quant_method": "omni_activation_real",
        "attention": {
            "enabled": True,
            "reference_without_forward_context": True,
            "qk_scale": 1.125,
            "pv_scale": 0.875,
        },
    }

    omni_quant_config = {
        "global_quant_config": {
            "enabled": True,
            "weight_dtype": "int8",
            "weight_scale_dtype": "float32",
            "input_scale_dtype": "float32",
            "quantize_inputs": True,
            "weight_scale_granularity": "per_tensor",
            "input_scale_granularity": "per_tensor",
            "tensor_names": {
                "weight": "weight",
                "weight_scale": "weight_scale",
                "input_scale": "input_scale",
            },
        },
        "attention": {
            "enabled": True,
            "reference_without_forward_context": True,
            "qk_scale": 1.125,
            "pv_scale": 0.875,
        },
        "checkpoint_metadata": {
            "format": "exaone-omni-act-real",
            "layout": "separate-proj",
        },
        "modules_to_not_convert": [],
    }

    (tmpdir / "config.json").write_text(json.dumps(hf_config, indent=2))
    (tmpdir / "omni_act_quant_config.json").write_text(
        json.dumps(omni_quant_config, indent=2)
    )
    return tmpdir


def _build_quant_config(model_dir: Path):
    hf_dict = json.loads((model_dir / "config.json").read_text())
    hf_config = Exaone4Config.from_dict(hf_dict)
    hf_quant_config = hf_dict["quantization_config"]
    quant_cls = get_quantization_config("omni_activation_real")
    detected_quant = quant_cls.override_quantization_method(hf_quant_config, None)
    assert detected_quant == "omni_activation_real"

    model_config = SimpleNamespace(
        model=str(model_dir),
        revision=None,
        quantization=detected_quant,
        hf_config=hf_config,
        hf_overrides={},
    )
    quant_config = get_quant_config(model_config, LoadConfig())
    assert quant_config.get_name() == "omni_activation_real"
    assert quant_config.config_source == "config.json+omni_act_quant_config.json"
    assert quant_config.checkpoint_metadata == {
        "format": "exaone-omni-act-real",
        "layout": "separate-proj",
    }
    return quant_config


def _verify_registration() -> dict[str, object]:
    quant_cls = get_quantization_config("omni_activation_real")
    assert quant_cls.__name__ == "OmniActivationRealConfig"
    return {
        "quant_config_class": quant_cls.__name__,
    }


def _load_fused_qkv(qkv: QKVParallelLinear) -> tuple[torch.Tensor, torch.Tensor]:
    q_weight = (
        torch.arange(1, 1 + 64 * 64, dtype=torch.int32).reshape(64, 64) % 11 - 5
    ).contiguous()
    k_weight = (
        torch.arange(1, 1 + 64 * 64, dtype=torch.int32).reshape(64, 64) % 7 - 3
    ).contiguous()
    v_weight = (
        torch.arange(1, 1 + 64 * 64, dtype=torch.int32).reshape(64, 64) % 13 - 6
    ).contiguous()
    q_weight = q_weight.to(torch.int8)
    k_weight = k_weight.to(torch.int8)
    v_weight = v_weight.to(torch.int8)

    qkv.weight_loader(qkv.weight, q_weight, "q")
    qkv.weight_loader(qkv.weight, k_weight, "k")
    qkv.weight_loader(qkv.weight, v_weight, "v")
    qkv.weight_loader(qkv.weight_scale, torch.tensor(0.25), "q")
    qkv.weight_loader(qkv.weight_scale, torch.tensor(0.50), "k")
    qkv.weight_loader(qkv.weight_scale, torch.tensor(0.75), "v")
    qkv.weight_loader(qkv.input_scale, torch.tensor(0.20), "q")
    qkv.weight_loader(qkv.input_scale, torch.tensor(0.40), "k")
    qkv.weight_loader(qkv.input_scale, torch.tensor(0.60), "v")
    qkv.quant_method.process_weights_after_loading(qkv)

    dense_q = q_weight.float() * 0.25
    dense_k = k_weight.float() * 0.50
    dense_v = v_weight.float() * 0.75
    dense_weight = torch.cat([dense_q, dense_k, dense_v], dim=0)
    input_scales = torch.tensor([0.20, 0.40, 0.60], dtype=torch.float32)
    return dense_weight, input_scales


def _verify_qkv(quant_config) -> dict[str, object]:
    qkv = QKVParallelLinear(
        hidden_size=64,
        head_size=32,
        total_num_heads=2,
        total_num_kv_heads=2,
        bias=False,
        params_dtype=torch.float32,
        quant_config=quant_config,
        prefix="model.layers.0.self_attn.qkv_proj",
        disable_tp=True,
    )
    dense_weight, input_scales = _load_fused_qkv(qkv)
    assert qkv.omni_weight.shape == dense_weight.shape
    assert len(qkv.omni_shard_runtimes) == 3

    x = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float32).reshape(1, 64)
    y, _ = qkv(x)
    reference = torch.cat(
        [
            F.linear(_fake_quantize_input(x, scale.item()), shard)
            for scale, shard in zip(
                input_scales,
                torch.split(dense_weight, [64, 64, 64], dim=0),
                strict=True,
            )
        ],
        dim=-1,
    )
    torch.testing.assert_close(y, reference)
    return {
        "attach_ok": qkv.quant_method.__class__.__name__ == "OmniActivationRealLinearMethod",
        "runtime_shards": len(qkv.omni_shard_runtimes),
        "weight_shape": list(qkv.omni_weight.shape),
        "output_shape": list(y.shape),
        "max_abs_diff": _max_abs_diff(y, reference),
    }


def _verify_gate_up_and_row(quant_config) -> dict[str, object]:
    merged = MergedColumnParallelLinear(
        input_size=64,
        output_sizes=[48, 80],
        bias=False,
        params_dtype=torch.float32,
        quant_config=quant_config,
        prefix="model.layers.0.mlp.gate_up_proj",
        disable_tp=True,
    )
    gate_weight = (
        torch.arange(48 * 64, dtype=torch.int32).reshape(48, 64) % 9 - 4
    ).contiguous()
    up_weight = (
        torch.arange(80 * 64, dtype=torch.int32).reshape(80, 64) % 5 - 2
    ).contiguous()
    gate_weight = gate_weight.to(torch.int8)
    up_weight = up_weight.to(torch.int8)
    merged.weight_loader(merged.weight, gate_weight, 0)
    merged.weight_loader(merged.weight, up_weight, 1)
    merged.weight_loader(merged.weight_scale, torch.tensor(0.30), 0)
    merged.weight_loader(merged.weight_scale, torch.tensor(0.45), 1)
    merged.weight_loader(merged.input_scale, torch.tensor(0.15), 0)
    merged.weight_loader(merged.input_scale, torch.tensor(0.35), 1)
    merged.quant_method.process_weights_after_loading(merged)

    x = torch.linspace(-0.5, 0.5, steps=128, dtype=torch.float32).reshape(2, 64)
    y, _ = merged(x)
    gate_ref = F.linear(_fake_quantize_input(x, 0.15), gate_weight.float() * 0.30)
    up_ref = F.linear(_fake_quantize_input(x, 0.35), up_weight.float() * 0.45)
    merged_reference = torch.cat([gate_ref, up_ref], dim=-1)
    torch.testing.assert_close(y, merged_reference)

    row = RowParallelLinear(
        input_size=128,
        output_size=64,
        bias=False,
        params_dtype=torch.float32,
        quant_config=quant_config,
        prefix="model.layers.0.mlp.down_proj",
        disable_tp=True,
    )
    row_weight = (
        torch.arange(64 * 128, dtype=torch.int32).reshape(64, 128) % 17 - 8
    ).contiguous()
    row_weight = row_weight.to(torch.int8)
    row.weight_loader(row.weight, row_weight)
    row.weight_loader(row.weight_scale, torch.tensor(0.125))
    row.weight_loader(row.input_scale, torch.tensor(0.20))
    row.quant_method.process_weights_after_loading(row)

    row_in = torch.linspace(-1.0, 1.0, steps=256, dtype=torch.float32).reshape(2, 128)
    row_out, _ = row(row_in)
    row_ref = F.linear(_fake_quantize_input(row_in, 0.20), row_weight.float() * 0.125)
    torch.testing.assert_close(row_out, row_ref)
    return {
        "merged_attach_ok": (
            merged.quant_method.__class__.__name__ == "OmniActivationRealLinearMethod"
        ),
        "merged_runtime_shards": len(merged.omni_shard_runtimes),
        "merged_weight_shape": list(merged.omni_weight.shape),
        "merged_output_shape": list(y.shape),
        "merged_max_abs_diff": _max_abs_diff(y, merged_reference),
        "row_attach_ok": row.quant_method.__class__.__name__ == "OmniActivationRealLinearMethod",
        "row_runtime_shards": len(row.omni_shard_runtimes),
        "row_weight_shape": list(row.omni_weight.shape),
        "row_output_shape": list(row_out.shape),
        "row_max_abs_diff": _max_abs_diff(row_out, row_ref),
    }


def _attention_config_shim(block_size: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
            model="exaone4-omni-activation-real-smoke",
        ),
        cache_config=CacheConfig(block_size=block_size, enable_prefix_caching=False),
        attention_config=AttentionConfig(),
        compilation_config=CompilationConfig(),
    )


def _verify_attention_helper(quant_config) -> dict[str, object]:
    shim = _attention_config_shim()
    with set_current_vllm_config(shim, prefix="model.layers.0.self_attn.attn"):
        attention = Attention(
            num_heads=2,
            head_size=32,
            scale=1 / math.sqrt(32),
            num_kv_heads=2,
            cache_config=None,
            quant_config=quant_config,
            prefix="model.layers.0.self_attn.attn",
            attn_backend=CPUAttentionBackend,
        )

    assert attention.quant_method is not None
    attention.quant_method.process_weights_after_loading(attention)
    attention.process_weights_after_loading(torch.float32)
    assert attention.omni_attention_helper is not None

    query = torch.linspace(-0.25, 0.25, steps=192, dtype=torch.float32).reshape(3, 64)
    key = torch.linspace(-0.50, 0.50, steps=192, dtype=torch.float32).reshape(3, 64)
    value = torch.linspace(0.10, 0.70, steps=192, dtype=torch.float32).reshape(3, 64)
    output = attention(query, key, value)

    q = query.view(3, 2, 32).transpose(0, 1)
    k = key.view(3, 2, 32).transpose(0, 1)
    v = value.view(3, 2, 32).transpose(0, 1)
    logits = torch.matmul(q, k.transpose(-2, -1)) * ((1 / math.sqrt(32)) * 1.125)
    causal_mask = torch.triu(
        torch.ones(3, 3, dtype=torch.bool),
        diagonal=1,
    )
    logits = logits.masked_fill(causal_mask, torch.finfo(logits.dtype).min)
    probs = logits.softmax(dim=-1)
    reference = (torch.matmul(probs, v) * 0.875).transpose(0, 1).reshape(3, 64)
    torch.testing.assert_close(output, reference)
    return {
        "helper_type": attention.omni_attention_helper.__class__.__name__,
        "output_shape": list(output.shape),
        "max_abs_diff": _max_abs_diff(output, reference),
        "qkt_runtime_calls": 1,
        "pv_runtime_calls": 1,
    }


def _run_attention_with_context(
    *,
    layer_name: str,
    attention: Attention,
    metadata: CPUAttentionMetadata,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    forward_context = ForwardContext(
        no_compile_layers={layer_name: attention},
        attn_metadata={layer_name: metadata},
        slot_mapping={layer_name: metadata.slot_mapping},
    )
    with override_forward_context(forward_context):
        return attention(query, key, value)


def _reference_suffix_attention(
    attention: Attention,
    full_query: torch.Tensor,
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    *,
    query_len: int,
) -> torch.Tensor:
    full_seq_len = full_query.shape[0]
    assert 0 < query_len <= full_seq_len
    q = full_query[-query_len:].view(query_len, attention.num_heads, attention.head_size).transpose(
        0, 1
    )
    k = full_key.view(full_seq_len, attention.num_kv_heads, attention.head_size).transpose(0, 1)
    v = full_value.view(
        full_seq_len,
        attention.num_kv_heads,
        attention.head_size_v,
    ).transpose(0, 1)
    logits = torch.matmul(q, k.transpose(-2, -1)) * (
        (1 / math.sqrt(attention.head_size)) * 1.125
    )
    key_positions = torch.arange(full_seq_len, dtype=torch.int64)
    query_positions = torch.arange(
        full_seq_len - query_len,
        full_seq_len,
        dtype=torch.int64,
    )
    logits = logits.masked_fill(
        (key_positions.unsqueeze(0) > query_positions.unsqueeze(1)).unsqueeze(0),
        torch.finfo(logits.dtype).min,
    )
    probs = logits.softmax(dim=-1)
    return (torch.matmul(probs, v) * 0.875).transpose(0, 1).reshape(query_len, -1)


def _verify_attention_engine_helper(quant_config) -> dict[str, object]:
    shim = _attention_config_shim()
    layer_name = "model.layers.0.self_attn.attn.engine"
    with set_current_vllm_config(shim, prefix=layer_name):
        attention = Attention(
            num_heads=2,
            head_size=32,
            scale=1 / math.sqrt(32),
            num_kv_heads=2,
            cache_config=None,
            quant_config=quant_config,
            prefix=layer_name,
            attn_backend=CPUAttentionBackend,
        )

    assert attention.quant_method is not None
    attention.quant_method.process_weights_after_loading(attention)
    attention.process_weights_after_loading(torch.float32)

    query = torch.linspace(-0.5, 0.5, steps=320, dtype=torch.float32).reshape(5, 64)
    key = torch.linspace(-0.75, 0.75, steps=320, dtype=torch.float32).reshape(5, 64)
    value = torch.linspace(0.20, 0.80, steps=320, dtype=torch.float32).reshape(5, 64)
    metadata = CPUAttentionMetadata(
        isa="unit-test",
        num_actual_tokens=5,
        max_query_len=3,
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        max_seq_len=3,
        seq_lens=torch.tensor([2, 3], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 16, 17, 18], dtype=torch.int32),
        scheduler_metadata=None,
    )
    output = _run_attention_with_context(
        layer_name=layer_name,
        attention=attention,
        metadata=metadata,
        query=query,
        key=key,
        value=value,
    )

    q0, q1 = query[:2], query[2:]
    k0, k1 = key[:2], key[2:]
    v0, v1 = value[:2], value[2:]
    ref0 = _reference_suffix_attention(
        attention,
        q0,
        k0,
        v0,
        query_len=2,
    )
    ref1 = _reference_suffix_attention(
        attention,
        q1,
        k1,
        v1,
        query_len=3,
    )
    reference = torch.cat([ref0, ref1], dim=0)

    assert attention.omni_last_attention_engine_status == "forward_context_reference_prefill"
    assert attention.omni_last_attention_metadata_summary["has_attn_metadata"]
    assert attention.omni_last_attention_metadata_summary["has_block_table"]
    assert attention.omni_last_attention_metadata_summary["has_slot_mapping"]
    assert attention.omni_last_attention_metadata_summary["kv_cache_shape"] == tuple(
        attention.kv_cache.shape
    )
    torch.testing.assert_close(output, reference)
    return {
        "engine_status": attention.omni_last_attention_engine_status,
        "cache_backend": attention.omni_last_attention_cache_backend,
        "cache_layout_compatible": bool(attention.omni_last_attention_cache_layout_compatible),
        "cache_write_calls": int(attention.omni_last_attention_cache_write_calls),
        "cache_read_calls": int(attention.omni_last_attention_cache_read_calls),
        "qkt_runtime_calls": int(attention.omni_last_attention_qkt_runtime_calls),
        "pv_runtime_calls": int(attention.omni_last_attention_pv_runtime_calls),
        "output_shape": list(output.shape),
        "max_abs_diff": _max_abs_diff(output, reference),
        "query_start_loc": metadata.query_start_loc.tolist(),
        "seq_lens": metadata.seq_lens.tolist(),
        "slot_mapping": metadata.slot_mapping.tolist(),
        "block_table_shape": list(metadata.block_table.shape),
    }


def _verify_attention_decode_engine_helper(quant_config) -> dict[str, object]:
    block_size = 2
    shim = _attention_config_shim(block_size=block_size)
    layer_name = "model.layers.0.self_attn.attn.decode"
    with set_current_vllm_config(shim, prefix=layer_name):
        attention = Attention(
            num_heads=2,
            head_size=32,
            scale=1 / math.sqrt(32),
            num_kv_heads=2,
            cache_config=None,
            quant_config=quant_config,
            prefix=layer_name,
            attn_backend=CPUAttentionBackend,
        )

    assert attention.quant_method is not None
    attention.quant_method.process_weights_after_loading(attention)
    attention.process_weights_after_loading(torch.float32)
    kv_cache_shape = CPUAttentionBackend.get_kv_cache_shape(
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=attention.num_kv_heads,
        head_size=attention.head_size,
    )
    attention.kv_cache = torch.zeros(kv_cache_shape, dtype=torch.float32)

    prompt_query = torch.linspace(-0.60, 0.60, steps=192, dtype=torch.float32).reshape(3, 64)
    prompt_key = torch.linspace(-0.90, 0.90, steps=192, dtype=torch.float32).reshape(3, 64)
    prompt_value = torch.linspace(0.05, 0.65, steps=192, dtype=torch.float32).reshape(3, 64)
    prompt_metadata = CPUAttentionMetadata(
        isa="decode-prefill",
        num_actual_tokens=3,
        max_query_len=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        max_seq_len=3,
        seq_lens=torch.tensor([3], dtype=torch.int32),
        block_table=torch.tensor([[0, 1, -1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
        scheduler_metadata=None,
    )
    prompt_output = _run_attention_with_context(
        layer_name=layer_name,
        attention=attention,
        metadata=prompt_metadata,
        query=prompt_query,
        key=prompt_key,
        value=prompt_value,
    )
    prompt_reference = _reference_suffix_attention(
        attention,
        prompt_query,
        prompt_key,
        prompt_value,
        query_len=3,
    )
    torch.testing.assert_close(prompt_output, prompt_reference)
    assert attention.omni_last_attention_engine_status == "forward_context_reference_prefill"
    assert attention.omni_last_attention_cache_backend == "physical"
    assert bool(attention.omni_last_attention_cache_layout_compatible)
    assert int(attention.omni_last_attention_cache_write_calls) == 3
    assert int(attention.omni_last_attention_cache_read_calls) == 1
    assert int(attention.omni_last_attention_qkt_runtime_calls) == 1
    assert int(attention.omni_last_attention_pv_runtime_calls) == 1

    decode1_query = torch.linspace(-0.15, 0.15, steps=64, dtype=torch.float32).reshape(1, 64)
    decode1_key = torch.linspace(-0.30, 0.30, steps=64, dtype=torch.float32).reshape(1, 64)
    decode1_value = torch.linspace(0.70, 1.10, steps=64, dtype=torch.float32).reshape(1, 64)
    decode1_metadata = CPUAttentionMetadata(
        isa="decode-step-1",
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=4,
        seq_lens=torch.tensor([4], dtype=torch.int32),
        block_table=torch.tensor([[0, 1, -1]], dtype=torch.int32),
        slot_mapping=torch.tensor([3], dtype=torch.int32),
        scheduler_metadata=None,
    )
    decode1_output = _run_attention_with_context(
        layer_name=layer_name,
        attention=attention,
        metadata=decode1_metadata,
        query=decode1_query,
        key=decode1_key,
        value=decode1_value,
    )
    full_query_step1 = torch.cat([prompt_query, decode1_query], dim=0)
    full_key_step1 = torch.cat([prompt_key, decode1_key], dim=0)
    full_value_step1 = torch.cat([prompt_value, decode1_value], dim=0)
    decode1_reference = _reference_suffix_attention(
        attention,
        full_query_step1,
        full_key_step1,
        full_value_step1,
        query_len=1,
    )
    torch.testing.assert_close(decode1_output, decode1_reference)
    decode1_summary = {
        "engine_status": attention.omni_last_attention_engine_status,
        "cache_backend": attention.omni_last_attention_cache_backend,
        "cache_layout_compatible": bool(attention.omni_last_attention_cache_layout_compatible),
        "cache_write_calls": int(attention.omni_last_attention_cache_write_calls),
        "cache_read_calls": int(attention.omni_last_attention_cache_read_calls),
        "qkt_runtime_calls": int(attention.omni_last_attention_qkt_runtime_calls),
        "pv_runtime_calls": int(attention.omni_last_attention_pv_runtime_calls),
        "slot_mapping": decode1_metadata.slot_mapping.tolist(),
        "block_table": decode1_metadata.block_table.tolist(),
        "output_shape": list(decode1_output.shape),
        "max_abs_diff": _max_abs_diff(decode1_output, decode1_reference),
    }
    assert decode1_summary["engine_status"] == "forward_context_reference_decode"
    assert decode1_summary["cache_backend"] == "physical"
    assert decode1_summary["cache_write_calls"] == 1
    assert decode1_summary["cache_read_calls"] == 1
    assert decode1_summary["qkt_runtime_calls"] == 1
    assert decode1_summary["pv_runtime_calls"] == 1

    decode2_query = torch.linspace(-0.05, 0.25, steps=64, dtype=torch.float32).reshape(1, 64)
    decode2_key = torch.linspace(-0.10, 0.40, steps=64, dtype=torch.float32).reshape(1, 64)
    decode2_value = torch.linspace(1.20, 1.60, steps=64, dtype=torch.float32).reshape(1, 64)
    decode2_metadata = CPUAttentionMetadata(
        isa="decode-step-2",
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=5,
        seq_lens=torch.tensor([5], dtype=torch.int32),
        block_table=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        slot_mapping=torch.tensor([4], dtype=torch.int32),
        scheduler_metadata=None,
    )
    decode2_output = _run_attention_with_context(
        layer_name=layer_name,
        attention=attention,
        metadata=decode2_metadata,
        query=decode2_query,
        key=decode2_key,
        value=decode2_value,
    )
    full_query_step2 = torch.cat([full_query_step1, decode2_query], dim=0)
    full_key_step2 = torch.cat([full_key_step1, decode2_key], dim=0)
    full_value_step2 = torch.cat([full_value_step1, decode2_value], dim=0)
    decode2_reference = _reference_suffix_attention(
        attention,
        full_query_step2,
        full_key_step2,
        full_value_step2,
        query_len=1,
    )
    torch.testing.assert_close(decode2_output, decode2_reference)
    decode2_summary = {
        "engine_status": attention.omni_last_attention_engine_status,
        "cache_backend": attention.omni_last_attention_cache_backend,
        "cache_layout_compatible": bool(attention.omni_last_attention_cache_layout_compatible),
        "cache_write_calls": int(attention.omni_last_attention_cache_write_calls),
        "cache_read_calls": int(attention.omni_last_attention_cache_read_calls),
        "qkt_runtime_calls": int(attention.omni_last_attention_qkt_runtime_calls),
        "pv_runtime_calls": int(attention.omni_last_attention_pv_runtime_calls),
        "slot_mapping": decode2_metadata.slot_mapping.tolist(),
        "block_table": decode2_metadata.block_table.tolist(),
        "output_shape": list(decode2_output.shape),
        "max_abs_diff": _max_abs_diff(decode2_output, decode2_reference),
    }
    assert decode2_summary["engine_status"] == "forward_context_reference_decode"
    assert decode2_summary["cache_backend"] == "physical"
    assert decode2_summary["cache_write_calls"] == 1
    assert decode2_summary["cache_read_calls"] == 1
    assert decode2_summary["qkt_runtime_calls"] == 1
    assert decode2_summary["pv_runtime_calls"] == 1
    assert torch.count_nonzero(attention.kv_cache).item() > 0

    return {
        "prefill": {
            "engine_status": "forward_context_reference_prefill",
            "cache_backend": "physical",
            "cache_layout_compatible": True,
            "cache_write_calls": 3,
            "cache_read_calls": 1,
            "qkt_runtime_calls": 1,
            "pv_runtime_calls": 1,
            "output_shape": list(prompt_output.shape),
            "max_abs_diff": _max_abs_diff(prompt_output, prompt_reference),
        },
        "first_decode": decode1_summary,
        "second_decode": decode2_summary,
        "kv_cache_shape": list(attention.kv_cache.shape),
        "kv_cache_nonzero": int(torch.count_nonzero(attention.kv_cache).item()),
    }


def main() -> None:
    summary: dict[str, object] = {}
    summary["registration"] = _verify_registration()
    model_dir = _build_model_dir()
    quant_config = _build_quant_config(model_dir)
    summary["config_source"] = quant_config.config_source
    summary["checkpoint_metadata"] = quant_config.checkpoint_metadata
    summary["qkv"] = _verify_qkv(quant_config)
    summary["merged_and_row"] = _verify_gate_up_and_row(quant_config)
    summary["attention_no_forward_context"] = _verify_attention_helper(quant_config)
    summary["attention_engine_prefill"] = _verify_attention_engine_helper(quant_config)
    summary["attention_engine_decode"] = _verify_attention_decode_engine_helper(quant_config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("omni_activation_real smoke test passed")


if __name__ == "__main__":
    main()
