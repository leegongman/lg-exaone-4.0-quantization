# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness-first omni activation real quantization support.

This module is intentionally Python-only. It focuses on:

1. Parsing omni activation real quantization configs from either
   ``config.json`` or ``omni_act_quant_config.json``.
2. Attaching a custom quant method to vLLM linear layers without any
   local-path assumptions.
3. Reusing vLLM's existing fused shard loaders for EXAONE packed modules.
4. Preserving a natural path to wheel packaging by auto-registering the
   quantization method on package import.

The initial implementation prioritizes correctness over performance. Quantized
weights are dequantized into dense tensors after loading, and activations are
optionally fake-quantized per shard during the forward pass to mimic runtime
rules used by the checkpoint export.
"""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.omni_cutlass_utils import (
    omni_w6a6_cutlass_linear,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    apply_w8a8_block_int8_linear,
)
from vllm.model_executor.layers.quantization.utils.omni_triton_utils import (
    omni_w6a6_linear,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_METHOD_NAME = "omni_activation_real"
_PACKED_4BIT_RUNTIME_FORMAT = "omni_activation_real_v2_packed_4bit"
_PACKED_6BIT_RUNTIME_FORMAT = "omni_activation_real_v2_packed_6bit"
_PACKED_UINT4X2 = "uint4x2"
_PACKED_UINT6X4 = "uint6x4"
_METHOD_ALIASES = frozenset(
    {
        _METHOD_NAME,
        "omni-act",
        "omni_act",
        "omni_activation",
        "omni_activation_real_quant",
    }
)
_RESERVED_TOP_LEVEL_KEYS = frozenset(
    {
        "quant_method",
        "format",
        "version",
        "global_quant_config",
        "linear",
        "attention",
        "layer_quant_config",
        "modules_to_not_convert",
        "modules_to_skip",
        "skip_modules",
        "skip_with_substr",
        "checkpoint_metadata",
        "config_source",
    }
)


def _normalize_method_name(name: Any) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip().lower()
    if normalized in _METHOD_ALIASES:
        return _METHOD_NAME
    return normalized


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(
                cast(dict[str, Any], merged[key]),
                cast(dict[str, Any], value),
            )
        else:
            merged[key] = value
    return merged


def _json_deep_equal(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    return json.dumps(lhs, sort_keys=True) == json.dumps(rhs, sort_keys=True)


def _json_value_equal(lhs: Any, rhs: Any) -> bool:
    return json.dumps(lhs, sort_keys=True) == json.dumps(rhs, sort_keys=True)


def _parse_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_list(config: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]
    return []


def _resolve_dtype(dtype: Any, default: torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return default

    dtype_name = str(dtype).strip().lower()
    dtype_map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported omni activation real dtype: {dtype}")
    return dtype_map[dtype_name]


def _metadata_granularity(config: dict[str, Any], key_prefix: str, default: str) -> str:
    value = config.get(f"{key_prefix}_granularity", config.get(f"{key_prefix}_mode"))
    if value is None:
        return default
    return str(value).strip().lower()


def _metadata_group_size(config: dict[str, Any], key_prefix: str) -> int | None:
    value = config.get(f"{key_prefix}_group_size", config.get("group_size"))
    if value in (None, "", -1):
        return None
    return int(value)


def _unpack_uint4x2(
    packed: torch.Tensor,
    *,
    signed: bool,
    unpacked_last_dim: int | None = None,
) -> torch.Tensor:
    packed_i16 = packed.to(torch.int16)
    unpacked = torch.empty(
        *packed.shape[:-1],
        packed.shape[-1] * 2,
        dtype=torch.int16,
        device=packed.device,
    )
    unpacked[..., ::2] = packed_i16 & 0x0F
    unpacked[..., 1::2] = (packed_i16 >> 4) & 0x0F
    if unpacked_last_dim is not None:
        unpacked = unpacked[..., :unpacked_last_dim]
    if signed:
        unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    return unpacked.contiguous()


def _unpack_uint6x4(
    packed: torch.Tensor,
    *,
    signed: bool,
    unpacked_last_dim: int | None = None,
) -> torch.Tensor:
    if packed.shape[-1] % 3 != 0:
        raise ValueError(
            f"Cannot unpack uint6 values with innermost size {packed.shape[-1]} "
            "that is not divisible by 3."
        )
    packed_i32 = packed.to(torch.int32)
    unpacked = torch.empty(
        *packed.shape[:-1],
        (packed.shape[-1] // 3) * 4,
        dtype=torch.int16,
        device=packed.device,
    )
    packed_view = packed_i32.view(*packed.shape[:-1], packed.shape[-1] // 3, 3)
    unpacked_view = unpacked.view(*packed.shape[:-1], packed.shape[-1] // 3, 4)
    b0 = packed_view[..., 0]
    b1 = packed_view[..., 1]
    b2 = packed_view[..., 2]
    unpacked_view[..., 0] = (b0 & 0x3F).to(torch.int16)
    unpacked_view[..., 1] = (((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)).to(
        torch.int16
    )
    unpacked_view[..., 2] = (((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)).to(
        torch.int16
    )
    unpacked_view[..., 3] = ((b2 >> 2) & 0x3F).to(torch.int16)
    if unpacked_last_dim is not None:
        unpacked = unpacked[..., :unpacked_last_dim]
    if signed:
        unpacked = torch.where(unpacked >= 32, unpacked - 64, unpacked)
    return unpacked.contiguous()


def _weight_packing_mode(
    runtime_format: str | None,
    layer_config: dict[str, Any],
) -> str:
    packing = str(layer_config.get("weight_packing", "")).strip().lower()
    if packing == _PACKED_UINT4X2:
        return _PACKED_UINT4X2
    if packing == _PACKED_UINT6X4:
        return _PACKED_UINT6X4

    normalized_runtime = str(runtime_format or "").strip().lower()
    weight_bits = int(layer_config.get("weight_bits", 0))
    if normalized_runtime == _PACKED_4BIT_RUNTIME_FORMAT and weight_bits == 4:
        return _PACKED_UINT4X2
    if normalized_runtime == _PACKED_6BIT_RUNTIME_FORMAT and weight_bits == 6:
        return _PACKED_UINT6X4
    return "plain"


def _expand_group_tensor(
    tensor: torch.Tensor,
    target_cols: int,
    group_size: int | None,
) -> torch.Tensor:
    if group_size is None:
        if tensor.shape[-1] == 1:
            return tensor.expand(*tensor.shape[:-1], target_cols)
        raise ValueError(
            "Group expansion requires `group_size` when the tensor is not "
            "already dense on the input dimension."
        )

    if tensor.shape[-1] * group_size != target_cols:
        raise ValueError(
            f"Cannot expand grouped tensor of shape {tuple(tensor.shape)} into "
            f"input width {target_cols} with group_size={group_size}."
        )
    return tensor.repeat_interleave(group_size, dim=-1)


def _expand_weight_metadata(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    granularity: str,
    group_size: int | None,
) -> torch.Tensor:
    if tensor.numel() == 1:
        return tensor.reshape(1, 1)

    if tensor.shape == weight.shape:
        return tensor

    out_features, in_features = weight.shape
    granularity = granularity.lower()
    if granularity in {"per_channel", "channel"}:
        if tensor.ndim == 1 and tensor.shape[0] == out_features:
            return tensor.reshape(out_features, 1)
        if tensor.ndim == 2 and tensor.shape == (out_features, 1):
            return tensor
    elif granularity in {"per_group", "group"}:
        if tensor.ndim == 2 and tensor.shape[0] == out_features:
            return _expand_group_tensor(tensor, in_features, group_size)
        if tensor.ndim == 2 and tensor.shape[1] == out_features:
            return _expand_group_tensor(tensor.transpose(0, 1), in_features, group_size)

    if tensor.ndim == 1 and tensor.shape[0] == out_features:
        return tensor.reshape(out_features, 1)

    raise ValueError(
        "Unsupported omni activation real weight metadata shape "
        f"{tuple(tensor.shape)} for weight shape {tuple(weight.shape)}."
    )


def _expand_input_metadata(
    tensor: torch.Tensor,
    x: torch.Tensor,
    granularity: str,
    group_size: int | None,
) -> torch.Tensor:
    if tensor.numel() == 1:
        return tensor.reshape(1, 1)

    in_features = x.shape[-1]
    granularity = granularity.lower()
    if granularity in {"per_channel", "channel"}:
        if tensor.ndim == 1 and tensor.shape[0] == in_features:
            return tensor.reshape(1, in_features)
    elif granularity in {"per_group", "group"}:
        if tensor.ndim == 1:
            return _expand_group_tensor(tensor.reshape(1, -1), in_features, group_size)

    if tensor.ndim == 2 and tensor.shape[-1] == in_features:
        return tensor

    raise ValueError(
        "Unsupported omni activation real input metadata shape "
        f"{tuple(tensor.shape)} for input width {in_features}."
    )


def _get_nested_config(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected `{key}` to be a dict, got {type(value)}.")
    return cast(dict[str, Any], value)


def _normalized_top_level_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized["quant_method"] = _METHOD_NAME

    global_quant_config = (
        _get_nested_config(normalized, "global_quant_config")
        or _get_nested_config(normalized, "linear")
    )
    if not global_quant_config:
        global_quant_config = {
            key: value
            for key, value in normalized.items()
            if key not in _RESERVED_TOP_LEVEL_KEYS
        }
    if "format" not in global_quant_config and normalized.get("format") is not None:
        global_quant_config["format"] = normalized["format"]
    if (
        "runtime_format" not in global_quant_config
        and normalized.get("runtime_format") is not None
    ):
        global_quant_config["runtime_format"] = normalized["runtime_format"]

    layer_quant_config = _get_nested_config(normalized, "layer_quant_config")
    if not layer_quant_config:
        # OmniQuant exports per-module runtime metadata under `modules`.
        # Treat that as the layer-level quant config when the newer vLLM
        # layout is not present so runtime parameters are preserved.
        layer_quant_config = _get_nested_config(normalized, "modules")

    # OmniQuant exports use tensor names compatible with the original
    # checkpoint layout rather than vLLM's generic defaults.
    default_tensor_names = {
        "weight": "qweight",
        "weight_scale": "w_scales",
        "weight_zero_point": "w_zero_points",
        "input_scale": None,
        "input_zero_point": None,
    }
    remapped_layer_quant_config: dict[str, dict[str, Any]] = {}
    for layer_name, layer_cfg in layer_quant_config.items():
        merged_layer_cfg = dict(layer_cfg)
        tensor_names = dict(default_tensor_names)
        tensor_names.update(_get_nested_config(merged_layer_cfg, "tensor_names"))
        merged_layer_cfg["tensor_names"] = tensor_names
        merged_layer_cfg.setdefault("weight_scale_granularity", "per_group")
        merged_layer_cfg.setdefault("weight_zero_point_granularity", "per_group")
        remapped_layer_quant_config[layer_name] = merged_layer_cfg
    layer_quant_config = remapped_layer_quant_config

    return {
        "quant_method": _METHOD_NAME,
        "version": normalized.get("version"),
        "global_quant_config": global_quant_config,
        "attention": _get_nested_config(normalized, "attention"),
        "layer_quant_config": layer_quant_config,
        "modules_to_not_convert": _parse_list(
            normalized,
            "modules_to_not_convert",
            "modules_to_skip",
            "skip_modules",
        ),
        "skip_with_substr": _parse_bool(normalized, "skip_with_substr", True),
        "checkpoint_metadata": _get_nested_config(normalized, "checkpoint_metadata"),
        "config_source": normalized.get("config_source"),
    }


@dataclass(frozen=True)
class _OmniShardRuntime:
    dense_weight: torch.Tensor
    input_scale: torch.Tensor | None
    input_inv_scale: torch.Tensor | None
    input_zero_point: torch.Tensor | None
    start: int
    end: int


_OMNI_FAST_PATH_GENERIC = 0
_OMNI_FAST_PATH_DENSE_LINEAR = 1
_OMNI_FAST_PATH_BLOCK_INT8 = 2
_OMNI_FAST_PATH_TRITON_W6A6 = 3
_OMNI_FAST_PATH_CUTLASS_W6A6 = 4


@dataclass(frozen=True)
class OmniActivationRealAttentionHelper:
    qk_scale: float = 1.0
    pv_scale: float = 1.0
    logits_soft_cap: float | None = None
    reference_without_forward_context: bool = True
    reference_with_forward_context: bool = True

    def can_run_without_forward_context(self) -> bool:
        return self.reference_without_forward_context

    def can_run_with_forward_context(self) -> bool:
        return self.reference_with_forward_context

    def _value_head_size(self, layer: torch.nn.Module) -> int:
        return int(getattr(layer, "head_size_v", layer.head_size))

    def _reshape_query(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        return query.view(num_tokens, layer.num_heads, layer.head_size).transpose(0, 1)

    def _reshape_key(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = key.shape[0]
        return key.view(num_tokens, layer.num_kv_heads, layer.head_size)

    def _reshape_value(
        self,
        layer: torch.nn.Module,
        value: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = value.shape[0]
        return value.view(num_tokens, layer.num_kv_heads, self._value_head_size(layer))

    def _expand_kv_heads(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        if layer.num_heads != layer.num_kv_heads:
            repeat_factor = layer.num_heads // layer.num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=0)
            value = value.repeat_interleave(repeat_factor, dim=0)
        return key, value

    def _reset_debug_state(
        self,
        layer: torch.nn.Module,
        metadata_summary: dict[str, Any],
    ) -> None:
        layer.omni_last_attention_metadata_summary = metadata_summary
        layer.omni_last_attention_engine_status = "metadata_read"
        layer.omni_last_attention_qkt_runtime_calls = 0
        layer.omni_last_attention_pv_runtime_calls = 0
        layer.omni_last_attention_cache_write_calls = 0
        layer.omni_last_attention_cache_read_calls = 0
        layer.omni_last_attention_cache_backend = None
        layer.omni_last_attention_cache_layout_compatible = False
        layer.omni_last_attention_debug = {
            "cache_backend": None,
            "cache_layout_compatible": False,
            "cache_write_calls": 0,
            "cache_read_calls": 0,
            "qkt_runtime_calls": 0,
            "pv_runtime_calls": 0,
        }

    def _bump_debug_counter(self, layer: torch.nn.Module, field: str, amount: int = 1) -> None:
        debug = getattr(layer, "omni_last_attention_debug", None)
        if isinstance(debug, dict):
            debug[field] = int(debug.get(field, 0)) + amount
        attr_name = f"omni_last_attention_{field}"
        if hasattr(layer, attr_name):
            setattr(layer, attr_name, int(getattr(layer, attr_name)) + amount)

    def _run_qkt_runtime(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        self._bump_debug_counter(layer, "qkt_runtime_calls")
        return torch.matmul(query, key.transpose(-2, -1)) * (
            layer.impl.scale * self.qk_scale
        )

    def _run_pv_runtime(
        self,
        layer: torch.nn.Module,
        probs: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        self._bump_debug_counter(layer, "pv_runtime_calls")
        return torch.matmul(probs, value) * self.pv_scale

    def _reference_attention_from_sequence(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_sequence: torch.Tensor,
        value_sequence: torch.Tensor,
        *,
        past_len: int,
    ) -> torch.Tensor:
        query_len = query.shape[0]
        seq_len = key_sequence.shape[0]
        query_heads = self._reshape_query(layer, query)
        key_heads, value_heads = self._expand_kv_heads(layer, key_sequence, value_sequence)

        scores = self._run_qkt_runtime(layer, query_heads, key_heads)
        if self.logits_soft_cap is not None and self.logits_soft_cap > 0:
            scores = torch.tanh(scores / self.logits_soft_cap) * self.logits_soft_cap

        key_positions = torch.arange(seq_len, device=scores.device, dtype=torch.int64)
        query_positions = torch.arange(
            past_len,
            past_len + query_len,
            device=scores.device,
            dtype=torch.int64,
        )
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = scores.softmax(dim=-1)
        attn_output = self._run_pv_runtime(layer, probs, value_heads)
        return attn_output.transpose(0, 1).reshape(query_len, -1)

    def _reference_attention(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self._reference_attention_from_sequence(
            layer,
            query,
            self._reshape_key(layer, key),
            self._reshape_value(layer, value),
            past_len=0,
        )

    def _cpu_cache_layout_compatible(
        self,
        layer: torch.nn.Module,
        kv_cache: torch.Tensor | None,
    ) -> bool:
        return bool(
            kv_cache is not None
            and kv_cache.ndim == 5
            and kv_cache.shape[0] == 2
            and kv_cache.shape[2] == layer.num_kv_heads
            and kv_cache.shape[3] > 0
            and kv_cache.shape[4] == layer.head_size
            and self._value_head_size(layer) == layer.head_size
        )

    def _infer_block_size(
        self,
        kv_cache: torch.Tensor | None,
    ) -> int | None:
        if kv_cache is not None and kv_cache.ndim >= 4:
            return int(kv_cache.shape[-2])

        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        except Exception:  # pragma: no cover - best effort fallback
            vllm_config = None

        cache_config = getattr(vllm_config, "cache_config", None)
        if cache_config is None:
            return None
        block_size = getattr(cache_config, "block_size", None)
        return int(block_size) if block_size else None

    def _shadow_num_blocks(
        self,
        kv_cache: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> int:
        if kv_cache is not None and kv_cache.ndim >= 2:
            return int(kv_cache.shape[1])
        if block_table is None or block_table.numel() == 0:
            return 0
        valid_blocks = block_table[block_table >= 0]
        if valid_blocks.numel() == 0:
            return 0
        return int(valid_blocks.max().item()) + 1

    def _ensure_shadow_cache(
        self,
        layer: torch.nn.Module,
        *,
        kv_cache: torch.Tensor | None,
        block_table: torch.Tensor | None,
        key_dtype: torch.dtype,
        key_device: torch.device,
        value_dtype: torch.dtype,
        value_device: torch.device,
    ) -> int | None:
        block_size = self._infer_block_size(kv_cache)
        if block_size is None or block_size <= 0:
            return None

        num_blocks = self._shadow_num_blocks(kv_cache, block_table)
        if num_blocks <= 0:
            return None

        key_shape = (num_blocks, layer.num_kv_heads, block_size, layer.head_size)
        value_shape = (
            num_blocks,
            layer.num_kv_heads,
            block_size,
            self._value_head_size(layer),
        )
        shadow_key = getattr(layer, "omni_shadow_key_cache", None)
        shadow_value = getattr(layer, "omni_shadow_value_cache", None)
        shadow_valid = getattr(layer, "omni_shadow_cache_valid", None)
        if (
            shadow_key is None
            or tuple(shadow_key.shape) != key_shape
            or shadow_key.device != key_device
            or shadow_key.dtype != key_dtype
        ):
            layer.omni_shadow_key_cache = torch.zeros(
                key_shape,
                dtype=key_dtype,
                device=key_device,
            )
        if (
            shadow_value is None
            or tuple(shadow_value.shape) != value_shape
            or shadow_value.device != value_device
            or shadow_value.dtype != value_dtype
        ):
            layer.omni_shadow_value_cache = torch.zeros(
                value_shape,
                dtype=value_dtype,
                device=value_device,
            )
        if (
            shadow_valid is None
            or tuple(shadow_valid.shape) != (num_blocks, block_size)
            or shadow_valid.device != key_device
        ):
            layer.omni_shadow_cache_valid = torch.zeros(
                (num_blocks, block_size),
                dtype=torch.bool,
                device=key_device,
            )
        return block_size

    def _write_current_tokens_to_caches(
        self,
        layer: torch.nn.Module,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> tuple[int | None, bool]:
        if slot_mapping.ndim != 1 or slot_mapping.shape[0] != key.shape[0]:
            return None, False

        key_sequence = self._reshape_key(layer, key)
        value_sequence = self._reshape_value(layer, value)
        block_size = self._ensure_shadow_cache(
            layer,
            kv_cache=kv_cache,
            block_table=block_table,
            key_dtype=key_sequence.dtype,
            key_device=key_sequence.device,
            value_dtype=value_sequence.dtype,
            value_device=value_sequence.device,
        )
        if block_size is None:
            return None, False

        shadow_key = cast(torch.Tensor, layer.omni_shadow_key_cache)
        shadow_value = cast(torch.Tensor, layer.omni_shadow_value_cache)
        shadow_valid = cast(torch.Tensor, layer.omni_shadow_cache_valid)
        physical_cache = self._cpu_cache_layout_compatible(layer, kv_cache)
        layer.omni_last_attention_cache_layout_compatible = physical_cache
        debug = getattr(layer, "omni_last_attention_debug", None)
        if isinstance(debug, dict):
            debug["cache_layout_compatible"] = physical_cache

        if physical_cache:
            key_cache, value_cache = cast(torch.Tensor, kv_cache).unbind(0)

        write_calls = 0
        for token_idx in range(slot_mapping.shape[0]):
            slot_idx = int(slot_mapping[token_idx].item())
            if slot_idx < 0:
                continue
            block_idx = slot_idx // block_size
            block_offset = slot_idx % block_size
            shadow_key[block_idx, :, block_offset, :] = key_sequence[token_idx]
            shadow_value[block_idx, :, block_offset, :] = value_sequence[token_idx]
            shadow_valid[block_idx, block_offset] = True
            if physical_cache:
                key_cache[block_idx, :, block_offset, :] = key_sequence[token_idx].to(
                    key_cache.dtype
                )
                value_cache[block_idx, :, block_offset, :] = value_sequence[token_idx].to(
                    value_cache.dtype
                )
            write_calls += 1

        self._bump_debug_counter(layer, "cache_write_calls", write_calls)
        return block_size, physical_cache

    def _gather_from_shadow_cache(
        self,
        layer: torch.nn.Module,
        *,
        block_table_row: torch.Tensor,
        seq_len: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shadow_key = cast(torch.Tensor, layer.omni_shadow_key_cache)
        shadow_value = cast(torch.Tensor, layer.omni_shadow_value_cache)
        shadow_valid = cast(torch.Tensor, layer.omni_shadow_cache_valid)

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for position in range(seq_len):
            logical_block = position // block_size
            if logical_block >= block_table_row.numel():
                raise ValueError(
                    "Block table does not cover the requested sequence length."
                )
            physical_block = int(block_table_row[logical_block].item())
            if physical_block < 0:
                raise ValueError("Encountered an invalid physical block in block_table.")
            block_offset = position % block_size
            if not bool(shadow_valid[physical_block, block_offset].item()):
                raise ValueError(
                    "Shadow KV cache is missing a token required for decode correctness."
                )
            keys.append(shadow_key[physical_block, :, block_offset, :])
            values.append(shadow_value[physical_block, :, block_offset, :])

        self._bump_debug_counter(layer, "cache_read_calls")
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)

    def _gather_from_physical_cache(
        self,
        layer: torch.nn.Module,
        *,
        kv_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        seq_len: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = kv_cache.unbind(0)

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for position in range(seq_len):
            logical_block = position // block_size
            if logical_block >= block_table_row.numel():
                raise ValueError(
                    "Block table does not cover the requested sequence length."
                )
            physical_block = int(block_table_row[logical_block].item())
            if physical_block < 0:
                raise ValueError("Encountered an invalid physical block in block_table.")
            block_offset = position % block_size
            keys.append(key_cache[physical_block, :, block_offset, :].to(torch.float32))
            values.append(value_cache[physical_block, :, block_offset, :].to(torch.float32))

        self._bump_debug_counter(layer, "cache_read_calls")
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)

    def forward_without_forward_context(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del output_shape
        return self._reference_attention(layer, query, key, value)

    def forward_with_forward_context(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_metadata: Any,
        kv_cache: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor | None:
        del output_shape

        metadata_summary = {
            "has_attn_metadata": attn_metadata is not None,
            "has_query_start_loc": hasattr(attn_metadata, "query_start_loc"),
            "has_seq_lens": hasattr(attn_metadata, "seq_lens"),
            "has_block_table": hasattr(attn_metadata, "block_table"),
            "has_slot_mapping": slot_mapping is not None,
            "kv_cache_shape": tuple(kv_cache.shape) if kv_cache is not None else None,
        }
        self._reset_debug_state(layer, metadata_summary)

        if not self.can_run_with_forward_context():
            return None
        if attn_metadata is None:
            return None

        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if query_start_loc is None or seq_lens is None:
            layer.omni_last_attention_engine_status = "metadata_missing_boundaries"
            return None

        if query_start_loc.ndim != 1 or seq_lens.ndim != 1:
            layer.omni_last_attention_engine_status = "metadata_bad_rank"
            return None

        total_query_tokens = query.shape[0]
        boundary_list = [int(x) for x in query_start_loc.tolist()]
        seq_lens_list = [int(x) for x in seq_lens.tolist()]
        if not boundary_list or boundary_list[0] != 0 or boundary_list[-1] != total_query_tokens:
            layer.omni_last_attention_engine_status = "metadata_bad_query_bounds"
            return None
        if len(seq_lens_list) != len(boundary_list) - 1:
            layer.omni_last_attention_engine_status = "metadata_bad_seq_count"
            return None

        block_table = getattr(attn_metadata, "block_table", None)
        if block_table is not None and block_table.ndim == 1:
            block_table = block_table.unsqueeze(0)

        if slot_mapping is None or slot_mapping.ndim != 1:
            if any(seq_len != (end - start) for start, end, seq_len in zip(
                boundary_list[:-1], boundary_list[1:], seq_lens_list, strict=True
            )):
                layer.omni_last_attention_engine_status = "metadata_missing_slot_mapping"
                return None

            outputs = [
                self._reference_attention(layer, query[start:end], key[start:end], value[start:end])
                for start, end in zip(boundary_list[:-1], boundary_list[1:], strict=True)
            ]
            layer.omni_last_attention_engine_status = "forward_context_reference_prefill"
            return torch.cat(outputs, dim=0)

        block_size, physical_cache = self._write_current_tokens_to_caches(
            layer,
            key=key,
            value=value,
            slot_mapping=slot_mapping,
            kv_cache=kv_cache,
            block_table=block_table,
        )
        if block_size is None:
            if any(seq_len != (end - start) for start, end, seq_len in zip(
                boundary_list[:-1], boundary_list[1:], seq_lens_list, strict=True
            )):
                layer.omni_last_attention_engine_status = "metadata_missing_cache_layout"
                return None

            outputs = [
                self._reference_attention(layer, query[start:end], key[start:end], value[start:end])
                for start, end in zip(boundary_list[:-1], boundary_list[1:], strict=True)
            ]
            layer.omni_last_attention_engine_status = "forward_context_reference_prefill"
            return torch.cat(outputs, dim=0)

        outputs: list[torch.Tensor] = []
        saw_decode = False
        cache_backend = "physical" if physical_cache else "shadow"
        if isinstance(getattr(layer, "omni_last_attention_debug", None), dict):
            layer.omni_last_attention_debug["cache_backend"] = cache_backend
        layer.omni_last_attention_cache_backend = cache_backend

        for seq_idx, (start, end, seq_len) in enumerate(
            zip(boundary_list[:-1], boundary_list[1:], seq_lens_list, strict=True)
        ):
            query_len = end - start
            past_len = seq_len - query_len
            if past_len < 0:
                layer.omni_last_attention_engine_status = (
                    f"metadata_bad_seq_len_seq_{seq_idx}"
                )
                return None
            if block_table is None or block_table.ndim != 2 or seq_idx >= block_table.shape[0]:
                if past_len != 0:
                    layer.omni_last_attention_engine_status = "metadata_missing_block_table"
                    return None
                outputs.append(
                    self._reference_attention(
                        layer,
                        query[start:end],
                        key[start:end],
                        value[start:end],
                    )
                )
                continue

            block_table_row = block_table[seq_idx]
            try:
                if physical_cache and kv_cache is not None:
                    key_sequence, value_sequence = self._gather_from_physical_cache(
                        layer,
                        kv_cache=kv_cache,
                        block_table_row=block_table_row,
                        seq_len=seq_len,
                        block_size=block_size,
                    )
                else:
                    key_sequence, value_sequence = self._gather_from_shadow_cache(
                        layer,
                        block_table_row=block_table_row,
                        seq_len=seq_len,
                        block_size=block_size,
                    )
            except ValueError as exc:
                layer.omni_last_attention_engine_status = f"cache_gather_failed_seq_{seq_idx}"
                if not torch.compiler.is_compiling():
                    logger.debug("Omni attention cache gather failed: %s", exc)
                return None

            outputs.append(
                self._reference_attention_from_sequence(
                    layer,
                    query[start:end],
                    key_sequence.to(query.dtype),
                    value_sequence.to(value.dtype),
                    past_len=past_len,
                )
            )
            saw_decode = saw_decode or past_len > 0

        layer.omni_last_attention_engine_status = (
            "forward_context_reference_decode"
            if saw_decode
            else "forward_context_reference_prefill"
        )
        return torch.cat(outputs, dim=0)


@register_quantization_config(_METHOD_NAME)
class OmniActivationRealConfig(QuantizationConfig):
    """Config class for omni activation real quantization."""

    def __init__(
        self,
        *,
        global_quant_config: dict[str, Any],
        attention_config: dict[str, Any],
        layer_quant_config: dict[str, dict[str, Any]],
        modules_to_not_convert: list[str],
        skip_with_substr: bool,
        checkpoint_metadata: dict[str, Any] | None = None,
        config_source: str | None = None,
    ) -> None:
        super().__init__()
        self.global_quant_config = global_quant_config
        self.attention_config = attention_config
        self.layer_quant_config = layer_quant_config
        self.modules_to_not_convert = modules_to_not_convert
        self.skip_with_substr = skip_with_substr
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.config_source = config_source
        self.module_runtime_metadata = self.layer_quant_config
        self.weight_bits = self.global_quant_config.get(
            "weight_bits",
            self.global_quant_config.get("bits"),
        )
        self.activation_bits = self.global_quant_config.get("activation_bits")
        self.group_size = self.global_quant_config.get("group_size")
        self.runtime_format = self.global_quant_config.get(
            "runtime_format",
            self.global_quant_config.get("format"),
        )

    def get_name(self) -> str:
        return _METHOD_NAME

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["omni_act_quant_config.json"]

    @classmethod
    def merge_with_file_config(
        cls,
        config_json_quant_config: dict[str, Any],
        file_quant_config: dict[str, Any],
    ) -> dict[str, Any]:
        merged = _deep_merge_dicts(file_quant_config, config_json_quant_config)
        merged["config_source"] = "config.json+omni_act_quant_config.json"
        return merged

    @classmethod
    def from_config_file(cls, config_file: str) -> "OmniActivationRealConfig":
        with open(config_file) as handle:
            return cls.from_config(json.load(handle))

    @classmethod
    def from_config_dict_json(cls, config_dict_json: str) -> "OmniActivationRealConfig":
        return cls.from_config(json.loads(config_dict_json))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OmniActivationRealConfig":
        normalized = _normalized_top_level_config(config)
        return cls(
            global_quant_config=cast(
                dict[str, Any], normalized["global_quant_config"]
            ),
            attention_config=cast(dict[str, Any], normalized["attention"]),
            layer_quant_config=cast(dict[str, dict[str, Any]], normalized["layer_quant_config"]),
            modules_to_not_convert=cast(list[str], normalized["modules_to_not_convert"]),
            skip_with_substr=cast(bool, normalized["skip_with_substr"]),
            checkpoint_metadata=cast(
                dict[str, Any], normalized.get("checkpoint_metadata", {})
            ),
            config_source=cast(str | None, normalized.get("config_source")),
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
    ) -> str | None:
        method_name = _normalize_method_name(
            hf_quant_cfg.get("quant_method", hf_quant_cfg.get("format"))
        )
        if method_name != _METHOD_NAME:
            return None
        if user_quant is None or _normalize_method_name(user_quant) == _METHOD_NAME:
            return _METHOD_NAME
        return None

    def maybe_update_config(self, model_name: str):
        config_paths = [
            os.path.join(model_name, filename)
            for filename in self.get_config_filenames()
        ]
        for config_path in config_paths:
            if not os.path.isfile(config_path):
                continue
            with open(config_path) as handle:
                file_quant_config = json.load(handle)
            merged = self.merge_with_file_config(
                {
                    "global_quant_config": self.global_quant_config,
                    "attention": self.attention_config,
                    "layer_quant_config": self.layer_quant_config,
                    "modules_to_not_convert": self.modules_to_not_convert,
                    "skip_with_substr": self.skip_with_substr,
                    "checkpoint_metadata": self.checkpoint_metadata,
                    "config_source": self.config_source,
                },
                file_quant_config,
            )
            updated = type(self).from_config(merged)
            self.global_quant_config = updated.global_quant_config
            self.attention_config = updated.attention_config
            self.layer_quant_config = updated.layer_quant_config
            self.modules_to_not_convert = updated.modules_to_not_convert
            self.skip_with_substr = updated.skip_with_substr
            self.checkpoint_metadata = updated.checkpoint_metadata
            self.config_source = updated.config_source
            self.module_runtime_metadata = self.layer_quant_config
            self.weight_bits = updated.weight_bits
            self.activation_bits = updated.activation_bits
            self.group_size = updated.group_size
            self.runtime_format = updated.runtime_format
            break

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.modules_to_not_convert:
            self.modules_to_not_convert = hf_to_vllm_mapper.apply_list(
                self.modules_to_not_convert
            )

        remapped_layer_quant_config: dict[str, dict[str, Any]] = {}
        for layer_name, layer_cfg in self.layer_quant_config.items():
            if "*" in layer_name:
                remapped_layer_quant_config[layer_name] = layer_cfg
                continue
            mapped_names = hf_to_vllm_mapper.apply_list([layer_name])
            if mapped_names:
                remapped_layer_quant_config[mapped_names[0]] = layer_cfg
        self.layer_quant_config = remapped_layer_quant_config

    def _find_matched_layer_config(self, layer_name: str) -> dict[str, Any]:
        for pattern, layer_config in self.layer_quant_config.items():
            if "*" in pattern:
                if fnmatch.fnmatch(layer_name, pattern):
                    return layer_config
            elif layer_name == pattern:
                return layer_config
        return {}

    def _get_effective_layer_config(self, prefix: str) -> dict[str, Any]:
        proj_name = prefix.split(".")[-1]
        if proj_name in self.packed_modules_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_name)
                for shard_name in self.packed_modules_mapping[proj_name]
            ]
            shard_configs = [
                _deep_merge_dicts(
                    self.global_quant_config,
                    self._find_matched_layer_config(shard_prefix),
                )
                for shard_prefix in shard_prefixes
            ]
            non_empty_configs = [
                cfg for cfg in shard_configs if cfg != self.global_quant_config
            ]
            if not non_empty_configs:
                return dict(self.global_quant_config)

            shared_config = dict(self.global_quant_config)
            missing = object()
            all_keys = set().union(*(cfg.keys() for cfg in non_empty_configs))
            for key in all_keys:
                first_value = non_empty_configs[0].get(key, missing)
                if first_value is missing:
                    continue
                if all(
                    shard_config.get(key, missing) is not missing
                    and _json_value_equal(first_value, shard_config[key])
                    for shard_config in non_empty_configs[1:]
                ):
                    shared_config[key] = first_value
            return shared_config

        return _deep_merge_dicts(
            self.global_quant_config,
            self._find_matched_layer_config(prefix),
        )

    def _attention_enabled(self, prefix: str) -> bool:
        attention_config = self.get_attention_config(prefix)
        return bool(attention_config) and _parse_bool(attention_config, "enabled", True)

    def get_attention_config(self, prefix: str) -> dict[str, Any]:
        layer_specific_attention = _get_nested_config(
            self._find_matched_layer_config(prefix),
            "attention",
        )
        return _deep_merge_dicts(self.attention_config, layer_specific_attention)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )

        if isinstance(layer, LinearBase):
            if self.modules_to_not_convert and is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=self.skip_with_substr,
            ):
                return UnquantizedLinearMethod()
            layer_config = self._get_effective_layer_config(prefix)
            if not _parse_bool(layer_config, "enabled", True):
                return UnquantizedLinearMethod()
            return OmniActivationRealLinearMethod(self, layer_config)

        if layer.__class__.__name__ == "Attention" and self._attention_enabled(prefix):
            return OmniActivationRealAttentionMethod(self, self.get_attention_config(prefix))

        return None


class OmniActivationRealLinearMethod(QuantizeMethodBase):
    """Correctness-first linear method for omni activation real checkpoints."""

    def __init__(
        self,
        quant_config: OmniActivationRealConfig,
        layer_config: dict[str, Any],
    ) -> None:
        self.quant_config = quant_config
        self.layer_config = layer_config

    def _tensor_name(
        self,
        logical_name: str,
        default_name: str | None = None,
    ) -> str | None:
        tensor_names = _get_nested_config(self.layer_config, "tensor_names")
        if logical_name in tensor_names:
            value = tensor_names[logical_name]
        else:
            value = self.layer_config.get(f"{logical_name}_name", default_name)
        if value in (None, False, ""):
            return None
        return str(value)

    def _logical_shard_count(self, output_partition_sizes: list[int]) -> int:
        return len(output_partition_sizes)

    def _register_metadata_param(
        self,
        layer: torch.nn.Module,
        *,
        param_name: str | None,
        dtype: torch.dtype,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        weight_loader: Any,
        granularity: str,
        group_size: int | None,
        role: str,
    ) -> None:
        if param_name is None:
            return

        shard_count = self._logical_shard_count(output_partition_sizes)
        output_size = sum(output_partition_sizes)
        attrs: dict[str, Any] = {
            "weight_loader": self._wrap_weight_loader(weight_loader),
            "ignore_warning": True,
        }
        granularity = granularity.lower()
        if granularity in {"per_tensor", "tensor"}:
            shape = (shard_count,) if shard_count > 1 else (1,)
            attrs["needs_scalar_to_array"] = shard_count > 1
        elif granularity in {"per_channel", "channel"}:
            shape = (output_size,)
            attrs["output_dim"] = 0
        elif granularity in {"per_group", "group"}:
            if group_size is None:
                raise ValueError(
                    f"{role} uses grouped metadata but no group size was specified."
                )
            if output_size == 0 or input_size_per_partition % group_size != 0:
                raise ValueError(
                    f"Cannot create grouped {role} metadata with output_size="
                    f"{output_size}, input_size_per_partition={input_size_per_partition}, "
                    f"group_size={group_size}."
                )
            shape = (output_size, input_size_per_partition // group_size)
            attrs["output_dim"] = 0
            # Grouped checkpoint tensors are stored as [out_features, num_groups]
            # for column/qkv paths and sharded on the group axis for row paths.
            attrs["input_dim"] = 1
        elif granularity in {"per_input", "input"}:
            shape = (input_size_per_partition,)
            attrs["input_dim"] = 0
        else:
            raise ValueError(f"Unsupported {role} granularity: {granularity}")

        param = Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        set_weight_attrs(param, attrs)
        layer.register_parameter(param_name, param)

    def _wrap_weight_loader(self, original_weight_loader: Any) -> Any:
        def wrapped_weight_loader(
            param: Parameter,
            loaded_weight: torch.Tensor,
            *args: Any,
        ) -> None:
            if len(loaded_weight.shape) == 0:
                loaded_weight = loaded_weight.reshape(1)

            # OmniQuant exports already store the final grouped tensor layout for
            # these parameters. When shapes already match, bypass the default vLLM
            # TP slicing logic and copy directly.
            if tuple(param.data.shape) == tuple(loaded_weight.shape):
                param.data.copy_(loaded_weight)
                return

            original_weight_loader(param, loaded_weight, *args)

        return wrapped_weight_loader

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size
        del output_size

        weight_loader = extra_weight_attrs["weight_loader"]
        weight_name = self._tensor_name("weight", "weight")
        assert weight_name is not None

        checkpoint_weight_dtype = _resolve_dtype(
            self.layer_config.get("checkpoint_weight_dtype")
            or self.layer_config.get("weight_dtype"),
            torch.int8,
        )
        weight_scale_granularity = _metadata_granularity(
            self.layer_config,
            "weight_scale",
            "per_tensor",
        )
        weight_scale_group_size = _metadata_group_size(self.layer_config, "weight_scale")
        grouped_weight_layout = (
            weight_scale_granularity in {"per_group", "group"}
            and weight_scale_group_size is not None
            and input_size_per_partition % weight_scale_group_size == 0
        )
        packing_mode = _weight_packing_mode(
            self.quant_config.runtime_format,
            self.layer_config,
        )
        if grouped_weight_layout:
            packed_group_size = weight_scale_group_size
            assert packed_group_size is not None
            if packing_mode == _PACKED_UINT4X2 and packed_group_size % 2 != 0:
                raise ValueError(
                    "Packed omni 4-bit weights require an even group size, got "
                    f"{packed_group_size}."
                )
            if packing_mode == _PACKED_UINT6X4 and packed_group_size % 4 != 0:
                raise ValueError(
                    "Packed omni 6-bit weights require a group size divisible "
                    f"by 4, got {packed_group_size}."
                )
            weight_shape = (
                sum(output_partition_sizes),
                input_size_per_partition // weight_scale_group_size,
                weight_scale_group_size // 2
                if packing_mode == _PACKED_UINT4X2
                else (
                    (weight_scale_group_size // 4) * 3
                    if packing_mode == _PACKED_UINT6X4
                    else weight_scale_group_size
                ),
            )
        else:
            if packing_mode != "plain":
                raise ValueError(
                    "Packed omni sub-byte weights currently require per-group "
                    "checkpoint layout."
                )
            weight_shape = (
                sum(output_partition_sizes),
                input_size_per_partition,
            )
        source_weight = Parameter(
            torch.empty(weight_shape, dtype=checkpoint_weight_dtype),
            requires_grad=False,
        )
        source_weight_attrs = dict(extra_weight_attrs)
        source_weight_attrs.update(
            {
                "input_dim": 1,
                "output_dim": 0,
                "omni_grouped_weight_layout": grouped_weight_layout,
                "omni_group_size": weight_scale_group_size,
                "weight_loader": self._wrap_weight_loader(weight_loader),
            }
        )
        set_weight_attrs(
            source_weight,
            source_weight_attrs,
        )
        layer.register_parameter(weight_name, source_weight)

        self._register_metadata_param(
            layer,
            param_name=self._tensor_name("weight_scale", "weight_scale"),
            dtype=_resolve_dtype(
                self.layer_config.get("weight_scale_dtype"), torch.float32
            ),
            output_partition_sizes=output_partition_sizes,
            input_size_per_partition=input_size_per_partition,
            weight_loader=weight_loader,
            granularity=weight_scale_granularity,
            group_size=weight_scale_group_size,
            role="weight_scale",
        )

        symmetric_weights = _parse_bool(self.layer_config, "symmetric_weights", True)
        weight_zero_point_name = self._tensor_name(
            "weight_zero_point",
            None if symmetric_weights else "weight_zero_point",
        )
        if weight_zero_point_name is not None:
            self._register_metadata_param(
                layer,
                param_name=weight_zero_point_name,
                dtype=_resolve_dtype(
                    self.layer_config.get("weight_zero_point_dtype"), torch.float32
                ),
                output_partition_sizes=output_partition_sizes,
                input_size_per_partition=input_size_per_partition,
                weight_loader=weight_loader,
                granularity=_metadata_granularity(
                    self.layer_config,
                    "weight_zero_point",
                    weight_scale_granularity,
                ),
                group_size=_metadata_group_size(
                    self.layer_config,
                    "weight_zero_point",
                )
                or weight_scale_group_size,
                role="weight_zero_point",
            )

        quantize_inputs = _parse_bool(self.layer_config, "quantize_inputs", True)
        input_scale_name = self._tensor_name(
            "input_scale",
            "input_scale" if quantize_inputs else None,
        )
        if input_scale_name is not None:
            self._register_metadata_param(
                layer,
                param_name=input_scale_name,
                dtype=_resolve_dtype(
                    self.layer_config.get("input_scale_dtype"), torch.float32
                ),
                output_partition_sizes=output_partition_sizes,
                input_size_per_partition=input_size_per_partition,
                weight_loader=weight_loader,
                granularity=_metadata_granularity(
                    self.layer_config,
                    "input_scale",
                    "per_tensor",
                ),
                group_size=_metadata_group_size(self.layer_config, "input_scale"),
                role="input_scale",
            )

        symmetric_inputs = _parse_bool(self.layer_config, "symmetric_inputs", True)
        input_zero_point_name = self._tensor_name(
            "input_zero_point",
            None if symmetric_inputs or not quantize_inputs else "input_zero_point",
        )
        if input_zero_point_name is not None:
            self._register_metadata_param(
                layer,
                param_name=input_zero_point_name,
                dtype=_resolve_dtype(
                    self.layer_config.get("input_zero_point_dtype"), torch.float32
                ),
                output_partition_sizes=output_partition_sizes,
                input_size_per_partition=input_size_per_partition,
                weight_loader=weight_loader,
                granularity=_metadata_granularity(
                    self.layer_config,
                    "input_zero_point",
                    _metadata_granularity(
                        self.layer_config,
                        "input_scale",
                        "per_tensor",
                    ),
                ),
                group_size=_metadata_group_size(self.layer_config, "input_zero_point")
                or _metadata_group_size(self.layer_config, "input_scale"),
                role="input_zero_point",
            )

    def _get_layer_param(
        self,
        layer: torch.nn.Module,
        logical_name: str,
        default_name: str | None = None,
    ) -> torch.Tensor | None:
        tensor_name = self._tensor_name(logical_name, default_name)
        if tensor_name is None or not hasattr(layer, tensor_name):
            return None
        return cast(torch.Tensor, getattr(layer, tensor_name))

    def _get_activation_runtime_config(self) -> tuple[int, bool, bool, str]:
        global_cfg = self.quant_config.global_quant_config
        activation_bits = int(
            self.layer_config.get(
                "activation_bits",
                global_cfg.get("activation_bits", 16),
            )
        )
        activation_symmetric = _parse_bool(
            self.layer_config,
            "activation_symmetric",
            _parse_bool(global_cfg, "activation_symmetric", False)
            or _parse_bool(global_cfg, "activation_sym", False),
        )
        activation_disable_zero_point = _parse_bool(
            self.layer_config,
            "activation_disable_zero_point",
            _parse_bool(global_cfg, "activation_disable_zero_point", False),
        )
        activation_dynamic_method = str(
            self.layer_config.get(
                "activation_dynamic_method",
                global_cfg.get("activation_dynamic_method", "per_token"),
            )
        ).strip().lower()
        return (
            activation_bits,
            activation_symmetric,
            activation_disable_zero_point,
            activation_dynamic_method,
        )

    def _resolve_shard_input_meta(
        self,
        input_meta: torch.Tensor | None,
        shard_idx: int,
        input_width: int,
        granularity: str,
        group_size: int | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if input_meta is None:
            return None

        shard_meta = input_meta
        if input_meta.ndim == 1 and input_meta.numel() > 1 and input_meta.numel() > shard_idx:
            if input_meta.numel() != input_width:
                shard_meta = input_meta[shard_idx].reshape(1, 1)

        meta_ref = torch.empty((1, input_width), dtype=dtype, device=device)
        return _expand_input_metadata(
            shard_meta.to(device=device, dtype=dtype),
            meta_ref,
            granularity,
            group_size,
        ).contiguous()

    def _build_runtime(
        self,
        layer: torch.nn.Module,
    ) -> tuple[torch.Tensor, tuple[_OmniShardRuntime, ...]]:
        source_weight = self._get_layer_param(layer, "weight", "weight")
        assert source_weight is not None
        weight_scale_granularity = _metadata_granularity(
            self.layer_config,
            "weight_scale",
            "per_tensor",
        )
        weight_group_size = _metadata_group_size(self.layer_config, "weight_scale")
        packing_mode = _weight_packing_mode(
            self.quant_config.runtime_format,
            self.layer_config,
        )

        if packing_mode == _PACKED_UINT4X2:
            if source_weight.ndim != 3:
                raise ValueError(
                    "Packed omni 4-bit weights must be stored as a 3D grouped "
                    f"tensor, got shape {tuple(source_weight.shape)}."
                )
            if weight_group_size is None:
                raise ValueError(
                    "Packed omni 4-bit weights require weight_scale group_size "
                    "metadata."
                )
            dense_source = _unpack_uint4x2(
                source_weight,
                signed=_parse_bool(self.layer_config, "weight_symmetric", False)
                or _parse_bool(self.layer_config, "weight_disable_zero_point", False),
                unpacked_last_dim=weight_group_size,
            ).to(torch.float32)
        elif packing_mode == _PACKED_UINT6X4:
            if source_weight.ndim != 3:
                raise ValueError(
                    "Packed omni 6-bit weights must be stored as a 3D grouped "
                    f"tensor, got shape {tuple(source_weight.shape)}."
                )
            if weight_group_size is None:
                raise ValueError(
                    "Packed omni 6-bit weights require weight_scale group_size "
                    "metadata."
                )
            if source_weight.shape[-1] * 4 != weight_group_size * 3:
                raise ValueError(
                    "Packed omni 6-bit grouped width does not match the "
                    f"configured group size {weight_group_size}."
                )
            dense_source = _unpack_uint6x4(
                source_weight,
                signed=_parse_bool(self.layer_config, "weight_symmetric", False)
                or _parse_bool(self.layer_config, "weight_disable_zero_point", False),
                unpacked_last_dim=weight_group_size,
            ).to(torch.float32)
        else:
            dense_source = source_weight.to(torch.float32)
        if dense_source.ndim == 2 and _parse_bool(self.layer_config, "transpose_weight", False):
            dense_source = dense_source.transpose(0, 1).contiguous()

        weight_scale = self._get_layer_param(layer, "weight_scale", "weight_scale")
        weight_zero_point = self._get_layer_param(
            layer,
            "weight_zero_point",
            None,
        )

        output_partition_sizes = list(getattr(layer, "output_partition_sizes", []))
        if not output_partition_sizes:
            output_partition_sizes = [dense_source.shape[0]]

        input_scale = self._get_layer_param(layer, "input_scale", None)
        input_zero_point = self._get_layer_param(layer, "input_zero_point", None)
        input_scale_granularity = _metadata_granularity(
            self.layer_config,
            "input_scale",
            "per_tensor",
        )
        input_group_size = _metadata_group_size(self.layer_config, "input_scale")
        runtime_meta_dtype = layer.params_dtype
        runtime_meta_device = dense_source.device
        input_width = (
            dense_source.shape[1] * dense_source.shape[2]
            if dense_source.ndim == 3
            else dense_source.shape[1]
        )

        shard_runtimes: list[_OmniShardRuntime] = []
        dense_weight_shards: list[torch.Tensor] = []
        start = 0
        for shard_idx, shard_size in enumerate(output_partition_sizes):
            end = start + shard_size
            shard_weight = dense_source[start:end]

            if weight_zero_point is not None:
                shard_zero = (
                    weight_zero_point[shard_idx]
                    if weight_zero_point.ndim == 1
                    and weight_zero_point.numel() == len(output_partition_sizes)
                    else weight_zero_point[start:end]
                    if weight_zero_point.ndim >= 1
                    and weight_zero_point.shape[0] == dense_source.shape[0]
                    else weight_zero_point
                )
                shard_zero = shard_zero.to(torch.float32)
                if shard_weight.ndim == 3:
                    if shard_zero.ndim == 2 and shard_zero.shape == shard_weight.shape[:2]:
                        shard_zero = shard_zero.unsqueeze(-1)
                    elif shard_zero.ndim != 3 or shard_zero.shape != shard_weight.shape:
                        flat_zero = _expand_weight_metadata(
                            shard_zero,
                            shard_weight.reshape(shard_weight.shape[0], -1),
                            _metadata_granularity(
                                self.layer_config,
                                "weight_zero_point",
                                weight_scale_granularity,
                            ),
                            _metadata_group_size(self.layer_config, "weight_zero_point")
                            or weight_group_size,
                        )
                        shard_zero = flat_zero.reshape_as(shard_weight)
                else:
                    shard_zero = _expand_weight_metadata(
                        shard_zero,
                        shard_weight,
                        _metadata_granularity(
                            self.layer_config,
                            "weight_zero_point",
                            weight_scale_granularity,
                        ),
                        _metadata_group_size(self.layer_config, "weight_zero_point")
                        or weight_group_size,
                    )
                shard_weight = shard_weight - shard_zero

            if weight_scale is not None:
                shard_scale = (
                    weight_scale[shard_idx]
                    if weight_scale.ndim == 1
                    and weight_scale.numel() == len(output_partition_sizes)
                    else weight_scale[start:end]
                    if weight_scale.ndim >= 1
                    and weight_scale.shape[0] == dense_source.shape[0]
                    else weight_scale
                )
                shard_scale = shard_scale.to(torch.float32)
                if shard_weight.ndim == 3:
                    if shard_scale.ndim == 2 and shard_scale.shape == shard_weight.shape[:2]:
                        shard_scale = shard_scale.unsqueeze(-1)
                    elif shard_scale.ndim != 3 or shard_scale.shape != shard_weight.shape:
                        flat_scale = _expand_weight_metadata(
                            shard_scale,
                            shard_weight.reshape(shard_weight.shape[0], -1),
                            weight_scale_granularity,
                            weight_group_size,
                        )
                        shard_scale = flat_scale.reshape_as(shard_weight)
                else:
                    shard_scale = _expand_weight_metadata(
                        shard_scale,
                        shard_weight,
                        weight_scale_granularity,
                        weight_group_size,
                    )
                shard_weight = shard_weight * shard_scale

            if shard_weight.ndim == 3:
                shard_weight = shard_weight.reshape(shard_weight.shape[0], -1)

            dense_weight_shards.append(shard_weight.to(dtype=layer.params_dtype))
            shard_input_scale = self._resolve_shard_input_meta(
                input_scale,
                shard_idx,
                input_width,
                input_scale_granularity,
                input_group_size,
                runtime_meta_dtype,
                runtime_meta_device,
            )
            shard_input_inv_scale = (
                torch.reciprocal(shard_input_scale) if shard_input_scale is not None else None
            )
            shard_input_zero_point = self._resolve_shard_input_meta(
                input_zero_point,
                shard_idx,
                input_width,
                input_scale_granularity,
                input_group_size,
                runtime_meta_dtype,
                runtime_meta_device,
            )
            shard_runtimes.append(
                _OmniShardRuntime(
                    dense_weight=dense_weight_shards[-1],
                    input_scale=shard_input_scale,
                    input_inv_scale=shard_input_inv_scale,
                    input_zero_point=shard_input_zero_point,
                    start=start,
                    end=end,
                )
            )
            start = end

        dense_weight = torch.cat(dense_weight_shards, dim=0).contiguous()
        if "omni_weight" in layer._buffers:
            layer._buffers["omni_weight"] = dense_weight
        else:
            layer.register_buffer("omni_weight", dense_weight, persistent=False)

        resolved_runtimes = []
        for shard_runtime in shard_runtimes:
            resolved_runtimes.append(
                _OmniShardRuntime(
                    dense_weight=shard_runtime.dense_weight,
                    input_scale=shard_runtime.input_scale,
                    input_inv_scale=shard_runtime.input_inv_scale,
                    input_zero_point=shard_runtime.input_zero_point,
                    start=shard_runtime.start,
                    end=shard_runtime.end,
                )
            )
        layer.omni_shard_runtimes = tuple(resolved_runtimes)
        first_shard = resolved_runtimes[0]
        layer.omni_shared_input_quant_params = all(
            shard.input_scale is first_shard.input_scale
            and shard.input_inv_scale is first_shard.input_inv_scale
            and shard.input_zero_point is first_shard.input_zero_point
            for shard in resolved_runtimes[1:]
        )
        return dense_weight, layer.omni_shard_runtimes

    def _build_block_int8_runtime(self, layer: torch.nn.Module) -> bool:
        (
            activation_bits,
            activation_symmetric,
            activation_disable_zero_point,
            activation_dynamic_method,
        ) = self._get_activation_runtime_config()
        source_weight = self._get_layer_param(layer, "weight", "weight")
        weight_scale = self._get_layer_param(layer, "weight_scale", "weight_scale")
        weight_zero_point = self._get_layer_param(layer, "weight_zero_point", None)
        input_scale = self._get_layer_param(layer, "input_scale", None)
        input_zero_point = self._get_layer_param(layer, "input_zero_point", None)

        if (
            activation_dynamic_method != "per_token"
            or activation_bits != 8
            or not activation_symmetric
            or not activation_disable_zero_point
        ):
            return False
        if input_scale is not None or input_zero_point is not None:
            return False
        if source_weight is None or weight_scale is None or weight_zero_point is None:
            return False
        if _parse_bool(self.layer_config, "transpose_weight", False):
            return False
        grouped_runtime = self._materialize_grouped_int_weight_runtime(
            source_weight=source_weight,
            weight_scale=weight_scale,
            weight_zero_point=weight_zero_point,
        )
        if grouped_runtime is None:
            return False

        adjusted_weight, grouped_scales, group_size = grouped_runtime
        if group_size != 128:
            return False

        block_weight = adjusted_weight.to(dtype=torch.int8).reshape(
            adjusted_weight.shape[0], -1
        ).contiguous()
        block_scale = grouped_scales.contiguous()

        if "omni_block_weight" in layer._buffers:
            layer._buffers["omni_block_weight"] = block_weight
        else:
            layer.register_buffer("omni_block_weight", block_weight, persistent=False)
        if "omni_block_scale" in layer._buffers:
            layer._buffers["omni_block_scale"] = block_scale
        else:
            layer.register_buffer("omni_block_scale", block_scale, persistent=False)
        layer.omni_block_group_size = int(group_size)
        return True

    def _build_triton_w6a6_runtime(self, layer: torch.nn.Module) -> bool:
        if os.getenv("OMNI_DISABLE_W6A6_FASTPATH", "0").strip() == "1":
            return False
        (
            activation_bits,
            activation_symmetric,
            activation_disable_zero_point,
            activation_dynamic_method,
        ) = self._get_activation_runtime_config()
        weight_bits = int(
            self.layer_config.get(
                "weight_bits",
                self.quant_config.global_quant_config.get("weight_bits", 16),
            )
        )
        if weight_bits < 6:
            return False
        if activation_bits <= 4:
            return False
        source_weight = self._get_layer_param(layer, "weight", "weight")
        weight_scale = self._get_layer_param(layer, "weight_scale", "weight_scale")
        weight_zero_point = self._get_layer_param(layer, "weight_zero_point", None)
        input_scale = self._get_layer_param(layer, "input_scale", None)
        input_zero_point = self._get_layer_param(layer, "input_zero_point", None)

        if activation_dynamic_method != "per_token" or not (1 <= activation_bits <= 8):
            return False
        if input_scale is not None or input_zero_point is not None:
            return False
        if source_weight is None or weight_scale is None or weight_zero_point is None:
            return False
        if _parse_bool(self.layer_config, "transpose_weight", False):
            return False
        grouped_runtime = self._materialize_grouped_int_weight_runtime(
            source_weight=source_weight,
            weight_scale=weight_scale,
            weight_zero_point=weight_zero_point,
        )
        if grouped_runtime is None:
            return False

        adjusted_weight, grouped_scales, group_size = grouped_runtime
        if group_size != 128:
            return False

        triton_weight = adjusted_weight.to(dtype=torch.int8).permute(1, 2, 0).contiguous()
        triton_scale = grouped_scales.transpose(0, 1).contiguous()

        if "omni_triton_weight" in layer._buffers:
            layer._buffers["omni_triton_weight"] = triton_weight
        else:
            layer.register_buffer("omni_triton_weight", triton_weight, persistent=False)
        if "omni_triton_scale" in layer._buffers:
            layer._buffers["omni_triton_scale"] = triton_scale
        else:
            layer.register_buffer("omni_triton_scale", triton_scale, persistent=False)
        layer.omni_fast_activation_bits = activation_bits
        layer.omni_fast_activation_symmetric = activation_symmetric
        layer.omni_fast_activation_disable_zero_point = activation_disable_zero_point
        return True

    def _materialize_grouped_int_weight_runtime(
        self,
        *,
        source_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        packing_mode = _weight_packing_mode(
            self.quant_config.runtime_format,
            self.layer_config,
        )
        weight_group_size = _metadata_group_size(self.layer_config, "weight_scale")

        if packing_mode == _PACKED_UINT4X2:
            if source_weight.ndim != 3 or weight_group_size is None:
                return None
            if source_weight.shape[-1] * 2 != weight_group_size:
                return None
            grouped_weight = _unpack_uint4x2(
                source_weight,
                signed=_parse_bool(self.layer_config, "weight_symmetric", False)
                or _parse_bool(
                    self.layer_config,
                    "weight_disable_zero_point",
                    False,
                ),
                unpacked_last_dim=weight_group_size,
            )
            group_size = weight_group_size
        elif packing_mode == _PACKED_UINT6X4:
            if source_weight.ndim != 3 or weight_group_size is None:
                return None
            if source_weight.shape[-1] * 4 != weight_group_size * 3:
                return None
            grouped_weight = _unpack_uint6x4(
                source_weight,
                signed=_parse_bool(self.layer_config, "weight_symmetric", False)
                or _parse_bool(
                    self.layer_config,
                    "weight_disable_zero_point",
                    False,
                ),
                unpacked_last_dim=weight_group_size,
            )
            group_size = weight_group_size
        else:
            if source_weight.ndim != 3:
                return None
            grouped_weight = source_weight.to(dtype=torch.int16)
            group_size = int(source_weight.shape[-1])

        if weight_scale.ndim != 2 or weight_zero_point.ndim != 2:
            return None
        if weight_scale.shape != grouped_weight.shape[:2]:
            return None
        if weight_zero_point.shape != grouped_weight.shape[:2]:
            return None

        adjusted_weight = (
            grouped_weight.to(dtype=torch.int16)
            - weight_zero_point.to(dtype=torch.int16).unsqueeze(-1)
        )
        if adjusted_weight.amin().item() < -128 or adjusted_weight.amax().item() > 127:
            return None
        return adjusted_weight, weight_scale.to(dtype=torch.float32), group_size

    def _build_cutlass_w6a6_runtime(self, layer: torch.nn.Module) -> bool:
        if os.getenv("OMNI_DISABLE_W6A6_FASTPATH", "0").strip() == "1":
            return False
        if not ops.cutlass_scaled_mm_supports_fp8(89):
            return False

        (
            activation_bits,
            activation_symmetric,
            activation_disable_zero_point,
            activation_dynamic_method,
        ) = self._get_activation_runtime_config()
        weight_bits = int(
            self.layer_config.get(
                "weight_bits",
                self.quant_config.global_quant_config.get("weight_bits", 16),
            )
        )
        if weight_bits < 6:
            return False
        if activation_bits <= 4:
            return False
        source_weight = self._get_layer_param(layer, "weight", "weight")
        weight_scale = self._get_layer_param(layer, "weight_scale", "weight_scale")
        weight_zero_point = self._get_layer_param(layer, "weight_zero_point", None)
        input_scale = self._get_layer_param(layer, "input_scale", None)
        input_zero_point = self._get_layer_param(layer, "input_zero_point", None)

        if activation_dynamic_method != "per_token" or not (1 <= activation_bits <= 8):
            return False
        if input_scale is not None or input_zero_point is not None:
            return False
        if source_weight is None or weight_scale is None or weight_zero_point is None:
            return False
        if _parse_bool(self.layer_config, "transpose_weight", False):
            return False
        if source_weight.ndim != 3 or source_weight.shape[-1] != 128:
            return False
        if weight_scale.ndim != 2 or weight_zero_point.ndim != 2:
            return False
        if weight_scale.shape != source_weight.shape[:2]:
            return False
        if weight_zero_point.shape != source_weight.shape[:2]:
            return False

        adjusted_weight = (
            source_weight.to(dtype=torch.int16)
            - weight_zero_point.to(dtype=torch.int16).unsqueeze(-1)
        )
        if adjusted_weight.amin().item() < -128 or adjusted_weight.amax().item() > 127:
            return False

        out_features, num_groups, group_size = source_weight.shape
        cutlass_weight = adjusted_weight.to(dtype=torch.int8).reshape(
            out_features, num_groups * group_size
        ).contiguous()
        cutlass_scale = weight_scale.transpose(0, 1).contiguous()

        if "omni_cutlass_weight" in layer._buffers:
            layer._buffers["omni_cutlass_weight"] = cutlass_weight.t()
        else:
            layer.register_buffer(
                "omni_cutlass_weight",
                cutlass_weight.t(),
                persistent=False,
            )
        if "omni_cutlass_scale" in layer._buffers:
            layer._buffers["omni_cutlass_scale"] = cutlass_scale
        else:
            layer.register_buffer("omni_cutlass_scale", cutlass_scale, persistent=False)
        layer.omni_fast_activation_bits = activation_bits
        layer.omni_fast_activation_symmetric = activation_symmetric
        layer.omni_fast_activation_disable_zero_point = activation_disable_zero_point
        return True

    def _get_or_build_runtime(
        self,
        layer: torch.nn.Module,
    ) -> tuple[torch.Tensor, tuple[_OmniShardRuntime, ...]]:
        dense_weight = layer._buffers.get("omni_weight")
        shard_runtimes = getattr(layer, "omni_shard_runtimes", None)
        if dense_weight is None or not isinstance(shard_runtimes, tuple):
            return self._build_runtime(layer)
        return cast(torch.Tensor, dense_weight), cast(tuple[_OmniShardRuntime, ...], shard_runtimes)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._build_triton_w6a6_runtime(layer):
            layer.omni_fast_path_kind = _OMNI_FAST_PATH_TRITON_W6A6
            self._release_source_tensors(layer)
            return
        if self._build_cutlass_w6a6_runtime(layer):
            layer.omni_fast_path_kind = _OMNI_FAST_PATH_CUTLASS_W6A6
            self._release_source_tensors(layer)
            return

        if self._build_block_int8_runtime(layer):
            layer.omni_fast_path_kind = _OMNI_FAST_PATH_BLOCK_INT8
            self._release_source_tensors(layer)
            return

        dense_weight, shard_runtimes = self._build_runtime(layer)
        if all(
            shard.input_scale is None and shard.input_zero_point is None
            for shard in shard_runtimes
        ):
            layer.omni_fast_path_kind = _OMNI_FAST_PATH_DENSE_LINEAR
            layer.omni_dense_weight = dense_weight
        else:
            layer.omni_fast_path_kind = _OMNI_FAST_PATH_GENERIC
        self._release_source_tensors(layer)

    def _release_source_tensors(self, layer: torch.nn.Module) -> None:
        tensor_names = _get_nested_config(self.layer_config, "tensor_names")
        for tensor_name in (
            tensor_names.get("weight"),
            tensor_names.get("weight_scale"),
            tensor_names.get("weight_zero_point"),
            tensor_names.get("input_scale"),
            tensor_names.get("input_zero_point"),
        ):
            if not tensor_name:
                continue
            attr_name = str(tensor_name)
            if attr_name in layer._parameters:
                replace_parameter(layer, attr_name, torch.empty(0))
            elif attr_name in layer._buffers:
                layer._buffers[attr_name] = torch.empty(0)

    def _fake_quantize_input(
        self,
        x: torch.Tensor,
        input_scale: torch.Tensor | None,
        input_inv_scale: torch.Tensor | None,
        input_zero_point: torch.Tensor | None,
    ) -> torch.Tensor:
        if input_scale is None:
            return x

        scale = input_scale if input_scale.dtype == x.dtype else input_scale.to(dtype=x.dtype)
        inv_scale = (
            input_inv_scale
            if input_inv_scale is not None and input_inv_scale.dtype == x.dtype
            else input_inv_scale.to(dtype=x.dtype)
            if input_inv_scale is not None
            else torch.reciprocal(scale)
        )
        zero_point = (
            input_zero_point
            if input_zero_point is not None and input_zero_point.dtype == x.dtype
            else input_zero_point.to(dtype=x.dtype)
            if input_zero_point is not None
            else None
        )
        if zero_point is None:
            return torch.round(x * inv_scale) * scale

        quantized = torch.round(x * inv_scale + zero_point)
        return (quantized - zero_point) * scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            getattr(layer, "omni_fast_path_kind", _OMNI_FAST_PATH_GENERIC)
            == _OMNI_FAST_PATH_CUTLASS_W6A6
        ):
            return omni_w6a6_cutlass_linear(
                x,
                layer.omni_cutlass_weight,
                layer.omni_cutlass_scale,
                bias=bias,
                activation_bits=int(layer.omni_fast_activation_bits),
                activation_symmetric=bool(layer.omni_fast_activation_symmetric),
                activation_disable_zero_point=bool(
                    layer.omni_fast_activation_disable_zero_point
                ),
            )

        if (
            getattr(layer, "omni_fast_path_kind", _OMNI_FAST_PATH_GENERIC)
            == _OMNI_FAST_PATH_TRITON_W6A6
        ):
            return omni_w6a6_linear(
                x,
                layer.omni_triton_weight,
                layer.omni_triton_scale,
                bias=bias,
                activation_bits=int(layer.omni_fast_activation_bits),
                activation_symmetric=bool(layer.omni_fast_activation_symmetric),
                activation_disable_zero_point=bool(
                    layer.omni_fast_activation_disable_zero_point
                ),
            )

        if (
            getattr(layer, "omni_fast_path_kind", _OMNI_FAST_PATH_GENERIC)
            == _OMNI_FAST_PATH_BLOCK_INT8
        ):
            return apply_w8a8_block_int8_linear(
                x,
                layer.omni_block_weight,
                [1, int(layer.omni_block_group_size)],
                layer.omni_block_scale,
                bias=bias,
            )

        if (
            getattr(layer, "omni_fast_path_kind", _OMNI_FAST_PATH_GENERIC)
            == _OMNI_FAST_PATH_DENSE_LINEAR
        ):
            return F.linear(x, layer.omni_dense_weight, bias)

        dense_weight, shard_runtimes = self._get_or_build_runtime(layer)
        if len(shard_runtimes) == 1:
            shard = shard_runtimes[0]
            quantized_x = self._fake_quantize_input(
                x,
                shard.input_scale,
                shard.input_inv_scale,
                shard.input_zero_point,
            )
            return F.linear(quantized_x, shard.dense_weight, bias)

        if bool(getattr(layer, "omni_shared_input_quant_params", False)):
            first_shard = shard_runtimes[0]
            quantized_x = self._fake_quantize_input(
                x,
                first_shard.input_scale,
                first_shard.input_inv_scale,
                first_shard.input_zero_point,
            )
            return F.linear(quantized_x, dense_weight, bias)

        outputs: list[torch.Tensor] = []
        quantized_input_cache: dict[tuple[int, int], torch.Tensor] = {}
        for shard in shard_runtimes:
            shard_bias = None if bias is None else bias[shard.start : shard.end]
            cache_key = (
                id(shard.input_scale),
                id(shard.input_zero_point),
            )
            quantized_x = quantized_input_cache.get(cache_key)
            if quantized_x is None:
                quantized_x = self._fake_quantize_input(
                    x,
                    shard.input_scale,
                    shard.input_inv_scale,
                    shard.input_zero_point,
                )
                quantized_input_cache[cache_key] = quantized_x
            outputs.append(F.linear(quantized_x, shard.dense_weight, shard_bias))
        return torch.cat(outputs, dim=-1)


class OmniActivationRealAttentionMethod(BaseKVCacheMethod):
    """Attention helper attachment for omni activation real checkpoints."""

    def __init__(
        self,
        quant_config: OmniActivationRealConfig,
        attention_config: dict[str, Any],
    ) -> None:
        super().__init__(quant_config)
        self.attention_config = attention_config

    def create_weights(self, layer: torch.nn.Module) -> None:
        super().create_weights(layer)
        qk_scale_name = self.attention_config.get("qk_scale_name")
        pv_scale_name = self.attention_config.get("pv_scale_name")
        if qk_scale_name:
            qk_scale = Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
            set_weight_attrs(qk_scale, {"ignore_warning": True})
            layer.register_parameter(str(qk_scale_name), qk_scale)
        if pv_scale_name:
            pv_scale = Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
            set_weight_attrs(pv_scale, {"ignore_warning": True})
            layer.register_parameter(str(pv_scale_name), pv_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # Most rebuilt OmniQuant checkpoints do not carry attention runtime
        # metadata. In that case, keep vLLM's native attention backend instead
        # of installing the Python reference helper on every attention layer.
        if not self.attention_config:
            if hasattr(layer, "omni_attention_helper"):
                delattr(layer, "omni_attention_helper")
            return

        qk_scale_name = self.attention_config.get("qk_scale_name")
        pv_scale_name = self.attention_config.get("pv_scale_name")
        qk_scale = (
            float(getattr(layer, str(qk_scale_name)).item())
            if qk_scale_name and hasattr(layer, str(qk_scale_name))
            else float(self.attention_config.get("qk_scale", 1.0))
        )
        pv_scale = (
            float(getattr(layer, str(pv_scale_name)).item())
            if pv_scale_name and hasattr(layer, str(pv_scale_name))
            else float(self.attention_config.get("pv_scale", 1.0))
        )

        layer.omni_attention_helper = OmniActivationRealAttentionHelper(
            qk_scale=qk_scale,
            pv_scale=pv_scale,
            logits_soft_cap=self.attention_config.get("logits_soft_cap"),
            reference_without_forward_context=_parse_bool(
                self.attention_config,
                "reference_without_forward_context",
                True,
            ),
            reference_with_forward_context=_parse_bool(
                self.attention_config,
                "reference_with_forward_context",
                True,
            ),
        )

        if qk_scale_name and hasattr(layer, str(qk_scale_name)):
            delattr(layer, str(qk_scale_name))
        if pv_scale_name and hasattr(layer, str(pv_scale_name)):
            delattr(layer, str(pv_scale_name))

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(
            "OmniActivationRealAttentionMethod.apply should never be called."
        )


def maybe_run_omni_attention(
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
) -> torch.Tensor | None:
    helper = getattr(layer, "omni_attention_helper", None)
    if helper is None:
        return None

    # Import lazily only when the helper is actually installed.
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if is_forward_context_available():
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(getattr(layer, "layer_name", ""))
        else:
            attn_metadata = None
        slot_mapping = forward_context.slot_mapping
        layer_slot_mapping = None
        if isinstance(slot_mapping, dict):
            layer_slot_mapping = slot_mapping.get(getattr(layer, "layer_name", ""))

        return helper.forward_with_forward_context(
            layer,
            query,
            key,
            value,
            attn_metadata=attn_metadata,
            kv_cache=getattr(layer, "kv_cache", None),
            slot_mapping=layer_slot_mapping,
            output_shape=output_shape,
        )

    if not helper.can_run_without_forward_context():
        return None
    if not torch.compiler.is_compiling():
        logger.debug(
            "Running omni activation real reference attention for %s without "
            "forward context.",
            getattr(layer, "layer_name", "<unknown>"),
        )
    return helper.forward_without_forward_context(layer, query, key, value, output_shape)
