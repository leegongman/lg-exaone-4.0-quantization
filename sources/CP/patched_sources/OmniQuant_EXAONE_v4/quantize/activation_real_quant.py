import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.omni_norm import OmniLlamaRMSNorm


RUNTIME_CONFIG_NAME = "omni_act_quant_config.json"
PACKED_4BIT_RUNTIME_FORMAT = "omni_activation_real_v2_packed_4bit"
PACKED_6BIT_RUNTIME_FORMAT = "omni_activation_real_v2_packed_6bit"
PACKED_UINT4X2 = "uint4x2"
PACKED_UINT6X4 = "uint6x4"


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.int8:
        return "int8"
    if dtype == torch.uint8:
        return "uint8"
    if dtype == torch.int16:
        return "int16"
    if dtype == torch.int32:
        return "int32"
    raise ValueError(f"Unsupported dtype name export for {dtype}.")


def _pack_uint4x2(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] % 2 != 0:
        raise ValueError(
            f"Cannot pack uint4 values with odd innermost size {values.shape[-1]}."
        )
    values_i16 = values.to(torch.int16)
    low = values_i16[..., ::2] & 0x0F
    high = (values_i16[..., 1::2] & 0x0F) << 4
    return (low | high).to(torch.uint8).contiguous()


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


def _pack_uint6x4(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] % 4 != 0:
        raise ValueError(
            f"Cannot pack uint6 values with innermost size {values.shape[-1]} that is not divisible by 4."
        )
    values_i32 = values.to(torch.int32) & 0x3F
    reshaped = values_i32.reshape(*values.shape[:-1], values.shape[-1] // 4, 4)
    v0 = reshaped[..., 0]
    v1 = reshaped[..., 1]
    v2 = reshaped[..., 2]
    v3 = reshaped[..., 3]
    packed = torch.empty(
        *values.shape[:-1],
        (values.shape[-1] // 4) * 3,
        dtype=torch.uint8,
        device=values.device,
    )
    packed_view = packed.view(*values.shape[:-1], values.shape[-1] // 4, 3)
    packed_view[..., 0] = (v0 | ((v1 & 0x03) << 6)).to(torch.uint8)
    packed_view[..., 1] = (((v1 >> 2) & 0x0F) | ((v2 & 0x0F) << 4)).to(torch.uint8)
    packed_view[..., 2] = (((v2 >> 4) & 0x03) | ((v3 & 0x3F) << 2)).to(torch.uint8)
    return packed.contiguous()


def _unpack_uint6x4(
    packed: torch.Tensor,
    *,
    signed: bool,
    unpacked_last_dim: int | None = None,
) -> torch.Tensor:
    if packed.shape[-1] % 3 != 0:
        raise ValueError(
            f"Cannot unpack uint6 values with innermost size {packed.shape[-1]} that is not divisible by 3."
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
    unpacked_view[..., 1] = (((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)).to(torch.int16)
    unpacked_view[..., 2] = (((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)).to(torch.int16)
    unpacked_view[..., 3] = ((b2 >> 2) & 0x3F).to(torch.int16)
    if unpacked_last_dim is not None:
        unpacked = unpacked[..., :unpacked_last_dim]
    if signed:
        unpacked = torch.where(unpacked >= 32, unpacked - 64, unpacked)
    return unpacked.contiguous()


def _storage_dtype_for_bits(n_bits: int) -> torch.dtype:
    return torch.int8 if n_bits <= 8 else torch.int16


def _quant_bounds(n_bits: int, symmetric: bool, disable_zero_point: bool) -> tuple[int, int]:
    if disable_zero_point or symmetric:
        return -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
    return 0, 2**n_bits - 1


def _quantize_activation_per_token(
    x: torch.Tensor,
    n_bits: int,
    symmetric: bool = False,
    disable_zero_point: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    qmin, qmax = _quant_bounds(n_bits, symmetric, disable_zero_point)
    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    if symmetric:
        abs_max = torch.max(xmax.abs(), xmin.abs())
        scale = (abs_max / max(qmax, 1)).clamp(min=1e-5)
        zero_point = None if disable_zero_point else torch.full_like(scale, qmax)
    else:
        scale = ((xmax - xmin) / max(qmax - qmin, 1)).clamp(min=1e-5)
        zero_point = None if disable_zero_point else (-(xmin) / scale).round().clamp(qmin, qmax)

    q = torch.round(x / scale)
    if zero_point is not None:
        q = q + zero_point
    q = q.clamp(qmin, qmax).to(torch.int16)
    return q, scale.to(torch.float32), None if zero_point is None else zero_point.to(torch.int16)


def _quantize_activation_runtime(
    x: torch.Tensor,
    n_bits: int,
    symmetric: bool = False,
    disable_zero_point: bool = False,
    metric: str = "minmax",
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if n_bits >= 16:
        return x.to(torch.float32), torch.ones_like(x[..., :1], dtype=torch.float32), None
    if metric == "fix0to1":
        qmax = 2**n_bits - 1
        q = torch.round(x.clamp(0, 1) * qmax).to(torch.int16)
        scale = torch.full_like(x[..., :1], 1.0 / max(qmax, 1), dtype=torch.float32)
        return q, scale, None
    return _quantize_activation_per_token(
        x,
        n_bits=n_bits,
        symmetric=symmetric,
        disable_zero_point=disable_zero_point,
    )


def _quantize_weight_from_module(module: QuantLinear) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    quantizer = module.weight_quantizer
    assert hasattr(quantizer, "scales") and hasattr(quantizer, "zeros"), (
        "Weight quantizer must have registered scales/zeros before activation-real export."
    )
    weight = module.weight.detach().to(torch.float32)
    group_size = quantizer.group_size or weight.shape[1]
    padded_in_features = ((weight.shape[1] + group_size - 1) // group_size) * group_size
    deficiency = padded_in_features - weight.shape[1]
    if deficiency > 0:
        weight = torch.cat(
            [weight, torch.zeros(weight.shape[0], deficiency, device=weight.device, dtype=weight.dtype)],
            dim=1,
        )

    num_groups = padded_in_features // group_size
    weight_groups = weight.view(weight.shape[0], num_groups, group_size)
    w_scales = quantizer.scales.detach().to(torch.float32).view(weight.shape[0], num_groups, 1)
    if quantizer.zeros is None:
        w_zero_points = torch.zeros_like(w_scales, dtype=torch.int16)
        has_zero_point = False
    else:
        w_zero_points = quantizer.zeros.detach().to(torch.int16).view(weight.shape[0], num_groups, 1)
        has_zero_point = True

    qmin, qmax = _quant_bounds(quantizer.n_bits, quantizer.symmetric, quantizer.disable_zero_point)
    storage_dtype = _storage_dtype_for_bits(quantizer.n_bits)
    qweight = torch.round(weight_groups / w_scales)
    if has_zero_point:
        qweight = qweight + w_zero_points.to(torch.float32)
    qweight = qweight.clamp(qmin, qmax).to(storage_dtype)
    weight_packing = "plain"
    checkpoint_weight_dtype = storage_dtype
    if quantizer.n_bits == 4:
        qweight = _pack_uint4x2(qweight)
        weight_packing = PACKED_UINT4X2
        checkpoint_weight_dtype = torch.uint8
    elif quantizer.n_bits == 6:
        qweight = _pack_uint6x4(qweight)
        weight_packing = PACKED_UINT6X4
        checkpoint_weight_dtype = torch.uint8

    w_zero_points = w_zero_points.to(torch.uint8)

    metadata = {
        "type": "linear",
        "in_features": module.in_features,
        "out_features": module.out_features,
        "group_size": group_size,
        "padded_in_features": padded_in_features,
        "weight_bits": quantizer.n_bits,
        "weight_symmetric": bool(quantizer.symmetric),
        "weight_disable_zero_point": bool(quantizer.disable_zero_point),
        "weight_packing": weight_packing,
        "checkpoint_weight_dtype": _dtype_name(checkpoint_weight_dtype),
        "weight_scale_dtype": _dtype_name(w_scales.dtype),
        "weight_zero_point_dtype": _dtype_name(w_zero_points.dtype),
        "activation_bits": getattr(module.act_quantizer, "n_bits", 16) if module.act_quantizer is not None else 16,
        "activation_symmetric": bool(getattr(module.act_quantizer, "symmetric", False)) if module.act_quantizer is not None else False,
        "activation_disable_zero_point": bool(getattr(module.act_quantizer, "disable_zero_point", False)) if module.act_quantizer is not None else False,
        "activation_dynamic_method": getattr(module.act_quantizer, "dynamic_method", "per_token") if module.act_quantizer is not None else "per_token",
    }
    return qweight.cpu(), w_scales.squeeze(-1).cpu(), w_zero_points.squeeze(-1).cpu(), metadata


def _collect_non_quant_tensors(model: nn.Module) -> Dict[str, torch.Tensor]:
    skip_prefixes = []
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantMatMul)):
            skip_prefixes.append(name)

    tensors: Dict[str, torch.Tensor] = {}
    seen_storages = set()
    for key, value in model.state_dict().items():
        if "smooth" in key or "bound_factor" in key:
            continue
        if any(key == prefix or key.startswith(prefix + ".") for prefix in skip_prefixes):
            continue
        storage_ptr = value.untyped_storage().data_ptr()
        if storage_ptr in seen_storages:
            continue
        seen_storages.add(storage_ptr)
        tensors[key] = value.detach().cpu()
    return tensors


def _collect_runtime_metadata(model: nn.Module) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    tensors = _collect_non_quant_tensors(model)
    runtime_cfg: Dict[str, Any] = {"format": "omni_activation_real_v1", "modules": {}}
    saw_packed_4bit = False
    saw_packed_6bit = False

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            qweight, w_scales, w_zero_points, metadata = _quantize_weight_from_module(module)
            tensors[f"{name}.qweight"] = qweight
            tensors[f"{name}.w_scales"] = w_scales
            tensors[f"{name}.w_zero_points"] = w_zero_points
            if module.bias is not None:
                tensors[f"{name}.bias"] = module.bias.detach().cpu()
            runtime_cfg["modules"][name] = metadata
            saw_packed_4bit = saw_packed_4bit or metadata["weight_packing"] == PACKED_UINT4X2
            saw_packed_6bit = saw_packed_6bit or metadata["weight_packing"] == PACKED_UINT6X4
        elif isinstance(module, QuantMatMul):
            runtime_cfg["modules"][name] = {
                "type": "matmul",
                "x1_bits": module.x1_quantizer.n_bits,
                "x2_bits": module.x2_quantizer.n_bits,
                "x1_metric": module.x1_quantizer.metric,
                "x2_metric": module.x2_quantizer.metric,
                "x1_symmetric": bool(module.x1_quantizer.symmetric),
                "x2_symmetric": bool(module.x2_quantizer.symmetric),
                "x1_disable_zero_point": bool(module.x1_quantizer.disable_zero_point),
                "x2_disable_zero_point": bool(module.x2_quantizer.disable_zero_point),
            }
    if saw_packed_6bit:
        runtime_cfg["format"] = PACKED_6BIT_RUNTIME_FORMAT
    elif saw_packed_4bit:
        runtime_cfg["format"] = PACKED_4BIT_RUNTIME_FORMAT
    return tensors, runtime_cfg


class ActivationRealQuantLinear(nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        w_scales: torch.Tensor,
        w_zero_points: torch.Tensor,
        bias: Optional[torch.Tensor],
        metadata: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.in_features = metadata["in_features"]
        self.out_features = metadata["out_features"]
        self.group_size = metadata["group_size"]
        self.padded_in_features = metadata["padded_in_features"]
        self.weight_bits = metadata["weight_bits"]
        self.weight_symmetric = metadata["weight_symmetric"]
        self.weight_disable_zero_point = metadata["weight_disable_zero_point"]
        self.weight_packing = metadata.get("weight_packing", "plain")
        self.activation_bits = metadata["activation_bits"]
        self.activation_symmetric = metadata["activation_symmetric"]
        self.activation_disable_zero_point = metadata["activation_disable_zero_point"]
        self.activation_dynamic_method = metadata["activation_dynamic_method"]

        if self.weight_packing == PACKED_UINT4X2:
            qweight = _unpack_uint4x2(
                qweight,
                signed=self.weight_symmetric or self.weight_disable_zero_point,
                unpacked_last_dim=self.group_size,
            )
        elif self.weight_packing == PACKED_UINT6X4:
            qweight = _unpack_uint6x4(
                qweight,
                signed=self.weight_symmetric or self.weight_disable_zero_point,
                unpacked_last_dim=self.group_size,
            )

        self.register_buffer("qweight", qweight.to(torch.int16))
        self.register_buffer("w_scales", w_scales.to(torch.float32))
        self.register_buffer("w_zero_points", w_zero_points.to(torch.int16))
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float32))
        else:
            self.bias = None
        self.runtime_act_quant_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.runtime_act_quant_calls += 1
        input_dtype = x.dtype
        qx, x_scale, x_zero_point = _quantize_activation_per_token(
            x.to(torch.float32),
            self.activation_bits,
            symmetric=self.activation_symmetric,
            disable_zero_point=self.activation_disable_zero_point,
        )
        if x_zero_point is None:
            centered_x = qx.to(torch.float32)
        else:
            centered_x = qx.to(torch.float32) - x_zero_point.to(torch.float32)

        if self.padded_in_features > self.in_features:
            deficiency = self.padded_in_features - self.in_features
            centered_x = torch.cat(
                [centered_x, torch.zeros(*centered_x.shape[:-1], deficiency, device=centered_x.device, dtype=centered_x.dtype)],
                dim=-1,
            )

        centered_x = centered_x.view(*centered_x.shape[:-1], -1, self.group_size)
        centered_w = self.qweight.to(torch.float32) - self.w_zero_points.to(torch.float32).unsqueeze(-1)
        acc = torch.einsum("...gc,ogc->...og", centered_x, centered_w)
        out = (acc * self.w_scales.view(*([1] * (acc.dim() - 2)), self.out_features, -1)).sum(dim=-1)
        out = out * x_scale.to(out.dtype)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.to(input_dtype)


class ActivationRealQuantMatMul(nn.Module):
    def __init__(self, metadata: Dict[str, Any], matmul_func=torch.matmul) -> None:
        super().__init__()
        self.x1_bits = metadata["x1_bits"]
        self.x2_bits = metadata["x2_bits"]
        self.x1_metric = metadata.get("x1_metric", "minmax")
        self.x2_metric = metadata.get("x2_metric", "minmax")
        self.x1_symmetric = metadata["x1_symmetric"]
        self.x2_symmetric = metadata["x2_symmetric"]
        self.x1_disable_zero_point = metadata["x1_disable_zero_point"]
        self.x2_disable_zero_point = metadata["x2_disable_zero_point"]
        self.matmul_func = matmul_func
        self.runtime_act_quant_calls = 0

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        transpose_x2: bool = False,
    ) -> torch.Tensor:
        self.runtime_act_quant_calls += 1
        input_dtype = x1.dtype
        q1, s1, z1 = _quantize_activation_runtime(
            x1.to(torch.float32),
            self.x1_bits,
            symmetric=self.x1_symmetric,
            disable_zero_point=self.x1_disable_zero_point,
            metric=self.x1_metric,
        )
        q2, s2, z2 = _quantize_activation_runtime(
            x2.to(torch.float32),
            self.x2_bits,
            symmetric=self.x2_symmetric,
            disable_zero_point=self.x2_disable_zero_point,
            metric=self.x2_metric,
        )
        x1_centered = q1.to(torch.float32) if z1 is None else q1.to(torch.float32) - z1.to(torch.float32)
        x2_centered = q2.to(torch.float32) if z2 is None else q2.to(torch.float32) - z2.to(torch.float32)
        rhs = x2_centered * s2.to(x2_centered.dtype)
        if transpose_x2:
            rhs = rhs.transpose(-1, -2)
        out = self.matmul_func(x1_centered, rhs)
        out = out * s1.to(out.dtype)
        return out.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RuntimeExaone4MLP(nn.Module):
    def __init__(self, org_module: nn.Module, module_prefix: str, tensors: Dict[str, torch.Tensor], runtime_cfg: Dict[str, Any]):
        super().__init__()
        self.gate_proj = ActivationRealQuantLinear(
            tensors[f"{module_prefix}.gate_proj.qweight"],
            tensors[f"{module_prefix}.gate_proj.w_scales"],
            tensors[f"{module_prefix}.gate_proj.w_zero_points"],
            tensors.get(f"{module_prefix}.gate_proj.bias"),
            runtime_cfg[f"{module_prefix}.gate_proj"],
        )
        self.up_proj = ActivationRealQuantLinear(
            tensors[f"{module_prefix}.up_proj.qweight"],
            tensors[f"{module_prefix}.up_proj.w_scales"],
            tensors[f"{module_prefix}.up_proj.w_zero_points"],
            tensors.get(f"{module_prefix}.up_proj.bias"),
            runtime_cfg[f"{module_prefix}.up_proj"],
        )
        self.down_proj = ActivationRealQuantLinear(
            tensors[f"{module_prefix}.down_proj.qweight"],
            tensors[f"{module_prefix}.down_proj.w_scales"],
            tensors[f"{module_prefix}.down_proj.w_zero_points"],
            tensors.get(f"{module_prefix}.down_proj.bias"),
            runtime_cfg[f"{module_prefix}.down_proj"],
        )
        self.act_fn = org_module.act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class RuntimeExaone4Attention(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        config,
        module_prefix: str,
        tensors: Dict[str, torch.Tensor],
        runtime_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.config = config
        self.layer_idx = org_module.layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        layer_type = config.layer_types[self.layer_idx] if hasattr(config, "layer_types") else None
        self.is_sliding = layer_type == "sliding_attention"

        self.q_proj = ActivationRealQuantLinear(
            tensors[f"{module_prefix}.q_proj.qweight"],
            tensors[f"{module_prefix}.q_proj.w_scales"],
            tensors[f"{module_prefix}.q_proj.w_zero_points"],
            tensors.get(f"{module_prefix}.q_proj.bias"),
            runtime_cfg[f"{module_prefix}.q_proj"],
        )
        self.k_proj = ActivationRealQuantLinear(
            tensors[f"{module_prefix}.k_proj.qweight"],
            tensors[f"{module_prefix}.k_proj.w_scales"],
            tensors[f"{module_prefix}.k_proj.w_zero_points"],
            tensors.get(f"{module_prefix}.k_proj.bias"),
            runtime_cfg[f"{module_prefix}.k_proj"],
        )
        self.v_proj = ActivationRealQuantLinear(
            tensors[f"{module_prefix}.v_proj.qweight"],
            tensors[f"{module_prefix}.v_proj.w_scales"],
            tensors[f"{module_prefix}.v_proj.w_zero_points"],
            tensors.get(f"{module_prefix}.v_proj.bias"),
            runtime_cfg[f"{module_prefix}.v_proj"],
        )
        self.o_proj = ActivationRealQuantLinear(
            tensors[f"{module_prefix}.o_proj.qweight"],
            tensors[f"{module_prefix}.o_proj.w_scales"],
            tensors[f"{module_prefix}.o_proj.w_zero_points"],
            tensors.get(f"{module_prefix}.o_proj.bias"),
            runtime_cfg[f"{module_prefix}.o_proj"],
        )
        self.q_norm = OmniLlamaRMSNorm(org_module.q_norm, eps=org_module.q_norm.variance_epsilon)
        self.k_norm = OmniLlamaRMSNorm(org_module.k_norm, eps=org_module.k_norm.variance_epsilon)
        self.qkt_matmul = ActivationRealQuantMatMul(runtime_cfg[f"{module_prefix}.qkt_matmul"], matmul_func=torch.matmul)
        self.pv_matmul = ActivationRealQuantMatMul(runtime_cfg[f"{module_prefix}.pv_matmul"], matmul_func=torch.matmul)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        del position_ids, cache_position, use_cache, kwargs
        input_shape = hidden_states.shape[:-1]
        query_states = self.q_proj(hidden_states).view(*input_shape, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        if self.sliding_window is None or self.is_sliding:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = self.qkt_matmul(query_states, key_states, transpose_x2=True) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout)
        attn_output = self.pv_matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class RuntimeExaone4DecoderLayer(nn.Module):
    def __init__(self, config, org_layer: nn.Module, layer_prefix: str, tensors: Dict[str, torch.Tensor], runtime_cfg: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = RuntimeExaone4Attention(
            org_layer.self_attn,
            config=config,
            module_prefix=f"{layer_prefix}.self_attn",
            tensors=tensors,
            runtime_cfg=runtime_cfg,
        )
        self.mlp = RuntimeExaone4MLP(
            org_layer.mlp,
            module_prefix=f"{layer_prefix}.mlp",
            tensors=tensors,
            runtime_cfg=runtime_cfg,
        )
        self.post_attention_layernorm = OmniLlamaRMSNorm(
            org_layer.post_attention_layernorm,
            eps=org_layer.post_attention_layernorm.variance_epsilon,
        )
        self.post_feedforward_layernorm = OmniLlamaRMSNorm(
            org_layer.post_feedforward_layernorm,
            eps=org_layer.post_feedforward_layernorm.variance_epsilon,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        del self_attn_weights, past_key_values, output_attentions, use_cache
        return hidden_states


def save_exaone_activation_real_quant_model(model, tokenizer, save_dir: str, args) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    tensors, runtime_cfg = _collect_runtime_metadata(model)
    save_file(tensors, str(save_path / "model.safetensors"))
    runtime_format = runtime_cfg.get("format", "omni_activation_real_v1")

    model.config.quantization_config = {
        "quant_method": "omni_activation_real",
        "format": runtime_format,
        "runtime_format": runtime_format,
        "bits": int(args.wbits),
        "activation_bits": int(args.abits),
        "group_size": args.group_size if args.group_size is not None else -1,
        "sym": bool(getattr(args, "symmetric", False) or getattr(args, "disable_zero_point", False)),
        "true_sequential": True,
        "runtime": "custom_hf",
    }
    # vLLM/HF loaders look at `torch_dtype` when deciding the runtime dtype.
    # Without this, the exported checkpoint can be treated as fp32 and then
    # downcasted to bf16 at load time, which changes the intended numerics.
    model.config.dtype = "float16"
    model.config.torch_dtype = "float16"
    model.config.save_pretrained(save_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    runtime_cfg["model_type"] = "exaone4_activation_real_quant"
    runtime_cfg["weight_bits"] = args.wbits
    runtime_cfg["activation_bits"] = args.abits
    runtime_cfg["group_size"] = args.group_size
    runtime_cfg["let"] = bool(args.let)
    runtime_cfg["lwc"] = bool(args.lwc)
    runtime_cfg["base_model"] = args.model
    (save_path / RUNTIME_CONFIG_NAME).write_text(
        json.dumps(runtime_cfg, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_exaone_activation_real_quant_model(
    save_dir: str,
    trust_remote_code: bool = False,
    device: str = "cpu",
):
    save_path = Path(save_dir)
    runtime_cfg = json.loads((save_path / RUNTIME_CONFIG_NAME).read_text(encoding="utf-8"))
    tensors = load_file(str(save_path / "model.safetensors"))

    config = AutoConfig.from_pretrained(save_dir, trust_remote_code=trust_remote_code, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=trust_remote_code, local_files_only=True)

    non_quant_state = {
        key: value
        for key, value in tensors.items()
        if not (
            key.endswith(".qweight")
            or key.endswith(".w_scales")
            or key.endswith(".w_zero_points")
        )
    }
    model.load_state_dict(non_quant_state, strict=False)

    for layer_idx in range(len(model.model.layers)):
        prefix = f"model.layers.{layer_idx}"
        model.model.layers[layer_idx] = RuntimeExaone4DecoderLayer(
            config,
            model.model.layers[layer_idx],
            prefix,
            tensors,
            runtime_cfg["modules"],
        )

    model.to(device)
    model.eval()
    return model, tokenizer, runtime_cfg


def repack_exaone_activation_real_checkpoint(
    source_dir: str,
    output_dir: str | None = None,
) -> str:
    source_path = Path(source_dir)
    output_path = Path(output_dir) if output_dir is not None else source_path.with_name(source_path.name + "_packed")
    output_path.mkdir(parents=True, exist_ok=True)

    runtime_cfg = json.loads((source_path / RUNTIME_CONFIG_NAME).read_text(encoding="utf-8"))
    tensors = load_file(str(source_path / "model.safetensors"))

    saw_packed_4bit = False
    saw_packed_6bit = False
    for module_name, metadata in runtime_cfg.get("modules", {}).items():
        if metadata.get("type") != "linear":
            continue

        qweight_key = f"{module_name}.qweight"
        zero_key = f"{module_name}.w_zero_points"
        if qweight_key not in tensors or zero_key not in tensors:
            continue

        qweight = tensors[qweight_key]
        weight_bits = int(metadata.get("weight_bits", 0))
        if weight_bits == 4:
            if qweight.ndim != 3 or qweight.shape[-1] % 2 != 0:
                raise ValueError(
                    f"Expected unpacked 4-bit qweight with even group width, got {qweight_key} {tuple(qweight.shape)}."
                )
            tensors[qweight_key] = _pack_uint4x2(qweight)
            metadata["weight_packing"] = PACKED_UINT4X2
            saw_packed_4bit = True
        elif weight_bits == 6:
            if qweight.ndim != 3 or qweight.shape[-1] % 4 != 0:
                raise ValueError(
                    f"Expected unpacked 6-bit qweight with group width divisible by 4, got {qweight_key} {tuple(qweight.shape)}."
                )
            tensors[qweight_key] = _pack_uint6x4(qweight)
            metadata["weight_packing"] = PACKED_UINT6X4
            saw_packed_6bit = True
        else:
            continue

        tensors[zero_key] = tensors[zero_key].to(torch.uint8)
        metadata["checkpoint_weight_dtype"] = "uint8"
        metadata["weight_zero_point_dtype"] = "uint8"

    if saw_packed_6bit:
        runtime_cfg["format"] = PACKED_6BIT_RUNTIME_FORMAT
    elif saw_packed_4bit:
        runtime_cfg["format"] = PACKED_4BIT_RUNTIME_FORMAT

    save_file(tensors, str(output_path / "model.safetensors"))
    (output_path / RUNTIME_CONFIG_NAME).write_text(
        json.dumps(runtime_cfg, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = json.loads((source_path / "config.json").read_text(encoding="utf-8"))
    quant_cfg = dict(config.get("quantization_config", {}))
    if saw_packed_6bit:
        quant_cfg["format"] = PACKED_6BIT_RUNTIME_FORMAT
        quant_cfg["runtime_format"] = PACKED_6BIT_RUNTIME_FORMAT
    elif saw_packed_4bit:
        quant_cfg["format"] = PACKED_4BIT_RUNTIME_FORMAT
        quant_cfg["runtime_format"] = PACKED_4BIT_RUNTIME_FORMAT
    config["quantization_config"] = quant_cfg
    (output_path / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for extra_name in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
    ):
        src = source_path / extra_name
        if src.exists():
            shutil.copy2(src, output_path / extra_name)

    return str(output_path)


def verify_activation_real_quant_checkpoint(save_dir: str) -> Dict[str, Any]:
    save_path = Path(save_dir)
    with (save_path / "model.safetensors").open("rb") as handle:
        import struct

        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    max_end = 0
    for tensor_name, tensor_info in header.items():
        if tensor_name == "__metadata__":
            continue
        begin, end = tensor_info["data_offsets"]
        if begin > end:
            raise ValueError(f"Invalid offsets for {tensor_name}")
        max_end = max(max_end, end)

    actual_size = (save_path / "model.safetensors").stat().st_size
    expected_size = 8 + header_len + max_end
    runtime_cfg = json.loads((save_path / RUNTIME_CONFIG_NAME).read_text(encoding="utf-8"))
    config = json.loads((save_path / "config.json").read_text(encoding="utf-8"))
    return {
        "valid_safetensors": actual_size == expected_size,
        "expected_size": expected_size,
        "actual_size": actual_size,
        "has_runtime_config": (save_path / RUNTIME_CONFIG_NAME).exists(),
        "module_count": len(runtime_cfg.get("modules", {})),
        "quantization_config": config.get("quantization_config"),
    }
