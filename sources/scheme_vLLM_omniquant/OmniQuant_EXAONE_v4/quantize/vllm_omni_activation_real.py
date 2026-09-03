import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import Parameter


try:
    from vllm.model_executor.layers.linear import (
        LinearMethodBase,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )
    from vllm.model_executor.utils import set_weight_attrs
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("vLLM is required to use the omni_activation_real quantization scaffold.") from exc


RUNTIME_CONFIG_NAME = "omni_act_quant_config.json"


def _storage_dtype_for_bits(n_bits: int) -> torch.dtype:
    return torch.int8 if n_bits <= 8 else torch.int16


def _quant_bounds(n_bits: int, symmetric: bool, disable_zero_point: bool) -> tuple[int, int]:
    if disable_zero_point or symmetric:
        return -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
    return 0, 2**n_bits - 1


def _quantize_activation_runtime(
    x: torch.Tensor,
    n_bits: int,
    symmetric: bool = False,
    disable_zero_point: bool = False,
    metric: str = "minmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if n_bits >= 16:
        return x.to(torch.float32), torch.ones_like(x[..., :1], dtype=torch.float32), None

    if metric == "fix0to1":
        qmax = 2**n_bits - 1
        q = torch.round(x.clamp(0, 1) * qmax).to(torch.int16)
        scale = torch.full_like(x[..., :1], 1.0 / max(qmax, 1), dtype=torch.float32)
        return q, scale, None

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


class OmniActivationRealMatMul(nn.Module):
    def __init__(self, metadata: dict[str, Any], matmul_func=torch.matmul) -> None:
        super().__init__()
        self.x1_bits = int(metadata["x1_bits"])
        self.x2_bits = int(metadata["x2_bits"])
        self.x1_metric = metadata.get("x1_metric", "minmax")
        self.x2_metric = metadata.get("x2_metric", "minmax")
        self.x1_symmetric = bool(metadata["x1_symmetric"])
        self.x2_symmetric = bool(metadata["x2_symmetric"])
        self.x1_disable_zero_point = bool(metadata["x1_disable_zero_point"])
        self.x2_disable_zero_point = bool(metadata["x2_disable_zero_point"])
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
            n_bits=self.x1_bits,
            symmetric=self.x1_symmetric,
            disable_zero_point=self.x1_disable_zero_point,
            metric=self.x1_metric,
        )
        q2, s2, z2 = _quantize_activation_runtime(
            x2.to(torch.float32),
            n_bits=self.x2_bits,
            symmetric=self.x2_symmetric,
            disable_zero_point=self.x2_disable_zero_point,
            metric=self.x2_metric,
        )
        x1_centered = q1.to(torch.float32) if z1 is None else q1.to(torch.float32) - z1.to(torch.float32)
        x2_centered = q2.to(torch.float32) if z2 is None else q2.to(torch.float32) - z2.to(torch.float32)
        rhs = x2_centered * s2.to(torch.float32)
        if transpose_x2:
            rhs = rhs.transpose(-1, -2)
        out = self.matmul_func(x1_centered, rhs)
        out = out * s1.to(torch.float32)
        return out.to(input_dtype)


def load_omni_activation_real_runtime_config(model_dir: str) -> dict[str, Any]:
    model_dir = Path(model_dir)
    return json.loads((model_dir / RUNTIME_CONFIG_NAME).read_text())


def build_omni_activation_real_config_from_model_dir(model_dir: str) -> "OmniActivationRealConfig":
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    quant_cfg = config["quantization_config"]
    return OmniActivationRealConfig.from_model_dir(str(model_dir), quant_cfg)


def summarize_omni_activation_real_checkpoint_for_vllm(model_dir: str) -> dict[str, Any]:
    runtime_cfg = load_omni_activation_real_runtime_config(model_dir)
    modules = runtime_cfg.get("modules", {})
    linear_modules = [name for name, meta in modules.items() if meta.get("type") == "linear"]
    matmul_modules = [name for name, meta in modules.items() if meta.get("type") == "matmul"]

    qkv_triplets = sorted(
        {
            name.rsplit(".", 1)[0]
            for name in linear_modules
            if name.endswith(".q_proj") or name.endswith(".k_proj") or name.endswith(".v_proj")
        }
    )
    gate_up_pairs = sorted(
        {
            name.rsplit(".", 1)[0]
            for name in linear_modules
            if name.endswith(".gate_proj") or name.endswith(".up_proj")
        }
    )
    direct_row_linears = sorted(
        [
            name
            for name in linear_modules
            if name.endswith(".o_proj") or name.endswith(".down_proj")
        ]
    )
    return {
        "format": runtime_cfg.get("format"),
        "model_type": runtime_cfg.get("model_type"),
        "module_count": len(modules),
        "linear_module_count": len(linear_modules),
        "matmul_module_count": len(matmul_modules),
        "qkv_fused_target_count": len(qkv_triplets),
        "gate_up_fused_target_count": len(gate_up_pairs),
        "direct_row_linear_count": len(direct_row_linears),
        "requires_qkv_fusion_loader": bool(qkv_triplets),
        "requires_gate_up_fusion_loader": bool(gate_up_pairs),
        "requires_attention_backend": bool(matmul_modules),
        "sample_qkv_prefixes": qkv_triplets[:3],
        "sample_gate_up_prefixes": gate_up_pairs[:3],
    }


def _slice_output_shard(loaded_weight: torch.Tensor, shard_size: int, shard_rank: int) -> torch.Tensor:
    if loaded_weight.shape[0] == shard_size:
        return loaded_weight
    start = shard_rank * shard_size
    return loaded_weight.narrow(0, start, shard_size)


def _slice_group_shard(loaded_weight: torch.Tensor, shard_groups: int, tp_rank: int) -> torch.Tensor:
    if loaded_weight.shape[1] == shard_groups:
        return loaded_weight
    start = tp_rank * shard_groups
    return loaded_weight.narrow(1, start, shard_groups)


def _load_row_param(param: Parameter, layer: RowParallelLinear, loaded_weight: torch.Tensor) -> None:
    target = param.data
    if target.dim() == 3:
        loaded_weight = _slice_group_shard(loaded_weight, target.shape[1], layer.tp_rank)
    elif target.dim() == 2 and param.omni_param_role in {"w_scales", "w_zero_points"}:
        loaded_weight = _slice_group_shard(loaded_weight, target.shape[1], layer.tp_rank)
    assert target.shape == loaded_weight.shape, (target.shape, loaded_weight.shape)
    target.copy_(loaded_weight.to(target.dtype))


def _load_merged_param(
    param: Parameter,
    layer: MergedColumnParallelLinear,
    loaded_weight: torch.Tensor,
    loaded_shard_id: int | None,
) -> None:
    target = param.data
    if loaded_shard_id is None:
        assert target.shape == loaded_weight.shape, (target.shape, loaded_weight.shape)
        target.copy_(loaded_weight.to(target.dtype))
        return

    local_sizes = [size // layer.tp_size for size in layer.output_sizes]
    fused_offset = sum(local_sizes[:loaded_shard_id])
    shard_size = local_sizes[loaded_shard_id]
    loaded_weight = _slice_output_shard(loaded_weight, shard_size, layer.tp_rank)
    target = target.narrow(0, fused_offset, shard_size)
    assert target.shape == loaded_weight.shape, (target.shape, loaded_weight.shape)
    target.copy_(loaded_weight.to(target.dtype))


def _load_qkv_param(
    param: Parameter,
    layer: QKVParallelLinear,
    loaded_weight: torch.Tensor,
    loaded_shard_id: str | None,
) -> None:
    target = param.data
    if loaded_shard_id is None:
        assert target.shape == loaded_weight.shape, (target.shape, loaded_weight.shape)
        target.copy_(loaded_weight.to(target.dtype))
        return

    shard_offset = layer._get_shard_offset_mapping(loaded_shard_id)
    shard_size = layer._get_shard_size_mapping(loaded_shard_id)
    shard_rank = layer.tp_rank if loaded_shard_id == "q" else layer.tp_rank // layer.num_kv_head_replicas
    loaded_weight = _slice_output_shard(loaded_weight, shard_size, shard_rank)
    target = target.narrow(0, shard_offset, shard_size)
    assert target.shape == loaded_weight.shape, (target.shape, loaded_weight.shape)
    target.copy_(loaded_weight.to(target.dtype))


def load_fused_linear_weights_from_state_dict(
    layer: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, Any]:
    if isinstance(layer, QKVParallelLinear):
        shard_map = [("q", f"{prefix}.q_proj"), ("k", f"{prefix}.k_proj"), ("v", f"{prefix}.v_proj")]
    elif isinstance(layer, MergedColumnParallelLinear):
        shard_map = [(0, f"{prefix}.gate_proj"), (1, f"{prefix}.up_proj")]
    elif isinstance(layer, RowParallelLinear):
        shard_map = [(None, prefix)]
    else:
        raise TypeError(f"Unsupported layer type: {type(layer).__name__}")

    loaded_shapes: dict[str, Any] = {}
    for shard_id, shard_prefix in shard_map:
        for tensor_name in ("qweight", "w_scales", "w_zero_points"):
            key = f"{shard_prefix}.{tensor_name}"
            loaded_shapes[key] = list(state_dict[key].shape)
            param = getattr(layer, tensor_name)
            if isinstance(layer, QKVParallelLinear):
                param.weight_loader(param, state_dict[key], shard_id)
            elif isinstance(layer, MergedColumnParallelLinear):
                param.weight_loader(param, state_dict[key], shard_id)
            else:
                param.weight_loader(param, state_dict[key])
    return loaded_shapes


class OmniActivationRealLinearMethod(LinearMethodBase):
    def __init__(
        self,
        quant_config: "OmniActivationRealConfig",
        prefix: str,
        layer_kind: str,
    ):
        self.quant_config = quant_config
        self.prefix = prefix
        self.layer_kind = layer_kind

    def _weight_loader(self, layer: nn.Module, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id=None):
        if isinstance(layer, QKVParallelLinear):
            _load_qkv_param(param, layer, loaded_weight, loaded_shard_id)
            return
        if isinstance(layer, MergedColumnParallelLinear):
            _load_merged_param(param, layer, loaded_weight, loaded_shard_id)
            return
        if isinstance(layer, RowParallelLinear):
            _load_row_param(param, layer, loaded_weight)
            return
        raise TypeError(f"Unsupported layer type for omni_activation_real: {type(layer).__name__}")

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size, params_dtype, extra_weight_attrs
        group_size = self.quant_config.group_size if self.quant_config.group_size != -1 else input_size_per_partition
        padded_in_features = ((input_size_per_partition + group_size - 1) // group_size) * group_size
        num_groups = padded_in_features // group_size
        storage_dtype = _storage_dtype_for_bits(self.quant_config.weight_bits)
        output_size_per_partition = sum(output_partition_sizes)

        qweight = Parameter(
            torch.empty(output_size_per_partition, num_groups, group_size, dtype=storage_dtype),
            requires_grad=False,
        )
        w_scales = Parameter(
            torch.empty(output_size_per_partition, num_groups, dtype=torch.float32),
            requires_grad=False,
        )
        w_zero_points = Parameter(
            torch.empty(output_size_per_partition, num_groups, dtype=storage_dtype),
            requires_grad=False,
        )

        for tensor_name, param in (
            ("qweight", qweight),
            ("w_scales", w_scales),
            ("w_zero_points", w_zero_points),
        ):
            set_weight_attrs(
                param,
                {
                    "weight_loader": lambda p, loaded_weight, loaded_shard_id=None, layer_ref=layer: self._weight_loader(
                        layer_ref, p, loaded_weight, loaded_shard_id
                    ),
                    "omni_param_role": tensor_name,
                    "ignore_warning": True,
                },
            )
            layer.register_parameter(tensor_name, param)

        layer.omni_layer_kind = self.layer_kind
        layer.omni_weight_bits = self.quant_config.weight_bits
        layer.omni_activation_bits = self.quant_config.activation_bits
        layer.omni_group_size = group_size
        layer.omni_padded_in_features = padded_in_features
        layer.omni_activation_symmetric = self.quant_config.activation_symmetric
        layer.omni_activation_disable_zero_point = self.quant_config.activation_disable_zero_point
        layer.omni_activation_metric = self.quant_config.activation_metric
        layer.omni_runtime_act_quant_calls = 0

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer.omni_runtime_act_quant_calls += 1
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        qx, x_scale, x_zero_point = _quantize_activation_runtime(
            x_float,
            n_bits=layer.omni_activation_bits,
            symmetric=layer.omni_activation_symmetric,
            disable_zero_point=layer.omni_activation_disable_zero_point,
            metric=layer.omni_activation_metric,
        )
        if x_zero_point is None:
            centered_x = qx.to(torch.float32)
        else:
            centered_x = qx.to(torch.float32) - x_zero_point.to(torch.float32)

        if layer.omni_padded_in_features > centered_x.shape[-1]:
            deficiency = layer.omni_padded_in_features - centered_x.shape[-1]
            centered_x = torch.cat(
                [
                    centered_x,
                    torch.zeros(*centered_x.shape[:-1], deficiency, dtype=centered_x.dtype, device=centered_x.device),
                ],
                dim=-1,
            )

        centered_x = centered_x.view(*centered_x.shape[:-1], -1, layer.omni_group_size)
        centered_w = layer.qweight.to(torch.float32) - layer.w_zero_points.to(torch.float32).unsqueeze(-1)
        acc = torch.einsum("...gc,ogc->...og", centered_x, centered_w)
        out = (acc * layer.w_scales.view(*([1] * (acc.dim() - 2)), acc.shape[-2], -1)).sum(dim=-1)
        out = out * x_scale.to(out.dtype)
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.to(input_dtype)


@register_quantization_config("omni_activation_real")
class OmniActivationRealConfig(QuantizationConfig):
    def __init__(self, config_dict: dict[str, Any]):
        super().__init__()
        self.config_dict = config_dict
        self.weight_bits = int(config_dict.get("bits", config_dict.get("weight_bits", 4)))
        self.activation_bits = int(config_dict.get("activation_bits", 16))
        self.group_size = int(config_dict.get("group_size", -1))
        self.runtime_format = config_dict.get("format", "omni_activation_real_v1")
        self.activation_symmetric = bool(config_dict.get("activation_sym", False))
        self.activation_disable_zero_point = bool(config_dict.get("activation_disable_zero_point", False))
        self.activation_metric = config_dict.get("activation_metric", "minmax")
        self.runtime_config: dict[str, Any] | None = None
        self.module_runtime_metadata: dict[str, Any] = {}
        self.model_dir: str | None = None

    def get_name(self) -> str:
        return "omni_activation_real"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json", RUNTIME_CONFIG_NAME]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OmniActivationRealConfig":
        return cls(config)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str,
        config_dict: dict[str, Any] | None = None,
    ) -> "OmniActivationRealConfig":
        if config_dict is None:
            config = json.loads((Path(model_dir) / "config.json").read_text())
            config_dict = config["quantization_config"]
        instance = cls(config_dict)
        instance.load_model_dir(model_dir)
        return instance

    def load_model_dir(self, model_dir: str) -> "OmniActivationRealConfig":
        self.model_dir = str(model_dir)
        self.runtime_config = load_omni_activation_real_runtime_config(model_dir)
        self.runtime_format = self.runtime_config.get("format", self.runtime_format)
        self.module_runtime_metadata = self.runtime_config.get("modules", {})
        return self

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: Any | None = None,
        revision: str | None = None,
    ):
        model_path = Path(model_name)
        if model_path.exists():
            runtime_path = model_path / RUNTIME_CONFIG_NAME
            if runtime_path.exists():
                self.load_model_dir(model_name)

    def get_quant_method(self, layer: nn.Module, prefix: str):
        if isinstance(layer, QKVParallelLinear):
            return OmniActivationRealLinearMethod(self, prefix=prefix, layer_kind="qkv_parallel")
        if isinstance(layer, MergedColumnParallelLinear):
            return OmniActivationRealLinearMethod(self, prefix=prefix, layer_kind="merged_column")
        if isinstance(layer, RowParallelLinear):
            return OmniActivationRealLinearMethod(self, prefix=prefix, layer_kind="row_parallel")
        return None


def load_omni_activation_real_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    return load_file(str(Path(model_dir) / "model.safetensors"))
