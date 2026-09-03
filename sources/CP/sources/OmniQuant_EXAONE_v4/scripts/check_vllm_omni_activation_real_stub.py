import json

import torch
from vllm.model_executor.layers.linear import RowParallelLinear

from quantize.vllm_omni_activation_real import OmniActivationRealConfig
from vllm.model_executor.layers.quantization import get_quantization_config


def main():
    config_cls = get_quantization_config("omni_activation_real")
    config = config_cls.from_config(
        {
            "quant_method": "omni_activation_real",
            "bits": 4,
            "activation_bits": 8,
            "group_size": 128,
        }
    )
    layer = RowParallelLinear(
        input_size=128,
        output_size=64,
        bias=False,
        quant_config=config,
        prefix="layer0.o_proj",
        disable_tp=True,
    )
    method = layer.quant_method
    try:
        x = torch.randn(1, 2, 128, dtype=torch.float16)
        layer(x)
    except Exception as exc:  # pragma: no cover - diagnostic path
        apply_error = f"{type(exc).__name__}: {exc}"
    else:
        apply_error = None

    summary = {
        "registered_config_class": config_cls.__name__,
        "instance_type": type(config).__name__,
        "method_type": type(method).__name__,
        "qweight_shape": list(layer.qweight.shape),
        "apply_error": apply_error,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
