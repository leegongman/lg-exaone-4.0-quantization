# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, get_args

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)

QuantizationMethods = Literal[
    "awq",
    "fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mixed",
    "gguf",
    "gptq_marlin",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "petit_nvfp4",
    "cpu_awq",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

DEPRECATED_QUANTIZATION_METHODS = [
    "tpu_int8",
    "fbgemm_fp8",
    "fp_quant",
    "experts_int8",
    "petit_nvfp4",
]

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.warning(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
        return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]

    # Import only the requested backend so unrelated optional dependencies
    # (for example compressed_tensors or extra Triton kernels) do not break
    # custom quantization methods that do not rely on them.
    if quantization == "awq":
        from .awq import AWQConfig

        return AWQConfig
    if quantization == "fp8":
        from .fp8 import Fp8Config

        return Fp8Config
    if quantization == "fbgemm_fp8":
        from .fbgemm_fp8 import FBGEMMFp8Config

        return FBGEMMFp8Config
    if quantization == "fp_quant":
        from .fp_quant import FPQuantConfig

        return FPQuantConfig
    if quantization == "modelopt":
        from .modelopt import ModelOptFp8Config

        return ModelOptFp8Config
    if quantization == "modelopt_fp4":
        from .modelopt import ModelOptNvFp4Config

        return ModelOptNvFp4Config
    if quantization == "modelopt_mixed":
        from .modelopt import ModelOptMixedPrecisionConfig

        return ModelOptMixedPrecisionConfig
    if quantization == "gguf":
        from .gguf import GGUFConfig

        return GGUFConfig
    if quantization == "gptq_marlin":
        from .gptq_marlin import GPTQMarlinConfig

        return GPTQMarlinConfig
    if quantization == "awq_marlin":
        from .awq_marlin import AWQMarlinConfig

        return AWQMarlinConfig
    if quantization == "gptq":
        from .gptq import GPTQConfig

        return GPTQConfig
    if quantization == "compressed-tensors":
        from .compressed_tensors.compressed_tensors import CompressedTensorsConfig

        return CompressedTensorsConfig
    if quantization == "bitsandbytes":
        from .bitsandbytes import BitsAndBytesConfig

        return BitsAndBytesConfig
    if quantization == "experts_int8":
        from .experts_int8 import ExpertsInt8Config

        return ExpertsInt8Config
    if quantization == "quark":
        from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

        return QuarkConfig
    if quantization == "moe_wna16":
        from .moe_wna16 import MoeWNA16Config

        return MoeWNA16Config
    if quantization == "torchao":
        from .torchao import TorchAOConfig

        return TorchAOConfig
    if quantization in {"auto-round", "inc"}:
        from .inc import INCConfig

        return INCConfig
    if quantization == "mxfp4":
        from .mxfp4 import Mxfp4Config

        return Mxfp4Config
    if quantization == "petit_nvfp4":
        from .petit import PetitNvFp4Config

        return PetitNvFp4Config
    if quantization == "cpu_awq":
        from .cpu_wna16 import CPUAWQConfig

        return CPUAWQConfig

    raise ValueError(f"Unsupported quantization method: {quantization}")


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]

# Import custom in-tree quant methods so they self-register on package import.
from . import omni_activation_real  # noqa: E402,F401
