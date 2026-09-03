# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

from .omni_triton_utils import quantize_omni_activation_per_token_int8


def omni_w6a6_cutlass_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation_bits: int,
    activation_symmetric: bool,
    activation_disable_zero_point: bool,
) -> torch.Tensor:
    assert current_platform.is_cuda(), "omni_w6a6_cutlass_linear requires CUDA"
    assert x.is_cuda and weight.is_cuda and weight_scale.is_cuda
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert weight.dtype == torch.int8
    assert weight_scale.dtype == torch.float32
    assert weight.ndim == 2
    assert weight_scale.ndim == 2
    assert weight.shape[0] % 128 == 0
    assert weight.stride(0) == 1
    assert weight_scale.shape[0] * 128 == weight.shape[0]
    assert weight_scale.shape[1] == weight.shape[1]

    x_2d = x.view(-1, x.shape[-1])
    q_input, x_scale = quantize_omni_activation_per_token_int8(
        x_2d.contiguous(),
        n_bits=activation_bits,
        symmetric=activation_symmetric,
        disable_zero_point=activation_disable_zero_point,
        group_size=128,
    )
    output = ops.cutlass_scaled_mm(
        q_input,
        weight,
        scale_a=x_scale,
        scale_b=weight_scale,
        out_dtype=x.dtype,
        bias=bias,
    )
    return output.view(*x.shape[:-1], weight.shape[1])
