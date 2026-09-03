# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    ],
    key=["N", "K"],
)
@triton.jit
def _omni_w6a6_linear_kernel(
    A,
    B,
    C,
    As,
    Bs,
    Ad,
    Corr,
    M,
    N,
    K,
    NUM_GROUPS: tl.constexpr,
    GROUP_K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_Bs_g,
    stride_Bs_n,
    stride_Ad_m,
    stride_Corr_n,
    APPLY_ACTIVATION_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_s = tl.load(As + offs_m * stride_As_m, mask=m_mask, other=0.0)

    for g in range(NUM_GROUPS):
        offs_k = tl.arange(0, GROUP_K)
        a_ptrs = A + offs_m[:, None] * stride_am + (g * GROUP_K + offs_k)[None, :] * stride_ak
        b_ptrs = B + g * stride_bg + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0).to(tl.int8)
        b = tl.load(b_ptrs, mask=n_mask[None, :], other=0).to(tl.int8)

        b_s = tl.load(
            Bs + g * stride_Bs_g + offs_n * stride_Bs_n,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(a, b).to(tl.float32) * a_s[:, None] * b_s[None, :]

    if APPLY_ACTIVATION_OFFSET:
        a_d = tl.load(Ad + offs_m * stride_Ad_m, mask=m_mask, other=0.0).to(tl.float32)
        corr = tl.load(Corr + offs_n * stride_Corr_n, mask=n_mask, other=0.0).to(
            tl.float32
        )
        acc += a_s[:, None] * a_d[:, None] * corr[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 64}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _omni_w6a6_linear_small_m_kernel(
    A,
    B,
    C,
    As,
    Bs,
    Ad,
    Corr,
    M,
    N,
    K,
    NUM_GROUPS: tl.constexpr,
    GROUP_K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_Bs_g,
    stride_Bs_n,
    stride_Ad_m,
    stride_Corr_n,
    APPLY_ACTIVATION_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_s = tl.load(As + offs_m * stride_As_m, mask=m_mask, other=0.0)

    for g in range(NUM_GROUPS):
        offs_k = tl.arange(0, GROUP_K)
        a_ptrs = A + offs_m[:, None] * stride_am + (g * GROUP_K + offs_k)[None, :] * stride_ak
        b_ptrs = B + g * stride_bg + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0).to(tl.int8)
        b = tl.load(b_ptrs, mask=n_mask[None, :], other=0).to(tl.int8)
        b_s = tl.load(
            Bs + g * stride_Bs_g + offs_n * stride_Bs_n,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(a, b).to(tl.float32) * a_s[:, None] * b_s[None, :]

    if APPLY_ACTIVATION_OFFSET:
        a_d = tl.load(Ad + offs_m * stride_Ad_m, mask=m_mask, other=0.0).to(tl.float32)
        corr = tl.load(Corr + offs_n * stride_Corr_n, mask=n_mask, other=0.0).to(
            tl.float32
        )
        acc += a_s[:, None] * a_d[:, None] * corr[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _omni_w6a6_linear_fused_small_m_kernel(
    A,
    B,
    C,
    Bs,
    Corr,
    M,
    N,
    K,
    qmin,
    qmax,
    qmid,
    NUM_GROUPS: tl.constexpr,
    GROUP_K: tl.constexpr,
    symmetric: tl.constexpr,
    disable_zero_point: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_Bs_g,
    stride_Bs_n,
    stride_Corr_n,
    APPLY_ACTIVATION_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    row_min = tl.full((BLOCK_M,), float("inf"), dtype=tl.float32)
    row_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    for g in range(NUM_GROUPS):
        offs_k = tl.arange(0, GROUP_K)
        a_ptrs = A + offs_m[:, None] * stride_am + (g * GROUP_K + offs_k)[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(a, axis=1))
        row_max = tl.maximum(row_max, tl.max(a, axis=1))

    if symmetric:
        abs_max = tl.maximum(tl.abs(row_min), tl.abs(row_max))
        scale = tl.maximum(abs_max / qmax, 1e-5)
        zero_point = tl.zeros((BLOCK_M,), dtype=tl.int32)
    else:
        scale = tl.maximum((row_max - row_min) / (qmax - qmin), 1e-5)
        if disable_zero_point:
            zero_point = tl.zeros((BLOCK_M,), dtype=tl.int32)
        else:
            zero_point = _round_to_int32(-(row_min) / scale)
            zero_point = tl.maximum(tl.minimum(zero_point, qmax), qmin)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    row_offset = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for g in range(NUM_GROUPS):
        offs_k = tl.arange(0, GROUP_K)
        a_ptrs = A + offs_m[:, None] * stride_am + (g * GROUP_K + offs_k)[None, :] * stride_ak
        b_ptrs = B + g * stride_bg + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=n_mask[None, :], other=0).to(tl.int8)
        b_s = tl.load(
            Bs + g * stride_Bs_g + offs_n * stride_Bs_n,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)

        centered = _round_to_int32(a / scale[:, None])
        if symmetric:
            centered = tl.maximum(tl.minimum(centered, qmax), qmin)
            if not disable_zero_point:
                centered = (
                    tl.maximum(tl.minimum(centered + qmax, qmax), qmin) - qmax
                )
        else:
            if disable_zero_point:
                centered = tl.maximum(tl.minimum(centered, qmax), qmin)
            else:
                quantized = tl.maximum(
                    tl.minimum(centered + zero_point[:, None], qmax),
                    qmin,
                )
                if APPLY_ACTIVATION_OFFSET:
                    centered = quantized - qmid
                    row_offset = (qmid - zero_point).to(tl.float32)
                else:
                    centered = quantized - zero_point[:, None]

        acc += tl.dot(centered.to(tl.int8), b).to(tl.float32) * scale[:, None] * b_s[None, :]

    if APPLY_ACTIVATION_OFFSET:
        corr = tl.load(Corr + offs_n * stride_Corr_n, mask=n_mask, other=0.0).to(
            tl.float32
        )
        acc += scale[:, None] * row_offset[:, None] * corr[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


if current_platform.is_rocm():

    @triton.jit
    def _round_to_int32(x):
        return tl.extra.hip.libdevice.round(x).to(tl.int32)

else:

    @triton.jit
    def _round_to_int32(x):
        return tl.extra.cuda.libdevice.rint(x).to(tl.int32)


@triton.jit
def _per_token_omni_quant_int8_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    offset_ptr,
    stride_x,
    stride_xq,
    stride_offset,
    N,
    qmin,
    qmax,
    qmid,
    symmetric: tl.constexpr,
    disable_zero_point: tl.constexpr,
    midpoint_centering: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    x_min = tl.min(tl.where(mask, x, float("inf")), axis=0)
    x_max = tl.max(tl.where(mask, x, float("-inf")), axis=0)

    if symmetric:
        abs_max = tl.maximum(tl.abs(x_min), tl.abs(x_max))
        scale = tl.maximum(abs_max / qmax, 1e-5)
        centered = _round_to_int32(x / scale)
        centered = tl.maximum(tl.minimum(centered, qmax), qmin)
        if not disable_zero_point:
            centered = tl.maximum(tl.minimum(centered + qmax, qmax), qmin) - qmax
    else:
        scale = tl.maximum((x_max - x_min) / (qmax - qmin), 1e-5)
        centered = _round_to_int32(x / scale)
        if disable_zero_point:
            centered = tl.maximum(tl.minimum(centered, qmax), qmin)
        else:
            zero_point = _round_to_int32(-(x_min) / scale)
            zero_point = tl.maximum(tl.minimum(zero_point, qmax), qmin)
            quantized = tl.maximum(tl.minimum(centered + zero_point, qmax), qmin)
            if midpoint_centering:
                centered = quantized - qmid
                tl.store(
                    offset_ptr + row_id * stride_offset,
                    (qmid - zero_point).to(tl.float32),
                )
            else:
                centered = quantized - zero_point

    tl.store(
        xq_ptr + row_id * stride_xq + cols,
        centered.to(tl.int8),
        mask=mask,
    )
    tl.store(scale_ptr + row_id, scale)


def _quant_bounds(
    n_bits: int,
    symmetric: bool,
    disable_zero_point: bool,
) -> tuple[int, int]:
    if disable_zero_point or symmetric:
        return -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
    return 0, 2**n_bits - 1


def quantize_omni_activation_per_token_int8(
    x: torch.Tensor,
    *,
    n_bits: int,
    symmetric: bool,
    disable_zero_point: bool,
    group_size: int,
    expand_group_scales: bool = True,
    midpoint_centering: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert x.is_cuda
    assert x.ndim == 2
    assert x.is_contiguous()
    assert 1 <= n_bits <= 8
    assert group_size > 0 and x.shape[-1] % group_size == 0

    qmin, qmax = _quant_bounds(n_bits, symmetric, disable_zero_point)
    m, k = x.shape
    num_groups = k // group_size

    x_q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    row_scales = torch.empty((m,), device=x.device, dtype=torch.float32)
    row_offsets = (
        torch.empty((m,), device=x.device, dtype=torch.float32)
        if midpoint_centering
        else None
    )
    block = triton.next_power_of_2(k)
    num_warps = min(max(block // 256, 1), 8)
    qmid = (qmin + qmax + 1) // 2

    _per_token_omni_quant_int8_kernel[(m,)](
        x,
        x_q,
        row_scales,
        row_offsets if row_offsets is not None else row_scales,
        stride_x=x.stride(0),
        stride_xq=x_q.stride(0),
        stride_offset=(row_offsets.stride(0) if row_offsets is not None else 0),
        N=k,
        qmin=qmin,
        qmax=qmax,
        qmid=qmid,
        symmetric=symmetric,
        disable_zero_point=disable_zero_point,
        midpoint_centering=midpoint_centering,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )

    if expand_group_scales:
        return (
            x_q,
            row_scales.view(m, 1).expand(-1, num_groups).contiguous(),
            row_offsets,
        )
    return x_q, row_scales, row_offsets


def omni_w6a6_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation_bits: int,
    activation_symmetric: bool,
    activation_disable_zero_point: bool,
    activation_weight_correction: torch.Tensor | None = None,
) -> torch.Tensor:
    assert current_platform.is_cuda(), "omni_w6a6_linear requires CUDA"
    assert x.is_cuda and weight.is_cuda and weight_scale.is_cuda
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert weight_scale.is_contiguous()
    assert weight.dtype == torch.int8
    assert weight_scale.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert weight.ndim == 3
    assert weight_scale.ndim == 2
    if activation_weight_correction is not None:
        assert activation_weight_correction.is_cuda
        assert activation_weight_correction.is_contiguous()
        assert activation_weight_correction.shape == (weight.shape[-1],)

    num_groups, group_k, n = weight.shape
    assert group_k == 128
    k = num_groups * 128
    assert x.shape[-1] == k
    assert weight_scale.shape == (num_groups, n)

    x_2d = x.view(-1, x.shape[-1]).contiguous()
    m = x_2d.shape[0]
    apply_activation_offset = activation_weight_correction is not None
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))

    if n > 4096:
        fused_m_limit = 0
    elif n > 2048:
        fused_m_limit = 128
    else:
        fused_m_limit = 256

    if m <= fused_m_limit:
        qmin, qmax = _quant_bounds(
            activation_bits,
            activation_symmetric,
            activation_disable_zero_point,
        )
        _omni_w6a6_linear_fused_small_m_kernel[grid](
            x_2d,
            weight,
            out,
            weight_scale,
            activation_weight_correction
            if activation_weight_correction is not None
            else weight_scale[0],
            m,
            n,
            k,
            qmin,
            qmax,
            (qmin + qmax + 1) // 2,
            NUM_GROUPS=num_groups,
            GROUP_K=128,
            symmetric=activation_symmetric,
            disable_zero_point=activation_disable_zero_point,
            stride_am=x_2d.stride(0),
            stride_ak=x_2d.stride(1),
            stride_bg=weight.stride(0),
            stride_bk=weight.stride(1),
            stride_bn=weight.stride(2),
            stride_cm=out.stride(0),
            stride_cn=out.stride(1),
            stride_Bs_g=weight_scale.stride(0),
            stride_Bs_n=weight_scale.stride(1),
            stride_Corr_n=(
                activation_weight_correction.stride(0)
                if activation_weight_correction is not None
                else 0
            ),
            APPLY_ACTIVATION_OFFSET=apply_activation_offset,
        )
    else:
        x_q, x_s, x_offsets = quantize_omni_activation_per_token_int8(
            x_2d,
            n_bits=activation_bits,
            symmetric=activation_symmetric,
            disable_zero_point=activation_disable_zero_point,
            group_size=128,
            expand_group_scales=False,
            midpoint_centering=apply_activation_offset,
        )

        kernel = _omni_w6a6_linear_small_m_kernel if m <= 8 else _omni_w6a6_linear_kernel
        kernel[grid](
            x_q,
            weight,
            out,
            x_s,
            weight_scale,
            x_offsets if x_offsets is not None else x_s,
            activation_weight_correction
            if activation_weight_correction is not None
            else weight_scale[0],
            m,
            n,
            k,
            NUM_GROUPS=num_groups,
            GROUP_K=128,
            stride_am=x_q.stride(0),
            stride_ak=x_q.stride(1),
            stride_bg=weight.stride(0),
            stride_bk=weight.stride(1),
            stride_bn=weight.stride(2),
            stride_cm=out.stride(0),
            stride_cn=out.stride(1),
            stride_As_m=x_s.stride(0),
            stride_Bs_g=weight_scale.stride(0),
            stride_Bs_n=weight_scale.stride(1),
            stride_Ad_m=x_offsets.stride(0) if x_offsets is not None else 0,
            stride_Corr_n=(
                activation_weight_correction.stride(0)
                if activation_weight_correction is not None
                else 0
            ),
            APPLY_ACTIVATION_OFFSET=apply_activation_offset,
        )

    if bias is not None:
        out.add_(bias)
    return out.view(*x.shape[:-1], n)
