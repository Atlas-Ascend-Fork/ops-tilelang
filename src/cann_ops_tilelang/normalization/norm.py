"""Public Ascend interfaces for fused normalization operators."""

from __future__ import annotations

from typing import Any

import torch
from tilelang.ascend import language as T

from cann_ops_tilelang.config import require_ascend

from .norm_kernel import _norm_fwd_tl_ascend


def norm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_scale: float = 1.0,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_out: torch.Tensor | None = None,
    residual_dtype: torch.dtype | None = None,
    is_rms_norm: bool = True,
    down_cast_before_norm: bool = False,
    # optional fused input update; only one of x2 and x_scale may be provided
    x2: torch.Tensor | None = None,
    x_scale: float | None = None,
    # optional FP8 cast
    do_fp8_cast: bool = False,
    num_per_channels: int = 0,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    return_origin_out: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    require_ascend()
    if x.device.type != 'npu':
        raise ValueError(f'x must be an NPU tensor, got {x.device}.')
    for name, tensor in (
        ('weight', weight),
        ('residual', residual),
        ('bias', bias),
        ('out', out),
        ('residual_out', residual_out),
        ('x2', x2),
    ):
        if tensor is not None and tensor.device != x.device:
            raise ValueError(f'{name} must be on {x.device}, got {tensor.device}.')
    assert not use_packed_ue8m0 or use_tma_aligned_col_major_sf
    if residual is not None:
        residual_dtype = residual.dtype
    if residual_dtype is not None:
        assert x.dtype == residual_dtype
    batch, seq_len, hidden_dim = x.shape
    if hidden_dim <= 0 or hidden_dim % 64 != 0:
        raise ValueError(f'the hidden dimension must be a positive multiple of 64, got {hidden_dim}.')
    if do_fp8_cast:
        if num_per_channels not in (32, 128):
            raise ValueError(f'num_per_channels must be 32 or 128 for FP8 output, got {num_per_channels}.')
        if hidden_dim % num_per_channels != 0:
            raise ValueError(
                f'the hidden dimension ({hidden_dim}) must be divisible by num_per_channels ({num_per_channels}).'
            )
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (batch, seq_len, hidden_dim)
    assert weight.shape == (hidden_dim,), f'{weight.shape=} {hidden_dim=}'
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.shape == (hidden_dim,)
        assert bias.stride(-1) == 1
        assert bias.dtype == weight.dtype  # bwd will return same type grads
    # allocate output
    if out is None:
        out = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    else:
        assert out.shape == x.shape
    assert out.stride(-1) == 1
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        if residual_out is None:
            residual_out = torch.empty(
                batch,
                seq_len,
                hidden_dim,
                device=x.device,
                dtype=residual_dtype if residual_dtype is not None else x.dtype,
            )
        else:
            assert residual_out.shape == x.shape
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    out_origin = torch.empty_like(x, dtype=x.dtype) if return_origin_out else None
    if do_fp8_cast:
        if use_tma_aligned_col_major_sf:
            if use_packed_ue8m0:
                num_groups = hidden_dim // num_per_channels
                num_group_pairs = (num_groups + 1) // 2
                sf_raw = torch.empty(batch, num_group_pairs, seq_len, dtype=torch.int16, device=out.device)
                sf_uint8_ptr = sf_raw.view(torch.uint8).as_strided(
                    (batch * num_group_pairs, 2 * seq_len), (2 * seq_len, 1)
                )
                sf = sf_raw
            else:
                col_range = hidden_dim // num_per_channels
                if use_packed_ue8m0:
                    col_range = (col_range + 3) // 4
                col_stride = (seq_len + 3) // 4 * 4
                batch_stride = col_range * col_stride
                if use_packed_ue8m0:
                    batch_stride *= 4
                sf = torch.empty(
                    batch,
                    col_stride,
                    col_range,
                    dtype=(torch.int32 if use_packed_ue8m0 else torch.float32),
                    device=out.device,
                )
                if use_packed_ue8m0:
                    sf_uint8_ptr = sf.view(torch.uint8).as_strided(
                        (batch, seq_len, col_range, 4), (batch_stride, 4, (seq_len + 3) // 4 * 16, 1)
                    )
                sf = torch.as_strided(sf, size=sf.shape, stride=(batch_stride, 1, col_stride))
                sf = sf[:, :seq_len, :]
        else:
            sf = torch.empty(batch, seq_len, hidden_dim // num_per_channels, dtype=torch.float32, device=out.device)
    else:
        sf = None

    dtype_str = ''
    if x.dtype == torch.float:
        dtype_str = 'float'
    elif x.dtype == torch.bfloat16:
        dtype_str = 'bfloat16'
    else:
        raise AssertionError('Unsupported dtype')
    if out_dtype is None:
        out_dtype_str = dtype_str
    elif out_dtype == torch.float:
        out_dtype_str = 'float'
    elif out_dtype == torch.bfloat16:
        out_dtype_str = 'bfloat16'
    elif out_dtype == torch.float8_e4m3fn:
        assert do_fp8_cast
        out_dtype_str = T.float8_e4m3fn
    else:
        raise AssertionError('Unsupported dtype')

    kernel_fwd = _norm_fwd_tl_ascend(
        hidden_dim,
        dtype_str,
        T.float8_e4m3fn if do_fp8_cast else out_dtype_str,
        residual is not None,
        residual_out is not None,
        bias is not None,
        is_rms_norm,
        eps,
        out_scale,
        down_cast_before_norm,
        return_origin_out,
        x2 is not None,
        1.0 if x_scale is None else x_scale,
        x_stride_s=x.stride(1),
        y_stride_s=out.stride(1),
        residual_stride_s=residual.stride(1) if residual is not None else x.stride(1),
        residual_out_stride_s=residual_out.stride(1) if residual_out is not None else x.stride(1),
        x2_stride_s=x2.stride(1) if x2 is not None else x.stride(1),
        y_origin_stride_s=out_origin.stride(1) if out_origin is not None else x.stride(1),
        do_fp8_cast=do_fp8_cast,
        num_per_channels=num_per_channels if do_fp8_cast else 1,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    kernel_fwd(
        x,
        out,
        weight,
        bias if bias is not None else weight,
        residual if residual is not None else x,
        residual_out if residual_out is not None else x,
        x2 if x2 is not None else x,
        (sf_uint8_ptr if use_packed_ue8m0 else sf) if do_fp8_cast else x,
        out_origin if return_origin_out else x,
    )
    if do_fp8_cast and use_packed_ue8m0:
        sf = sf_raw.transpose(1, 2)
    return (
        out,
        sf,
        out_origin,
    )


def layer_norm_fn(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, float],
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
    is_rms_norm: bool = False,
    out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    # fp8 cast params
    fmt: str = '',
    num_per_channels: int = 0,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    # fp8 cast params end
    return_bf16: bool = False,
    out_scale: float = 1.0,
    down_cast_before_norm: bool = False,
) -> Any:
    require_ascend()
    if down_cast_before_norm:
        assert residual is not None
    # Notes: fmt == "" and num_per_channels == 0 is equivalent, content of fmt is not used.
    if fmt == '':
        assert num_per_channels == 0
        do_fp8_cast = False
    else:
        assert num_per_channels != 0
        do_fp8_cast = True
    # Notes: `out_scale` is only for no bias version
    assert out_scale == 1.0 or bias is None
    # x_type represents the actual structure of x: 0 for a tensor,
    # 1 for (tensor, tensor), and 2 for (tensor, float).
    x_type = 0
    if isinstance(x, tuple):
        assert len(x) == 2
        if isinstance(x[1], torch.Tensor):
            x_type = 1
            x, x2 = x
            x_scale = None
            assert x.dtype == x2.dtype, f'{x.dtype=} {x2.dtype=}'
            assert x.shape == x2.shape and x.stride() == x2.stride(), f'{x.shape=} {x2.shape=}'
        else:
            assert isinstance(x[1], float)
            x_type = 2
            x, x_scale = x
            x2 = None
    else:
        x2 = None
        x_scale = None

    x_shape_og = x.shape
    assert x.stride(-1) == 1
    assert x.ndim == 2 or x.ndim == 3
    if x.ndim == 2:
        x = x.view(1, x.shape[0], x.shape[1])
    if x_type == 1:
        assert x2.ndim == 2 or x2.ndim == 3
        if x2.ndim == 2:
            x2 = x2.view(1, x2.shape[0], x2.shape[1])
    assert weight.shape[-1] == x.shape[-1]
    assert weight.is_contiguous()
    assert not use_packed_ue8m0 or use_tma_aligned_col_major_sf
    if bias is not None:
        assert bias.shape[-1] == x.shape[-1]
        assert bias.is_contiguous()
    if residual is not None:
        assert residual.shape == x_shape_og
        assert residual.stride(-1) == 1
        if residual.ndim == 2:
            residual = residual.view(1, residual.shape[0], residual.shape[1])
    if out is not None:
        assert out.shape == x_shape_og
        assert out.stride(-1) == 1
        if out.ndim == 2:
            out = out.view(1, out.shape[0], out.shape[1])
        assert not do_fp8_cast
    elif not do_fp8_cast:
        out = torch.empty_like(x)
    if residual_out is not None:
        assert residual is not None
        assert residual_out.shape == x_shape_og
        assert residual_out.stride(-1) == 1
        if residual_out.ndim == 2:
            residual_out = residual_out.view(1, residual_out.shape[0], residual_out.shape[1])
    elif residual is not None:
        residual_out = torch.empty_like(residual)
    if do_fp8_cast:
        out_norm_dtype = torch.float8_e4m3fn
        out = torch.empty_like(x, dtype=out_norm_dtype)
    else:
        out_norm_dtype = out.dtype

    out, sf, out_norm = norm_fn(
        x,
        weight=weight,
        residual=residual,
        bias=None if bias is None else bias,
        eps=eps,
        out_scale=out_scale,
        out=out,
        out_dtype=out_norm_dtype,
        residual_out=None if residual_out is None else residual_out,
        residual_dtype=None if residual_out is None else residual_out.dtype,
        is_rms_norm=is_rms_norm,
        down_cast_before_norm=down_cast_before_norm,
        x2=x2,
        x_scale=x_scale,
        # fp8 cast params
        do_fp8_cast=do_fp8_cast,
        num_per_channels=num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        return_origin_out=return_bf16,
    )

    # return:
    #     out
    #     sf(when do_fp8_cast)
    #     out_bf16 (when return_bf16)
    #     residual (optional)
    out = out.view(*x_shape_og)
    if do_fp8_cast:
        if out.ndim == 2:
            assert sf.shape[0] == 1
            sf = sf[0]
        out = (out, sf)
    if return_bf16:
        out_bf16 = out_norm.to(torch.bfloat16) if do_fp8_cast else out.to(torch.bfloat16)
        out_bf16 = out_bf16.view(*x_shape_og)
        out = (out, out_bf16)
    if residual is not None:
        out = (out, residual_out.view(*x_shape_og))
    return out
