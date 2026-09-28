from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.normalization.norm import layer_norm_fn
from cann_ops_tilelang.quant.per_token_cast import per_token_cast

pytestmark = pytest.mark.npu


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str = '',
) -> None:
    """Standalone tensor comparison helper."""
    try:
        if actual.dtype == torch.bfloat16:
            torch.testing.assert_close(actual, expected, atol=3e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(actual, expected)
    except AssertionError as error:
        if name:
            raise AssertionError(f'{name}: {error}') from error
        raise


def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm reference: x -> w * x / sqrt(E[x^2] + eps)."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight
    if residual is None:
        return x
    return x, residual


def layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm reference: x -> w * (x - mean) / sqrt(var + eps) + b."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)
    x = x - x.mean(dim=-1, keepdim=True)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight + bias
    if residual is None:
        return x
    return x, residual


_EPS = 1e-6


def _norm_test_params() -> list[tuple]:
    """Full-range correctness: all hidden_sizes × all branches."""
    params: list[tuple] = []
    for hidden_size in [512, 1024, 1536, 2048, 4096, 5120, 7168]:
        for x_type in [0, 1, 2]:
            for is_rms_norm in [True, False]:
                for has_residual in [False, True]:
                    x_out_scale = 1.5 if is_rms_norm else 1.0
                    params.append(
                        (
                            x_type,
                            16384,
                            hidden_size,
                            is_rms_norm,
                            has_residual,
                            torch.bfloat16,
                            x_out_scale,
                        )
                    )
    return params


def _norm_fp8_test_params() -> list[tuple]:
    """Full-range correctness: all hidden_sizes × all fp8 branches × all x_types."""
    params: list[tuple] = []
    for hidden_size in [512, 1024, 1536, 2048, 4096, 5120, 7168]:
        for num_per_channels in [32, 128]:
            for use_tma_aligned_col_major_sf in [False, True]:
                for round_sf in [False, True]:
                    for use_packed_ue8m0 in [False, True]:
                        if use_packed_ue8m0 and (not use_tma_aligned_col_major_sf or not round_sf):
                            continue
                        if num_per_channels == 32 and use_tma_aligned_col_major_sf:
                            continue
                        for x_type in [0, 1, 2]:
                            for has_residual in [False, True]:
                                params.append(
                                    (
                                        16384,
                                        hidden_size,
                                        has_residual,
                                        x_type,
                                        num_per_channels,
                                        use_tma_aligned_col_major_sf,
                                        round_sf,
                                        use_packed_ue8m0,
                                        1.5,
                                    )
                                )
    return params


def _make_inputs(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    x_type: int,
    has_residual: bool,
):
    """Build x (possibly tuple), x_real, residual tensors.

    x_type:
        0 — plain tensor
        1 — (x, x2) tuple  (x_real = x + x2)
        2 — (x, scale) tuple  (x_real = x * scale)
    """
    # Non-contiguous x (matches the original test)
    x = torch.randn(num_tokens, hidden_size + 64, dtype=dtype, device=get_device())
    x = x.split([hidden_size, 64], dim=-1)[0]
    assert not x.is_contiguous() and x.stride(1) == 1

    residual = torch.randn_like(x) if has_residual else None

    x_real = x
    if x_type == 1:
        x2 = torch.randn(num_tokens, hidden_size + 64, dtype=dtype, device=get_device())
        x2 = x2.split([hidden_size, 64], dim=-1)[0]
        x_real = x + x2
        x_input = (x, x2)
    elif x_type == 2:
        x_scale = 2.3333
        x_real = x * x_scale
        x_input = (x, x_scale)
    else:
        x_input = x

    return x_input, x_real, residual


@pytest.mark.parametrize(
    'x_type,num_tokens,hidden_size,is_rms_norm,has_residual,dtype,x_out_scale',
    _norm_test_params(),
)
def test_norm(
    x_type: int,
    num_tokens: int,
    hidden_size: int,
    is_rms_norm: bool,
    has_residual: bool,
    dtype: torch.dtype,
    x_out_scale: float,
) -> None:
    weight = torch.empty((hidden_size,), dtype=dtype, device=get_device())
    weight.normal_(mean=1.0, std=0.1)
    bias = None
    if not is_rms_norm:
        bias = torch.empty((hidden_size,), dtype=dtype, device=get_device())
        bias.normal_(mean=1.0, std=0.1)

    x_input, x_real, residual = _make_inputs(num_tokens, hidden_size, dtype, x_type, has_residual)

    # Reference (pure PyTorch)
    ref_out = (
        rms_norm_ref(x_real, weight, _EPS, residual)
        if is_rms_norm
        else layer_norm_ref(x_real, weight, bias, _EPS, residual)
    )

    # Kernel under test
    out = layer_norm_fn(
        x_input,
        weight,
        bias=bias,
        residual=residual,
        eps=_EPS,
        is_rms_norm=is_rms_norm,
        out_scale=x_out_scale,
    )

    if has_residual:
        assert_close(out[0], ref_out[0] * x_out_scale, name='out')
        assert_close(out[1], ref_out[1], name='residual')
    else:
        assert_close(out, ref_out * x_out_scale, name='out')


@pytest.mark.parametrize(
    'num_tokens,hidden_size,has_residual,x_type,fp8_npc,use_col_major,round_sf,use_packed_ue8m0,x_out_scale',
    _norm_fp8_test_params(),
)
def test_norm_fp8_cast(
    num_tokens: int,
    hidden_size: int,
    has_residual: bool,
    x_type: int,
    fp8_npc: int,
    use_col_major: bool,
    round_sf: bool,
    use_packed_ue8m0: bool,
    x_out_scale: float,
) -> None:
    dtype = torch.bfloat16

    weight = torch.empty((hidden_size,), dtype=dtype, device=get_device())
    weight.normal_(mean=1.0, std=0.1)

    x_input, x_real, residual = _make_inputs(num_tokens, hidden_size, dtype, x_type, has_residual)

    # Step 1: Get bf16 norm output via the non-fp8 path
    out_bf16_result = layer_norm_fn(
        x_input,
        weight,
        bias=None,
        residual=residual,
        eps=_EPS,
        is_rms_norm=True,
        out_scale=x_out_scale,
    )
    out_bf16 = out_bf16_result[0] if has_residual else out_bf16_result

    # Step 2: Cast bf16 output to fp8 independently
    ref_fp8, ref_scale = per_token_cast(
        out_bf16,
        'e4m3',
        num_per_channels=fp8_npc,
        round_sf=round_sf,
        use_tma_aligned_col_major_sf=use_col_major,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    # Step 3: Get fused norm+fp8 output
    fused_out = layer_norm_fn(
        x_input,
        weight,
        bias=None,
        residual=residual,
        eps=_EPS,
        is_rms_norm=True,
        out=None,
        fmt='e4m3',
        num_per_channels=fp8_npc,
        use_tma_aligned_col_major_sf=use_col_major,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        return_bf16=False,
        out_scale=x_out_scale,
        down_cast_before_norm=False,
    )
    if has_residual:
        fused_out = fused_out[0]

    # fused_out is ((fp8_tensor, sf),) or (fp8_tensor, sf)
    fused_fp8, fused_scale = fused_out

    # Binary-exact comparison
    fp8_max_diff = (fused_fp8.view(torch.uint8).int() - ref_fp8.view(torch.uint8).int()).abs().max().item()
    assert torch.equal(fused_fp8.view(torch.uint8), ref_fp8.view(torch.uint8)), (
        f'fp8 values mismatch: max diff = {fp8_max_diff}'
    )
    assert torch.equal(fused_scale, ref_scale), (
        f'sf mismatch: fused shape={fused_scale.shape} dtype={fused_scale.dtype}, '
        f'ref shape={ref_scale.shape} dtype={ref_scale.dtype}'
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'num_tokens,hidden_size,has_residual',
    [
        (16384, 4096, True),
        (16384, 7168, True),
        (16384, 7168, False),
    ],
)
def test_norm_benchmark(
    num_tokens: int,
    hidden_size: int,
    has_residual: bool,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    dtype = torch.bfloat16
    weight = torch.empty((hidden_size,), dtype=dtype, device=get_device())
    weight.normal_(mean=1.0, std=0.1)

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=get_device())
    residual = torch.randn_like(x) if has_residual else None

    def fn():
        return layer_norm_fn(
            x,
            weight,
            bias=None,
            residual=residual,
            eps=_EPS,
            is_rms_norm=True,
            out_scale=1.5,
        )

    t_us = benchmark_timer(fn)

    # I/O bytes: read (x + residual + weight) + write (out + residual_out)
    io_bytes = (x.numel() + weight.numel()) * dtype.itemsize
    if residual is not None:
        io_bytes += residual.numel() * dtype.itemsize
    out = fn()
    if isinstance(out, tuple):
        for o in out:
            if isinstance(o, torch.Tensor):
                io_bytes += o.numel() * o.element_size()
    else:
        io_bytes += out.numel() * out.element_size()

    benchmark_record(
        kernel='norm',
        operation='rms_norm',
        params={
            'has_residual': int(has_residual),
            'hidden_size': hidden_size,
            'num_tokens': num_tokens,
        },
        time_us=t_us,
        extras={'bandwidth_gbs': io_bytes / 1e9 / (t_us / 1e6)},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'num_tokens,hidden_size,fp8_npc',
    [
        (16384, 4096, 128),
        (16384, 7168, 128),
        (16384, 7168, 32),
    ],
)
def test_norm_fp8_cast_benchmark(
    num_tokens: int,
    hidden_size: int,
    fp8_npc: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    dtype = torch.bfloat16
    weight = torch.empty((hidden_size,), dtype=dtype, device=get_device())
    weight.normal_(mean=1.0, std=0.1)

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=get_device())
    residual = torch.randn_like(x)

    def fn():
        return layer_norm_fn(
            x,
            weight,
            bias=None,
            residual=residual,
            eps=_EPS,
            is_rms_norm=True,
            fmt='e4m3',
            num_per_channels=fp8_npc,
            use_tma_aligned_col_major_sf=False,
            round_sf=False,
            use_packed_ue8m0=False,
            return_bf16=False,
            out_scale=1.5,
        )

    t_us = benchmark_timer(fn)

    # I/O: read (x + residual + weight) + write (fp8_out + sf + residual_out)
    io_bytes = (x.numel() + residual.numel() + weight.numel()) * dtype.itemsize
    io_bytes += num_tokens * hidden_size  # fp8 output (1 byte each)
    io_bytes += num_tokens * (hidden_size // fp8_npc) * 4  # sf (float32)
    io_bytes += num_tokens * hidden_size * dtype.itemsize  # residual_out

    benchmark_record(
        kernel='norm',
        operation='rms_norm_fp8_cast',
        params={
            'fp8_npc': fp8_npc,
            'hidden_size': hidden_size,
            'num_tokens': num_tokens,
        },
        time_us=t_us,
        extras={'bandwidth_gbs': io_bytes / 1e9 / (t_us / 1e6)},
    )
