import torch

from cann_ops_tilelang import config
from cann_ops_tilelang.quant.common import (
    QuantTensor,
    alloc_scaling_factors,
    cast_epilogue,
    get_cast_input_and_config,
    get_cast_output_config_impl,
    get_logical_hidden,
)

from .per_block_cast_lossless_kernel import get_per_block_cast_lossless_kernel_asc


@torch.library.custom_op(
    'cann_ops_tilelang::per_block_cast_lossless',
    mutates_args=(),
    schema='(Tensor[] x, str fmt, SymInt[] x_block_size, SymInt[] out_block_size, '
    'bool use_tma_aligned_col_major_sf=False, bool round_sf=False, bool use_packed_ue8m0=False) -> (Tensor, Tensor)',
)
def per_block_cast_lossless(
    x: QuantTensor,
    fmt: str,
    x_block_size: tuple[int, int],
    out_block_size: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> QuantTensor:
    """Losslessly re-quantize an FP4 (E2M1) tensor to FP8 (E4M3) with new block sizes.

    Args:
        x: Quantized tensor pair ``(data, sf_factors)`` in E2M1 format.
        fmt: Target format (must be ``'e4m3'``).
        x_block_size: Input scaling block size as (num_per_tokens, num_per_channels).
        out_block_size: Output scaling block size as (num_per_tokens, num_per_channels).
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and sf-factor tensor.
    """
    options = [use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0]
    return _per_block_cast_lossless_impl(x, fmt, x_block_size, out_block_size, options)


def _validate_per_block_cast_lossless(x, fmt, x_block_size, out_block_size, options, clamp_min_value):
    x, x_sf, in_config = get_cast_input_and_config(x, x_block_size)

    assert fmt == 'e4m3' and x.dtype == torch.int8, 'lossless cast only supports e2m1 -> e4m3 input'
    assert x.device.type == 'npu'
    assert x.is_contiguous() and x.dim() == 2
    num_tokens, hidden = x.shape
    hidden = get_logical_hidden(hidden, x.dtype)
    assert num_tokens % 128 == 0, f'lossless cast requires num_tokens % 128 == 0, got {num_tokens}'
    assert hidden % 32 == 0

    out_config = get_cast_output_config_impl(fmt, out_block_size, *options, clamp_min_value)

    assert x_block_size == (1, 32), 'ascend lossless only supports in (1, 32)'
    assert out_block_size in (
        (32, 32),
        (1, 128),
        (128, 128),
    ), 'ascend lossless only supports out (32, 32) / (1, 128) / (128, 128)'
    return x, x_sf, in_config, out_config


def _per_block_cast_lossless_impl(
    x: list[torch.Tensor],
    fmt: str,
    x_block_size: list[int],
    out_block_size: list[int],
    options: list[bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    config.require_ascend()
    clamp_min_value = config.get_amax_clamp_for_quant(fmt)
    x, x_sf, in_config, out_config = _validate_per_block_cast_lossless(
        tuple(x), fmt, tuple(x_block_size), tuple(out_block_size), options, clamp_min_value
    )
    num_tokens, physical_hidden = x.shape
    hidden = get_logical_hidden(physical_hidden, x.dtype)
    # FP4 is stored as packed int8; reinterpret as the 2x-packed fp4 dtype the kernel expects.
    x = x.view(torch.float4_e2m1fn_x2)
    kernel = get_per_block_cast_lossless_kernel_asc(
        hidden=hidden,
        in_config=in_config,
        out_config=out_config,
        num_cores=config.get_num_vector_cores(),
    )

    if config.should_print_kernel_source():
        print(kernel.get_kernel_source())

    out = torch.empty((num_tokens, hidden), dtype=out_config.torch_dtype, device=x.device)
    out_sf = alloc_scaling_factors(
        (num_tokens, hidden),
        out_config=out_config,
        device=x.device,
    )

    if num_tokens > 0:
        kernel(x, x_sf, out, out_sf)

    out_sf = cast_epilogue(out_sf, out_config)

    return out, out_sf


@per_block_cast_lossless.register_fake
def _per_block_cast_lossless_fake(
    x, fmt, x_block_size, out_block_size, use_tma_aligned_col_major_sf=False, round_sf=False, use_packed_ue8m0=False
):
    options = (use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)
    x, _, _, out_config = _validate_per_block_cast_lossless(
        tuple(x), fmt, tuple(x_block_size), tuple(out_block_size), options, None
    )
    shape = (x.shape[0], get_logical_hidden(x.shape[1], x.dtype))
    out = torch.empty(shape, dtype=out_config.torch_dtype, device=x.device)
    scale = cast_epilogue(alloc_scaling_factors(shape, out_config, device=x.device), out_config)
    return out, scale
