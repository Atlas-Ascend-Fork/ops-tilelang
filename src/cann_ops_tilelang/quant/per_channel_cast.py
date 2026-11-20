import torch
from tilelang.ascend import language as T

from cann_ops_tilelang.config import (
    get_amax_clamp_for_quant,
    get_num_vector_cores,
    require_ascend,
    should_print_kernel_source,
)
from cann_ops_tilelang.quant.common import (
    alloc_scaling_factors,
    cast_epilogue,
    get_cast_input_and_config,
    get_cast_output_config_impl,
    get_sf_shape,
)

from .per_channel_cast_kernel import get_per_channel_cast_kernel_asc


@torch.library.custom_op('cann_ops_tilelang::per_channel_cast', mutates_args=())
def per_channel_cast(
    data: torch.Tensor,
    scale: torch.Tensor | None,
    fmt: str,
    num_per_tokens: int,
    round_sf: bool = False,
    num_per_channels: int | None = None,
    use_packed_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel cast to FP8 (row-major SF).

    Args:
        data: Input 2D tensor of shape ``(num_tokens, hidden)``.
        scale: Input scaling factors for pre-quantized FP8 data, or ``None``
            for unquantized input.
        fmt: Target FP8 format (must be ``'e4m3'``).
        num_per_tokens: Number of tokens sharing each scaling factor block (``32`` or ``128``).
        round_sf: Whether to round scaling factors to powers of two.
        num_per_channels: Number of channels in each input scaling block. Must be
            ``32`` or ``128`` when ``scale`` is provided, ``None`` otherwise.
        use_packed_ue8m0: Whether to store sf factors as packed UE8M0 bytes.

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and row-major sf-factor tensor.

    Fixed-shape npugraph_ex inference is supported. Configuration updates through
    the config API invalidate graphs; no autograd formula is registered.
    """
    return _per_channel_cast_impl(data, scale, fmt, num_per_tokens, round_sf, num_per_channels, use_packed_ue8m0)


def _validate_per_channel_cast(x, fmt, num_per_tokens, round_sf, num_per_channels, use_packed_ue8m0, clamp_min_value):
    x, x_sf_invs, in_config = get_cast_input_and_config(x, None if num_per_channels is None else (1, num_per_channels))

    assert fmt == 'e4m3'
    if in_config.with_sf:
        input_num_per_channels = in_config.sf_block[1]
        assert input_num_per_channels in (32, 128), 'Ascend rescale supports num_per_channels in (32, 128) only'
    else:
        assert in_config.dtype == T.bfloat16, 'per_channel_cast supports bf16 or e4m3 (rescale) input only'
    assert x.device.type == 'npu'
    assert x.dim() == 2 and x.is_contiguous()
    assert num_per_tokens in (32, 128)
    num_tokens, hidden = x.shape
    assert hidden % 128 == 0, f'Per channel cast requires hidden % 128 == 0, got {hidden}'
    if x_sf_invs is not None:
        assert num_per_channels in (32, 128)
        assert x.dtype == torch.float8_e4m3fn
        assert x_sf_invs.dim() == 2 and x_sf_invs.is_contiguous()
        assert tuple(x_sf_invs.shape) == get_sf_shape((num_tokens, hidden), in_config)
        if in_config.use_tma_aligned_col_major_sf and in_config.use_packed_ue8m0:
            x_sf_invs = x_sf_invs.view(torch.int16)

    out_config = get_cast_output_config_impl(
        fmt, (num_per_tokens, 1), False, round_sf, use_packed_ue8m0, clamp_min_value
    )
    quant_max = 448.0
    assert out_config.clamp_min_value >= quant_max * (2 ** (-126)), (
        'Ascend scale exponent fast path requires clamp to normal number'
    )
    assert num_tokens % 128 == 0
    return x, x_sf_invs, in_config, out_config


def _per_channel_cast_impl(data, scale, fmt, num_per_tokens, round_sf, num_per_channels, use_packed_ue8m0):
    x = data if scale is None else (data, scale)
    clamp_min_value = get_amax_clamp_for_quant(fmt)
    require_ascend()
    x, x_sf_invs, in_config, out_config = _validate_per_channel_cast(
        x,
        fmt,
        num_per_tokens,
        round_sf,
        num_per_channels,
        use_packed_ue8m0,
        clamp_min_value,
    )
    num_tokens, hidden = x.shape
    out = torch.empty((num_tokens, hidden), dtype=out_config.torch_dtype, device=x.device)
    out_sf = alloc_scaling_factors((num_tokens, hidden), out_config, device=x.device)
    kernel = get_per_channel_cast_kernel_asc(
        hidden, in_config=in_config, out_config=out_config, num_cores=get_num_vector_cores()
    )

    if should_print_kernel_source():
        print(kernel.get_kernel_source())

    if num_tokens > 0:
        kernel(x, out, out_sf, x_sf_invs)
    out_sf = cast_epilogue(out_sf, out_config)
    return out, out_sf


@per_channel_cast.register_fake
def _per_channel_cast_fake(
    data, scale, fmt, num_per_tokens, round_sf=False, num_per_channels=None, use_packed_ue8m0=False
):
    x = data if scale is None else (data, scale)
    clamp_min_value = get_amax_clamp_for_quant(fmt)
    data, _, _, config = _validate_per_channel_cast(
        x,
        fmt,
        num_per_tokens,
        round_sf,
        num_per_channels,
        use_packed_ue8m0,
        clamp_min_value,
    )
    output = torch.empty(data.shape, dtype=config.torch_dtype, device=data.device)
    scales = alloc_scaling_factors(tuple(data.shape), config, data.device)
    return output, cast_epilogue(scales, config)
