import torch
from tilelang.ascend import language as T

from cann_ops_tilelang.config import (
    get_amax_clamp_for_quant,
    get_num_vector_cores,
    require_ascend,
    should_print_kernel_source,
)

from .common import get_cast_output_config_impl
from .per_token_cast_and_cast_back_kernel import get_per_token_cast_and_cast_back_kernel_asc


@torch.library.custom_op('cann_ops_tilelang::per_token_cast_and_cast_back', mutates_args=('x',))
def per_token_cast_and_cast_back(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    round_sf: bool = False,
) -> None:
    """Apply an in-place lossy per-token cast-and-cast-back to ``x``.

    Args:
        x: Input 2D tensor of shape (num_tokens, hidden). The innermost stride must
            be 1; ``x.stride(0)`` may exceed ``hidden`` (non-contiguous rows allowed).
        fmt: Quantized format used for cast-and-cast-back (``'e4m3'`` or ``'e2m1'``).
        num_per_channels: Number of channels in each scaling block (32, 64 or 128).
        round_sf: Whether to round scaling factors to powers of two.

    Returns None and mutates only x. Fixed-shape npugraph_ex inference is
    supported; the registered operator declares the same mutation contract.
    """
    _per_token_cast_and_cast_back_impl(x, fmt, num_per_channels, round_sf)
    return None


def _validate_per_token_cast_and_cast_back(x, fmt, num_per_channels, round_sf, clamp_min_value):
    if x.device.type != 'npu':
        raise ValueError(f'x must be an NPU tensor, got {x.device}.')
    assert x.dim() == 2
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        'Ascend per-token cast and cast-back only supports bf16, fp16 or fp32 input'
    )
    assert x.stride(1) == 1
    assert fmt in ('e4m3', 'e2m1')
    assert num_per_channels in (32, 64, 128), (
        'Ascend per-token cast and cast-back only supports num_per_channels=32/64/128'
    )

    _, hidden = x.shape
    assert hidden % 64 == 0, 'Ascend per-token cast and cast-back hidden must be a multiple of 64'
    assert hidden % num_per_channels == 0

    out_config = get_cast_output_config_impl(fmt, (1, num_per_channels), False, round_sf, False, clamp_min_value)
    assert out_config.sf_block[0] == 1

    return out_config


def _per_token_cast_and_cast_back_impl(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    round_sf: bool,
) -> None:
    require_ascend()
    clamp_min_value = get_amax_clamp_for_quant(fmt)
    out_config = _validate_per_token_cast_and_cast_back(x, fmt, num_per_channels, round_sf, clamp_min_value)
    num_tokens, hidden = x.shape
    kernel = get_per_token_cast_and_cast_back_kernel_asc(
        hidden=hidden,
        token_stride=x.stride(0),
        dtype=T.dtype(x.dtype),
        out_config=out_config,
        n_cores=get_num_vector_cores(),
    )

    if should_print_kernel_source():
        print(kernel.get_kernel_source())

    if num_tokens > 0:
        kernel(x)


@per_token_cast_and_cast_back.register_fake
def _per_token_cast_and_cast_back_fake(x, fmt, num_per_channels, round_sf=False):
    clamp_min_value = get_amax_clamp_for_quant(fmt)
    _validate_per_token_cast_and_cast_back(x, fmt, num_per_channels, round_sf, clamp_min_value)
