import torch
from tilelang.ascend import language as T

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .common import get_cast_output_config
from .per_token_cast_and_cast_back_kernel import get_per_token_cast_and_cast_back_kernel_asc


def per_token_cast_and_cast_back(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    round_sf: bool = False,
):
    """Apply an in-place lossy per-token cast-and-cast-back to ``x``.

    Args:
        x: Input 2D tensor of shape (num_tokens, hidden). The innermost stride must
            be 1; ``x.stride(0)`` may exceed ``hidden`` (non-contiguous rows allowed).
        fmt: Quantized format used for cast-and-cast-back (``'e4m3'`` or ``'e2m1'``).
        num_per_channels: Number of channels in each scaling block (32, 64 or 128).
        round_sf: Whether to round scaling factors to powers of two.
    """
    require_ascend()
    if x.device.type != 'npu':
        raise ValueError(f'x must be an NPU tensor, got {x.device}.')
    assert x.dim() == 2
    assert x.stride(1) == 1
    assert fmt in ('e4m3', 'e2m1')
    assert num_per_channels in (32, 64, 128)

    num_tokens, hidden = x.shape
    assert hidden % num_per_channels == 0

    out_config = get_cast_output_config(fmt, (1, num_per_channels), round_sf=round_sf)
    assert out_config.sf_block[0] == 1

    kernel = get_per_token_cast_and_cast_back_kernel_asc(
        hidden=hidden,
        token_stride=x.stride(0),
        dtype=T.dtype(x.dtype),
        out_config=out_config,
    )

    if should_print_kernel_source():
        print(kernel.get_kernel_source())

    if num_tokens > 0:
        kernel(x)
