import torch
from tilelang.ascend import language as T

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .cast_back_kernel import get_cast_back_kernel_ascend
from .common import get_cast_input_and_config, get_logical_hidden
from .types import QuantTensor


def cast_back(
    x: QuantTensor,
    fmt: str,
    x_block_size: tuple[int, int],
    x_special_fmt: str | None = None,
) -> torch.Tensor:
    """Dequantize an FP8/FP4 tensor back to BF16 or FP32.

    Args:
        x: Quantized tensor pair ``(data, sf_factors)``.
        fmt: Target output format, either ``'bf16'`` or ``'fp32'``.
        x_block_size: Scaling block size as ``(num_per_tokens, num_per_channels)``.
        x_special_fmt: Reserved, must be ``None``.

    Returns:
        Dequantized tensor in the requested format.
    """
    require_ascend()
    assert fmt in ('bf16', 'fp32')
    out_dtype = torch.bfloat16 if fmt == 'bf16' else torch.float32

    x, x_sf = x
    if x.device.type != 'npu' or x_sf.device.type != 'npu':
        raise ValueError('x and its scaling factors must be NPU tensors.')

    assert x_special_fmt is None, f'Unsupported special format: {x_special_fmt}'

    num_tokens, hidden = x.shape
    assert x_sf.dim() == 2
    assert x_sf.dtype in (torch.int16, torch.int32, torch.float32)

    hidden = get_logical_hidden(hidden, x.dtype)
    assert hidden % 64 == 0

    x, x_sf, in_config = get_cast_input_and_config((x, x_sf), x_block_size)
    out = torch.empty((num_tokens, hidden), dtype=out_dtype, device=x.device)
    if num_tokens == 0:
        return out

    kernel = get_cast_back_kernel_ascend(hidden=hidden, in_config=in_config, out_dtype=T.dtype(out_dtype))

    if should_print_kernel_source():
        print(kernel.get_kernel_source())

    kernel(x, x_sf, out)

    return out


def per_token_cast_back(
    x: QuantTensor,
    fmt: str,
    num_per_channels: int,
    x_special_fmt: str | None = None,
) -> torch.Tensor:
    """Dequantize an FP8/FP4 tensor back to BF16 or FP32 with per-token scaling.

    Args:
        x: Quantized tensor pair ``(data, sf_factors)``.
        fmt: Target output format, either ``'bf16'`` or ``'fp32'``.
        num_per_channels: Number of channels per scaling block.
        x_special_fmt: Reserved, must be ``None``.

    Returns:
        Dequantized tensor in the requested format.
    """
    return cast_back(x, fmt, (1, num_per_channels), x_special_fmt=x_special_fmt)
