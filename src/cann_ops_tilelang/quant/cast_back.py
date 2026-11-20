import torch
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend, should_print_kernel_source

from .cast_back_kernel import get_cast_back_kernel_ascend
from .common import get_cast_input_and_config, get_logical_hidden
from .types import QuantTensor


@torch.library.custom_op(
    'cann_ops_tilelang::cast_back',
    mutates_args=(),
    schema='(Tensor[] x, str fmt, SymInt[] x_block_size, str? x_special_fmt=None) -> Tensor',
)
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

    Fixed-shape npugraph_ex inference is supported. No autograd formula is registered.
    """
    return _cast_back_impl(x, fmt, x_block_size, x_special_fmt)


def _validate_cast_back(x, x_sf, fmt, x_block_size, x_special_fmt):
    assert fmt in ('bf16', 'fp32')
    out_dtype = torch.bfloat16 if fmt == 'bf16' else torch.float32

    if x.device.type != 'npu' or x_sf.device.type != 'npu':
        raise ValueError('x and its scaling factors must be NPU tensors.')

    assert x_special_fmt is None, f'Unsupported special format: {x_special_fmt}'

    _, hidden = x.shape
    assert x_sf.dim() == 2
    assert x_sf.dtype in (torch.int16, torch.int32, torch.float32)

    hidden = get_logical_hidden(hidden, x.dtype)
    assert hidden % 64 == 0

    _, x_sf, in_config = get_cast_input_and_config((x, x_sf), x_block_size)
    # Mirror the kernel's static block/layout restrictions in fake execution.
    assert in_config.sf_block in ((1, 32), (1, 128), (32, 1), (32, 32), (128, 1), (128, 128))
    assert not (in_config.sf_block == (32, 1) and in_config.use_tma_aligned_col_major_sf)
    return x_sf, in_config, hidden, out_dtype


def _cast_back_impl(x, fmt, x_block_size, x_special_fmt):
    x_block_size = tuple(x_block_size)
    require_ascend()
    data, scale = x
    x_sf, in_config, hidden, out_dtype = _validate_cast_back(data, scale, fmt, x_block_size, x_special_fmt)
    x = data
    num_tokens = x.shape[0]
    out = torch.empty((num_tokens, hidden), dtype=out_dtype, device=x.device)
    if num_tokens == 0:
        return out

    kernel = get_cast_back_kernel_ascend(
        hidden=hidden, in_config=in_config, out_dtype=T.dtype(out_dtype), n_cores=get_num_vector_cores()
    )

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

    Fixed-shape npugraph_ex inference is supported. No autograd formula is registered.
    """
    return cast_back(x, fmt, (1, num_per_channels), x_special_fmt=x_special_fmt)


@cast_back.register_fake
def _cast_back_fake(x, fmt, x_block_size, x_special_fmt=None):
    data, scale = x
    _, _, hidden, out_dtype = _validate_cast_back(data, scale, fmt, tuple(x_block_size), x_special_fmt)
    return torch.empty((data.shape[0], hidden), dtype=out_dtype, device=data.device)
