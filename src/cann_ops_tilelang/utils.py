"""Small helpers shared by operator wrappers."""

import torch


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f'divisor must be positive, got {divisor}.')
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


_STR_TO_DTYPE = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
    'e4m3': torch.float8_e4m3fn,
    'e2m1': torch.int8,
}


def str_to_dtype(fmt: str) -> torch.dtype:
    """Return the torch storage dtype used by a quantization format."""

    try:
        return _STR_TO_DTYPE[fmt]
    except KeyError as error:
        raise ValueError(f'Unsupported format: {fmt}. Supported: {list(_STR_TO_DTYPE)}') from error
