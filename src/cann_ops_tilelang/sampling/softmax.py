"""Softmax operators for Ascend."""

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .softmax_kernel import _softmax_tl_ascend, get_block_size

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_input(x: torch.Tensor) -> None:
    require_ascend()
    if not isinstance(x, torch.Tensor):
        raise TypeError('x must be a torch.Tensor.')
    if x.device.type != 'npu':
        raise ValueError(f'x must be on an NPU, got {x.device}.')
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f'x must have one of {_SUPPORTED_DTYPES}, got {x.dtype}.')
    if x.ndim != 2 or x.stride(1) != 1:
        raise ValueError('x must be a two-dimensional tensor contiguous in its last dimension.')
    if x.shape[1] <= 0:
        raise ValueError('x must have a non-empty vocabulary dimension.')


def _run_softmax(
    x: torch.Tensor,
    *,
    do_softmax: bool,
    do_logsoftmax: bool,
    inplace_softmax: bool = False,
    inplace_logsoftmax: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    _validate_input(x)
    softmax_out = (
        (x if inplace_softmax else torch.empty_like(x, memory_format=torch.contiguous_format)) if do_softmax else None
    )
    logsoftmax_out = (
        (x if inplace_logsoftmax else torch.empty_like(x, memory_format=torch.contiguous_format))
        if do_logsoftmax
        else None
    )
    if x.shape[0] == 0:
        return softmax_out, logsoftmax_out

    kernel = _softmax_tl_ascend(
        vocab_size=x.shape[1],
        vocab_block_size=get_block_size(x.shape[1]),
        dtype=x.dtype,
        do_softmax=do_softmax,
        do_logsoftmax=do_logsoftmax,
        inplace_softmax=inplace_softmax,
        inplace_logsoftmax=inplace_logsoftmax,
    )
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(x, softmax_out, logsoftmax_out)
    return softmax_out, logsoftmax_out


def softmax(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Compute softmax along the last dimension."""
    output, _ = _run_softmax(x, do_softmax=True, do_logsoftmax=False, inplace_softmax=inplace)
    assert output is not None
    return output


def log_softmax(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Compute log-softmax along the last dimension."""
    _, output = _run_softmax(x, do_softmax=False, do_logsoftmax=True, inplace_logsoftmax=inplace)
    assert output is not None
    return output


def softmax_and_logsoftmax(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute softmax and log-softmax with one kernel launch."""
    softmax_out, logsoftmax_out = _run_softmax(x, do_softmax=True, do_logsoftmax=True)
    assert softmax_out is not None and logsoftmax_out is not None
    return softmax_out, logsoftmax_out
