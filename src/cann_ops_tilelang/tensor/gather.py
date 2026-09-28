"""Public interface for generic row gathering on Ascend."""

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .gather_kernel import get_gather_kernel


def gather(x: torch.Tensor, index: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Gather rows from ``x``, filling invalid indices with zero."""
    if not isinstance(x, torch.Tensor):
        raise TypeError('x must be a torch.Tensor.')
    if not isinstance(index, torch.Tensor):
        raise TypeError('index must be a torch.Tensor.')
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError('out must be a torch.Tensor or None.')

    require_ascend()
    if x.device.type != 'npu' or index.device.type != 'npu':
        raise ValueError('x and index must be NPU tensors.')
    if index.device != x.device:
        raise ValueError('x and index must be on the same device.')
    if x.ndim == 0:
        raise ValueError('x must have at least one dimension.')
    if index.ndim != 1:
        raise ValueError('index must be a 1D tensor.')
    if index.dtype not in (torch.int32, torch.int64):
        raise TypeError(f'index must use torch.int32 or torch.int64, got {index.dtype}.')
    if not x.is_contiguous() or not index.is_contiguous():
        raise ValueError('x and index must be contiguous.')

    output_shape = (index.shape[0], *x.shape[1:])
    if out is not None:
        if out.shape != output_shape:
            raise ValueError(f'out must have shape {output_shape}, got {tuple(out.shape)}.')
        if out.dtype != x.dtype:
            raise TypeError(f'out must use {x.dtype}, got {out.dtype}.')
        if out.device != x.device:
            raise ValueError('out must be on the same device as x.')
        if not out.is_contiguous():
            raise ValueError('out must be contiguous.')

    x_shape = x.shape
    x_dtype = x.dtype
    x_bytes = x.view(x_shape[0], -1).view(torch.int8)
    if out is None:
        output_bytes = torch.empty((index.shape[0], x_bytes.shape[1]), dtype=torch.int8, device=x.device)
    else:
        output_bytes = out.view(out.shape[0], -1).view(torch.int8)

    kernel = get_gather_kernel(x_bytes.shape[1], x_bytes.dtype, index.dtype)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(x_bytes, index, output_bytes)
    return output_bytes.view(x_dtype).view(output_shape)
