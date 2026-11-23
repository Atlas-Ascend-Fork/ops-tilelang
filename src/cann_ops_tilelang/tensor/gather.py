# MIT License
#
# Copyright (c) 2026 ops-tilelang contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Public interface for generic row gathering on Ascend."""

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .gather_kernel import get_gather_kernel


def _validate_inputs(x: torch.Tensor, index: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError('x must be a torch.Tensor.')
    if not isinstance(index, torch.Tensor):
        raise TypeError('index must be a torch.Tensor.')
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


def _gather_impl(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    _validate_inputs(x, index)
    require_ascend()
    output_shape = (index.shape[0], *x.shape[1:])
    if x.shape[0] == 0:
        return torch.zeros(output_shape, dtype=x.dtype, device=x.device)
    if index.numel() == 0 or x.numel() == 0:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    x_shape = x.shape
    x_dtype = x.dtype
    x_bytes = x.view(x_shape[0], -1).view(torch.int8)
    row_bytes = x_bytes.shape[1]
    if row_bytes % 4:
        # The byte-copy kernel requires aligned row starts. Pad only rows that
        # would otherwise issue misaligned NPU accesses, then strip the padding.
        x_bytes = torch.nn.functional.pad(x_bytes, (0, (-row_bytes) % 32))
    output_bytes = torch.empty((index.shape[0], x_bytes.shape[1]), dtype=torch.int8, device=x.device)
    kernel = get_gather_kernel(x_bytes.shape[1], x_bytes.dtype, index.dtype)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(x_bytes, index, output_bytes)
    if row_bytes != output_bytes.shape[1]:
        output_bytes = output_bytes[:, :row_bytes].contiguous()
    return output_bytes.view(x_dtype).view(output_shape)


@torch.library.custom_op('cann_ops_tilelang::gather', mutates_args=())
def gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather rows into a new tensor, filling out-of-range indices with zero.

    Inputs must be contiguous tensors on the same NPU, with a 1D int32/int64
    index. The result has shape ``(index.numel(), *x.shape[1:])`` and ``x.dtype``.
    No input is modified and no optional ``out`` buffer is accepted.

    Supports fixed-shape inference with ``torch.compile(...,
    backend='npugraph_ex', fullgraph=True, dynamic=False)``. The fake kernel
    only validates metadata and constructs output metadata. TileLang JIT and
    launch remain in the real implementation; no autograd formula is registered.
    """
    return _gather_impl(x, index)


@gather.register_fake
def _gather_fake(x, index):
    _validate_inputs(x, index)
    return torch.empty((index.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device)
