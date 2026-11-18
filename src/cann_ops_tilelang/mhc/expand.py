# MIT License
#
# Copyright (c) 2026 DeepSeek
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

"""Public interfaces for expanding activations across MHC streams."""

from __future__ import annotations

import torch

from cann_ops_tilelang.common.compile_utils import resolve_vector_cores
from cann_ops_tilelang.config import require_ascend

from .expand_kernel import expand_to_mhc_backward_kernel, expand_to_mhc_forward_kernel

_FORWARD_HIDDEN_ALIGNMENT = 128
_BACKWARD_HIDDEN_ALIGNMENT = 64
_FORWARD_NUM_STAGES = 2
_BACKWARD_NUM_STAGES = 3
_BACKWARD_HIDDEN_BLOCK = 4096


def _check_npu_bfloat16_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...] | None = None,
    device: torch.device | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor.')
    if tensor.device.type != 'npu':
        raise ValueError(f'{name} must be an NPU tensor, got {tensor.device}.')
    if device is not None and tensor.device != device:
        raise ValueError(f'{name} must be on {device}, got {tensor.device}.')
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f'{name} must use torch.bfloat16, got {tensor.dtype}.')
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f'{name} must have shape {shape}, got {tuple(tensor.shape)}.')
    if not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous.')


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.numel() > 0
        and right.numel() > 0
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _forward_hidden_block(num_tokens: int, hidden_size: int) -> int:
    is_short = num_tokens <= 512
    if is_short and hidden_size >= 7168:
        return 1024
    if is_short or hidden_size <= 4096:
        return 512
    return 1024


def _validate_expand_to_mhc_forward_metadata(x: torch.Tensor, output: torch.Tensor) -> tuple:
    if not isinstance(x, torch.Tensor):
        raise TypeError('x must be a torch.Tensor.')
    if x.ndim != 2:
        raise ValueError(f'x must have shape (num_tokens, hidden_size), got {tuple(x.shape)}.')
    num_tokens, hidden_size = x.shape
    if hidden_size <= 0 or hidden_size % _FORWARD_HIDDEN_ALIGNMENT:
        raise ValueError(f'hidden_size must be a positive multiple of {_FORWARD_HIDDEN_ALIGNMENT}, got {hidden_size}.')
    if not isinstance(output, torch.Tensor):
        raise TypeError('output must be a torch.Tensor.')
    if output.ndim != 3:
        raise ValueError('output must have shape (num_tokens, mhc_mult, hidden_size).')
    mhc_mult = output.shape[1]
    if mhc_mult <= 0:
        raise ValueError(f'mhc_mult must be positive, got {mhc_mult}.')
    _check_npu_bfloat16_tensor('x', x)
    _check_npu_bfloat16_tensor('output', output, shape=(num_tokens, mhc_mult, hidden_size), device=x.device)
    return (hidden_size, num_tokens)


def _expand_to_mhc_forward_impl(x: torch.Tensor, output: torch.Tensor, *, vector_cores: int) -> None:
    """Copy every BF16 activation row into each MHC stream on Ascend."""
    hidden_size, num_tokens = _validate_expand_to_mhc_forward_metadata(x, output)
    require_ascend()
    if _shares_storage(output, x):
        raise ValueError('output and x must not share storage.')
    if num_tokens == 0:
        return
    expand_to_mhc_forward_kernel(
        x,
        output,
        hidden_block=_forward_hidden_block(num_tokens, hidden_size),
        num_stages=_FORWARD_NUM_STAGES,
        num_vec_cores=vector_cores,
    )


@torch.library.custom_op(
    'cann_ops_tilelang::expand_to_mhc_forward', mutates_args=('output',), schema='(Tensor x, Tensor(a!) output) -> ()'
)
def expand_to_mhc_forward(x: torch.Tensor, output: torch.Tensor) -> None:
    """Use runtime core configuration; configuration APIs invalidate compiled graphs."""
    _validate_expand_to_mhc_forward_metadata(x, output)
    cores = 1 if x.numel() == 0 else resolve_vector_cores(x)
    _expand_to_mhc_forward_impl(x, output, vector_cores=cores)


@expand_to_mhc_forward.register_fake
def _expand_to_mhc_forward_fake(x: torch.Tensor, output: torch.Tensor) -> None:
    _validate_expand_to_mhc_forward_metadata(x, output)


def _validate_expand_to_mhc_backward_metadata(output_grad: torch.Tensor, x_grad: torch.Tensor):
    if not isinstance(output_grad, torch.Tensor):
        raise TypeError('output_grad must be a torch.Tensor.')
    if output_grad.ndim != 3:
        raise ValueError('output_grad must have shape (num_tokens, mhc_mult, hidden_size).')
    num_tokens, mhc_mult, hidden_size = output_grad.shape
    if mhc_mult <= 0:
        raise ValueError(f'mhc_mult must be positive, got {mhc_mult}.')
    if hidden_size <= 0 or hidden_size % _BACKWARD_HIDDEN_ALIGNMENT:
        raise ValueError(f'hidden_size must be a positive multiple of {_BACKWARD_HIDDEN_ALIGNMENT}, got {hidden_size}.')
    _check_npu_bfloat16_tensor('output_grad', output_grad)
    _check_npu_bfloat16_tensor('x_grad', x_grad, shape=(num_tokens, hidden_size), device=output_grad.device)
    return num_tokens


def _expand_to_mhc_backward_impl(output_grad: torch.Tensor, x_grad: torch.Tensor, *, vector_cores: int) -> None:
    """Sum BF16 MHC stream gradients into a caller-owned activation gradient."""
    num_tokens = _validate_expand_to_mhc_backward_metadata(output_grad, x_grad)
    require_ascend()
    if _shares_storage(x_grad, output_grad):
        raise ValueError('x_grad and output_grad must not share storage.')
    if num_tokens == 0:
        return
    expand_to_mhc_backward_kernel(
        output_grad,
        x_grad,
        hidden_block=_BACKWARD_HIDDEN_BLOCK,
        num_stages=_BACKWARD_NUM_STAGES,
        num_vec_cores=vector_cores,
    )


@torch.library.custom_op(
    'cann_ops_tilelang::expand_to_mhc_backward',
    mutates_args=('x_grad',),
    schema='(Tensor output_grad, Tensor(a!) x_grad) -> ()',
)
def expand_to_mhc_backward(output_grad: torch.Tensor, x_grad: torch.Tensor) -> None:
    """Use runtime core configuration; configuration APIs invalidate compiled graphs."""
    _validate_expand_to_mhc_backward_metadata(output_grad, x_grad)
    cores = 1 if output_grad.numel() == 0 else resolve_vector_cores(output_grad)
    _expand_to_mhc_backward_impl(output_grad, x_grad, vector_cores=cores)


@expand_to_mhc_backward.register_fake
def _expand_to_mhc_backward_fake(output_grad: torch.Tensor, x_grad: torch.Tensor) -> None:
    _validate_expand_to_mhc_backward_metadata(output_grad, x_grad)
