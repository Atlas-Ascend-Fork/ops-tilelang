"""Public interfaces for expanding activations across MHC streams."""

from __future__ import annotations

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend

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


def expand_to_mhc_forward(x: torch.Tensor, output: torch.Tensor) -> None:
    """Copy every BF16 activation row into each MHC stream on Ascend."""

    require_ascend()
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
    _check_npu_bfloat16_tensor(
        'output',
        output,
        shape=(num_tokens, mhc_mult, hidden_size),
        device=x.device,
    )
    if _shares_storage(output, x):
        raise ValueError('output and x must not share storage.')
    if num_tokens == 0:
        return

    expand_to_mhc_forward_kernel(
        x,
        output,
        hidden_block=_forward_hidden_block(num_tokens, hidden_size),
        num_stages=_FORWARD_NUM_STAGES,
        num_vec_cores=get_num_vector_cores(),
    )


def expand_to_mhc_backward(output_grad: torch.Tensor, x_grad: torch.Tensor) -> None:
    """Sum BF16 MHC stream gradients into a caller-owned activation gradient."""

    require_ascend()
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
    _check_npu_bfloat16_tensor(
        'x_grad',
        x_grad,
        shape=(num_tokens, hidden_size),
        device=output_grad.device,
    )
    if _shares_storage(x_grad, output_grad):
        raise ValueError('x_grad and output_grad must not share storage.')
    if num_tokens == 0:
        return

    expand_to_mhc_backward_kernel(
        output_grad,
        x_grad,
        hidden_block=_BACKWARD_HIDDEN_BLOCK,
        num_stages=_BACKWARD_NUM_STAGES,
        num_vec_cores=get_num_vector_cores(),
    )
