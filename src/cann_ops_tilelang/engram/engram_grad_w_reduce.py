"""Public interface for reducing Engram fused-weight gradients."""

from __future__ import annotations

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .engram_grad_w_reduce_kernel import get_engram_grad_w_reduce_kernel


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor.')
    if tensor.device.type != 'npu':
        raise ValueError(f'{name} must be an NPU tensor, got {tensor.device}.')
    if device is not None and tensor.device != device:
        raise ValueError(f'{name} must be on {device}, got {tensor.device}.')
    if tensor.dtype != dtype:
        raise TypeError(f'{name} must use {dtype}, got {tensor.dtype}.')
    if not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous.')


def engram_grad_w_reduce(
    grad_w_partial: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    grad_weight_hidden: torch.Tensor,
    grad_weight_embed: torch.Tensor,
) -> None:
    """Accumulate partial fused-weight gradients into both unfused weights.

    ``grad_weight_hidden`` and ``grad_weight_embed`` are FP32 accumulation
    buffers and are updated in place. The function returns ``None``.
    """

    require_ascend()
    _check_tensor('grad_w_partial', grad_w_partial, dtype=torch.float32)
    _check_tensor('weight_hidden', weight_hidden, dtype=torch.bfloat16, device=grad_w_partial.device)
    _check_tensor('weight_embed', weight_embed, dtype=torch.bfloat16, device=grad_w_partial.device)
    _check_tensor(
        'grad_weight_hidden',
        grad_weight_hidden,
        dtype=torch.float32,
        device=grad_w_partial.device,
    )
    _check_tensor(
        'grad_weight_embed',
        grad_weight_embed,
        dtype=torch.float32,
        device=grad_w_partial.device,
    )

    if grad_w_partial.ndim != 3:
        raise ValueError('grad_w_partial must have shape (num_partials, hc_mult, hidden_size).')
    num_partials, hc_mult, hidden_size = grad_w_partial.shape
    if num_partials < 1:
        raise ValueError('grad_w_partial must contain at least one partial gradient.')
    if hc_mult != 4:
        raise ValueError(f'Engram gradient reduction currently requires hc_mult == 4, got {hc_mult}.')
    if hidden_size == 0 or hidden_size % 256:
        raise ValueError(f'hidden_size must be a positive multiple of 256, got {hidden_size}.')
    expected_shape = (hc_mult, hidden_size)
    for name, tensor in (
        ('weight_hidden', weight_hidden),
        ('weight_embed', weight_embed),
        ('grad_weight_hidden', grad_weight_hidden),
        ('grad_weight_embed', grad_weight_embed),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f'{name} must have shape {expected_shape}, got {tuple(tensor.shape)}.')

    kernel = get_engram_grad_w_reduce_kernel(hidden_size, num_partials, hc_mult)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed)
