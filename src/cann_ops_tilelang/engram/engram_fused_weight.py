"""Public interface for the Engram fused-weight operator."""

from __future__ import annotations

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .engram_fused_weight_kernel import get_engram_fused_weight_kernel


def _validate_inputs(
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    out: torch.Tensor | None,
) -> None:
    if not isinstance(weight_hidden, torch.Tensor) or not isinstance(weight_embed, torch.Tensor):
        raise TypeError('weight_hidden and weight_embed must be torch.Tensor instances.')
    require_ascend()
    if weight_hidden.device.type != 'npu' or weight_embed.device.type != 'npu':
        raise ValueError(
            f'weight_hidden and weight_embed must be NPU tensors, got {weight_hidden.device} and {weight_embed.device}.'
        )
    if weight_hidden.device != weight_embed.device:
        raise ValueError('weight_hidden and weight_embed must be on the same NPU device.')
    if weight_hidden.ndim != 2 or weight_embed.ndim != 2:
        raise ValueError('weight_hidden and weight_embed must be two-dimensional.')
    if weight_hidden.shape != weight_embed.shape:
        raise ValueError(
            f'weight_hidden and weight_embed must have the same shape, got '
            f'{tuple(weight_hidden.shape)} and {tuple(weight_embed.shape)}.'
        )
    if weight_hidden.dtype != torch.bfloat16 or weight_embed.dtype != torch.bfloat16:
        raise TypeError('weight_hidden and weight_embed must both use torch.bfloat16.')
    if not weight_hidden.is_contiguous() or not weight_embed.is_contiguous():
        raise ValueError('weight_hidden and weight_embed must be contiguous.')
    if weight_hidden.numel() and weight_hidden.numel() % 256:
        raise ValueError('the total number of input elements must be divisible by 256.')

    if out is None:
        return
    if not isinstance(out, torch.Tensor):
        raise TypeError('out must be a torch.Tensor when provided.')
    if out.shape != weight_hidden.shape:
        raise ValueError('out must have the same shape as weight_hidden.')
    if out.dtype != torch.float32 or out.device != weight_hidden.device:
        raise ValueError('out must be float32 and on the same NPU device as weight_hidden.')
    if not out.is_contiguous():
        raise ValueError('out must be contiguous.')


def engram_fused_weight(
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multiply two BF16 Engram weights and return an FP32 tensor.

    The inputs must be contiguous NPU tensors with shape
    ``(hc_mult, hidden_size)``. The Ascend kernel converts both operands to
    FP32 before multiplication.
    """

    _validate_inputs(weight_hidden, weight_embed, out)
    result = torch.empty_like(weight_hidden, dtype=torch.float32) if out is None else out
    if weight_hidden.numel() == 0:
        return result

    hc_mult, hidden_size = weight_hidden.shape
    kernel = get_engram_fused_weight_kernel(hidden_size, hc_mult)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(weight_hidden.view(-1), weight_embed.view(-1), result.view(-1))
    return result
