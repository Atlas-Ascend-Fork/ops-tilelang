"""Public interface for the Engram fused-weight operator."""

from __future__ import annotations

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .engram_fused_weight_kernel import get_engram_fused_weight_kernel


def _validate_inputs(weight_hidden: torch.Tensor, weight_embed: torch.Tensor) -> None:
    if not isinstance(weight_hidden, torch.Tensor) or not isinstance(weight_embed, torch.Tensor):
        raise TypeError('weight_hidden and weight_embed must be torch.Tensor instances.')
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


def _engram_fused_weight_impl(
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
) -> torch.Tensor:
    _validate_inputs(weight_hidden, weight_embed)
    require_ascend()
    result = torch.empty_like(weight_hidden, dtype=torch.float32)
    if weight_hidden.numel() == 0:
        return result

    hc_mult, hidden_size = weight_hidden.shape
    kernel = get_engram_fused_weight_kernel(hidden_size, hc_mult)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(weight_hidden.view(-1), weight_embed.view(-1), result.view(-1))
    return result


@torch.library.custom_op('cann_ops_tilelang::engram_fused_weight', mutates_args=())
def _compiled_fused_weight(weight_hidden: torch.Tensor, weight_embed: torch.Tensor) -> torch.Tensor:
    return _engram_fused_weight_impl(weight_hidden, weight_embed)


@_compiled_fused_weight.register_fake
def _fused_weight_fake(weight_hidden, weight_embed):
    _validate_inputs(weight_hidden, weight_embed)
    return torch.empty_like(weight_hidden, dtype=torch.float32)


def engram_fused_weight(
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
) -> torch.Tensor:
    """Multiply two BF16 Engram weights and return an FP32 tensor.

    The inputs must be contiguous NPU tensors with shape
    ``(hc_mult, hidden_size)``. The Ascend kernel converts both operands to
    FP32 before multiplication and returns the result without modifying
    either input.

    With a compatible torch_npu, supports inference with
    ``torch.compile(..., backend='npugraph_ex', fullgraph=True, dynamic=False)``.
    The compiled path uses an opaque custom operator with a fake implementation;
    tracing does not compile or launch the TileLang kernel. Shapes specialize
    at compilation. No autograd formula is registered for this operator.
    """
    if torch.compiler.is_compiling():
        return _compiled_fused_weight(weight_hidden, weight_embed)
    return _engram_fused_weight_impl(weight_hidden, weight_embed)
