"""Ascend MHC pre-apply mix forward and backward operators."""

from __future__ import annotations

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend
from cann_ops_tilelang.utils import align_up, ceil_div

from .pre_apply_mix_kernel import mhc_pre_apply_mix_bwd_asc, mhc_pre_apply_mix_fwd_asc


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    dtype: torch.dtype,
    ndim: int,
    *,
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
    if tensor.ndim != ndim:
        raise ValueError(f'{name} must be {ndim}-dimensional, got shape {tuple(tensor.shape)}.')
    if not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous.')


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.numel() > 0
        and right.numel() > 0
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _check_shape(num_tokens: int, mhc: int, hidden: int) -> None:
    if num_tokens < 0:
        raise ValueError(f'num_tokens must be non-negative, got {num_tokens}.')
    if mhc <= 0:
        raise ValueError(f'mhc must be positive, got {mhc}.')
    if hidden <= 0 or hidden % 64:
        raise ValueError(f'hidden must be a positive multiple of 64, got {hidden}.')


def mhc_pre_apply_mix_forward(
    x: torch.Tensor,
    mix: torch.Tensor,
    o: torch.Tensor,
) -> None:
    """Apply token-local MHC mixing into a preallocated BF16 output.

    ``x`` has shape ``(num_tokens, mhc, hidden)``, ``mix`` has shape
    ``(num_tokens, mhc)``, and ``o`` has shape
    ``(num_tokens, hidden)``.
    """

    if not all(isinstance(tensor, torch.Tensor) for tensor in (x, mix, o)):
        raise TypeError('x, mix and o must be torch.Tensor instances.')
    require_ascend()
    _check_tensor('x', x, torch.bfloat16, 3)
    _check_tensor('mix', mix, torch.float32, 2, device=x.device)
    _check_tensor('o', o, torch.bfloat16, 2, device=x.device)

    num_tokens, mhc, hidden = x.shape
    _check_shape(num_tokens, mhc, hidden)
    if mix.shape != (num_tokens, mhc):
        raise ValueError(f'mix must have shape {(num_tokens, mhc)}, got {tuple(mix.shape)}.')
    if o.shape != (num_tokens, hidden):
        raise ValueError(f'o must have shape {(num_tokens, hidden)}, got {tuple(o.shape)}.')
    if _shares_storage(o, x) or _shares_storage(o, mix):
        raise ValueError('o must not share storage with x or mix.')
    if num_tokens == 0:
        return

    hidden_block = hidden if hidden <= 6144 else min(ceil_div(hidden, 2), 6144)
    hidden_block = align_up(hidden_block, 64)
    mhc_pre_apply_mix_fwd_asc(
        x,
        mix,
        o,
        h_blk=hidden_block,
        num_vec_cores=get_num_vector_cores(),
    )


def mhc_pre_apply_mix_backward(
    o_grad: torch.Tensor,
    x: torch.Tensor,
    mix: torch.Tensor,
    x_grad: torch.Tensor,
) -> torch.Tensor:
    """Accumulate ``x`` gradients and return the FP32 mix gradient."""

    tensors = (o_grad, x, mix, x_grad)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('o_grad, x, mix and x_grad must be torch.Tensor instances.')
    require_ascend()
    _check_tensor('x', x, torch.bfloat16, 3)
    _check_tensor('o_grad', o_grad, torch.bfloat16, 2, device=x.device)
    _check_tensor('mix', mix, torch.float32, 2, device=x.device)
    _check_tensor('x_grad', x_grad, torch.bfloat16, 3, device=x.device)

    num_tokens, mhc, hidden = x.shape
    _check_shape(num_tokens, mhc, hidden)
    if o_grad.shape != (num_tokens, hidden):
        raise ValueError(f'o_grad must have shape {(num_tokens, hidden)}, got {tuple(o_grad.shape)}.')
    if mix.shape != (num_tokens, mhc):
        raise ValueError(f'mix must have shape {(num_tokens, mhc)}, got {tuple(mix.shape)}.')
    if x_grad.shape != x.shape:
        raise ValueError(f'x_grad must have shape {tuple(x.shape)}, got {tuple(x_grad.shape)}.')
    for name, tensor in (('o_grad', o_grad), ('x', x), ('mix', mix)):
        if _shares_storage(x_grad, tensor):
            raise ValueError(f'x_grad must not share storage with {name}.')
    if num_tokens == 0:
        return torch.empty((0, mhc), dtype=torch.float32, device=x.device)

    if hidden <= 4096:
        hidden_block = hidden
        num_stages = 3
    else:
        hidden_block = align_up(min(ceil_div(hidden, 4), 3072), 64)
        num_stages = 4
    return mhc_pre_apply_mix_bwd_asc(
        o_grad,
        x,
        mix,
        x_grad,
        h_blk=hidden_block,
        num_stages=num_stages,
        num_vec_cores=get_num_vector_cores(),
    )
