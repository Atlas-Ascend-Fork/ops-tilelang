"""Public interfaces for the MHC head mix transform."""

from __future__ import annotations

import math
from numbers import Real

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend

from .head_compute_mix_kernel import (
    mhc_head_compute_mix_backward_kernel,
    mhc_head_compute_mix_forward_kernel,
)

_MHC_MULT = 4
_MAX_TOKEN_BLOCK_SIZE = 64
_ASCEND_TOKEN_BLOCK_SIZE = 512


def _check_npu_fp32_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor.')
    if tensor.device.type != 'npu':
        raise ValueError(f'{name} must be an NPU tensor, got {tensor.device}.')
    if device is not None and tensor.device != device:
        raise ValueError(f'{name} must be on {device}, got {tensor.device}.')
    if tensor.dtype != torch.float32:
        raise TypeError(f'{name} must use torch.float32, got {tensor.dtype}.')
    if tuple(tensor.shape) != shape:
        raise ValueError(f'{name} must have shape {shape}, got {tuple(tensor.shape)}.')
    if not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous.')


def _check_eps(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError('mhc_pre_eps must be a real number.')
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('mhc_pre_eps must be finite and non-negative.')
    return result


def _check_num_vec_cores(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('num_vec_cores must be an integer.')
    detected = get_num_vector_cores()
    if value <= 0 or value > detected:
        raise ValueError(f'num_vec_cores must be in [1, {detected}], got {value}.')
    return value


def _choose_token_block_size(token_block_size: int) -> int:
    if isinstance(token_block_size, bool) or not isinstance(token_block_size, int):
        raise TypeError('token_block_size must be an integer.')
    if token_block_size < 0:
        raise ValueError(f'token_block_size must be non-negative, got {token_block_size}.')
    if token_block_size and (token_block_size > _MAX_TOKEN_BLOCK_SIZE or token_block_size & (token_block_size - 1)):
        raise ValueError(f'token_block_size must be a power of two no greater than {_MAX_TOKEN_BLOCK_SIZE}.')
    # Match TileKernels-Nightly's Ascend specialization. The public tuning
    # argument remains accepted and validated for API compatibility, but the
    # Ascend implementation always uses its optimized 512-token tile.
    return _ASCEND_TOKEN_BLOCK_SIZE


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.numel() > 0
        and right.numel() > 0
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _check_output_aliases(
    inputs: tuple[tuple[str, torch.Tensor], ...],
    outputs: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    for output_index, (output_name, output) in enumerate(outputs):
        for other_name, other in inputs + outputs[:output_index]:
            if _shares_storage(output, other):
                raise ValueError(f'{output_name} and {other_name} must not share storage.')


def _validate_inputs(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
) -> tuple[int, torch.device]:
    if not isinstance(input_mix, torch.Tensor):
        raise TypeError('input_mix must be a torch.Tensor.')
    if input_mix.ndim != 2 or input_mix.shape[1] != _MHC_MULT:
        raise ValueError(f'input_mix must have shape (num_tokens, {_MHC_MULT}).')
    num_tokens = input_mix.shape[0]
    device = input_mix.device
    _check_npu_fp32_tensor('input_mix', input_mix, shape=(num_tokens, _MHC_MULT))
    _check_npu_fp32_tensor('mhc_scale', mhc_scale, shape=(1,), device=device)
    _check_npu_fp32_tensor('mhc_base', mhc_base, shape=(_MHC_MULT,), device=device)
    return num_tokens, device


def mhc_head_compute_mix_forward(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    output_mix: torch.Tensor,
    mhc_pre_eps: float,
    token_block_size: int = 0,
) -> None:
    """Apply the FP32 affine sigmoid transform for four MHC streams."""

    eps = _check_eps(mhc_pre_eps)
    require_ascend()
    num_tokens, device = _validate_inputs(input_mix, mhc_scale, mhc_base)
    _check_npu_fp32_tensor(
        'output_mix',
        output_mix,
        shape=(num_tokens, _MHC_MULT),
        device=device,
    )
    _check_output_aliases(
        (('input_mix', input_mix), ('mhc_scale', mhc_scale), ('mhc_base', mhc_base)),
        (('output_mix', output_mix),),
    )
    selected_token_block_size = _choose_token_block_size(token_block_size)
    if num_tokens == 0:
        return

    mhc_head_compute_mix_forward_kernel(
        input_mix,
        mhc_scale,
        mhc_base,
        output_mix,
        eps,
        token_block_size=selected_token_block_size,
    )


def mhc_head_compute_mix_backward(
    output_mix_grad: torch.Tensor,
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    input_mix_grad: torch.Tensor,
    mhc_scale_grad_partial: torch.Tensor,
    mhc_base_grad_partial: torch.Tensor,
    num_vec_cores: int,
    token_block_size: int = 0,
) -> None:
    """Compute input and partial parameter gradients for the MHC head mix."""

    require_ascend()
    num_vec_cores = _check_num_vec_cores(num_vec_cores)
    num_tokens, device = _validate_inputs(input_mix, mhc_scale, mhc_base)
    tensor_specs = (
        ('output_mix_grad', output_mix_grad, (num_tokens, _MHC_MULT)),
        ('input_mix_grad', input_mix_grad, (num_tokens, _MHC_MULT)),
        ('mhc_scale_grad_partial', mhc_scale_grad_partial, (num_vec_cores, 1)),
        ('mhc_base_grad_partial', mhc_base_grad_partial, (num_vec_cores, _MHC_MULT)),
    )
    for name, tensor, shape in tensor_specs:
        _check_npu_fp32_tensor(name, tensor, shape=shape, device=device)

    inputs = (
        ('output_mix_grad', output_mix_grad),
        ('input_mix', input_mix),
        ('mhc_scale', mhc_scale),
        ('mhc_base', mhc_base),
    )
    outputs = (
        ('input_mix_grad', input_mix_grad),
        ('mhc_scale_grad_partial', mhc_scale_grad_partial),
        ('mhc_base_grad_partial', mhc_base_grad_partial),
    )
    _check_output_aliases(inputs, outputs)
    selected_token_block_size = _choose_token_block_size(token_block_size)
    if num_tokens == 0:
        mhc_scale_grad_partial.zero_()
        mhc_base_grad_partial.zero_()
        return

    mhc_head_compute_mix_backward_kernel(
        output_mix_grad,
        input_mix,
        mhc_scale,
        mhc_base,
        input_mix_grad,
        mhc_scale_grad_partial,
        mhc_base_grad_partial,
        token_block_size=selected_token_block_size,
        num_vec_cores=num_vec_cores,
    )
