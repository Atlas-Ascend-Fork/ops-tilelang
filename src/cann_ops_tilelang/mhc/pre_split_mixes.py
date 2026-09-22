"""Public interfaces for the MHC pre-split-mixes operators."""

from __future__ import annotations

import math
from numbers import Real

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend

from .pre_split_mixes_kernel import (
    mhc_pre_split_mixes_backward_kernel,
    mhc_pre_split_mixes_forward_kernel,
)

_MHC_MULT = 4
_MHC_MULT2 = _MHC_MULT * _MHC_MULT
_MHC_MULT3 = _MHC_MULT * 2 + _MHC_MULT2
_VEC = 64


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


def _check_real(
    name: str,
    value: float,
    *,
    non_negative: bool = False,
    non_zero: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number.')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite.')
    if non_negative and result < 0:
        raise ValueError(f'{name} must be non-negative.')
    if non_zero and result == 0:
        raise ValueError(f'{name} must be non-zero.')
    return result


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


def mhc_pre_split_mixes_forward(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    pre_layer_mix: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> None:
    """Split affine MHC mixes into pre, post, and residual-combination outputs.

    All tensors are caller-owned contiguous FP32 NPU buffers. ``input_mixes``
    has shape ``(num_tokens, 24)``; the three outputs have shapes
    ``(num_tokens, 4)``, ``(num_tokens, 4)``, and ``(num_tokens, 16)``.
    """

    post_mult_value = _check_real('mhc_post_mult_value', mhc_post_mult_value)
    pre_eps = _check_real('mhc_pre_eps', mhc_pre_eps, non_negative=True)
    require_ascend()

    if not isinstance(input_mixes, torch.Tensor):
        raise TypeError('input_mixes must be a torch.Tensor.')
    if input_mixes.ndim != 2 or input_mixes.shape[1] != _MHC_MULT3:
        raise ValueError(f'input_mixes must have shape (num_tokens, {_MHC_MULT3}).')
    num_tokens = input_mixes.shape[0]
    device = input_mixes.device
    _check_npu_fp32_tensor('input_mixes', input_mixes, shape=(num_tokens, _MHC_MULT3))
    _check_npu_fp32_tensor('mhc_scale', mhc_scale, shape=(3,), device=device)
    _check_npu_fp32_tensor('mhc_base', mhc_base, shape=(_MHC_MULT3,), device=device)
    _check_npu_fp32_tensor('pre_layer_mix', pre_layer_mix, shape=(num_tokens, _MHC_MULT), device=device)
    _check_npu_fp32_tensor('post_layer_mix', post_layer_mix, shape=(num_tokens, _MHC_MULT), device=device)
    _check_npu_fp32_tensor('comb_res_mix', comb_res_mix, shape=(num_tokens, _MHC_MULT2), device=device)

    inputs = (
        ('input_mixes', input_mixes),
        ('mhc_scale', mhc_scale),
        ('mhc_base', mhc_base),
    )
    outputs = (
        ('pre_layer_mix', pre_layer_mix),
        ('post_layer_mix', post_layer_mix),
        ('comb_res_mix', comb_res_mix),
    )
    _check_output_aliases(inputs, outputs)
    if num_tokens == 0:
        return

    use_8192_forward = num_tokens == 8192
    token_block_size = 128 if use_8192_forward else (16 if num_tokens <= 4096 else 128)
    mhc_pre_split_mixes_forward_kernel(
        input_mixes,
        mhc_scale,
        mhc_base,
        pre_layer_mix,
        post_layer_mix,
        comb_res_mix,
        post_mult_value,
        pre_eps,
        static_num_tokens=num_tokens if use_8192_forward else 0,
        token_block_size=token_block_size,
        pre_vecs=math.ceil(token_block_size * _MHC_MULT / _VEC),
        comb_vecs=math.ceil(token_block_size * _MHC_MULT2 / _VEC),
        full_tiles=num_tokens % token_block_size == 0,
        num_vec_cores=get_num_vector_cores(),
    )


def mhc_pre_split_mixes_backward(
    pre_layer_mix_grad: torch.Tensor,
    post_layer_mix_grad: torch.Tensor,
    comb_res_mix_grad: torch.Tensor,
    input_mixes: torch.Tensor,
    pre_layer_mix: torch.Tensor,
    post_layer_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    input_mixes_grad: torch.Tensor,
    mhc_scale_grad: torch.Tensor,
    mhc_base_grad: torch.Tensor,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> None:
    """Compute input, scale, and base gradients for MHC pre-split mixes."""

    post_mult_value = _check_real('mhc_post_mult_value', mhc_post_mult_value, non_zero=True)
    pre_eps = _check_real('mhc_pre_eps', mhc_pre_eps, non_negative=True)
    require_ascend()

    if not isinstance(input_mixes, torch.Tensor):
        raise TypeError('input_mixes must be a torch.Tensor.')
    if input_mixes.ndim != 2 or input_mixes.shape[1] != _MHC_MULT3:
        raise ValueError(f'input_mixes must have shape (num_tokens, {_MHC_MULT3}).')
    num_tokens = input_mixes.shape[0]
    device = input_mixes.device

    input_specs = (
        ('pre_layer_mix_grad', pre_layer_mix_grad, (num_tokens, _MHC_MULT)),
        ('post_layer_mix_grad', post_layer_mix_grad, (num_tokens, _MHC_MULT)),
        ('comb_res_mix_grad', comb_res_mix_grad, (num_tokens, _MHC_MULT2)),
        ('input_mixes', input_mixes, (num_tokens, _MHC_MULT3)),
        ('pre_layer_mix', pre_layer_mix, (num_tokens, _MHC_MULT)),
        ('post_layer_mix', post_layer_mix, (num_tokens, _MHC_MULT)),
        ('mhc_scale', mhc_scale, (3,)),
    )
    for name, tensor, shape in input_specs:
        _check_npu_fp32_tensor(name, tensor, shape=shape, device=device)

    output_specs = (
        ('input_mixes_grad', input_mixes_grad, (num_tokens, _MHC_MULT3)),
        ('mhc_scale_grad', mhc_scale_grad, (3,)),
        ('mhc_base_grad', mhc_base_grad, (_MHC_MULT3,)),
    )
    for name, tensor, shape in output_specs:
        _check_npu_fp32_tensor(name, tensor, shape=shape, device=device)

    inputs = tuple((name, tensor) for name, tensor, _shape in input_specs)
    outputs = tuple((name, tensor) for name, tensor, _shape in output_specs)
    _check_output_aliases(inputs, outputs)
    if num_tokens == 0:
        mhc_scale_grad.zero_()
        mhc_base_grad.zero_()
        return

    use_8192_backward = num_tokens == 8192
    token_block_size = 128 if use_8192_backward else (16 if num_tokens <= 4096 else 64)
    num_vec_cores = 32 if use_8192_backward else get_num_vector_cores()
    mhc_grad_partial = torch.empty(
        (num_vec_cores, _MHC_MULT3 + 3),
        dtype=torch.float32,
        device=device,
    )
    mhc_pre_split_mixes_backward_kernel(
        pre_layer_mix_grad,
        post_layer_mix_grad,
        comb_res_mix_grad,
        input_mixes,
        pre_layer_mix,
        post_layer_mix,
        mhc_scale,
        input_mixes_grad,
        mhc_grad_partial,
        post_mult_value,
        pre_eps,
        static_num_tokens=num_tokens if use_8192_backward else 0,
        token_block_size=token_block_size,
        full_tiles=use_8192_backward,
        use_simd_index=use_8192_backward,
        pre_index_shift=2 if use_8192_backward else 0,
        comb_index_shift=4 if use_8192_backward else 0,
        num_vec_cores=num_vec_cores,
    )
    mhc_grad = mhc_grad_partial.sum(0)
    mhc_scale_grad.copy_(mhc_grad[:3])
    mhc_base_grad.copy_(mhc_grad[3:])
