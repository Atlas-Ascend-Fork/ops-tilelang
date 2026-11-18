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

"""Public interfaces for the MHC head mix transform."""

from __future__ import annotations

import math
from numbers import Real

import torch

from cann_ops_tilelang.common.compile_utils import resolve_vector_cores
from cann_ops_tilelang.config import require_ascend

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


def _check_num_vec_cores(value: int, *, vector_cores: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('num_vec_cores must be an integer.')
    detected = vector_cores
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


def _check_num_vec_cores_metadata(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('num_vec_cores must be an integer.')
    if value <= 0:
        raise ValueError('num_vec_cores must be positive.')
    return value


def _validate_mhc_head_compute_mix_forward_metadata(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    output_mix: torch.Tensor,
    mhc_pre_eps: float,
    token_block_size: int = 0,
) -> tuple:
    eps = _check_eps(mhc_pre_eps)
    num_tokens, device = _validate_inputs(input_mix, mhc_scale, mhc_base)
    _check_npu_fp32_tensor('output_mix', output_mix, shape=(num_tokens, _MHC_MULT), device=device)
    selected_token_block_size = _choose_token_block_size(token_block_size)
    return (eps, num_tokens, selected_token_block_size)


def _mhc_head_compute_mix_forward_impl(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    output_mix: torch.Tensor,
    mhc_pre_eps: float,
    token_block_size: int = 0,
) -> None:
    """Apply the FP32 affine sigmoid transform for four MHC streams."""
    eps, num_tokens, selected_token_block_size = _validate_mhc_head_compute_mix_forward_metadata(
        input_mix, mhc_scale, mhc_base, output_mix, mhc_pre_eps, token_block_size
    )
    require_ascend()
    _check_output_aliases(
        (('input_mix', input_mix), ('mhc_scale', mhc_scale), ('mhc_base', mhc_base)), (('output_mix', output_mix),)
    )
    if num_tokens == 0:
        return
    mhc_head_compute_mix_forward_kernel(
        input_mix, mhc_scale, mhc_base, output_mix, eps, token_block_size=selected_token_block_size
    )


@torch.library.custom_op(
    'cann_ops_tilelang::mhc_head_compute_mix_forward',
    mutates_args=('output_mix',),
    schema='(Tensor input_mix, Tensor mhc_scale, Tensor mhc_base, Tensor(a!) output_mix, '
    'Any? mhc_pre_eps, Any? token_block_size=0) -> ()',
)
def mhc_head_compute_mix_forward(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    output_mix: torch.Tensor,
    mhc_pre_eps: float,
    token_block_size: int = 0,
) -> None:
    """Apply the affine sigmoid transform, writing output_mix and returning None."""
    _mhc_head_compute_mix_forward_impl(input_mix, mhc_scale, mhc_base, output_mix, mhc_pre_eps, token_block_size)


@mhc_head_compute_mix_forward.register_fake
def _mhc_head_compute_mix_forward_fake(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    output_mix: torch.Tensor,
    mhc_pre_eps: float,
    token_block_size: int = 0,
) -> None:
    _validate_mhc_head_compute_mix_forward_metadata(
        input_mix, mhc_scale, mhc_base, output_mix, mhc_pre_eps, token_block_size
    )


def _validate_mhc_head_compute_mix_backward_metadata(
    output_mix_grad: torch.Tensor,
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    input_mix_grad: torch.Tensor,
    mhc_scale_grad_partial: torch.Tensor,
    mhc_base_grad_partial: torch.Tensor,
    num_vec_cores: int,
    token_block_size: int = 0,
) -> tuple:
    num_vec_cores = _check_num_vec_cores_metadata(num_vec_cores)
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
    selected_token_block_size = _choose_token_block_size(token_block_size)
    return (inputs, num_tokens, outputs, selected_token_block_size, num_vec_cores)


def _mhc_head_compute_mix_backward_impl(arguments, *, vector_cores: int) -> None:
    """Validate and launch backward using the caller's gradient buffers."""
    inputs, num_tokens, outputs, tile, requested = _validate_mhc_head_compute_mix_backward_metadata(*arguments)
    require_ascend()
    _check_output_aliases(inputs, outputs)
    requested = _check_num_vec_cores(requested, vector_cores=vector_cores)
    if num_tokens == 0:
        for _, partial in outputs[1:]:
            partial.zero_()
        return
    mhc_head_compute_mix_backward_kernel(*arguments[:7], token_block_size=tile, num_vec_cores=requested)


@torch.library.custom_op(
    'cann_ops_tilelang::mhc_head_compute_mix_backward',
    mutates_args=('input_mix_grad', 'mhc_scale_grad_partial', 'mhc_base_grad_partial'),
    schema='(Tensor output_mix_grad, Tensor input_mix, Tensor mhc_scale, Tensor mhc_base, '
    'Tensor(a!) input_mix_grad, Tensor(b!) mhc_scale_grad_partial, Tensor(c!) mhc_base_grad_partial, '
    'Any? num_vec_cores, Any? token_block_size=0) -> ()',
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
    """Write input/partial gradients; num_vec_cores determines caller buffer shapes.

    Runtime configuration supplies the launch limit; configuration APIs
    invalidate compiled graphs when that configuration changes. Fake checks
    the supplied buffer shapes without querying runtime configuration.
    """
    arguments = (
        output_mix_grad,
        input_mix,
        mhc_scale,
        mhc_base,
        input_mix_grad,
        mhc_scale_grad_partial,
        mhc_base_grad_partial,
        num_vec_cores,
        token_block_size,
    )
    _validate_mhc_head_compute_mix_backward_metadata(*arguments)
    vector_cores = resolve_vector_cores(output_mix_grad)
    _mhc_head_compute_mix_backward_impl(arguments, vector_cores=vector_cores)


@mhc_head_compute_mix_backward.register_fake
def _mhc_head_compute_mix_backward_fake(*args, **kwargs) -> None:
    if len(args) < 9:
        kwargs.setdefault('token_block_size', 0)
    _validate_mhc_head_compute_mix_backward_metadata(*args, **kwargs)
