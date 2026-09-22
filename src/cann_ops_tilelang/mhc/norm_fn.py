"""Ascend MHC norm, GEMM, and partial-reduction operators."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch

from cann_ops_tilelang.config import get_num_ai_cores, require_ascend

from .norm_fn_kernel import (
    mhc_fn_normw_merge_bwd_asc,
    mhc_fn_normw_merge_fwd_asc,
    mhc_gemm_with_sqrsum_bwd_asc,
    mhc_gemm_with_sqrsum_fwd_asc,
    mhc_reduce_partials_and_rmsnorm_bwd_asc,
    mhc_reduce_partials_and_rmsnorm_fwd_asc,
)

_MHC_MULT3 = 24


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


def _check_output_aliases(inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
    output_items = list(outputs.items())
    for output_name, output in output_items:
        for input_name, input_tensor in inputs.items():
            if _shares_storage(output, input_tensor):
                raise ValueError(f'{output_name} must not share storage with {input_name}.')
    for index, (left_name, left) in enumerate(output_items):
        for right_name, right in output_items[index + 1 :]:
            if _shares_storage(left, right):
                raise ValueError(f'{left_name} and {right_name} must not share storage.')


def _check_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer.')
    converted = int(value)
    if converted <= 0:
        raise ValueError(f'{name} must be positive, got {converted}.')
    return converted


def _check_non_negative_real(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number.')
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f'{name} must be finite and non-negative, got {value}.')
    return converted


def _check_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tensor.shape != expected:
        raise ValueError(f'{name} must have shape {expected}, got {tuple(tensor.shape)}.')


def _ascend_forward_config(mhc_hidden_size: int, mhc_mult3: int, num_tokens: int) -> dict[str, int]:
    block_num_tokens = 128
    block_hidden = 256
    l0b_size_bytes = 64 * 1024
    l0c_size_bytes = 256 * 1024

    num_ai_cores = get_num_ai_cores()
    block_mhc_mult3 = (mhc_mult3 + 15) // 16 * 16
    num_hidden_blocks = mhc_hidden_size // block_hidden
    num_token_blocks = math.ceil(num_tokens / block_num_tokens)
    num_l0b_stages = l0b_size_bytes // (block_mhc_mult3 * block_hidden * 4)
    num_l0c_stages = l0c_size_bytes // (block_num_tokens * block_mhc_mult3 * 4)

    num_k_splits_by_tokens = max(1, num_ai_cores * num_l0c_stages // num_token_blocks)
    num_k_splits_by_hidden = num_hidden_blocks & -num_hidden_blocks
    num_k_splits_by_cores = num_ai_cores & -num_ai_cores
    split_limit = min(
        num_k_splits_by_tokens,
        num_k_splits_by_hidden,
        num_k_splits_by_cores,
    )
    num_k_splits = 1 << (split_limit.bit_length() - 1)
    active_ai_cores = num_k_splits * min(num_ai_cores // num_k_splits, num_token_blocks)
    cores_per_split = active_ai_cores // num_k_splits
    num_hidden_blocks_per_split = num_hidden_blocks // num_k_splits
    num_token_blocks_per_chunk = cores_per_split * num_l0c_stages

    return {
        'block_mhc_mult3': block_mhc_mult3,
        'num_aic_cores': active_ai_cores,
        'num_k_splits': num_k_splits,
        'cores_per_split': cores_per_split,
        'num_hidden_blocks_per_split': num_hidden_blocks_per_split,
        'num_l0b_stages': num_l0b_stages,
        'num_l0c_stages': num_l0c_stages,
        'num_token_blocks_per_chunk': num_token_blocks_per_chunk,
    }


def mhc_fn_normw_merge_forward(
    fn: torch.Tensor,
    normw: torch.Tensor,
    out_fn: torch.Tensor,
) -> None:
    """Multiply an FP32 MHC function matrix by its FP32 normalization weights."""

    if not all(isinstance(tensor, torch.Tensor) for tensor in (fn, normw, out_fn)):
        raise TypeError('fn, normw and out_fn must be torch.Tensor instances.')
    require_ascend()
    _check_tensor('fn', fn, torch.float32, 2)
    _check_tensor('normw', normw, torch.float32, 1, device=fn.device)
    _check_tensor('out_fn', out_fn, torch.float32, 2, device=fn.device)
    if fn.shape[0] != _MHC_MULT3 or fn.shape[1] <= 0 or fn.shape[1] % 256:
        raise ValueError(f'fn must have shape (24, hidden) with hidden divisible by 256, got {tuple(fn.shape)}.')
    _check_shape('normw', normw, (fn.shape[1],))
    _check_shape('out_fn', out_fn, tuple(fn.shape))
    _check_output_aliases({'fn': fn, 'normw': normw}, {'out_fn': out_fn})

    mhc_fn_normw_merge_fwd_asc(
        fn,
        normw,
        out_fn,
        num_stages=2,
    )


def mhc_fn_normw_merge_backward(
    fn: torch.Tensor,
    normw: torch.Tensor,
    out_fn_grad: torch.Tensor,
    fn_grad: torch.Tensor,
    normw_grad: torch.Tensor,
) -> None:
    """Accumulate gradients into the FP32 function and normalization buffers."""

    tensors = (fn, normw, out_fn_grad, fn_grad, normw_grad)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('all norm-weight merge inputs and outputs must be torch.Tensor instances.')
    require_ascend()
    _check_tensor('fn', fn, torch.float32, 2)
    device = fn.device
    _check_tensor('normw', normw, torch.float32, 1, device=device)
    _check_tensor('out_fn_grad', out_fn_grad, torch.float32, 2, device=device)
    _check_tensor('fn_grad', fn_grad, torch.float32, 2, device=device)
    _check_tensor('normw_grad', normw_grad, torch.float32, 1, device=device)
    if fn.shape[0] != _MHC_MULT3 or fn.shape[1] <= 0 or fn.shape[1] % 128:
        raise ValueError(f'fn must have shape (24, hidden) with hidden divisible by 128, got {tuple(fn.shape)}.')
    _check_shape('normw', normw, (fn.shape[1],))
    _check_shape('out_fn_grad', out_fn_grad, tuple(fn.shape))
    _check_shape('fn_grad', fn_grad, tuple(fn.shape))
    _check_shape('normw_grad', normw_grad, (fn.shape[1],))
    _check_output_aliases(
        {'fn': fn, 'normw': normw, 'out_fn_grad': out_fn_grad},
        {'fn_grad': fn_grad, 'normw_grad': normw_grad},
    )

    mhc_fn_normw_merge_bwd_asc(
        fn,
        normw,
        out_fn_grad,
        fn_grad,
        normw_grad,
        num_stages=2,
    )


def mhc_gemm_with_sqrsum_forward(
    x: torch.Tensor,
    fn: torch.Tensor,
    out_mul_splitted: torch.Tensor,
    sqrsum_splitted: torch.Tensor,
    n_splits: int = 16,
) -> None:
    """Compute the MHC GEMM and squared sums into one internally reduced partial."""

    tensors = (x, fn, out_mul_splitted, sqrsum_splitted)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('all GEMM forward inputs and outputs must be torch.Tensor instances.')
    require_ascend()
    _check_tensor('x', x, torch.bfloat16, 2)
    device = x.device
    _check_tensor('fn', fn, torch.float32, 2, device=device)
    _check_tensor('out_mul_splitted', out_mul_splitted, torch.float32, 4, device=device)
    _check_tensor('sqrsum_splitted', sqrsum_splitted, torch.float32, 3, device=device)
    _check_positive_int('n_splits', n_splits)

    num_tokens, mhc_hidden_size = x.shape
    if fn.shape != (_MHC_MULT3, mhc_hidden_size):
        raise ValueError(f'fn must have shape {(24, mhc_hidden_size)}, got {tuple(fn.shape)}.')
    if mhc_hidden_size <= 0 or mhc_hidden_size % 256:
        raise ValueError(f'x hidden size must be a positive multiple of 256, got {mhc_hidden_size}.')
    _check_shape('out_mul_splitted', out_mul_splitted, (1, num_tokens, 1, _MHC_MULT3))
    _check_shape('sqrsum_splitted', sqrsum_splitted, (1, num_tokens, 1))
    _check_output_aliases(
        {'x': x, 'fn': fn},
        {'out_mul_splitted': out_mul_splitted, 'sqrsum_splitted': sqrsum_splitted},
    )
    if num_tokens == 0:
        return

    config = _ascend_forward_config(mhc_hidden_size, fn.shape[0], num_tokens)
    mhc_gemm_with_sqrsum_fwd_asc(
        x,
        fn,
        out_mul_splitted[0, :, 0, :],
        sqrsum_splitted[0, :, 0],
        block_mhc_mult3=config['block_mhc_mult3'],
        num_aic_cores=config['num_aic_cores'],
        num_k_splits=config['num_k_splits'],
        cores_per_split=config['cores_per_split'],
        num_hidden_blocks_per_split=config['num_hidden_blocks_per_split'],
        num_l0b_stages=config['num_l0b_stages'],
        num_l0c_stages=config['num_l0c_stages'],
        num_token_blocks_per_chunk=config['num_token_blocks_per_chunk'],
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )


def mhc_gemm_with_sqrsum_backward(
    out_mul_grad: torch.Tensor,
    sqrsum_grad: torch.Tensor,
    x: torch.Tensor,
    fn: torch.Tensor,
    x_grad: torch.Tensor,
    fn_grad: torch.Tensor,
) -> None:
    """Accumulate the BF16 input gradient and write the FP32 function gradient."""

    tensors = (out_mul_grad, sqrsum_grad, x, fn, x_grad, fn_grad)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('all GEMM backward inputs and outputs must be torch.Tensor instances.')
    require_ascend()
    _check_tensor('x', x, torch.bfloat16, 2)
    device = x.device
    _check_tensor('out_mul_grad', out_mul_grad, torch.float32, 3, device=device)
    _check_tensor('sqrsum_grad', sqrsum_grad, torch.float32, 2, device=device)
    _check_tensor('fn', fn, torch.float32, 2, device=device)
    _check_tensor('x_grad', x_grad, torch.bfloat16, 2, device=device)
    _check_tensor('fn_grad', fn_grad, torch.float32, 2, device=device)

    num_tokens, mhc_hidden_size = x.shape
    if fn.shape != (_MHC_MULT3, mhc_hidden_size):
        raise ValueError(f'fn must have shape {(24, mhc_hidden_size)}, got {tuple(fn.shape)}.')
    if mhc_hidden_size <= 0 or mhc_hidden_size % 128:
        raise ValueError(f'x hidden size must be a positive multiple of 128, got {mhc_hidden_size}.')
    _check_shape('out_mul_grad', out_mul_grad, (num_tokens, 1, _MHC_MULT3))
    _check_shape('sqrsum_grad', sqrsum_grad, (num_tokens, 1))
    _check_shape('x_grad', x_grad, tuple(x.shape))
    _check_shape('fn_grad', fn_grad, tuple(fn.shape))
    _check_output_aliases(
        {'out_mul_grad': out_mul_grad, 'sqrsum_grad': sqrsum_grad, 'x': x, 'fn': fn},
        {'x_grad': x_grad, 'fn_grad': fn_grad},
    )
    if num_tokens == 0:
        fn_grad.zero_()
        return

    kernel_out_mul_grad = out_mul_grad
    kernel_sqrsum_grad = sqrsum_grad
    kernel_x = x
    kernel_x_grad = x_grad
    if num_tokens % 128:
        padded_tokens = (num_tokens + 127) // 128 * 128
        kernel_out_mul_grad = out_mul_grad.new_zeros((padded_tokens, 1, _MHC_MULT3))
        kernel_sqrsum_grad = sqrsum_grad.new_zeros((padded_tokens, 1))
        kernel_x = x.new_zeros((padded_tokens, mhc_hidden_size))
        kernel_x_grad = x_grad.new_zeros((padded_tokens, mhc_hidden_size))
        kernel_out_mul_grad[:num_tokens].copy_(out_mul_grad)
        kernel_sqrsum_grad[:num_tokens].copy_(sqrsum_grad)
        kernel_x[:num_tokens].copy_(x)
        kernel_x_grad[:num_tokens].copy_(x_grad)

    mhc_gemm_with_sqrsum_bwd_asc(
        kernel_out_mul_grad,
        kernel_sqrsum_grad,
        kernel_x,
        fn,
        kernel_x_grad,
        fn_grad,
        token_block=128,
        hidden_block=128,
    )
    if kernel_x_grad is not x_grad:
        x_grad.copy_(kernel_x_grad[:num_tokens])


def mhc_reduce_partials_and_rmsnorm_forward(
    out_mul_splitted: torch.Tensor,
    sqrsum_splitted: torch.Tensor,
    out_mul: torch.Tensor,
    sqrsum: torch.Tensor,
    out: torch.Tensor,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
) -> None:
    """Reduce split partials and apply RMS normalization into caller buffers."""

    tensors = (out_mul_splitted, sqrsum_splitted, out_mul, sqrsum, out)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('all reduce/RMSNorm forward inputs and outputs must be torch.Tensor instances.')
    require_ascend()
    _check_tensor('out_mul_splitted', out_mul_splitted, torch.float32, 4)
    device = out_mul_splitted.device
    _check_tensor('sqrsum_splitted', sqrsum_splitted, torch.float32, 3, device=device)
    _check_tensor('out_mul', out_mul, torch.float32, 3, device=device)
    _check_tensor('sqrsum', sqrsum, torch.float32, 2, device=device)
    _check_tensor('out', out, torch.float32, 2, device=device)
    n_splits = _check_positive_int('n_splits', n_splits)
    rms_group_size = _check_positive_int('rms_group_size', rms_group_size)
    rms_eps = _check_non_negative_real('rms_eps', rms_eps)

    if out_mul_splitted.shape[0] != n_splits:
        raise ValueError('out_mul_splitted first dimension must equal n_splits.')
    _, num_tokens, num_rms_groups, mhc_mult3 = out_mul_splitted.shape
    if num_rms_groups != 1 or mhc_mult3 != _MHC_MULT3:
        raise ValueError(
            f'out_mul_splitted must have shape (n_splits, num_tokens, 1, 24), got {tuple(out_mul_splitted.shape)}.'
        )
    _check_shape('sqrsum_splitted', sqrsum_splitted, (n_splits, num_tokens, 1))
    _check_shape('out_mul', out_mul, (num_tokens, 1, _MHC_MULT3))
    _check_shape('sqrsum', sqrsum, (num_tokens, 1))
    _check_shape('out', out, (num_tokens, _MHC_MULT3))
    _check_output_aliases(
        {'out_mul_splitted': out_mul_splitted, 'sqrsum_splitted': sqrsum_splitted},
        {'out_mul': out_mul, 'sqrsum': sqrsum, 'out': out},
    )
    if num_tokens == 0:
        return

    mhc_reduce_partials_and_rmsnorm_fwd_asc(
        out_mul_splitted,
        sqrsum_splitted,
        out_mul,
        sqrsum,
        out,
        rms_group_size,
        rms_eps,
        n_splits,
    )


def mhc_reduce_partials_and_rmsnorm_backward(
    out_grad: torch.Tensor,
    out_mul: torch.Tensor,
    sqrsum: torch.Tensor,
    out_mul_grad: torch.Tensor,
    sqrsum_grad: torch.Tensor,
    rms_group_size: int,
    rms_eps: float,
) -> None:
    """Backpropagate through the MHC partial reduction and RMS normalization."""

    tensors = (out_grad, out_mul, sqrsum, out_mul_grad, sqrsum_grad)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('all reduce/RMSNorm backward inputs and outputs must be torch.Tensor instances.')
    require_ascend()
    _check_tensor('out_grad', out_grad, torch.float32, 2)
    device = out_grad.device
    _check_tensor('out_mul', out_mul, torch.float32, 3, device=device)
    _check_tensor('sqrsum', sqrsum, torch.float32, 2, device=device)
    _check_tensor('out_mul_grad', out_mul_grad, torch.float32, 3, device=device)
    _check_tensor('sqrsum_grad', sqrsum_grad, torch.float32, 2, device=device)
    rms_group_size = _check_positive_int('rms_group_size', rms_group_size)
    rms_eps = _check_non_negative_real('rms_eps', rms_eps)

    num_tokens = out_grad.shape[0]
    _check_shape('out_grad', out_grad, (num_tokens, _MHC_MULT3))
    _check_shape('out_mul', out_mul, (num_tokens, 1, _MHC_MULT3))
    _check_shape('sqrsum', sqrsum, (num_tokens, 1))
    _check_shape('out_mul_grad', out_mul_grad, (num_tokens, 1, _MHC_MULT3))
    _check_shape('sqrsum_grad', sqrsum_grad, (num_tokens, 1))
    _check_output_aliases(
        {'out_grad': out_grad, 'out_mul': out_mul, 'sqrsum': sqrsum},
        {'out_mul_grad': out_mul_grad, 'sqrsum_grad': sqrsum_grad},
    )
    if num_tokens == 0:
        return

    mhc_reduce_partials_and_rmsnorm_bwd_asc(
        out_grad,
        out_mul,
        sqrsum,
        out_mul_grad,
        sqrsum_grad,
        rms_group_size,
        rms_eps,
    )
