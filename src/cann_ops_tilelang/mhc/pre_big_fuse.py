"""Ascend fused MHC preprocessing operator."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend

from .pre_big_fuse_kernel import (
    _mhc_reduce_partials_and_rmsnorm_forward_kernel,
    mhc_pre_big_fuse_fwd_asc,
)

_ASCEND_MAX_H_BLK = 6144
_ASCEND_STATIC_TOKEN_COUNTS = (512, 1024, 2048, 8192)


def _ascend_pre_big_fuse_config(num_tokens: int, hidden_size: int) -> dict[str, int | bool]:
    """Mirror the current TileKernels main launch policy for Ascend."""

    use_batched_residual = num_tokens == 8192 and hidden_size == 7168
    if use_batched_residual:
        hidden_block = 256
        residual_token_block = 16
        residual_num_stages = 4
    else:
        hidden_block = min(hidden_size, _ASCEND_MAX_H_BLK)
        hidden_block -= hidden_block % 64
        while hidden_block > 0 and hidden_size % hidden_block:
            hidden_block -= 64
        if hidden_block == 0:
            raise ValueError(f'No 64-aligned hidden block divides hidden={hidden_size}.')
        residual_token_block = 1
        residual_num_stages = 3

    return {
        'h_blk': hidden_block,
        'residual_token_block': residual_token_block,
        'residual_num_stages': residual_num_stages,
        'static_num_tokens': num_tokens if num_tokens in _ASCEND_STATIC_TOKEN_COUNTS else 0,
        'use_fma_affine': num_tokens == 8192,
    }


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


def _check_real(name: str, value: float, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number.')
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be finite.')
    if non_negative and converted < 0:
        raise ValueError(f'{name} must be non-negative.')
    return converted


def _check_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer.')
    converted = int(value)
    if converted <= 0:
        raise ValueError(f'{name} must be positive, got {converted}.')
    return converted


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


def _mhc_reduce_partials_and_rmsnorm_forward(
    out_mul_splitted: torch.Tensor,
    sqrsum_splitted: torch.Tensor,
    out_mul: torch.Tensor,
    sqrsum: torch.Tensor,
    out: torch.Tensor,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
) -> None:
    """Run the private reduce/RMSNorm reference stage used by fused tests."""

    tensors = (out_mul_splitted, sqrsum_splitted, out_mul, sqrsum, out)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('all reduce/RMSNorm inputs and outputs must be torch.Tensor instances.')
    require_ascend()
    _check_tensor('out_mul_splitted', out_mul_splitted, torch.float32, 4)
    device = out_mul_splitted.device
    _check_tensor('sqrsum_splitted', sqrsum_splitted, torch.float32, 3, device=device)
    _check_tensor('out_mul', out_mul, torch.float32, 3, device=device)
    _check_tensor('sqrsum', sqrsum, torch.float32, 2, device=device)
    _check_tensor('out', out, torch.float32, 2, device=device)

    n_splits = _check_positive_int('n_splits', n_splits)
    rms_group_size = _check_positive_int('rms_group_size', rms_group_size)
    rms_eps = _check_real('rms_eps', rms_eps, non_negative=True)
    if out_mul_splitted.shape[0] != n_splits:
        raise ValueError(
            f'out_mul_splitted first dimension must equal n_splits={n_splits}, got {out_mul_splitted.shape[0]}.'
        )
    _, num_tokens, num_rms_groups, mhc_mult3 = out_mul_splitted.shape
    if num_rms_groups != 1 or mhc_mult3 != 24:
        raise ValueError(
            f'out_mul_splitted must have shape (n_splits, num_tokens, 1, 24), got {tuple(out_mul_splitted.shape)}.'
        )
    expected_shapes = {
        'sqrsum_splitted': (n_splits, num_tokens, 1),
        'out_mul': (num_tokens, 1, 24),
        'sqrsum': (num_tokens, 1),
        'out': (num_tokens, 24),
    }
    actual_tensors = {
        'sqrsum_splitted': sqrsum_splitted,
        'out_mul': out_mul,
        'sqrsum': sqrsum,
        'out': out,
    }
    for name, expected_shape in expected_shapes.items():
        if actual_tensors[name].shape != expected_shape:
            raise ValueError(f'{name} must have shape {expected_shape}, got {tuple(actual_tensors[name].shape)}.')
    _check_output_aliases(
        {
            'out_mul_splitted': out_mul_splitted,
            'sqrsum_splitted': sqrsum_splitted,
        },
        {'out_mul': out_mul, 'sqrsum': sqrsum, 'out': out},
    )
    if num_tokens == 0:
        return

    _mhc_reduce_partials_and_rmsnorm_forward_kernel(
        out_mul_splitted,
        sqrsum_splitted,
        out_mul,
        sqrsum,
        out,
        rms_group_size,
        rms_eps,
        n_splits,
        get_num_vector_cores(),
    )


def mhc_pre_big_fuse(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
) -> None:
    """Run fused MHC preprocessing into caller-provided output buffers."""

    tensors = (
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('all MHC pre-big-fuse inputs and outputs must be torch.Tensor instances.')
    require_ascend()
    _check_tensor('gemm_out_mul', gemm_out_mul, torch.float32, 3)
    device = gemm_out_mul.device
    _check_tensor('gemm_out_sqrsum', gemm_out_sqrsum, torch.float32, 2, device=device)
    _check_tensor('mhc_scale', mhc_scale, torch.float32, 1, device=device)
    _check_tensor('mhc_base', mhc_base, torch.float32, 1, device=device)
    _check_tensor('residual', residual, torch.bfloat16, 3, device=device)
    _check_tensor('post_mix', post_mix, torch.float32, 2, device=device)
    _check_tensor('comb_mix', comb_mix, torch.float32, 2, device=device)
    _check_tensor('layer_input', layer_input, torch.bfloat16, 2, device=device)

    rms_eps = _check_real('rms_eps', rms_eps, non_negative=True)
    mhc_pre_eps = _check_real('mhc_pre_eps', mhc_pre_eps, non_negative=True)
    mhc_sinkhorn_eps = _check_real('mhc_sinkhorn_eps', mhc_sinkhorn_eps, non_negative=True)
    mhc_post_mult_value = _check_real('mhc_post_mult_value', mhc_post_mult_value)
    sinkhorn_repeat = _check_positive_int('sinkhorn_repeat', sinkhorn_repeat)
    _check_positive_int('n_splits', n_splits)

    num_tokens, mhc, hidden = residual.shape
    if mhc != 4:
        raise ValueError(f'residual must use mHC4, got mhc={mhc}.')
    if hidden <= 0 or hidden % 64:
        raise ValueError(f'hidden must be a positive multiple of 64, got {hidden}.')
    expected_shapes = {
        'gemm_out_mul': (1, num_tokens, 24),
        'gemm_out_sqrsum': (1, num_tokens),
        'mhc_scale': (3,),
        'mhc_base': (24,),
        'post_mix': (num_tokens, 4),
        'comb_mix': (num_tokens, 16),
        'layer_input': (num_tokens, hidden),
    }
    actual_tensors = {
        'gemm_out_mul': gemm_out_mul,
        'gemm_out_sqrsum': gemm_out_sqrsum,
        'mhc_scale': mhc_scale,
        'mhc_base': mhc_base,
        'post_mix': post_mix,
        'comb_mix': comb_mix,
        'layer_input': layer_input,
    }
    for name, expected_shape in expected_shapes.items():
        if actual_tensors[name].shape != expected_shape:
            raise ValueError(f'{name} must have shape {expected_shape}, got {tuple(actual_tensors[name].shape)}.')
    _check_output_aliases(
        {
            'gemm_out_mul': gemm_out_mul,
            'gemm_out_sqrsum': gemm_out_sqrsum,
            'mhc_scale': mhc_scale,
            'mhc_base': mhc_base,
            'residual': residual,
        },
        {'post_mix': post_mix, 'comb_mix': comb_mix, 'layer_input': layer_input},
    )
    if num_tokens == 0:
        return

    config = _ascend_pre_big_fuse_config(num_tokens, hidden)
    mhc_pre_big_fuse_fwd_asc(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        h_blk=config['h_blk'],
        residual_token_block=config['residual_token_block'],
        residual_num_stages=config['residual_num_stages'],
        use_fma_affine=config['use_fma_affine'],
        static_num_tokens=config['static_num_tokens'],
        n_splits=1,
    )
