"""Public interfaces for MHC Sinkhorn forward and backward."""

from __future__ import annotations

import math
from numbers import Real

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend

from .sinkhorn_kernel import mhc_sinkhorn_backward_kernel, mhc_sinkhorn_forward_kernel

_MHC = 4
_TOKEN_TILE = 64


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


def _check_parameters(repeat: int, eps: float) -> tuple[int, float]:
    if isinstance(repeat, bool) or not isinstance(repeat, int):
        raise TypeError('repeat must be an integer.')
    if repeat < 1:
        raise ValueError('repeat must be at least one.')
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise TypeError('eps must be a real number.')
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value < 0:
        raise ValueError('eps must be finite and non-negative.')
    return repeat, eps_value


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.numel() > 0
        and right.numel() > 0
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _launch_config(num_tokens: int) -> tuple[int, int, int]:
    """Choose a shape-aware PTO launch for one 64-token SIMD tile."""

    tile_num = math.ceil(num_tokens / _TOKEN_TILE)
    num_cores = min(get_num_vector_cores(), tile_num)
    threads = min(num_tokens, _TOKEN_TILE) * (_MHC * _MHC)
    return tile_num, num_cores, threads


def mhc_sinkhorn_forward(
    comb_res_mix: torch.Tensor,
    comb_res_mix_out: torch.Tensor,
    repeat: int,
    eps: float,
) -> None:
    """Normalize batches of 4-by-4 MHC residual-combination matrices."""

    repeat, eps_value = _check_parameters(repeat, eps)
    require_ascend()
    if not isinstance(comb_res_mix, torch.Tensor):
        raise TypeError('comb_res_mix must be a torch.Tensor.')
    if comb_res_mix.ndim != 3 or tuple(comb_res_mix.shape[1:]) != (_MHC, _MHC):
        raise ValueError(f'comb_res_mix must have shape (num_tokens, {_MHC}, {_MHC}).')
    shape = tuple(comb_res_mix.shape)
    _check_npu_fp32_tensor('comb_res_mix', comb_res_mix, shape=shape)
    _check_npu_fp32_tensor('comb_res_mix_out', comb_res_mix_out, shape=shape, device=comb_res_mix.device)
    if _shares_storage(comb_res_mix_out, comb_res_mix):
        raise ValueError('comb_res_mix_out and comb_res_mix must not share storage.')
    if shape[0] == 0:
        return

    tile_num, num_cores, threads = _launch_config(shape[0])
    mhc_sinkhorn_forward_kernel(
        comb_res_mix,
        comb_res_mix_out,
        repeat,
        eps_value,
        tile_num=tile_num,
        num_cores=num_cores,
        threads=threads,
    )


def mhc_sinkhorn_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    grad_input: torch.Tensor,
    repeat: int,
    eps: float,
) -> None:
    """Compute the input gradient for 4-by-4 MHC Sinkhorn normalization."""

    repeat, eps_value = _check_parameters(repeat, eps)
    require_ascend()
    if not isinstance(x, torch.Tensor):
        raise TypeError('x must be a torch.Tensor.')
    if x.ndim != 3 or tuple(x.shape[1:]) != (_MHC, _MHC):
        raise ValueError(f'x must have shape (num_tokens, {_MHC}, {_MHC}).')
    shape = tuple(x.shape)
    _check_npu_fp32_tensor('x', x, shape=shape)
    _check_npu_fp32_tensor('grad_output', grad_output, shape=shape, device=x.device)
    _check_npu_fp32_tensor('grad_input', grad_input, shape=shape, device=x.device)
    for input_name, input_tensor in (('grad_output', grad_output), ('x', x)):
        if _shares_storage(grad_input, input_tensor):
            raise ValueError(f'grad_input and {input_name} must not share storage.')
    if shape[0] == 0:
        return

    tile_num, num_cores, threads = _launch_config(shape[0])
    mhc_sinkhorn_backward_kernel(
        grad_output,
        x,
        grad_input,
        repeat,
        eps_value,
        tile_num=tile_num,
        num_cores=num_cores,
        threads=threads,
    )
