"""Public interfaces for the Engram Sinkhorn operator family."""

from __future__ import annotations

import math
from numbers import Real

import torch
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend, should_print_kernel_source

from .engram_sinkhorn_kernel import (
    get_engram_sinkhorn_finalize_kernel,
    get_engram_sinkhorn_momentum_update_kernel,
    get_engram_sinkhorn_step_kernel,
)

_PHYSICAL_UB_CAPACITY_BYTES = 248 * 1024
# SimtVF reserves 32 KiB of the physical UB for thread context.
_SIMT_SHARED_UB_CAPACITY_BYTES = (248 - 32) * 1024
_MOMENTUM_TILE_SIZE = 4096


def _check_npu_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtypes: tuple[torch.dtype, ...],
    device: torch.device | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor.')
    if tensor.device.type != 'npu':
        raise ValueError(f'{name} must be an NPU tensor, got {tensor.device}.')
    if device is not None and tensor.device != device:
        raise ValueError(f'{name} must be on {device}, got {tensor.device}.')
    if tensor.dtype not in dtypes:
        expected = ' or '.join(str(dtype) for dtype in dtypes)
        raise TypeError(f'{name} must use {expected}, got {tensor.dtype}.')
    if not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous.')


def _check_real(name: str, value: float, *, non_negative: bool = False) -> float:
    if not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number.')
    value = float(value)
    if not math.isfinite(value) or (non_negative and value < 0):
        requirement = 'finite and non-negative' if non_negative else 'finite'
        raise ValueError(f'{name} must be {requirement}.')
    return value


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.numel() > 0
        and right.numel() > 0
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _reject_aliases(inputs: dict[str, torch.Tensor]) -> None:
    input_items = list(inputs.items())
    for index, (left_name, left) in enumerate(input_items):
        for right_name, right in input_items[index + 1 :]:
            if _shares_storage(left, right):
                raise ValueError(f'{left_name} and {right_name} must not share storage.')


def _validate_matrix_inputs(
    matrix: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    eps: float,
    *,
    head_dim_alignment: int,
) -> tuple[int, int, float]:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (matrix, row_scale, col_scale)):
        raise TypeError('matrix, row_scale and col_scale must be torch.Tensor instances.')
    eps = _check_real('eps', eps, non_negative=True)

    require_ascend()
    fp32 = (torch.float32,)
    _check_npu_tensor('matrix', matrix, dtypes=fp32)
    _check_npu_tensor('row_scale', row_scale, dtypes=fp32, device=matrix.device)
    _check_npu_tensor('col_scale', col_scale, dtypes=fp32, device=matrix.device)

    if matrix.ndim != 2:
        raise ValueError('matrix must have shape (num_rows, head_dim).')
    num_rows, head_dim = matrix.shape
    if head_dim == 0 or head_dim % head_dim_alignment:
        raise ValueError(f'head_dim must be a positive multiple of {head_dim_alignment}, got {head_dim}.')
    if row_scale.ndim != 1 or row_scale.shape[0] != num_rows:
        raise ValueError(f'row_scale must have shape ({num_rows},), got {tuple(row_scale.shape)}.')
    if col_scale.ndim != 1 or col_scale.shape[0] != head_dim:
        raise ValueError(f'col_scale must have shape ({head_dim},), got {tuple(col_scale.shape)}.')

    _reject_aliases({'matrix': matrix, 'row_scale': row_scale, 'col_scale': col_scale})
    return num_rows, head_dim, eps


def engram_sinkhorn_momentum_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta1: float,
    nesterov: bool = True,
) -> torch.Tensor:
    """Update Engram momentum in place and return the FP32 Sinkhorn input.

    ``grad`` and ``momentum_buffer`` must be contiguous one-dimensional NPU
    tensors with equal shape. Each tensor may use BF16 or FP32, independently,
    and its length must be a multiple of 256. Computation is performed in FP32;
    the updated momentum is converted back to ``momentum_buffer.dtype``.
    """

    if not isinstance(grad, torch.Tensor) or not isinstance(momentum_buffer, torch.Tensor):
        raise TypeError('grad and momentum_buffer must be torch.Tensor instances.')
    beta1 = _check_real('beta1', beta1)
    if not isinstance(nesterov, bool):
        raise TypeError('nesterov must be a bool.')

    require_ascend()
    supported_dtypes = (torch.bfloat16, torch.float32)
    _check_npu_tensor('grad', grad, dtypes=supported_dtypes)
    _check_npu_tensor('momentum_buffer', momentum_buffer, dtypes=supported_dtypes, device=grad.device)
    if grad.ndim != 1 or momentum_buffer.ndim != 1:
        raise ValueError('grad and momentum_buffer must be one-dimensional.')
    if grad.shape != momentum_buffer.shape:
        raise ValueError(
            f'grad and momentum_buffer must have the same shape, got {tuple(grad.shape)} and '
            f'{tuple(momentum_buffer.shape)}.'
        )
    if grad.numel() % 256:
        raise ValueError(f'grad length must be a multiple of 256, got {grad.numel()}.')
    _reject_aliases({'grad': grad, 'momentum_buffer': momentum_buffer})

    sinkhorn_input = torch.empty_like(momentum_buffer, dtype=torch.float32)
    if grad.numel() == 0:
        return sinkhorn_input

    num_tiles = (grad.numel() + _MOMENTUM_TILE_SIZE - 1) // _MOMENTUM_TILE_SIZE
    num_cores = min(get_num_vector_cores(), num_tiles)
    kernel = get_engram_sinkhorn_momentum_update_kernel(
        T.dtype(grad.dtype),
        T.dtype(momentum_buffer.dtype),
        nesterov,
        beta1,
        num_cores,
    )
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(grad, momentum_buffer, sinkhorn_input)
    return sinkhorn_input


def engram_sinkhorn_step(
    matrix: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize one Engram Sinkhorn iterate over rows.

    All inputs must be contiguous FP32 NPU tensors and ``head_dim`` must be a
    positive multiple of 256. The function updates ``row_scale`` in place and
    returns the pre-normalization row norms and normalized column squared sums.
    The caller must provide non-negative Sinkhorn scaling factors in
    ``row_scale``.
    """

    num_rows, head_dim, eps = _validate_matrix_inputs(
        matrix,
        row_scale,
        col_scale,
        eps,
        head_dim_alignment=256,
    )
    if num_rows == 0:
        return torch.empty_like(row_scale), torch.zeros_like(col_scale)

    num_cores = min(get_num_vector_cores(), num_rows)
    row_bytes = head_dim * matrix.element_size()
    if row_bytes >= 16 * 1024:
        chunk_rows, num_stages = 1, 6
    else:
        chunk_rows = max(1, (50 * 1024) // row_bytes)
        num_stages = 3
    rows_per_core = (num_rows + num_cores - 1) // num_cores
    shared_ub_bytes = (
        num_stages * chunk_rows * row_bytes
        + 2 * row_bytes
        + 3 * rows_per_core * row_scale.element_size()
        + (row_bytes if head_dim > 512 else 0)
    )
    if shared_ub_bytes > _SIMT_SHARED_UB_CAPACITY_BYTES:
        raise ValueError(
            'the requested num_rows/head_dim combination exceeds the 216 KiB usable shared UB capacity: '
            f'num_rows={num_rows}, head_dim={head_dim}, required_bytes={shared_ub_bytes}.'
        )
    row_norm = torch.empty_like(row_scale)
    col_sumsq = torch.zeros_like(col_scale)
    kernel = get_engram_sinkhorn_step_kernel(
        head_dim,
        eps,
        num_cores,
        rows_per_core,
        chunk_rows,
        num_stages,
    )
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(matrix, row_scale, col_scale, row_norm, col_sumsq)
    return row_norm, col_sumsq


def engram_sinkhorn_finalize(
    matrix: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    eps: float,
    alpha: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the final row-normalized Engram Sinkhorn matrix.

    The three inputs must be contiguous FP32 NPU tensors and ``head_dim`` must
    be a positive multiple of 64. ``out`` may be supplied as a separate,
    contiguous FP32 output tensor with the same shape and device as ``matrix``.
    """

    num_rows, head_dim, eps = _validate_matrix_inputs(
        matrix,
        row_scale,
        col_scale,
        eps,
        head_dim_alignment=64,
    )
    alpha = _check_real('alpha', alpha)
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError('out must be a torch.Tensor when provided.')
        _check_npu_tensor('out', out, dtypes=(torch.float32,), device=matrix.device)
        if out.shape != matrix.shape:
            raise ValueError(f'out must have shape {tuple(matrix.shape)}, got {tuple(out.shape)}.')
        _reject_aliases({'matrix': matrix, 'row_scale': row_scale, 'col_scale': col_scale, 'out': out})
    else:
        out = torch.empty_like(matrix)

    if num_rows == 0:
        return out

    num_stages = 6 if head_dim <= 512 else 3
    row_bytes = head_dim * matrix.element_size()
    shared_ub_bytes = (2 * num_stages + 1) * row_bytes + num_stages * row_scale.element_size()
    if shared_ub_bytes > _PHYSICAL_UB_CAPACITY_BYTES:
        raise ValueError(
            'the requested head_dim exceeds the 248 KiB shared UB capacity: '
            f'head_dim={head_dim}, required_bytes={shared_ub_bytes}.'
        )
    num_cores = min(get_num_vector_cores(), num_rows)
    kernel = get_engram_sinkhorn_finalize_kernel(head_dim, eps, alpha, num_cores)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(matrix, row_scale, col_scale, out)
    return out
