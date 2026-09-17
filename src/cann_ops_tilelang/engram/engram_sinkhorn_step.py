"""Public interface for one Engram Sinkhorn normalization step."""

from __future__ import annotations

import math
from numbers import Real

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend, should_print_kernel_source

from .engram_sinkhorn_step_kernel import get_engram_sinkhorn_step_kernel

# SimtVF reserves 32 KiB of the physical 248 KiB UB for thread context.
_SHARED_UB_CAPACITY_BYTES = (248 - 32) * 1024


def _check_npu_fp32_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
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
    if not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous.')


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.numel() > 0
        and right.numel() > 0
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _validate_inputs(
    matrix: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    eps: float,
) -> tuple[int, int]:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (matrix, row_scale, col_scale)):
        raise TypeError('matrix, row_scale and col_scale must be torch.Tensor instances.')
    if not isinstance(eps, Real):
        raise TypeError('eps must be a real number.')
    if not math.isfinite(float(eps)) or eps < 0:
        raise ValueError('eps must be finite and non-negative.')

    require_ascend()
    _check_npu_fp32_tensor('matrix', matrix)
    _check_npu_fp32_tensor('row_scale', row_scale, device=matrix.device)
    _check_npu_fp32_tensor('col_scale', col_scale, device=matrix.device)

    if matrix.ndim != 2:
        raise ValueError('matrix must have shape (num_rows, head_dim).')
    num_rows, head_dim = matrix.shape
    if head_dim == 0 or head_dim % 256:
        raise ValueError(f'head_dim must be a positive multiple of 256, got {head_dim}.')
    if row_scale.ndim != 1 or row_scale.shape[0] != num_rows:
        raise ValueError(f'row_scale must have shape ({num_rows},), got {tuple(row_scale.shape)}.')
    if col_scale.ndim != 1 or col_scale.shape[0] != head_dim:
        raise ValueError(f'col_scale must have shape ({head_dim},), got {tuple(col_scale.shape)}.')

    inputs = {'matrix': matrix, 'row_scale': row_scale, 'col_scale': col_scale}
    input_items = list(inputs.items())
    for index, (left_name, left) in enumerate(input_items):
        for right_name, right in input_items[index + 1 :]:
            if _shares_storage(left, right):
                raise ValueError(f'{left_name} and {right_name} must not share storage.')
    return num_rows, head_dim


def engram_sinkhorn_step(
    matrix: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize one Engram Sinkhorn iterate over rows.

    ``matrix`` has shape ``(num_rows, head_dim)`` and both scale tensors are
    one-dimensional. All tensors must be contiguous FP32 NPU tensors;
    ``head_dim`` must be a positive multiple of 256 and ``row_scale`` must
    contain non-negative Sinkhorn scaling factors. The function updates
    ``row_scale`` in place and returns the pre-normalization ``row_norm`` and
    normalized per-column squared sums ``col_sumsq``.

    The three inputs must not share storage. ``col_sumsq`` is initialized to
    zero before the cross-core atomic reduction.
    """

    num_rows, head_dim = _validate_inputs(matrix, row_scale, col_scale, eps)
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
    if shared_ub_bytes > _SHARED_UB_CAPACITY_BYTES:
        raise ValueError(
            'the requested num_rows/head_dim combination exceeds the 216 KiB usable shared UB capacity: '
            f'num_rows={num_rows}, head_dim={head_dim}, required_bytes={shared_ub_bytes}.'
        )
    row_norm = torch.empty_like(row_scale)
    col_sumsq = torch.zeros_like(col_scale)
    kernel = get_engram_sinkhorn_step_kernel(
        head_dim,
        float(eps),
        num_cores,
        rows_per_core,
        chunk_rows,
        num_stages,
    )
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(matrix, row_scale, col_scale, row_norm, col_sumsq)
    return row_norm, col_sumsq
