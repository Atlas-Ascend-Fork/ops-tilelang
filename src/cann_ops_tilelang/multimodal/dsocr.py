"""Public Ascend interface for relative-position attention."""

from __future__ import annotations

import math

import torch

from cann_ops_tilelang.config import get_num_ai_cores, require_ascend

from .dsocr_kernel import _attn_rel_h_rel_w_tl_ascend


def _validate_attn_rel_h_rel_w_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_h: torch.Tensor,
    rel_w: torch.Tensor,
) -> tuple[int, int, int, int]:
    tensors = (q, k, v, rel_h, rel_w)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('q, k, v, rel_h and rel_w must be torch.Tensor instances.')
    require_ascend()
    if any(tensor.device.type != 'npu' for tensor in tensors):
        raise ValueError('all inputs must be NPU tensors.')
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError('all inputs must be on the same NPU device.')
    if q.ndim != 4:
        raise ValueError(f'q must have shape (batch, sequence, heads, dim), got {tuple(q.shape)}.')

    batch, sequence, heads, dim = q.shape
    side = math.isqrt(sequence)
    width_aligned = ((side + 7) // 8) * 8
    if side * side != sequence:
        raise ValueError('the sequence length must be a perfect square.')
    if dim != 64:
        raise ValueError(f'the head dimension must be 64, got {dim}.')
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError('k and v must have the same shape as q.')
    expected_rel_shape = (batch, sequence, heads, width_aligned)
    if rel_h.shape != expected_rel_shape or rel_w.shape != expected_rel_shape:
        raise ValueError(f'rel_h and rel_w must have shape {expected_rel_shape}.')
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError('all inputs must use torch.bfloat16.')
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError('all inputs must have unit stride in their last dimension.')
    return batch, sequence, heads, dim


def attn_rel_h_rel_w(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_h: torch.Tensor,
    rel_w: torch.Tensor,
) -> torch.Tensor:
    """Compute relative-height/width attention on an Ascend NPU."""

    batch, sequence, heads, dim = _validate_attn_rel_h_rel_w_inputs(q, k, v, rel_h, rel_w)
    query_blocks = (sequence + 127) // 128
    num_tasks = batch * heads * query_blocks
    kernel = _attn_rel_h_rel_w_tl_ascend(
        heads=heads,
        seq_len=sequence,
        dim=dim,
        num_tasks=num_tasks,
        num_cores=min(get_num_ai_cores(), num_tasks),
    )
    return kernel(q, k, v, rel_h, rel_w)
