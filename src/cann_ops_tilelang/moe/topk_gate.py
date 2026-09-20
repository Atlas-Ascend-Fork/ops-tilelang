"""Public interface for stable MoE top-k expert selection."""

import os

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend, should_print_kernel_source
from cann_ops_tilelang.utils import align_up

from .topk_gate_kernel import get_topk_gate_kernel

_TOPK_MIN_CORES_ENV = 'OPS_TILELANG_TOPK_MIN_CORES'

# Cache compiled kernels to bypass JIT decorator overhead on every call.
# Key: (num_experts, num_topk, num_cores). Value: compiled kernel callable.
_kernel_cache: dict = {}


def _get_topk_min_cores(num_vector_cores: int) -> int:
    default = max(1, num_vector_cores // 2)
    raw = os.getenv(_TOPK_MIN_CORES_ENV)
    if raw is None:
        return default
    try:
        return max(1, min(int(raw), num_vector_cores))
    except ValueError:
        return default


def topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Select stable top-k expert indices for each token on Ascend.

    ``scores`` must be a contiguous FP32 NPU tensor with shape
    ``(num_tokens, num_experts)``. Equal scores prefer the lower expert index.
    """

    if not isinstance(scores, torch.Tensor):
        raise TypeError('scores must be a torch.Tensor.')
    if not isinstance(num_topk, int) or isinstance(num_topk, bool):
        raise TypeError(f'num_topk must be an int, got {type(num_topk).__name__}.')
    require_ascend()
    if scores.device.type != 'npu':
        raise ValueError(f'scores must be an NPU tensor, got {scores.device}.')
    if scores.dtype != torch.float32:
        raise TypeError(f'scores must use torch.float32, got {scores.dtype}.')
    if scores.ndim != 2:
        raise ValueError('scores must have shape (num_tokens, num_experts).')
    if not scores.is_contiguous():
        raise ValueError('scores must be contiguous.')

    num_tokens, num_experts = scores.shape
    if num_topk <= 0 or num_topk > num_experts:
        raise ValueError(f'num_topk must be in [1, {num_experts}], got {num_topk}.')

    # The padded backing allocation lets the kernel issue whole-tile stores.
    topk_idx = torch.empty(
        (align_up(num_tokens, 8), num_topk),
        dtype=torch.int64,
        device=scores.device,
    )[:num_tokens]
    if num_tokens == 0:
        return topk_idx

    num_vector_cores = get_num_vector_cores()
    num_cores = min(num_vector_cores, num_tokens)
    min_cores = _get_topk_min_cores(num_vector_cores)
    if 1 < num_cores < min_cores:
        num_cores = min_cores

    cache_key = (num_experts, num_topk, num_cores)
    kernel = _kernel_cache.get(cache_key)
    if kernel is None:
        kernel = get_topk_gate_kernel(num_experts, num_topk, num_cores)
        _kernel_cache[cache_key] = kernel

    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(scores, topk_idx)
    return topk_idx
