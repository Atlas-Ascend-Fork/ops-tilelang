"""Probability prefix reduction for Ascend sampling."""

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend, should_print_kernel_source

from .sum_smaller_probs_kernel import _sum_smaller_probs_tl_ascend, _sum_smaller_probs_ub_bytes


def get_sum_smaller_probs(probs: torch.Tensor, sample_results: torch.Tensor) -> torch.Tensor:
    """Sum probabilities smaller than each sampled value with a stable tie break."""
    require_ascend()
    if not isinstance(probs, torch.Tensor) or not isinstance(sample_results, torch.Tensor):
        raise TypeError('probs and sample_results must be torch.Tensor instances.')
    if probs.device.type != 'npu' or sample_results.device != probs.device:
        raise ValueError('probs and sample_results must be on the same NPU device.')
    if probs.dtype != torch.float32:
        raise TypeError(f'probs must have dtype torch.float32, got {probs.dtype}.')
    if sample_results.dtype != torch.int32:
        raise TypeError(f'sample_results must have dtype torch.int32, got {sample_results.dtype}.')
    if probs.ndim != 2 or probs.stride(1) != 1:
        raise ValueError('probs must be two-dimensional and contiguous in its last dimension.')
    if sample_results.shape != (probs.shape[0],) or not sample_results.is_contiguous():
        raise ValueError('sample_results must be contiguous with shape (num_tokens,).')
    if probs.shape[1] <= 0:
        raise ValueError('vocab_size must be greater than zero.')

    output = torch.empty((probs.shape[0],), dtype=probs.dtype, device=probs.device)
    if probs.shape[0] == 0:
        return output
    num_cores = min(probs.shape[0], get_num_vector_cores())
    pipeline_stages = 2 if probs.shape[0] >= 2 * num_cores else 1
    max_rows_per_core = (probs.shape[0] + num_cores - 1) // num_cores
    ub_bytes = _sum_smaller_probs_ub_bytes(probs.shape[1], pipeline_stages, max_rows_per_core)
    if ub_bytes > 240000:
        raise ValueError(
            f'get_sum_smaller_probs requires {ub_bytes} bytes of UB per core, exceeding the 240000-byte limit.'
        )
    kernel = _sum_smaller_probs_tl_ascend(probs.shape[1], num_cores, pipeline_stages, max_rows_per_core)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(probs, sample_results, output)
    return output
