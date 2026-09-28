"""MTP variable-length scheduling for Ascend."""

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .schedule_mtp_varlen_kernel import (
    _compute_seqlens_q_by_confidence_rates_tl_ascend,
    _compute_seqlens_q_by_confidence_rates_tl_ascend_fallback,
)


def compute_seqlens_q_by_confidence_rates(
    total_q: int,
    remap: torch.Tensor,
    prev_confidence_rates: torch.Tensor,
    seqlens_q: torch.Tensor,
) -> None:
    """Distribute a query-token budget using MTP confidence rates."""
    require_ascend()
    if not all(isinstance(value, torch.Tensor) for value in (remap, prev_confidence_rates, seqlens_q)):
        raise TypeError('remap, prev_confidence_rates and seqlens_q must be torch.Tensor instances.')
    if isinstance(total_q, bool) or not isinstance(total_q, int):
        raise TypeError('total_q must be an integer.')
    if remap.device.type != 'npu' or prev_confidence_rates.device != remap.device or seqlens_q.device != remap.device:
        raise ValueError('all tensors must be on the same NPU device.')
    if remap.dtype != torch.int32 or seqlens_q.dtype != torch.int32:
        raise TypeError('remap and seqlens_q must have dtype torch.int32.')
    if prev_confidence_rates.dtype != torch.float32:
        raise TypeError('prev_confidence_rates must have dtype torch.float32.')
    if remap.ndim != 1 or prev_confidence_rates.ndim != 2 or seqlens_q.shape != remap.shape:
        raise ValueError(
            'expected remap [num_seqs], prev_confidence_rates [n_prev_seq, nextn], and matching seqlens_q.'
        )
    if not remap.is_contiguous() or not prev_confidence_rates.is_contiguous() or not seqlens_q.is_contiguous():
        raise ValueError('all tensors must be contiguous.')

    num_seqs = remap.shape[0]
    nextn = prev_confidence_rates.shape[1]
    if nextn <= 0:
        raise ValueError('nextn must be greater than zero.')
    if total_q < num_seqs or total_q > num_seqs * (nextn + 1):
        raise ValueError('total_q must be between num_seqs and num_seqs * (nextn + 1).')
    if num_seqs == 0:
        return

    total_rates = num_seqs * nextn
    block_size = next(
        (
            candidate
            for candidate in (448, 512, 896, 1024, 1792, 2048, 3584, 4096, 7168, 8192)
            if total_rates <= candidate
        ),
        None,
    )
    if block_size is None:
        if nextn > 19792:
            raise ValueError('nextn must be at most 19792 for the Ascend fallback kernel.')
        cum_rates = torch.empty((total_rates,), dtype=torch.float32, device=remap.device)
        kernel = _compute_seqlens_q_by_confidence_rates_tl_ascend_fallback(nextn)
        if should_print_kernel_source():
            print(kernel.get_kernel_source())
        kernel(remap, prev_confidence_rates, cum_rates, seqlens_q, total_q)
        return

    kernel = _compute_seqlens_q_by_confidence_rates_tl_ascend(nextn=nextn, block_size=block_size)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(remap, prev_confidence_rates, seqlens_q, total_q)
