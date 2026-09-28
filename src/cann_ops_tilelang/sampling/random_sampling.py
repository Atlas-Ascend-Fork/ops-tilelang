"""Random sampling for Ascend."""

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .random_sampling_kernel import _random_sampling_tl_ascend


def random_sampling(
    probs: torch.Tensor,
    seq_seeds: torch.Tensor,
    sample_positions: torch.Tensor,
    is_nextn_predict: bool = False,
) -> torch.Tensor:
    """Sample one token from each probability row."""
    require_ascend()
    if not isinstance(probs, torch.Tensor):
        raise TypeError('probs must be a torch.Tensor.')
    if not isinstance(seq_seeds, torch.Tensor) or not isinstance(sample_positions, torch.Tensor):
        raise TypeError('seq_seeds and sample_positions must be torch.Tensor instances.')
    if probs.device.type != 'npu':
        raise ValueError(f'probs must be on an NPU, got {probs.device}.')
    if seq_seeds.device != probs.device or sample_positions.device != probs.device:
        raise ValueError('all inputs must be on the same NPU device.')
    if probs.dtype != torch.float32:
        raise TypeError(f'probs must have dtype torch.float32, got {probs.dtype}.')
    if seq_seeds.dtype != torch.uint64:
        raise TypeError(f'seq_seeds must have dtype torch.uint64, got {seq_seeds.dtype}.')
    if sample_positions.dtype != torch.int32:
        raise TypeError(f'sample_positions must have dtype torch.int32, got {sample_positions.dtype}.')
    if probs.ndim != 2 or probs.stride(1) != 1:
        raise ValueError('probs must be a two-dimensional tensor contiguous in its last dimension.')

    num_tokens, vocab_size = probs.shape
    if vocab_size <= 0:
        raise ValueError('vocab_size must be greater than zero.')
    if seq_seeds.shape != (num_tokens,) or sample_positions.shape != (num_tokens,):
        raise ValueError('seq_seeds and sample_positions must have shape (num_tokens,).')
    if not seq_seeds.is_contiguous() or not sample_positions.is_contiguous():
        raise ValueError('seq_seeds and sample_positions must be contiguous.')

    out = torch.empty((num_tokens,), dtype=torch.int32, device=probs.device)
    if num_tokens == 0:
        return out

    kernel = _random_sampling_tl_ascend(vocab_size, is_nextn_predict)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(probs, seq_seeds, sample_positions, out)
    return out
