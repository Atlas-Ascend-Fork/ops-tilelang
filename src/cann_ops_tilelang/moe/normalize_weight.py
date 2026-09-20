"""Public interface for MoE routing-weight normalization."""

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .normalize_weight_kernel import get_normalize_weight_kernel


def normalize_weight(topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each token's FP32 top-k weights on Ascend.

    ``topk_weights`` must be a contiguous two-dimensional NPU tensor with a
    positive top-k dimension. The returned denominator includes the kernel's
    ``1e-20`` numerical-stability term.
    """

    if not isinstance(topk_weights, torch.Tensor):
        raise TypeError('topk_weights must be a torch.Tensor.')
    require_ascend()
    if topk_weights.device.type != 'npu':
        raise ValueError(f'topk_weights must be an NPU tensor, got {topk_weights.device}.')
    if topk_weights.dtype != torch.float32:
        raise TypeError(f'topk_weights must use torch.float32, got {topk_weights.dtype}.')
    if topk_weights.ndim != 2:
        raise ValueError('topk_weights must have shape (num_tokens, num_topk).')
    if not topk_weights.is_contiguous():
        raise ValueError('topk_weights must be contiguous.')

    num_tokens, num_topk = topk_weights.shape
    if num_topk <= 0:
        raise ValueError(f'num_topk must be positive, got {num_topk}.')

    denominator = torch.empty((num_tokens,), dtype=torch.float32, device=topk_weights.device)
    normalized_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32, device=topk_weights.device)
    if num_tokens == 0:
        return denominator, normalized_weights

    kernel = get_normalize_weight_kernel(num_topk=num_topk)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(topk_weights, denominator, normalized_weights)
    return denominator, normalized_weights
