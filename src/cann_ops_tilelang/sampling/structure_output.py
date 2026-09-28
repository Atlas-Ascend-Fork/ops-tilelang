"""Public Ascend interface for applying packed token masks."""

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .structure_output_kernel import BITS_PER_BLOCK, _apply_token_bitmask_inplace_tl_ascend


def apply_token_bitmask_inplace(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Mask disallowed token logits in selected rows in place."""
    for name, tensor in (('logits', logits), ('bitmask', bitmask), ('indices', indices)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{name} must be a torch.Tensor.')
    require_ascend()
    if logits.device.type != 'npu' or bitmask.device != logits.device or indices.device != logits.device:
        raise ValueError('logits, bitmask, and indices must be on the same NPU device.')
    if logits.ndim != 2 or bitmask.ndim != 2 or indices.ndim != 1:
        raise ValueError('logits and bitmask must be 2D, and indices must be 1D.')
    if logits.dtype != torch.float32:
        raise TypeError('logits must use torch.float32.')
    if bitmask.dtype != torch.int32 or indices.dtype != torch.int32:
        raise TypeError('bitmask and indices must use torch.int32.')
    if logits.shape[0] != bitmask.shape[0]:
        raise ValueError('logits and bitmask must have the same number of rows.')
    if logits.stride(1) != 1 or bitmask.stride(1) != 1:
        raise ValueError('logits and bitmask must have unit inner stride.')
    if logits.stride(0) < logits.shape[1] or bitmask.stride(0) < bitmask.shape[1]:
        raise ValueError('logits and bitmask rows must not overlap.')
    if not indices.is_contiguous():
        raise ValueError('indices must be contiguous.')
    if (logits.shape[1] + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK < bitmask.shape[1]:
        raise ValueError('logits vocabulary size must cover the packed bitmask.')

    vocab_size = min(logits.shape[1], bitmask.shape[1] * BITS_PER_BLOCK)
    if vocab_size == 0 or indices.numel() == 0:
        return
    use_single_row_direct = indices.numel() == 1
    use_large_tile = indices.numel() >= 8
    use_l_tile = indices.numel() >= 512
    max_block_size = 10816 if use_l_tile else (8192 if use_large_tile else 4096)
    num_stages = 1 if use_large_tile or use_single_row_direct else 2
    kernel = _apply_token_bitmask_inplace_tl_ascend(
        logits.shape[1],
        bitmask.shape[1],
        max_block_size=max_block_size,
        num_stages=num_stages,
    )
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(logits, bitmask, indices)
