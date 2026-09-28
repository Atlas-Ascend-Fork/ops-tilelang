import torch

from cann_ops_tilelang.attention.flash_mla_utils_kernel import _convert_kv_index_tl_ascend
from cann_ops_tilelang.config import require_ascend


def convert_kv_index(
    topk_idxs: torch.Tensor, block_size: int, context_lens: torch.Tensor, block_tables: torch.Tensor
) -> None:
    """
    Args:
        topk_idxs: [bsz, seqlen_q, topk_tokens]
        context_lens: [bsz, seqlen_q]
        block_tables: [bsz, max_num_blocks]
    """
    require_ascend()
    if any(tensor.device.type != 'npu' for tensor in (topk_idxs, context_lens, block_tables)):
        raise ValueError('topk_idxs, context_lens, and block_tables must be NPU tensors.')
    if not (topk_idxs.device == context_lens.device == block_tables.device):
        raise ValueError('topk_idxs, context_lens, and block_tables must be on the same device.')
    if any(tensor.dtype != torch.int32 for tensor in (topk_idxs, context_lens, block_tables)):
        raise TypeError('topk_idxs, context_lens, and block_tables must have dtype torch.int32.')
    if block_size <= 0:
        raise ValueError('block_size must be positive.')
    assert topk_idxs.is_contiguous()
    assert context_lens.is_contiguous()
    assert block_tables.is_contiguous()
    assert topk_idxs.ndim == 3
    assert context_lens.ndim == 2
    assert block_tables.ndim == 2
    b, s, k = topk_idxs.shape
    _b, n = block_tables.shape
    assert b == _b
    if context_lens.shape != (b, s):
        raise ValueError(f'context_lens must have shape {(b, s)}, got {tuple(context_lens.shape)}.')
    kernel = _convert_kv_index_tl_ascend(s, k, n, block_size)
    kernel(topk_idxs, context_lens, block_tables)
