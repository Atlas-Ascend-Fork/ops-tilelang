"""Public interface for gathering ZOS input token IDs."""

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .gather_input_ids_zos_kernel import get_gather_input_ids_zos_kernel


def gather_input_ids_zos(
    fresh_tokens: torch.Tensor,
    prev_sampled_token_ids: torch.Tensor,
    prev_predict_token_ids: torch.Tensor,
    remap: torch.Tensor,
    seqlens_q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    out_input_ids: torch.Tensor,
    *,
    max_seqlen_q: int,
) -> None:
    """Gather current or previous token IDs into ``out_input_ids`` on Ascend."""
    tensors = (
        fresh_tokens,
        prev_sampled_token_ids,
        prev_predict_token_ids,
        remap,
        seqlens_q,
        cu_seqlens_q,
        out_input_ids,
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('all inputs and outputs must be torch.Tensor instances.')
    if not isinstance(max_seqlen_q, int) or isinstance(max_seqlen_q, bool):
        raise TypeError('max_seqlen_q must be an integer.')
    if max_seqlen_q <= 0 or max_seqlen_q > 128:
        raise ValueError(f'max_seqlen_q must be in [1, 128], got {max_seqlen_q}.')

    require_ascend()
    if any(tensor.device.type != 'npu' for tensor in tensors):
        raise ValueError('all inputs and outputs must be NPU tensors.')
    if any(tensor.device != fresh_tokens.device for tensor in tensors):
        raise ValueError('all inputs and outputs must be on the same device.')
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise TypeError('all inputs and outputs must use torch.int32.')
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError('all inputs and outputs must be contiguous.')

    expected_ranks = (2, 1, 2, 1, 1, 1, 1)
    if any(tensor.ndim != rank for tensor, rank in zip(tensors, expected_ranks, strict=True)):
        raise ValueError('input and output tensor ranks do not match the operator contract.')

    if fresh_tokens.shape[1] != max_seqlen_q:
        raise ValueError(f'fresh_tokens must have shape (n_seq, {max_seqlen_q}).')
    n_seq = fresh_tokens.shape[0]
    prev_n_seq = prev_sampled_token_ids.shape[0]
    nextn = max_seqlen_q - 1
    if prev_predict_token_ids.shape != (prev_n_seq, nextn):
        raise ValueError(f'prev_predict_token_ids must have shape ({prev_n_seq}, {nextn}).')
    if remap.shape != (n_seq,) or seqlens_q.shape != (n_seq,) or cu_seqlens_q.shape != (n_seq,):
        raise ValueError('remap, seqlens_q, and cu_seqlens_q must have shape (n_seq,).')
    kernel = get_gather_input_ids_zos_kernel(max_seqlen_q)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(*tensors)


def gather_input_ids_zos_ref(
    fresh_tokens: torch.Tensor,
    prev_sampled_token_ids: torch.Tensor,
    prev_predict_token_ids: torch.Tensor,
    remap: torch.Tensor,
    seqlens_q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    out_input_ids: torch.Tensor,
    *,
    max_seqlen_q: int,
) -> None:
    """PyTorch reference for ZOS input token gathering."""
    n_seq = fresh_tokens.shape[0]
    token_idx = torch.arange(max_seqlen_q, device=fresh_tokens.device, dtype=torch.int32)
    valid_mask = token_idx.unsqueeze(0) < seqlens_q.unsqueeze(1)

    q_base = torch.zeros(n_seq, device=fresh_tokens.device, dtype=torch.int32)
    if n_seq > 1:
        q_base[1:] = cu_seqlens_q[:-1]
    out_indices = q_base.unsqueeze(1) + token_idx.unsqueeze(0)

    is_old = (remap >= 0).unsqueeze(1)
    remap_clamped = remap.clamp(min=0)
    sampled_values = prev_sampled_token_ids[remap_clamped.long()]
    previous_idx = (token_idx - 1).clamp(min=0)
    predicted_values = prev_predict_token_ids[
        remap_clamped.long().unsqueeze(1).expand(-1, max_seqlen_q),
        previous_idx.unsqueeze(0).long().expand(n_seq, -1),
    ]

    is_first = (token_idx == 0).unsqueeze(0)
    old_values = torch.where(is_first, sampled_values.unsqueeze(1), predicted_values)
    values = torch.where(is_old, old_values, fresh_tokens)

    flat_indices = out_indices.reshape(-1).long()
    flat_valid = valid_mask.reshape(-1)
    out_input_ids.scatter_(0, flat_indices[flat_valid], values.reshape(-1)[flat_valid])
