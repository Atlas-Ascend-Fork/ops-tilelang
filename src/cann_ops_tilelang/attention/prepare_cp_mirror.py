"""Public Ascend interface for mirror context-parallel metadata."""

import torch

from cann_ops_tilelang.config import require_ascend

from .prepare_cp_mirror_kernel import _precompute_tl_ascend, _scatter_tl_ascend


def prepare_for_mirror_cp_tilelang(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    compress_ratio: int,
    total_q: int,
    output_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TileLang Ascend implementation of prepare_for_mirror_cp.
    This function is sync-free: it does not call .item() or perform a
    device-to-host synchronization. The caller must provide pre-computed sizes.

    Args:
        cu_seqlens_q: [batch_size + 1], contiguous int32 device tensor
        cu_seqlens_k: [batch_size + 1], contiguous int32 device tensor
        cp_size: context parallelism size
        cp_rank: current rank
        compress_ratio: KV compression ratio
        total_q: total number of query tokens (= cu_seqlens_q[-1], pre-synced to CPU)
        output_len: total output length for cp_rank (= sum of 2*partial_seqlen, pre-synced to CPU)

    Returns:
        mirror_cp_index_tensor,
        mirror_cp_reverse_index_tensor,
        ks_tensor,
        ke_tensor,
        cu_seqlens_q_cp_tensor,
        cu_seqlens_k_cp_tensor
    """
    assert cu_seqlens_q.ndim == 1, 'cu_seqlens_q must be a 1D tensor'
    assert cu_seqlens_k.ndim == 1, 'cu_seqlens_k must be a 1D tensor'
    assert cu_seqlens_q.dtype == torch.int32, 'cu_seqlens_q must use int32'
    assert cu_seqlens_k.dtype == torch.int32, 'cu_seqlens_k must use int32'
    assert cu_seqlens_q.is_contiguous(), 'cu_seqlens_q must be contiguous'
    assert cu_seqlens_k.is_contiguous(), 'cu_seqlens_k must be contiguous'
    assert cu_seqlens_q.device == cu_seqlens_k.device, 'both inputs must be on the same device'
    assert cp_size > 0, 'cp_size must be positive'
    assert 0 <= cp_rank < cp_size, 'cp_rank must be in [0, cp_size)'
    assert compress_ratio > 0, 'compress_ratio must be positive'
    assert total_q >= 0, 'total_q must be non-negative'
    assert output_len >= 0, 'output_len must be non-negative'

    require_ascend()
    assert cu_seqlens_q.device.type == 'npu', 'input tensors must be on an NPU device'
    device = cu_seqlens_q.device
    batch_size = cu_seqlens_q.numel() - 1
    assert batch_size >= 0, 'cu_seqlens_q must contain at least one element'
    assert cu_seqlens_k.numel() == batch_size + 1

    if batch_size == 0 or output_len == 0 or total_q == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        z1 = torch.zeros(1, dtype=torch.int32, device=device)
        return (
            empty,
            torch.zeros(total_q, dtype=torch.int32, device=device),
            empty,
            empty,
            z1,
            z1,
        )

    # ---- Phase 1: precompute per-batch quantities (NPU kernel, no sync) ---- #
    k_c4_lens = torch.empty(batch_size, dtype=torch.int32, device=device)
    partial_seqlens = torch.empty(batch_size, dtype=torch.int32, device=device)
    elems_per_batch = torch.empty(batch_size, dtype=torch.int32, device=device)
    delta_sq = torch.empty(batch_size, 4, dtype=torch.int32, device=device)
    delta_sk = torch.empty(batch_size, 4, dtype=torch.int32, device=device)

    precompute_kern = _precompute_tl_ascend(cp_size=cp_size, cp_rank=cp_rank, compress_ratio=compress_ratio)
    precompute_kern(
        cu_seqlens_q,
        cu_seqlens_k,
        k_c4_lens,
        partial_seqlens,
        elems_per_batch,
        delta_sq,
        delta_sk,
    )

    # ---- Phase 2: cumulative sums (NPU ops, no sync) ---- #
    cu_seqlens_k_c4 = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k_c4[1:] = torch.cumsum(k_c4_lens, dim=0)

    cu_elems = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_elems[1:] = torch.cumsum(elems_per_batch, dim=0)

    total_per_rank = cu_elems[batch_size : batch_size + 1].contiguous()  # [1] tensor on NPU

    # ---- Phase 3: main scatter kernel (NPU kernel, no sync) ---- #
    mirror_cp_reverse_index = torch.zeros(total_q, dtype=torch.int32, device=device)
    mirror_cp_index = torch.empty(output_len, dtype=torch.int32, device=device)
    ks_out = torch.empty(output_len, dtype=torch.int32, device=device)
    ke_out = torch.empty(output_len, dtype=torch.int32, device=device)

    scatter_kern = _scatter_tl_ascend(cp_size=cp_size, cp_rank=cp_rank, compress_ratio=compress_ratio)
    scatter_kern(
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_c4,
        partial_seqlens,
        cu_elems,
        total_per_rank,
        mirror_cp_reverse_index,
        mirror_cp_index,
        ks_out,
        ke_out,
    )

    # ---- Phase 4: cu_seqlens_q_cp / cu_seqlens_k_cp (NPU ops, no sync) ---- #
    delta_sq_flat = delta_sq.reshape(-1)
    delta_sk_flat = delta_sk.reshape(-1)

    cu_seqlens_q_cp = torch.zeros(batch_size * 4 + 1, dtype=torch.int32, device=device)
    cu_seqlens_k_cp = torch.zeros(batch_size * 4 + 1, dtype=torch.int32, device=device)
    cu_seqlens_q_cp[1:] = torch.cumsum(delta_sq_flat, dim=0)
    cu_seqlens_k_cp[1:] = torch.cumsum(delta_sk_flat, dim=0)

    return (
        mirror_cp_index,
        mirror_cp_reverse_index,
        ks_out,
        ke_out,
        cu_seqlens_q_cp,
        cu_seqlens_k_cp,
    )
