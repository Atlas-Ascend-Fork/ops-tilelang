"""Public Ascend interface for mirror context-parallel metadata."""

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend

from .prepare_cp_mirror_kernel import _precompute_tl_ascend, _scatter_tl_ascend


def _validate_mirror_inputs(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    compress_ratio: int,
    total_q: int,
    output_len: int,
) -> int:
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

    assert cu_seqlens_q.device.type == 'npu', 'input tensors must be on an NPU device'
    batch_size = cu_seqlens_q.numel() - 1
    assert batch_size >= 0, 'cu_seqlens_q must contain at least one element'
    assert cu_seqlens_k.numel() == batch_size + 1

    return batch_size


def _precompute_mirror(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    compress_ratio: int,
    num_cores: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = cu_seqlens_q.numel() - 1
    device = cu_seqlens_q.device
    k_c4_lens = torch.empty(batch_size, dtype=torch.int32, device=device)
    partial_seqlens = torch.empty(batch_size, dtype=torch.int32, device=device)
    elems_per_batch = torch.empty(batch_size, dtype=torch.int32, device=device)
    delta_sq = torch.empty(batch_size, 4, dtype=torch.int32, device=device)
    delta_sk = torch.empty(batch_size, 4, dtype=torch.int32, device=device)

    kernel = _precompute_tl_ascend(cp_size=cp_size, cp_rank=cp_rank, compress_ratio=compress_ratio, num_cores=num_cores)
    kernel(cu_seqlens_q, cu_seqlens_k, k_c4_lens, partial_seqlens, elems_per_batch, delta_sq, delta_sk)
    return k_c4_lens, partial_seqlens, elems_per_batch, delta_sq, delta_sk


def _prefix_sum_on_device(values: torch.Tensor, length: int) -> torch.Tensor:
    result = torch.zeros(length + 1, dtype=torch.int32, device=values.device)
    result[1:] = torch.cumsum(values, dim=0)
    return result


def _empty_mirror_outputs(total_q: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    empty = torch.empty(0, dtype=torch.int32, device=device)
    z1 = torch.zeros(1, dtype=torch.int32, device=device)
    outputs = (
        empty,
        torch.zeros(total_q, dtype=torch.int32, device=device),
        empty.clone(),
        empty.clone(),
        z1,
        z1.clone(),
    )
    return outputs


def _allocate_scatter_outputs(
    output_len: int, total_q: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(total_q, dtype=torch.int32, device=device),
        torch.empty(output_len, dtype=torch.int32, device=device),
        torch.empty(output_len, dtype=torch.int32, device=device),
        torch.empty(output_len, dtype=torch.int32, device=device),
    )


def _prepare_for_mirror_cp_impl(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    compress_ratio: int,
    total_q: int,
    output_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = _validate_mirror_inputs(
        cu_seqlens_q, cu_seqlens_k, cp_size, cp_rank, compress_ratio, total_q, output_len
    )
    require_ascend()
    device = cu_seqlens_q.device

    if batch_size == 0 or output_len == 0 or total_q == 0:
        return _empty_mirror_outputs(total_q, device)

    num_cores = get_num_vector_cores()
    # ---- Phase 1: precompute per-batch quantities (NPU kernel, no sync) ---- #
    k_c4_lens, partial_seqlens, elems_per_batch, delta_sq, delta_sk = _precompute_mirror(
        cu_seqlens_q, cu_seqlens_k, cp_size, cp_rank, compress_ratio, num_cores
    )

    # ---- Phase 2: cumulative sums (NPU ops, no sync) ---- #
    cu_seqlens_k_c4 = _prefix_sum_on_device(k_c4_lens, batch_size)
    cu_elems = _prefix_sum_on_device(elems_per_batch, batch_size)

    total_per_rank = cu_elems[batch_size : batch_size + 1].contiguous()  # [1] tensor on NPU

    # ---- Phase 3: main scatter kernel (NPU kernel, no sync) ---- #
    mirror_cp_reverse_index, mirror_cp_index, ks_out, ke_out = _allocate_scatter_outputs(output_len, total_q, device)

    scatter_kern = _scatter_tl_ascend(
        cp_size=cp_size, cp_rank=cp_rank, compress_ratio=compress_ratio, num_cores=num_cores
    )
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
    cu_seqlens_q_cp = _prefix_sum_on_device(delta_sq.reshape(-1), batch_size * 4)
    cu_seqlens_k_cp = _prefix_sum_on_device(delta_sk.reshape(-1), batch_size * 4)

    return (
        mirror_cp_index,
        mirror_cp_reverse_index,
        ks_out,
        ke_out,
        cu_seqlens_q_cp,
        cu_seqlens_k_cp,
    )


@torch.library.custom_op('cann_ops_tilelang::prepare_for_mirror_cp', mutates_args=())
def prepare_for_mirror_cp(
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
    return _prepare_for_mirror_cp_impl(
        cu_seqlens_q,
        cu_seqlens_k,
        cp_size,
        cp_rank,
        compress_ratio,
        total_q,
        output_len,
    )


@prepare_for_mirror_cp.register_fake
def _prepare_for_mirror_cp_fake(cu_seqlens_q, cu_seqlens_k, cp_size, cp_rank, compress_ratio, total_q, output_len):
    batch_size = _validate_mirror_inputs(
        cu_seqlens_q, cu_seqlens_k, cp_size, cp_rank, compress_ratio, total_q, output_len
    )
    if batch_size == 0 or output_len == 0 or total_q == 0:
        lengths = (0, total_q, 0, 0, 1, 1)
    else:
        lengths = (output_len, total_q, output_len, output_len, batch_size * 4 + 1, batch_size * 4 + 1)
    return tuple(torch.empty(length, dtype=torch.int32, device=cu_seqlens_q.device) for length in lengths)
