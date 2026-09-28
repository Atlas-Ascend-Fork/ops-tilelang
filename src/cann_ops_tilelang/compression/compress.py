"""Public Ascend interfaces for state-compression operators."""

from __future__ import annotations

import math

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .compress_kernel import (
    _compress_and_update_state_decode_tl_ascend,
    _compress_and_update_state_prefill_tl_ascend,
)

_UB_BUDGET_BYTES = 200 * 1024


def _validate_kernel_capacity(
    dim: int,
    compress_ratio: int,
    overlap_ratio: int,
    *,
    has_ape: bool,
    is_prefill: bool,
) -> None:
    if dim <= 0 or compress_ratio <= 0 or overlap_ratio <= 0:
        raise ValueError('dim, compress_ratio, and overlap_ratio must be positive.')

    compress_tokens = compress_ratio * overlap_ratio
    dim_block_base = (128 if compress_tokens <= 64 else 64) if is_prefill else 128
    dim_block = math.gcd(dim_block_base, dim)
    compute_buffers = (3 if has_ape else 2) if is_prefill else 2
    compute_stage_bytes = compute_buffers * compress_tokens * dim_block * 4 + dim_block * 2
    store_versions = 3 if is_prefill else 6
    store_buffers = 3 if has_ape else 2
    store_bytes = store_versions * store_buffers * overlap_ratio * dim * 4
    if not is_prefill:
        store_bytes += dim * 2

    if compute_stage_bytes > _UB_BUDGET_BYTES:
        raise ValueError(
            f'compression compute requires {compute_stage_bytes} bytes of UB per stage, '
            f'exceeding the {_UB_BUDGET_BYTES}-byte limit.'
        )
    if store_bytes > _UB_BUDGET_BYTES:
        raise ValueError(
            f'compression state update requires {store_bytes} bytes of UB, exceeding the {_UB_BUDGET_BYTES}-byte limit.'
        )


def _validate_common_inputs(
    score: torch.Tensor,
    latent: torch.Tensor,
    ape: torch.Tensor | None,
    positions: torch.Tensor | None,
    state_cache: torch.Tensor,
    state_block_idx: torch.Tensor,
) -> None:
    tensors = (score, latent, state_cache, state_block_idx)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('score, latent, state_cache and state_block_idx must be tensors.')
    if score.device.type != 'npu' or any(tensor.device != score.device for tensor in tensors[1:]):
        raise ValueError('all required inputs must be on the same NPU device.')
    if score.dtype != torch.float32 or latent.dtype != torch.float32 or state_cache.dtype != torch.float32:
        raise TypeError('score, latent and state_cache must use torch.float32.')
    if state_block_idx.dtype != torch.int32:
        raise TypeError('state_block_idx must use torch.int32.')
    if score.shape != latent.shape or score.ndim != 3:
        raise ValueError('score and latent must have the same three-dimensional shape.')
    for name, tensor in (('ape', ape), ('positions', positions)):
        if tensor is not None and tensor.device != score.device:
            raise ValueError(f'{name} must be on {score.device}.')


def compress_and_update_state_prefill(
    score: torch.Tensor,
    latent: torch.Tensor,
    ape: torch.Tensor | None,
    positions: torch.Tensor | None,
    state_cache: torch.Tensor,
    state_block_idx: torch.Tensor,
    compressed_batch_mapping: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_compressed_seqlens_q: torch.Tensor,
    mirror_cp_reverse_index: torch.Tensor | None = None,
    compress_ratio: int | None = None,
    *,
    bitwise_consistent: bool = False,
) -> torch.Tensor:
    """
    Args:
        score: [total_q | cp_size * cp_mirror_len, overlap_ratio, dim], fp32
        latent: [total_q | cp_size * cp_mirror_len, overlap_ratio, dim], fp32
        ape: [compress_ratio, overlap_ratio, dim], fp32
        positions: [total_q], int32
        state_cache: [num_state_slots, state_cache_size, 2, overlap_ratio, dim], fp32
        state_block_idx: [num_seqs], int32
        compressed_batch_mapping: [total_num_compressed_tokens], int32
        cu_seqlens_q: [num_seqs + 1], int32
        cu_seqlens_k: [num_seqs + 1], int32
        cu_compressed_seqlens_q: [num_seqs + 1], int32. Each query must be >=1,
            or state_cache will not be updated.
        mirror_cp_reverse_index: [total_q], int32, in [0, cp_size * cp_mirror_len)

    Returns:
        kv_compressed: [total_num_compressed_tokens, dim], bf16
    """
    require_ascend()
    _validate_common_inputs(score, latent, ape, positions, state_cache, state_block_idx)
    _, overlap_ratio, dim = score.shape
    if ape is not None:
        has_ape = True
        compress_ratio = ape.shape[0]
    else:
        has_ape = False
        assert compress_ratio is not None
    _validate_kernel_capacity(
        dim,
        compress_ratio,
        overlap_ratio,
        has_ape=has_ape,
        is_prefill=True,
    )
    total_num_compressed_tokens = compressed_batch_mapping.shape[0]
    kv_compressed = torch.empty(total_num_compressed_tokens, dim, dtype=torch.bfloat16, device=score.device)

    if not isinstance(bitwise_consistent, bool):
        raise TypeError('bitwise_consistent must be a bool.')
    get_kernel = _compress_and_update_state_prefill_tl_ascend

    compute_kernel = get_kernel(
        dim=dim,
        compress_ratio=compress_ratio,
        overlap_ratio=overlap_ratio,
        has_ape=has_ape,
        has_index=mirror_cp_reverse_index is not None,
        bitwise_consistent=bitwise_consistent,
        stage='compute',
    )
    store_kernel = get_kernel(
        dim=dim,
        compress_ratio=compress_ratio,
        overlap_ratio=overlap_ratio,
        has_ape=has_ape,
        has_index=mirror_cp_reverse_index is not None,
        bitwise_consistent=bitwise_consistent,
        stage='store',
    )
    if should_print_kernel_source():
        print(compute_kernel.get_kernel_source())
        print(store_kernel.get_kernel_source())
    compute_kernel(
        score,
        latent,
        ape,
        positions,
        state_cache,
        state_block_idx,
        compressed_batch_mapping,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_compressed_seqlens_q,
        mirror_cp_reverse_index,
        kv_compressed,
    )
    store_kernel(
        score,
        latent,
        ape,
        positions,
        state_cache,
        state_block_idx,
        cu_seqlens_q,
        cu_seqlens_k,
        mirror_cp_reverse_index,
    )
    return kv_compressed


def compress_and_update_state_decode(
    score: torch.Tensor,
    latent: torch.Tensor,
    ape: torch.Tensor | None,
    positions: torch.Tensor | None,
    state_cache: torch.Tensor,
    state_block_idx: torch.Tensor,
    context_lens: torch.Tensor,
    max_seqlen_q: int,
    compress_ratio: int | None = None,
    *,
    bitwise_consistent: bool = False,
) -> torch.Tensor:
    """
    Args:
        score: [total_q, overlap_ratio, dim], fp32
        latent: [total_q, overlap_ratio, dim], fp32
        ape: [compress_ratio, overlap_ratio, dim], fp32
        positions: [total_q], int32
        state_cache: [num_state_slots, state_cache_size, 2, overlap_ratio, dim], fp32
        state_block_idx: [total_q], int32
        context_lens: [total_q], int32

    Returns:
        kv_compressed: [total_q, dim], bf16
    """
    require_ascend()
    _validate_common_inputs(score, latent, ape, positions, state_cache, state_block_idx)
    total_q, overlap_ratio, dim = score.shape
    if ape is not None:
        has_ape = True
        compress_ratio = ape.shape[0]
    else:
        has_ape = False
        assert compress_ratio is not None
    _validate_kernel_capacity(
        dim,
        compress_ratio,
        overlap_ratio,
        has_ape=has_ape,
        is_prefill=False,
    )
    assert state_cache.shape[1] >= compress_ratio * overlap_ratio + max_seqlen_q - 1
    kv_compressed = torch.empty(total_q, dim, dtype=torch.bfloat16, device=score.device)

    if not isinstance(bitwise_consistent, bool):
        raise TypeError('bitwise_consistent must be a bool.')
    get_kernel = _compress_and_update_state_decode_tl_ascend

    store_kernel = get_kernel(
        dim=dim,
        compress_ratio=compress_ratio,
        overlap_ratio=overlap_ratio,
        has_ape=has_ape,
        bitwise_consistent=bitwise_consistent,
        stage='store',
    )
    compute_kernel = get_kernel(
        dim=dim,
        compress_ratio=compress_ratio,
        overlap_ratio=overlap_ratio,
        has_ape=has_ape,
        bitwise_consistent=bitwise_consistent,
        stage='compute',
    )
    if should_print_kernel_source():
        print(store_kernel.get_kernel_source())
        print(compute_kernel.get_kernel_source())
    store_kernel(
        score,
        latent,
        ape,
        positions,
        state_cache,
        state_block_idx,
        context_lens,
        kv_compressed,
    )
    compute_kernel(
        state_cache,
        state_block_idx,
        context_lens,
        kv_compressed,
    )
    return kv_compressed
