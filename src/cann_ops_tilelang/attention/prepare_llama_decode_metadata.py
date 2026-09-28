"""Public Ascend interface for LLaMA decode metadata."""

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend

from .prepare_llama_decode_metadata_kernel import _prepare_llama_decode_metadata_tl_ascend


def prepare_llama_decode_metadata(
    seqlens_q: torch.Tensor,
    seqlens_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables_in: torch.Tensor,
    block_tables_offsets: torch.Tensor,
    is_placeholder: torch.Tensor,
    positions: torch.Tensor,
    valid_token_mask: torch.Tensor,
    gate_force_random: torch.Tensor,
    fix_routing_mask: torch.Tensor,
    context_lens: torch.Tensor,
    slot_mapping_store: torch.Tensor,
    query_indices: torch.Tensor,
    block_tables_out: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_num_blocks: int,
    block_size: int,
    force_load_balance: bool = False,
) -> None:
    """Compute LLaMA decode metadata on the active accelerator via TileLang.

    Standard (non-compressed, non-SWA) attention: each query token maps to a slot in
    a paged KV cache addressed by its block table. Supports variable-length seqlen_q.
    block_tables_in is compact 1D with block_tables_offsets for variable-length per-seq
    block tables; block_tables_out is the per-token expanded [total_q, max_num_blocks] form.
    """
    require_ascend()
    n_seq = seqlens_q.shape[0]
    num_cores = get_num_vector_cores()
    long_block_table_threshold = 512
    balance_sequences = num_cores < n_seq <= 64 or block_tables_in.numel() >= n_seq * long_block_table_threshold
    average_num_blocks = block_tables_in.numel() // max(n_seq, 1)
    use_double_buffer = n_seq >= 4 * num_cores or (n_seq > 2 * num_cores and average_num_blocks >= 128)
    kernel = _prepare_llama_decode_metadata_tl_ascend(
        max_seqlen_q, max_num_blocks, block_size, force_load_balance, balance_sequences, 2 if use_double_buffer else 1
    )
    kernel(
        seqlens_q,
        seqlens_k,
        cu_seqlens_q,
        block_tables_in,
        block_tables_offsets,
        is_placeholder,
        positions,
        valid_token_mask,
        gate_force_random,
        fix_routing_mask,
        context_lens,
        slot_mapping_store,
        query_indices,
        block_tables_out,
    )
