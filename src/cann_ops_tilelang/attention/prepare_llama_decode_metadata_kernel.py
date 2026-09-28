"""TileLang Ascend kernel for LLaMA decode metadata."""

import tilelang
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(
    out_idx=None,
    target='ascend',
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _prepare_llama_decode_metadata_tl_ascend(
    max_seqlen_q: int,
    max_num_blocks: int,
    block_size: int,
    force_load_balance: bool,
    balance_sequences: bool = False,
    num_stages: int = 1,
) -> None:
    n_seq = T.dynamic('n_seq')
    total_q = T.dynamic('total_q')
    total_blocks = T.dynamic('total_blocks')
    num_cores = get_num_vector_cores()

    @T.prim_func
    def _prepare_llama_decode_metadata(
        seqlens_q: T.Tensor((n_seq,), 'int32'),  # type: ignore
        seqlens_k: T.Tensor((n_seq,), 'int32'),  # type: ignore
        cu_seqlens_q: T.Tensor((n_seq,), 'int32'),  # type: ignore
        block_tables_in: T.Tensor((total_blocks,), 'int32'),  # type: ignore
        block_tables_offsets: T.Tensor((n_seq + 1,), 'int32'),  # type: ignore
        is_placeholder: T.Tensor((n_seq,), T.bool),  # type: ignore
        positions: T.Tensor((total_q,), 'int32'),  # type: ignore
        valid_token_mask: T.Tensor((total_q,), T.bool),  # type: ignore
        gate_force_random: T.Tensor((total_q,), T.bool),  # type: ignore
        fix_routing_mask: T.Tensor((total_q,), T.bool),  # type: ignore
        context_lens: T.Tensor((total_q,), 'int32'),  # type: ignore
        slot_mapping_store: T.Tensor((total_q,), 'int32'),  # type: ignore
        query_indices: T.Tensor((total_q,), 'int32'),  # type: ignore
        block_tables_out: T.Tensor((total_q, max_num_blocks), 'int32'),  # type: ignore
    ):
        with T.Kernel(num_cores) as pid:
            block_table_ub = T.alloc_shared((max_num_blocks,), 'int32')
            if balance_sequences:
                base_sequences = T.floordiv(n_seq, num_cores)
                remainder = T.floormod(n_seq, num_cores)
                seq_begin = pid * base_sequences + T.min(pid, remainder)
                seq_end = seq_begin + base_sequences + T.if_then_else(pid < remainder, 1, 0)
            else:
                sequences_per_core = T.ceildiv(n_seq, num_cores)
                seq_begin = pid * sequences_per_core
                seq_end = T.min(seq_begin + sequences_per_core, n_seq)
            for seq_idx in T.Pipelined(
                seq_begin,
                seq_end,
                num_stages=num_stages,
                annotations={'multi_buffer_eligible': [block_table_ub]},
            ):
                sq = seqlens_q[seq_idx]
                sk = seqlens_k[seq_idx]
                is_pad = is_placeholder[seq_idx]
                q_base = T.if_then_else(seq_idx == 0, 0, cu_seqlens_q[seq_idx - 1])
                bt_offset = block_tables_offsets[seq_idx]
                num_blocks = block_tables_offsets[seq_idx + 1] - bt_offset

                T.assume(sq > 0)
                T.assume(sq <= max_seqlen_q)
                T.assume(q_base >= 0)
                T.assume(q_base + sq <= total_q)
                T.assume(num_blocks > 0)
                T.assume(num_blocks <= max_num_blocks)
                T.assume(bt_offset >= 0)
                T.assume(bt_offset + num_blocks <= total_blocks)

                # Cache each sequence's compact block table once, then reuse it for
                # both slot lookup and all per-token expanded block-table rows.
                T.copy(
                    block_tables_in[bt_offset : bt_offset + num_blocks],
                    block_table_ub[:num_blocks],
                )

                with T.SimtVF(threads=max_seqlen_q):
                    j = T.get_thread_binding()
                    if j < sq:
                        pos = sk - sq + j
                        q_off = q_base + j
                        block_number = block_table_ub[pos // block_size]

                        positions[q_off] = pos
                        valid_token_mask[q_off] = T.if_then_else(is_pad, False, True)
                        gate_force_random[q_off] = force_load_balance
                        fix_routing_mask[q_off] = False
                        context_lens[q_off] = pos + 1
                        query_indices[q_off] = T.if_then_else(is_pad, 998244353, seq_idx)
                        slot_mapping_store[q_off] = block_number * block_size + pos % block_size

                for j in T.serial(sq):
                    T.copy(
                        block_table_ub[:num_blocks],
                        block_tables_out[q_base + j, :num_blocks],
                    )

    return _prepare_llama_decode_metadata
