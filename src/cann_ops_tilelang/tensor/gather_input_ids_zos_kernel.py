import tilelang
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(
    out_idx=None,
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def get_gather_input_ids_zos_kernel(max_seqlen_q: int):
    """Build the Ascend kernel that gathers ZOS input token IDs."""
    n_seq = T.dynamic('n_seq')
    prev_n_seq = T.dynamic('prev_n_seq')
    total_q = T.dynamic('total_q')
    nextn = max_seqlen_q - 1
    num_cores = get_num_vector_cores()
    threads = 128
    block_size = threads // max_seqlen_q

    @T.prim_func
    def gather_input_ids_zos_kernel(
        fresh_tokens: T.Tensor((n_seq, max_seqlen_q), 'int32'),
        prev_sampled_token_ids: T.Tensor((prev_n_seq,), 'int32'),
        prev_predict_token_ids: T.Tensor((prev_n_seq, nextn), 'int32'),
        remap: T.Tensor((n_seq,), 'int32'),
        seqlens_q: T.Tensor((n_seq,), 'int32'),
        cu_seqlens_q: T.Tensor((n_seq,), 'int32'),
        out_input_ids: T.Tensor((total_q,), 'int32'),
    ):
        with T.Kernel(num_cores) as core_id:
            seqlens_q_ub = T.alloc_shared((block_size,), 'int32')
            cu_seqlens_q_ub = T.alloc_shared((block_size,), 'int32')
            remap_ub = T.alloc_shared((block_size,), 'int32')

            for block in T.Persistent(
                [T.ceildiv(n_seq, block_size)],
                num_cores,
                core_id,
                group_size=1,
            ):
                batch_base = block * block_size
                n_remaining = T.min(block_size, n_seq - batch_base)

                T.copy(seqlens_q[batch_base : batch_base + n_remaining], seqlens_q_ub[:n_remaining])
                T.copy(cu_seqlens_q[batch_base : batch_base + n_remaining], cu_seqlens_q_ub[:n_remaining])
                T.copy(remap[batch_base : batch_base + n_remaining], remap_ub[:n_remaining])

                with T.SimtVF(threads=threads):
                    thread_id = T.get_thread_binding()
                    local_seq = thread_id // max_seqlen_q
                    token_idx = thread_id % max_seqlen_q
                    seq_idx = batch_base + local_seq
                    if local_seq < n_remaining:
                        seq_len = seqlens_q_ub[local_seq]
                        if token_idx < seq_len:
                            q_base = cu_seqlens_q_ub[local_seq] - seq_len
                            prev_idx = remap_ub[local_seq]
                            if prev_idx >= 0:
                                if token_idx == 0:
                                    out_input_ids[q_base + token_idx] = prev_sampled_token_ids[prev_idx]
                                else:
                                    out_input_ids[q_base + token_idx] = prev_predict_token_ids[prev_idx, token_idx - 1]
                            else:
                                out_input_ids[q_base + token_idx] = fresh_tokens[seq_idx, token_idx]

    return gather_input_ids_zos_kernel
