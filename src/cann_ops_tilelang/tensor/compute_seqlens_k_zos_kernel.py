import tilelang
from tilelang.ascend import language as T


@tilelang.jit(
    out_idx=None,
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def get_compute_seqlens_k_zos_kernel(nextn: int, num_cores: int):
    """Build the Ascend kernel that updates ZOS key sequence lengths."""
    n_seq = T.dynamic('n_seq')
    n_prev_seq = T.dynamic('n_prev_seq')
    block_size = 128

    @T.prim_func
    def compute_seqlens_k_zos_kernel(
        seqlens_k: T.Tensor((n_seq,), 'int32'),
        seqlens_q: T.Tensor((n_seq,), 'int32'),
        remap: T.Tensor((n_seq,), 'int32'),
        prev_num_accepted_tokens: T.Tensor((n_prev_seq,), 'int32'),
    ):
        with T.Kernel(num_cores) as core_id:
            for block in T.Persistent(
                [T.ceildiv(n_seq, block_size)],
                num_cores,
                core_id,
                group_size=1,
            ):
                with T.SimtVF(threads=block_size):
                    thread_id = T.get_thread_binding()
                    seq_idx = block * block_size + thread_id
                    if seq_idx < n_seq:
                        prev_idx = remap[seq_idx]
                        safe_prev_idx = T.Select(prev_idx == -1, 0, prev_idx)
                        adjustment = T.Select(
                            prev_idx == -1,
                            0,
                            prev_num_accepted_tokens[safe_prev_idx] - nextn,
                        )
                        seqlens_k[seq_idx] = seqlens_k[seq_idx] + seqlens_q[seq_idx] + adjustment

    return compute_seqlens_k_zos_kernel
