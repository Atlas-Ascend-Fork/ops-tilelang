"""Ascend kernel for gathering logits at the current decoding step."""

from typing import Any

import tilelang
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _get_logits_cur_by_step_tl_ascend(
    vocab_size: int,
) -> Any:
    total_q = T.dynamic('total_q')
    logits_stride_0 = T.dynamic('logits_stride_0')
    n = T.dynamic('n')
    block_dim = 4096
    num_dim_blocks = T.ceildiv(vocab_size, block_dim)
    num_cores = get_num_vector_cores()

    @T.prim_func
    def _get_logits_cur_by_step(
        logits: T.StridedTensor[(total_q, vocab_size), (logits_stride_0, 1), T.float],
        ftp: T.Tensor[n, T.int32],
        has_token: T.Tensor[n, T.bool],
        out: T.Tensor[(n, vocab_size), T.float],
        step: T.int32,
    ) -> None:
        T.assume(logits_stride_0 % 16 == 0)
        with T.Kernel(num_cores) as core_id:
            logits_ub = T.alloc_shared((block_dim,), T.float32)
            for row, dim_block in T.Persistent(
                [n, num_dim_blocks],
                num_cores,
                core_id,
                group_size=1,
                num_stages=2,
            ):
                dim_start = dim_block * block_dim
                copy_length = T.min(block_dim, vocab_size - dim_start)
                if has_token[row]:
                    gather_idx = ftp[row] + step
                    T.assume(0 <= gather_idx < total_q)
                    T.copy(
                        logits[gather_idx, dim_start : dim_start + copy_length],
                        logits_ub[:copy_length],
                    )
                else:
                    T.fill(logits_ub[:copy_length], 0)
                T.copy(
                    logits_ub[:copy_length],
                    out[row, dim_start : dim_start + copy_length],
                )

    return _get_logits_cur_by_step
