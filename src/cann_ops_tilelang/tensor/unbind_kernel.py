"""TileLang Ascend kernels for copying tensor slices into independent tensors."""

from typing import Any

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
def _unbind_tl_ascend(
    from_stride0: int,
) -> Any:
    req_cnt = T.dynamic('req_cnt')
    nbytes = T.dynamic('nbytes')
    num_cores = get_num_vector_cores()
    block_bytes = 32768

    @T.prim_func
    def _unbind(
        from_bytes: T.StridedTensor[(req_cnt, nbytes), (from_stride0, 1), T.int8],
        out_ptrs: T.Tensor[(req_cnt,), T.ptr],
    ) -> None:
        with T.Kernel(num_cores) as pid:
            copy_ub = T.alloc_shared((block_bytes,), T.int8)
            for pid_n, pid_b in T.Persistent(
                [req_cnt, T.ceildiv(nbytes, block_bytes)],
                num_cores,
                pid,
                group_size=1,
                num_stages=0,
            ):
                out = T.make_tensor(out_ptrs[pid_n], (nbytes,), T.int8)
                offset = pid_b * block_bytes
                valid_bytes = T.min(block_bytes, nbytes - offset)
                T.copy(
                    from_bytes[pid_n, offset : offset + valid_bytes],
                    copy_ub[:valid_bytes],
                )
                T.copy(copy_ub[:valid_bytes], out[offset : offset + valid_bytes])

    return _unbind


@tilelang.jit(
    out_idx=None,
    target='ascend',
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _unbind_tl_ascend_small(
    from_stride0: int,
    num_stages: int,
    num_cores: int,
    req_cnt: int,
    requests_per_task: int,
) -> Any:
    block_bytes = 4096
    task_cnt = req_cnt // requests_per_task

    @T.prim_func
    def _unbind(
        from_bytes: T.StridedTensor[(req_cnt, block_bytes), (from_stride0, 1), T.int8],
        out_ptrs: T.Tensor[(req_cnt,), T.ptr],
    ) -> None:
        with T.Kernel(num_cores) as pid:
            if requests_per_task == 1:
                # Preserve the lower-overhead one-dimensional kernel for the
                # 64-request case; grouping only pays off for larger batches.
                copy_ub = T.alloc_shared((block_bytes,), T.int8)
                T.annotate_buffer_versions({copy_ub: num_stages})
                for req_id in T.Persistent(
                    [req_cnt],
                    num_cores,
                    pid,
                    group_size=1,
                    num_stages=num_stages,
                ):
                    out = T.make_tensor(out_ptrs[req_id], (block_bytes,), T.int8)
                    T.copy(from_bytes[req_id, :], copy_ub)
                    T.copy(copy_ub, out)
            else:
                copy_ub = T.alloc_shared((requests_per_task, block_bytes), T.int8)
                T.annotate_buffer_versions({copy_ub: num_stages})
                for task_id in T.Persistent(
                    [task_cnt],
                    num_cores,
                    pid,
                    group_size=1,
                    num_stages=num_stages,
                ):
                    req_start = task_id * requests_per_task
                    T.copy(
                        from_bytes[req_start : req_start + requests_per_task, :],
                        copy_ub,
                    )
                    for req_offset in T.unroll(requests_per_task, explicit=True):
                        req_id = req_start + req_offset
                        out = T.make_tensor(out_ptrs[req_id], (block_bytes,), T.int8)
                        T.copy(copy_ub[req_offset, :], out)

    return _unbind
