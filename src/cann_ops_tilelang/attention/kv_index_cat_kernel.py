import tilelang
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(
    out_idx=None,
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
    },
)
def _kv_index_cat_tl_ascend(
    dtype: str,
    window_size: int,
    topk_tokens: int,
    out_stride: int,
) -> None:
    total_q = T.dynamic('total_q')

    if max(window_size, topk_tokens) <= 128:
        block_size = 128
        rows_per_task = 64
        num_stages = 2
    else:
        block_size = 6336
        rows_per_task = 5
        num_stages = 2
    vec_len = 64
    blocks_l = (window_size + block_size - 1) // block_size
    blocks_r = (topk_tokens + block_size - 1) // block_size
    num_cores = get_num_vector_cores()

    @T.prim_func
    def _kv_index_cat(
        kv_index: T.Tensor((total_q, window_size), dtype),  # type: ignore
        kv_ca_index: T.Tensor((total_q, topk_tokens), dtype),  # type: ignore
        kv_index_out: T.StridedTensor((total_q, window_size), (out_stride, 1), dtype),  # type: ignore
        kv_ca_index_out: T.StridedTensor((total_q, topk_tokens), (out_stride, 1), dtype),  # type: ignore
        total_k_swa: T.int32,
        total_k: T.int32,
    ) -> None:
        with T.Kernel(num_cores) as pid:
            data_ub = T.alloc_shared((num_stages, rows_per_task, block_size), dtype)

            for stage in T.serial(num_stages):
                T.ascend_set_flag('MTE3_MTE2', stage)

            for bid_l in T.Persistent(
                [T.ceildiv(total_q, rows_per_task) * blocks_l],
                num_cores,
                pid,
                group_size=1,
                num_stages=0,
            ):
                row_l = (bid_l // blocks_l) * rows_per_task
                block_l = bid_l % blocks_l
                offset_l = block_l * block_size
                copy_rows_l = T.min(rows_per_task, total_q - row_l)
                copy_length_l = T.min(block_size, window_size - offset_l)
                slot_l = (bid_l // num_cores) % num_stages

                T.ascend_wait_flag('MTE3_MTE2', slot_l)
                T.copy(
                    kv_index[row_l : row_l + copy_rows_l, offset_l : offset_l + copy_length_l],
                    data_ub[slot_l, :copy_rows_l, :copy_length_l],
                    l2_cache_ctrl='NOTALLOC_KEEP',
                )
                T.ascend_set_flag('MTE2_MTE3', slot_l)
                T.ascend_wait_flag('MTE2_MTE3', slot_l)
                T.copy(
                    data_ub[slot_l, :copy_rows_l, :copy_length_l],
                    kv_index_out[row_l : row_l + copy_rows_l, offset_l : offset_l + copy_length_l],
                    l2_cache_ctrl='NOTALLOC_KEEP',
                )
                T.ascend_set_flag('MTE3_MTE2', slot_l)

            for stage in T.serial(num_stages):
                T.ascend_wait_flag('MTE3_MTE2', stage)
                T.ascend_set_flag('MTE3_MTE2', stage)

            for bid_r in T.Persistent(
                [T.ceildiv(total_q, rows_per_task) * blocks_r],
                num_cores,
                pid,
                group_size=1,
                num_stages=0,
            ):
                row_r = (bid_r // blocks_r) * rows_per_task
                block_r = bid_r % blocks_r
                offset_r = block_r * block_size
                copy_rows_r = T.min(rows_per_task, total_q - row_r)
                copy_length_r = T.min(block_size, topk_tokens - offset_r)
                slot_r = (bid_r // num_cores) % num_stages

                T.ascend_wait_flag('MTE3_MTE2', slot_r)
                T.copy(
                    kv_ca_index[row_r : row_r + copy_rows_r, offset_r : offset_r + copy_length_r],
                    data_ub[slot_r, :copy_rows_r, :copy_length_r],
                    l2_cache_ctrl='NOTALLOC_KEEP',
                )
                T.ascend_set_flag('MTE2_V', slot_r)
                T.ascend_wait_flag('MTE2_V', slot_r)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for tile_row in T.serial(copy_rows_r):
                        for v in T.serial(T.ceildiv(copy_length_r, vec_len)):
                            value = T.simd.vld(data_ub[slot_r, tile_row, v * vec_len])
                            value_added = T.simd.vadds(value, total_k + total_k_swa, mask)
                            value_out = T.simd.vsel(value, value_added, T.simd.vcmps(value, -1, mask, 'eq'))
                            T.simd.vsts(data_ub[slot_r, tile_row, v * vec_len], value_out, mask)

                T.ascend_set_flag('V_MTE3', slot_r)
                T.ascend_wait_flag('V_MTE3', slot_r)
                T.copy(
                    data_ub[slot_r, :copy_rows_r, :copy_length_r],
                    kv_ca_index_out[row_r : row_r + copy_rows_r, offset_r : offset_r + copy_length_r],
                    l2_cache_ctrl='NOTALLOC_KEEP',
                )
                T.ascend_set_flag('MTE3_MTE2', slot_r)

            for stage in T.serial(num_stages):
                T.ascend_wait_flag('MTE3_MTE2', stage)

    return _kv_index_cat
