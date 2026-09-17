"""TileLang Ascend implementation of one Engram Sinkhorn step."""

import tilelang
from tilelang import language as T
from tilelang.language import simd as S


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
        'tl.enable_fast_math': True,
    },
)
def get_engram_sinkhorn_step_kernel(
    head_dim: int,
    eps: float,
    num_cores: int,
    rows_per_core: int,
    chunk_rows: int,
    num_stages: int,
):
    """Build the chunked, multi-stage Ascend Sinkhorn step kernel."""

    tile_size = tilelang.next_power_of_2(head_dim)
    num_threads = 256 if tile_size > 4096 else 128
    vector_size = 64
    assert head_dim % 256 == 0
    num_vectors = head_dim // vector_size
    use_local_acc = num_vectors <= 8
    num_rows = T.dynamic('num_rows')

    @T.prim_func
    def engram_sinkhorn_step_kernel(
        matrix: T.Tensor[(num_rows, head_dim), T.float32],
        row_scale: T.Tensor[(num_rows,), T.float32],
        col_scale: T.Tensor[(head_dim,), T.float32],
        row_norm: T.Tensor[(num_rows,), T.float32],
        col_sumsq: T.Tensor[(head_dim,), T.float32],
    ):
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((num_stages, chunk_rows, head_dim), T.float32)
            col_scale_ub = T.alloc_shared((head_dim,), T.float32)
            col_acc_ub = T.alloc_shared((head_dim,), T.float32)
            row_scale_in_ub = T.alloc_shared((rows_per_core,), T.float32)
            row_scale_ub = T.alloc_shared((rows_per_core,), T.float32)
            row_norm_ub = T.alloc_shared((rows_per_core,), T.float32)
            if not use_local_acc:
                x_vec_ub = T.alloc_shared((head_dim,), T.float32)

            T.copy(col_scale[:head_dim], col_scale_ub[:head_dim])
            with T.SimdVF():
                T.clear(col_acc_ub)
            row_base = num_rows // num_cores
            row_rem = num_rows % num_cores
            row_begin = core_id * row_base + T.min(core_id, row_rem)
            row_end = row_begin + row_base + T.cast(core_id < row_rem, 'int32')
            row_count = row_end - row_begin
            num_chunks = T.ceildiv(row_count, chunk_rows)
            T.copy(row_scale[row_begin:row_end], row_scale_in_ub[:row_count])
            for stage in range(num_stages):
                T.ascend_set_flag('V_MTE2', stage)
            for chunk in T.serial(num_chunks):
                stage = chunk % num_stages
                T.ascend_wait_flag('V_MTE2', stage)
                row_start = row_begin + chunk * chunk_rows
                copy_rows = T.min(chunk_rows, row_end - row_start)
                T.copy(
                    matrix[row_start : row_start + copy_rows, :head_dim],
                    x_ub[stage, :copy_rows, :head_dim],
                )
                T.ascend_set_flag('MTE2_V', stage)
                T.ascend_wait_flag('MTE2_V', stage)
                if use_local_acc:
                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        one_lane = S.pset(32, 'PAT_VL1')
                        zero = S.vdup(0.0, T.float32, full)
                        sum_x2 = S.alloc_var(T.float32)
                        col_acc_l = S.alloc_local((num_vectors, vector_size), T.float32)
                        xcs_l = S.alloc_local((num_vectors, vector_size), T.float32)
                        cs_l = S.alloc_local((num_vectors, vector_size), T.float32)
                        for vector in T.unroll(num_vectors, explicit=True):
                            col_acc_l[vector, 0] = S.vld(col_acc_ub[vector * vector_size])
                            cs_l[vector, 0] = S.vld(col_scale_ub[vector * vector_size])
                        for row in T.serial(copy_rows):
                            row_id = row_start + row
                            row_local = row_id - row_begin
                            scale = S.vld(row_scale_in_ub[row_local], dist='BRC_B32')
                            sum_x2 = zero
                            for vector in T.unroll(num_vectors, explicit=True):
                                xcs_l[vector, 0] = S.vmul(
                                    S.vld(x_ub[stage, row, vector * vector_size]),
                                    cs_l[vector, 0],
                                    full,
                                )
                                S.vmula(sum_x2, xcs_l[vector, 0], xcs_l[vector, 0], full)
                            base_norm = S.vsqrt(S.vdupv(S.vcadd(sum_x2, full), full))
                            norm = S.vmul(scale, base_norm, full)
                            normalized_scale = S.vdiv(scale, S.vadds(norm, eps))
                            S.vsts(row_norm_ub[row_local], norm, one_lane, dist='ONEPT_B32')
                            S.vsts(row_scale_ub[row_local], normalized_scale, one_lane, dist='ONEPT_B32')
                            normalized_scale_sq = S.vmul(normalized_scale, normalized_scale, full)
                            for vector in T.unroll(num_vectors, explicit=True):
                                xcs_sq = S.vmul(xcs_l[vector, 0], xcs_l[vector, 0], full)
                                S.vmula(col_acc_l[vector, 0], xcs_sq, normalized_scale_sq, full)
                        for vector in T.unroll(num_vectors, explicit=True):
                            S.vsts(col_acc_ub[vector * vector_size], col_acc_l[vector, 0], full)
                else:
                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        one_lane = S.pset(32, 'PAT_VL1')
                        zero = S.vdup(0.0, T.float32, full)
                        sum_x2 = S.alloc_var(T.float32)
                        for row in T.serial(copy_rows):
                            row_id = row_start + row
                            row_local = row_id - row_begin
                            scale = S.vld(row_scale_in_ub[row_local], dist='BRC_B32')
                            sum_x2 = zero
                            for vector in range(num_vectors):
                                offset = vector * vector_size
                                x_vec = S.vmul(
                                    S.vld(x_ub[stage, row, offset]),
                                    S.vld(col_scale_ub[offset]),
                                    full,
                                )
                                S.vsts(x_vec_ub[offset], x_vec, full)
                                S.vmula(sum_x2, x_vec, x_vec, full)
                            base_norm = S.vsqrt(S.vdupv(S.vcadd(sum_x2, full), full))
                            norm = S.vmul(scale, base_norm, full)
                            normalized_scale = S.vdiv(scale, S.vadds(norm, eps))
                            S.vsts(row_norm_ub[row_local], norm, one_lane, dist='ONEPT_B32')
                            S.vsts(row_scale_ub[row_local], normalized_scale, one_lane, dist='ONEPT_B32')
                            normalized_scale_sq = S.vmul(normalized_scale, normalized_scale, full)
                            for vector in range(num_vectors):
                                offset = vector * vector_size
                                x_vec_sq = S.vmul(S.vld(x_vec_ub[offset]), S.vld(x_vec_ub[offset]), full)
                                S.vsts(
                                    col_acc_ub[offset],
                                    S.vadd(
                                        S.vld(col_acc_ub[offset]),
                                        S.vmul(x_vec_sq, normalized_scale_sq, full),
                                        full,
                                    ),
                                    full,
                                )
                T.ascend_set_flag('V_MTE2', stage)
            for stage in range(num_stages):
                T.ascend_wait_flag('V_MTE2', stage)
            T.ascend_set_flag('V_MTE3', 2)
            T.ascend_wait_flag('V_MTE3', 2)
            T.copy(row_norm_ub[:row_count], row_norm[row_begin : row_begin + row_count])
            T.copy(row_scale_ub[:row_count], row_scale[row_begin : row_begin + row_count])
            with T.SimtVF(threads=num_threads):
                for index in T.Parallel(tile_size):
                    if index < head_dim:
                        T.atomic_add(col_sumsq[index], col_acc_ub[index])

    return engram_sinkhorn_step_kernel
