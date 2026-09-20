"""TileLang Ascend kernels for the Engram Sinkhorn operator family."""

import tilelang
from tilelang import language as T
from tilelang.language import simd as S


@tilelang.jit
def get_engram_sinkhorn_momentum_update_kernel(
    grad_dtype: T.dtype,
    momentum_dtype: T.dtype,
    nesterov: bool,
    beta1: float,
    num_cores: int,
):
    """Build the vectorized Ascend Sinkhorn momentum-update kernel."""

    tile_size = 4096
    num_stages = 2
    vector_size = 64
    grad_is_bf16 = grad_dtype == T.bfloat16
    momentum_is_bf16 = momentum_dtype == T.bfloat16
    numel = T.dynamic('numel')

    @T.prim_func
    def engram_sinkhorn_momentum_update_kernel(
        grad: T.Tensor[(numel,), grad_dtype],
        momentum_buffer: T.Tensor[(numel,), momentum_dtype],
        sinkhorn_input: T.Tensor[(numel,), T.float32],
    ):
        total_tiles = T.ceildiv(numel, tile_size)

        with T.Kernel(num_cores) as core_id:
            grad_ub = T.alloc_shared((tile_size,), grad_dtype)
            momentum_in_ub = T.alloc_shared((tile_size,), momentum_dtype)
            momentum_out_ub = T.alloc_shared((tile_size,), momentum_dtype)
            sinkhorn_ub = T.alloc_shared((tile_size,), T.float32)
            T.annotate_buffer_versions(
                {
                    grad_ub: num_stages,
                    momentum_in_ub: num_stages,
                    momentum_out_ub: num_stages,
                    sinkhorn_ub: num_stages,
                }
            )

            for tile_id in T.Persistent([total_tiles], num_cores, core_id, num_stages=num_stages):
                base = tile_id * tile_size
                valid_elements = T.min(tile_size, numel - base)
                T.copy(grad[base : base + valid_elements], grad_ub[:valid_elements])
                T.copy(
                    momentum_buffer[base : base + valid_elements],
                    momentum_in_ub[:valid_elements],
                )

                with T.SimdVF():
                    full = S.pset(32, 'PAT_ALL')
                    for vector in range(tile_size // vector_size):
                        offset = vector * vector_size
                        if grad_is_bf16:
                            grad_vec = S.vcvt(
                                S.vld(grad_ub[offset], dist='UNPK_B16'),
                                T.float32,
                                part=0,
                            )
                        else:
                            grad_vec = S.vld(grad_ub[offset])

                        if momentum_is_bf16:
                            momentum_vec = S.vcvt(
                                S.vld(momentum_in_ub[offset], dist='UNPK_B16'),
                                T.float32,
                                part=0,
                            )
                        else:
                            momentum_vec = S.vld(momentum_in_ub[offset])

                        updated_momentum = S.vadd(
                            S.vmuls(momentum_vec, beta1, full),
                            S.vmuls(grad_vec, 1.0 - beta1, full),
                            full,
                        )
                        if nesterov:
                            sinkhorn_vec = S.vadd(
                                S.vmuls(updated_momentum, beta1, full),
                                S.vmuls(grad_vec, 1.0 - beta1, full),
                                full,
                            )
                        else:
                            sinkhorn_vec = updated_momentum

                        if momentum_is_bf16:
                            S.vsts(
                                momentum_out_ub[offset],
                                S.vcvt(updated_momentum, T.bfloat16),
                                full,
                                dist='PK_B32',
                            )
                        else:
                            S.vsts(momentum_out_ub[offset], updated_momentum, full)
                        S.vsts(sinkhorn_ub[offset], sinkhorn_vec, full)

                T.copy(
                    momentum_out_ub[:valid_elements],
                    momentum_buffer[base : base + valid_elements],
                )
                T.copy(
                    sinkhorn_ub[:valid_elements],
                    sinkhorn_input[base : base + valid_elements],
                )

    return engram_sinkhorn_momentum_update_kernel


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


@tilelang.jit
def get_engram_sinkhorn_finalize_kernel(
    head_dim: int,
    eps: float,
    alpha: float,
    num_cores: int,
):
    """Build the vectorized Ascend Sinkhorn finalize kernel."""

    vector_size = 64
    assert head_dim % vector_size == 0
    num_vectors = head_dim // vector_size
    num_stages = 6 if head_dim <= 512 else 3
    num_rows = T.dynamic('num_rows')

    @T.prim_func
    def engram_sinkhorn_finalize_kernel(
        matrix: T.Tensor[(num_rows, head_dim), T.float32],
        row_scale: T.Tensor[(num_rows,), T.float32],
        col_scale: T.Tensor[(head_dim,), T.float32],
        out: T.Tensor[(num_rows, head_dim), T.float32],
    ):
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((head_dim,), T.float32)
            out_ub = T.alloc_shared((head_dim,), T.float32)
            col_scale_ub = T.alloc_shared((head_dim,), T.float32)
            row_scale_ub = T.alloc_shared((1,), T.float32)
            T.annotate_buffer_versions(
                {
                    x_ub: num_stages,
                    out_ub: num_stages,
                    row_scale_ub: num_stages,
                }
            )

            T.copy(col_scale[:head_dim], col_scale_ub[:head_dim])
            rows_per_core = T.ceildiv(num_rows, num_cores)
            row_begin = core_id * rows_per_core
            row_end = T.min(row_begin + rows_per_core, num_rows)
            iterations = row_end - row_begin
            for row_iter in T.Pipelined(
                iterations,
                num_stages=num_stages,
                annotations={'multi_buffer_eligible': [x_ub, out_ub, row_scale_ub]},
            ):
                row_id = row_begin + row_iter
                T.copy(matrix[row_id, :head_dim], x_ub[:head_dim])
                row_scale_ub[0] = row_scale[row_id]

                with T.SimdVF():
                    full = S.pset(32, 'PAT_ALL')
                    zero = S.vdup(0.0, T.float32, full)
                    alpha_vec = S.vdup(alpha, T.float32, full)
                    scale = S.vld(row_scale_ub[0], dist='BRC_B32')
                    sum_x2 = S.alloc_var(T.float32)
                    sum_x2 = zero

                    for vector in range(num_vectors):
                        offset = vector * vector_size
                        x_vec = S.vmul(
                            S.vmul(S.vld(x_ub[offset]), scale, full),
                            S.vld(col_scale_ub[offset]),
                            full,
                        )
                        S.vmula(sum_x2, x_vec, x_vec, full)

                    norm = S.vsqrt(S.vdupv(S.vcadd(sum_x2, full), full))
                    output_scale = S.vdiv(alpha_vec, S.vadds(norm, eps))
                    final_scale = S.vmul(scale, output_scale, full)
                    for vector in range(num_vectors):
                        offset = vector * vector_size
                        out_vec = S.vmul(
                            S.vmul(S.vld(x_ub[offset]), final_scale, full),
                            S.vld(col_scale_ub[offset]),
                            full,
                        )
                        S.vsts(out_ub[offset], out_vec, full)

                T.copy(out_ub[:head_dim], out[row_id, :head_dim])

    return engram_sinkhorn_finalize_kernel
