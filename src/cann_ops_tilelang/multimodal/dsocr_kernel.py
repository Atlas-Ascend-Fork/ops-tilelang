"""TileLang Ascend kernel for relative-position attention."""

import math

import tilelang
from tilelang.ascend import language as T
from tilelang.layout import make_ascend_compact_nz_layout


@tilelang.jit(
    out_idx=[5],
    target='ascend',
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _attn_rel_h_rel_w_tl_ascend(
    heads: int,
    seq_len: int,
    dim: int,
    num_tasks: int,
    num_cores: int,
    block_m: int = 128,
    block_n: int = 128,
) -> object:
    compact_nz_min_key_blocks = 13
    pipeline_3stage_min_key_blocks = 13

    batch = T.dynamic('batch')
    q_stride_b = T.dynamic('q_stride_b')
    q_stride_s = T.dynamic('q_stride_s')
    q_stride_h = T.dynamic('q_stride_h')
    k_stride_b = T.dynamic('k_stride_b')
    k_stride_s = T.dynamic('k_stride_s')
    k_stride_h = T.dynamic('k_stride_h')
    v_stride_b = T.dynamic('v_stride_b')
    v_stride_s = T.dynamic('v_stride_s')
    v_stride_h = T.dynamic('v_stride_h')
    rel_h_stride_b = T.dynamic('rel_h_stride_b')
    rel_h_stride_s = T.dynamic('rel_h_stride_s')
    rel_h_stride_h = T.dynamic('rel_h_stride_h')
    rel_w_stride_b = T.dynamic('rel_w_stride_b')
    rel_w_stride_s = T.dynamic('rel_w_stride_s')
    rel_w_stride_h = T.dynamic('rel_w_stride_h')
    side = math.isqrt(seq_len)
    assert side * side == seq_len
    assert dim == 64, 'the Ascend SIMD softmax currently requires dim=64'
    assert block_m == 128 and block_n == 128

    dtype = T.bfloat16
    accum_dtype = T.float32
    rows_per_aiv = block_m // 2
    vector_lanes = 64
    query_blocks = T.ceildiv(seq_len, block_m)
    key_blocks = T.ceildiv(seq_len, block_n)
    key_block_count = (seq_len + block_n - 1) // block_n
    use_contiguous_rel_bias = side % vector_lanes == 0
    use_compact_nz = key_block_count >= compact_nz_min_key_blocks
    pipeline_stages = 3 if key_block_count >= pipeline_3stage_min_key_blocks else 2
    pipeline_annotations = {'enable_offset': True}
    w_aligned = ((side + 7) // 8) * 8
    scale = math.sqrt(1.0 / dim)
    input_shape = [batch, seq_len, heads, dim]
    rel_shape = [batch, seq_len, heads, w_aligned]

    @T.macro
    def ordinary_softmax(score_ub, prob_ub, max_ub, sum_ub, alpha_ub):
        with T.SimdVF(latency=948):
            softmax_full = T.simd.pset(32, 'PAT_ALL')
            one = T.simd.pset(32, 'PAT_VL1')
            for row in range(0, rows_per_aiv, 2):
                old_max_0 = T.simd.vld(max_ub[row], 'BRC_B32')
                old_max_1 = T.simd.vld(max_ub[row + 1], 'BRC_B32')

                score_00 = T.simd.vld(score_ub[row, 0], 'NORM')
                score_01 = T.simd.vld(score_ub[row, vector_lanes], 'NORM')
                score_10 = T.simd.vld(score_ub[row + 1, 0], 'NORM')
                score_11 = T.simd.vld(score_ub[row + 1, vector_lanes], 'NORM')

                max_pair_00 = T.simd.vmax(old_max_0, score_00, softmax_full)
                max_pair_01 = T.simd.vmax(max_pair_00, score_01, softmax_full)
                new_max_0 = T.simd.vcmax(max_pair_01, softmax_full)
                T.simd.vsts(max_ub[row], new_max_0, one, 'ONEPT_B32')
                new_max_broadcast_0 = T.simd.vdupv(new_max_0, softmax_full, 'POS_LOWEST')

                max_pair_10 = T.simd.vmax(old_max_1, score_10, softmax_full)
                max_pair_11 = T.simd.vmax(max_pair_10, score_11, softmax_full)
                new_max_1 = T.simd.vcmax(max_pair_11, softmax_full)
                T.simd.vsts(max_ub[row + 1], new_max_1, one, 'ONEPT_B32')
                new_max_broadcast_1 = T.simd.vdupv(new_max_1, softmax_full, 'POS_LOWEST')

                alpha_0 = T.simd.vexpdif(old_max_0, new_max_broadcast_0, softmax_full)
                alpha_1 = T.simd.vexpdif(old_max_1, new_max_broadcast_1, softmax_full)
                T.simd.vsts(alpha_ub[row], alpha_0, one, 'ONEPT_B32')
                T.simd.vsts(alpha_ub[row + 1], alpha_1, one, 'ONEPT_B32')

                prob_00 = T.simd.vexpdif(score_00, new_max_broadcast_0, softmax_full)
                prob_01 = T.simd.vexpdif(score_01, new_max_broadcast_0, softmax_full)
                prob_10 = T.simd.vexpdif(score_10, new_max_broadcast_1, softmax_full)
                prob_11 = T.simd.vexpdif(score_11, new_max_broadcast_1, softmax_full)

                T.simd.vsts(prob_ub[row, 0], T.simd.vcvt(prob_00, 'bfloat16', softmax_full), softmax_full, 'PK_B32')
                T.simd.vsts(
                    prob_ub[row, vector_lanes], T.simd.vcvt(prob_01, 'bfloat16', softmax_full), softmax_full, 'PK_B32'
                )
                T.simd.vsts(prob_ub[row + 1, 0], T.simd.vcvt(prob_10, 'bfloat16', softmax_full), softmax_full, 'PK_B32')
                T.simd.vsts(
                    prob_ub[row + 1, vector_lanes],
                    T.simd.vcvt(prob_11, 'bfloat16', softmax_full),
                    softmax_full,
                    'PK_B32',
                )

                prob_sum_0 = T.simd.vcadd(T.simd.vadd(prob_00, prob_01, softmax_full), softmax_full)
                prob_sum_1 = T.simd.vcadd(T.simd.vadd(prob_10, prob_11, softmax_full), softmax_full)
                old_sum_0 = T.simd.vld(sum_ub[row], 'BRC_B32')
                old_sum_1 = T.simd.vld(sum_ub[row + 1], 'BRC_B32')
                new_sum_0 = T.simd.vadd(T.simd.vmul(old_sum_0, alpha_0, softmax_full), prob_sum_0, one)
                new_sum_1 = T.simd.vadd(T.simd.vmul(old_sum_1, alpha_1, softmax_full), prob_sum_1, one)
                T.simd.vsts(sum_ub[row], new_sum_0, one, 'ONEPT_B32')
                T.simd.vsts(sum_ub[row + 1], new_sum_1, one, 'ONEPT_B32')

    @T.macro
    def compact_softmax(
        score_ub,
        p_nz_ub,
        max_ub,
        sum_ub,
        alpha_ub,
        old_m_ub,
        row_sum_ub,
    ):
        with T.SimdVF():
            softmax_full = T.simd.pset(32, 'PAT_ALL')
            one = T.simd.pset(32, 'PAT_VL1')
            softmax_b16 = T.simd.pset(16, 'PAT_VL128')

            T.simd.vsts(old_m_ub[0], T.simd.vld(max_ub[0], 'NORM'), softmax_full, 'NORM_B32')

            for row in range(0, rows_per_aiv, 2):
                score_00 = T.simd.vld(score_ub[row, 0], 'NORM')
                score_01 = T.simd.vld(score_ub[row, vector_lanes], 'NORM')
                score_10 = T.simd.vld(score_ub[row + 1, 0], 'NORM')
                score_11 = T.simd.vld(score_ub[row + 1, vector_lanes], 'NORM')

                new_max_0 = T.simd.vcmax(T.simd.vmax(score_00, score_01, softmax_full), softmax_full)
                T.simd.vsts(max_ub[row], new_max_0, one, 'ONEPT_B32')
                new_max_1 = T.simd.vcmax(T.simd.vmax(score_10, score_11, softmax_full), softmax_full)
                T.simd.vsts(max_ub[row + 1], new_max_1, one, 'ONEPT_B32')

            T.simd.mem_bar('VST_VLD')
            merged_max = T.simd.vmax(T.simd.vld(max_ub[0], 'NORM'), T.simd.vld(old_m_ub[0], 'NORM'), softmax_full)
            T.simd.vsts(max_ub[0], merged_max, softmax_full, 'NORM_B32')
            T.simd.mem_bar('VST_VLD')

            for row in range(0, rows_per_aiv, 2):
                new_max_0 = T.simd.vld(max_ub[row], 'BRC_B32')
                new_max_1 = T.simd.vld(max_ub[row + 1], 'BRC_B32')
                score_even_0, score_odd_0 = T.simd.vld2(score_ub[row, 0], 'DINTLV_B32')
                score_even_1, score_odd_1 = T.simd.vld2(score_ub[row + 1, 0], 'DINTLV_B32')
                prob_00 = T.simd.vexpdif(score_even_0, new_max_0, softmax_full)
                prob_01 = T.simd.vexpdif(score_odd_0, new_max_0, softmax_full)
                prob_10 = T.simd.vexpdif(score_even_1, new_max_1, softmax_full)
                prob_11 = T.simd.vexpdif(score_odd_1, new_max_1, softmax_full)

                prob_sum_0 = T.simd.vcadd(T.simd.vadd(prob_00, prob_01, softmax_full), softmax_full)
                prob_sum_1 = T.simd.vcadd(T.simd.vadd(prob_10, prob_11, softmax_full), softmax_full)
                T.simd.vsts(row_sum_ub[row], prob_sum_0, one, 'ONEPT_B32')
                T.simd.vsts(row_sum_ub[row + 1], prob_sum_1, one, 'ONEPT_B32')

                even_bf16_0 = T.simd.vcvt(prob_00, 'bfloat16', softmax_full, sat=False, part=0)
                even_bf16_1 = T.simd.vcvt(prob_10, 'bfloat16', softmax_full, sat=False, part=0)
                odd_bf16_0 = T.simd.vcvt(prob_01, 'bfloat16', softmax_full, sat=False, part=1)
                odd_bf16_1 = T.simd.vcvt(prob_11, 'bfloat16', softmax_full, sat=False, part=1)

                merged_bits_0 = T.simd.vor(
                    T.reinterpret(even_bf16_0, 'uint16x128'),
                    T.reinterpret(odd_bf16_0, 'uint16x128'),
                    softmax_b16,
                )
                merged_bits_1 = T.simd.vor(
                    T.reinterpret(even_bf16_1, 'uint16x128'),
                    T.reinterpret(odd_bf16_1, 'uint16x128'),
                    softmax_b16,
                )
                T.simd.vsstb(
                    T.reinterpret(merged_bits_0, 'bfloat16x128'),
                    p_nz_ub[row, 0],
                    T.int32((rows_per_aiv + 1) << 16),
                    softmax_b16,
                )
                T.simd.vsstb(
                    T.reinterpret(merged_bits_1, 'bfloat16x128'),
                    p_nz_ub[row + 1, 0],
                    T.int32((rows_per_aiv + 1) << 16),
                    softmax_b16,
                )

            T.simd.mem_bar('VST_VLD')
            alpha_vec = T.simd.vexpdif(T.simd.vld(old_m_ub[0], 'NORM'), T.simd.vld(max_ub[0], 'NORM'), softmax_full)
            T.simd.vsts(alpha_ub[0], alpha_vec, softmax_full, 'NORM_B32')
            scaled_sum = T.simd.vmul(T.simd.vld(sum_ub[0], 'NORM'), alpha_vec, softmax_full)
            updated_sum = T.simd.vadd(scaled_sum, T.simd.vld(row_sum_ub[0], 'NORM'), softmax_full)
            T.simd.vsts(sum_ub[0], updated_sum, softmax_full, 'NORM_B32')

    @T.prim_func
    def _attn_rel_h_rel_w(
        q: T.StridedTensor(  # type: ignore
            input_shape, [q_stride_b, q_stride_s, q_stride_h, 1], dtype
        ),
        k: T.StridedTensor(  # type: ignore
            input_shape, [k_stride_b, k_stride_s, k_stride_h, 1], dtype
        ),
        v: T.StridedTensor(  # type: ignore
            input_shape, [v_stride_b, v_stride_s, v_stride_h, 1], dtype
        ),
        rel_h: T.StridedTensor(  # type: ignore
            rel_shape,
            [rel_h_stride_b, rel_h_stride_s, rel_h_stride_h, 1],
            dtype,
        ),
        rel_w: T.StridedTensor(  # type: ignore
            rel_shape,
            [rel_w_stride_b, rel_w_stride_s, rel_w_stride_h, 1],
            dtype,
        ),
        output: T.Tensor(input_shape, dtype),  # type: ignore
    ) -> None:
        # Keep query and width as the trailing axes for dual_copy split inference.
        # Explicit strides preserve the original storage without moving data.
        rel_h_bhsw = T.view(
            rel_h,
            shape=(batch, heads, seq_len, w_aligned),
            strides=(rel_h_stride_b, rel_h_stride_h, rel_h_stride_s, 1),
        )
        rel_w_bhsw = T.view(
            rel_w,
            shape=(batch, heads, seq_len, w_aligned),
            strides=(rel_w_stride_b, rel_w_stride_h, rel_w_stride_s, 1),
        )
        output_bhsd = T.view(
            output,
            shape=(batch, heads, seq_len, dim),
            strides=(seq_len * heads * dim, dim, heads * dim, 1),
        )
        with T.Kernel(num_cores) as core_id:
            q_l1 = T.alloc_l1((block_m, dim), dtype)
            k_l1 = T.alloc_l1((block_n, dim), dtype)
            v_l1 = T.alloc_l1((dim, block_n), dtype)
            p_l1 = T.alloc_l1((block_m, block_n), dtype)

            score_l0c = T.alloc_l0c((block_m, block_n), accum_dtype)
            output_l0c = T.alloc_l0c((block_m, dim), accum_dtype)

            score_ub = T.alloc_shared((rows_per_aiv, block_n), accum_dtype)
            if use_compact_nz:
                p_nz_ub = T.alloc_shared((rows_per_aiv + 1, block_n), dtype)
                T.annotate_layout({p_nz_ub: make_ascend_compact_nz_layout(p_nz_ub)})
                old_m_ub = T.alloc_shared((rows_per_aiv,), accum_dtype)
                row_sum_ub = T.alloc_shared((rows_per_aiv,), accum_dtype)
            else:
                prob_ub = T.alloc_shared((rows_per_aiv, block_n), dtype)
            output_ub = T.alloc_shared((rows_per_aiv, dim), accum_dtype)
            output_tmp_ub = T.alloc_shared((rows_per_aiv, dim), accum_dtype)
            output_bf16_ub = T.alloc_shared((rows_per_aiv, dim), dtype)
            rel_h_ub = T.alloc_shared((rows_per_aiv, w_aligned), dtype)
            rel_w_ub = T.alloc_shared((rows_per_aiv, w_aligned), dtype)
            max_ub = T.alloc_shared((rows_per_aiv,), accum_dtype)
            sum_ub = T.alloc_shared((rows_per_aiv,), accum_dtype)
            alpha_ub = T.alloc_shared((rows_per_aiv,), accum_dtype)

            for task_id in T.Persistent([num_tasks], num_cores, core_id, group_size=1):
                query_block = task_id % query_blocks
                matrix_idx = task_id // query_blocks
                batch_idx = matrix_idx // heads
                head_idx = matrix_idx % heads
                query_start = query_block * block_m

                T.copy(
                    q[
                        batch_idx,
                        query_start : query_start + block_m,
                        head_idx,
                        :,
                    ],
                    q_l1,
                )
                T.dual_copy(
                    rel_h_bhsw[
                        batch_idx,
                        head_idx,
                        query_start : query_start + block_m,
                        :,
                    ],
                    rel_h_ub,
                )
                T.dual_copy(
                    rel_w_bhsw[
                        batch_idx,
                        head_idx,
                        query_start : query_start + block_m,
                        :,
                    ],
                    rel_w_ub,
                )

                with T.SimdVF():
                    T.fill(output_ub, 0.0)
                    T.fill(sum_ub, 0.0)
                    T.fill(max_ub, -T.infinity(accum_dtype))

                for key_block in T.Pipelined(
                    key_blocks,
                    num_stages=pipeline_stages,
                    annotations=pipeline_annotations,
                ):
                    key_start = key_block * block_n
                    T.copy(
                        k[
                            batch_idx,
                            key_start : key_start + block_n,
                            head_idx,
                            :,
                        ],
                        k_l1,
                    )
                    T.copy(
                        v[
                            batch_idx,
                            key_start : key_start + block_n,
                            head_idx,
                            :,
                        ],
                        v_l1,
                        transpose=True,
                    )

                    T.gemm(q_l1, k_l1, score_l0c, transpose_B=True, clear_accum=True)
                    T.dual_copy(score_l0c, score_ub)

                    with T.SimdVF():
                        bias_full = T.simd.pset(32, 'PAT_ALL')
                        if not use_contiguous_rel_bias:
                            gather_mask = T.simd.pset(16, 'PAT_ALL')
                            side_vec = T.simd.vdup(T.int16(side), 'int16', gather_mask)
                            zero_b16 = T.simd.vdup(T.int16(0), 'int16', gather_mask)
                            negative_infinity = T.simd.vdup(-T.infinity(accum_dtype), accum_dtype, bias_full)
                        for vector_id in range(block_n // vector_lanes):
                            vector_offset = vector_id * vector_lanes
                            rel_offset = key_start + vector_offset
                            if not use_contiguous_rel_bias:
                                packed_key_indices_b16 = T.simd.vci(T.int16(2 * (key_start + vector_offset)), 'int16')
                                key_indices_b16 = T.simd.vshrs(packed_key_indices_b16, 1)
                                valid_gather_mask = T.simd.vcmps(key_indices_b16, seq_len, gather_mask, 'lt')
                                safe_key_indices_b16 = T.simd.vsel(key_indices_b16, zero_b16, valid_gather_mask)
                                rel_h_index = T.simd.vdiv(safe_key_indices_b16, side_vec, gather_mask)
                                rel_w_index = T.simd.vsub(
                                    safe_key_indices_b16,
                                    T.simd.vmul(rel_h_index, side_vec, gather_mask),
                                    gather_mask,
                                )
                                key_indices = T.simd.vci(T.int32(key_start + vector_offset), 'int32')
                                valid_mask = T.simd.vcmps(key_indices, seq_len, bias_full, 'lt')
                            for i in range(rows_per_aiv):
                                if use_contiguous_rel_bias:
                                    rel_h_vec = T.simd.vcvt(
                                        T.simd.vld(
                                            rel_h_ub[i, rel_offset // side],
                                            'BRC_B16',
                                        ),
                                        accum_dtype,
                                        part=0,
                                    )
                                    rel_w_vec = T.simd.vcvt(
                                        T.simd.vld(
                                            rel_w_ub[i, rel_offset % side],
                                            'UNPK_B16',
                                        ),
                                        accum_dtype,
                                        part=0,
                                    )
                                else:
                                    rel_h_vec = T.simd.vcvt(
                                        T.simd.vgather2(
                                            rel_h_ub[i, 0],
                                            T.reinterpret('uint16x128', rel_h_index),
                                            gather_mask,
                                        ),
                                        accum_dtype,
                                        part=0,
                                    )
                                    rel_w_vec = T.simd.vcvt(
                                        T.simd.vgather2(
                                            rel_w_ub[i, 0],
                                            T.reinterpret('uint16x128', rel_w_index),
                                            gather_mask,
                                        ),
                                        accum_dtype,
                                        part=0,
                                    )
                                scaled_score = T.simd.vmuls(
                                    T.simd.vld(score_ub[i, vector_offset]),
                                    T.float32(scale),
                                    bias_full,
                                )
                                biased_score = T.simd.vadd(
                                    T.simd.vadd(scaled_score, rel_h_vec, bias_full),
                                    rel_w_vec,
                                    bias_full,
                                )
                                score_to_store = (
                                    biased_score
                                    if use_contiguous_rel_bias
                                    else T.simd.vsel(
                                        biased_score,
                                        negative_infinity,
                                        valid_mask,
                                    )
                                )
                                T.simd.vsts(score_ub[i, vector_offset], score_to_store, bias_full)

                    if use_compact_nz:
                        compact_softmax(
                            score_ub,
                            p_nz_ub,
                            max_ub,
                            sum_ub,
                            alpha_ub,
                            old_m_ub,
                            row_sum_ub,
                        )
                        T.dual_copy(p_nz_ub[0:rows_per_aiv, 0:block_n], p_l1)
                    else:
                        ordinary_softmax(
                            score_ub,
                            prob_ub,
                            max_ub,
                            sum_ub,
                            alpha_ub,
                        )
                        T.dual_copy(prob_ub, p_l1)
                    T.gemm(p_l1, v_l1, output_l0c, transpose_B=True, clear_accum=True)
                    T.dual_copy(output_l0c, output_tmp_ub)

                    with T.SimdVF():
                        for i, j in T.Parallel(rows_per_aiv, dim):
                            output_ub[i, j] = output_ub[i, j] * alpha_ub[i] + output_tmp_ub[i, j]

                with T.SimdVF():
                    full = T.simd.pset(32, 'PAT_ALL')
                    for i in range(rows_per_aiv):
                        output_vec = T.simd.vld(output_ub[i, 0], 'NORM')
                        row_sum = T.simd.vld(sum_ub[i], 'BRC_B32')
                        output_norm = T.simd.vdiv(output_vec, row_sum, full)
                        output_vec_bf16 = T.simd.vcvt(output_norm, 'bfloat16', full)
                        T.simd.vsts(output_bf16_ub[i, 0], output_vec_bf16, full, 'PK_B32')

                T.dual_copy(
                    output_bf16_ub,
                    output_bhsd[
                        batch_idx,
                        head_idx,
                        query_start : query_start + block_m,
                        :,
                    ],
                )

    return _attn_rel_h_rel_w
