"""TileLang Ascend kernels for mirror context-parallel metadata."""

import tilelang
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _precompute_tl_ascend(cp_size: int, cp_rank: int, compress_ratio: int) -> None:
    """Ascend AIV implementation of the per-batch precomputation."""
    batch_size = T.dynamic('batch_size')
    num_cores = get_num_vector_cores()

    @T.prim_func
    def _precompute(
        cu_seqlens_q: T.Tensor([batch_size + 1], 'int32'),  # type: ignore
        cu_seqlens_k: T.Tensor([batch_size + 1], 'int32'),  # type: ignore
        k_c4_lens: T.Tensor([batch_size], 'int32'),  # type: ignore
        partial_seqlens: T.Tensor([batch_size], 'int32'),  # type: ignore
        elems_per_batch: T.Tensor([batch_size], 'int32'),  # type: ignore
        delta_sq: T.Tensor([batch_size, 4], 'int32'),  # type: ignore
        delta_sk: T.Tensor([batch_size, 4], 'int32'),  # type: ignore
    ) -> None:
        with T.Kernel(num_cores) as pid:
            for i in T.Persistent([batch_size], num_cores, pid, group_size=1):
                sq = T.int32(cu_seqlens_q[i + 1]) - T.int32(cu_seqlens_q[i])
                sk = T.int32(cu_seqlens_k[i + 1]) - T.int32(cu_seqlens_k[i])

                k_c4_val = sk // compress_ratio
                k_c4_lens[i] = T.max(1, k_c4_val)

                ps = (sq + cp_size * 2 - 1) // (cp_size * 2)
                partial_seqlens[i] = ps
                elems_per_batch[i] = 2 * ps

                nv_h0 = T.min(ps, T.max(0, sq - ps * cp_rank))
                mirror_rank = cp_size * 2 - 1 - cp_rank
                nv_h1 = T.min(ps, T.max(0, sq - ps * mirror_rank))
                kl_h0 = sk - sq + ps * cp_rank + nv_h0
                kl_h1 = sk - sq + ps * mirror_rank + nv_h1

                delta_sk[i, 0] = kl_h0 // compress_ratio
                delta_sq[i, 0] = kl_h0 // compress_ratio - (kl_h0 - nv_h0) // compress_ratio

                delta_sk[i, 1] = -(kl_h0 // compress_ratio)
                delta_sq[i, 1] = -(kl_h0 // compress_ratio) + (kl_h0 + ps - nv_h0) // compress_ratio

                delta_sk[i, 2] = kl_h1 // compress_ratio
                delta_sq[i, 2] = kl_h1 // compress_ratio - (kl_h1 - nv_h1) // compress_ratio

                sk_arg3 = sq - ps * mirror_rank - nv_h1
                diff3 = sk_arg3 - (ps - nv_h1)
                td_sk3 = T.if_then_else(sk_arg3 >= 0, sk_arg3 // compress_ratio, -((-sk_arg3) // compress_ratio))
                td_diff3 = T.if_then_else(diff3 >= 0, diff3 // compress_ratio, -((-diff3) // compress_ratio))
                delta_sk[i, 3] = td_sk3
                delta_sq[i, 3] = td_sk3 - td_diff3

    return _precompute


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _scatter_tl_ascend(cp_size: int, cp_rank: int, compress_ratio: int) -> None:
    """Ascend AIV scatter with bounded UB tiles and independent work domains."""
    batch_size = T.dynamic('batch_size')
    total_q = T.dynamic('total_q')
    output_len = T.dynamic('output_len')

    block_size = 8192
    vec_len = 64
    num_cores = get_num_vector_cores()
    ratio_is_pow2 = (compress_ratio & (compress_ratio - 1)) == 0
    ratio_shift = compress_ratio.bit_length() - 1

    @T.prim_func
    def _scatter(
        cu_seqlens_q: T.Tensor([batch_size + 1], 'int32'),  # type: ignore
        cu_seqlens_k: T.Tensor([batch_size + 1], 'int32'),  # type: ignore
        cu_seqlens_k_c4: T.Tensor([batch_size + 1], 'int32'),  # type: ignore
        partial_seqlens: T.Tensor([batch_size], 'int32'),  # type: ignore
        cu_elems: T.Tensor([batch_size + 1], 'int32'),  # type: ignore
        total_per_rank: T.Tensor([1], 'int32'),  # type: ignore
        mirror_cp_reverse_index: T.Tensor([total_q], 'int32'),  # type: ignore
        mirror_cp_index: T.Tensor([output_len], 'int32'),  # type: ignore
        ks_out: T.Tensor([output_len], 'int32'),  # type: ignore
        ke_out: T.Tensor([output_len], 'int32'),  # type: ignore
    ) -> None:
        with T.Kernel(num_cores) as pid:
            total_per_rank_value = T.int32(total_per_rank[0])
            reverse_ub = T.alloc_shared((block_size,), 'int32')
            ks_ub = T.alloc_shared((block_size,), 'int32')
            ke_ub = T.alloc_shared((block_size,), 'int32')

            for i_rev0, rank_rev0 in T.Persistent([batch_size, cp_size], num_cores, pid, group_size=1):
                sq_rev0 = T.int32(cu_seqlens_q[i_rev0 + 1]) - T.int32(cu_seqlens_q[i_rev0])
                cu_q_rev0 = T.int32(cu_seqlens_q[i_rev0])
                cu_elem_rev0 = T.int32(cu_elems[i_rev0])
                ps_rev0 = T.int32(partial_seqlens[i_rev0])
                local_start_rev0 = ps_rev0 * rank_rev0
                valid_count_rev0 = T.min(ps_rev0, T.max(0, sq_rev0 - local_start_rev0))

                for block_rev0 in T.serial(T.ceildiv(valid_count_rev0, block_size)):
                    offset_rev0 = block_rev0 * block_size
                    copy_length_rev0 = T.min(block_size, valid_count_rev0 - offset_rev0)
                    with T.SimdVF():
                        mask_rev0 = T.simd.pset(32)
                        for v_rev0 in T.serial(T.ceildiv(copy_length_rev0, vec_len)):
                            value_rev0 = T.simd.vci(
                                rank_rev0 * total_per_rank_value + cu_elem_rev0 + offset_rev0 + v_rev0 * vec_len,
                                'int32',
                            )
                            T.simd.vsts(reverse_ub[v_rev0 * vec_len], value_rev0, mask_rev0)
                    reverse_start_rev0 = cu_q_rev0 + local_start_rev0 + offset_rev0
                    T.copy(
                        reverse_ub[:copy_length_rev0],
                        mirror_cp_reverse_index[reverse_start_rev0 : reverse_start_rev0 + copy_length_rev0],
                    )

            for i_rev1, rank_rev1 in T.Persistent([batch_size, cp_size], num_cores, pid, group_size=1):
                sq_rev1 = T.int32(cu_seqlens_q[i_rev1 + 1]) - T.int32(cu_seqlens_q[i_rev1])
                cu_q_rev1 = T.int32(cu_seqlens_q[i_rev1])
                cu_elem_rev1 = T.int32(cu_elems[i_rev1])
                ps_rev1 = T.int32(partial_seqlens[i_rev1])
                local_start_rev1 = ps_rev1 * (cp_size * 2 - 1 - rank_rev1)
                valid_count_rev1 = T.min(ps_rev1, T.max(0, sq_rev1 - local_start_rev1))

                for block_rev1 in T.serial(T.ceildiv(valid_count_rev1, block_size)):
                    offset_rev1 = block_rev1 * block_size
                    copy_length_rev1 = T.min(block_size, valid_count_rev1 - offset_rev1)
                    with T.SimdVF():
                        mask_rev1 = T.simd.pset(32)
                        for v_rev1 in T.serial(T.ceildiv(copy_length_rev1, vec_len)):
                            value_rev1 = T.simd.vci(
                                rank_rev1 * total_per_rank_value
                                + cu_elem_rev1
                                + ps_rev1
                                + offset_rev1
                                + v_rev1 * vec_len,
                                'int32',
                            )
                            T.simd.vsts(reverse_ub[v_rev1 * vec_len], value_rev1, mask_rev1)
                    reverse_start_rev1 = cu_q_rev1 + local_start_rev1 + offset_rev1
                    T.copy(
                        reverse_ub[:copy_length_rev1],
                        mirror_cp_reverse_index[reverse_start_rev1 : reverse_start_rev1 + copy_length_rev1],
                    )

            # Forward outputs include valid and padded positions for cp_rank.
            for i_fwd0 in T.Persistent([batch_size], num_cores, pid, group_size=1):
                sq_fwd0 = T.int32(cu_seqlens_q[i_fwd0 + 1]) - T.int32(cu_seqlens_q[i_fwd0])
                sk_fwd0 = T.int32(cu_seqlens_k[i_fwd0 + 1]) - T.int32(cu_seqlens_k[i_fwd0])
                cu_q_fwd0 = T.int32(cu_seqlens_q[i_fwd0])
                cu_k_c4_fwd0 = T.int32(cu_seqlens_k_c4[i_fwd0])
                ps_fwd0 = T.int32(partial_seqlens[i_fwd0])
                local_start_fwd0 = ps_fwd0 * cp_rank
                out_start_fwd0 = T.int32(cu_elems[i_fwd0])

                for block_fwd0 in T.serial(T.ceildiv(ps_fwd0, block_size)):
                    offset_fwd0 = block_fwd0 * block_size
                    copy_length_fwd0 = T.min(block_size, ps_fwd0 - offset_fwd0)
                    with T.SimdVF():
                        mask_fwd0 = T.simd.pset(32)
                        neg1_fwd0 = T.simd.vdup(-1, 'int32', mask_fwd0)
                        ks_vec_fwd0 = T.simd.vdup(cu_k_c4_fwd0, 'int32', mask_fwd0)
                        if compress_ratio > 1 and not ratio_is_pow2:
                            ratio_vec_fwd0 = T.simd.vdup(compress_ratio, 'int32', mask_fwd0)
                        for v_fwd0 in T.serial(T.ceildiv(copy_length_fwd0, vec_len)):
                            local_idx_vec_fwd0 = T.simd.vci(local_start_fwd0 + offset_fwd0 + v_fwd0 * vec_len, 'int32')
                            valid_fwd0 = T.simd.vcmps(local_idx_vec_fwd0, sq_fwd0, mask_fwd0, 'lt')
                            reverse_vec_fwd0 = T.simd.vadds(local_idx_vec_fwd0, cu_q_fwd0, mask_fwd0)
                            ke_numerator_fwd0 = T.simd.vadds(local_idx_vec_fwd0, sk_fwd0 - sq_fwd0 + 1, mask_fwd0)
                            if compress_ratio == 1:
                                ke_delta_fwd0 = ke_numerator_fwd0
                            elif ratio_is_pow2:
                                ke_delta_fwd0 = T.simd.vshrs(ke_numerator_fwd0, ratio_shift, mask_fwd0)
                            else:
                                ke_delta_fwd0 = T.simd.vdiv(ke_numerator_fwd0, ratio_vec_fwd0, mask_fwd0)
                            ke_vec_fwd0 = T.simd.vadds(ke_delta_fwd0, cu_k_c4_fwd0, mask_fwd0)
                            T.simd.vsts(
                                reverse_ub[v_fwd0 * vec_len],
                                T.simd.vsel(reverse_vec_fwd0, neg1_fwd0, valid_fwd0),
                                mask_fwd0,
                            )
                            T.simd.vsts(ks_ub[v_fwd0 * vec_len], ks_vec_fwd0, mask_fwd0)
                            T.simd.vsts(
                                ke_ub[v_fwd0 * vec_len], T.simd.vsel(ke_vec_fwd0, ks_vec_fwd0, valid_fwd0), mask_fwd0
                            )

                    T.copy(
                        reverse_ub[:copy_length_fwd0],
                        mirror_cp_index[out_start_fwd0 + offset_fwd0 : out_start_fwd0 + offset_fwd0 + copy_length_fwd0],
                    )
                    T.copy(
                        ks_ub[:copy_length_fwd0],
                        ks_out[out_start_fwd0 + offset_fwd0 : out_start_fwd0 + offset_fwd0 + copy_length_fwd0],
                    )
                    T.copy(
                        ke_ub[:copy_length_fwd0],
                        ke_out[out_start_fwd0 + offset_fwd0 : out_start_fwd0 + offset_fwd0 + copy_length_fwd0],
                    )

            for i_fwd1 in T.Persistent([batch_size], num_cores, pid, group_size=1):
                sq_fwd1 = T.int32(cu_seqlens_q[i_fwd1 + 1]) - T.int32(cu_seqlens_q[i_fwd1])
                sk_fwd1 = T.int32(cu_seqlens_k[i_fwd1 + 1]) - T.int32(cu_seqlens_k[i_fwd1])
                cu_q_fwd1 = T.int32(cu_seqlens_q[i_fwd1])
                cu_k_c4_fwd1 = T.int32(cu_seqlens_k_c4[i_fwd1])
                ps_fwd1 = T.int32(partial_seqlens[i_fwd1])
                local_start_fwd1 = ps_fwd1 * (cp_size * 2 - 1 - cp_rank)
                out_start_fwd1 = T.int32(cu_elems[i_fwd1]) + ps_fwd1

                for block_fwd1 in T.serial(T.ceildiv(ps_fwd1, block_size)):
                    offset_fwd1 = block_fwd1 * block_size
                    copy_length_fwd1 = T.min(block_size, ps_fwd1 - offset_fwd1)
                    with T.SimdVF():
                        mask_fwd1 = T.simd.pset(32)
                        neg1_fwd1 = T.simd.vdup(-1, 'int32', mask_fwd1)
                        ks_vec_fwd1 = T.simd.vdup(cu_k_c4_fwd1, 'int32', mask_fwd1)
                        if compress_ratio > 1 and not ratio_is_pow2:
                            ratio_vec_fwd1 = T.simd.vdup(compress_ratio, 'int32', mask_fwd1)
                        for v_fwd1 in T.serial(T.ceildiv(copy_length_fwd1, vec_len)):
                            local_idx_vec_fwd1 = T.simd.vci(local_start_fwd1 + offset_fwd1 + v_fwd1 * vec_len, 'int32')
                            valid_fwd1 = T.simd.vcmps(local_idx_vec_fwd1, sq_fwd1, mask_fwd1, 'lt')
                            reverse_vec_fwd1 = T.simd.vadds(local_idx_vec_fwd1, cu_q_fwd1, mask_fwd1)
                            ke_numerator_fwd1 = T.simd.vadds(local_idx_vec_fwd1, sk_fwd1 - sq_fwd1 + 1, mask_fwd1)
                            if compress_ratio == 1:
                                ke_delta_fwd1 = ke_numerator_fwd1
                            elif ratio_is_pow2:
                                ke_delta_fwd1 = T.simd.vshrs(ke_numerator_fwd1, ratio_shift, mask_fwd1)
                            else:
                                ke_delta_fwd1 = T.simd.vdiv(ke_numerator_fwd1, ratio_vec_fwd1, mask_fwd1)
                            ke_vec_fwd1 = T.simd.vadds(ke_delta_fwd1, cu_k_c4_fwd1, mask_fwd1)
                            T.simd.vsts(
                                reverse_ub[v_fwd1 * vec_len],
                                T.simd.vsel(reverse_vec_fwd1, neg1_fwd1, valid_fwd1),
                                mask_fwd1,
                            )
                            T.simd.vsts(ks_ub[v_fwd1 * vec_len], ks_vec_fwd1, mask_fwd1)
                            T.simd.vsts(
                                ke_ub[v_fwd1 * vec_len], T.simd.vsel(ke_vec_fwd1, ks_vec_fwd1, valid_fwd1), mask_fwd1
                            )

                    T.copy(
                        reverse_ub[:copy_length_fwd1],
                        mirror_cp_index[out_start_fwd1 + offset_fwd1 : out_start_fwd1 + offset_fwd1 + copy_length_fwd1],
                    )
                    T.copy(
                        ks_ub[:copy_length_fwd1],
                        ks_out[out_start_fwd1 + offset_fwd1 : out_start_fwd1 + offset_fwd1 + copy_length_fwd1],
                    )
                    T.copy(
                        ke_ub[:copy_length_fwd1],
                        ke_out[out_start_fwd1 + offset_fwd1 : out_start_fwd1 + offset_fwd1 + copy_length_fwd1],
                    )

    return _scatter
