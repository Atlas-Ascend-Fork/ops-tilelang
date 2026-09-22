import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit()
def _mhc_reduce_partials_and_rmsnorm_forward_kernel(
    out_mul_splitted,
    sqrsum_splitted,
    out_mul,
    sqrsum,
    out,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
    num_vec_cores: int,
):
    """Main-compatible reference reduction with caller-owned diagnostic outputs."""
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    n_rms_group = T.const('n_rms_group')
    assert n_rms_group == 1, 'Ascend reduce/rmsnorm fwd currently supports one RMS group'

    out_mul_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32]
    out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32]
    out: T.Tensor[(num_tokens, mhc_mult3), T.float32]

    with T.Kernel(num_vec_cores) as core_id:
        partial_mul_ub = T.alloc_shared((n_splits, mhc_mult3), T.float32)
        partial_sqrsum_ub = T.alloc_shared((n_splits,), T.float32)
        raw_mix_ub = T.alloc_shared((mhc_mult3,), T.float32)
        sqrsum_ub = T.alloc_shared((1,), T.float32)
        out_ub = T.alloc_shared((mhc_mult3,), T.float32)

        for pid in T.Persistent([num_tokens], num_vec_cores, core_id, group_size=1, num_stages=0):
            for split in T.serial(n_splits):
                T.copy(
                    out_mul_splitted[split, pid, 0, 0:mhc_mult3],
                    partial_mul_ub[split, 0:mhc_mult3],
                )
                partial_sqrsum_ub[split] = sqrsum_splitted[split, pid, 0]

            with T.SimdVF():
                zero = S.vdup(0.0, T.float32)
                one = S.vdup(1.0, T.float32)
                rms = S.alloc_var(T.float32)
                rms = zero
                for split in range(n_splits):
                    rms = S.vadd(rms, S.vld(partial_sqrsum_ub[split], dist='BRC_B32'))
                S.vsts(sqrsum_ub[0], rms, dist='ONEPT_B32')
                rms = S.vadds(S.vmuls(rms, 1.0 / rms_group_size), rms_eps)
                rms = S.vdiv(one, S.vsqrt(rms))

                mix = S.alloc_var(T.float32)
                for j in range(mhc_mult3):
                    mix = zero
                    for split in range(n_splits):
                        mix = S.vadd(mix, S.vld(partial_mul_ub[split, j], dist='BRC_B32'))
                    S.vsts(raw_mix_ub[j], mix, dist='ONEPT_B32')
                    S.vsts(out_ub[j], S.vmul(mix, rms), dist='ONEPT_B32')

            T.copy(raw_mix_ub, out_mul[pid, 0, 0])
            T.copy(sqrsum_ub, sqrsum[pid, 0])
            T.copy(out_ub, out[pid, 0])


_VEC = 64
_TOKEN_TILE = 64
_TOKEN_GROUP = 64
_PREFIX_CAPACITY = max(
    2 * _TOKEN_TILE,
    (_TOKEN_GROUP + _TOKEN_TILE - 1) // _TOKEN_TILE * _TOKEN_TILE,
)


def _get_balanced_group_core_count(num_tokens: int) -> int:
    """Use the largest available core count that evenly owns token groups."""
    total_groups = num_tokens // _TOKEN_GROUP
    num_cores = min(get_num_vector_cores(), total_groups)
    while total_groups % num_cores != 0:
        num_cores -= 1
    return num_cores


@tilelang.jit(pass_configs={'tl.enable_fast_math': True})
def mhc_pre_big_fuse_fwd_asc(
    gemm_out_mul,
    gemm_out_sqrsum,
    mhc_scale,
    mhc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    h_blk: int,
    residual_token_block: int,
    residual_num_stages: int,
    use_fma_affine: bool,
    static_num_tokens: int,
    n_splits: int = 16,
):
    """Ascend implementation of the fused MHC pre kernel.

    The arithmetic order intentionally matches the CUDA implementation: split
    partials are accumulated serially, Sinkhorn alternates row and column
    normalization, and the residual mix is accumulated in MHC-index order.
    """
    # Common production shapes are JIT constants.  Passing zero keeps the
    # dynamic fallback for arbitrary/ragged token counts.
    num_tokens = static_num_tokens if static_num_tokens > 0 else T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')
    mhc_mult3 = T.const('mhc_mult3')
    hidden_size = T.const('hidden_size')

    gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32]
    gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    mhc_base: T.Tensor[(mhc_mult3,), T.float32]
    residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16]
    post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32]
    layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16]

    # The 8192-token layout assigns complete 64-token groups round-robin.
    # Choose the largest supported core count that divides those groups so
    # every wave is full and balanced on devices with different core counts.
    n_cores = _get_balanced_group_core_count(static_num_tokens) if static_num_tokens == 8192 else get_num_vector_cores()
    num_chunks = h_blk // _VEC
    num_hidden_tiles = hidden_size // h_blk
    token_blocks_per_group = _TOKEN_GROUP // residual_token_block
    use_wave16_mapping = static_num_tokens == 8192 and residual_token_block == 16

    with T.Kernel(n_cores) as core_id:
        T.assume(core_id >= 0)
        T.assume(core_id < n_cores)
        # The fixed-size prefix used to broadcast every logical scalar to all
        # 64 SIMD lanes for one token.  Keep the same arithmetic order, but put
        # up to 64 independent tokens in separate lanes.  The remaining lanes
        # are masked to zero, while the hidden-size apply-mix below still uses
        # the full vector width for one token at a time.
        sqrsum_ub = T.alloc_shared((_VEC,), T.float32)
        scale_ub = T.alloc_shared((8,), T.float32)
        base_ub = T.alloc_shared((mhc_mult3,), T.float32)
        mixes_ub = T.alloc_shared((mhc_mult3, _VEC), T.float32)
        if use_wave16_mapping:
            gemm_input_ub = T.alloc_shared((_TOKEN_TILE * mhc_mult3,), T.float32)
        pre_mix_ub = T.alloc_shared((mhc_mult, _PREFIX_CAPACITY), T.float32)
        post_mix_ub = T.alloc_shared((mhc_mult, _VEC), T.float32)
        comb_mix_ub = T.alloc_shared((mhc_mult * mhc_mult, _VEC), T.float32)
        residual_ub = T.alloc_shared((residual_token_block, mhc_mult, h_blk), T.bfloat16)
        layer_input_ub = T.alloc_shared((residual_token_block, h_blk), T.bfloat16)
        T.annotate_buffer_versions({residual_ub: residual_num_stages, layer_input_ub: residual_num_stages})

        # These tensors are invariant across tokens.  Load them once per core
        # instead of issuing the same small multi-core GM read for every token.
        T.copy(mhc_scale, scale_ub[0:3])
        T.copy(mhc_base, base_ub)

        if static_num_tokens > 0:
            # Static shapes still need remainder-aware distribution: for
            # example, 512 tokens on 72 vector cores is 8 tokens on the first
            # eight cores and 7 on the rest.  Plain integer division used to
            # leave the final eight tokens unwritten.
            tokens_per_core = static_num_tokens // n_cores
            extra_tokens = static_num_tokens % n_cores
            core_tokens = T.alloc_var(T.int32)
            core_tokens = tokens_per_core
            if core_id < extra_tokens:
                core_tokens += 1
            core_token_start = core_id * tokens_per_core + T.min(core_id, extra_tokens)
            token_groups = T.ceildiv(core_tokens, _TOKEN_GROUP)
        else:
            tokens_per_core = num_tokens // n_cores
            extra_tokens = num_tokens % n_cores
            core_tokens = T.alloc_var(T.int32)
            core_tokens = tokens_per_core
            if core_id < extra_tokens:
                core_tokens += 1
            core_token_start = core_id * tokens_per_core + T.min(core_id, extra_tokens)
            token_groups = T.ceildiv(core_tokens, _TOKEN_GROUP)

        for token_group in T.serial(token_groups):
            if static_num_tokens == 8192:
                # Round-robin 64-token groups across vector cores.  Each wave
                # now covers one contiguous 4096-token GM range instead of
                # making neighboring cores start 128 tokens apart.
                group_token_start = (token_group * n_cores + core_id) * _TOKEN_GROUP
            else:
                group_token_start = core_token_start + token_group * _TOKEN_GROUP
            if static_num_tokens > 0:
                group_valid_tokens = core_tokens if core_tokens < _TOKEN_GROUP else _TOKEN_GROUP
                token_subbatches = 1
            else:
                group_valid_tokens = T.min(
                    _TOKEN_GROUP,
                    core_tokens - token_group * _TOKEN_GROUP,
                )
                token_subbatches = T.ceildiv(group_valid_tokens, _TOKEN_TILE)

            # Fill the SIMD-sized prefix tiles first, then pipeline one larger
            # residual group.  This removes repeated MTE2/VEC pipeline flushes
            # without introducing a shape branch or another kernel.
            for token_subbatch in T.serial(token_subbatches):
                batch_token_start = group_token_start + token_subbatch * _TOKEN_TILE
                if static_num_tokens > 0:
                    valid_tokens = group_valid_tokens
                else:
                    valid_tokens = T.min(
                        _TOKEN_TILE,
                        T.max(
                            0,
                            group_valid_tokens - token_subbatch * _TOKEN_TILE,
                        ),
                    )

                # The wave16 layout consists of four disjoint but individually
                # contiguous 16-token ranges.  Move each range with MTE2 first,
                # then transpose inside UB instead of issuing scalar GM loads
                # from all 1024 SIMT threads.
                if use_wave16_mapping:
                    for input_block in T.serial(token_blocks_per_group):
                        input_token_start = (
                            (token_group * token_blocks_per_group + input_block) * n_cores + core_id
                        ) * residual_token_block
                        input_lane_start = input_block * residual_token_block
                        T.copy(
                            gemm_out_mul[
                                0:1,
                                input_token_start : input_token_start + residual_token_block,
                                0:mhc_mult3,
                            ],
                            gemm_input_ub[
                                input_lane_start * mhc_mult3 : (input_lane_start + residual_token_block) * mhc_mult3
                            ],
                        )
                        T.copy(
                            gemm_out_sqrsum[
                                0:1,
                                input_token_start : input_token_start + residual_token_block,
                            ],
                            sqrsum_ub[input_lane_start : input_lane_start + residual_token_block],
                        )

                # Transpose token-major input to element-major UB, where one
                # SIMD register represents 64 tokens.  Wave16 reads the staged
                # UB above; the other shapes retain their coalesced GM path.
                with T.SimtVF(threads=1024):
                    load_thread = T.get_thread_binding()
                    for load_round in T.serial(T.ceildiv(_TOKEN_TILE * mhc_mult3, 1024)):
                        load_index = load_thread + load_round * 1024
                        if load_index < _TOKEN_TILE * mhc_mult3:
                            token_lane = load_index // mhc_mult3
                            mix_index = load_index % mhc_mult3
                            if use_wave16_mapping:
                                mixes_ub[mix_index, token_lane] = gemm_input_ub[load_index]
                            elif static_num_tokens == 8192:
                                pid = batch_token_start + token_lane
                                mix_sum = T.alloc_var(T.float32, 0.0)
                                for split in T.serial(n_splits):
                                    mix_sum += gemm_out_mul[split, pid, mix_index]
                                mixes_ub[mix_index, token_lane] = mix_sum
                            else:
                                if token_lane < valid_tokens:
                                    pid = batch_token_start + token_lane
                                    mix_sum = T.alloc_var(T.float32, 0.0)
                                    for split in T.serial(n_splits):
                                        mix_sum += gemm_out_mul[split, pid, mix_index]
                                    mixes_ub[mix_index, token_lane] = mix_sum
                                else:
                                    mixes_ub[mix_index, token_lane] = 0.0

                    # Keep sqrsum on one contiguous thread range instead of
                    # the sparse ``mix_index == 0`` branch in the mix loop.
                    if load_thread < _TOKEN_TILE and not use_wave16_mapping:
                        token_lane = load_thread
                        if static_num_tokens == 8192:
                            pid = batch_token_start + token_lane
                            rms_sum = T.alloc_var(T.float32, 0.0)
                            for split in T.serial(n_splits):
                                rms_sum += gemm_out_sqrsum[split, pid]
                            sqrsum_ub[token_lane] = rms_sum
                        else:
                            if token_lane < valid_tokens:
                                pid = batch_token_start + token_lane
                                rms_sum = T.alloc_var(T.float32, 0.0)
                                for split in T.serial(n_splits):
                                    rms_sum += gemm_out_sqrsum[split, pid]
                                sqrsum_ub[token_lane] = rms_sum
                            else:
                                sqrsum_ub[token_lane] = 0.0

                with T.SimdVF():
                    zero1 = S.vdup(0.0, T.float32)
                    one1 = S.vdup(1.0, T.float32)
                    rms = S.alloc_var(T.float32)
                    if static_num_tokens == 8192:
                        rms = S.vld(sqrsum_ub[0])
                    else:
                        lane_id = S.vci(T.int32(0), T.int32)
                        valid_tokens_vec = S.vdup(valid_tokens, T.int32)
                        valid_mask = S.vcmp(lane_id, valid_tokens_vec, op='lt')
                        rms = S.vsel(S.vld(sqrsum_ub[0]), zero1, valid_mask)
                    rms = S.vadds(
                        S.vmuls(rms, 1.0 / (mhc_mult * hidden_size)),
                        rms_eps,
                    )
                    # The fused kernel uses fast division for sigmoid and
                    # Sinkhorn to match their standalone kernels.  Norm-fn,
                    # however, keeps precise fp32 division.  Reproduce PTO's
                    # vdiv_precise_f32 correction locally so this RMS inverse
                    # remains bit-identical without splitting another kernel.
                    rms_denom = S.vsqrt(rms)
                    div_z0 = S.vdiv(one1, rms_denom)
                    div_neg_rhs = S.vmuls(rms_denom, -1.0)

                    div_r0 = S.alloc_var(T.float32)
                    div_r0 = one1
                    S.vmula(div_r0, div_z0, div_neg_rhs)

                    div_z_i32 = T.reinterpret(div_z0, 'int32x64')
                    div_z_pre = T.reinterpret(S.vadds(div_z_i32, -1), 'float32x64')
                    div_z_next = T.reinterpret(S.vadds(div_z_i32, 1), 'float32x64')

                    div_r_pre = S.alloc_var(T.float32)
                    div_r_pre = one1
                    S.vmula(div_r_pre, div_z_pre, div_neg_rhs)
                    div_r_next = S.alloc_var(T.float32)
                    div_r_next = one1
                    S.vmula(div_r_next, div_z_next, div_neg_rhs)

                    div_r0_abs = S.vabs(div_r0)
                    div_r_pre_abs = S.vabs(div_r_pre)
                    div_r_next_abs = S.vabs(div_r_next)
                    div_keep_current = S.vcmp(div_r0_abs, div_r_pre_abs, op='lt')
                    div_best_r = S.vsel(div_r0_abs, div_r_pre_abs, div_keep_current)
                    div_best_z = S.vsel(div_z0, div_z_pre, div_keep_current)
                    div_better_next = S.vcmp(div_r_next_abs, div_best_r, op='lt')
                    rms = S.vsel(div_z_next, div_best_z, div_better_next)
                    scale_pre = S.vld(scale_ub[0], dist='BRC_B32')
                    scale_post = S.vld(scale_ub[1], dist='BRC_B32')
                    scale_comb = S.vld(scale_ub[2], dist='BRC_B32')

                    for j in range(mhc_mult):
                        if static_num_tokens == 8192:
                            mix_pre = S.vmul(S.vld(mixes_ub[j, 0]), rms)
                        else:
                            mix_pre = S.vmul(
                                S.vsel(
                                    S.vld(mixes_ub[j, 0]),
                                    zero1,
                                    valid_mask,
                                ),
                                rms,
                            )
                        pre_base = S.vld(base_ub[j], dist='BRC_B32')
                        z_pre = S.alloc_var(T.float32)
                        if use_fma_affine:
                            z_pre = mix_pre
                            S.vmadd(z_pre, scale_pre, pre_base)
                        else:
                            z_pre = S.vadd(S.vmul(mix_pre, scale_pre), pre_base)
                        pre = S.vadds(
                            S.vdiv(
                                one1,
                                S.vadds(S.vexpdif(zero1, z_pre), 1.0),
                            ),
                            mhc_pre_eps,
                        )
                        if static_num_tokens == 8192:
                            mix_post = S.vmul(S.vld(mixes_ub[j + mhc_mult, 0]), rms)
                        else:
                            mix_post = S.vmul(
                                S.vsel(
                                    S.vld(mixes_ub[j + mhc_mult, 0]),
                                    zero1,
                                    valid_mask,
                                ),
                                rms,
                            )
                        post_base = S.vld(base_ub[j + mhc_mult], dist='BRC_B32')
                        z_post = S.alloc_var(T.float32)
                        if use_fma_affine:
                            z_post = mix_post
                            S.vmadd(z_post, scale_post, post_base)
                        else:
                            z_post = S.vadd(S.vmul(mix_post, scale_post), post_base)
                        post = S.vmuls(
                            S.vdiv(
                                one1,
                                S.vadds(S.vexpdif(zero1, z_post), 1.0),
                            ),
                            mhc_post_mult_value,
                        )
                        S.vsts(
                            pre_mix_ub[j, token_subbatch * _TOKEN_TILE],
                            pre,
                        )
                        S.vsts(post_mix_ub[j, 0], post)

                    for j in range(mhc_mult):
                        for k in range(mhc_mult):
                            index = j * mhc_mult + k
                            offset = index + mhc_mult * 2
                            if static_num_tokens == 8192:
                                mix_comb = S.vmul(S.vld(mixes_ub[offset, 0]), rms)
                            else:
                                mix_comb = S.vmul(
                                    S.vsel(
                                        S.vld(mixes_ub[offset, 0]),
                                        zero1,
                                        valid_mask,
                                    ),
                                    rms,
                                )
                            comb_base = S.vld(base_ub[offset], dist='BRC_B32')
                            comb_value = S.alloc_var(T.float32)
                            if use_fma_affine:
                                comb_value = mix_comb
                                S.vmadd(comb_value, scale_comb, comb_base)
                            else:
                                comb_value = S.vadd(S.vmul(mix_comb, scale_comb), comb_base)
                            S.vsts(comb_mix_ub[index, 0], comb_value)

                # Keep one 4x4 matrix in vector registers for a complete
                # row/column pair.  The arithmetic order is unchanged, but the
                # intermediate row result no longer makes a round trip through
                # UB before column normalization.
                with T.SimdVF():
                    zero3 = S.vdup(0.0, T.float32)
                    one3 = S.vdup(1.0, T.float32)
                    initial_eps3 = S.vdup(mhc_sinkhorn_eps, T.float32)
                    matrix_values = S.alloc_local((mhc_mult * mhc_mult,), T.float32)
                    row_max = S.alloc_var(T.float32)
                    reduction = S.alloc_var(T.float32)
                    recip = S.alloc_var(T.float32)
                    for index in T.unroll(mhc_mult * mhc_mult, explicit=True):
                        matrix_values[index] = S.vld(comb_mix_ub[index, 0])
                    for j in T.unroll(mhc_mult, explicit=True):
                        row_offset = j * mhc_mult
                        row_max = matrix_values[row_offset]
                        for k in T.unroll(mhc_mult - 1, explicit=True):
                            row_max = S.vmax(row_max, matrix_values[row_offset + k + 1])
                        reduction = zero3
                        for k in T.unroll(mhc_mult, explicit=True):
                            matrix_values[row_offset + k] = S.vexpdif(matrix_values[row_offset + k], row_max)
                            reduction = S.vadd(reduction, matrix_values[row_offset + k])
                        recip = S.vdiv(one3, reduction)
                        for k in T.unroll(mhc_mult, explicit=True):
                            matrix_values[row_offset + k] = S.vadds(
                                S.vmul(matrix_values[row_offset + k], recip),
                                mhc_sinkhorn_eps,
                            )
                    for k in T.unroll(mhc_mult, explicit=True):
                        reduction = initial_eps3
                        for j in T.unroll(mhc_mult, explicit=True):
                            reduction = S.vadd(
                                reduction,
                                matrix_values[j * mhc_mult + k],
                            )
                        recip = S.vdiv(one3, reduction)
                        for j in T.unroll(mhc_mult, explicit=True):
                            matrix_values[j * mhc_mult + k] = S.vmul(matrix_values[j * mhc_mult + k], recip)
                    for _ in T.serial(sinkhorn_repeat - 1):
                        for j in T.unroll(mhc_mult, explicit=True):
                            row_offset = j * mhc_mult
                            reduction = initial_eps3
                            for k in T.unroll(mhc_mult, explicit=True):
                                reduction = S.vadd(
                                    reduction,
                                    matrix_values[row_offset + k],
                                )
                            recip = S.vdiv(one3, reduction)
                            for k in T.unroll(mhc_mult, explicit=True):
                                matrix_values[row_offset + k] = S.vmul(matrix_values[row_offset + k], recip)
                        for k in T.unroll(mhc_mult, explicit=True):
                            reduction = initial_eps3
                            for j in T.unroll(mhc_mult, explicit=True):
                                reduction = S.vadd(
                                    reduction,
                                    matrix_values[j * mhc_mult + k],
                                )
                            recip = S.vdiv(one3, reduction)
                            for j in T.unroll(mhc_mult, explicit=True):
                                matrix_values[j * mhc_mult + k] = S.vmul(matrix_values[j * mhc_mult + k], recip)
                    for index in T.unroll(mhc_mult * mhc_mult, explicit=True):
                        S.vsts(comb_mix_ub[index, 0], matrix_values[index])

                # Transpose the small token-batched outputs back to token-major GM.
                with T.SimtVF(threads=1024):
                    store_thread = T.get_thread_binding()
                    token_lane = store_thread // 16
                    matrix_index = store_thread % 16
                    if token_lane < valid_tokens:
                        if use_wave16_mapping:
                            pid = (
                                (token_group * token_blocks_per_group + token_lane // residual_token_block) * n_cores
                                + core_id
                            ) * residual_token_block + token_lane % residual_token_block
                        else:
                            pid = batch_token_start + token_lane
                        comb_mix[pid, matrix_index] = comb_mix_ub[matrix_index, token_lane]
                        if matrix_index < mhc_mult:
                            post_mix[pid, matrix_index] = post_mix_ub[matrix_index, token_lane]

            # Multi-buffer the only large GM path at hidden-tile granularity.
            # Keep token-block and hidden-tile as separate loop coordinates.
            # This avoids a scalar divide/modulo pair on every residual tile;
            # the inner hidden loop is still a four-stage MTE2/VEC/MTE3 pipe.
            residual_token_blocks = group_valid_tokens // residual_token_block
            for token_block_lane in T.serial(residual_token_blocks):
                token_lane_start = token_block_lane * residual_token_block
                if use_wave16_mapping:
                    token_start = (
                        (token_group * token_blocks_per_group + token_block_lane) * n_cores + core_id
                    ) * residual_token_block
                else:
                    token_start = group_token_start + token_lane_start
                for hidden_tile in T.Pipelined(
                    num_hidden_tiles,
                    num_stages=residual_num_stages,
                ):
                    hidden_start = hidden_tile * h_blk
                    T.copy(
                        residual[
                            token_start : token_start + residual_token_block,
                            0:mhc_mult,
                            hidden_start : hidden_start + h_blk,
                        ],
                        residual_ub,
                    )

                    with T.SimdVF():
                        pre_reg = S.alloc_local((mhc_mult,), T.float32)
                        residual_reg = S.alloc_local((mhc_mult,), T.float32)
                        acc = S.alloc_var(T.float32)
                        for token_offset in T.serial(residual_token_block):
                            token_lane = token_lane_start + token_offset
                            for m in T.unroll(mhc_mult, explicit=True):
                                pre_reg[m] = S.vld(
                                    pre_mix_ub[m, token_lane],
                                    dist='BRC_B32',
                                )
                            for chunk in range(num_chunks):
                                col = chunk * _VEC
                                for m in T.unroll(mhc_mult, explicit=True):
                                    residual_reg[m] = S.vcvt(
                                        S.vld(
                                            residual_ub[token_offset, m, col],
                                            dist='UNPK_B16',
                                        ),
                                        T.float32,
                                        part=0,
                                    )
                                acc = S.vmul(residual_reg[0], pre_reg[0])
                                for m in T.unroll(mhc_mult - 1, explicit=True):
                                    S.vmula(
                                        acc,
                                        residual_reg[m + 1],
                                        pre_reg[m + 1],
                                    )
                                S.vsts(
                                    layer_input_ub[token_offset, col],
                                    S.vcvt(acc, T.bfloat16),
                                    dist='PK_B32',
                                )

                    T.copy(
                        layer_input_ub,
                        layer_input[
                            token_start : token_start + residual_token_block,
                            hidden_start : hidden_start + h_blk,
                        ],
                    )


# Keep the historical symbol for local profiling tools while exposing the
# upstream-compatible forward name to the shared launcher.
mhc_pre_big_fuse_asc = mhc_pre_big_fuse_fwd_asc
