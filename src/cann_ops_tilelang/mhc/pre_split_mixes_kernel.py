import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

_VEC = 64
_FWD_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}


@tilelang.jit(pass_configs=_FWD_PASS_CONFIGS)
def mhc_pre_split_mixes_forward_kernel(
    input_mixes,
    mhc_scale,
    mhc_base,
    pre_layer_mix,
    post_layer_mix,
    comb_res_mix,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    static_num_tokens: int,
    token_block_size: int,
    pre_vecs: int,
    comb_vecs: int,
    full_tiles: bool,
    num_vec_cores: int,
):
    """Ascend forward with shape-aware token tiling and a static 8192 path."""

    num_tokens = static_num_tokens if static_num_tokens else T.dynamic('num_tokens')
    mhc_mult = 4
    mhc_mult2 = 16
    mhc_mult3 = 24

    input_mixes: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    mhc_base: T.Tensor[(mhc_mult3,), T.float32]
    pre_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    comb_res_mix: T.Tensor[(num_tokens, mhc_mult2), T.float32]

    n_cores = 32 if static_num_tokens else num_vec_cores
    with T.Kernel(n_cores) as core_id:
        input_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        pre_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        post_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        comb_ub = T.alloc_shared((token_block_size, mhc_mult2), T.float32)
        scale_ub = T.alloc_shared((8,), T.float32)
        base_ub = T.alloc_shared((mhc_mult3,), T.float32)
        pre_index_ub = T.alloc_shared((_VEC,), T.uint32)
        comb_index_ub = T.alloc_shared((_VEC,), T.uint32)
        pre_base_pattern_ub = T.alloc_shared((_VEC,), T.float32)
        post_base_pattern_ub = T.alloc_shared((_VEC,), T.float32)
        comb_base_pattern_ub = T.alloc_shared((_VEC,), T.float32)

        if static_num_tokens:
            T.copy(mhc_scale, scale_ub[0:3])
            T.copy(mhc_base, base_ub[0:mhc_mult3])
            with T.SimtVF(threads=_VEC):
                for lane in T.Parallel(_VEC):
                    pre_component = lane % mhc_mult
                    comb_component = lane % mhc_mult2
                    pre_index_ub[lane] = T.cast(
                        (lane // mhc_mult) * mhc_mult3 + pre_component,
                        T.uint32,
                    )
                    comb_index_ub[lane] = T.cast(
                        (lane // mhc_mult2) * mhc_mult3 + comb_component,
                        T.uint32,
                    )
                    pre_base_pattern_ub[lane] = base_ub[pre_component]
                    post_base_pattern_ub[lane] = base_ub[mhc_mult + pre_component]
                    comb_base_pattern_ub[lane] = base_ub[mhc_mult * 2 + comb_component]
        else:
            if core_id < T.ceildiv(num_tokens, token_block_size):
                T.copy(mhc_scale, scale_ub[0:3])
                T.copy(mhc_base, base_ub[0:mhc_mult3])
                with T.SimtVF(threads=_VEC):
                    for lane in T.Parallel(_VEC):
                        pre_component = lane % mhc_mult
                        comb_component = lane % mhc_mult2
                        pre_index_ub[lane] = T.cast(
                            (lane // mhc_mult) * mhc_mult3 + pre_component,
                            T.uint32,
                        )
                        comb_index_ub[lane] = T.cast(
                            (lane // mhc_mult2) * mhc_mult3 + comb_component,
                            T.uint32,
                        )
                        pre_base_pattern_ub[lane] = base_ub[pre_component]
                        post_base_pattern_ub[lane] = base_ub[mhc_mult + pre_component]
                        comb_base_pattern_ub[lane] = base_ub[mhc_mult * 2 + comb_component]

        for tile_id in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            n_cores,
            core_id,
            group_size=1,
            num_stages=2,
        ):
            token_offset = tile_id * token_block_size
            valid_tokens = token_block_size if full_tiles else T.min(token_block_size, num_tokens - token_offset)
            T.copy(
                input_mixes[token_offset : token_offset + valid_tokens, 0:mhc_mult3],
                input_ub[0:valid_tokens, 0:mhc_mult3],
            )

            with T.SimdVF():
                zeros = S.vdup(0.0, T.float32)
                ones = S.vdup(1.0, T.float32)
                if full_tiles:
                    full_mask = S.pset(32, 'PAT_ALL')
                else:
                    lanes = S.vci(0, T.int32)

                if token_block_size == 16:
                    pre_valid = (
                        full_mask if full_tiles else S.vcmp(lanes, S.vdup(valid_tokens * mhc_mult, T.int32), op='lt')
                    )
                    pre_index = S.vld(pre_index_ub[0])
                    pre_x_raw = S.vgather2(input_ub[0, 0], pre_index, pre_valid)
                    post_x_raw = S.vgather2(input_ub[0, 0], S.vadds(pre_index, mhc_mult), pre_valid)
                    pre_x = pre_x_raw if full_tiles else S.vsel(pre_x_raw, zeros, pre_valid)
                    post_x = post_x_raw if full_tiles else S.vsel(post_x_raw, zeros, pre_valid)
                    pre_affine = S.vadd(
                        S.vmul(pre_x, S.vld(scale_ub[0], dist='BRC_B32')),
                        S.vld(pre_base_pattern_ub[0]),
                    )
                    post_affine = S.vadd(
                        S.vmul(post_x, S.vld(scale_ub[1], dist='BRC_B32')),
                        S.vld(post_base_pattern_ub[0]),
                    )
                    pre_sigmoid = S.vdiv(ones, S.vadds(S.vexpdif(zeros, pre_affine), 1.0))
                    post_sigmoid = S.vdiv(ones, S.vadds(S.vexpdif(zeros, post_affine), 1.0))
                    S.vsts(pre_ub[0, 0], S.vadds(pre_sigmoid, mhc_pre_eps), pre_valid)
                    S.vsts(post_ub[0, 0], S.vmuls(post_sigmoid, mhc_post_mult_value), pre_valid)

                    comb_valid_elements = valid_tokens * mhc_mult2
                    comb_index = S.vld(comb_index_ub[0])
                    for vec_id in T.serial(comb_vecs):
                        lane_offset = vec_id * _VEC
                        comb_valid = (
                            full_mask
                            if full_tiles
                            else S.vcmp(
                                S.vci(T.int32(lane_offset), T.int32),
                                S.vdup(comb_valid_elements, T.int32),
                                op='lt',
                            )
                        )
                        input_offset = vec_id * (_VEC // mhc_mult2) * mhc_mult3 + mhc_mult * 2
                        comb_x_raw = S.vgather2(
                            input_ub[0, 0],
                            S.vadds(comb_index, input_offset),
                            comb_valid,
                        )
                        comb_x = comb_x_raw if full_tiles else S.vsel(comb_x_raw, zeros, comb_valid)
                        comb_affine = S.vadd(
                            S.vmul(comb_x, S.vld(scale_ub[2], dist='BRC_B32')),
                            S.vld(comb_base_pattern_ub[0]),
                        )
                        S.vsts(
                            comb_ub[vec_id * (_VEC // mhc_mult2), 0],
                            comb_affine,
                            comb_valid,
                        )
                else:
                    pre_index = S.vld(pre_index_ub[0])
                    scale_pre = S.vld(scale_ub[0], dist='BRC_B32')
                    scale_post = S.vld(scale_ub[1], dist='BRC_B32')
                    pre_base = S.vld(pre_base_pattern_ub[0])
                    post_base = S.vld(post_base_pattern_ub[0])
                    pre_valid_elements = valid_tokens * mhc_mult
                    for vec_id in T.serial(pre_vecs):
                        lane_offset = vec_id * _VEC
                        pre_valid = (
                            full_mask
                            if full_tiles
                            else S.vcmp(
                                S.vadds(lanes, lane_offset),
                                S.vdup(pre_valid_elements, T.int32),
                                op='lt',
                            )
                        )
                        input_offset = vec_id * (_VEC // mhc_mult) * mhc_mult3
                        group_index = S.vadds(pre_index, input_offset)
                        pre_x_raw = S.vgather2(input_ub[0, 0], group_index, pre_valid)
                        post_x_raw = S.vgather2(
                            input_ub[0, 0],
                            S.vadds(group_index, mhc_mult),
                            pre_valid,
                        )
                        pre_x = pre_x_raw if full_tiles else S.vsel(pre_x_raw, zeros, pre_valid)
                        post_x = post_x_raw if full_tiles else S.vsel(post_x_raw, zeros, pre_valid)
                        pre_affine = S.alloc_var(T.float32)
                        post_affine = S.alloc_var(T.float32)
                        pre_affine = pre_x
                        post_affine = post_x
                        S.vmadd(pre_affine, scale_pre, pre_base, pre_valid)
                        S.vmadd(post_affine, scale_post, post_base, pre_valid)
                        pre_sigmoid = S.vdiv(ones, S.vadds(S.vexpdif(zeros, pre_affine), 1.0))
                        post_sigmoid = S.vdiv(ones, S.vadds(S.vexpdif(zeros, post_affine), 1.0))
                        S.vsts(
                            pre_ub[vec_id * (_VEC // mhc_mult), 0],
                            S.vadds(pre_sigmoid, mhc_pre_eps),
                            pre_valid,
                        )
                        S.vsts(
                            post_ub[vec_id * (_VEC // mhc_mult), 0],
                            S.vmuls(post_sigmoid, mhc_post_mult_value),
                            pre_valid,
                        )

                    comb_valid_elements = valid_tokens * mhc_mult2
                    comb_index = S.vld(comb_index_ub[0])
                    scale_comb = S.vld(scale_ub[2], dist='BRC_B32')
                    comb_base = S.vld(comb_base_pattern_ub[0])
                    for vec_id in T.serial(comb_vecs):
                        lane_offset = vec_id * _VEC
                        comb_valid = (
                            full_mask
                            if full_tiles
                            else S.vcmp(
                                S.vadds(lanes, lane_offset),
                                S.vdup(comb_valid_elements, T.int32),
                                op='lt',
                            )
                        )
                        input_offset = vec_id * (_VEC // mhc_mult2) * mhc_mult3 + mhc_mult * 2
                        comb_x_raw = S.vgather2(
                            input_ub[0, 0],
                            S.vadds(comb_index, input_offset),
                            comb_valid,
                        )
                        comb_x = comb_x_raw if full_tiles else S.vsel(comb_x_raw, zeros, comb_valid)
                        comb_affine = S.alloc_var(T.float32)
                        comb_affine = comb_x
                        S.vmadd(comb_affine, scale_comb, comb_base, comb_valid)
                        S.vsts(
                            comb_ub[vec_id * (_VEC // mhc_mult2), 0],
                            comb_affine,
                            comb_valid,
                        )

            T.copy(
                pre_ub[0:valid_tokens, 0:mhc_mult],
                pre_layer_mix[token_offset : token_offset + valid_tokens, 0:mhc_mult],
            )
            T.copy(
                post_ub[0:valid_tokens, 0:mhc_mult],
                post_layer_mix[token_offset : token_offset + valid_tokens, 0:mhc_mult],
            )
            T.copy(
                comb_ub[0:valid_tokens, 0:mhc_mult2],
                comb_res_mix[token_offset : token_offset + valid_tokens, 0:mhc_mult2],
            )


@tilelang.jit()
def mhc_pre_split_mixes_backward_kernel(
    pre_layer_mix_grad,
    post_layer_mix_grad,
    comb_res_mix_grad,
    input_mixes,
    pre_layer_mix,
    post_layer_mix,
    mhc_scale,
    input_mixes_grad,
    mhc_grad_partial,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    static_num_tokens: int,
    token_block_size: int,
    full_tiles: bool,
    use_simd_index: bool,
    pre_index_shift: int,
    comb_index_shift: int,
    num_vec_cores: int,
):
    """Shape-aware backward with per-core scale and base reductions."""

    num_tokens = static_num_tokens if static_num_tokens else T.dynamic('num_tokens')
    mhc_mult = 4
    mhc_mult2 = 16
    mhc_mult3 = 24

    pre_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    post_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    comb_res_mix_grad: T.Tensor[(num_tokens, mhc_mult2), T.float32]
    input_mixes: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    pre_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    input_mixes_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    mhc_grad_partial: T.Tensor[(num_vec_cores, mhc_mult3 + 3), T.float32]

    with T.Kernel(num_vec_cores) as core_id:
        pre_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        post_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        comb_grad_ub = T.alloc_shared((token_block_size, mhc_mult2), T.float32)
        input_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        pre_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        post_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        input_grad_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        scale_ub = T.alloc_shared((8,), T.float32)
        pre_index_ub = T.alloc_shared((_VEC,), T.uint32)
        comb_index_ub = T.alloc_shared((_VEC,), T.uint32)
        scale_acc_ub = T.alloc_shared((3, _VEC), T.float32)
        base_acc_ub = T.alloc_shared((3, _VEC), T.float32)
        grad_partial_ub = T.alloc_shared((mhc_mult3 + 3,), T.float32)

        if use_simd_index:
            with T.SimdVF():
                lane = T.reinterpret(S.vci(0, T.int32), 'uint32x64')
                pre_index_init = S.vadd(
                    lane,
                    S.vmuls(S.vshrs(lane, pre_index_shift), mhc_mult3 - mhc_mult),
                )
                comb_index_init = S.vadd(
                    lane,
                    S.vmuls(S.vshrs(lane, comb_index_shift), mhc_mult3 - mhc_mult2),
                )
                S.vsts(pre_index_ub[0], pre_index_init)
                S.vsts(comb_index_ub[0], comb_index_init)

        T.copy(mhc_scale, scale_ub[0:3])
        if not use_simd_index:
            with T.SimtVF(threads=_VEC):
                for lane in T.Parallel(_VEC):
                    pre_index_ub[lane] = T.cast(
                        (lane // mhc_mult) * mhc_mult3 + lane % mhc_mult,
                        T.uint32,
                    )
                    comb_index_ub[lane] = T.cast(
                        (lane // mhc_mult2) * mhc_mult3 + lane % mhc_mult2,
                        T.uint32,
                    )

        with T.SimdVF():
            init_zeros = S.vdup(0.0, T.float32)
            for group in T.unroll(3, explicit=True):
                S.vsts(scale_acc_ub[group, 0], init_zeros)
                S.vsts(base_acc_ub[group, 0], init_zeros)

        for tile_id in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=2,
        ):
            token_offset = tile_id * token_block_size
            valid_tokens = token_block_size if full_tiles else T.min(token_block_size, num_tokens - token_offset)
            T.copy(
                pre_layer_mix_grad[token_offset : token_offset + valid_tokens, 0:mhc_mult],
                pre_grad_ub[0:valid_tokens, 0:mhc_mult],
            )
            T.copy(
                post_layer_mix_grad[token_offset : token_offset + valid_tokens, 0:mhc_mult],
                post_grad_ub[0:valid_tokens, 0:mhc_mult],
            )
            T.copy(
                comb_res_mix_grad[token_offset : token_offset + valid_tokens, 0:mhc_mult2],
                comb_grad_ub[0:valid_tokens, 0:mhc_mult2],
            )
            T.copy(
                input_mixes[token_offset : token_offset + valid_tokens, 0:mhc_mult3],
                input_ub[0:valid_tokens, 0:mhc_mult3],
            )
            T.copy(
                pre_layer_mix[token_offset : token_offset + valid_tokens, 0:mhc_mult],
                pre_ub[0:valid_tokens, 0:mhc_mult],
            )
            T.copy(
                post_layer_mix[token_offset : token_offset + valid_tokens, 0:mhc_mult],
                post_ub[0:valid_tokens, 0:mhc_mult],
            )

            with T.SimdVF():
                zeros = S.vdup(0.0, T.float32)
                one = S.vdup(1.0, T.float32)
                if full_tiles:
                    full_mask = S.pset(32, 'PAT_ALL')
                else:
                    inv_post_mult = S.vdup(1.0 / mhc_post_mult_value, T.float32)
                    lanes = S.vci(0, T.int32)

                pre_index = S.vld(pre_index_ub[0])
                scale_pre = S.vld(scale_ub[0], dist='BRC_B32')
                scale_post = S.vld(scale_ub[1], dist='BRC_B32')
                scale_acc_pre = S.alloc_var(T.float32)
                scale_acc_post = S.alloc_var(T.float32)
                base_acc_pre = S.alloc_var(T.float32)
                base_acc_post = S.alloc_var(T.float32)
                scale_acc_pre = S.vld(scale_acc_ub[0, 0])
                scale_acc_post = S.vld(scale_acc_ub[1, 0])
                base_acc_pre = S.vld(base_acc_ub[0, 0])
                base_acc_post = S.vld(base_acc_ub[1, 0])
                pre_valid_elements = valid_tokens * mhc_mult

                for group in (
                    T.serial(token_block_size * mhc_mult // _VEC)
                    if token_block_size >= 64
                    else T.unroll(token_block_size * mhc_mult // _VEC, explicit=True)
                ):
                    lane_offset = group * _VEC
                    pre_valid = (
                        full_mask
                        if full_tiles
                        else S.vcmp(
                            S.vadds(lanes, lane_offset),
                            S.vdup(pre_valid_elements, T.int32),
                            op='lt',
                        )
                    )
                    compact_row = group * (_VEC // mhc_mult)
                    input_offset = compact_row * mhc_mult3
                    group_index = S.vadds(pre_index, input_offset)
                    post_index = S.vadds(group_index, mhc_mult)
                    pre_grad = (
                        S.vld(pre_grad_ub[compact_row, 0])
                        if full_tiles
                        else S.vsel(S.vld(pre_grad_ub[compact_row, 0]), zeros, pre_valid)
                    )
                    post_grad = (
                        S.vld(post_grad_ub[compact_row, 0])
                        if full_tiles
                        else S.vsel(S.vld(post_grad_ub[compact_row, 0]), zeros, pre_valid)
                    )
                    pre_saved = (
                        S.vld(pre_ub[compact_row, 0])
                        if full_tiles
                        else S.vsel(S.vld(pre_ub[compact_row, 0]), zeros, pre_valid)
                    )
                    post_saved = (
                        S.vld(post_ub[compact_row, 0])
                        if full_tiles
                        else S.vsel(S.vld(post_ub[compact_row, 0]), zeros, pre_valid)
                    )
                    input_pre = (
                        S.vgather2(input_ub[0, 0], group_index, pre_valid)
                        if full_tiles
                        else S.vsel(
                            S.vgather2(input_ub[0, 0], group_index, pre_valid),
                            zeros,
                            pre_valid,
                        )
                    )
                    input_post = (
                        S.vgather2(input_ub[0, 0], post_index, pre_valid)
                        if full_tiles
                        else S.vsel(
                            S.vgather2(input_ub[0, 0], post_index, pre_valid),
                            zeros,
                            pre_valid,
                        )
                    )
                    sigmoid_pre = S.vadds(pre_saved, -mhc_pre_eps)
                    pre_deriv = S.vmul(S.vmul(pre_grad, sigmoid_pre), S.vsub(one, sigmoid_pre))
                    post_deriv = S.vmul(
                        S.vmul(post_grad, post_saved),
                        S.vsub(
                            one,
                            S.vmuls(post_saved, 1.0 / mhc_post_mult_value)
                            if full_tiles
                            else S.vmul(post_saved, inv_post_mult),
                        ),
                    )
                    scale_acc_pre = S.vadd(scale_acc_pre, S.vmul(pre_deriv, input_pre))
                    scale_acc_post = S.vadd(scale_acc_post, S.vmul(post_deriv, input_post))
                    base_acc_pre = S.vadd(base_acc_pre, pre_deriv)
                    base_acc_post = S.vadd(base_acc_post, post_deriv)
                    S.vscatter(S.vmul(pre_deriv, scale_pre), input_grad_ub[0, 0], group_index, pre_valid)
                    S.vscatter(S.vmul(post_deriv, scale_post), input_grad_ub[0, 0], post_index, pre_valid)

                scale_acc_comb = S.alloc_var(T.float32)
                base_acc_comb = S.alloc_var(T.float32)
                scale_acc_comb = S.vld(scale_acc_ub[2, 0])
                base_acc_comb = S.vld(base_acc_ub[2, 0])
                comb_index = S.vld(comb_index_ub[0])
                comb_valid_elements = valid_tokens * mhc_mult2
                scale_comb = S.vld(scale_ub[2], dist='BRC_B32')
                for group in (
                    T.serial(token_block_size * mhc_mult2 // _VEC)
                    if token_block_size >= 64
                    else T.unroll(token_block_size * mhc_mult2 // _VEC, explicit=True)
                ):
                    lane_offset = group * _VEC
                    comb_valid = (
                        full_mask
                        if full_tiles
                        else S.vcmp(
                            S.vadds(lanes, lane_offset),
                            S.vdup(comb_valid_elements, T.int32),
                            op='lt',
                        )
                    )
                    input_offset = group * (_VEC // mhc_mult2) * mhc_mult3 + mhc_mult * 2
                    group_index = S.vadds(comb_index, input_offset)
                    comb_grad = (
                        S.vld(comb_grad_ub[group * (_VEC // mhc_mult2), 0])
                        if full_tiles
                        else S.vsel(
                            S.vld(comb_grad_ub[group * (_VEC // mhc_mult2), 0]),
                            zeros,
                            comb_valid,
                        )
                    )
                    input_comb = (
                        S.vgather2(input_ub[0, 0], group_index, comb_valid)
                        if full_tiles
                        else S.vsel(
                            S.vgather2(input_ub[0, 0], group_index, comb_valid),
                            zeros,
                            comb_valid,
                        )
                    )
                    scale_acc_comb = S.vadd(scale_acc_comb, S.vmul(comb_grad, input_comb))
                    base_acc_comb = S.vadd(base_acc_comb, comb_grad)
                    S.vscatter(
                        S.vmul(comb_grad, scale_comb),
                        input_grad_ub[0, 0],
                        group_index,
                        comb_valid,
                    )

                S.vsts(scale_acc_ub[0, 0], scale_acc_pre)
                S.vsts(scale_acc_ub[1, 0], scale_acc_post)
                S.vsts(scale_acc_ub[2, 0], scale_acc_comb)
                S.vsts(base_acc_ub[0, 0], base_acc_pre)
                S.vsts(base_acc_ub[1, 0], base_acc_post)
                S.vsts(base_acc_ub[2, 0], base_acc_comb)

            T.copy(
                input_grad_ub[0:valid_tokens, 0:mhc_mult3],
                input_mixes_grad[token_offset : token_offset + valid_tokens, 0:mhc_mult3],
            )

        with T.SimdVF():
            zeros = S.vdup(0.0, T.float32)
            one_lane = S.pset(32, 'PAT_VL1')
            lanes = S.vci(0, T.int32)
            pre_reduce_mask = S.vcmp(lanes, S.vdup(_VEC // mhc_mult, T.int32), op='lt')
            comb_reduce_mask = S.vcmp(lanes, S.vdup(_VEC // mhc_mult2, T.int32), op='lt')
            pre_stride = S.vmuls(lanes, mhc_mult)
            comb_stride = S.vmuls(lanes, mhc_mult2)

            for group in T.unroll(3, explicit=True):
                scale_sum = S.vcadd(S.vld(scale_acc_ub[group, 0]))
                S.vsts(grad_partial_ub[group], scale_sum, one_lane, dist='ONEPT_B32')

            for component in T.unroll(mhc_mult, explicit=True):
                pre_values = S.vsel(
                    S.vgather2(
                        base_acc_ub[0, 0],
                        T.reinterpret(S.vadds(pre_stride, component), 'uint32x64'),
                        pre_reduce_mask,
                    ),
                    zeros,
                    pre_reduce_mask,
                )
                post_values = S.vsel(
                    S.vgather2(
                        base_acc_ub[1, 0],
                        T.reinterpret(S.vadds(pre_stride, component), 'uint32x64'),
                        pre_reduce_mask,
                    ),
                    zeros,
                    pre_reduce_mask,
                )
                S.vsts(
                    grad_partial_ub[3 + component],
                    S.vcadd(pre_values),
                    one_lane,
                    dist='ONEPT_B32',
                )
                S.vsts(
                    grad_partial_ub[3 + mhc_mult + component],
                    S.vcadd(post_values),
                    one_lane,
                    dist='ONEPT_B32',
                )

            for component in T.unroll(mhc_mult2, explicit=True):
                comb_values = S.vsel(
                    S.vgather2(
                        base_acc_ub[2, 0],
                        T.reinterpret(S.vadds(comb_stride, component), 'uint32x64'),
                        comb_reduce_mask,
                    ),
                    zeros,
                    comb_reduce_mask,
                )
                S.vsts(
                    grad_partial_ub[3 + mhc_mult * 2 + component],
                    S.vcadd(comb_values),
                    one_lane,
                    dist='ONEPT_B32',
                )

        T.copy(
            grad_partial_ub[0 : mhc_mult3 + 3],
            mhc_grad_partial[core_id, 0 : mhc_mult3 + 3],
        )
