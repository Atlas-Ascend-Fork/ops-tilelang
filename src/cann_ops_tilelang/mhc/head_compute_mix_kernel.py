"""TileLang kernels for the MHC head mix transform."""

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

_VECTOR_WIDTH = 64
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}


@T.macro()
def _split_accumulators(base_acc, width: int) -> None:
    """Split interleaved component accumulators once in the reduction tree."""
    for component in T.unroll(width, explicit=True):
        low, high = S.vdintlv(base_acc[component], base_acc[component])
        base_acc[component] = low
        base_acc[component + width] = high


@T.macro
def _split_accumulator_tree(base_acc, mhc_mult: int) -> None:
    """Separate interleaved accumulators into one accumulator per component."""
    if mhc_mult > 2:
        _split_accumulators(base_acc, 2)
    if mhc_mult > 4:
        _split_accumulators(base_acc, 4)
    if mhc_mult > 8:
        _split_accumulators(base_acc, 8)
    if mhc_mult > 16:
        _split_accumulators(base_acc, 16)


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_head_compute_mix_forward_kernel(
    input_mix,
    mhc_scale,
    mhc_base,
    output_mix,
    mhc_pre_eps: float,
    token_block_size: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')

    input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale: T.Tensor[(1,), T.float32]
    mhc_base: T.Tensor[(mhc_mult,), T.float32]
    output_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]

    with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as block_id:
        input_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        output_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        scale_ub = T.alloc_shared((8,), T.float32)
        base_ub = T.alloc_shared((T.ceildiv(mhc_mult, 8) * 8,), T.float32)

        token_offset = block_id * token_block_size
        valid_tokens = T.min(token_block_size, num_tokens - token_offset)

        T.copy(mhc_scale[0], scale_ub[0:1])
        T.copy(mhc_base[0:mhc_mult], base_ub[0:mhc_mult])
        T.copy(
            input_mix[token_offset : token_offset + valid_tokens, 0:mhc_mult],
            input_ub[0:valid_tokens, 0:mhc_mult],
        )

        with T.SimdVF():
            zeros = S.vdup(T.float32(0.0), T.float32)
            ones = S.vdup(T.float32(1.0), T.float32)
            scale = S.vld(scale_ub[0], dist='BRC_B32')
            base_indices = S.vand(
                T.reinterpret(S.vci(T.int32(0), T.int32), 'uint32x64'),
                S.vdup(T.uint32(mhc_mult - 1), T.uint32),
            )
            base_pattern = S.vgather2(base_ub[0], base_indices)
            for vector_id in T.serial(token_block_size * mhc_mult // _VECTOR_WIDTH):
                row = vector_id * _VECTOR_WIDTH // mhc_mult
                x = S.vld(input_ub[row, 0])
                affine = S.vadd(S.vmul(x, scale), base_pattern)
                sigmoid = S.vdiv(ones, S.vadds(S.vexpdif(zeros, affine), 1.0))
                S.vsts(output_ub[row, 0], S.vadds(sigmoid, mhc_pre_eps))

        T.copy(
            output_ub[0:valid_tokens, 0:mhc_mult],
            output_mix[token_offset : token_offset + valid_tokens, 0:mhc_mult],
        )


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_head_compute_mix_backward_kernel(
    output_mix_grad,
    input_mix,
    mhc_scale,
    mhc_base,
    input_mix_grad,
    mhc_scale_grad_partial,
    mhc_base_grad_partial,
    token_block_size: int,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')

    output_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale: T.Tensor[(1,), T.float32]
    mhc_base: T.Tensor[(mhc_mult,), T.float32]
    input_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale_grad_partial: T.Tensor[(num_vec_cores, 1), T.float32]
    mhc_base_grad_partial: T.Tensor[(num_vec_cores, mhc_mult), T.float32]

    num_tiles = T.ceildiv(num_tokens, token_block_size)

    with T.Kernel(num_vec_cores) as block_id:
        output_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        input_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        scale_ub = T.alloc_shared((8,), T.float32)
        base_ub = T.alloc_shared((T.ceildiv(mhc_mult, 8) * 8,), T.float32)
        input_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        scale_grad_partial_ub = T.alloc_shared((8,), T.float32)
        base_grad_partial_ub = T.alloc_shared((T.ceildiv(mhc_mult, 8) * 8,), T.float32)
        scale_acc_ub = T.alloc_shared((_VECTOR_WIDTH,), T.float32)
        seed_acc_ub = T.alloc_shared((2, _VECTOR_WIDTH), T.float32)

        T.copy(mhc_base[0:mhc_mult], base_ub[0:mhc_mult])
        T.copy(mhc_scale[0], scale_ub[0:1])

        with T.SimtVF(threads=_VECTOR_WIDTH):
            T.clear(scale_acc_ub)
            T.clear(seed_acc_ub)

        for tile_id in T.Persistent(
            [num_tiles],
            num_vec_cores,
            block_id,
            group_size=1,
            num_stages=2,
            annotations={'multi_buffer_eligible': [output_grad_ub, input_ub, input_grad_ub]},
        ):
            token_offset = tile_id * token_block_size
            valid_tokens = T.min(token_block_size, num_tokens - token_offset)

            T.copy(
                output_mix_grad[token_offset : token_offset + valid_tokens, 0:mhc_mult],
                output_grad_ub[0:valid_tokens, 0:mhc_mult],
            )
            T.copy(
                input_mix[token_offset : token_offset + valid_tokens, 0:mhc_mult],
                input_ub[0:valid_tokens, 0:mhc_mult],
            )

            with T.SimdVF():
                zeros = S.vdup(T.float32(0.0), T.float32)
                ones = S.vdup(T.float32(1.0), T.float32)
                scale = S.vld(scale_ub[0], dist='BRC_B32')
                base_indices = S.vand(
                    T.reinterpret(S.vci(T.int32(0), T.int32), 'uint32x64'),
                    S.vdup(T.uint32(mhc_mult - 1), T.uint32),
                )
                base_pattern = S.vgather2(base_ub[0], base_indices)

                S.mem_bar('VST_VLD')
                scale_acc = S.alloc_var(T.float32)
                scale_acc = S.vld(scale_acc_ub[0])
                seed_acc = S.alloc_local((2,), T.float32)
                seed_acc[0] = S.vld(seed_acc_ub[0, 0])
                seed_acc[1] = S.vld(seed_acc_ub[1, 0])

                for vector_id in T.serial(token_block_size * mhc_mult // _VECTOR_WIDTH):
                    row = vector_id * _VECTOR_WIDTH // mhc_mult
                    x = S.alloc_var(T.float32)
                    x = S.vld(input_ub[row, 0])
                    output_grad = S.alloc_var(T.float32)
                    output_grad = S.vld(output_grad_ub[row, 0])

                    if valid_tokens < token_block_size:
                        valid_mask = S.vcmp(
                            S.vci(T.int32(row * mhc_mult), T.int32),
                            S.vdup(T.int32((num_tokens % token_block_size) * mhc_mult), T.int32),
                            op='lt',
                        )
                        x = S.vsel(x, zeros, valid_mask)
                        output_grad = S.vsel(output_grad, zeros, valid_mask)

                    affine = S.alloc_var(T.float32)
                    affine = x
                    S.vmadd(affine, scale, base_pattern)
                    sigmoid = S.vdiv(ones, S.vadds(S.vexpdif(zeros, affine), 1.0))
                    grad = S.vmul(S.vmul(sigmoid, S.vsub(ones, sigmoid)), output_grad)
                    S.vsts(input_grad_ub[row, 0], S.vmul(grad, scale))

                    S.vmula(scale_acc, grad, x)
                    even, odd = S.vdintlv(grad, grad)
                    seed_acc[0] = S.vadd(seed_acc[0], even)
                    seed_acc[1] = S.vadd(seed_acc[1], odd)

                S.vsts(scale_acc_ub[0], scale_acc)
                S.vsts(seed_acc_ub[0, 0], seed_acc[0])
                S.vsts(seed_acc_ub[1, 0], seed_acc[1])

            T.copy(
                input_grad_ub[0:valid_tokens, 0:mhc_mult],
                input_mix_grad[token_offset : token_offset + valid_tokens, 0:mhc_mult],
            )

        with T.SimdVF():
            S.mem_bar('VST_VLD')
            scale_acc = S.alloc_var(T.float32)
            scale_acc = S.vld(scale_acc_ub[0])
            S.vsts(scale_grad_partial_ub[0], S.vcadd(scale_acc), dist='ONEPT_B32')

            seed_acc = S.alloc_local((2,), T.float32)
            seed_acc[0] = S.vld(seed_acc_ub[0, 0])
            seed_acc[1] = S.vld(seed_acc_ub[1, 0])

            base_acc = S.alloc_local((mhc_mult,), T.float32)
            base_acc[0] = seed_acc[0]
            base_acc[1] = seed_acc[1]
            _split_accumulator_tree(base_acc, mhc_mult)
            component_lanes_mask = S.pset(32, f'PAT_VL{_VECTOR_WIDTH // mhc_mult}')
            for component in T.unroll(mhc_mult, explicit=True):
                S.vsts(
                    base_grad_partial_ub[component],
                    S.vcadd(base_acc[component], component_lanes_mask),
                    dist='ONEPT_B32',
                )

        T.copy(scale_grad_partial_ub[0:1], mhc_scale_grad_partial[block_id, 0:1])
        T.copy(base_grad_partial_ub[0:mhc_mult], mhc_base_grad_partial[block_id, 0:mhc_mult])
