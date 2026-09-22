"""TileLang Ascend kernels for MHC post forward and backward."""

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

_VECTOR_LANES = 64


@tilelang.jit
def get_mhc_post_forward_kernel(
    mhc_mult: int,
    hidden_size: int,
    hidden_block: int,
    num_vector_cores: int,
):
    """Build the Ascend MHC post forward kernel."""
    num_tokens = T.dynamic('num_tokens')
    num_hidden_tiles = T.ceildiv(hidden_size, hidden_block)
    num_chunks = hidden_block // _VECTOR_LANES
    num_stages = 3

    @T.prim_func
    def mhc_post_forward_kernel(
        comb_res_mix: T.Tensor[(num_tokens, mhc_mult, mhc_mult), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        post_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        x: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
        out: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_vector_cores) as core_id:
            residual_ub = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
            x_ub = T.alloc_shared((hidden_block,), T.bfloat16)
            comb_res_mix_ub = T.alloc_shared((mhc_mult, mhc_mult), T.float32)
            post_layer_mix_ub = T.alloc_shared((T.ceildiv(mhc_mult, 8) * 8,), T.float32)
            out_ub = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
            T.annotate_buffer_versions(
                {
                    residual_ub: num_stages,
                    x_ub: num_stages,
                    comb_res_mix_ub: num_stages,
                    post_layer_mix_ub: num_stages,
                    out_ub: num_stages,
                }
            )

            for token_id, hidden_tile in T.Persistent(
                [num_tokens, num_hidden_tiles],
                num_vector_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                T.copy(comb_res_mix[token_id, 0, 0], comb_res_mix_ub)
                T.copy(post_layer_mix[token_id, 0], post_layer_mix_ub[0:mhc_mult])
                T.copy(residual[token_id, 0, hidden_tile * hidden_block], residual_ub)
                T.copy(x[token_id, hidden_tile * hidden_block], x_ub)

                with T.SimdVF():
                    comb_mix = S.alloc_local((mhc_mult * mhc_mult,), T.float32)
                    post_mix = S.alloc_local((mhc_mult,), T.float32)
                    for input_head in T.unroll(mhc_mult, explicit=True):
                        for output_head in T.unroll(mhc_mult, explicit=True):
                            comb_mix[input_head * mhc_mult + output_head] = S.vld(
                                comb_res_mix_ub[input_head, output_head],
                                dist='BRC_B32',
                            )
                    for output_head in T.unroll(mhc_mult, explicit=True):
                        post_mix[output_head] = S.vld(
                            post_layer_mix_ub[output_head],
                            dist='BRC_B32',
                        )
                    out_reg = S.alloc_local((mhc_mult,), T.float32)
                    residual_reg = S.alloc_local((mhc_mult,), T.float32)
                    for chunk in T.serial(num_chunks):
                        column = chunk * _VECTOR_LANES
                        x_fp32 = S.vcvt(
                            S.vld(x_ub[column], dist='UNPK_B16'),
                            T.float32,
                            part=0,
                        )
                        for input_head in T.unroll(mhc_mult, explicit=True):
                            residual_reg[input_head] = S.vcvt(
                                S.vld(residual_ub[input_head, column], dist='UNPK_B16'),
                                T.float32,
                                part=0,
                            )
                        for output_head in T.unroll(mhc_mult, explicit=True):
                            out_reg[output_head] = S.vmul(x_fp32, post_mix[output_head])
                        for input_head in T.unroll(mhc_mult, explicit=True):
                            for output_head in T.unroll(mhc_mult, explicit=True):
                                S.vmula(
                                    out_reg[output_head],
                                    residual_reg[input_head],
                                    comb_mix[input_head * mhc_mult + output_head],
                                )
                        for output_head in T.unroll(mhc_mult, explicit=True):
                            S.vsts(
                                out_ub[output_head, column],
                                S.vcvt(out_reg[output_head], T.bfloat16),
                                dist='PK_B32',
                            )

                T.copy(out_ub, out[token_id, 0, hidden_tile * hidden_block])

    return mhc_post_forward_kernel


@tilelang.jit
def get_mhc_post_backward_kernel(
    mhc_mult: int,
    hidden_size: int,
    hidden_block: int,
    num_stages: int,
    num_vector_cores: int,
):
    """Build the Ascend MHC post backward kernel."""
    num_tokens = T.dynamic('num_tokens')
    num_hidden_tiles = T.ceildiv(hidden_size, hidden_block)
    num_chunks = hidden_block // _VECTOR_LANES

    @T.prim_func
    def mhc_post_backward_kernel(
        grad_out: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        comb_res_mix: T.Tensor[(num_tokens, mhc_mult, mhc_mult), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        post_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        x: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
        grad_comb_res_mix: T.Tensor[(num_tokens, mhc_mult, mhc_mult), T.float32],
        grad_residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        grad_post_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        grad_x: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_vector_cores) as core_id:
            grad_out_ub = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
            x_ub = T.alloc_shared((hidden_block,), T.bfloat16)
            residual_ub = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
            comb_res_mix_ub = T.alloc_shared((mhc_mult, mhc_mult), T.float32)
            post_layer_mix_ub = T.alloc_shared((T.ceildiv(mhc_mult, 8) * 8,), T.float32)
            grad_x_ub = T.alloc_shared((hidden_block,), T.bfloat16)
            grad_residual_ub = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
            grad_post_mix_ub = T.alloc_shared((T.ceildiv(mhc_mult, 8) * 8,), T.float32)
            grad_comb_mix_ub = T.alloc_shared((mhc_mult, mhc_mult), T.float32)
            post_mix_acc = T.alloc_shared((mhc_mult, _VECTOR_LANES), T.float32)
            comb_mix_acc = T.alloc_shared((mhc_mult * mhc_mult, _VECTOR_LANES), T.float32)
            T.annotate_buffer_versions(
                {
                    grad_out_ub: num_stages,
                    x_ub: num_stages,
                    residual_ub: num_stages,
                    grad_x_ub: num_stages,
                    grad_residual_ub: num_stages,
                    comb_res_mix_ub: num_stages,
                    post_layer_mix_ub: num_stages,
                    grad_post_mix_ub: 1,
                    grad_comb_mix_ub: 1,
                    post_mix_acc: 1,
                    comb_mix_acc: 1,
                }
            )

            for token_id in T.Persistent(
                [num_tokens],
                num_vector_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                T.copy(comb_res_mix[token_id, 0, 0], comb_res_mix_ub)
                T.copy(post_layer_mix[token_id, 0], post_layer_mix_ub[0:mhc_mult])

                with T.SimdVF():
                    zero = S.vdup(0.0, T.float32)
                    for output_head in T.unroll(mhc_mult, explicit=True):
                        S.vsts(post_mix_acc[output_head, 0], zero)
                    for index in T.unroll(mhc_mult * mhc_mult, explicit=True):
                        S.vsts(comb_mix_acc[index, 0], zero)

                for hidden_tile in T.serial(num_hidden_tiles):
                    T.copy(grad_out[token_id, 0, hidden_tile * hidden_block], grad_out_ub)
                    T.copy(x[token_id, hidden_tile * hidden_block], x_ub)
                    T.copy(residual[token_id, 0, hidden_tile * hidden_block], residual_ub)

                    with T.SimdVF():
                        post_mix = S.alloc_local((mhc_mult,), T.float32)
                        for output_head in T.unroll(mhc_mult, explicit=True):
                            post_mix[output_head] = S.vld(
                                post_layer_mix_ub[output_head],
                                dist='BRC_B32',
                            )
                        comb_mix = S.alloc_local((mhc_mult * mhc_mult,), T.float32)
                        for input_head in T.unroll(mhc_mult, explicit=True):
                            for output_head in T.unroll(mhc_mult, explicit=True):
                                comb_mix[input_head * mhc_mult + output_head] = S.vld(
                                    comb_res_mix_ub[input_head, output_head],
                                    dist='BRC_B32',
                                )

                        grad_out_reg = S.alloc_local((mhc_mult,), T.float32)
                        grad_x_reg = S.alloc_var(T.float32)
                        grad_residual_reg = S.alloc_var(T.float32)
                        post_acc_reg = S.alloc_var(T.float32)
                        grad_comb_mix_input = S.alloc_local((mhc_mult,), T.float32)

                        for chunk in T.serial(num_chunks):
                            column = chunk * _VECTOR_LANES
                            x_fp32 = S.vcvt(
                                S.vld(x_ub[column], dist='UNPK_B16'),
                                T.float32,
                                part=0,
                            )
                            for output_head in T.unroll(mhc_mult, explicit=True):
                                grad_out_reg[output_head] = S.vcvt(
                                    S.vld(grad_out_ub[output_head, column], dist='UNPK_B16'),
                                    T.float32,
                                    part=0,
                                )

                            grad_x_reg = S.vdup(0.0, T.float32)
                            for output_head in T.unroll(mhc_mult, explicit=True):
                                S.vmula(grad_x_reg, grad_out_reg[output_head], post_mix[output_head])
                            S.vsts(
                                grad_x_ub[column],
                                S.vcvt(grad_x_reg, T.bfloat16),
                                dist='PK_B32',
                            )

                            for output_head in T.unroll(mhc_mult, explicit=True):
                                post_acc_reg = S.vld(post_mix_acc[output_head, 0])
                                S.vmula(post_acc_reg, grad_out_reg[output_head], x_fp32)
                                S.vsts(post_mix_acc[output_head, 0], post_acc_reg)

                            for input_head in T.unroll(mhc_mult, explicit=True):
                                residual_input = S.vcvt(
                                    S.vld(residual_ub[input_head, column], dist='UNPK_B16'),
                                    T.float32,
                                    part=0,
                                )
                                for output_head in T.unroll(mhc_mult, explicit=True):
                                    grad_comb_mix_input[output_head] = S.vld(
                                        comb_mix_acc[input_head * mhc_mult + output_head, 0]
                                    )
                                grad_residual_reg = S.vdup(0.0, T.float32)
                                for output_head in T.unroll(mhc_mult, explicit=True):
                                    S.vmula(
                                        grad_residual_reg,
                                        grad_out_reg[output_head],
                                        comb_mix[input_head * mhc_mult + output_head],
                                    )
                                    S.vmula(
                                        grad_comb_mix_input[output_head],
                                        grad_out_reg[output_head],
                                        residual_input,
                                    )
                                S.vsts(
                                    grad_residual_ub[input_head, column],
                                    S.vcvt(grad_residual_reg, T.bfloat16),
                                    dist='PK_B32',
                                )
                                for output_head in T.unroll(mhc_mult, explicit=True):
                                    S.vsts(
                                        comb_mix_acc[input_head * mhc_mult + output_head, 0],
                                        grad_comb_mix_input[output_head],
                                    )

                    T.copy(grad_x_ub, grad_x[token_id, hidden_tile * hidden_block])
                    T.copy(
                        grad_residual_ub,
                        grad_residual[token_id, 0, hidden_tile * hidden_block],
                    )

                with T.SimdVF():
                    for output_head in T.unroll(mhc_mult, explicit=True):
                        S.vsts(
                            grad_post_mix_ub[output_head],
                            S.vcadd(S.vld(post_mix_acc[output_head, 0])),
                            dist='ONEPT_B32',
                        )
                    for input_head in T.unroll(mhc_mult, explicit=True):
                        for output_head in T.unroll(mhc_mult, explicit=True):
                            S.vsts(
                                grad_comb_mix_ub[input_head, output_head],
                                S.vcadd(S.vld(comb_mix_acc[input_head * mhc_mult + output_head, 0])),
                                dist='ONEPT_B32',
                            )

                T.copy(grad_post_mix_ub[0:mhc_mult], grad_post_layer_mix[token_id, 0])
                T.copy(grad_comb_mix_ub, grad_comb_res_mix[token_id, 0, 0])

    return mhc_post_backward_kernel
