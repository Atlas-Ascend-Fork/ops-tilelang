"""TileLang kernels for expanding activations across MHC streams."""

import tilelang
from tilelang.ascend import language as T

_VECTOR_WIDTH = 64
_FORWARD_TOKEN_BLOCK = 32

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_LOOP_UNSWITCHING: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def expand_to_mhc_forward_kernel(x, output, hidden_block: int, num_stages: int, num_vec_cores: int):
    num_tokens = T.dynamic('num_tokens')
    hidden_size, mhc_mult = T.const('hidden_size, mhc_mult')
    x: T.Tensor[(num_tokens, hidden_size), T.bfloat16]
    output: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16]

    with T.Kernel(num_vec_cores) as core_id:
        x_ub = T.alloc_shared((_FORWARD_TOKEN_BLOCK, hidden_block), T.bfloat16)
        T.annotate_buffer_versions({x_ub: num_stages})

        for token_index, hidden_index in T.Persistent(
            [T.ceildiv(num_tokens, _FORWARD_TOKEN_BLOCK), T.ceildiv(hidden_size, hidden_block)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            row_base = token_index * _FORWARD_TOKEN_BLOCK
            column_base = hidden_index * hidden_block
            T.copy(x[row_base, column_base], x_ub)
            for mhc_index in T.serial(mhc_mult):
                T.copy(
                    x_ub,
                    output[
                        row_base : row_base + _FORWARD_TOKEN_BLOCK,
                        mhc_index,
                        column_base : column_base + hidden_block,
                    ],
                )


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def expand_to_mhc_backward_kernel(
    output_grad,
    x_grad,
    hidden_block: int,
    num_stages: int,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    hidden_size, mhc_mult = T.const('hidden_size, mhc_mult')
    output_grad: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16]
    x_grad: T.Tensor[(num_tokens, hidden_size), T.bfloat16]

    with T.Kernel(num_vec_cores) as core_id:
        output_grad_ub = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
        x_grad_ub = T.alloc_shared((hidden_block,), T.bfloat16)
        T.annotate_buffer_versions({output_grad_ub: num_stages, x_grad_ub: num_stages})

        for token_index, hidden_index in T.Persistent(
            [num_tokens, T.ceildiv(hidden_size, hidden_block)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            T.copy(output_grad[token_index, 0, hidden_index * hidden_block], output_grad_ub)

            with T.SimdVF():
                accumulator = T.simd.alloc_var('float32')
                ones = T.simd.vdup(1.0, T.float32)
                num_chunks = hidden_block // _VECTOR_WIDTH
                for chunk in range(num_chunks):
                    column = chunk * _VECTOR_WIDTH
                    accumulator = T.simd.vcvt(
                        T.simd.vld(output_grad_ub[0, column], dist='UNPK_B16'),
                        T.float32,
                        part=0,
                    )
                    for mhc_index in T.unroll(mhc_mult - 1, explicit=True):
                        output_grad_value = T.simd.vcvt(
                            T.simd.vld(output_grad_ub[mhc_index + 1, column], dist='UNPK_B16'),
                            T.float32,
                            part=0,
                        )
                        T.simd.vmula(accumulator, output_grad_value, ones)
                    T.simd.vsts(
                        x_grad_ub[column],
                        T.simd.vcvt(accumulator, T.bfloat16),
                        dist='PK_B32',
                    )

            T.copy(x_grad_ub, x_grad[token_index, hidden_index * hidden_block])
