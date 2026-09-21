"""TileLang Ascend implementation of Engram fused weight."""

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit
def get_engram_fused_weight_kernel(hidden_size: int, hc_mult: int):
    """Build an elementwise BF16 x BF16 -> FP32 Ascend kernel."""

    total_elements = hc_mult * hidden_size
    if total_elements <= 12288:
        block_size = 512
    elif total_elements <= 131072:
        block_size = 2048
    elif total_elements <= 524288:
        block_size = 4096
    else:
        block_size = 8192
    while total_elements % block_size != 0:
        block_size //= 2
    assert block_size >= 256

    vector_size = 64
    num_vectors = block_size // vector_size
    num_blocks = total_elements // block_size
    num_vec_cores = get_num_vector_cores()
    use_persistent = num_blocks > num_vec_cores
    num_cores = min(num_vec_cores, num_blocks)

    @T.prim_func
    def engram_fused_weight_kernel(
        weight_hidden: T.Tensor[(total_elements,), T.bfloat16],
        weight_embed: T.Tensor[(total_elements,), T.bfloat16],
        weight_fused: T.Tensor[(total_elements,), T.float32],
    ) -> None:
        if use_persistent:
            with T.Kernel(num_cores) as core_id:
                hidden_ub = T.alloc_shared((block_size,), T.bfloat16)
                embed_ub = T.alloc_shared((block_size,), T.bfloat16)
                fused_ub = T.alloc_shared((block_size,), T.float32)
                T.annotate_buffer_versions({hidden_ub: 2, embed_ub: 2, fused_ub: 2})
                for block in T.Persistent(
                    [num_blocks],
                    num_cores,
                    core_id,
                    group_size=1,
                    num_stages=2,
                ):
                    offset = block * block_size
                    T.copy(weight_hidden[offset : offset + block_size], hidden_ub[:])
                    T.copy(weight_embed[offset : offset + block_size], embed_ub[:])
                    with T.SimdVF():
                        for vector in range(num_vectors):
                            vector_offset = vector * vector_size
                            hidden_vector = S.vcvt(
                                S.vld(hidden_ub[vector_offset], dist='UNPK_B16'),
                                T.float32,
                                part=0,
                            )
                            embed_vector = S.vcvt(
                                S.vld(embed_ub[vector_offset], dist='UNPK_B16'),
                                T.float32,
                                part=0,
                            )
                            S.vsts(fused_ub[vector_offset], S.vmul(hidden_vector, embed_vector))
                    T.copy(fused_ub[:], weight_fused[offset : offset + block_size])
        else:
            with T.Kernel(num_blocks) as block:
                hidden_ub = T.alloc_shared((block_size,), T.bfloat16)
                embed_ub = T.alloc_shared((block_size,), T.bfloat16)
                fused_ub = T.alloc_shared((block_size,), T.float32)
                offset = block * block_size
                T.copy(weight_hidden[offset : offset + block_size], hidden_ub[:])
                T.copy(weight_embed[offset : offset + block_size], embed_ub[:])
                with T.SimdVF():
                    for vector in range(num_vectors):
                        vector_offset = vector * vector_size
                        hidden_vector = S.vcvt(
                            S.vld(hidden_ub[vector_offset], dist='UNPK_B16'),
                            T.float32,
                            part=0,
                        )
                        embed_vector = S.vcvt(
                            S.vld(embed_ub[vector_offset], dist='UNPK_B16'),
                            T.float32,
                            part=0,
                        )
                        S.vsts(fused_ub[vector_offset], S.vmul(hidden_vector, embed_vector))
                T.copy(fused_ub[:], weight_fused[offset : offset + block_size])

    return engram_fused_weight_kernel
