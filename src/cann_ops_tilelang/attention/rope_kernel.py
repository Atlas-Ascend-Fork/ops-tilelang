"""TileLang Ascend kernel for rotary embeddings."""

import tilelang
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(
    out_idx=None,
    target='ascend',
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _rotary_embedding_tl_ascend(
    dtype: str,
    positions_dtype: str,
    nheads: int,
    head_size: int,
    rotary_dim: int,
    has_position_tensor: bool,
    interleaved: bool,
    conjugate: bool,
    cos_sin_stride: int,
    x_stride_0: int,
    x_stride_1: int,
    y_stride_0: int,
    y_stride_1: int,
) -> None:
    seqlen = T.dynamic('seqlen')
    seqlen_ro = T.dynamic('seqlen_ro')
    half_d = rotary_dim // 2
    dtype_bytes = 4 if dtype == 'float32' else 2
    supported_simd_dtype = dtype in ('bfloat16', 'float32')
    simd_half_aligned = half_d * dtype_bytes % 32 == 0
    use_simd_fast_path = supported_simd_dtype and half_d <= 64 and not interleaved and simd_half_aligned
    simd_mask_pattern = 'PAT_ALL' if half_d >= 64 else f'PAT_VL{half_d}'
    if use_simd_fast_path:
        min_padded_head_size = max(head_size, half_d + 64)
        ub_align_elements = 32 // dtype_bytes
        padded_head_size = (min_padded_head_size + ub_align_elements - 1) // ub_align_elements * ub_align_elements
    else:
        padded_head_size = head_size
    padded_half_d = (half_d + 63) // 64 * 64 if use_simd_fast_path else half_d
    tile_elements_per_head = padded_head_size
    max_heads_per_tile = 64 if use_simd_fast_path else 32
    heads_per_tile = max(1, min(nheads, max_heads_per_tile, 32 * 1024 // (tile_elements_per_head * dtype_bytes)))
    while nheads % heads_per_tile != 0:
        heads_per_tile -= 1
    head_tiles = nheads // heads_per_tile
    threads = 256 if heads_per_tile * half_d >= 4096 else 128
    num_cores = get_num_vector_cores()
    num_stages = 4 if use_simd_fast_path else 1

    @T.macro
    def rotate_simd_block(x0_buffer, x1_buffer, cos_sin_buffer, x0_offset, x1_offset, cos_offset, mask):
        cos_value = T.simd.vld(cos_sin_buffer[cos_offset])
        sin_value = T.simd.vld(cos_sin_buffer[half_d + cos_offset])
        for head in T.unroll(heads_per_tile, explicit=True):
            if dtype == 'bfloat16':
                x0 = T.simd.vcvt(T.simd.vld(x0_buffer[head, x0_offset], dist='UNPK_B16'), T.float32, mask, part=0)
                x1 = T.simd.vcvt(T.simd.vld(x1_buffer[head, x1_offset], dist='UNPK_B16'), T.float32, mask, part=0)
            else:
                x0 = T.simd.vld(x0_buffer[head, x0_offset])
                x1 = T.simd.vld(x1_buffer[head, x1_offset])
            if conjugate:
                x0_result = T.simd.vadd(T.simd.vmul(x0, cos_value, mask), T.simd.vmul(x1, sin_value, mask), mask)
                x1_result = T.simd.vsub(T.simd.vmul(x1, cos_value, mask), T.simd.vmul(x0, sin_value, mask), mask)
            else:
                x0_result = T.simd.vsub(T.simd.vmul(x0, cos_value, mask), T.simd.vmul(x1, sin_value, mask), mask)
                x1_result = T.simd.vadd(T.simd.vmul(x0, sin_value, mask), T.simd.vmul(x1, cos_value, mask), mask)
            if dtype == 'bfloat16':
                T.simd.vsts(x0_buffer[head, x0_offset], T.simd.vcvt(x0_result, T.bfloat16, mask), mask, 'PK_B32')
                T.simd.vsts(x1_buffer[head, x1_offset], T.simd.vcvt(x1_result, T.bfloat16, mask), mask, 'PK_B32')
            else:
                T.simd.vsts(x0_buffer[head, x0_offset], x0_result, mask)
                T.simd.vsts(x1_buffer[head, x1_offset], x1_result, mask)

    @T.prim_func
    def _rotary_embedding_ascend(
        x: T.StridedTensor((seqlen, nheads, head_size), (x_stride_0, x_stride_1, 1), dtype),  # type: ignore
        cos_sin: T.StridedTensor((seqlen_ro, rotary_dim), (cos_sin_stride, 1), 'float32'),  # type: ignore
        positions: T.Tensor((seqlen,), positions_dtype),  # type: ignore
        y: T.StridedTensor((seqlen, nheads, head_size), (y_stride_0, y_stride_1, 1), dtype),  # type: ignore
        seqlen_offsets: T.int32,
    ) -> None:
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((heads_per_tile, padded_head_size), dtype)
            cos_sin_ub = T.alloc_shared((padded_half_d * 2,), 'float32')
            T.annotate_buffer_versions({x_ub: num_stages, cos_sin_ub: num_stages})

            for task_id in T.Persistent(
                [seqlen * head_tiles],
                num_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
                annotations={'enable_offset': True},
            ):
                token_id = task_id // head_tiles
                head_begin = task_id % head_tiles * heads_per_tile
                position = T.alloc_var(positions_dtype, seqlen_offsets)
                if has_position_tensor:
                    position += positions[token_id]
                T.copy(x[token_id, head_begin : head_begin + heads_per_tile, :head_size], x_ub[:, :head_size])
                T.copy(cos_sin[position, :rotary_dim], cos_sin_ub[:rotary_dim])

                if use_simd_fast_path:
                    with T.SimdVF():
                        mask = T.simd.pset(32, simd_mask_pattern)
                        rotate_simd_block(x_ub, x_ub, cos_sin_ub, 0, half_d, 0, mask)
                else:
                    with T.SimtVF(threads=threads):
                        for head, lane in T.Parallel(heads_per_tile, half_d):
                            if interleaved:
                                x0_col = lane * 2
                                x1_col = lane * 2 + 1
                            else:
                                x0_col = lane
                                x1_col = half_d + lane
                            x0 = T.cast(x_ub[head, x0_col], 'float32')
                            x1 = T.cast(x_ub[head, x1_col], 'float32')
                            cos_value = cos_sin_ub[lane]
                            sin_value = cos_sin_ub[half_d + lane]
                            if conjugate:
                                x_ub[head, x0_col] = T.cast(x0 * cos_value + x1 * sin_value, dtype)
                                x_ub[head, x1_col] = T.cast(x1 * cos_value - x0 * sin_value, dtype)
                            else:
                                x_ub[head, x0_col] = T.cast(x0 * cos_value - x1 * sin_value, dtype)
                                x_ub[head, x1_col] = T.cast(x0 * sin_value + x1 * cos_value, dtype)

                T.copy(x_ub[:, :head_size], y[token_id, head_begin : head_begin + heads_per_tile, :head_size])

    return _rotary_embedding_ascend
