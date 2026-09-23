import math

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.config import get_num_vector_cores
from cann_ops_tilelang.utils import align_up as align
from cann_ops_tilelang.utils import ceil_div

from .common import CastInputConfig, get_packed_ue8m0_pack_factor, get_sf_shape


@tilelang.jit()
def get_cast_back_kernel_ascend(
    hidden: int,
    in_config: CastInputConfig,
    out_dtype: T.dtype = T.bfloat16,
):
    n_cores = get_num_vector_cores()

    num_per_tokens, num_per_channels = in_config.sf_block
    pack_factor = get_packed_ue8m0_pack_factor() if in_config.use_packed_ue8m0 else 1
    assert (num_per_tokens, num_per_channels) in ((1, 32), (1, 128), (32, 1), (32, 32), (128, 1), (128, 128))
    assert not ((num_per_tokens, num_per_channels) == (32, 1) and in_config.use_tma_aligned_col_major_sf)

    use_e2b_scale = (
        num_per_tokens == 1
        and num_per_channels == 32
        and in_config.use_packed_ue8m0
        and hidden % 256 == 0
        and in_config.dtype == T.float4_e2m1fn
        and out_dtype != T.float32
    )
    block_k_align = 128 if in_config.dtype == T.float4_e2m1fn else 64
    if num_per_tokens == 1:
        block_k = max(math.gcd(hidden, 256 if use_e2b_scale else 512 // out_dtype.bytes), block_k_align)
        block_m_limit = 64 if use_e2b_scale or out_dtype != T.float32 else 128
    else:
        block_k = max(math.gcd(hidden, 256), block_k_align)
        block_m_limit = 256 // out_dtype.bytes
        if num_per_tokens == 128 and out_dtype == T.float32:
            block_k = max(math.gcd(hidden, 128), block_k_align)
    block_m = 128 if num_per_tokens == 128 else min(block_m_limit, align(8 * ceil_div(hidden, block_k), 64))

    num_sf_rows = ceil_div(block_m, num_per_tokens)
    num_sf_cols = ceil_div(block_k, num_per_channels)

    sf_raw_rows, sf_raw_cols = get_sf_shape((block_m, block_k), in_config)
    sf_raw_cols //= pack_factor  # uint8 -> uint16 if packed
    sf_raw_dtype = T.uint16 if in_config.use_packed_ue8m0 else in_config.sf_dtype
    sf_ub_dtype = T.bfloat16 if use_e2b_scale else T.float32

    if num_per_tokens == 128:
        if hidden >= 512 and block_k == 64 and out_dtype != T.float32:
            num_stages = 3
        else:
            num_stages = 2 if hidden >= 512 else (2 if num_per_channels == 128 and out_dtype == T.float32 else 1)
    else:
        num_stages = 3 if num_per_tokens == 1 and out_dtype != T.float32 else 2

    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    sf_stride = T.dynamic('sf_stride')
    num_token_tiles = T.ceildiv(num_tokens, block_m)
    num_hidden_tiles = T.ceildiv(hidden, block_k)
    num_hidden_sf = ceil_div(hidden, num_per_channels)
    hidden_tail_cols = hidden % block_k

    @T.macro
    def load_scale_value(x_sf_raw: T.Tensor, token_index: int, channel_index: int):
        if in_config.use_packed_ue8m0:
            if in_config.sf_col_pack:
                packed = x_sf_raw[token_index // pack_factor, channel_index]
                packed_index = token_index
            elif in_config.use_tma_aligned_col_major_sf:
                packed = x_sf_raw[channel_index // pack_factor, token_index]
                packed_index = channel_index
            else:
                packed = x_sf_raw[token_index, channel_index // pack_factor]
                packed_index = channel_index
            exponent = T.if_then_else(packed_index % pack_factor == 0, packed & 0xFF, packed >> 8)
            return T.reinterpret(T.uint32(exponent) << 23, T.float32)
        elif in_config.use_tma_aligned_col_major_sf:
            return x_sf_raw[channel_index, token_index]
        else:
            return x_sf_raw[token_index, channel_index]

    @T.macro
    def load_scale_tile(x_sf_raw: T.Tensor, sf_raw_ub, sf_ub, token_base: int, channel_base: int):
        if not in_config.use_packed_ue8m0 and not in_config.use_tma_aligned_col_major_sf:
            # Scale rows can be shorter than one MTE block and their dynamic
            # GM stride is not necessarily 32-byte aligned (for example,
            # hidden=576 and sf_block=(32, 32)). Load them on the vector cores
            # exactly as the 128-channel path already does.
            with T.SimtVF(threads=num_sf_rows * num_sf_cols):
                for idx in T.Parallel(num_sf_rows * num_sf_cols):
                    sf_i = idx // num_sf_cols
                    sf_j = idx % num_sf_cols
                    if token_base + sf_i < sf_shape[0]:
                        sf_ub[sf_i, sf_j] = x_sf_raw[token_base + sf_i, channel_base + sf_j]
        elif num_per_channels == 128:
            with T.SimtVF(threads=num_sf_rows * num_sf_cols):
                for idx in T.Parallel(num_sf_rows * num_sf_cols):
                    sf_i = idx // num_sf_cols
                    sf_j = idx % num_sf_cols
                    sf_ub[sf_i, sf_j] = load_scale_value(x_sf_raw, token_base + sf_i, channel_base + sf_j)
        else:
            if in_config.use_tma_aligned_col_major_sf:
                source = x_sf_raw[channel_base // pack_factor, token_base]
                T.copy(source, sf_raw_ub[:sf_raw_rows, :sf_raw_cols])
            else:
                source_token = token_base // pack_factor if in_config.sf_col_pack else token_base
                source_channel = (
                    channel_base // pack_factor
                    if in_config.use_packed_ue8m0 and not in_config.sf_col_pack
                    else channel_base
                )
                T.copy(x_sf_raw[source_token, source_channel], sf_raw_ub)

            with T.SimtVF(threads=num_sf_rows * num_sf_cols):
                for idx in T.Parallel(num_sf_rows * num_sf_cols):
                    sf_i = idx // num_sf_cols
                    sf_j = idx % num_sf_cols
                    raw_i, raw_j = (sf_j, sf_i) if in_config.use_tma_aligned_col_major_sf else (sf_i, sf_j)
                    if in_config.use_packed_ue8m0:
                        if in_config.sf_col_pack or in_config.use_tma_aligned_col_major_sf:
                            packed = sf_raw_ub[raw_i // pack_factor, raw_j]
                        else:
                            packed = sf_raw_ub[raw_i, raw_j // pack_factor]
                        packed_index = token_base + sf_i if in_config.sf_col_pack else channel_base + sf_j
                        odd = packed_index % pack_factor == 1
                        exponent = T.if_then_else(odd, packed >> 8, packed & 0xFF)
                        if use_e2b_scale:
                            scale_bits = T.uint16(exponent) << 7
                            sf_ub[sf_i, sf_j] = T.reinterpret(scale_bits, T.bfloat16)
                        else:
                            scale_bits = T.uint32(exponent) << 23
                            sf_ub[sf_i, sf_j] = T.reinterpret(scale_bits, T.float32)
                    else:
                        sf_ub[sf_i, sf_j] = sf_raw_ub[raw_i, raw_j]

    @T.macro
    def load_scale_tail(
        x_sf_raw: T.Tensor,
        sf_ub,
        token_base: int,
        channel_base: int,
        valid_sf_rows: int,
        valid_sf_cols: int,
    ):
        with T.SimtVF(threads=num_sf_rows * num_sf_cols):
            for idx in T.Parallel(num_sf_rows * num_sf_cols):
                sf_i = idx // num_sf_cols
                sf_j = idx % num_sf_cols
                if sf_i < valid_sf_rows and sf_j < valid_sf_cols:
                    sf_ub[sf_i, sf_j] = load_scale_value(
                        x_sf_raw,
                        token_base + sf_i,
                        channel_base + sf_j,
                    )
                else:
                    sf_ub[sf_i, sf_j] = 0.0

    @T.macro
    def load_scale(sf_ub: T.SharedBuffer, group: int, col: int, mask_low_f32):
        if num_per_channels == 1:
            return S.vld(sf_ub[group, col])
        elif num_per_channels == 32:
            scale_low = S.vld(sf_ub[group, col // 32], dist='BRC_B32')
            scale_high = S.vld(sf_ub[group, col // 32 + 1], dist='BRC_B32')
            return S.vsel(scale_low, scale_high, mask_low_f32)
        else:  # num_per_channels % 64 == 0
            return S.vld(sf_ub[group, col // num_per_channels], dist='BRC_B32')

    @T.macro
    def store_out(out_ub: T.SharedBuffer, row: int, col: int, out_f32):
        if out_dtype == T.float32:
            S.vsts(out_ub[row, col], out_f32)
        else:
            out_bf16 = S.vcvt(out_f32, T.bfloat16)
            S.vsts(out_ub[row, col], out_bf16, dist='PK_B32')

    @T.macro
    def dequant_fp8_chunk(x_ub, out_ub, row, col, scale):
        x_fp8 = S.vld(x_ub[row, col], dist='UNPK4_B8')
        x_f32 = S.vcvt(x_fp8, T.float32)
        store_out(out_ub, row, col, S.vmul(x_f32, scale))

    @T.macro
    def dequant_fp4_chunk(x_ub, out_ub, row, col, scale_low, scale_high, zero_bf16):
        x_fp4 = S.vld(x_ub[row, col], dist='UNPK4_B8')
        x_bf16 = S.vcvt(x_fp4, T.bfloat16)
        x_uint_low, x_uint_high = S.vintlv(zero_bf16, x_bf16)
        x_low = T.reinterpret(x_uint_low, 'float32x64')
        x_high = T.reinterpret(x_uint_high, 'float32x64')
        store_out(out_ub, row, col, S.vmul(x_low, scale_low))
        store_out(out_ub, row, col + 64, S.vmul(x_high, scale_high))

    @T.macro
    def transform_tile(x_ub: T.SharedBuffer, sf_ub: T.SharedBuffer, out_ub: T.SharedBuffer):
        with T.SimdVF():
            mask_low_f32 = S.pset(32, 'PAT_VL32') if num_per_channels == 32 else None

            if use_e2b_scale:
                for row in T.serial(block_m):
                    for chunk in T.serial(block_k // 256):
                        col = chunk * 256
                        scale_e2b = S.vld(sf_ub[row, chunk * 8], dist='E2B_B16')
                        scale_bf16_low, scale_bf16_high = S.vintlv(scale_e2b, scale_e2b)
                        x_low = S.vcvt(S.vld(x_ub[row, col], dist='UNPK4_B8'), T.bfloat16, part=0)
                        x_high = S.vcvt(S.vld(x_ub[row, col + 128], dist='UNPK4_B8'), T.bfloat16, part=0)
                        out_low = S.vmul(x_low, scale_bf16_low)
                        out_high = S.vmul(x_high, scale_bf16_high)
                        S.vsts(out_ub[row, col], out_low, dist='NORM_B16')
                        S.vsts(out_ub[row, col + 128], out_high, dist='NORM_B16')
            elif in_config.dtype == T.float4_e2m1fn:
                zero_bf16 = S.vdup(0.0, T.bfloat16)
                for group in T.serial(num_sf_rows):
                    row_base = group * num_per_tokens
                    for chunk in T.serial(block_k // 128):
                        col = chunk * 128
                        scale_low = load_scale(sf_ub, group, col, mask_low_f32)
                        scale_high = load_scale(sf_ub, group, col + 64, mask_low_f32)
                        for row in T.serial(num_per_tokens):
                            dequant_fp4_chunk(x_ub, out_ub, row_base + row, col, scale_low, scale_high, zero_bf16)
            else:
                for group in T.serial(num_sf_rows):
                    row_base = group * num_per_tokens
                    for chunk in T.serial(block_k // 64):
                        col = chunk * 64
                        scale = load_scale(sf_ub, group, col, mask_low_f32)
                        for row in T.serial(num_per_tokens):
                            dequant_fp8_chunk(x_ub, out_ub, row_base + row, col, scale)

    @T.prim_func
    def cast_back_kernel_ascend(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        x_sf: T.StridedTensor[sf_shape, (sf_stride, 1), in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
    ):
        with T.Kernel(n_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            sf_raw_ub = T.alloc_shared(
                (
                    sf_raw_rows,
                    align(
                        sf_raw_cols,
                        32 // sf_raw_dtype.bytes if in_config.use_tma_aligned_col_major_sf else 1,
                    ),
                ),
                sf_raw_dtype,
            )
            sf_ub = T.alloc_shared((num_sf_rows, align(num_sf_cols, 32 // sf_ub_dtype.bytes)), sf_ub_dtype)
            out_ub = T.alloc_shared((block_m, block_k), out_dtype)
            x_sf_raw = (
                T.Tensor((sf_shape[0], sf_shape[1] // pack_factor), T.uint16, x_sf.data)
                if in_config.use_packed_ue8m0
                else x_sf
            )
            T.annotate_buffer_versions(
                {
                    x_ub: num_stages,
                    sf_raw_ub: num_stages if num_per_tokens == 1 else 1,
                    sf_ub: num_stages if num_per_tokens == 1 else 1,
                    out_ub: num_stages,
                }
            )

            for pid_token, pid_hidden in T.Persistent(
                [num_token_tiles, num_hidden_tiles],
                n_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                token_base = T.alloc_var(T.int32, init=pid_token * block_m // num_per_tokens)
                channel_base = T.alloc_var(T.int32, init=pid_hidden * block_k // num_per_channels)
                token_offset = pid_token * block_m
                hidden_offset = pid_hidden * block_k
                valid_rows = T.min(block_m, num_tokens - token_offset)
                valid_sf_cols = T.min(num_sf_cols, num_hidden_sf - channel_base)
                if hidden_tail_cols != 0:
                    if pid_hidden == num_hidden_tiles - 1:
                        T.copy(
                            x[
                                token_offset : token_offset + valid_rows,
                                hidden_offset : hidden_offset + hidden_tail_cols,
                            ],
                            x_ub[:valid_rows, :hidden_tail_cols],
                        )
                    else:
                        if valid_rows == block_m:
                            T.copy(x[token_offset, hidden_offset], x_ub)
                        else:
                            T.copy(
                                x[
                                    token_offset : token_offset + valid_rows,
                                    hidden_offset : hidden_offset + block_k,
                                ],
                                x_ub[:valid_rows, :block_k],
                            )
                else:
                    if valid_rows == block_m:
                        T.copy(x[token_offset, hidden_offset], x_ub)
                    else:
                        T.copy(
                            x[
                                token_offset : token_offset + valid_rows,
                                hidden_offset : hidden_offset + block_k,
                            ],
                            x_ub[:valid_rows, :block_k],
                        )
                if valid_rows == block_m and valid_sf_cols == num_sf_cols:
                    load_scale_tile(x_sf_raw, sf_raw_ub, sf_ub, token_base, channel_base)
                else:
                    load_scale_tail(
                        x_sf_raw,
                        sf_ub,
                        token_base,
                        channel_base,
                        T.ceildiv(valid_rows, num_per_tokens),
                        valid_sf_cols,
                    )
                transform_tile(x_ub, sf_ub, out_ub)
                if hidden_tail_cols != 0:
                    if pid_hidden == num_hidden_tiles - 1:
                        T.copy(
                            out_ub[:valid_rows, :hidden_tail_cols],
                            out[
                                token_offset : token_offset + valid_rows,
                                hidden_offset : hidden_offset + hidden_tail_cols,
                            ],
                        )
                    elif valid_rows == block_m:
                        T.copy(out_ub, out[token_offset, hidden_offset])
                    else:
                        T.copy(
                            out_ub[:valid_rows, :block_k],
                            out[
                                token_offset : token_offset + valid_rows,
                                hidden_offset : hidden_offset + block_k,
                            ],
                        )
                elif valid_rows == block_m:
                    T.copy(out_ub, out[token_offset, hidden_offset])
                else:
                    T.copy(
                        out_ub[:valid_rows, :block_k],
                        out[
                            token_offset : token_offset + valid_rows,
                            hidden_offset : hidden_offset + block_k,
                        ],
                    )

    return cast_back_kernel_ascend
