import math

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.config import get_num_vector_cores
from cann_ops_tilelang.quant.common import (
    CastInputConfig,
    CastOutputConfig,
    get_packed_ue8m0_pack_factor,
    get_sf_shape,
)
from cann_ops_tilelang.utils import ceil_div


@tilelang.jit
def get_per_channel_cast_kernel_asc(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
):
    # bf16 hidden >= 2048 时用 nbg=1（迭代数翻倍，MTE2 请求更频繁，bf16 2048 改善 ~4%）；
    # 小 hidden(128) 保持 nbg=2 以聚合 pack_row，避免退化。
    bf16_nbg1_min_hidden = 2048
    # hidden tile 数 > 4 时 hidden 维度置外（L2 复用）；e4m3 4096 不 swap 快 2.5%，
    # 6144+ 仍需 swap 防退化（去掉后 6144 时延暴涨 ~31%）。
    swap_min_num_hidden_tiles = 4

    num_per_tokens = out_config.sf_block[0]
    assert num_per_tokens in (32, 128)
    if in_config.with_sf:
        num_per_channels = in_config.sf_block[1]
        assert num_per_channels in (32, 128), 'Ascend rescale supports num_per_channels in (32, 128) only'
    else:
        assert in_config.dtype == T.bfloat16, 'per_channel_cast supports bf16 or e4m3 (rescale) input only'

    quant_max = 448.0
    assert out_config.clamp_min_value >= quant_max * (2 ** (-126)), (
        'Ascend scale exponent fast path requires clamp to normal number'
    )

    # num_per_tokens=128 packed 路径用 block_k=256（而非 1024），使 block_m=128，
    # groups_per_tile=1；num_per_tokens=32 保持 block_k=1024 高性能路径不受影响
    use_large_block_k = in_config.with_sf and in_config.use_packed_ue8m0 and num_per_tokens != 128
    # num_per_channels=128 non-tma non-packed: dequant_ub=(num_per_tokens, block_k) float32 超出共享内存，
    # 减小 block_k 到 128
    if not use_large_block_k and in_config.with_sf and num_per_tokens == 128 and not in_config.use_packed_ue8m0:
        max_block_k = 128
    else:
        max_block_k = 1024 if use_large_block_k else 256
    block_k = math.gcd(max_block_k, hidden)
    block_m = min(128, 32768 // block_k)
    # num_per_tokens=128 时 block_m 需保持 128（=num_per_tokens），不能因 hidden==block_k 而降到 64
    if hidden // block_k == 1 and num_per_tokens != 128:
        block_m = min(64, 32768 // block_k)
    if not in_config.with_sf:
        # bf16 block_m 不得小于 num_per_tokens，否则 groups_per_tile=0 使量化循环跳过、
        # 未初始化的 out_ub 被写回（npt=128 时触发）；block_m=128 时 UB 容不下 3 级流水，故 stages=2。
        block_m = min(64, block_m)
        if block_m < num_per_tokens:
            block_m = num_per_tokens
        num_stages = 3 if block_m <= 64 else 2
    else:
        # num_per_channels=128 non-tma non-packed: block_k=128 时双缓冲仍超出 UB，降为单缓冲
        num_stages = 1 if (num_per_tokens == 128 and not in_config.use_packed_ue8m0) else 2
    groups_per_tile = block_m // num_per_tokens
    # block_m 需为 num_per_tokens 的整数倍，否则 groups_per_tile 为 0 或非整数，
    # 会导致量化循环跳过输出未初始化（num_per_tokens=128 触发）。
    assert block_m >= num_per_tokens and block_m % num_per_tokens == 0, (
        f'Unsupported Ascend tile: block_m={block_m}, num_per_tokens={num_per_tokens}'
    )

    assert block_k % 128 == 0
    num_hidden_tiles = hidden // block_k

    pack_factor = get_packed_ue8m0_pack_factor()
    assert pack_factor == 2
    if in_config.with_sf:
        num_packed_per_channels = num_per_channels * pack_factor if in_config.use_packed_ue8m0 else num_per_channels
    if out_config.use_packed_ue8m0:
        # bf16 2048+ 用 nbg=1 提升 MTE2 请求频率；e4m3 保持 nbg=2（packed e4m3 的 pack_row 依赖 nbg=2 的聚合布局）。
        # nbg=1 要求 groups_per_tile 为偶数（待 pack SF 行数 = groups_per_tile*1 须能被 pack_factor=2 整除），
        # 否则 num_packed_rows = groups_per_tile*1//2 不正确（num_per_tokens=128 时 groups_per_tile=1 触发）。
        num_blocks_per_group = (
            2
            if in_config.with_sf
            else (1 if (hidden >= bf16_nbg1_min_hidden and groups_per_tile % 2 == 0 and groups_per_tile > 0) else 2)
        )
        num_packed_rows = groups_per_tile * num_blocks_per_group // pack_factor
    else:
        num_blocks_per_group = 1

    num_cores = get_num_vector_cores()

    num_tokens = T.dynamic('num_tokens')
    sf_stride = T.dynamic('sf_stride')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    packed_col_major_input = in_config.use_tma_aligned_col_major_sf and in_config.use_packed_ue8m0
    if packed_col_major_input:
        x_sf_shape = (x_sf_shape[0], x_sf_shape[1] // 2)
    x_sf_dtype = T.int16 if packed_col_major_input else in_config.sf_dtype

    @T.macro
    def compute_scale(amax):
        amax = S.vmax(amax, S.vdup(out_config.clamp_min_value, T.float32))
        if not out_config.round_sf:
            scale = S.vdiv(amax, S.vdup(quant_max, T.float32))
            sf_inv = S.vdiv(S.vdup(quant_max, T.float32), amax)
            return scale, sf_inv
        scale_raw = S.vmul(amax, S.vdup(1.0 / quant_max, T.float32))
        scale_bits = T.reinterpret(scale_raw, 'uint32x64')
        scale_exponent = S.vadds(S.vshrs(S.vsub(scale_bits, S.vdup(1, T.uint32)), 23), 1)
        inverse_exponent = S.vsub(S.vdup(254, T.uint32), scale_exponent)
        inverse = T.reinterpret(S.vshls(inverse_exponent, 23), 'float32x64')
        if out_config.use_packed_ue8m0:
            return scale_exponent, inverse
        scale = T.reinterpret(S.vshls(scale_exponent, 23), 'float32x64')
        return scale, inverse

    @T.macro
    def quantize_block_bf16(x_ub, o_ub, sf_dst, sf_input, dequant_ub, dst_base=0):
        with T.SimdVF():
            sf_dist = 'PK4_B32' if out_config.use_packed_ue8m0 else 'NORM_B32'
            zero_bf16 = S.vdup(0.0, T.bfloat16)
            abs_mask = T.reinterpret(S.vdup(0x7FFF, T.uint16), 'bfloat16x128')
            amax_values = S.alloc_local((2,), T.bfloat16)
            inverse = S.alloc_local((2,), T.float32)
            if in_config.with_sf:
                lane_channel = T.reinterpret(S.vshrs(S.vci(0, T.int16), 5), 'uint16x128')  # 0x32 1x32 2x32 3x32
                # num_per_channels=32 时，两行 packed SF 用 vsel 按 lane 组对选择；
                # num_per_channels=128 时，单行 packed SF 用 chunk*8 选字节。
                if in_config.use_tma_aligned_col_major_sf and num_per_channels != 128:
                    use_pair1 = S.vcmps(lane_channel, 2, op='ge')  # 0x64 1x64
                    byte_shift = T.reinterpret(
                        S.vshls(S.vand(lane_channel, S.vdup(1, T.uint16)), 3), 'int16x128'
                    )  # 0x32 8x32 0x32 8x32
                    byte_mask = S.vdup(0x00FF, T.int16)
                elif in_config.use_tma_aligned_col_major_sf:
                    byte_mask = S.vdup(0x00FF, T.int16)
            for group in T.serial(groups_per_tile):
                row_base = group * num_per_tokens
                for chunk in T.serial(block_k // 128):
                    col = chunk * 128
                    amax = S.alloc_var(T.bfloat16)
                    if in_config.with_sf:
                        amax = S.vdup(0.0, T.bfloat16)
                        # num_per_channels=128 时，每个 chunk 对应一个 SF 组，直接用 chunk 索引；
                        # num_per_channels=32 时，用 lane_channel 派生索引。
                        if num_per_channels == 128:
                            scale_index = S.vdup(chunk, T.uint16)
                        else:
                            scale_index = S.vadds(lane_channel, chunk * 4)
                        for row in T.serial(num_per_tokens):
                            raw0 = S.vcvt(S.vld(x_ub[row_base + row, col], dist='UNPK4_B8'), T.float32)
                            raw1 = S.vcvt(S.vld(x_ub[row_base + row, col + 64], dist='UNPK4_B8'), T.float32)
                            _, raw_bf16 = S.vdintlv(
                                T.reinterpret(raw0, 'bfloat16x128'), T.reinterpret(raw1, 'bfloat16x128')
                            )
                            if in_config.use_tma_aligned_col_major_sf:
                                sf_row = row_base + row
                                # num_per_channels=128 时，sf_in_ub 仅一行，从 sf_input[0]
                                # 读取并用 chunk*8 选字节；num_per_channels=32 时使用双行 vsel。
                                if num_per_channels == 128:
                                    packed = S.vld(sf_input[0, sf_row], dist='BRC_B16')
                                    byte_shift = S.vdup(chunk * 8, T.int16)
                                    exponents = S.vand(S.vshr(packed, byte_shift), byte_mask)
                                    scale_bits = S.vshls(exponents, 7)
                                else:
                                    packed0 = S.vld(sf_input[chunk * 2, sf_row], dist='BRC_B16')
                                    packed1 = S.vld(sf_input[chunk * 2 + 1, sf_row], dist='BRC_B16')
                                    packed = S.vsel(packed1, packed0, use_pair1)
                                    exponents = S.vand(S.vshr(packed, byte_shift), byte_mask)
                                    scale_bits = S.vshls(exponents, 7)
                            else:
                                exponents = T.reinterpret(
                                    S.vld(sf_input[row_base + row, 0], dist='UNPK_B8'), 'uint16x128'
                                )
                                scale_bits = S.vshls(S.vselr(exponents, scale_index), 7)
                            values = S.vmul(raw_bf16, T.reinterpret(scale_bits, 'bfloat16x128'))
                            S.vsts(dequant_ub[row, 0], values, dist='NORM_B16')
                            amax = S.vmax(amax, S.vand(values, abs_mask))
                    else:
                        amax_acc = S.alloc_local((4,), T.bfloat16)
                        for acc_i in T.unroll(4, explicit=True):
                            amax_acc[acc_i] = S.vdup(0.0, T.bfloat16)
                        for row4 in T.serial(num_per_tokens // 4):
                            for acc_i in T.unroll(4, explicit=True):
                                row = row4 * 4 + acc_i
                                values = S.vld(x_ub[row_base + row, col])
                                amax_acc[acc_i] = S.vmax(amax_acc[acc_i], S.vand(values, abs_mask))
                        amax = S.vmax(S.vmax(amax_acc[0], amax_acc[1]), S.vmax(amax_acc[2], amax_acc[3]))
                    amax_values[0], amax_values[1] = S.vintlv(zero_bf16, amax)
                    # NOTE: round sf/not round sf path can use different instruction order for faster execution
                    # If manual instruction reorder is required, refer to commit aa73858.
                    for half in T.unroll(2, explicit=True):
                        scale, inverse[half] = compute_scale(T.reinterpret(amax_values[half], 'float32x64'))
                        S.vsts(sf_dst[dst_base + group, col + half * 64], scale, dist=sf_dist)
                    for half in T.unroll(2, explicit=True):
                        for row in T.serial(num_per_tokens):
                            if in_config.with_sf:
                                values = S.vcvt(S.vld(dequant_ub[row, half * 64], dist='UNPK_B16'), T.float32)
                            else:
                                values = S.vcvt(
                                    S.vld(x_ub[row_base + row, col + half * 64], dist='UNPK_B16'), T.float32
                                )
                            quantized = S.vmul(values, inverse[half])
                            S.vsts(
                                o_ub[row_base + row, col + half * 64],
                                S.vcvt(quantized, T.float8_e4m3fn),
                                dist='PK4_B32',
                            )

    def dequant_f32_scaled(x_ub, row, col, scale_vec):
        values = S.vcvt(S.vld(x_ub[row, col], dist='UNPK4_B8'), T.float32)
        return S.vmul(values, scale_vec)

    @T.macro
    def quantize_block_f32(x_ub, o_ub, sf_dst, sf_ub, dequant_ub, dst_base=0):
        with T.SimdVF():
            sf_dist = 'PK4_B32' if out_config.use_packed_ue8m0 else 'NORM_B32'
            abs_mask = T.reinterpret(S.vdup(0x7FFFFFFF, T.uint32), 'float32x64')
            num_subvectors = block_k // 64
            amax_values = S.alloc_local((num_subvectors,), T.float32)
            input_scale_values = S.alloc_local((num_subvectors,), T.float32)
            scale_values = S.alloc_local((num_subvectors,), T.uint32 if out_config.use_packed_ue8m0 else T.float32)
            if in_config.use_tma_aligned_col_major_sf:
                sf_lane_channel = T.reinterpret(S.vshrs(S.vci(0, T.int32), 5), 'uint32x64')  # 0x32 1x32
            for g in T.serial(groups_per_tile):
                row_base = g * num_per_tokens
                for j in T.unroll(num_subvectors, explicit=True):
                    amax_values[j] = S.vdup(0.0, T.float32)
                for r in T.serial(num_per_tokens):
                    if in_config.use_tma_aligned_col_major_sf:
                        # num_per_channels=128 时，sf_in_ub 布局为
                        # (block_k//num_per_channels, block_m)，每两个 subvector
                        # 共享一行 SF，并通过 BRC_B32 广播。
                        if num_per_channels == 128:
                            for subvector in T.unroll(num_subvectors, explicit=True):
                                sf_group = subvector // 2
                                input_scale_values[subvector] = S.vld(sf_ub[sf_group, row_base + r], dist='BRC_B32')
                        else:
                            sf_gather_base = S.vadds(S.vmuls(sf_lane_channel, block_m), row_base + r)
                            for subvector in T.unroll(num_subvectors, explicit=True):
                                input_scale_values[subvector] = S.vgather2(
                                    sf_ub[0, 0], S.vadds(sf_gather_base, subvector * 2 * block_m)
                                )
                    else:
                        # num_per_channels=128 时，每个 SF 组覆盖 128 列，按
                        # sv//subvectors_per_group 索引行；num_per_channels=32 时用 E2B 广播。
                        if num_per_channels == 128:
                            num_groups = block_k // num_per_channels
                            subvectors_per_group = num_subvectors // num_groups
                            for sv in T.unroll(num_subvectors, explicit=True):
                                input_scale_values[sv] = S.vld(
                                    sf_ub[row_base + r, sv // subvectors_per_group], dist='BRC_B32'
                                )
                        else:
                            expanded = S.vld(sf_ub[row_base + r, 0], dist='E2B_B32')
                            low, high = S.vintlv(expanded, expanded)
                            input_scale_values[0], input_scale_values[1] = S.vintlv(low, low)
                            if block_k == 256:
                                input_scale_values[2], input_scale_values[3] = S.vintlv(high, high)
                    for subvector in T.unroll(num_subvectors, explicit=True):
                        col = subvector * 64
                        values = dequant_f32_scaled(x_ub, row_base + r, col, input_scale_values[subvector])
                        S.vsts(dequant_ub[r, col], values)
                        amax_values[subvector] = S.vmax(amax_values[subvector], S.vand(values, abs_mask))
                if out_config.round_sf:
                    for pair in T.unroll(block_k // 128, explicit=True):
                        for half in T.unroll(2, explicit=True):
                            subvector = pair * 2 + half
                            scale_values[subvector], amax_values[subvector] = compute_scale(amax_values[subvector])
                            S.vsts(sf_dst[dst_base + g, subvector * 64], scale_values[subvector], dist=sf_dist)
                        for r in T.serial(num_per_tokens):
                            for half in T.unroll(2, explicit=True):
                                subvector = pair * 2 + half
                                quantized = S.vmul(S.vld(dequant_ub[r, subvector * 64]), amax_values[subvector])
                                S.vsts(
                                    o_ub[row_base + r, subvector * 64],
                                    S.vcvt(quantized, T.float8_e4m3fn),
                                    dist='PK4_B32',
                                )
                else:
                    for subvector in T.unroll(num_subvectors, explicit=True):
                        scale, inverse = compute_scale(amax_values[subvector])
                        S.vsts(sf_dst[g, subvector * 64], scale)
                        for r in T.serial(num_per_tokens):
                            q = S.vmul(S.vld(dequant_ub[r, subvector * 64]), inverse)
                            S.vsts(o_ub[row_base + r, subvector * 64], S.vcvt(q, T.float8_e4m3fn), dist='PK4_B32')

    @T.macro
    def pack_row(exp_src, pk_dst, exp_row, dst_row):
        with T.SimdVF():
            for base in T.serial(0, block_k, 256):
                lo, hi = S.vintlv(S.vld(exp_src[exp_row, base]), S.vld(exp_src[exp_row + 1, base]))
                S.vsts(pk_dst[dst_row, base * 2], lo, dist='NORM_B8')
                if block_k - base >= 256:
                    S.vsts(pk_dst[dst_row, base * 2 + 256], hi, dist='NORM_B8')

    @T.prim_func
    def per_channel_cast_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
        x_sf_invs: T.Tensor[x_sf_shape, x_sf_dtype],
    ):
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)
            versions = {x_ub: num_stages, out_ub: num_stages}

            if in_config.with_sf:
                if in_config.use_tma_aligned_col_major_sf:
                    if in_config.use_packed_ue8m0:
                        # block_k 可能小于 num_per_channels*pack_factor（例如
                        # num_per_tokens=128 且 block_k=128），需用 ceil_div 避免分配零行。
                        sf_in_ub = T.alloc_shared((ceil_div(block_k, num_per_channels * pack_factor), block_m), T.int16)
                    else:
                        sf_in_ub = T.alloc_shared((block_k // num_per_channels, block_m), in_config.sf_dtype)
                elif in_config.use_packed_ue8m0:
                    sf_in_ub = T.alloc_shared((block_m, max(block_k // num_per_channels, 64)), in_config.sf_dtype)
                else:
                    sf_in_ub = T.alloc_shared((block_m, max(block_k // num_per_channels, 8)), in_config.sf_dtype)
                dequant_dtype = T.bfloat16 if in_config.use_packed_ue8m0 else T.float32
                dequant_columns = 128 if in_config.use_packed_ue8m0 else block_k
                dequant_ub = T.alloc_shared((num_per_tokens, dequant_columns), dequant_dtype)
                versions[sf_in_ub] = num_stages

            if out_config.use_packed_ue8m0:
                sf_out_ub = T.alloc_shared((groups_per_tile * num_blocks_per_group, max(block_k, 256)), T.uint8)
                pk_ub = T.alloc_shared((num_packed_rows, block_k * pack_factor), T.uint8)
                # num_per_tokens=128 时 UB 紧凑（block_k 减小），版本数从 num_stages+2 降为 num_stages
                versions[sf_out_ub] = num_stages
                versions[pk_ub] = num_stages
            else:
                sf_out_ub = T.alloc_shared((groups_per_tile, block_k), T.float32)
                # 同 packed 路径：num_per_tokens=128 UB 紧凑，版本数从 num_stages+2 降为 num_stages
                versions[sf_out_ub] = num_stages
            T.annotate_buffer_versions(versions)

            if in_config.with_sf:
                use_swap = num_hidden_tiles > swap_min_num_hidden_tiles
                if use_swap:
                    dim_outer, dim_inner = num_hidden_tiles, T.ceildiv(num_tokens, block_m * num_blocks_per_group)
                else:
                    dim_outer, dim_inner = T.ceildiv(num_tokens, block_m * num_blocks_per_group), num_hidden_tiles
                for iter_id, tile_id in T.Persistent(
                    [dim_outer, dim_inner],
                    num_cores,
                    core_id,
                    group_size=4,
                    num_stages=num_stages,
                ):
                    col_offset = (iter_id if use_swap else tile_id) * block_k
                    sf_idx = tile_id if use_swap else iter_id
                    for bg in T.serial(num_blocks_per_group):
                        tile_row = sf_idx * num_blocks_per_group + bg
                        if num_blocks_per_group == 1 or tile_row * block_m < num_tokens:
                            T.copy(x[tile_row * block_m, col_offset], x_ub, l2_cache_ctrl='NOTALLOC_KEEP')
                            if in_config.use_tma_aligned_col_major_sf:
                                T.copy(
                                    x_sf_invs[col_offset // num_packed_per_channels, tile_row * block_m],
                                    sf_in_ub[: ceil_div(block_k, num_packed_per_channels), :block_m],
                                )  # ceil_div: 防止 block_k < num_per_channels*pack_factor 时拷贝 0 行
                            else:
                                T.copy(
                                    x_sf_invs[tile_row * block_m, col_offset // num_per_channels],
                                    sf_in_ub[:, : block_k // num_per_channels],
                                )
                            if in_config.use_packed_ue8m0:
                                quantize_block_bf16(
                                    x_ub, out_ub, sf_out_ub, sf_in_ub, dequant_ub, dst_base=bg * groups_per_tile
                                )
                            else:
                                quantize_block_f32(
                                    x_ub, out_ub, sf_out_ub, sf_in_ub, dequant_ub, dst_base=bg * groups_per_tile
                                )
                            T.copy(out_ub, out[tile_row * block_m, col_offset])
                    if not out_config.use_packed_ue8m0:
                        T.copy(sf_out_ub, out_sf[sf_idx * groups_per_tile, col_offset])
                    else:
                        for p in T.serial(num_packed_rows):
                            if sf_idx * (block_m * num_blocks_per_group) + 2 * p * num_per_tokens < num_tokens:
                                pack_row(sf_out_ub, pk_ub, 2 * p, p)
                        pk_base = T.alloc_var(T.int32, init=sf_idx * num_packed_rows)
                        valid_rows = T.alloc_var(T.int32, init=T.min(num_packed_rows, sf_shape[0] - pk_base))
                        T.copy(pk_ub[0:valid_rows, :], out_sf[pk_base, col_offset * 2])
            else:
                use_swap = num_hidden_tiles > swap_min_num_hidden_tiles
                if use_swap:
                    dim_outer, dim_inner = num_hidden_tiles, T.ceildiv(num_tokens, block_m * num_blocks_per_group)
                else:
                    dim_outer, dim_inner = T.ceildiv(num_tokens, block_m * num_blocks_per_group), num_hidden_tiles
                for iter_id, tile_id in T.Persistent(
                    [dim_outer, dim_inner],
                    num_cores,
                    core_id,
                    group_size=4,
                    num_stages=num_stages,
                ):
                    col_offset = (iter_id if use_swap else tile_id) * block_k
                    sf_idx = tile_id if use_swap else iter_id
                    for bg in T.serial(num_blocks_per_group):
                        tile_row = sf_idx * num_blocks_per_group + bg
                        if num_blocks_per_group == 1 or tile_row * block_m < num_tokens:
                            T.copy(x[tile_row * block_m, col_offset], x_ub, l2_cache_ctrl='NOTALLOC_KEEP')
                            quantize_block_bf16(x_ub, out_ub, sf_out_ub, None, None, dst_base=bg * groups_per_tile)
                            T.copy(out_ub, out[tile_row * block_m, col_offset])
                    if not out_config.use_packed_ue8m0:
                        T.copy(sf_out_ub, out_sf[sf_idx * groups_per_tile, col_offset])
                    else:
                        for p in T.serial(num_packed_rows):
                            if sf_idx * (block_m * num_blocks_per_group) + 2 * p * num_per_tokens < num_tokens:
                                pack_row(sf_out_ub, pk_ub, 2 * p, p)
                        pk_base = T.alloc_var(T.int32, init=sf_idx * num_packed_rows)
                        valid_rows = T.alloc_var(T.int32, init=T.min(num_packed_rows, sf_shape[0] - pk_base))
                        T.copy(pk_ub[0:valid_rows, :], out_sf[pk_base, col_offset * 2])

    return per_channel_cast_kernel
