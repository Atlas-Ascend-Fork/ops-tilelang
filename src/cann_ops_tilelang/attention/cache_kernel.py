import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.attention._simd import (
    ASCEND_VSTSX2_BF16_PRELUDE as _ASCEND_VSTSX2_BF16_PRELUDE,
)
from cann_ops_tilelang.attention._simd import (
    is_pto_target as _is_pto_target,
)
from cann_ops_tilelang.attention._simd import (
    vstsx2_bf16 as _vstsx2_bf16,
)
from cann_ops_tilelang.config import get_num_vector_cores


def _pack_prmt_selector(*sources: int) -> int:
    return sum(source << (lane * 4) for lane, source in enumerate(sources))


_PRMT_UE8M0_PAIR01 = _pack_prmt_selector(4, 0, 5, 0)


_PRMT_UE8M0_PAIR23 = _pack_prmt_selector(6, 0, 7, 0)


_BYTE_BITS = 8


_BF16_BITS = 16


_LOW_U32_MASK = (1 << 32) - 1


_SWAR_KEEP_U16_PAIRS = ((1 << _BF16_BITS) - 1) * ((1 << 32) + 1)


_SWAR_KEEP_U8_LANES = sum(((1 << _BYTE_BITS) - 1) << (lane * 16) for lane in range(4))


_UE8M0_TO_BF16_EXP_SHIFT = 7


@tilelang.jit(out_idx=None, pass_configs={tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True})
def _store_kv_cache_tl_ascend(
    block_size: int,
    num_stages: int,
    full_tiles: bool = False,
    block_m: int = 64,
    store_lanes_per_row: int = 16,
    cache_block_stride: int = 0,
    static_n: int = 0,
    static_num_blocks: int = 0,
) -> object:
    """Ascend fused per-64 FP8 quantization and paged-cache store."""
    is_pto = _is_pto_target()
    n = static_n if static_n > 0 else T.dynamic('n')
    num_blocks = static_num_blocks if static_num_blocks > 0 else T.dynamic('num_blocks')
    stride_kvcache_block = cache_block_stride
    num_cores = get_num_vector_cores()
    num_groups = 8
    num_sf_slots = block_m * num_groups
    num_per32_slots = num_sf_slots * 2
    vec_size = 64
    if store_lanes_per_row not in (1, 2, 4, 8, 16):
        raise ValueError('store_lanes_per_row must be one of 1, 2, 4, 8, 16')
    value_bytes = 448 + 64 * 2
    value_qwords = value_bytes // 8
    scale_qwords = 8 // 8
    combined_qwords_per_block = block_size * (value_qwords + scale_qwords)
    kv_ub_stride = 512

    @T.prim_func
    def _store_kv_cache_ascend(
        kv: T.Tensor((n, 512), 'bfloat16'),
        kv_cache_value: T.StridedTensor(
            (num_blocks, block_size * (value_bytes + 8)), (stride_kvcache_block, 1), T.uint8
        ),
        slot_mapping: T.Tensor((n,), 'int32'),
    ) -> None:
        T.assume(stride_kvcache_block % 32 == 0)
        if full_tiles and is_pto:
            T.assume(n % block_m == 0)
        kv_cache_block_u64 = T.view(
            kv_cache_value,
            shape=(num_blocks, combined_qwords_per_block),
            dtype=T.uint64,
            strides=(stride_kvcache_block // 8, 1),
        )
        with T.Kernel(num_cores) as core_id:
            kv_ub = T.alloc_shared((block_m, kv_ub_stride), T.bfloat16)
            slot_ub = T.alloc_shared((block_m,), T.int32)
            amax_per32_ub = T.alloc_shared((num_per32_slots,), T.bfloat16)
            sf_ub = T.alloc_shared((num_sf_slots,), T.uint8)
            kv_fp8_ub = T.view(kv_ub, shape=(block_m, kv_ub_stride * 2), dtype=T.float8_e4m3fn)
            kv_u64_ub = T.view(kv_ub, shape=(block_m, kv_ub_stride // 4), dtype=T.uint64)
            T.annotate_buffer_versions({kv_ub: num_stages, slot_ub: 1})
            for token_tile in T.Persistent(
                [T.ceildiv(n, block_m)], num_cores, core_id, group_size=1, num_stages=num_stages
            ):
                token_base = token_tile * block_m
                valid_rows = block_m if full_tiles else T.min(block_m, n - token_base)
                T.copy(
                    kv[token_base : token_base + valid_rows, 0:512],
                    kv_ub[:valid_rows, 0:512],
                    l2_cache_ctrl='NOTALLOC_KEEP',
                )
                T.copy(slot_mapping[token_base : token_base + valid_rows], slot_ub[:valid_rows])
                with T.SimdVF():
                    if not full_tiles:
                        for sf_batch in T.unroll(num_per32_slots // 128, explicit=True):
                            S.vsts(amax_per32_ub[sf_batch * 128], S.vdup(0.0, T.bfloat16))
                    amax_store_mask = S.pset(16, 'PAT_VL8')
                    amax_math_mask = S.pset(16, 'PAT_ALL')
                    amax_scatter_index = T.reinterpret(S.vci(0, T.int16), 'uint16x128')
                    abs_mask_u16 = S.vdup(0x7FFF, T.uint16, amax_math_mask)
                    for row in T.serial(block_m):
                        if full_tiles or row < valid_rows:
                            for half in T.unroll(2, explicit=True):
                                values0, values1 = S.vld2(kv_ub[row, half * 256], dist='DINTLV_B16')
                                abs0_u16 = S.vand(T.reinterpret(values0, 'uint16x128'), abs_mask_u16, amax_math_mask)
                                abs1_u16 = S.vand(T.reinterpret(values1, 'uint16x128'), abs_mask_u16, amax_math_mask)
                                abs_max_u16 = S.vmax(abs0_u16, abs1_u16, amax_math_mask)
                                amax_u16 = S.vcgmax(abs_max_u16, amax_math_mask)
                                S.vscatter(
                                    T.reinterpret(amax_u16, 'bfloat16x128'),
                                    amax_per32_ub[row * num_groups * 2 + half * 8],
                                    amax_scatter_index,
                                    amax_store_mask,
                                )
                with T.SimdVF():
                    b16_all = S.pset(16, 'PAT_ALL')
                    inv_bias = S.vdup(254, T.uint16, b16_all)
                    inf_exp = S.vdup(255, T.uint16, b16_all)
                    nan_exp = S.vdup(256, T.uint16, b16_all)
                    for scale_batch in T.unroll(num_sf_slots // 128, explicit=True):
                        pair_offset = scale_batch * 256
                        per32_even, per32_odd = S.vld2(amax_per32_ub[pair_offset], dist='DINTLV_B16')
                        per64_bits = S.vmax(
                            T.reinterpret(per32_even, 'uint16x128'), T.reinterpret(per32_odd, 'uint16x128'), b16_all
                        )
                        clamped_bits = S.vmaxs(per64_bits, 0x38D2, b16_all)
                        normal_exponent = S.vshrs(S.vadds(clamped_bits, -993, b16_all), 7, b16_all)
                        special_mask = S.vcmps(clamped_bits, 0x7F80, b16_all, op='ge')
                        nan_mask = S.vcmps(clamped_bits, 0x7F80, special_mask, op='gt')
                        special_exponent = S.vsel(inf_exp, normal_exponent, special_mask)
                        scale_exponent = S.vsel(nan_exp, special_exponent, nan_mask)
                        sf_offset = scale_batch * 128
                        S.vsts(sf_ub[sf_offset], scale_exponent, b16_all, dist='PK_B16')
                        inverse_exponent = S.vsub(inv_bias, scale_exponent, b16_all)
                        inverse_bf16 = T.reinterpret(S.vshls(inverse_exponent, 7, b16_all), 'bfloat16x128')
                        S.vsts(amax_per32_ub[sf_offset], inverse_bf16, b16_all, dist='NORM_B16')
                with T.SimdVF():
                    bf16_vl64 = S.pset(16, 'PAT_VL64')
                    bf16_all = S.pset(16, 'PAT_ALL')
                    fp8_all = S.pset(8, 'PAT_ALL')
                    for row in T.serial(block_m):
                        if full_tiles or row < valid_rows:
                            raw_bf16_tail = S.vld(kv_ub[row, 448])
                            for group in T.unroll(num_groups - 1, explicit=True):
                                values_bf16 = S.vld(kv_ub[row, group * vec_size], dist='UNPK_B16')
                                sf_inv = S.vld(amax_per32_ub[row * num_groups + group], dist='BRC_B16')
                                values = S.vcvt(S.vmul(values_bf16, sf_inv, bf16_all), T.float32, mask=bf16_all, part=0)
                                quantized = S.vcvt(values, T.float8_e4m3fn, mask=fp8_all)
                                S.vsts(kv_fp8_ub[row, group * vec_size], quantized, fp8_all, dist='PK4_B32')
                            S.vsts(kv_ub[row, 224], raw_bf16_tail, bf16_vl64)
                with T.SimtVF(threads=block_m * store_lanes_per_row):
                    store_thread = T.get_thread_binding()
                    store_row = store_thread // store_lanes_per_row
                    store_lane = store_thread - store_row * store_lanes_per_row
                    if full_tiles or store_row < valid_rows:
                        slot = T.int32(slot_ub[store_row])
                        T.assume(slot >= -1)
                        T.assume(slot < num_blocks * block_size)
                        if slot != -1:
                            slot_u = T.cast(slot, 'uint32')
                            cache_block = slot_u // block_size
                            if static_num_blocks > 0:
                                cache_offset = slot_u % block_size
                            else:
                                cache_offset = slot_u - cache_block * block_size
                            value_row_base = T.alloc_var(T.uint32, init=cache_offset * value_qwords)
                            T.assume(value_row_base + value_qwords <= combined_qwords_per_block)
                            for store_round in T.unroll(T.ceildiv(value_qwords, store_lanes_per_row), explicit=True):
                                store_word = store_lane + store_round * store_lanes_per_row
                                if store_word < value_qwords:
                                    kv_cache_block_u64[cache_block, value_row_base + store_word] = kv_u64_ub[
                                        store_row, store_word
                                    ]
                            if store_lane < num_groups:
                                kv_cache_value[
                                    cache_block, block_size * value_bytes + cache_offset * num_groups + store_lane
                                ] = sf_ub[store_row * num_groups + store_lane]

    return _store_kv_cache_ascend


@tilelang.jit(out_idx=None, pass_configs={tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True})
def _store_kv_cache_per32_tl_ascend(block_size: int, full_tiles: bool = False, block_m: int = 64) -> object:
    """Ascend fused per-32 FP8 quantization and paged-cache store."""
    n = T.dynamic('n')
    num_blocks = T.dynamic('num_blocks')
    stride_kvcache_block = T.dynamic('stride_kvcache_block')
    stride_kvcache_fp8 = T.dynamic('stride_kvcache_fp8')
    stride_kvcache_sf = T.dynamic('stride_kvcache_sf')
    num_cores = get_num_vector_cores()
    num_stages = 2
    num_groups = 16
    num_scale_slots = block_m * num_groups
    store_lanes_per_row = 16
    fp8_qwords = 512 // 8
    scale_qwords = 16 // 8
    value_qwords = fp8_qwords + scale_qwords

    @T.prim_func
    def _store_kv_cache_per32_ascend(
        kv: T.Tensor((n, 512), 'bfloat16'),
        kv_cache_fp8: T.StridedTensor(
            (num_blocks, block_size, 512), (stride_kvcache_block, stride_kvcache_fp8, 1), T.float8_e4m3fn
        ),
        kv_cache_use_packed_ue8m0: T.StridedTensor(
            (num_blocks, block_size, 16), (stride_kvcache_block, stride_kvcache_sf, 1), 'uint8'
        ),
        slot_mapping: T.Tensor((n,), 'int32'),
    ) -> None:
        T.assume(stride_kvcache_block % 32 == 0)
        T.assume(stride_kvcache_fp8 % 32 == 0)
        T.assume(stride_kvcache_sf % 16 == 0)
        kv_cache_fp8_u64 = T.view(
            kv_cache_fp8,
            shape=(num_blocks, block_size, fp8_qwords),
            dtype=T.uint64,
            strides=(stride_kvcache_block // 8, stride_kvcache_fp8 // 8, 1),
        )
        kv_cache_sf_u64 = T.view(
            kv_cache_use_packed_ue8m0,
            shape=(num_blocks, block_size, scale_qwords),
            dtype=T.uint64,
            strides=(stride_kvcache_block // 8, stride_kvcache_sf // 8, 1),
        )
        with T.Kernel(num_cores) as core_id:
            kv_ub = T.alloc_shared((block_m, 512), T.bfloat16)
            slot_ub = T.alloc_shared((block_m,), T.int32)
            amax_ub = T.alloc_shared((num_scale_slots,), T.float32)
            sf_ub = T.alloc_shared((num_scale_slots,), T.uint8)
            sf_inv_ub = T.alloc_shared((num_scale_slots,), T.bfloat16)
            kv_fp8_ub = T.view(kv_ub, shape=(block_m, 1024), dtype=T.float8_e4m3fn)
            kv_u64_ub = T.view(kv_ub, shape=(block_m, 128), dtype=T.uint64)
            sf_u64_ub = T.view(sf_ub, shape=(block_m, scale_qwords), dtype=T.uint64)
            T.annotate_buffer_versions(
                {kv_ub: num_stages, slot_ub: num_stages, amax_ub: num_stages, sf_ub: num_stages, sf_inv_ub: num_stages}
            )
            for token_tile in T.Persistent(
                [T.ceildiv(n, block_m)], num_cores, core_id, group_size=1, num_stages=num_stages
            ):
                token_base = token_tile * block_m
                valid_rows = block_m if full_tiles else T.min(block_m, n - token_base)
                T.copy(
                    kv[token_base : token_base + valid_rows, 0:512],
                    kv_ub[:valid_rows, 0:512],
                    l2_cache_ctrl='NOTALLOC_KEEP',
                )
                T.copy(slot_mapping[token_base : token_base + valid_rows], slot_ub[:valid_rows])
                with T.SimdVF():
                    if not full_tiles:
                        for scale_batch in T.unroll(num_scale_slots // 64, explicit=True):
                            S.vsts(amax_ub[scale_batch * 64], S.vdup(0.0, T.float32))
                    amax_store_mask = S.pset(32, 'PAT_VL8')
                    abs_mask_u16 = S.vdup(0x7FFF, T.uint16)
                    zero_bf16 = S.vdup(0.0, T.bfloat16)
                    for row in T.serial(block_m):
                        if full_tiles or row < valid_rows:
                            for half in T.unroll(2, explicit=True):
                                values0, values1 = S.vld2(kv_ub[row, half * 256], dist='DINTLV_B16')
                                abs0_u16 = S.vand(T.reinterpret(values0, 'uint16x128'), abs_mask_u16)
                                abs1_u16 = S.vand(T.reinterpret(values1, 'uint16x128'), abs_mask_u16)
                                amax_u16 = S.vcgmax(S.vmax(abs0_u16, abs1_u16))
                                amax_low, _amax_high = S.vintlv(zero_bf16, T.reinterpret(amax_u16, 'bfloat16x128'))
                                S.vsts(
                                    amax_ub[row * num_groups + half * 8],
                                    T.reinterpret(amax_low, 'float32x64'),
                                    amax_store_mask,
                                    dist='NORM_B32',
                                )
                with T.SimdVF():
                    scale_store_mask = S.pset(32, 'PAT_ALL')
                    for scale_batch in T.unroll(num_scale_slots // 64, explicit=True):
                        scale_offset = scale_batch * 64
                        clamped_amax = S.vmaxs(S.vld(amax_ub[scale_offset]), 0.0001)
                        scale_raw = S.vmuls(clamped_amax, 1.0 / 448.0)
                        scale_bits = T.reinterpret(scale_raw, 'uint32x64')
                        scale_exponent = S.vshrs(S.vadds(scale_bits, 0x7FFFFF), 23)
                        inverse_exponent = S.vsub(S.vdup(254, T.uint32), scale_exponent)
                        inverse_bf16 = T.reinterpret(S.vshls(inverse_exponent, 7), 'bfloat16x128')
                        S.vsts(sf_ub[scale_offset], scale_exponent, dist='PK4_B32')
                        S.vsts(sf_inv_ub[scale_offset], inverse_bf16, scale_store_mask, dist='PK_B32')
                with T.SimdVF():
                    for row in T.serial(block_m):
                        if full_tiles or row < valid_rows:
                            for half in T.unroll(2, explicit=True):
                                src_even, src_odd = S.vld2(kv_ub[row, half * 256], dist='DINTLV_B16')
                                scale_e2b = S.vld(sf_inv_ub[row * num_groups + half * 8], dist='E2B_B16')
                                scaled_even = S.vmul(src_even, scale_e2b)
                                scaled_odd = S.vmul(src_odd, scale_e2b)
                                src_even_even = S.vcvt(scaled_even, T.float32, part=0)
                                src_even_odd = S.vcvt(scaled_even, T.float32, part=1)
                                src_odd_even = S.vcvt(scaled_odd, T.float32, part=0)
                                src_odd_odd = S.vcvt(scaled_odd, T.float32, part=1)
                                packed_p0 = S.vcvt(src_even_even, T.float8_e4m3fn, part=0)
                                packed_p1 = S.vcvt(src_odd_even, T.float8_e4m3fn, part=1)
                                packed_p2 = S.vcvt(src_even_odd, T.float8_e4m3fn, part=2)
                                packed_p3 = S.vcvt(src_odd_odd, T.float8_e4m3fn, part=3)
                                packed_window_u8 = S.vor(
                                    S.vor(T.reinterpret(packed_p0, 'uint8x256'), T.reinterpret(packed_p1, 'uint8x256')),
                                    S.vor(T.reinterpret(packed_p2, 'uint8x256'), T.reinterpret(packed_p3, 'uint8x256')),
                                )
                                S.vsts(
                                    kv_fp8_ub[row, half * 256],
                                    T.reinterpret(packed_window_u8, 'float8_e4m3fnx256'),
                                    dist='NORM_B8',
                                )
                with T.SimtVF(threads=block_m * store_lanes_per_row):
                    store_thread = T.get_thread_binding()
                    store_row = store_thread // store_lanes_per_row
                    store_lane = store_thread - store_row * store_lanes_per_row
                    if full_tiles or store_row < valid_rows:
                        slot = T.int32(slot_ub[store_row])
                        T.assume(slot >= -1)
                        T.assume(slot < num_blocks * block_size)
                        if slot != -1:
                            slot_u = T.cast(slot, 'uint32')
                            cache_block = slot_u // block_size
                            cache_offset = slot_u - cache_block * block_size
                            for store_round in T.unroll(T.ceildiv(value_qwords, store_lanes_per_row), explicit=True):
                                store_word = store_lane + store_round * store_lanes_per_row
                                if store_word < value_qwords:
                                    if store_word < fp8_qwords:
                                        kv_cache_fp8_u64[cache_block, cache_offset, store_word] = kv_u64_ub[
                                            store_row, store_word
                                        ]
                                    else:
                                        scale_word = store_word - fp8_qwords
                                        kv_cache_sf_u64[cache_block, cache_offset, scale_word] = sf_u64_ub[
                                            store_row, scale_word
                                        ]

    return _store_kv_cache_per32_ascend


@tilelang.jit(out_idx=None, pass_configs={tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True})
def _load_kv_cache_tl_ascend(block_size: int, num_stages: int, full_tiles: bool = False, block_m: int = 32) -> object:
    """Ascend paged-cache load and per-64 FP8 dequantization."""
    is_pto = _is_pto_target()
    n = T.dynamic('n')
    num_blocks = T.dynamic('num_blocks')
    stride_kvcache_block = T.dynamic('stride_kvcache_block')
    stride_kvcache_value = T.dynamic('stride_kvcache_value')
    stride_kvcache_sf = T.dynamic('stride_kvcache_sf')
    num_cores = get_num_vector_cores()
    num_groups = 8
    value_bytes = 448 + 64 * 2
    block_size_is_power_of_two = block_size > 0 and block_size & block_size - 1 == 0
    nan_lut_bits = 0x0000 if is_pto else 0x7F81

    @T.prim_func
    def _load_kv_cache_ascend(
        kv: T.Tensor((n, 512), 'bfloat16'),
        kv_cache_value: T.StridedTensor(
            (num_blocks, block_size, value_bytes), (stride_kvcache_block, stride_kvcache_value, 1), 'uint8'
        ),
        kv_cache_use_packed_ue8m0: T.StridedTensor(
            (num_blocks, block_size, 8), (stride_kvcache_block, stride_kvcache_sf, 1), 'uint8'
        ),
        slot_mapping: T.Tensor((n,), 'int32'),
    ) -> None:
        T.assume(stride_kvcache_block % 32 == 0)
        T.assume(stride_kvcache_value % 32 == 0)
        T.assume(stride_kvcache_sf % 8 == 0)
        kv_u8 = T.view(kv, shape=(n, 1024), dtype=T.uint8)
        kv_cache_sf_u64 = T.view(
            kv_cache_use_packed_ue8m0,
            shape=(num_blocks, block_size, 1),
            dtype=T.uint64,
            strides=(stride_kvcache_block // 8, stride_kvcache_sf // 8, 1),
        )
        with T.Kernel(num_cores, prelude=None if is_pto else _ASCEND_VSTSX2_BF16_PRELUDE) as core_id:
            slot_ub = T.alloc_shared((block_m,), T.int32)
            sf_raw_ub = T.alloc_shared((block_m, num_groups), T.uint8)
            sf_bf16_ub = T.alloc_shared((block_m, num_groups), T.bfloat16)
            raw_u8_ub = T.alloc_shared((block_m, value_bytes), T.uint8)
            raw_bf16_ub = T.view(raw_u8_ub, shape=(block_m, value_bytes // 2), dtype=T.bfloat16)
            out_ub = T.alloc_shared((block_m, 512), T.bfloat16)
            lut_raw_ub = T.alloc_shared((256,), T.uint8)
            lut_fp8_ub = T.view(lut_raw_ub, shape=(256,), dtype=T.float8_e4m3fn)
            lut_bf16_ub = T.alloc_shared((256,), T.bfloat16)
            lut_bf16_u16_ub = T.view(lut_bf16_ub, shape=(256,), dtype=T.uint16)
            lut_bf16_u8_ub = T.view(lut_bf16_ub, shape=(512,), dtype=T.uint8)
            lut_bf16_low_u8_ub = T.alloc_shared((256,), T.uint8)
            lut_bf16_high_u8_ub = T.alloc_shared((256,), T.uint8)
            sf_raw_u64_ub = T.view(sf_raw_ub, shape=(block_m, 1), dtype=T.uint64)
            T.annotate_buffer_versions(
                {
                    slot_ub: num_stages,
                    sf_raw_ub: num_stages,
                    sf_bf16_ub: num_stages,
                    raw_u8_ub: num_stages,
                    out_ub: num_stages,
                }
            )
            with T.SimtVF(threads=256):
                init_linear = T.get_thread_binding()
                lut_raw_ub[init_linear] = T.cast(init_linear, T.uint8)
            with T.SimdVF():
                for lut_part in T.unroll(4, explicit=True):
                    lut_fp8 = S.vld(lut_fp8_ub[lut_part * 64], dist='UNPK4_B8')
                    lut_f32 = S.vcvt(lut_fp8, T.float32)
                    lut_bf16 = S.vcvt(lut_f32, T.bfloat16)
                    S.vsts(lut_bf16_ub[lut_part * 64], lut_bf16, dist='PK_B32')
            with T.SimtVF(threads=2):
                nan_sign = T.get_thread_binding()
                lut_bf16_u16_ub[127 + nan_sign * 128] = T.cast(nan_lut_bits, T.uint16)
            with T.SimdVF():
                all_b8 = S.pset(8)
                lut_low_init, lut_high_init = S.vld2(lut_bf16_u8_ub[0], dist='DINTLV_B8')
                S.vsts(lut_bf16_low_u8_ub[0], lut_low_init, all_b8, dist='NORM_B8')
                S.vsts(lut_bf16_high_u8_ub[0], lut_high_init, all_b8, dist='NORM_B8')
            for token_tile in T.Persistent(
                [T.ceildiv(n, block_m)], num_cores, core_id, group_size=1, num_stages=num_stages
            ):
                token_base = token_tile * block_m
                valid_rows = block_m if full_tiles else T.min(block_m, n - token_base)
                T.copy(slot_mapping[token_base : token_base + valid_rows], slot_ub[:valid_rows])
                for row in T.serial(block_m):
                    if full_tiles or row < valid_rows:
                        slot = T.alloc_var(T.int32, init=slot_ub[row])
                        T.assume(slot >= -1)
                        T.assume(slot < num_blocks * block_size)
                        if slot != -1:
                            slot_u = T.alloc_var(T.uint32, init=T.cast(slot, 'uint32'))
                            cache_block = T.alloc_var(T.uint32, init=slot_u // block_size)
                            cache_offset = T.alloc_var(
                                T.uint32,
                                init=slot_u & block_size - 1
                                if block_size_is_power_of_two
                                else slot_u - cache_block * block_size,
                            )
                            T.assume(cache_block < T.cast(num_blocks, T.uint32))
                            T.assume(cache_offset < T.cast(block_size, T.uint32))
                            T.copy(
                                kv_cache_value[cache_block, cache_offset, 0:value_bytes], raw_u8_ub[row, 0:value_bytes]
                            )
                with T.SimtVF(threads=block_m):
                    load_row = T.get_thread_binding()
                    if full_tiles or load_row < valid_rows:
                        load_slot = T.int32(slot_ub[load_row])
                        T.assume(load_slot >= -1)
                        T.assume(load_slot < num_blocks * block_size)
                        if load_slot != -1:
                            load_slot_u = T.cast(load_slot, 'uint32')
                            load_block = load_slot_u // block_size
                            load_offset = (
                                load_slot_u & block_size - 1
                                if block_size_is_power_of_two
                                else load_slot_u - load_block * block_size
                            )
                            T.assume(load_block < T.cast(num_blocks, T.uint32))
                            T.assume(load_offset < T.cast(block_size, T.uint32))
                            sf_raw_u64_ub[load_row, 0] = kv_cache_sf_u64[load_block, load_offset, 0]
                with T.SimdVF():
                    for scale_batch in T.unroll(block_m * num_groups // 128, explicit=True):
                        scale_row = scale_batch * (128 // num_groups)
                        scale_bytes = S.vld(sf_raw_ub[scale_row, 0], dist='UNPK_B8')
                        scale_u16 = T.reinterpret(scale_bytes, 'uint16x128')
                        scale_bits = S.vshls(scale_u16, 7)
                        scale_bf16 = T.reinterpret(scale_bits, 'bfloat16x128')
                        S.vsts(sf_bf16_ub[scale_row, 0], scale_bf16, dist='NORM_B16')
                with T.SimdVF():
                    zero = S.vdup(0.0, T.bfloat16)
                    all_b16 = S.pset(16)
                    bf16_vl64 = S.pset(16, 'PAT_VL64')
                    lut_low_bytes = S.vld(lut_bf16_low_u8_ub[0], dist='NORM')
                    lut_high_bytes = S.vld(lut_bf16_high_u8_ub[0], dist='NORM')
                    for row in T.serial(block_m):
                        if full_tiles or row < valid_rows:
                            if slot_ub[row] != -1:
                                raw_even, raw_odd = S.vld2(raw_u8_ub[row, 0], dist='DINTLV_B8')
                                scale_compact = S.vld(sf_bf16_ub[row, 0], dist='E2B_B16')
                                scale_g0, scale_g1 = S.vintlv(scale_compact, scale_compact)
                                low_even = S.vselr(lut_low_bytes, raw_even)
                                high_even = S.vselr(lut_high_bytes, raw_even)
                                even_bytes_g0, even_bytes_g1 = S.vintlv(low_even, high_even)
                                low_odd = S.vselr(lut_low_bytes, raw_odd)
                                high_odd = S.vselr(lut_high_bytes, raw_odd)
                                odd_bytes_g0, odd_bytes_g1 = S.vintlv(low_odd, high_odd)
                                value_even_g0 = T.reinterpret(even_bytes_g0, 'bfloat16x128')
                                value_even_g1 = T.reinterpret(even_bytes_g1, 'bfloat16x128')
                                value_odd_g0 = T.reinterpret(odd_bytes_g0, 'bfloat16x128')
                                value_odd_g1 = T.reinterpret(odd_bytes_g1, 'bfloat16x128')
                                dequant_even_g0 = S.vmul(value_even_g0, scale_g0)
                                dequant_odd_g0 = S.vmul(value_odd_g0, scale_g0)
                                dequant_even_g1 = S.vmul(value_even_g1, scale_g1)
                                dequant_odd_g1 = S.vmul(value_odd_g1, scale_g1)
                                _vstsx2_bf16(out_ub[row, 0], dequant_even_g0, dequant_odd_g0, all_b16)
                                _vstsx2_bf16(out_ub[row, 256], dequant_even_g1, dequant_odd_g1, all_b16)
                            else:
                                for vector in T.unroll(4, explicit=True):
                                    S.vsts(out_ub[row, vector * 128], zero)
                                S.vsts(raw_bf16_ub[row, 224], zero, bf16_vl64)
                T.copy(out_ub[:valid_rows, 0:448], kv[token_base : token_base + valid_rows, 0:448])
                T.copy(raw_u8_ub[:valid_rows, 448:576], kv_u8[token_base : token_base + valid_rows, 896:1024])

    return _load_kv_cache_ascend


@tilelang.jit(out_idx=None)
def _load_kv_cache_per32_tl_ascend(
    block_size: int,
    full_tiles: bool = False,
    block_m: int = 32,
    cache_block_stride: int = 0,
    prefetch_slot_waves: bool = False,
) -> object:
    """Ascend per-32 FP8 dequantization in native DINTLV byte layout.

    ``cache_block_stride`` is read from the runtime tensor and used as a
    TileLang JIT specialization parameter. The 512/16 inner strides are fixed
    by the packed views. Specializing the outer stride removes dynamic address
    arithmetic from the random cache loads without assuming allocator layout.
    """
    if cache_block_stride <= 0 or cache_block_stride % 32 != 0:
        raise ValueError('cache_block_stride must be a positive 32-byte multiple')
    if prefetch_slot_waves and (block_m != 64 or not full_tiles):
        raise ValueError('slot-wave specialization requires full N=65536/B64 tiles')
    n = 65536 if prefetch_slot_waves else T.dynamic('n')
    num_blocks = T.dynamic('num_blocks')
    stride_kvcache_block = cache_block_stride
    stride_kvcache_fp8 = 512
    stride_kvcache_sf = 16
    num_cores = get_num_vector_cores()
    num_stages = 3
    num_groups = 16
    scale_qwords = num_groups // 8
    is_pto = _is_pto_target()
    nan_lut_bits = 0x0000 if is_pto else 0x7F81
    token_tiles = 65536 // block_m if prefetch_slot_waves else 0
    # Preserve the tuned 4,864-byte slot-wave region; shrinking it changes the
    # UB offsets of every following buffer and can alter generated performance.
    slot_wave_buffer_rows = 19 if prefetch_slot_waves else 0

    @T.macro
    def _store_ue8m0x4_prmt(dst, row, col, packed):
        dst[row, col] = (
            T.call_extern('uint32', 'pto.prmt', packed, T.int32(0), T.int32(_PRMT_UE8M0_PAIR01))
            << _UE8M0_TO_BF16_EXP_SHIFT
        )
        dst[row, col + 1] = (
            T.call_extern('uint32', 'pto.prmt', packed, T.int32(0), T.int32(_PRMT_UE8M0_PAIR23))
            << _UE8M0_TO_BF16_EXP_SHIFT
        )

    @T.macro
    def _expand_ue8m0x4_swar(packed):
        spread_u16 = (packed | packed << _BF16_BITS) & T.uint64(_SWAR_KEEP_U16_PAIRS)
        return (spread_u16 | spread_u16 << _BYTE_BITS) & T.uint64(_SWAR_KEEP_U8_LANES)

    @T.macro
    def _as_bf16(value):
        return T.reinterpret(value, 'bfloat16x128')

    @T.macro
    def _mul_pair(value_even, value_odd, scale):
        return (S.vmul(value_even, scale), S.vmul(value_odd, scale))

    @T.macro
    def _zero_bf16_row(dst, row, zero):
        for vector in T.unroll(4, explicit=True):
            S.vsts(dst[row * 512 + vector * 128], zero)

    @T.prim_func
    def _load_kv_cache_per32_ascend(
        kv: T.Tensor((n, 512), 'bfloat16'),
        kv_cache_fp8: T.StridedTensor(
            (num_blocks, block_size, 512), (stride_kvcache_block, stride_kvcache_fp8, 1), T.float8_e4m3fn
        ),
        kv_cache_use_packed_ue8m0: T.StridedTensor(
            (num_blocks, block_size, 16), (stride_kvcache_block, stride_kvcache_sf, 1), 'uint8'
        ),
        slot_mapping: T.Tensor((n,), 'int32'),
    ) -> None:
        kv_flat = T.view(kv, shape=(n * 512,), dtype=T.bfloat16)
        kv_cache_fp8_u8 = T.view(
            kv_cache_fp8,
            shape=(num_blocks, block_size, 512),
            dtype=T.uint8,
            strides=(stride_kvcache_block, stride_kvcache_fp8, 1),
        )
        kv_cache_sf_u64 = T.view(
            kv_cache_use_packed_ue8m0,
            shape=(num_blocks, block_size, scale_qwords),
            dtype=T.uint64,
            strides=(stride_kvcache_block // 8, stride_kvcache_sf // 8, 1),
        )
        with T.Kernel(num_cores, prelude=None if is_pto else _ASCEND_VSTSX2_BF16_PRELUDE) as core_id:
            if prefetch_slot_waves:
                slot_wave_ub = T.alloc_shared((slot_wave_buffer_rows, block_m), T.int32)
            else:
                slot_ub = T.alloc_shared((block_m,), T.int32)
            sf_bf16_ub = T.alloc_shared((block_m, num_groups), T.bfloat16)
            sf_bf16_u64_ub = T.view(sf_bf16_ub, shape=(block_m, num_groups // 4), dtype=T.uint64)
            sf_bf16_u32_ub = T.view(sf_bf16_ub, shape=(block_m, num_groups // 2), dtype=T.uint32)
            out_ub = T.alloc_shared((block_m * 512 + 48,), T.bfloat16)
            fp8_u8_ub = T.view(out_ub, shape=(block_m * 1024 + 96,), dtype=T.uint8)
            lut_raw_ub = T.alloc_shared((256,), T.uint8)
            lut_fp8_ub = T.view(lut_raw_ub, shape=(256,), dtype=T.float8_e4m3fn)
            lut_bf16_ub = T.alloc_shared((256,), T.bfloat16)
            lut_bf16_u16_ub = T.view(lut_bf16_ub, shape=(256,), dtype=T.uint16)
            lut_bf16_u8_ub = T.view(lut_bf16_ub, shape=(512,), dtype=T.uint8)
            lut_bf16_low_u8_ub = T.alloc_shared((256,), T.uint8)
            lut_bf16_high_u8_ub = T.alloc_shared((256,), T.uint8)
            if prefetch_slot_waves:
                T.annotate_buffer_versions({sf_bf16_ub: num_stages, out_ub: num_stages})
            else:
                T.annotate_buffer_versions({slot_ub: num_stages, sf_bf16_ub: num_stages, out_ub: num_stages})
            with T.SimtVF(threads=256):
                init_linear = T.get_thread_binding()
                lut_raw_ub[init_linear] = T.cast(init_linear, T.uint8)
            with T.SimdVF():
                for lut_part in T.unroll(4, explicit=True):
                    lut_fp8 = S.vld(lut_fp8_ub[lut_part * 64], dist='UNPK4_B8')
                    lut_f32 = S.vcvt(lut_fp8, T.float32)
                    lut_bf16 = S.vcvt(lut_f32, T.bfloat16)
                    S.vsts(lut_bf16_ub[lut_part * 64], lut_bf16, dist='PK_B32')
            with T.SimtVF(threads=2):
                nan_sign = T.get_thread_binding()
                lut_bf16_u16_ub[127 + nan_sign * 128] = T.cast(nan_lut_bits, T.uint16)
            with T.SimdVF():
                all_b8 = S.pset(8)
                lut_low_init, lut_high_init = S.vld2(lut_bf16_u8_ub[0], dist='DINTLV_B8')
                S.vsts(lut_bf16_low_u8_ub[0], lut_low_init, all_b8, dist='NORM_B8')
                S.vsts(lut_bf16_high_u8_ub[0], lut_high_init, all_b8, dist='NORM_B8')
            if prefetch_slot_waves:
                slot_wave_count = T.cast(
                    (T.int32(token_tiles) + T.int32(num_cores - 1) - T.int32(core_id)) // T.int32(num_cores), T.int32
                )
                slot_wave_src = T.make_tensor(
                    T.address_of(slot_mapping[T.int32(core_id) * block_m]),
                    (slot_wave_buffer_rows, block_m),
                    'int32',
                    strides=(num_cores * block_m, 1),
                )
                T.copy(slot_wave_src[0:slot_wave_count, 0:block_m], slot_wave_ub[0:slot_wave_count, 0:block_m])
            for token_tile in T.Persistent(
                [T.ceildiv(n, block_m)], num_cores, core_id, group_size=1, num_stages=num_stages
            ):
                token_base = token_tile * block_m
                valid_rows = block_m if full_tiles else T.min(block_m, n - token_base)
                if prefetch_slot_waves:
                    slot_wave = (token_tile - core_id) // num_cores
                else:
                    T.copy(slot_mapping[token_base : token_base + valid_rows], slot_ub[:valid_rows])
                for load_group in T.serial(block_m // 8):
                    for load_lane in T.unroll(8, explicit=True):
                        row = load_group * 8 + load_lane
                        if full_tiles or row < valid_rows:
                            if prefetch_slot_waves:
                                slot = T.alloc_var(T.int32, init=slot_wave_ub[slot_wave, row])
                            else:
                                slot = T.alloc_var(T.int32, init=slot_ub[row])
                            T.assume(slot >= -1)
                            T.assume(slot < num_blocks * block_size)
                            safe_slot = T.alloc_var(T.int32, init=T.max(slot, T.int32(0)))
                            T.assume(safe_slot >= 0)
                            T.assume(safe_slot < num_blocks * block_size)
                            slot_u = T.alloc_var(T.uint32, init=T.cast(safe_slot, 'uint32'))
                            cache_block = T.alloc_var(T.uint32, init=slot_u // block_size)
                            cache_offset = T.alloc_var(T.uint32, init=slot_u - cache_block * block_size)
                            T.copy(
                                kv_cache_fp8_u8[cache_block, cache_offset, 0:512],
                                fp8_u8_ub[row * 1024 : row * 1024 + 512],
                                l2_cache_ctrl='NOTALLOC_KEEP',
                            )
                with T.SimtVF(threads=block_m * scale_qwords):
                    scale_linear = T.get_thread_binding()
                    scale_row = scale_linear // scale_qwords
                    scale_word = scale_linear - scale_row * scale_qwords
                    if full_tiles or scale_row < valid_rows:
                        if prefetch_slot_waves:
                            scale_slot = T.int32(slot_wave_ub[slot_wave, scale_row])
                        else:
                            scale_slot = T.int32(slot_ub[scale_row])
                        T.assume(scale_slot >= -1)
                        T.assume(scale_slot < num_blocks * block_size)
                        scale_safe_slot = T.max(scale_slot, T.int32(0))
                        T.assume(scale_safe_slot >= 0)
                        T.assume(scale_safe_slot < num_blocks * block_size)
                        scale_slot_u = T.cast(scale_safe_slot, 'uint32')
                        scale_block = scale_slot_u // block_size
                        scale_offset = scale_slot_u - scale_block * block_size
                        scale_packed = kv_cache_sf_u64[scale_block, scale_offset, scale_word]
                        if is_pto:
                            scale_low4 = scale_packed & T.uint64(_LOW_U32_MASK)
                            scale_high4 = scale_packed >> 32
                            _store_ue8m0x4_prmt(sf_bf16_u32_ub, scale_row, scale_word * 4, scale_low4)
                            _store_ue8m0x4_prmt(sf_bf16_u32_ub, scale_row, scale_word * 4 + 2, scale_high4)
                        else:
                            scale_low4 = scale_packed & T.uint64(_LOW_U32_MASK)
                            scale_high4 = scale_packed >> 32 & T.uint64(_LOW_U32_MASK)
                            sf_bf16_u64_ub[scale_row, scale_word * 2] = (
                                _expand_ue8m0x4_swar(scale_low4) << _UE8M0_TO_BF16_EXP_SHIFT
                            )
                            sf_bf16_u64_ub[scale_row, scale_word * 2 + 1] = (
                                _expand_ue8m0x4_swar(scale_high4) << _UE8M0_TO_BF16_EXP_SHIFT
                            )
                with T.SimdVF():
                    zero = S.vdup(0.0, T.bfloat16)
                    all_b16 = S.pset(16)
                    lut_low_bytes = S.vld(lut_bf16_low_u8_ub[0], dist='NORM')
                    lut_high_bytes = S.vld(lut_bf16_high_u8_ub[0], dist='NORM')
                    # Keep the explicit four-row schedule: its instruction order bounds
                    # register lifetimes and overlaps LUT, multiply, and store phases.
                    for row_window in T.serial(block_m // 4):
                        row0 = row_window * 4
                        row1 = row0 + 1
                        row2 = row0 + 2
                        row3 = row0 + 3
                        if full_tiles or row3 < valid_rows:
                            raw_even_r0, raw_odd_r0 = S.vld2(fp8_u8_ub[row0 * 1024], dist='DINTLV_B8')
                            scale_g0_r0 = S.vld(sf_bf16_ub[row0, 0], dist='E2B_B16')
                            scale_g1_r0 = S.vld(sf_bf16_ub[row0, 8], dist='E2B_B16')
                            raw_even_r1, raw_odd_r1 = S.vld2(fp8_u8_ub[row1 * 1024], dist='DINTLV_B8')
                            low_even_r0 = S.vselr(lut_low_bytes, raw_even_r0)
                            high_even_r0 = S.vselr(lut_high_bytes, raw_even_r0)
                            even_bytes_g0_r0, even_bytes_g1_r0 = S.vintlv(low_even_r0, high_even_r0)
                            low_odd_r0 = S.vselr(lut_low_bytes, raw_odd_r0)
                            high_odd_r0 = S.vselr(lut_high_bytes, raw_odd_r0)
                            odd_bytes_g0_r0, odd_bytes_g1_r0 = S.vintlv(low_odd_r0, high_odd_r0)
                            value_even_g0_r0 = _as_bf16(even_bytes_g0_r0)
                            value_even_g1_r0 = _as_bf16(even_bytes_g1_r0)
                            value_odd_g0_r0 = _as_bf16(odd_bytes_g0_r0)
                            value_odd_g1_r0 = _as_bf16(odd_bytes_g1_r0)
                            dequant_even_g0_r0, dequant_odd_g0_r0 = _mul_pair(
                                value_even_g0_r0, value_odd_g0_r0, scale_g0_r0
                            )
                            low_even_r1 = S.vselr(lut_low_bytes, raw_even_r1)
                            low_odd_r1 = S.vselr(lut_low_bytes, raw_odd_r1)
                            _vstsx2_bf16(out_ub[row0 * 512 + 0], dequant_even_g0_r0, dequant_odd_g0_r0, all_b16)
                            scale_g0_r1 = S.vld(sf_bf16_ub[row1, 0], dist='E2B_B16')
                            scale_g1_r1 = S.vld(sf_bf16_ub[row1, 8], dist='E2B_B16')
                            dequant_even_g1_r0, dequant_odd_g1_r0 = _mul_pair(
                                value_even_g1_r0, value_odd_g1_r0, scale_g1_r0
                            )
                            high_even_r1 = S.vselr(lut_high_bytes, raw_even_r1)
                            high_odd_r1 = S.vselr(lut_high_bytes, raw_odd_r1)
                            _vstsx2_bf16(out_ub[row0 * 512 + 256], dequant_even_g1_r0, dequant_odd_g1_r0, all_b16)
                            even_bytes_g0_r1, even_bytes_g1_r1 = S.vintlv(low_even_r1, high_even_r1)
                            odd_bytes_g0_r1, odd_bytes_g1_r1 = S.vintlv(low_odd_r1, high_odd_r1)
                            value_even_g0_r1 = _as_bf16(even_bytes_g0_r1)
                            value_even_g1_r1 = _as_bf16(even_bytes_g1_r1)
                            value_odd_g0_r1 = _as_bf16(odd_bytes_g0_r1)
                            value_odd_g1_r1 = _as_bf16(odd_bytes_g1_r1)
                            raw_even_r2, raw_odd_r2 = S.vld2(fp8_u8_ub[row2 * 1024], dist='DINTLV_B8')
                            dequant_even_g0_r1, dequant_odd_g0_r1 = _mul_pair(
                                value_even_g0_r1, value_odd_g0_r1, scale_g0_r1
                            )
                            low_even_r2 = S.vselr(lut_low_bytes, raw_even_r2)
                            low_odd_r2 = S.vselr(lut_low_bytes, raw_odd_r2)
                            _vstsx2_bf16(out_ub[row1 * 512 + 0], dequant_even_g0_r1, dequant_odd_g0_r1, all_b16)
                            scale_g0_r2 = S.vld(sf_bf16_ub[row2, 0], dist='E2B_B16')
                            scale_g1_r2 = S.vld(sf_bf16_ub[row2, 8], dist='E2B_B16')
                            dequant_even_g1_r1, dequant_odd_g1_r1 = _mul_pair(
                                value_even_g1_r1, value_odd_g1_r1, scale_g1_r1
                            )
                            high_even_r2 = S.vselr(lut_high_bytes, raw_even_r2)
                            high_odd_r2 = S.vselr(lut_high_bytes, raw_odd_r2)
                            _vstsx2_bf16(out_ub[row1 * 512 + 256], dequant_even_g1_r1, dequant_odd_g1_r1, all_b16)
                            even_bytes_g0_r2, even_bytes_g1_r2 = S.vintlv(low_even_r2, high_even_r2)
                            odd_bytes_g0_r2, odd_bytes_g1_r2 = S.vintlv(low_odd_r2, high_odd_r2)
                            value_even_g0_r2 = _as_bf16(even_bytes_g0_r2)
                            value_even_g1_r2 = _as_bf16(even_bytes_g1_r2)
                            value_odd_g0_r2 = _as_bf16(odd_bytes_g0_r2)
                            value_odd_g1_r2 = _as_bf16(odd_bytes_g1_r2)
                            raw_even_r3, raw_odd_r3 = S.vld2(fp8_u8_ub[row3 * 1024], dist='DINTLV_B8')
                            dequant_even_g0_r2, dequant_odd_g0_r2 = _mul_pair(
                                value_even_g0_r2, value_odd_g0_r2, scale_g0_r2
                            )
                            low_even_r3 = S.vselr(lut_low_bytes, raw_even_r3)
                            low_odd_r3 = S.vselr(lut_low_bytes, raw_odd_r3)
                            _vstsx2_bf16(out_ub[row2 * 512 + 0], dequant_even_g0_r2, dequant_odd_g0_r2, all_b16)
                            scale_g0_r3 = S.vld(sf_bf16_ub[row3, 0], dist='E2B_B16')
                            scale_g1_r3 = S.vld(sf_bf16_ub[row3, 8], dist='E2B_B16')
                            dequant_even_g1_r2, dequant_odd_g1_r2 = _mul_pair(
                                value_even_g1_r2, value_odd_g1_r2, scale_g1_r2
                            )
                            high_even_r3 = S.vselr(lut_high_bytes, raw_even_r3)
                            high_odd_r3 = S.vselr(lut_high_bytes, raw_odd_r3)
                            _vstsx2_bf16(out_ub[row2 * 512 + 256], dequant_even_g1_r2, dequant_odd_g1_r2, all_b16)
                            even_bytes_g0_r3, even_bytes_g1_r3 = S.vintlv(low_even_r3, high_even_r3)
                            odd_bytes_g0_r3, odd_bytes_g1_r3 = S.vintlv(low_odd_r3, high_odd_r3)
                            value_even_g0_r3 = _as_bf16(even_bytes_g0_r3)
                            value_even_g1_r3 = _as_bf16(even_bytes_g1_r3)
                            value_odd_g0_r3 = _as_bf16(odd_bytes_g0_r3)
                            value_odd_g1_r3 = _as_bf16(odd_bytes_g1_r3)
                            dequant_even_g0_r3, dequant_odd_g0_r3 = _mul_pair(
                                value_even_g0_r3, value_odd_g0_r3, scale_g0_r3
                            )
                            _vstsx2_bf16(out_ub[row3 * 512 + 0], dequant_even_g0_r3, dequant_odd_g0_r3, all_b16)
                            dequant_even_g1_r3, dequant_odd_g1_r3 = _mul_pair(
                                value_even_g1_r3, value_odd_g1_r3, scale_g1_r3
                            )
                            _vstsx2_bf16(out_ub[row3 * 512 + 256], dequant_even_g1_r3, dequant_odd_g1_r3, all_b16)
                            if prefetch_slot_waves:
                                invalid_slots = T.int32(
                                    slot_wave_ub[slot_wave, row0]
                                    | slot_wave_ub[slot_wave, row1]
                                    | slot_wave_ub[slot_wave, row2]
                                    | slot_wave_ub[slot_wave, row3]
                                )
                            else:
                                invalid_slots = T.int32(slot_ub[row0] | slot_ub[row1] | slot_ub[row2] | slot_ub[row3])
                            if invalid_slots < T.int32(0):
                                for invalid_lane in T.unroll(4, explicit=True):
                                    invalid_row = row0 + invalid_lane
                                    if (
                                        slot_wave_ub[slot_wave, invalid_row] == -1
                                        if prefetch_slot_waves
                                        else slot_ub[invalid_row] == -1
                                    ):
                                        _zero_bf16_row(out_ub, invalid_row, zero)
                        else:
                            for window_lane in T.unroll(4, explicit=True):
                                fallback_row = row0 + window_lane
                                if full_tiles or fallback_row < valid_rows:
                                    if (
                                        slot_wave_ub[slot_wave, fallback_row] != -1
                                        if prefetch_slot_waves
                                        else slot_ub[fallback_row] != -1
                                    ):
                                        raw_even, raw_odd = S.vld2(fp8_u8_ub[fallback_row * 1024], dist='DINTLV_B8')
                                        scale_g0 = S.vld(sf_bf16_ub[fallback_row, 0], dist='E2B_B16')
                                        scale_g1 = S.vld(sf_bf16_ub[fallback_row, 8], dist='E2B_B16')
                                        low_even = S.vselr(lut_low_bytes, raw_even)
                                        low_odd = S.vselr(lut_low_bytes, raw_odd)
                                        high_even = S.vselr(lut_high_bytes, raw_even)
                                        high_odd = S.vselr(lut_high_bytes, raw_odd)
                                        even_bytes_g0, even_bytes_g1 = S.vintlv(low_even, high_even)
                                        odd_bytes_g0, odd_bytes_g1 = S.vintlv(low_odd, high_odd)
                                        value_even_g0 = _as_bf16(even_bytes_g0)
                                        value_even_g1 = _as_bf16(even_bytes_g1)
                                        value_odd_g0 = _as_bf16(odd_bytes_g0)
                                        value_odd_g1 = _as_bf16(odd_bytes_g1)
                                        dequant_even_g0 = S.vmul(value_even_g0, scale_g0)
                                        dequant_odd_g0 = S.vmul(value_odd_g0, scale_g0)
                                        dequant_even_g1 = S.vmul(value_even_g1, scale_g1)
                                        dequant_odd_g1 = S.vmul(value_odd_g1, scale_g1)
                                        _vstsx2_bf16(
                                            out_ub[fallback_row * 512 + 0], dequant_even_g0, dequant_odd_g0, all_b16
                                        )
                                        _vstsx2_bf16(
                                            out_ub[fallback_row * 512 + 256], dequant_even_g1, dequant_odd_g1, all_b16
                                        )
                                    else:
                                        _zero_bf16_row(out_ub, fallback_row, zero)
                T.copy(out_ub[0 : valid_rows * 512], kv_flat[token_base * 512 : (token_base + valid_rows) * 512])

    return _load_kv_cache_per32_ascend


@tilelang.jit(out_idx=None)
def _store_index_kv_cache_quant_tl_ascend(
    block_size: int,
    quant_format: str,
    stride_kvcache_data: int,
    stride_kvcache_sf: int,
) -> object:
    """Ascend AIV implementation shared by the three index-cache formats."""
    assert quant_format in ('fp8', 'mxfp8', 'fp4')

    n = T.dynamic('n')
    num_blocks = T.dynamic('num_blocks')
    stride_kvcache_block = T.dynamic('stride_kvcache_block')

    is_fp4 = quant_format == 'fp4'
    is_per32 = quant_format != 'fp8'
    num_groups = 4 if is_per32 else 1
    quant_max = 6.0 if is_fp4 else 448.0
    clamp_min = quant_max * (2**-126) if is_fp4 else 1e-4
    data_dtype = T.float4_e2m1fn if is_fp4 else T.float8_e4m3fn
    sf_dtype = T.uint8 if is_per32 else T.float32
    num_cores = get_num_vector_cores()
    block_m = 128
    num_stages = 2
    data_bytes_per_row = 64 if is_fp4 else 128
    data_qwords = data_bytes_per_row // 8
    store_lanes_per_row = 16

    @T.macro
    def load_bf16_as_fp32(x_ub, row, col):
        return S.vcvt(S.vld(x_ub[row, col], dist='UNPK_B16'), T.float32, part=0)

    @T.macro
    def collect_amax(x_ub, row, amax_ub):
        with T.SimdVF():
            S.vsts(amax_ub[row, 0], S.vdup(0.0, T.float32))
            if is_per32:
                mask_low = S.pset(32, 'PAT_VL32')
                mask_high = S.vcmps(S.vci(0, T.int32), 31, op='gt')
                for vector in T.unroll(2, explicit=True):
                    values = S.vabs(load_bf16_as_fp32(x_ub, row, vector * 64))
                    S.vsts(amax_ub[row, vector * 2], S.vcmax(values, mask_low), dist='ONEPT_B32')
                    S.vsts(amax_ub[row, vector * 2 + 1], S.vcmax(values, mask_high), dist='ONEPT_B32')
            else:
                values0 = S.vabs(load_bf16_as_fp32(x_ub, row, 0))
                values1 = S.vabs(load_bf16_as_fp32(x_ub, row, 64))
                S.vsts(amax_ub[row, 0], S.vcmax(S.vmax(values0, values1)), dist='ONEPT_B32')

    @T.macro
    def compute_sf(amax_ub, row, sf_ub, sf_inv_ub):
        with T.SimdVF():
            amax = S.vmaxs(S.vld(amax_ub[row, 0]), clamp_min)
            scale_raw = S.vmuls(amax, 1.0 / quant_max)
            scale_bits = T.reinterpret(scale_raw, 'uint32x64')
            scale_exponent = S.vadds(S.vshrs(S.vsub(scale_bits, S.vdup(1, T.uint32)), 23), 1)
            inverse_exponent = S.vsub(S.vdup(254, T.uint32), scale_exponent)
            inverse = T.reinterpret(S.vshls(inverse_exponent, 23), 'float32x64')
            S.vsts(sf_inv_ub[row, 0], inverse)
            if is_per32:
                S.vsts(sf_ub[row, 0], scale_exponent, dist='PK4_B32')
            else:
                scale = T.reinterpret(S.vshls(scale_exponent, 23), 'float32x64')
                S.vsts(sf_ub[row, 0], scale)

    @T.macro
    def quantize(x_ub, row, sf_inv_ub, out_ub):
        with T.SimdVF():
            if is_per32:
                expanded = S.vld(sf_inv_ub[row, 0], dist='E2B_B32')
                low, high = S.vintlv(expanded, expanded)
                sf_inv0, sf_inv1 = S.vintlv(low, low)
                sf_inv2, sf_inv3 = S.vintlv(high, high)

                if is_fp4:
                    scaled0 = S.vmul(load_bf16_as_fp32(x_ub, row, 0), sf_inv0)
                    scaled1 = S.vmul(load_bf16_as_fp32(x_ub, row, 64), sf_inv1)
                    low_bits, high_bits = S.vdintlv(
                        T.reinterpret(scaled0, 'uint16x128'),
                        T.reinterpret(scaled1, 'uint16x128'),
                    )
                    scaled_bf16 = T.reinterpret(S.vor(high_bits, S.vmins(low_bits, 1)), 'bfloat16x128')
                    quantized = S.vcvt(scaled_bf16, T.float4_e2m1fn)
                    S.vsts(out_ub[row, 0], quantized, dist='PK4_B32')
                else:
                    sf_invs = S.alloc_local((4,), T.float32)
                    sf_invs[0], sf_invs[1], sf_invs[2], sf_invs[3] = sf_inv0, sf_inv1, sf_inv2, sf_inv3
                    for vector in T.unroll(2, explicit=True):
                        quantized = S.vcvt(
                            S.vmul(load_bf16_as_fp32(x_ub, row, vector * 64), sf_invs[vector]),
                            T.float8_e4m3fn,
                        )
                        S.vsts(out_ub[row, vector * 64], quantized, dist='PK4_B32')
            else:
                sf_inv = S.vld(sf_inv_ub[row, 0], dist='BRC_B32')
                for vector in T.unroll(2, explicit=True):
                    quantized = S.vcvt(
                        S.vmul(load_bf16_as_fp32(x_ub, row, vector * 64), sf_inv),
                        T.float8_e4m3fn,
                    )
                    S.vsts(out_ub[row, vector * 64], quantized, dist='PK4_B32')

    @T.prim_func
    def _store_index_kv_cache_ascend(
        kv: T.Tensor((n, 128), T.bfloat16),  # type: ignore
        kv_cache_data: T.StridedTensor(
            (num_blocks, block_size, 128),
            (
                stride_kvcache_block * 2 if is_fp4 else stride_kvcache_block,
                stride_kvcache_data,
                1,
            ),
            data_dtype,
        ),  # type: ignore
        kv_cache_sf: T.StridedTensor(
            (num_blocks, block_size, num_groups),
            (
                stride_kvcache_block if is_per32 else stride_kvcache_block // 4,
                stride_kvcache_sf,
                1,
            ),
            sf_dtype,
        ),  # type: ignore
        slot_mapping: T.Tensor((n,), T.int32),  # type: ignore
    ) -> None:
        kv_cache_data_u64 = T.view(
            kv_cache_data,
            shape=(num_blocks, block_size, data_qwords),
            dtype=T.uint64,
            strides=(
                stride_kvcache_block // 8,
                stride_kvcache_data // 16 if is_fp4 else stride_kvcache_data // 8,
                1,
            ),
        )
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((block_m, 128), T.bfloat16)
            amax_ub = T.alloc_shared((block_m, 64), T.float32)
            sf_ub = T.alloc_shared((block_m, 64), sf_dtype)
            sf_inv_ub = T.alloc_shared((block_m, 64), T.float32)
            out_ub = T.alloc_shared((block_m, 128), data_dtype)
            slot_ub = T.alloc_shared((block_m,), T.int32)
            out_u64_ub = T.view(out_ub, shape=(block_m, data_qwords), dtype=T.uint64)

            T.annotate_buffer_versions({out_ub: num_stages, sf_ub: num_stages})

            for token_tile in T.Persistent(
                [T.ceildiv(n, block_m)],
                num_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                token_base = token_tile * block_m
                valid_rows = T.min(block_m, n - token_base)

                T.copy(slot_mapping[token_base : token_base + valid_rows], slot_ub[:valid_rows])
                T.copy(kv[token_base : token_base + valid_rows, 0:128], x_ub[:valid_rows, 0:128])

                for row in T.serial(block_m):
                    if row < valid_rows:
                        slot_idx = T.int32(slot_ub[row])
                        T.assume(-1 <= slot_idx < num_blocks * block_size)
                        if slot_idx != -1:
                            collect_amax(x_ub, row, amax_ub)
                            compute_sf(amax_ub, row, sf_ub, sf_inv_ub)
                            quantize(x_ub, row, sf_inv_ub, out_ub)

                with T.SimtVF(threads=block_m * store_lanes_per_row):
                    store_thread = T.get_thread_binding()
                    store_row = store_thread // store_lanes_per_row
                    store_lane = store_thread - store_row * store_lanes_per_row
                    if store_row < valid_rows:
                        slot = T.int32(slot_ub[store_row])
                        T.assume(-1 <= slot < num_blocks * block_size)
                        if slot != -1:
                            slot_u = T.cast(slot, 'uint32')
                            cache_block = T.alloc_var(T.uint32, init=slot_u // block_size)
                            cache_offset = T.alloc_var(T.uint32, init=slot_u - cache_block * block_size)
                            for store_round in T.unroll(T.ceildiv(data_qwords, store_lanes_per_row), explicit=True):
                                store_word = store_lane + store_round * store_lanes_per_row
                                if store_word < data_qwords:
                                    kv_cache_data_u64[cache_block, cache_offset, store_word] = out_u64_ub[
                                        store_row, store_word
                                    ]
                            if store_lane < num_groups:
                                kv_cache_sf[cache_block, cache_offset, store_lane] = sf_ub[store_row, store_lane]

    return _store_index_kv_cache_ascend


def _store_index_kv_cache_tl_ascend(
    block_size: int,
    stride_kvcache_fp8: int,
    stride_kvcache_sf: int,
) -> object:
    return _store_index_kv_cache_quant_tl_ascend(
        block_size,
        'fp8',
        stride_kvcache_fp8,
        stride_kvcache_sf,
    )


def _store_index_kv_cache_fp4_tl_ascend(
    block_size: int,
    stride_kvcache_fp4: int,
    stride_kvcache_sf: int,
) -> object:
    return _store_index_kv_cache_quant_tl_ascend(
        block_size,
        'fp4',
        stride_kvcache_fp4,
        stride_kvcache_sf,
    )


def _store_index_kv_cache_fp8_per32_tl_ascend(
    block_size: int,
    stride_kvcache_fp8: int,
    stride_kvcache_sf: int,
) -> object:
    return _store_index_kv_cache_quant_tl_ascend(
        block_size,
        'mxfp8',
        stride_kvcache_fp8,
        stride_kvcache_sf,
    )


@tilelang.jit(out_idx=None)
def _load_index_kv_cache_quant_tl_ascend(
    block_size: int,
    data_bytes: int,
    stride_kvcache_data: int,
    stride_kvcache_sf: int,
) -> object:
    """Ascend AIV byte-gather shared by the three index-cache formats."""
    assert data_bytes in (64, 128)
    n = T.dynamic('n')
    num_blocks = T.dynamic('num_blocks')
    stride_kvcache_block = T.dynamic('stride_kvcache_block')
    num_cores = get_num_vector_cores()
    block_m = 128
    num_stages = 2
    group_size = 2

    @T.prim_func
    def _load_index_kv_cache_ascend(
        kv_data: T.Tensor((n, data_bytes), T.uint8),  # type: ignore
        kv_sf: T.Tensor((n,), T.int32),  # type: ignore
        kv_cache_data: T.StridedTensor(
            (num_blocks, block_size, data_bytes),
            (stride_kvcache_block, stride_kvcache_data, 1),
            T.uint8,
        ),  # type: ignore
        kv_cache_sf: T.StridedTensor(
            (num_blocks, block_size, 1),
            (stride_kvcache_block // 4, stride_kvcache_sf, 1),
            T.int32,
        ),  # type: ignore
        slot_mapping: T.Tensor((n,), T.int32),  # type: ignore
    ) -> None:
        with T.Kernel(num_cores) as core_id:
            data_ub = T.alloc_shared((block_m, data_bytes), T.uint8)
            sf_ub = T.alloc_shared((block_m,), T.int32)
            slot_ub = T.alloc_shared((block_m,), T.int32)
            T.annotate_buffer_versions({data_ub: num_stages, sf_ub: num_stages, slot_ub: num_stages})

            for token_tile in T.Persistent(
                [T.ceildiv(n, block_m)],
                num_cores,
                core_id,
                group_size=group_size,
                num_stages=num_stages,
            ):
                token_base = token_tile * block_m
                valid_rows = T.min(block_m, n - token_base)
                T.copy(slot_mapping[token_base : token_base + valid_rows], slot_ub[:valid_rows])
                for row in T.serial(block_m):
                    if row < valid_rows:
                        slot_idx = T.int32(slot_ub[row])
                        T.assume(-1 <= slot_idx < num_blocks * block_size)
                        if slot_idx != -1:
                            cache_block = slot_idx // block_size
                            cache_offset = slot_idx % block_size
                            T.copy(kv_cache_data[cache_block, cache_offset, 0:data_bytes], data_ub[row, 0:data_bytes])
                            sf_ub[row] = kv_cache_sf[cache_block, cache_offset, 0]
                        else:
                            with T.SimdVF():
                                valid_bytes = S.pset(8, f'PAT_VL{data_bytes}')
                                zeros = S.vdup(0, T.uint8, valid_bytes)
                                S.vsts(data_ub[row, 0], zeros, valid_bytes, dist='NORM_B8', extent=data_bytes)
                            sf_ub[row] = T.int32(0)
                T.copy(data_ub[:valid_rows, 0:data_bytes], kv_data[token_base : token_base + valid_rows, 0:data_bytes])
                T.copy(sf_ub[:valid_rows], kv_sf[token_base : token_base + valid_rows])

    return _load_index_kv_cache_ascend


def _load_index_kv_cache_tl_ascend(
    block_size: int,
    stride_kvcache_fp8: int,
    stride_kvcache_sf: int,
) -> object:
    return _load_index_kv_cache_quant_tl_ascend(
        block_size,
        128,
        stride_kvcache_fp8,
        stride_kvcache_sf,
    )


def _load_index_kv_cache_fp4_tl_ascend(
    block_size: int,
    stride_kvcache_fp4: int,
    stride_kvcache_sf: int,
) -> object:
    return _load_index_kv_cache_quant_tl_ascend(
        block_size,
        64,
        stride_kvcache_fp4,
        stride_kvcache_sf,
    )


def _load_index_kv_cache_fp8_per32_tl_ascend(
    block_size: int,
    stride_kvcache_fp8: int,
    stride_kvcache_sf: int,
) -> object:
    return _load_index_kv_cache_quant_tl_ascend(
        block_size,
        128,
        stride_kvcache_fp8,
        stride_kvcache_sf,
    )


@tilelang.jit(out_idx=None)
def _store_kv_cache_llama_tl_ascend(
    block_size: int,
    element_per_token: int,
) -> object:
    """Ascend AIV SIMD paged-cache store for BF16 LLaMA KV cache."""
    n = T.dynamic('n')
    num_blocks = T.dynamic('num_blocks')
    stride_kvcache_block = T.dynamic('stride_kvcache_block')
    stride_kv_cache = T.dynamic('stride_kv_cache')
    stride_kv = T.dynamic('stride_kv')

    num_cores = get_num_vector_cores()
    num_stages = 2
    ub_budget = 262144 - 32768
    row_bytes = element_per_token * 2 + 4
    if num_stages * row_bytes > ub_budget:
        num_stages = 1
    max_blk = max(min(64, ub_budget // (num_stages * row_bytes)), 1)
    blk = max_blk // 8 * 8 if max_blk >= 8 else max_blk
    assert element_per_token % 128 == 0
    store_lanes_per_row = 16 if blk % 2 == 0 else 32
    element_qwords = element_per_token * 2 // 8

    @T.prim_func
    def _store_kv_cache_llama_ascend(
        kv: T.StridedTensor((n, element_per_token), (stride_kv, 1), T.bfloat16),  # type: ignore
        kv_cache: T.StridedTensor(
            (num_blocks, block_size, element_per_token),
            (stride_kvcache_block, stride_kv_cache, 1),
            T.bfloat16,
        ),  # type: ignore
        slot_mapping: T.Tensor((n,), T.int32),  # type: ignore
    ) -> None:
        T.assume(stride_kv % 16 == 0)
        T.assume(stride_kv_cache % 16 == 0)
        T.assume(stride_kvcache_block % 16 == 0)

        kv_cache_u64 = T.view(
            kv_cache,
            shape=(num_blocks, block_size, element_qwords),
            dtype=T.uint64,
            strides=(stride_kvcache_block // 4, stride_kv_cache // 4, 1),
        )
        with T.Kernel(num_cores) as core_id:
            kv_ub = T.alloc_shared((blk, element_per_token), T.bfloat16)
            slot_ub = T.alloc_shared((blk,), T.int32)
            kv_u64_ub = T.view(kv_ub, shape=(blk, element_qwords), dtype=T.uint64)
            T.annotate_buffer_versions({kv_ub: num_stages, slot_ub: num_stages})

            for block_id in T.Persistent([T.ceildiv(n, blk)], num_cores, core_id, group_size=1, num_stages=num_stages):
                base = T.alloc_var(T.int32, init=block_id * blk)
                valid = T.alloc_var(T.int32, init=T.min(blk, n - base))
                T.copy(slot_mapping[base : base + valid], slot_ub[:valid])
                T.copy(kv[base : base + valid, 0:element_per_token], kv_ub[:valid, 0:element_per_token])
                with T.SimtVF(threads=blk * store_lanes_per_row):
                    store_thread = T.get_thread_binding()
                    store_row = store_thread // store_lanes_per_row
                    store_lane = store_thread - store_row * store_lanes_per_row
                    if store_row < valid:
                        slot = T.int32(slot_ub[store_row])
                        T.assume(-1 <= slot < num_blocks * block_size)
                        if slot != -1:
                            slot_u = T.cast(slot, 'uint32')
                            cache_block = T.alloc_var(T.uint32, init=slot_u // block_size)
                            cache_offset = T.alloc_var(T.uint32, init=slot_u - cache_block * block_size)
                            for store_round in T.unroll(T.ceildiv(element_qwords, store_lanes_per_row), explicit=True):
                                store_word = store_lane + store_round * store_lanes_per_row
                                if store_word < element_qwords:
                                    kv_cache_u64[cache_block, cache_offset, store_word] = kv_u64_ub[
                                        store_row, store_word
                                    ]

    return _store_kv_cache_llama_ascend


@tilelang.jit(out_idx=None)
def _load_kv_cache_llama_tl_ascend(
    block_size: int,
    element_per_token: int,
) -> object:
    """Ascend paged-cache gather for BF16 LLaMA KV cache."""
    assert element_per_token % 128 == 0

    n = T.dynamic('n')
    num_blocks = T.dynamic('num_blocks')
    stride_kvcache_block = T.dynamic('stride_kvcache_block')
    stride_kv_cache = T.dynamic('stride_kv_cache')

    num_cores = get_num_vector_cores()
    blk = max(min(64, 192 * 1024 // (element_per_token * 2)), 1)
    num_stages = 2

    out_bytes = blk * element_per_token * T.bfloat16.bytes
    slot_bytes = blk * T.int32.bytes
    ub_limit = 248 * 1024
    out_stages = 2 if num_stages * out_bytes + num_stages * slot_bytes <= ub_limit else 1

    @T.prim_func
    def _load_kv_cache_llama_ascend(
        kv: T.Tensor((n, element_per_token), T.bfloat16),  # type: ignore
        kv_cache: T.StridedTensor(
            (num_blocks, block_size, element_per_token),
            (stride_kvcache_block, stride_kv_cache, 1),
            T.bfloat16,
        ),  # type: ignore
        slot_mapping: T.Tensor((n,), T.int32),  # type: ignore
    ) -> None:
        T.assume(stride_kv_cache % 16 == 0)
        T.assume(stride_kvcache_block % 16 == 0)

        with T.Kernel(num_cores) as core_id:
            out_ub = T.alloc_shared((blk, element_per_token), T.bfloat16)
            slot_ub = T.alloc_shared((blk,), T.int32)

            T.annotate_buffer_versions({out_ub: out_stages, slot_ub: num_stages})

            for block_id in T.Persistent([T.ceildiv(n, blk)], num_cores, core_id, group_size=1, num_stages=num_stages):
                base = T.alloc_var(T.int32, init=block_id * blk)
                valid = T.alloc_var(T.int32, init=T.min(blk, n - base))
                T.copy(slot_mapping[base : base + valid], slot_ub[:valid])
                for i in T.serial(valid):
                    slot = T.alloc_var(T.int32, init=slot_ub[i])
                    T.assume(-1 <= slot < num_blocks * block_size)
                    if slot != -1:
                        cache_block = T.alloc_var(T.int32, init=slot // block_size)
                        cache_offset = T.alloc_var(T.int32, init=slot % block_size)
                        T.copy(
                            kv_cache[cache_block, cache_offset, 0:element_per_token],
                            out_ub[i, 0:element_per_token],
                        )
                    else:
                        T.fill(out_ub[i, 0:element_per_token], 0)
                T.copy(out_ub[:valid, 0:element_per_token], kv[base : base + valid, 0:element_per_token])

    return _load_kv_cache_llama_ascend
