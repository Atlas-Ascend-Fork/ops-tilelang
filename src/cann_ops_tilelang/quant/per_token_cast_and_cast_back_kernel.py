import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.config import get_num_vector_cores
from cann_ops_tilelang.utils import align_up as align

from .common import CastOutputConfig

# BF16 bit-field constants for power-of-two round_sf.
_ABS_MASK_BF16 = 0x7FFF
_BF16_EXP_BIAS = 0x7F00
_EXP_MASK_F32 = 0x7F800000
_F32_EXP_BIAS = 0x7F000000
_PARALLELISM_TARGET_TOKENS = 512


def _parallelism_cap(block_m: int, block_k: int, hidden: int, n_cores: int) -> int:
    """Cap block_m to maximise wave utilisation for small batch sizes.

    When the UB-fitted ``block_m`` would produce fewer tiles than ``n_cores``
    for a 512-token batch, reducing ``block_m`` can increase the tile count
    so that all vector cores are kept busy without leaving a near-empty last
    wave.  The search picks the ``block_m`` (1 .. ``block_m``) whose tile count
    gives the highest wave utilisation, breaking ties toward larger
    ``block_m`` (less overhead for large batches).
    """
    hidden_tiles = hidden // block_k
    best_cap = block_m
    best_score = -1
    for cap in range(1, block_m + 1):
        tiles = (_PARALLELISM_TARGET_TOKENS + cap - 1) // cap * hidden_tiles
        waves = (tiles + n_cores - 1) // n_cores
        util = tiles / (waves * n_cores) if waves > 0 else 0
        score = int(util * 10000) + cap
        if score > best_score:
            best_score = score
            best_cap = cap
    return best_cap


def _get_block_shape(
    hidden: int, dtype: T.dtype, num_stages: int, num_per_channels: int, n_cores: int
) -> tuple[int, int, int]:
    """Choose the widest valid block_k, largest UB-fitting block_m and tile width.

    The tile width is picked from {256, 128, 64} (largest first) so that it
    divides ``block_k`` and covers at least ``num_per_channels`` columns. This
    keeps the 8-slot E2B SF loader unchanged while supporting hidden sizes that
    are not multiples of 256.

    For hidden <= 2048 an additional search over valid (block_k, tile_width,
    block_m) combinations is tried first, preferring the largest ``block_m``
    (then largest ``tile_width``, then largest ``block_k``).  A larger
    ``block_m`` increases compute density per tile for small hidden sizes.
    The kernel copies exactly ``block_k`` columns via explicit slices, so
    splitting hidden into smaller blocks is safe for any ``block_k`` that
    divides ``hidden`` (with at most 4 hidden tiles).  If no valid split
    exists, the greedy strategy below is used.
    """
    stage_budget = 248 * 1024 // num_stages

    if hidden <= 2048:
        best = None
        for block_k in range(hidden, 63, -64):
            if hidden % block_k != 0:
                continue
            if hidden // block_k > 4:
                continue
            for tile_width in (256, 128, 64):
                if block_k % tile_width != 0 or tile_width < num_per_channels:
                    continue
                for block_m in range(32, 0, -1):
                    num_sf = align(block_m * (block_k // tile_width) * 8, 64)
                    ub_usage = block_m * align(block_k, 128) * dtype.bits // 4 + num_sf * 12
                    if ub_usage <= stage_budget:
                        score = tile_width * 1000000 + block_k * 10000 + block_m * 100
                        if best is None or score > best[0]:
                            best = (score, block_m, block_k, tile_width)
                        break
        if best is not None:
            cap = _parallelism_cap(best[1], best[2], hidden, n_cores)
            return cap, best[2], best[3]

    for block_k in range(hidden, 63, -64):
        if hidden % block_k != 0:
            continue
        for tile_width in (256, 128, 64):
            if block_k % tile_width != 0 or tile_width < num_per_channels:
                continue
            for block_m in range(32, 0, -1):
                num_sf = align(block_m * (block_k // tile_width) * 8, 64)
                ub_usage = block_m * align(block_k, 128) * dtype.bits // 4 + num_sf * 12
                if ub_usage <= stage_budget:
                    cap = _parallelism_cap(block_m, block_k, hidden, n_cores)
                    return cap, block_k, tile_width


@tilelang.jit
def get_per_token_cast_and_cast_back_kernel_asc(
    hidden: int,
    token_stride: int,
    dtype: T.dtype,
    out_config: CastOutputConfig,
):
    num_per_channels = out_config.sf_block[1]
    is_bf16_in = dtype == T.bfloat16
    is_fp16_in = dtype == T.float16
    assert dtype in (T.bfloat16, T.float16, T.float32), (
        'Ascend per-token cast and cast-back only supports bf16, fp16 or fp32 input'
    )
    assert num_per_channels in (32, 64, 128), (
        'Ascend per-token cast and cast-back only supports num_per_channels=32/64/128'
    )
    assert hidden % 64 == 0, 'Ascend per-token cast and cast-back hidden must be a multiple of 64'

    is_fp4_out = out_config.dtype == T.float4_e2m1fn
    round_sf = out_config.round_sf
    quant_max = 6.0 if is_fp4_out else 448.0
    round_carry = 0x003FFFFF if is_fp4_out else 0x001FFFFF
    quant_max_exp = 0x01000000 if is_fp4_out else 0x04000000

    n_cores = get_num_vector_cores()
    num_stages = 4 if hidden >= 2560 else 3

    block_m, block_k, tile_width = _get_block_shape(hidden, dtype, num_stages, num_per_channels, n_cores)
    # SF slots are laid out at a fixed 32-column granularity so that the E2B
    # broadcast loaders stay unchanged; for num_per_channels>32 every amax/sf
    # value is duplicated across consecutive slots. Each tile keeps 8 slots so
    # the E2B loaders can read a full vector.
    num_tiles = block_k // tile_width
    tile_slots = 8

    num_sf = align(block_m * num_tiles * tile_slots, 64)
    use_bf16_sf = round_sf and is_bf16_in and tile_width == 256
    sf_inv_dtype = T.bfloat16 if use_bf16_sf else T.float32

    num_tokens = T.dynamic('num_tokens')

    @T.macro
    def get_sf_from_inv_f32(sf_inv, sf_inv_bias_u32):
        sf_bits_u32 = S.vsub(sf_inv_bias_u32, T.reinterpret(sf_inv, 'uint32x64'))
        return T.reinterpret(sf_bits_u32, 'float32x64')

    @T.macro
    def load_sf(sf_ub, sf_offset):
        """E2B-load eight compact SF values and broadcast each over 32 lanes."""
        expanded = S.vld(sf_ub[sf_offset], dist='E2B_B32')
        low, high = S.vintlv(expanded, expanded)
        sf0, sf1 = S.vintlv(low, low)
        sf2, sf3 = S.vintlv(high, high)
        return sf0, sf1, sf2, sf3

    @T.macro
    def load_sf_dintlv(sf_ub, sf_offset):
        """Broadcast eight compact SF values over DINTLV 16-element halves."""
        expanded = S.vld(sf_ub[sf_offset], dist='E2B_B32')
        return S.vintlv(expanded, expanded)

    @T.macro
    def cast_and_cast_back_f32_pair(scaled0, scaled1, sf0, sf1):
        """Round-trip two 64-lane f32 chunks through e4m3/e2m1 and dequant."""
        if is_fp4_out:
            low, high = S.vdintlv(
                T.reinterpret(scaled0, 'uint16x128'),
                T.reinterpret(scaled1, 'uint16x128'),
            )
            scaled_bf16 = T.reinterpret(S.vor(high, S.vmins(low, 1)), 'bfloat16x128')
            recovered = S.vcvt(S.vcvt(scaled_bf16, T.float4_e2m1fn), T.bfloat16, part=0)
            cast_back0, cast_back1 = S.vintlv(
                S.vcvt(recovered, T.float32, part=0),
                S.vcvt(recovered, T.float32, part=1),
            )
        else:
            cast_back0 = S.vcvt(S.vcvt(scaled0, T.float8_e4m3fn), T.float32, part=0)
            cast_back1 = S.vcvt(S.vcvt(scaled1, T.float8_e4m3fn), T.float32, part=0)
        return S.vmul(cast_back0, sf0), S.vmul(cast_back1, sf1)

    @T.macro
    def cast_and_cast_back(x_ub, out_ub, row, col, sf0, sf1, sf_inv0, sf_inv1, zero_bf16):
        if is_bf16_in:
            values0_bits, values1_bits = S.vintlv(zero_bf16, S.vld(x_ub[row, col]))
            scaled0 = S.vmul(T.reinterpret(values0_bits, 'float32x64'), sf_inv0)
            scaled1 = S.vmul(T.reinterpret(values1_bits, 'float32x64'), sf_inv1)
        elif is_fp16_in:
            values_f16 = S.vld(x_ub[row, col])
            rep0, rep1 = S.vintlv(values_f16, values_f16)
            scaled0 = S.vmul(S.vcvt(rep0, T.float32, part=0), sf_inv0)
            scaled1 = S.vmul(S.vcvt(rep1, T.float32, part=0), sf_inv1)
        else:
            values0 = S.vld(x_ub[row, col])
            values1 = S.vld(x_ub[row, col + 64])
            scaled0 = S.vmul(values0, sf_inv0)
            scaled1 = S.vmul(values1, sf_inv1)
        output0, output1 = cast_and_cast_back_f32_pair(scaled0, scaled1, sf0, sf1)
        if is_bf16_in:
            S.vsts(out_ub[row, col], S.vcvt(output0, T.bfloat16), dist='PK_B32')
            S.vsts(out_ub[row, col + 64], S.vcvt(output1, T.bfloat16), dist='PK_B32')
        elif is_fp16_in:
            S.vsts(out_ub[row, col], S.vcvt(output0, T.float16), dist='PK_B32')
            S.vsts(out_ub[row, col + 64], S.vcvt(output1, T.float16), dist='PK_B32')
        else:
            S.vsts(out_ub[row, col], output0)
            S.vsts(out_ub[row, col + 64], output1)

    @T.macro
    def cast_and_cast_back_64(x_ub, out_ub, row, col, sf0, sf_inv0, zero_bf16):
        if is_bf16_in:
            values0_bits, _values1_bits = S.vintlv(zero_bf16, S.vld(x_ub[row, col]))
            scaled0 = S.vmul(T.reinterpret(values0_bits, 'float32x64'), sf_inv0)
        elif is_fp16_in:
            values_f16 = S.vld(x_ub[row, col])
            rep0, _rep1 = S.vintlv(values_f16, values_f16)
            scaled0 = S.vmul(S.vcvt(rep0, T.float32, part=0), sf_inv0)
        else:
            scaled0 = S.vmul(S.vld(x_ub[row, col]), sf_inv0)
        output0, _output1 = cast_and_cast_back_f32_pair(scaled0, scaled0, sf0, sf0)
        if is_bf16_in:
            S.vsts(out_ub[row, col], S.vcvt(output0, T.bfloat16), dist='PK_B32')
        elif is_fp16_in:
            S.vsts(out_ub[row, col], S.vcvt(output0, T.float16), dist='PK_B32')
        else:
            S.vsts(out_ub[row, col], output0)

    @T.macro
    def cast_and_cast_back_fp16(x_ub, out_ub, row, col, sf0, sf1, sf2, sf3, sf_inv0, sf_inv1, sf_inv2, sf_inv3):
        values_f16_lo = S.vld(x_ub[row, col])
        values_f16_hi = S.vld(x_ub[row, col + 128])
        rep0_lo, rep0_hi = S.vintlv(values_f16_lo, values_f16_lo)
        rep1_lo, rep1_hi = S.vintlv(values_f16_hi, values_f16_hi)
        values00 = S.vcvt(rep0_lo, T.float32, part=0)
        values01 = S.vcvt(rep0_hi, T.float32, part=0)
        values10 = S.vcvt(rep1_lo, T.float32, part=0)
        values11 = S.vcvt(rep1_hi, T.float32, part=0)
        output0, output1 = cast_and_cast_back_f32_pair(S.vmul(values00, sf_inv0), S.vmul(values01, sf_inv1), sf0, sf1)
        output2, output3 = cast_and_cast_back_f32_pair(S.vmul(values10, sf_inv2), S.vmul(values11, sf_inv3), sf2, sf3)
        S.vsts(out_ub[row, col], S.vcvt(output0, T.float16), dist='PK_B32')
        S.vsts(out_ub[row, col + 64], S.vcvt(output1, T.float16), dist='PK_B32')
        S.vsts(out_ub[row, col + 128], S.vcvt(output2, T.float16), dist='PK_B32')
        S.vsts(out_ub[row, col + 192], S.vcvt(output3, T.float16), dist='PK_B32')

    @T.macro
    def cast_and_cast_back_bf16_no_round(
        x_ub, out_ub, row, col, sf_lower, sf_upper, sf_inv_lower, sf_inv_upper, zero_bf16
    ):
        values0, values1 = S.vld2(x_ub[row, col], dist='DINTLV_B16')
        values0_lower_bits, values0_upper_bits = S.vintlv(zero_bf16, values0)
        values1_lower_bits, values1_upper_bits = S.vintlv(zero_bf16, values1)
        values0_lower = T.reinterpret(values0_lower_bits, 'float32x64')
        values0_upper = T.reinterpret(values0_upper_bits, 'float32x64')
        values1_lower = T.reinterpret(values1_lower_bits, 'float32x64')
        values1_upper = T.reinterpret(values1_upper_bits, 'float32x64')

        output0_lower, output1_lower = cast_and_cast_back_f32_pair(
            S.vmul(values0_lower, sf_inv_lower), S.vmul(values1_lower, sf_inv_lower), sf_lower, sf_lower
        )
        output_lower0, output_lower1 = S.vintlv(output0_lower, output1_lower)
        S.vsts(out_ub[row, col], S.vcvt(output_lower0, T.bfloat16), dist='PK_B32')
        S.vsts(out_ub[row, col + 64], S.vcvt(output_lower1, T.bfloat16), dist='PK_B32')

        output0_upper, output1_upper = cast_and_cast_back_f32_pair(
            S.vmul(values0_upper, sf_inv_upper), S.vmul(values1_upper, sf_inv_upper), sf_upper, sf_upper
        )
        output_upper0, output_upper1 = S.vintlv(output0_upper, output1_upper)
        S.vsts(out_ub[row, col + 128], S.vcvt(output_upper0, T.bfloat16), dist='PK_B32')
        S.vsts(out_ub[row, col + 192], S.vcvt(output_upper1, T.bfloat16), dist='PK_B32')

    @T.macro
    def cast_and_cast_back_bf16(scaled, sf):
        scaled0 = S.vcvt(scaled, T.float32, part=0)
        scaled1 = S.vcvt(scaled, T.float32, part=1)
        quantized0 = S.vcvt(scaled0, T.float8_e4m3fn)
        quantized1 = S.vcvt(scaled1, T.float8_e4m3fn)
        cast_back0 = S.vcvt(quantized0, T.float32, part=0)
        cast_back1 = S.vcvt(quantized1, T.float32, part=0)
        rescaled0 = S.vmul(cast_back0, sf)
        rescaled1 = S.vmul(cast_back1, sf)
        packed0, packed1 = S.vintlv(rescaled0, rescaled1)
        _rounding_bits, result = S.vdintlv(
            T.reinterpret(packed0, 'uint16x128'),
            T.reinterpret(packed1, 'uint16x128'),
        )
        return T.reinterpret(result, 'bfloat16x128')

    @T.macro
    def compute_sf(amax_ub, sf_ub, sf_inv_ub):
        with T.SimdVF():
            if round_sf:
                exp_mask_u32 = S.vdup(_EXP_MASK_F32, T.uint32)
                sf_inv_bias_u32 = S.vdup(_F32_EXP_BIAS + quant_max_exp, T.uint32)
            else:
                quant_max_f32 = S.vdup(quant_max, T.float32)

            for sf_batch in T.serial(num_sf // 64):
                sf_offset = sf_batch * 64
                amax = S.vld(amax_ub[sf_offset])
                clamped_amax = S.vmaxs(amax, out_config.clamp_min_value)
                if round_sf:
                    rounded_amax_exp_u32 = S.alloc_var(T.uint32)
                    rounded_amax_exp_u32 = S.vadds(T.reinterpret(clamped_amax, 'uint32x64'), round_carry)
                    rounded_amax_exp_u32 = S.vand(rounded_amax_exp_u32, exp_mask_u32)
                    sf_inv_bits_u32 = S.vsub(sf_inv_bias_u32, rounded_amax_exp_u32)
                    sf_inv = T.reinterpret(sf_inv_bits_u32, 'float32x64')
                    if use_bf16_sf:
                        S.vsts(sf_inv_ub[sf_offset], S.vcvt(sf_inv, T.bfloat16), dist='PK_B32')
                    else:
                        S.vsts(sf_inv_ub[sf_offset], sf_inv)
                else:
                    sf = S.vdiv(clamped_amax, quant_max_f32)
                    S.vsts(sf_ub[sf_offset], sf)
                    sf_inv = S.vdiv(quant_max_f32, clamped_amax)
                    S.vsts(sf_inv_ub[sf_offset], sf_inv)

    @T.macro
    def cast_and_cast_back_tile(x_ub, sf_ub, sf_inv_ub, out_ub):
        with T.SimdVF():
            bias_u16 = S.vdup(_BF16_EXP_BIAS, T.uint16)
            sf_inv_bias_u32 = S.vdup(_F32_EXP_BIAS, T.uint32)
            zero_bf16 = S.vdup(0.0, T.bfloat16)

            for row in T.serial(block_m):
                for tile in T.serial(num_tiles):
                    col = tile * tile_width
                    sf_offset = row * (num_tiles * 8) + tile * 8
                    if tile_width < 256:
                        if round_sf:
                            sf_invs = load_sf(sf_inv_ub, sf_offset)
                            sfs = (
                                get_sf_from_inv_f32(sf_invs[0], sf_inv_bias_u32),
                                get_sf_from_inv_f32(sf_invs[1], sf_inv_bias_u32),
                            )
                        else:
                            sfs = load_sf(sf_ub, sf_offset)
                            sf_invs = load_sf(sf_inv_ub, sf_offset)
                        if tile_width == 64:
                            cast_and_cast_back_64(x_ub, out_ub, row, col, sfs[0], sf_invs[0], zero_bf16)
                        else:
                            cast_and_cast_back(
                                x_ub, out_ub, row, col, sfs[0], sfs[1], sf_invs[0], sf_invs[1], zero_bf16
                            )
                    elif use_bf16_sf:
                        values0, values1 = S.vld2(x_ub[row, col], dist='DINTLV_B16')
                        sf_inv = S.vld(sf_inv_ub[sf_offset], dist='E2B_B16')
                        scaled0 = S.vmul(values0, sf_inv)
                        scaled1 = S.vmul(values1, sf_inv)
                        sf_inv_u16 = T.reinterpret(sf_inv, 'uint16x128')
                        sf_u16 = S.vsub(bias_u16, sf_inv_u16)
                        sf = T.reinterpret(sf_u16, 'bfloat16x128')

                        if is_fp4_out:
                            quantized0 = S.vcvt(scaled0, T.float4_e2m1fn)
                            quantized1 = S.vcvt(scaled1, T.float4_e2m1fn)
                            cast_back0 = S.vcvt(quantized0, T.bfloat16, part=0)
                            cast_back1 = S.vcvt(quantized1, T.bfloat16, part=0)
                            output0 = S.vmul(cast_back0, sf)
                            output1 = S.vmul(cast_back1, sf)
                        else:
                            sf_f32 = S.vcvt(sf, T.float32, part=0)
                            output0 = cast_and_cast_back_bf16(scaled0, sf_f32)
                            output1 = cast_and_cast_back_bf16(scaled1, sf_f32)
                        output_lower_u16, output_upper_u16 = S.vintlv(
                            T.reinterpret(output0, 'uint16x128'),
                            T.reinterpret(output1, 'uint16x128'),
                        )
                        S.vsts(out_ub[row, col], T.reinterpret(output_lower_u16, 'bfloat16x128'), dist='NORM_B16')
                        S.vsts(out_ub[row, col + 128], T.reinterpret(output_upper_u16, 'bfloat16x128'), dist='NORM_B16')
                    else:
                        if is_bf16_in and not round_sf:
                            sf_lower, sf_upper = load_sf_dintlv(sf_ub, sf_offset)
                            sf_inv_lower, sf_inv_upper = load_sf_dintlv(sf_inv_ub, sf_offset)
                            cast_and_cast_back_bf16_no_round(
                                x_ub, out_ub, row, col, sf_lower, sf_upper, sf_inv_lower, sf_inv_upper, zero_bf16
                            )
                        elif is_bf16_in:
                            if round_sf:
                                sf_invs = load_sf(sf_inv_ub, sf_offset)
                                sfs = (
                                    get_sf_from_inv_f32(sf_invs[0], sf_inv_bias_u32),
                                    get_sf_from_inv_f32(sf_invs[1], sf_inv_bias_u32),
                                    get_sf_from_inv_f32(sf_invs[2], sf_inv_bias_u32),
                                    get_sf_from_inv_f32(sf_invs[3], sf_inv_bias_u32),
                                )
                            else:
                                sfs = load_sf(sf_ub, sf_offset)
                                sf_invs = load_sf(sf_inv_ub, sf_offset)
                            cast_and_cast_back(
                                x_ub, out_ub, row, col, sfs[0], sfs[1], sf_invs[0], sf_invs[1], zero_bf16
                            )
                            cast_and_cast_back(
                                x_ub, out_ub, row, col + 128, sfs[2], sfs[3], sf_invs[2], sf_invs[3], zero_bf16
                            )
                        elif is_fp16_in:
                            if round_sf:
                                sf_invs = load_sf(sf_inv_ub, sf_offset)
                                sfs = (
                                    get_sf_from_inv_f32(sf_invs[0], sf_inv_bias_u32),
                                    get_sf_from_inv_f32(sf_invs[1], sf_inv_bias_u32),
                                    get_sf_from_inv_f32(sf_invs[2], sf_inv_bias_u32),
                                    get_sf_from_inv_f32(sf_invs[3], sf_inv_bias_u32),
                                )
                            else:
                                sfs = load_sf(sf_ub, sf_offset)
                                sf_invs = load_sf(sf_inv_ub, sf_offset)
                            cast_and_cast_back_fp16(
                                x_ub,
                                out_ub,
                                row,
                                col,
                                sfs[0],
                                sfs[1],
                                sfs[2],
                                sfs[3],
                                sf_invs[0],
                                sf_invs[1],
                                sf_invs[2],
                                sf_invs[3],
                            )
                        elif round_sf:
                            sf_invs = load_sf(sf_inv_ub, sf_offset)
                            sfs = (
                                get_sf_from_inv_f32(sf_invs[0], sf_inv_bias_u32),
                                get_sf_from_inv_f32(sf_invs[1], sf_inv_bias_u32),
                                get_sf_from_inv_f32(sf_invs[2], sf_inv_bias_u32),
                                get_sf_from_inv_f32(sf_invs[3], sf_inv_bias_u32),
                            )
                            cast_and_cast_back(
                                x_ub, out_ub, row, col, sfs[0], sfs[1], sf_invs[0], sf_invs[1], zero_bf16
                            )
                            cast_and_cast_back(
                                x_ub, out_ub, row, col + 128, sfs[2], sfs[3], sf_invs[2], sf_invs[3], zero_bf16
                            )
                        else:
                            sfs = load_sf(sf_ub, sf_offset)
                            sf_invs = load_sf(sf_inv_ub, sf_offset)
                            cast_and_cast_back(
                                x_ub, out_ub, row, col, sfs[0], sfs[1], sf_invs[0], sf_invs[1], zero_bf16
                            )
                            cast_and_cast_back(
                                x_ub, out_ub, row, col + 128, sfs[2], sfs[3], sf_invs[2], sf_invs[3], zero_bf16
                            )

    @T.macro
    def collect_amax_from_f32(amax_ub, values, amax_base, mask_low, mask_high):
        abs_values = S.vabs(values)
        amax_lower = S.vcmax(abs_values, mask_low)
        amax_upper = S.vcmax(abs_values, mask_high)
        S.vsts(amax_ub[amax_base], amax_lower, dist='ONEPT_B32')
        S.vsts(amax_ub[amax_base + 1], amax_upper, dist='ONEPT_B32')

    @T.macro
    def collect_amax_from_f32_all(amax_ub, values, amax_base, mask_all, repeat):
        abs_values = S.vabs(values)
        amax = S.vcmax(abs_values, mask_all)
        for r in range(repeat):
            S.vsts(amax_ub[amax_base + r], amax, dist='ONEPT_B32')

    @T.macro
    def load_tile_as_f32(x_ub, row, col):
        if tile_width == 256:
            if is_bf16_in:
                zero_bf16 = S.vdup(0.0, T.bfloat16)
                v0_bits, v1_bits = S.vintlv(zero_bf16, S.vld(x_ub[row, col]))
                v2_bits, v3_bits = S.vintlv(zero_bf16, S.vld(x_ub[row, col + 128]))
                return (
                    T.reinterpret(v0_bits, 'float32x64'),
                    T.reinterpret(v1_bits, 'float32x64'),
                    T.reinterpret(v2_bits, 'float32x64'),
                    T.reinterpret(v3_bits, 'float32x64'),
                )
            if is_fp16_in:
                lo = S.vld(x_ub[row, col])
                hi = S.vld(x_ub[row, col + 128])
                rep0_lo, rep0_hi = S.vintlv(lo, lo)
                rep1_lo, rep1_hi = S.vintlv(hi, hi)
                return (
                    S.vcvt(rep0_lo, T.float32, part=0),
                    S.vcvt(rep0_hi, T.float32, part=0),
                    S.vcvt(rep1_lo, T.float32, part=0),
                    S.vcvt(rep1_hi, T.float32, part=0),
                )
            return (
                S.vld(x_ub[row, col]),
                S.vld(x_ub[row, col + 64]),
                S.vld(x_ub[row, col + 128]),
                S.vld(x_ub[row, col + 192]),
            )
        if tile_width == 128:
            if is_bf16_in:
                zero_bf16 = S.vdup(0.0, T.bfloat16)
                v0_bits, v1_bits = S.vintlv(zero_bf16, S.vld(x_ub[row, col]))
                v0 = T.reinterpret(v0_bits, 'float32x64')
                v1 = T.reinterpret(v1_bits, 'float32x64')
            elif is_fp16_in:
                lo = S.vld(x_ub[row, col])
                rep0, rep1 = S.vintlv(lo, lo)
                v0 = S.vcvt(rep0, T.float32, part=0)
                v1 = S.vcvt(rep1, T.float32, part=0)
            else:
                v0 = S.vld(x_ub[row, col])
                v1 = S.vld(x_ub[row, col + 64])
            return v0, v1, v1, v1
        if is_bf16_in:
            zero_bf16 = S.vdup(0.0, T.bfloat16)
            v0_bits, _v1_bits = S.vintlv(zero_bf16, S.vld(x_ub[row, col]))
            v0 = T.reinterpret(v0_bits, 'float32x64')
        elif is_fp16_in:
            lo = S.vld(x_ub[row, col])
            rep0, _rep1 = S.vintlv(lo, lo)
            v0 = S.vcvt(rep0, T.float32, part=0)
        else:
            v0 = S.vld(x_ub[row, col])
        return v0, v0, v0, v0

    @T.macro
    def collect_amax(x_ub, amax_ub):
        with T.SimdVF():
            if is_bf16_in:
                zero_bf16 = S.vdup(0.0, T.bfloat16)
            if tile_width == 256 and is_bf16_in and num_per_channels == 32:
                amax_store_mask = S.pset(32, 'PAT_VL8')
                abs_mask_u16 = S.vdup(_ABS_MASK_BF16, T.uint16)
            else:
                mask_group_low = S.pset(32, 'PAT_VL32')
                mask_group_high = S.vcmps(S.vci(0, T.int32), 31, op='gt')
                mask_all = S.pset(32, 'PAT_ALL')
                zero_f32 = S.vdup(0.0, T.float32)

            for row in T.serial(block_m):
                for tile in T.serial(num_tiles):
                    col = tile * tile_width
                    amax_offset = row * (num_tiles * 8) + tile * 8
                    if tile_width == 256 and is_bf16_in and num_per_channels == 32:
                        values0, values1 = S.vld2(x_ub[row, col], dist='DINTLV_B16')
                        abs0_u16 = S.vand(T.reinterpret(values0, 'uint16x128'), abs_mask_u16)
                        abs1_u16 = S.vand(T.reinterpret(values1, 'uint16x128'), abs_mask_u16)
                        abs_max_u16 = S.vmax(abs0_u16, abs1_u16)
                        amax_u16 = S.vcgmax(abs_max_u16)
                        amax_low, _amax_high = S.vintlv(zero_bf16, T.reinterpret(amax_u16, 'bfloat16x128'))
                        S.vsts(
                            amax_ub[amax_offset],
                            T.reinterpret(amax_low, 'float32x64'),
                            amax_store_mask,
                            dist='NORM_B32',
                        )
                    elif tile_width == 256:
                        v0, v1, v2, v3 = load_tile_as_f32(x_ub, row, col)
                        if num_per_channels == 32:
                            collect_amax_from_f32(amax_ub, v0, amax_offset, mask_group_low, mask_group_high)
                            collect_amax_from_f32(amax_ub, v1, amax_offset + 2, mask_group_low, mask_group_high)
                            collect_amax_from_f32(amax_ub, v2, amax_offset + 4, mask_group_low, mask_group_high)
                            collect_amax_from_f32(amax_ub, v3, amax_offset + 6, mask_group_low, mask_group_high)
                        elif num_per_channels == 64:
                            collect_amax_from_f32_all(amax_ub, v0, amax_offset, mask_all, 2)
                            collect_amax_from_f32_all(amax_ub, v1, amax_offset + 2, mask_all, 2)
                            collect_amax_from_f32_all(amax_ub, v2, amax_offset + 4, mask_all, 2)
                            collect_amax_from_f32_all(amax_ub, v3, amax_offset + 6, mask_all, 2)
                        else:
                            collect_amax_from_f32_all(amax_ub, S.vmax(S.vabs(v0), S.vabs(v1)), amax_offset, mask_all, 4)
                            collect_amax_from_f32_all(
                                amax_ub, S.vmax(S.vabs(v2), S.vabs(v3)), amax_offset + 4, mask_all, 4
                            )
                    elif tile_width == 128:
                        v0, v1, _v2, _v3 = load_tile_as_f32(x_ub, row, col)
                        if num_per_channels == 32:
                            collect_amax_from_f32(amax_ub, v0, amax_offset, mask_group_low, mask_group_high)
                            collect_amax_from_f32(amax_ub, v1, amax_offset + 2, mask_group_low, mask_group_high)
                        elif num_per_channels == 64:
                            collect_amax_from_f32_all(amax_ub, v0, amax_offset, mask_all, 2)
                            collect_amax_from_f32_all(amax_ub, v1, amax_offset + 2, mask_all, 2)
                        else:
                            collect_amax_from_f32_all(amax_ub, S.vmax(S.vabs(v0), S.vabs(v1)), amax_offset, mask_all, 4)
                        for p in range(4):
                            S.vsts(amax_ub[amax_offset + 4 + p], zero_f32, dist='ONEPT_B32')
                    else:
                        v0, _v1, _v2, _v3 = load_tile_as_f32(x_ub, row, col)
                        if num_per_channels == 32:
                            collect_amax_from_f32(amax_ub, v0, amax_offset, mask_group_low, mask_group_high)
                        else:
                            collect_amax_from_f32_all(amax_ub, v0, amax_offset, mask_all, 2)
                        for p in range(6):
                            S.vsts(amax_ub[amax_offset + 2 + p], zero_f32, dist='ONEPT_B32')

    @T.prim_func
    def per_token_cast_and_cast_back_kernel(x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), dtype]):
        with T.Kernel(n_cores) as core_id:
            x_ub = T.alloc_shared((block_m, align(block_k, 128)), dtype)
            amax_ub = T.alloc_shared((num_sf,), T.float32)
            sf_ub = T.alloc_shared((num_sf,), T.float32)
            sf_inv_ub = T.alloc_shared((num_sf,), sf_inv_dtype)
            out_ub = T.alloc_shared((block_m, align(block_k, 128)), dtype)
            T.annotate_buffer_versions(
                {x_ub: num_stages, amax_ub: num_stages, sf_ub: num_stages, sf_inv_ub: num_stages, out_ub: num_stages}
            )

            for pid_token, pid_hidden in T.Persistent(
                [T.ceildiv(num_tokens, block_m), hidden // block_k],
                n_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
                annotations={'enable_offset': True},
            ):
                if pid_token < T.ceildiv(num_tokens, block_m) and pid_hidden < hidden // block_k:
                    token_base = pid_token * block_m
                    hidden_base = pid_hidden * block_k
                    T.copy(
                        x[token_base : token_base + block_m, hidden_base : hidden_base + block_k],
                        x_ub[:block_m, :block_k],
                        l2_cache_ctrl='NOTALLOC_KEEP',
                    )
                    collect_amax(x_ub, amax_ub)
                    compute_sf(amax_ub, sf_ub, sf_inv_ub)
                    cast_and_cast_back_tile(x_ub, sf_ub, sf_inv_ub, out_ub)
                    T.copy(
                        out_ub[:block_m, :block_k],
                        x[token_base : token_base + block_m, hidden_base : hidden_base + block_k],
                        l2_cache_ctrl='NOTALLOC_KEEP',
                    )

    return per_token_cast_and_cast_back_kernel
