# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

import tilelang.ascend.language as T


def _half_block_maximum(dtype):
    @T.macro
    def block_maximum(x_ub, valid, full, neg):
        block_max = T.simd.alloc_var(T.float32)
        half_full = T.simd.pset(16)
        half_neg = T.simd.vdup(-T.infinity(dtype), dtype, half_full)
        h0 = T.simd.alloc_var(dtype)
        h1 = T.simd.alloc_var(dtype)
        h2 = T.simd.alloc_var(dtype)
        h3 = T.simd.alloc_var(dtype)
        h0 = half_neg
        h1 = half_neg
        h2 = half_neg
        h3 = half_neg
        if valid >= 512:
            for vec in T.serial(valid // 512):
                h0 = T.simd.vmax(h0, T.simd.vld(x_ub[vec * 512]), half_full)
                h1 = T.simd.vmax(h1, T.simd.vld(x_ub[vec * 512 + 128]), half_full)
                h2 = T.simd.vmax(h2, T.simd.vld(x_ub[vec * 512 + 256]), half_full)
                h3 = T.simd.vmax(h3, T.simd.vld(x_ub[vec * 512 + 384]), half_full)
        h0 = T.simd.vmax(h0, h1, half_full)
        h0 = T.simd.vmax(h0, h2, half_full)
        h0 = T.simd.vmax(h0, h3, half_full)
        if valid // 128 % 4 > 0:
            for vec in T.serial(valid // 128 % 4):
                h0 = T.simd.vmax(h0, T.simd.vld(x_ub[(valid // 512 * 4 + vec) * 128]), half_full)
        if valid % 128 != 0:
            half_tail = T.simd.update_mask(valid % 128, 16)
            half_value = T.simd.vld(x_ub[valid // 128 * 128])
            h0 = T.simd.vmax(h0, T.simd.vsel(half_value, half_neg, half_tail), half_full)
        max_even = T.simd.vcvt(h0, T.float32, part=0)
        max_odd = T.simd.vcvt(h0, T.float32, part=1)
        block_max = T.simd.vdupv(T.simd.vcmax(T.simd.vmax(max_even, max_odd, full), full), full)
        return block_max

    return block_maximum


def _float_block_maximum():
    @T.macro
    def block_maximum(x_ub, valid, full, neg):
        block_max = T.simd.alloc_var(T.float32)
        lm = T.simd.alloc_var(T.float32)
        lm = neg
        lm1 = T.simd.alloc_var(T.float32)
        lm1 = neg
        lm2 = T.simd.alloc_var(T.float32)
        lm2 = neg
        lm3 = T.simd.alloc_var(T.float32)
        lm3 = neg
        if valid >= 256:
            for vec in T.serial(valid // 256):
                group_value0 = T.simd.vld(x_ub[(vec * 4 + 0) * 64])
                lm = T.simd.vmax(lm, group_value0, full)
                group_value1 = T.simd.vld(x_ub[(vec * 4 + 1) * 64])
                lm1 = T.simd.vmax(lm1, group_value1, full)
                group_value2 = T.simd.vld(x_ub[(vec * 4 + 2) * 64])
                lm2 = T.simd.vmax(lm2, group_value2, full)
                group_value3 = T.simd.vld(x_ub[(vec * 4 + 3) * 64])
                lm3 = T.simd.vmax(lm3, group_value3, full)
        lm = T.simd.vmax(lm, lm1, full)
        lm = T.simd.vmax(lm, lm2, full)
        lm = T.simd.vmax(lm, lm3, full)
        if valid // 64 % 4 > 0:
            for vec in T.serial(valid // 64 % 4):
                value = T.simd.vld(x_ub[(valid // 256 * 4 + vec) * 64])
                lm = T.simd.vmax(lm, value, full)
        if valid % 64 != 0:
            tail_value = T.simd.vld(x_ub[valid // 64 * 64])
            tail = T.simd.update_mask(valid % 64, 32)
            lm = T.simd.vmax(lm, T.simd.vsel(tail_value, neg, tail), full)
        block_max = T.simd.vdupv(T.simd.vcmax(lm, full), full)
        return block_max

    return block_maximum


def _block_maximum(dtype):
    if dtype == 'float32':
        return _float_block_maximum()
    return _half_block_maximum(dtype)


def _exponential_lanes(narrow, fused_exp, broadcast):
    if narrow:

        @T.macro
        def load_lane(x_ub, offset):
            return T.simd.vcvt(T.simd.vld(x_ub[offset], dist='UNPK_B16'), T.float32, part=0)
    else:

        @T.macro
        def load_lane(x_ub, offset):
            return T.simd.vld(x_ub[offset])

    if fused_exp:

        @T.macro
        def selected_exp(value, m, full):
            return T.simd.vexpdif(value, m, full)
    else:

        @T.macro
        def selected_exp(value, m, full):
            return T.simd.vexp(T.simd.vsub(value, m, full), full)

    @T.macro
    def exponential_lanes(x_ub, valid, m, full, neg):
        ls = T.simd.alloc_var(T.float32)
        ls = T.simd.vdup(0.0, T.float32, full)
        ls1 = T.simd.alloc_var(T.float32)
        ls1 = T.simd.vdup(0.0, T.float32, full)
        ls2 = T.simd.alloc_var(T.float32)
        ls2 = T.simd.vdup(0.0, T.float32, full)
        ls3 = T.simd.alloc_var(T.float32)
        ls3 = T.simd.vdup(0.0, T.float32, full)
        if valid >= 256:
            for vec in T.serial(valid // 256):
                if narrow:
                    packed_group0 = T.simd.vld(x_ub[(vec * 4 + 0) * 64])
                    group_value0 = T.simd.vcvt(packed_group0, T.float32, part=0)
                else:
                    group_value0 = T.simd.vld(x_ub[(vec * 4 + 0) * 64])
                ls = T.simd.vadd(ls, selected_exp(group_value0, m, full), full)
                if narrow:
                    group_value1 = T.simd.vcvt(packed_group0, T.float32, part=1)
                else:
                    group_value1 = T.simd.vld(x_ub[(vec * 4 + 1) * 64])
                ls1 = T.simd.vadd(ls1, selected_exp(group_value1, m, full), full)
                if narrow:
                    packed_group1 = T.simd.vld(x_ub[(vec * 4 + 2) * 64])
                    group_value2 = T.simd.vcvt(packed_group1, T.float32, part=0)
                else:
                    group_value2 = T.simd.vld(x_ub[(vec * 4 + 2) * 64])
                ls2 = T.simd.vadd(ls2, selected_exp(group_value2, m, full), full)
                if narrow:
                    group_value3 = T.simd.vcvt(packed_group1, T.float32, part=1)
                else:
                    group_value3 = T.simd.vld(x_ub[(vec * 4 + 3) * 64])
                ls3 = T.simd.vadd(ls3, selected_exp(group_value3, m, full), full)
        ls = T.simd.vadd(ls, ls1, full)
        ls = T.simd.vadd(ls, ls2, full)
        ls = T.simd.vadd(ls, ls3, full)
        if valid // 64 % 4 > 0:
            for vec in T.serial(valid // 64 % 4):
                value2 = load_lane(x_ub, (valid // 256 * 4 + vec) * 64)
                ls = T.simd.vadd(ls, selected_exp(value2, m, full), full)
        if valid % 64 != 0:
            tail_value2 = load_lane(x_ub, valid // 64 * 64)
            tail2 = T.simd.update_mask(valid % 64, 32)
            tail_value2 = T.simd.vsel(tail_value2, neg, tail2)
            ls = T.simd.vadd(ls, selected_exp(tail_value2, m, full), full)
        block_sum = T.simd.vdupv(T.simd.vcadd(ls, full), full) if broadcast else ls
        return block_sum

    return exponential_lanes


def _block_exponential_lanes(dtype):
    return _exponential_lanes(narrow=dtype != 'float32', fused_exp=dtype != 'float32', broadcast=False)


def _block_exponential_sum(dtype):
    return _exponential_lanes(narrow=dtype != 'float32', fused_exp=dtype != 'float32', broadcast=True)


def _plain_exponential_lanes(narrow):
    return _exponential_lanes(narrow, fused_exp=False, broadcast=False)


def _row_block_statistics_update(dtype, single_block):
    narrow = dtype != 'float32'
    block_maximum = _block_maximum(dtype)
    block_exponential_sum = _block_exponential_sum(dtype)

    @T.macro
    def update_row_statistics(x_ub, valid, max_buffer, sum_buffer, wave, sum_offset):
        full = T.simd.pset(32)
        neg = T.simd.vdup(-T.infinity('float32'), T.float32, full)
        block_max = block_maximum(x_ub, valid, full, neg)
        if single_block:
            m = block_max
        else:
            run_max = T.simd.vld(max_buffer[wave], dist='BRC_B32')
            run_sum = T.simd.vld(sum_buffer[sum_offset], dist='BRC_B32')
            m = T.simd.vmax(block_max, run_max, full)
        block_sum = block_exponential_sum(x_ub, valid, m, full, neg)
        if single_block:
            s = block_sum
        else:
            s = T.simd.vadd(
                T.simd.vmul(
                    run_sum,
                    T.simd.vexpdif(run_max, m, full) if narrow else T.simd.vexp(T.simd.vsub(run_max, m, full), full),
                    full,
                ),
                block_sum,
                full,
            )
        T.simd.vsts(max_buffer[wave], m, full, 'ONEPT_B32')
        T.simd.vsts(sum_buffer[sum_offset], s, full, 'ONEPT_B32')

    return update_row_statistics
