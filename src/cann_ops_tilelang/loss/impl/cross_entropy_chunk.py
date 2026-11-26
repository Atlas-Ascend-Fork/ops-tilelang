# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

from functools import cache

import tilelang.ascend.language as T

from .cross_entropy_common import _compile
from .cross_entropy_simd import _block_exponential_lanes, _block_maximum, _plain_exponential_lanes


@cache
def _chunk_kernel(dtype, block, cores):
    (n, v, k) = (T.dynamic(name, dtype='int64') for name in ('n', 'v', 'k'))
    stride = T.dynamic('stride', dtype='int64')
    narrow = dtype != 'float32'
    stages = 2

    block_maximum = _block_maximum(dtype)
    exponential_lanes = _plain_exponential_lanes(narrow)

    @T.macro
    def _plain_chunk_task(buffer_x, buffer_max, buffer_sum, x_ub, row_max, row_sum, row, part):
        valid = T.min(T.int64(block), v - part * block)
        T.copy(buffer_x[row, slice(part * block, part * block + valid)], x_ub[:valid])
        with T.SimdVF():
            full = T.simd.pset(32)
            neg = T.simd.vdup(-T.infinity('float32'), T.float32, full)
            block_max = block_maximum(x_ub, valid, full, neg)
            m = block_max
            ls = exponential_lanes(x_ub, valid, m, full, neg)
            s = T.simd.vcadd(ls, full)
            T.simd.vsts(row_max[0], m, full, 'ONEPT_B32')
            T.simd.vsts(row_sum[0], s, full, 'ONEPT_B32')
        T.copy(row_max[0:1], buffer_max[row, slice(part, part + 1)])
        T.copy(row_sum[0:1], buffer_sum[row, slice(part, part + 1)])

    @T.prim_func
    def chunk(
        buffer_x: T.StridedTensor((n, v), (stride, 1), dtype),
        buffer_max: T.Tensor((n, k), 'float32'),
        buffer_sum: T.Tensor((n, k), 'float32'),
    ):
        with T.Kernel(cores) as core:
            x_ub = T.alloc_shared((block + 128,), dtype)
            row_max = T.alloc_shared((1,), 'float32')
            row_sum = T.alloc_shared((1,), 'float32')
            T.annotate_buffer_versions({x_ub: stages, row_max: stages, row_sum: stages})
            for wave in T.Pipelined(T.ceildiv(n * k, cores), num_stages=stages):
                task = wave * cores + core
                if task < n * k:
                    row = task // k
                    part = task % k
                    _plain_chunk_task(buffer_x, buffer_max, buffer_sum, x_ub, row_max, row_sum, row, part)

    return _compile(f'chunk_{dtype}_{block}_{cores}', chunk)


@cache
def _fully_simd_fused_kernel(dtype, block, cores, slots):
    (n, v) = (T.dynamic(name, dtype='int32') for name in ('n', 'v'))
    k = T.ceildiv(v, block)
    stride = T.dynamic('stride', dtype='int64')
    narrow = dtype != 'float32'
    stages = 2
    out_size = T.dynamic('out_size', dtype='int32')
    row_capacity = (2 * cores + 63) // 64 * 64

    block_maximum = _block_maximum(dtype)
    exponential_lanes = _block_exponential_lanes(dtype)

    @T.macro
    def _label_words(labels, label_index, full):
        low = T.simd.vgather2(
            T.access_ptr(labels[0], 'r', extent=4 * cores + 64),
            T.reinterpret(label_index, 'uint32x64'),
            full,
        )
        high = T.simd.vgather2(
            T.access_ptr(labels[0], 'r', extent=4 * cores + 64),
            T.reinterpret(T.simd.vadd(label_index, T.simd.vdup(1, T.int32, full), full), 'uint32x64'),
            full,
        )
        return low, high

    @T.macro
    def _chunk_statistics_vf(x_ub, labels, stats, row, part, valid, wave):
        with T.SimdVF():
            full = T.simd.pset(32)
            neg = T.simd.vdup(-T.infinity('float32'), T.float32, full)
            block_max = block_maximum(x_ub, valid, full, neg)
            m = block_max
            ls = exponential_lanes(x_ub, valid, m, full, neg)
            label_index = T.simd.vdup(row * 2, T.int32, full)
            low, high = _label_words(labels, label_index, full)
            in_block = T.simd.pand(
                T.simd.vcmps(low, part * block, full, 'ge'),
                T.simd.vcmps(low, part * block + valid, full, 'lt'),
                full,
            )
            in_block = T.simd.pand(in_block, T.simd.vcmps(high, 0, full, 'eq'), full)
            offset = T.simd.vsub(low, T.simd.vdup(part * block, T.int32, full), full)
            safe_offset = T.simd.vsel(offset, T.simd.vdup(0, T.int32, full), in_block)
            if narrow:
                gather_full = T.simd.pset(16)
                gather_index = T.simd.vdupv(T.simd.vcvt(safe_offset, T.uint16, full), gather_full)
                answer = T.simd.vcvt(
                    T.simd.vgather2(T.access_ptr(x_ub[0], 'r', extent=block), gather_index, gather_full),
                    T.float32,
                    part=0,
                )
            else:
                answer = T.simd.vgather2(
                    T.access_ptr(x_ub[0], 'r', extent=block), T.reinterpret(safe_offset, 'uint32x64'), full
                )
            answer = T.simd.vsel(answer, T.simd.vdup(0.0, T.float32, full), in_block)
            s = T.simd.vcadd(ls, full)
            T.simd.vsts(stats[wave * 8], m, full, 'ONEPT_B32')
            T.simd.vsts(stats[wave * 8 + 1], s, full, 'ONEPT_B32')
            T.simd.vsts(stats[wave * 8 + 2], answer, full, 'ONEPT_B32')

    @T.macro
    def _row_lanes(n, group, full):
        active = T.simd.update_mask(T.min(64, n - group * 64), 32)
        rows = T.simd.vadd(T.simd.vci(0, 'int32'), T.simd.vdup(group * 64, T.int32, full), full)
        rows = T.simd.vsel(rows, T.simd.vdup(0, T.int32, full), active)
        return active, rows

    @T.macro
    def _label_ignored(low, high, ignore, full):
        ignored = T.simd.pand(
            T.simd.vcmps(low, T.cast(ignore, 'int32'), full, 'eq'),
            T.simd.vcmps(high, T.cast(ignore >> 32, 'int32'), full, 'eq'),
            full,
        )
        return ignored

    @T.macro
    def _valid_row_count(labels, n, ignore, full, zero, one):
        count_lanes = T.simd.alloc_var(T.float32)
        count_lanes = zero
        for group in T.serial(T.ceildiv(n, 64)):
            active, rows = _row_lanes(n, group, full)
            label_index = T.simd.vmul(rows, T.simd.vdup(2, T.int32, full), full)
            low, high = _label_words(labels, label_index, full)
            ignored = _label_ignored(low, high, ignore, full)
            valid_count = T.simd.vsel(zero, one, ignored)
            valid_count = T.simd.vsel(valid_count, zero, active)
            count_lanes = T.simd.vadd(count_lanes, valid_count, full)
        count = T.simd.vdupv(T.simd.vcadd(count_lanes, full), full)
        return count

    @T.macro
    def _corrected_task_quotient(task0, full):
        q0 = T.simd.vcvt(
            T.simd.vmul(T.simd.vcvt(task0, T.float32, full), T.simd.vdup(1.0 / cores, T.float32, full), full),
            T.int32,
            full,
            round='ROUND_Z',
        )
        rem0 = T.simd.vsub(task0, T.simd.vmul(q0, T.simd.vdup(cores, T.int32, full), full), full)
        q0 = T.simd.vsel(
            T.simd.vadd(q0, T.simd.vdup(1, T.int32, full), full),
            q0,
            T.simd.vcmps(rem0, cores, full, 'ge'),
        )
        q0 = T.simd.vsel(T.simd.vsub(q0, T.simd.vdup(1, T.int32, full), full), q0, T.simd.vcmps(rem0, 0, full, 'lt'))
        return q0

    @T.macro
    def _loss_from_merged_statistics(mode, maximum, answer, total, inverse, full):
        value = T.simd.alloc_var(T.float32)
        if mode == 2:
            half = T.simd.vdup(0.5, T.float32, full)
            gap = T.simd.vsub(T.simd.vmul(maximum, half, full), T.simd.vmul(answer, half, full), full)
            value = T.simd.vadd(
                T.simd.vmul(T.simd.vln(total, full), inverse, full),
                T.simd.vmul(gap, T.simd.vmul(T.simd.vdup(2.0, T.float32, full), inverse, full), full),
                full,
            )
        else:
            value = T.simd.vsub(T.simd.vln(total, full), T.simd.vsub(answer, maximum, full), full)
        return value

    @T.macro
    def _group_origin(rows, full):
        task0 = T.simd.vmul(rows, T.simd.vdup(k, T.int32, full), full)
        q0 = _corrected_task_quotient(task0, full)
        owner0 = T.simd.vsub(task0, T.simd.vmul(q0, T.simd.vdup(cores, T.int32, full), full), full)
        index0 = T.simd.vadd(
            T.simd.vmul(owner0, T.simd.vdup(slots * 8, T.int32, full), full),
            T.simd.vmul(q0, T.simd.vdup(8, T.int32, full), full),
            full,
        )
        return owner0, index0

    @T.macro
    def _merged_max_load(merged, index, full):
        chunk_max = T.simd.vgather2(
            T.access_ptr(merged[0, 0], 'r', extent=cores * slots * 8),
            T.reinterpret(index, 'uint32x64'),
            full,
        )
        return chunk_max

    @T.macro
    def _merged_chunk_loads(merged, index, full):
        chunk_max = _merged_max_load(merged, index, full)
        chunk_sum = T.simd.vgather2(
            T.access_ptr(merged[0, 0], 'r', extent=cores * slots * 8),
            T.reinterpret(T.simd.vadd(index, T.simd.vdup(1, T.int32, full), full), 'uint32x64'),
            full,
        )
        chunk_answer = T.simd.vgather2(
            T.access_ptr(merged[0, 0], 'r', extent=cores * slots * 8),
            T.reinterpret(T.simd.vadd(index, T.simd.vdup(2, T.int32, full), full), 'uint32x64'),
            full,
        )
        return chunk_max, chunk_sum, chunk_answer

    @T.macro
    def _masked_group_value(labels, rows, ignore, value, zero, active, full):
        label_index = T.simd.vmul(rows, T.simd.vdup(2, T.int32, full), full)
        low, high = _label_words(labels, label_index, full)
        ignored = _label_ignored(low, high, ignore, full)
        masked = T.simd.vsel(zero, value, ignored)
        return T.simd.vsel(masked, zero, active)

    @T.macro
    def _merged_group(merged, labels, rows, active, ignore, mode, inverse, full, zero):
        maximum = T.simd.alloc_var(T.float32)
        maximum = T.simd.vdup(-T.infinity('float32'), T.float32, full)
        owner0, index0 = _group_origin(rows, full)
        owner = T.simd.alloc_var(T.int32)
        index = T.simd.alloc_var(T.int32)
        owner = owner0
        index = index0
        for _ in T.serial(k):
            chunk_max = _merged_max_load(merged, index, full)
            maximum = T.simd.vmax(maximum, chunk_max, full)
            owner = T.simd.vadd(owner, T.simd.vdup(1, T.int32, full), full)
            wrap = T.simd.vcmps(owner, cores, full, 'ge')
            owner = T.simd.vsel(T.simd.vsub(owner, T.simd.vdup(cores, T.int32, full), full), owner, wrap)
            index = T.simd.vadd(index, T.simd.vdup(slots * 8, T.int32, full), full)
            index = T.simd.vsel(
                T.simd.vsub(index, T.simd.vdup(cores * slots * 8 - 8, T.int32, full), full), index, wrap
            )
        total = T.simd.alloc_var(T.float32)
        answer = T.simd.alloc_var(T.float32)
        total = zero
        answer = zero
        owner = owner0
        index = index0
        for _ in T.serial(k):
            chunk_max, chunk_sum, chunk_answer = _merged_chunk_loads(merged, index, full)
            if narrow:
                total = T.simd.vadd(total, T.simd.vmul(chunk_sum, T.simd.vexpdif(chunk_max, maximum, full), full), full)
            else:
                total = T.simd.vadd(
                    total,
                    T.simd.vmul(chunk_sum, T.simd.vexp(T.simd.vsub(chunk_max, maximum, full), full), full),
                    full,
                )
            answer = T.simd.vadd(answer, chunk_answer, full)
            owner = T.simd.vadd(owner, T.simd.vdup(1, T.int32, full), full)
            wrap = T.simd.vcmps(owner, cores, full, 'ge')
            owner = T.simd.vsel(T.simd.vsub(owner, T.simd.vdup(cores, T.int32, full), full), owner, wrap)
            index = T.simd.vadd(index, T.simd.vdup(slots * 8, T.int32, full), full)
            index = T.simd.vsel(
                T.simd.vsub(index, T.simd.vdup(cores * slots * 8 - 8, T.int32, full), full), index, wrap
            )
        value = _loss_from_merged_statistics(mode, maximum, answer, total, inverse, full)
        value = _masked_group_value(labels, rows, ignore, value, zero, active, full)
        return value

    @T.macro
    def _merge_statistics_vf(merged, labels, output, n, ignore, mode):
        with T.SimdVF():
            full = T.simd.pset(32)
            zero = T.simd.vdup(0.0, T.float32, full)
            one = T.simd.vdup(1.0, T.float32, full)
            count = _valid_row_count(labels, n, ignore, full, zero, one)
            inverse = T.simd.vdiv(one, count, full)
            combined = T.simd.alloc_var(T.float32)
            combined = zero
            for group in T.serial(T.ceildiv(n, 64)):
                active, rows = _row_lanes(n, group, full)
                value = _merged_group(merged, labels, rows, active, ignore, mode, inverse, full, zero)
                if mode == 0:
                    T.simd.vsts(output[group * 64], value, active)
                else:
                    combined = T.simd.vadd(combined, value, full)
            if mode != 0:
                result = T.simd.alloc_var(T.float32)
                result = T.simd.vcadd(combined, full)
                if mode == 2:
                    result = T.simd.vsel(
                        T.simd.vdup(T.float32(float('nan')), T.float32, full),
                        result,
                        T.simd.vcmps(count, 0.0, full, 'eq'),
                    )
                T.simd.vsts(output[0], result, full, 'ONEPT_B32')

    @T.prim_func
    def chunk(
        buffer_x: T.StridedTensor((n, v), (stride, 1), dtype),
        buffer_y: T.Tensor((n * 2,), 'int32'),
        buffer_workspace: T.Tensor((cores, slots * 8), 'float32'),
        buffer_loss: T.Tensor((out_size,), 'float32'),
        ignore: T.int64,
        mode: T.int32,
    ):
        with T.Kernel(cores) as core:
            x_ub = T.alloc_shared((block + 128,), dtype)
            stats = T.alloc_shared((slots * 8,), 'float32')
            labels = T.alloc_shared((4 * cores + 64,), 'int32')
            merged = T.alloc_shared((cores, slots * 8), 'float32')
            output = T.alloc_shared((row_capacity,), 'float32')
            T.annotate_buffer_versions({x_ub: stages, stats: 1, labels: 1, merged: 1, output: 1})
            T.copy(buffer_y[: n * 2], labels[: n * 2])
            for wave in T.Pipelined(T.ceildiv(n * k, cores), num_stages=stages):
                task = wave * cores + core
                if task < n * k:
                    row = T.alloc_var('int32', init=task // k)
                    part = T.alloc_var('int32', init=task % k)
                    valid = T.alloc_var('int32', init=T.min(T.int32(block), v - part * block))
                    T.copy(buffer_x[row, part * block], x_ub[:valid])
                    _chunk_statistics_vf(x_ub, labels, stats, row, part, valid, wave)
            with T.PerCoreTask():
                T.copy(stats[: slots * 8], buffer_workspace[core, : slots * 8], l2_cache_ctrl=0)
                T.ascend_sync_inter_arrive('PIPE_MTE3', 15)
            T.ascend_sync_inter_wait('PIPE_S', 15)
            if core == 0:
                T.copy(buffer_workspace[:cores, : slots * 8], merged)
                _merge_statistics_vf(merged, labels, output, n, ignore, mode)
                if mode == 0:
                    T.copy(output[:n], buffer_loss[:n], l2_cache_ctrl=0)
                else:
                    T.copy(output[:1], buffer_loss[:1], l2_cache_ctrl=0)

    return _compile(f'row_fully_simd_fused_{dtype}_{block}_{cores}_{slots}', chunk)
