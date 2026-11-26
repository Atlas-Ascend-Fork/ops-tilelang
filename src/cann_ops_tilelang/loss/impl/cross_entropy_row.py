# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

from functools import cache

import tilelang.ascend.language as T

from .cross_entropy_common import _compile
from .cross_entropy_simd import _block_maximum, _plain_exponential_lanes, _row_block_statistics_update

_ROW_PIPELINE_STAGES = 2


def _row_block_accumulator(dtype, block, v, single_block, stages):
    update_row_statistics = _row_block_statistics_update(dtype, single_block)

    @T.macro
    def accumulate_block(buffer_x, row, x_ub, maxbuf, sumbuf, wave, sidx):
        for part in T.Pipelined(T.ceildiv(v, block), num_stages=stages):
            valid = T.min(T.int64(block), v - part * block)
            T.copy(buffer_x[row, part * block], x_ub[:valid])
            with T.SimdVF():
                update_row_statistics(x_ub, valid, maxbuf, sumbuf, wave, sidx)

    return accumulate_block


@T.macro
def _simt_mean_half_gap(buffer_x, buffer_y, row, maxbuf, midx):
    return maxbuf[midx] * T.float32(0.5) - T.cast(buffer_x[row, buffer_y[row]], 'float32') * T.float32(0.5)


def _simt_mean_inverse(n, prescale):
    @T.macro
    def inverse(buffer_total):
        if prescale:
            return T.float32(1) / T.cast(n, 'float32')
        else:
            return T.float32(1) / T.cast(buffer_total[0], 'float32')

    return inverse


def _batched_simt_finisher(n, batch_slots, prescale):
    mean_inverse = _simt_mean_inverse(n, prescale)

    @T.macro
    def store_ignored(buffer_loss, owned_row, mean):
        if mean != 0 and n == 1:
            buffer_loss[owned_row] = T.float32(float('nan'))
        else:
            buffer_loss[owned_row] = T.float32(0)

    @T.macro
    def prepare_inverse(buffer_total, inv_count):
        if n == 1:
            inv_count[0] = T.float32(1)
        else:
            inv_count[0] = mean_inverse(buffer_total)

    @T.macro
    def finish(buffer_x, buffer_y, buffer_loss, buffer_total, owned_row, slot, ignore, mean, stats):
        if owned_row < n:
            if buffer_y[owned_row] == ignore:
                if prescale:
                    buffer_total[owned_row] = 0
                store_ignored(buffer_loss, owned_row, mean)
            else:
                if prescale:
                    buffer_total[owned_row] = 1
                if mean != 0:
                    half_gap = _simt_mean_half_gap(buffer_x, buffer_y, owned_row, stats, slot)
                    inv_count = T.alloc_fragment((1,), 'float32')
                    prepare_inverse(buffer_total, inv_count)
                    buffer_loss[owned_row] = T.log(stats[batch_slots + slot]) * inv_count[0] + half_gap * (
                        T.float32(2) * inv_count[0]
                    )
                else:
                    buffer_loss[owned_row] = T.log(stats[batch_slots + slot]) - (
                        T.cast(buffer_x[owned_row, buffer_y[owned_row]], 'float32') - stats[slot]
                    )

    return finish


def _wave_simt_finisher(n, prescale):
    mean_inverse = _simt_mean_inverse(n, prescale)
    sidx = 8 if prescale else 0

    @T.macro
    def finish_row(buffer_x, buffer_y, buffer_loss, buffer_total, row, ignore, mean, row_max, row_sum):
        if buffer_y[row] == ignore:
            if mean != 0 and n == 1:
                buffer_loss[row] = T.float32(float('nan'))
            else:
                buffer_loss[row] = T.float32(0)
        elif mean != 0:
            half_gap = _simt_mean_half_gap(buffer_x, buffer_y, row, row_max, 0)
            inv_count = T.alloc_fragment((1,), 'float32')
            if n == 1:
                inv_count[0] = T.float32(1)
            else:
                inv_count[0] = mean_inverse(buffer_total)
            buffer_loss[row] = T.log(row_sum[sidx]) * inv_count[0] + half_gap * (T.float32(2) * inv_count[0])
        else:
            buffer_loss[row] = T.log(row_sum[sidx]) - (T.cast(buffer_x[row, buffer_y[row]], 'float32') - row_max[0])

    @T.macro
    def finish(buffer_x, buffer_y, buffer_loss, buffer_total, row, ignore, mean, row_max, row_sum):
        with T.SimtVF(threads=32):
            for _ in T.Parallel(1):
                finish_row(buffer_x, buffer_y, buffer_loss, buffer_total, row, ignore, mean, row_max, row_sum)

    return finish


def _row_factory_setup(dtype, block, single_block, batch_slots, static_v):
    if static_v is not None and not (single_block and 0 < static_v <= block):
        raise ValueError('static_v requires a single block and 0 < static_v <= block')
    n = T.dynamic('n', dtype='int64')
    v = T.dynamic('v', dtype='int64') if static_v is None else static_v
    stride = T.dynamic('stride', dtype='int64')

    if batch_slots and not single_block:
        raise ValueError('batch_slots requires a single block')
    stats_size = 2 * batch_slots if batch_slots else 16

    accumulate_row_blocks = _row_block_accumulator(dtype, block, v, single_block, _ROW_PIPELINE_STAGES)
    return n, v, stride, stats_size, accumulate_row_blocks


@cache
def _row_kernel(dtype, block, cores, single_block=False, batch_slots=0, static_v=None):
    n, v, stride, stats_size, accumulate_row_blocks = _row_factory_setup(
        dtype, block, single_block, batch_slots, static_v
    )

    @T.macro
    def initialize_row_statistics(row_max, row_sum):
        with T.SimdVF():
            full_init = T.simd.pset(32)
            T.simd.vsts(
                row_max[0],
                T.simd.vdup(-T.infinity('float32'), T.float32, full_init),
                full_init,
                'ONEPT_B32',
            )
            T.simd.vsts(row_sum[0], T.simd.vdup(0.0, T.float32, full_init), full_init, 'ONEPT_B32')

    store_row_loss = _wave_simt_finisher(n, False)
    store_batched_row_loss = _batched_simt_finisher(n, batch_slots, False)

    @T.macro
    def _row_loss_body(buffer_x, buffer_y, buffer_loss, buffer_total, ignore, mean, core):
        x_ub = T.alloc_shared((block + 128,), dtype)
        if batch_slots:
            stats = T.alloc_shared((stats_size,), 'float32')
            T.annotate_buffer_versions({x_ub: _ROW_PIPELINE_STAGES, stats: 1})
        else:
            row_max = T.alloc_shared((1,), 'float32')
            row_sum = T.alloc_shared((1,), 'float32')
            T.annotate_buffer_versions({x_ub: _ROW_PIPELINE_STAGES, row_max: 1, row_sum: 1})
        for wave in T.serial(T.ceildiv(n, cores)):
            row = wave * cores + core
            if row < n:
                if not single_block:
                    initialize_row_statistics(row_max, row_sum)
                if batch_slots:
                    accumulate_row_blocks(buffer_x, row, x_ub, stats, stats, wave, batch_slots + wave)
                else:
                    accumulate_row_blocks(buffer_x, row, x_ub, row_max, row_sum, 0, 0)
                if not batch_slots:
                    store_row_loss(buffer_x, buffer_y, buffer_loss, buffer_total, row, ignore, mean, row_max, row_sum)
        if batch_slots:
            T.ascend_pipe_barrier('PIPE_ALL')
            with T.SimtVF(threads=max(32, batch_slots)):
                slot = T.get_thread_binding()
                if slot < batch_slots:
                    owned_row = slot * cores + core
                    store_batched_row_loss(
                        buffer_x, buffer_y, buffer_loss, buffer_total, owned_row, slot, ignore, mean, stats
                    )

    @T.prim_func
    def row_loss(
        buffer_x: T.StridedTensor((n, v), (stride, 1), dtype),
        buffer_y: T.Tensor((n,), 'int64'),
        buffer_loss: T.Tensor((n,), 'float32'),
        buffer_total: T.Tensor((1,), 'int64'),
        ignore: T.int64,
        mean: T.int32,
    ):
        with T.Kernel(cores) as core:
            _row_loss_body(buffer_x, buffer_y, buffer_loss, buffer_total, ignore, mean, core)

    return _compile(f'row_{dtype}_{block}_{cores}_{int(single_block)}_{batch_slots}_{static_v}', row_loss)


@cache
def _simd_row_kernel(dtype, block, cores, inline_count=False):
    (n, v, k) = (T.dynamic(name, dtype='int64') for name in ('n', 'v', 'k'))
    stride = T.dynamic('stride', dtype='int64')
    narrow = dtype != 'float32'
    stages = 2

    block_maximum = _block_maximum(dtype)
    exponential_lanes = _plain_exponential_lanes(narrow)

    @T.macro
    def count_nonignored_pair(buffer_y, i, ignore, count):
        if buffer_y[2 * i] != T.cast(ignore, 'int32') or buffer_y[2 * i + 1] != T.cast(ignore >> 32, 'int32'):
            T.reducer_update(count[0], T.int32(1))

    @T.macro
    def reduce_inline_count(buffer_y, buffer_total, count_ub, ignore, core):
        with T.SimtVF(threads=32):
            count = T.alloc_reducer((1,), 'int32', op='sum')
            T.reducer_init(count)
            for i in T.Parallel(32):
                if i < n:
                    count_nonignored_pair(buffer_y, i, ignore, count)
            result = T.alloc_fragment((1,), 'int32')
            T.finalize_reducer(count, result)
            for _ in T.Parallel(1):
                count_ub[0] = result[0]
                if core == 0:
                    buffer_total[0] = result[0]
                    buffer_total[1] = T.int32(0)

    @T.macro
    def initialize_simd_row_statistics(row_max, row_sum, row_answer):
        with T.SimdVF():
            full_init = T.simd.pset(32)
            T.simd.vsts(
                row_max[0],
                T.simd.vdup(-T.infinity('float32'), T.float32, full_init),
                full_init,
                'ONEPT_B32',
            )
            T.simd.vsts(row_sum[0], T.simd.vdup(0.0, T.float32, full_init), full_init, 'ONEPT_B32')
            T.simd.vsts(row_answer[0], T.simd.vdup(0.0, T.float32, full_init), full_init, 'ONEPT_B32')

    @T.macro
    def compute_online_statistics(x_ub, valid, row_max, row_sum, full):
        neg = T.simd.vdup(-T.infinity('float32'), T.float32, full)
        block_max = block_maximum(x_ub, valid, full, neg)
        run_max = T.simd.vld(row_max[0], dist='BRC_B32')
        run_sum = T.simd.vld(row_sum[0], dist='BRC_B32')
        m = T.simd.vmax(block_max, run_max, full)
        ls = exponential_lanes(x_ub, valid, m, full, neg)
        block_sum = T.simd.vdupv(T.simd.vcadd(ls, full), full)
        s = T.simd.vadd(
            T.simd.vmul(run_sum, T.simd.vexp(T.simd.vsub(run_max, m, full), full), full),
            block_sum,
            full,
        )
        return m, s

    @T.macro
    def store_final_row_result(m, s, answer, ignored, mean, count_ub, output_ub, wave, full):
        result = T.simd.alloc_var(T.float32)
        if mean != 0:
            inverse = T.simd.alloc_var(T.float32)
            if n == 1:
                inverse = T.simd.vdup(1.0, T.float32, full)
            else:
                count = T.simd.vcvt(T.simd.vld(count_ub[0], dist='BRC_B32'), T.float32, full)
                inverse = T.simd.vdiv(T.simd.vdup(1.0, T.float32, full), count, full)
            half = T.simd.vdup(0.5, T.float32, full)
            gap = T.simd.vsub(T.simd.vmul(m, half, full), T.simd.vmul(answer, half, full), full)
            result = T.simd.vadd(
                T.simd.vmul(T.simd.vln(s, full), inverse, full),
                T.simd.vmul(gap, T.simd.vmul(T.simd.vdup(2.0, T.float32, full), inverse, full), full),
                full,
            )
        else:
            result = T.simd.vsub(T.simd.vln(s, full), T.simd.vsub(answer, m, full), full)
        if mean != 0 and n == 1:
            result = T.simd.vsel(T.simd.vdup(T.float32(float('nan')), T.float32, full), result, ignored)
        else:
            result = T.simd.vsel(T.simd.vdup(0.0, T.float32, full), result, ignored)
        T.simd.vsts(output_ub[wave], result, full, 'ONEPT_B32')

    @T.macro
    def update_simd_block_and_store(x_ub, valid, row_stats, label_ub, ignore, part, mean, count_ub, output_ub, wave):
        row_max, row_sum, row_answer = row_stats
        with T.SimdVF():
            full = T.simd.pset(32)
            m, s = compute_online_statistics(x_ub, valid, row_max, row_sum, full)
            low = T.simd.vld(label_ub[0], dist='BRC_B32')
            high = T.simd.vld(label_ub[1], dist='BRC_B32')
            ignored = T.simd.pand(
                T.simd.vcmps(low, T.cast(ignore, 'int32'), full, 'eq'),
                T.simd.vcmps(high, T.cast(ignore >> 32, 'int32'), full, 'eq'),
                full,
            )
            in_block = T.simd.pand(
                T.simd.vcmps(low, T.cast(part * block, 'int32'), full, 'ge'),
                T.simd.vcmps(low, T.cast(part * block + valid, 'int32'), full, 'lt'),
                full,
            )
            in_block = T.simd.pand(in_block, T.simd.vcmps(high, 0, full, 'eq'), full)
            offset = T.simd.vsub(low, T.simd.vdup(T.cast(part * block, 'int32'), T.int32, full), full)
            safe_offset = T.simd.vsel(offset, T.simd.vdup(0, T.int32, full), in_block)
            answer = T.simd.vld(row_answer[0], dist='BRC_B32')
            if narrow:
                gather_full = T.simd.pset(16)
                gather_index = T.simd.vdupv(T.simd.vcvt(safe_offset, T.uint16, full), gather_full)
                gathered = T.simd.vcvt(
                    T.simd.vgather2(T.access_ptr(x_ub[0], 'r', extent=block), gather_index, gather_full),
                    T.float32,
                    part=0,
                )
            else:
                gathered = T.simd.vgather2(
                    T.access_ptr(x_ub[0], 'r', extent=block),
                    T.reinterpret(safe_offset, 'uint32x64'),
                    full,
                )
            answer = T.simd.vsel(gathered, answer, in_block)
            if part + 1 == T.ceildiv(v, block):
                store_final_row_result(m, s, answer, ignored, mean, count_ub, output_ub, wave, full)
            else:
                T.simd.vsts(row_max[0], m, full, 'ONEPT_B32')
                T.simd.vsts(row_sum[0], s, full, 'ONEPT_B32')
                T.simd.vsts(row_answer[0], answer, full, 'ONEPT_B32')

    @T.macro
    def _simd_row_loss_body(buffer_x, buffer_y, buffer_loss, buffer_total, ignore, mean, core):
        x_ub = T.alloc_shared((block + 128,), dtype)
        row_max = T.alloc_shared((1,), 'float32')
        row_sum = T.alloc_shared((1,), 'float32')
        row_answer = T.alloc_shared((1,), 'float32')
        output_ub = T.alloc_shared((8,), 'float32')
        label_ub = T.alloc_shared((8,), 'int32')
        count_ub = T.alloc_shared((1 if inline_count else 8,), 'int32')
        T.annotate_buffer_versions({x_ub: stages, row_max: 1, row_sum: 1, row_answer: 1, output_ub: 1})
        if inline_count:
            reduce_inline_count(buffer_y, buffer_total, count_ub, ignore, core)
            T.ascend_pipe_barrier('PIPE_ALL')
        elif mean != 0 and n > 1:
            T.copy(buffer_total[0:2], count_ub[0:2])
        row_begin = n * core // cores
        row_end = n * (core + 1) // cores
        for wave in T.Pipelined(row_end - row_begin, num_stages=2):
            row = row_begin + wave
            if row < row_end:
                T.copy(buffer_y[2 * row], label_ub[0:2])
                initialize_simd_row_statistics(row_max, row_sum, row_answer)
                for part in T.Pipelined(T.ceildiv(v, block), num_stages=stages):
                    valid = T.min(T.int64(block), v - part * block)
                    T.copy(buffer_x[row, part * block], x_ub[:valid])
                    update_simd_block_and_store(
                        x_ub,
                        valid,
                        (row_max, row_sum, row_answer),
                        label_ub,
                        ignore,
                        part,
                        mean,
                        count_ub,
                        output_ub,
                        wave,
                    )
        if row_begin < row_end:
            T.copy(output_ub[: row_end - row_begin], buffer_loss[row_begin:row_end], l2_cache_ctrl=0)

    @T.prim_func
    def row_loss(
        buffer_x: T.StridedTensor((n, v), (stride, 1), dtype),
        buffer_y: T.Tensor((2 * n,), 'int32'),
        buffer_loss: T.Tensor((n,), 'float32'),
        buffer_total: T.Tensor((2,), 'int32'),
        ignore: T.int64,
        mean: T.int32,
    ):
        with T.Kernel(cores) as core:
            _simd_row_loss_body(buffer_x, buffer_y, buffer_loss, buffer_total, ignore, mean, core)

    return _compile(f'row_fullsimd_{dtype}_{block}_{cores}_{int(inline_count)}', row_loss)


@cache
def _scaled_row_kernel(dtype, block, cores, single_block=False, batch_slots=0, static_v=None):
    n, v, stride, stats_size, accumulate_row_blocks = _row_factory_setup(
        dtype, block, single_block, batch_slots, static_v
    )

    @T.macro
    def initialize_scaled_statistics(stats):
        with T.SimdVF():
            full_init = T.simd.pset(32)
            T.simd.vsts(
                stats[0],
                T.simd.vdup(-T.infinity('float32'), T.float32, full_init),
                full_init,
                'ONEPT_B32',
            )
            T.simd.vsts(stats[8], T.simd.vdup(0.0, T.float32, full_init), full_init, 'ONEPT_B32')

    store_scaled_row_loss = _wave_simt_finisher(n, True)
    store_batched_scaled_loss = _batched_simt_finisher(n, batch_slots, True)

    @T.macro
    def _scaled_row_loss_body(buffer_x, buffer_y, buffer_loss, buffer_total, ignore, mean, core):
        x_ub = T.alloc_shared((block + 128,), dtype)
        stats = T.alloc_shared((stats_size,), 'float32')
        T.annotate_buffer_versions({x_ub: _ROW_PIPELINE_STAGES, stats: 1})
        for wave in T.serial(T.ceildiv(n, cores)):
            row = wave * cores + core
            if row < n:
                if not single_block:
                    initialize_scaled_statistics(stats)
                if batch_slots:
                    accumulate_row_blocks(buffer_x, row, x_ub, stats, stats, wave, batch_slots + wave)
                else:
                    accumulate_row_blocks(buffer_x, row, x_ub, stats, stats, 0, 8)
                if not batch_slots:
                    store_scaled_row_loss(buffer_x, buffer_y, buffer_loss, stats, row, ignore, mean, stats, stats)
        if batch_slots:
            T.ascend_pipe_barrier('PIPE_ALL')
            with T.SimtVF(threads=max(32, batch_slots)):
                slot = T.get_thread_binding()
                if slot < batch_slots:
                    owned_row = slot * cores + core
                    store_batched_scaled_loss(
                        buffer_x, buffer_y, buffer_loss, buffer_total, owned_row, slot, ignore, mean, stats
                    )

    @T.prim_func
    def row_loss(
        buffer_x: T.StridedTensor((n, v), (stride, 1), dtype),
        buffer_y: T.Tensor((n,), 'int64'),
        buffer_loss: T.Tensor((n,), 'float32'),
        buffer_total: T.Tensor((n,), 'int32'),
        ignore: T.int64,
        mean: T.int32,
    ):
        with T.Kernel(cores) as core:
            _scaled_row_loss_body(buffer_x, buffer_y, buffer_loss, buffer_total, ignore, mean, core)

    return _compile(f'row_{dtype}_{block}_{cores}_{int(single_block)}_{batch_slots}_{static_v}', row_loss)
