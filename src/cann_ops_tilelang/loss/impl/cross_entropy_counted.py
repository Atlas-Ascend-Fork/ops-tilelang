# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

from functools import cache

import tilelang.ascend.language as T

from .cross_entropy_common import _compile
from .cross_entropy_simd import _block_exponential_sum, _block_maximum


def _row_statistics_update(dtype):
    narrow = dtype != 'float32'
    block_maximum = _block_maximum(dtype)
    block_exponential_sum = _block_exponential_sum(dtype)

    @T.macro
    def update_row_statistics(x_ub, valid, row_max, row_sum):
        full = T.simd.pset(32)
        neg = T.simd.vdup(-T.infinity('float32'), T.float32, full)
        block_max = block_maximum(x_ub, valid, full, neg)
        run_max = T.simd.vld(row_max[0], dist='BRC_B32')
        run_sum = T.simd.vld(row_sum[0], dist='BRC_B32')
        m = T.simd.vmax(block_max, run_max, full)
        block_sum = block_exponential_sum(x_ub, valid, m, full, neg)
        s = T.simd.vadd(
            T.simd.vmul(
                run_sum,
                T.simd.vexpdif(run_max, m, full) if narrow else T.simd.vexp(T.simd.vsub(run_max, m, full), full),
                full,
            ),
            block_sum,
            full,
        )
        T.simd.vsts(row_max[0], m, full, 'ONEPT_B32')
        T.simd.vsts(row_sum[0], s, full, 'ONEPT_B32')

    return update_row_statistics


@cache
def _counted_row_kernel(dtype, block, cores, scaled):
    (n, v, k) = (T.dynamic(name, dtype='int64') for name in ('n', 'v', 'k'))
    stride = T.dynamic('stride', dtype='int64')
    stages = 2

    update_row_statistics = _row_statistics_update(dtype)

    @T.macro
    def counted_statistics_block(buffer_x, row, part, x_ub, row_max, row_sum):
        valid = T.min(T.int64(block), v - part * block)
        T.copy(buffer_x[row, part * block], x_ub[:valid])
        with T.SimdVF():
            update_row_statistics(x_ub, valid, row_max, row_sum)

    @T.macro
    def counted_store_row_loss(buffer_x, buffer_y, buffer_loss, buffer_counts, row, ignore, row_max, row_sum):
        for _ in T.Parallel(1):
            if buffer_y[row] == ignore:
                buffer_counts[row] = 0
                buffer_loss[row] = T.float32(0)
            else:
                buffer_counts[row] = 1
                if scaled:
                    half_gap = row_max[0] * T.float32(0.5) - T.cast(
                        buffer_x[row, buffer_y[row]], 'float32'
                    ) * T.float32(0.5)
                    inv_n = T.float32(1) / T.cast(n, 'float32')
                    buffer_loss[row] = T.log(row_sum[0]) * inv_n + half_gap * (T.float32(2) * inv_n)
                else:
                    buffer_loss[row] = T.log(row_sum[0]) - (
                        T.cast(buffer_x[row, buffer_y[row]], 'float32') - row_max[0]
                    )

    @T.macro
    def _counted_row_loss_body(buffer_x, buffer_y, buffer_loss, buffer_counts, ignore, core):
        x_ub = T.alloc_shared((block + 128,), dtype)
        row_max = T.alloc_shared((1,), 'float32')
        row_sum = T.alloc_shared((1,), 'float32')
        T.annotate_buffer_versions({x_ub: stages, row_max: 1, row_sum: 1})
        for wave in T.serial(T.ceildiv(n, cores)):
            row = wave * cores + core
            if row < n:
                with T.SimdVF():
                    full_init = T.simd.pset(32)
                    T.simd.vsts(
                        row_max[0],
                        T.simd.vdup(-T.infinity('float32'), T.float32, full_init),
                        full_init,
                        'ONEPT_B32',
                    )
                    T.simd.vsts(row_sum[0], T.simd.vdup(0.0, T.float32, full_init), full_init, 'ONEPT_B32')
                for part in T.Pipelined(T.ceildiv(v, block), num_stages=stages):
                    counted_statistics_block(buffer_x, row, part, x_ub, row_max, row_sum)
                with T.SimtVF(threads=32):
                    counted_store_row_loss(
                        buffer_x, buffer_y, buffer_loss, buffer_counts, row, ignore, row_max, row_sum
                    )

    @T.prim_func
    def row_loss(
        buffer_x: T.StridedTensor((n, v), (stride, 1), dtype),
        buffer_y: T.Tensor((n,), 'int64'),
        buffer_loss: T.Tensor((n,), 'float32'),
        buffer_counts: T.Tensor((n,), 'int32'),
        ignore: T.int64,
    ):
        with T.Kernel(cores) as core:
            _counted_row_loss_body(buffer_x, buffer_y, buffer_loss, buffer_counts, ignore, core)

    return _compile(f'row_count_{dtype}_{block}_{cores}_{scaled}', row_loss)


@cache
def _partial_launch_kernel(dtype, block, cores, scaled):
    (n, v, k) = (T.dynamic(name, dtype='int64') for name in ('n', 'v', 'k'))
    stride = T.dynamic('stride', dtype='int64')
    stages = 2

    update_row_statistics = _row_statistics_update(dtype)

    @T.macro
    def partial_statistics_block(buffer_x, row, part, x_ub, row_max, row_sum):
        valid = T.min(T.int64(block), v - part * block)
        T.copy(buffer_x[row, part * block], x_ub[:valid])
        with T.SimdVF():
            update_row_statistics(x_ub, valid, row_max, row_sum)

    @T.macro
    def partial_accumulate_row_loss(
        buffer_x, buffer_y, row, ignore, row_max, row_sum, partial_count, partial_sum, compensation
    ):
        for _ in T.Parallel(1):
            if buffer_y[row] != ignore:
                partial_count[0] = partial_count[0] + 1
                half_gap = row_max[0] * T.float32(0.5) - T.cast(buffer_x[row, buffer_y[row]], 'float32') * T.float32(
                    0.5
                )
                inv_n = T.float32(1) / T.cast(n, 'float32')
                row_loss_scaled = T.log(row_sum[0]) * inv_n + half_gap * (T.float32(2) * inv_n)
                old_sum = partial_sum[0]
                corrected = row_loss_scaled - compensation[0]
                updated = T.call_pure_extern('float32', 'fmaf', old_sum, T.float32(1), corrected)
                new_compensation = T.if_then_else(T.isfinite(updated), updated - old_sum - corrected, T.float32(0))
                compensation[0] = new_compensation
                partial_sum[0] = updated

    @T.macro
    def partial_reduce_and_store(buffer_loss, buffer_counts, buffer_out):
        sums = T.alloc_reducer((1,), 'float32', op='sum')
        counts = T.alloc_reducer((1,), 'int32', op='sum')
        T.reducer_init(sums)
        T.reducer_init(counts)
        for i in T.Parallel(cores):
            T.reducer_update(sums[0], buffer_loss[i])
            T.reducer_update(counts[0], buffer_counts[i])
        total = T.alloc_fragment((1,), 'float32')
        count = T.alloc_fragment((1,), 'int32')
        T.finalize_reducer(sums, total)
        T.finalize_reducer(counts, count)
        for _ in T.Parallel(1):
            if scaled:
                buffer_out[0] = total[0] * (T.cast(n, 'float32') / T.cast(count[0], 'float32'))
            else:
                buffer_out[0] = total[0] * T.cast(n, 'float32')

    @T.macro
    def _partial_launch_loss_body(buffer_x, buffer_y, buffer_loss, buffer_counts, buffer_out, ignore, core):
        x_ub = T.alloc_shared((block + 128,), dtype)
        row_max = T.alloc_shared((1,), 'float32')
        row_sum = T.alloc_shared((1,), 'float32')
        T.annotate_buffer_versions({x_ub: stages, row_max: 1, row_sum: 1})
        partial_sum = T.alloc_shared((1,), 'float32')
        partial_count = T.alloc_shared((1,), 'int32')
        compensation = T.alloc_shared((1,), 'float32')
        T.annotate_buffer_versions({partial_sum: 1, partial_count: 1, compensation: 1})
        partial_sum[0] = T.float32(0)
        partial_count[0] = 0
        compensation[0] = T.float32(0)
        for wave in T.serial(T.ceildiv(n, cores)):
            row = wave * cores + core
            if row < n:
                with T.SimdVF():
                    row_max[0] = -T.infinity('float32')
                    row_sum[0] = T.float32(0)
                for part in T.Pipelined(T.ceildiv(v, block), num_stages=stages):
                    partial_statistics_block(buffer_x, row, part, x_ub, row_max, row_sum)
                with T.SimtVF(threads=32):
                    partial_accumulate_row_loss(
                        buffer_x, buffer_y, row, ignore, row_max, row_sum, partial_count, partial_sum, compensation
                    )
        with T.PerCoreTask():
            with T.SimtVF(threads=32):
                for _ in T.Parallel(1):
                    buffer_loss[core] = partial_sum[0]
                    buffer_counts[core] = partial_count[0]
                T.ascend_threadfence()
            T.ascend_cross_core_set_flag(0, 'PIPE_V', 14)
            T.ascend_cross_core_wait_flag(0, 'PIPE_V', 14)
        if core == 0:
            with T.SimtVF(threads=256):
                partial_reduce_and_store(buffer_loss, buffer_counts, buffer_out)

    @T.prim_func
    def row_loss(
        buffer_x: T.StridedTensor((n, v), (stride, 1), dtype),
        buffer_y: T.Tensor((n,), 'int64'),
        buffer_loss: T.Tensor((cores,), 'float32'),
        buffer_counts: T.Tensor((cores,), 'int32'),
        buffer_out: T.Tensor((1,), 'float32'),
        ignore: T.int64,
    ):
        with T.Kernel(cores) as core:
            _partial_launch_loss_body(buffer_x, buffer_y, buffer_loss, buffer_counts, buffer_out, ignore, core)

    return _compile(f'row_partial_compensated_{dtype}_{block}_{cores}_{scaled}', row_loss)
