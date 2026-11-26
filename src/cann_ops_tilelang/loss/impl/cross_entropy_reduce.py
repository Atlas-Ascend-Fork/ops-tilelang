# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

from functools import cache

import tilelang.ascend.language as T

from .cross_entropy_common import ROW_BLOCK, _compile


def _make_small_loss_batch(n, v, width, rows):
    @T.macro
    def loss_batch(buffer_x, buffer_y, buffer_loss, buffer_total, start, ignore, mean):
        values = T.alloc_fragment((rows, width), 'float32')
        exponential = T.alloc_fragment((rows, width), 'float32')
        maximum = T.alloc_fragment((rows,), 'float32')
        total = T.alloc_fragment((rows,), 'float32')
        T.annotate_layout(
            {
                values: T.Fragment(
                    (rows, width),
                    forward_thread_fn=lambda r, c: r * 32 + c % 32,
                    forward_index_fn=lambda r, c: c // 32,
                ),
                exponential: T.Fragment(
                    (rows, width),
                    forward_thread_fn=lambda r, c: r * 32 + c % 32,
                    forward_index_fn=lambda r, c: c // 32,
                ),
            }
        )
        for r, col in T.Parallel(rows, width):
            values[r, col] = -T.infinity('float32')
            if start + r < n and col < v:
                values[r, col] = T.cast(buffer_x[start + r, col], 'float32')
        T.reduce_max(values, maximum, dim=1)
        for r, col in T.Parallel(rows, width):
            exponential[r, col] = T.float32(0)
            if start + r < n and col < v:
                exponential[r, col] = T.exp(values[r, col] - maximum[r])
        T.reduce_sum(exponential, total, dim=1)
        for r in T.Parallel(rows):
            row = start + r
            if row < n:
                if buffer_y[row] == ignore:
                    buffer_loss[row] = T.float32(0)
                elif mean != 0:
                    half_gap = maximum[r] * T.float32(0.5) - T.cast(
                        buffer_x[row, buffer_y[row]], 'float32'
                    ) * T.float32(0.5)
                    inv_count = T.float32(1) / T.cast(buffer_total[0], 'float32')
                    buffer_loss[row] = T.log(total[r]) * inv_count + half_gap * (T.float32(2) * inv_count)
                else:
                    buffer_loss[row] = T.log(total[r]) - (T.cast(buffer_x[row, buffer_y[row]], 'float32') - maximum[r])

    return loss_batch


@cache
def _batched_small_kernel(dtype, width, rows, cores):
    (n, v) = (T.dynamic(name, dtype='int64') for name in ('n', 'v'))
    stride = T.dynamic('stride', dtype='int64')
    loss_batch = _make_small_loss_batch(n, v, width, rows)

    @T.prim_func
    def small_loss(
        buffer_x: T.StridedTensor((n, v), (stride, 1), dtype),
        buffer_y: T.Tensor((n,), 'int64'),
        buffer_loss: T.Tensor((n,), 'float32'),
        buffer_total: T.Tensor((1,), 'int64'),
        ignore: T.int64,
        mean: T.int32,
    ):
        with T.Kernel(cores) as core, T.SimtVF(threads=32 * rows):
            for wave in T.serial(T.ceildiv(n, cores * rows)):
                start = (wave * cores + core) * rows
                loss_batch(buffer_x, buffer_y, buffer_loss, buffer_total, start, ignore, mean)

    return _compile(f'small_{dtype}_{width}_{rows}_{cores}', small_loss)


def _make_valid_count_tile(n, cores, acc_dtype):
    @T.macro
    def count_nonignored(buffer_y, row, ignore, count):
        if buffer_y[row] != ignore:
            T.reducer_update(count[0], T.cast(1, acc_dtype))

    @T.macro
    def count_tile(buffer_y, count, tile, core, ignore):
        for i in T.Parallel(ROW_BLOCK):
            row = (tile * cores + core) * ROW_BLOCK + i
            if row < n:
                count_nonignored(buffer_y, row, ignore, count)

    return count_tile


@cache
def _valid_count_kernel(wide=False, cores=1):
    acc_dtype = 'int64' if wide else 'int32'
    threads = 32 if wide else 256
    n = T.dynamic('n', dtype='int64')
    count_tile = _make_valid_count_tile(n, cores, acc_dtype)

    @T.prim_func
    def count_valid(buffer_y: T.Tensor((n,), 'int64'), buffer_total: T.Tensor((cores,), 'int64'), ignore: T.int64):
        with T.Kernel(cores) as core, T.SimtVF(threads=threads):
            count = T.alloc_reducer((1,), acc_dtype, op='sum')
            T.reducer_init(count)
            for tile in T.serial(T.ceildiv(n, ROW_BLOCK * cores)):
                count_tile(buffer_y, count, tile, core, ignore)
            result = T.alloc_fragment((1,), acc_dtype)
            T.finalize_reducer(count, result)
            for i in T.Parallel(1):
                buffer_total[core] = T.cast(result[i], 'int64')

    return _compile(f'count_valid_{acc_dtype}_{cores}', count_valid)


def _make_count_sum_tile(n):
    @T.macro
    def sum_tile(partial_counts, count, tile):
        for i in T.Parallel(32):
            row = tile * 32 + i
            if row < n:
                T.reducer_update(count[0], partial_counts[row])

    return sum_tile


@cache
def _valid_count_merge_kernel():
    n = T.dynamic('n', dtype='int64')
    sum_tile = _make_count_sum_tile(n)

    @T.prim_func
    def count_sum(partial_counts: T.Tensor((n,), 'int64'), total_count: T.Tensor((1,), 'int64')):
        with T.Kernel(1) as _, T.SimtVF(threads=32):
            count = T.alloc_reducer((1,), 'int64', op='sum')
            T.reducer_init(count)
            for tile in T.serial(T.ceildiv(n, 32)):
                sum_tile(partial_counts, count, tile)
            result = T.alloc_fragment((1,), 'int64')
            T.finalize_reducer(count, result)
            for _ in T.Parallel(1):
                total_count[0] = result[0]

    return _compile('count_sum', count_sum)


def _make_merge_row(n, k, width):
    @T.macro
    def store_loss(buffer_x, buffer_y, buffer_loss, buffer_total, row, mean, m, s):
        for _ in T.Parallel(1):
            if mean != 0:
                half_gap = m[0] * T.float32(0.5) - T.cast(buffer_x[row, buffer_y[row]], 'float32') * T.float32(0.5)
                inv_count = T.alloc_fragment((1,), 'float32')
                if n == 1:
                    inv_count[0] = T.float32(1)
                else:
                    inv_count[0] = T.float32(1) / T.cast(buffer_total[0], 'float32')
                buffer_loss[row] = T.log(s[0]) * inv_count[0] + half_gap * (T.float32(2) * inv_count[0])
            else:
                buffer_loss[row] = T.log(s[0]) - (T.cast(buffer_x[row, buffer_y[row]], 'float32') - m[0])

    @T.macro
    def merge_row(buffer_x, buffer_y, buffer_max, buffer_sum, buffer_loss, ignore, buffer_total, mean, row):
        if buffer_y[row] == ignore:
            for _ in T.Parallel(1):
                if mean != 0 and n == 1:
                    buffer_loss[row] = T.float32(float('nan'))
                else:
                    buffer_loss[row] = T.float32(0)
        else:
            mx = T.alloc_reducer((1,), 'float32', op='max')
            T.reducer_init(mx)
            for i in T.Parallel(width):
                if i < k:
                    T.reducer_update(mx[0], buffer_max[row, i])
            m = T.alloc_fragment((1,), 'float32')
            T.finalize_reducer(mx, m)
            sm = T.alloc_reducer((1,), 'float32', op='sum')
            T.reducer_init(sm)
            for i in T.Parallel(width):
                if i < k:
                    T.reducer_update(sm[0], buffer_sum[row, i] * T.exp(buffer_max[row, i] - m[0]))
            s = T.alloc_fragment((1,), 'float32')
            T.finalize_reducer(sm, s)
            store_loss(buffer_x, buffer_y, buffer_loss, buffer_total, row, mean, m, s)

    return merge_row


@cache
def _merge_kernel(dtype, width, cores):
    (n, v, k) = (T.dynamic(name, dtype='int64') for name in ('n', 'v', 'k'))
    stride = T.dynamic('stride', dtype='int64')
    merge_row = _make_merge_row(n, k, width)

    @T.prim_func
    def merge(
        buffer_x: T.StridedTensor((n, v), (stride, 1), dtype),
        buffer_y: T.Tensor((n,), 'int64'),
        buffer_max: T.Tensor((n, k), 'float32'),
        buffer_sum: T.Tensor((n, k), 'float32'),
        buffer_loss: T.Tensor((n,), 'float32'),
        ignore: T.int64,
        buffer_total: T.Tensor((1,), 'int64'),
        mean: T.int32,
    ):
        with T.Kernel(cores) as core, T.SimtVF(threads=32):
            for wave in T.serial(T.ceildiv(n, cores)):
                row = wave * cores + core
                if row < n:
                    merge_row(buffer_x, buffer_y, buffer_max, buffer_sum, buffer_loss, ignore, buffer_total, mean, row)

    return _compile(f'merge_{dtype}_{width}_{cores}', merge)


def _make_reduce_rows_block(n, capacity):
    @T.macro
    def reduce_block(buffer_loss, buffer_out, buffer_total, mean, block):
        sums = T.alloc_reducer((1,), 'float32', op='sum')
        T.reducer_init(sums)
        for i in T.Parallel(capacity):
            row = block * capacity + i
            if row < n:
                T.reducer_update(sums[0], buffer_loss[row])
        total = T.alloc_fragment((1,), 'float32')
        T.finalize_reducer(sums, total)
        for _ in T.Parallel(1):
            if mean != 0:
                buffer_out[block] = total[0] / T.cast(T.min(buffer_total[0], T.int64(1)), 'float32')
            else:
                buffer_out[block] = total[0]

    return reduce_block


@cache
def _reduce_kernel(cores, capacity=ROW_BLOCK):
    (n, blocks) = (T.dynamic('n', dtype='int64'), T.dynamic('blocks', dtype='int64'))
    reduce_block = _make_reduce_rows_block(n, capacity)

    @T.prim_func
    def reduce_rows(
        buffer_loss: T.Tensor((n,), 'float32'),
        buffer_out: T.Tensor((blocks,), 'float32'),
        buffer_total: T.Tensor((1,), 'int64'),
        mean: T.int32,
    ):
        with T.Kernel(cores) as core, T.SimtVF(threads=256):
            for wave in T.serial(T.ceildiv(blocks, cores)):
                block = wave * cores + core
                if block < blocks:
                    reduce_block(buffer_loss, buffer_out, buffer_total, mean, block)

    return _compile(f'reduce_rows_{cores}_{capacity}', reduce_rows)


def _make_sum_count_block(n, capacity, first=False, final=False, joint=False):
    if joint:

        @T.macro
        def update_count(buffer_counts, counts, row, ignore):
            T.reducer_update(counts[0], buffer_counts[row])

        @T.macro
        def store_result(buffer_out, buffer_outcounts, total, valid, block, original_n, final_step, scaled):
            buffer_outcounts[block] = valid[0]
            if final_step != 0:
                if scaled != 0:
                    buffer_out[block] = total[0] * (T.cast(original_n, 'float32') / T.cast(valid[0], 'float32'))
                else:
                    buffer_out[block] = total[0] / T.cast(valid[0], 'float32')
            else:
                buffer_out[block] = total[0]
    else:

        @T.macro
        def update_count(buffer_counts, counts, row, ignore):
            if first:
                T.reducer_update(counts[0], T.cast(buffer_counts[row] != ignore, 'int32'))
            else:
                T.reducer_update(counts[0], T.cast(buffer_counts[row], 'int32'))

        @T.macro
        def store_result(buffer_out, buffer_outcounts, total, valid, block, original_n, final_step, scaled):
            buffer_outcounts[block] = T.cast(valid[0], 'int64')
            if final:
                buffer_out[block] = total[0] / T.cast(valid[0], 'float32')
            else:
                buffer_out[block] = total[0]

    @T.macro
    def reduce_block(
        buffer_values, buffer_counts, buffer_out, buffer_outcounts, block, ignore, original_n, final_step, scaled
    ):
        sums = T.alloc_reducer((1,), 'float32', op='sum')
        counts = T.alloc_reducer((1,), 'int32', op='sum')
        T.reducer_init(sums)
        T.reducer_init(counts)
        for i in T.Parallel(capacity):
            row = block * capacity + i
            if row < n:
                T.reducer_update(sums[0], buffer_values[row])
                update_count(buffer_counts, counts, row, ignore)
        total = T.alloc_fragment((1,), 'float32')
        valid = T.alloc_fragment((1,), 'int32')
        T.finalize_reducer(sums, total)
        T.finalize_reducer(counts, valid)
        for _ in T.Parallel(1):
            store_result(buffer_out, buffer_outcounts, total, valid, block, original_n, final_step, scaled)

    return reduce_block


@cache
def _reduce_sum_count_kernel(cores, capacity=ROW_BLOCK, first=False, final=False):
    (n, blocks) = (T.dynamic('n', dtype='int64'), T.dynamic('blocks', dtype='int64'))
    reduce_block = _make_sum_count_block(n, capacity, first=first, final=final)

    @T.prim_func
    def reduce_sum_count(
        buffer_values: T.Tensor((n,), 'float32'),
        buffer_counts: T.Tensor((n,), 'int64'),
        buffer_out: T.Tensor((blocks,), 'float32'),
        buffer_outcounts: T.Tensor((blocks,), 'int64'),
        ignore: T.int64,
    ):
        with T.Kernel(cores) as core, T.SimtVF(threads=256):
            for wave in T.serial(T.ceildiv(blocks, cores)):
                block = wave * cores + core
                if block < blocks:
                    reduce_block(
                        buffer_values,
                        buffer_counts,
                        buffer_out,
                        buffer_outcounts,
                        block,
                        ignore,
                        T.int64(0),
                        T.int32(0),
                        T.int32(0),
                    )

    return _compile(f'reduce_sum_count_{cores}_{capacity}_{int(first)}_{int(final)}', reduce_sum_count)


@cache
def _joint_reduce_kernel(cores, capacity=ROW_BLOCK):
    (n, original_n, blocks) = (T.dynamic(name, dtype='int64') for name in ('n', 'original_n', 'blocks'))
    reduce_block = _make_sum_count_block(n, capacity, joint=True)

    @T.prim_func
    def joint_reduce(
        buffer_loss: T.Tensor((n,), 'float32'),
        buffer_counts: T.Tensor((n,), 'int32'),
        buffer_out: T.Tensor((blocks,), 'float32'),
        buffer_outcount: T.Tensor((blocks,), 'int32'),
        original_n: T.int64,
        final: T.int32,
        scaled: T.int32,
    ):
        with T.Kernel(cores) as core, T.SimtVF(threads=256):
            for wave in T.serial(T.ceildiv(blocks, cores)):
                block = wave * cores + core
                if block < blocks:
                    reduce_block(
                        buffer_loss,
                        buffer_counts,
                        buffer_out,
                        buffer_outcount,
                        block,
                        T.int64(0),
                        original_n,
                        final,
                        scaled,
                    )

    return _compile(f'joint_reduce_{cores}_{capacity}', joint_reduce)
