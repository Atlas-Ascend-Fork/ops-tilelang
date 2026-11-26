# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

import tilelang
import torch

from cann_ops_tilelang.config import get_num_vector_cores

from .cross_entropy_kernel import (
    FUSED_UB_BUDGET,
    ROW_BLOCK,
    SIMD_ROW_MAX_WAVES,
    SMALL_MAX_LANE_WORK,
    VOCAB_BYTES,
    _batched_small_kernel,
    _chunk_kernel,
    _counted_row_kernel,
    _fully_simd_fused_kernel,
    _joint_reduce_kernel,
    _merge_kernel,
    _partial_launch_kernel,
    _reduce_kernel,
    _reduce_sum_count_kernel,
    _row_kernel,
    _scaled_row_kernel,
    _simd_row_kernel,
    _valid_count_kernel,
    _valid_count_merge_kernel,
)


def _validate(logits, targets, ignore_index, reduction):
    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError('logits and targets must be tensors')
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError('expected logits [N, V] and targets [N]')
    if logits.shape[0] != targets.shape[0] or logits.shape[1] < 1:
        raise ValueError('matching N and V >= 1 are required')
    if logits.dtype not in (torch.float32, torch.bfloat16, torch.float16) or targets.dtype != torch.int64:
        raise TypeError('expected FP32/BF16/FP16 logits and int64 targets')
    if logits.device.type != 'npu' or targets.device != logits.device:
        raise ValueError('inputs must be on the same NPU')
    if logits.stride(1) != 1 or (logits.shape[0] > 1 and logits.stride(0) < logits.shape[1]):
        raise ValueError('logits require contiguous columns and nonoverlapping rows')
    if not targets.is_contiguous():
        raise ValueError('targets must be contiguous')
    if type(ignore_index) is not int:
        raise TypeError('ignore_index must be a Python int, not bool')
    if not -(2**63) <= ignore_index < 2**63:
        raise ValueError('ignore_index must fit int64')
    if reduction not in ('none', 'sum', 'mean'):
        raise ValueError('reduction must be none, sum, or mean')


def _active_cores(tasks, available):
    return min(available, tilelang.next_power_of_2(tasks))


@torch.no_grad()
def _cross_entropy_forward_impl(logits, targets, *, ignore_index=-100, reduction='none'):
    _validate(logits, targets, ignore_index, reduction)
    n, v = logits.shape
    if n == 0:
        return _empty_loss(logits, reduction)
    dtype = str(logits.dtype).removeprefix('torch.')
    element_size = logits.element_size()
    small_width = max(32, tilelang.next_power_of_2(v))
    block = min(VOCAB_BYTES // element_size, small_width)
    k = (v + block - 1) // block
    cores = get_num_vector_cores()
    waves = (n + cores - 1) // cores
    route = _whole_row_route(n, v, block, cores, reduction, waves)
    if route:
        if reduction == 'sum':
            return _partial_sum(logits, targets, ignore_index, dtype, block, cores)
        return _whole_row_mean(logits, targets, ignore_index, dtype, block, cores, route)
    row_work = waves * v
    small_rows = min(8, 2048 // small_width)
    lane_work = small_width // 32 * (1 + int(dtype != 'float32'))
    uses_small = lane_work <= SMALL_MAX_LANE_WORK and small_rows >= 4 and n >= 2 * cores
    if k > 1:
        fused, fused_block, slots = _fused_route(v, k, n, cores, row_work, element_size)
        if fused:
            return _launch_fused(logits, targets, ignore_index, reduction, dtype, n, cores, fused_block, slots)
    inline_count = post_mean = False
    if reduction == 'mean':
        inline_count, post_mean, row_cores, row_slots = _mean_plan(
            logits, dtype, block, k, cores, waves, uses_small, n, v, element_size
        )
        if row_slots:
            return _single_block_scaled_mean(logits, targets, ignore_index, dtype, block, row_cores, row_slots)
    loss = torch.empty((n,), device=logits.device, dtype=torch.float32)
    total_count = torch.empty((1,), device=logits.device, dtype=torch.int64)
    if reduction == 'mean' and not (inline_count or post_mean):
        _prepare_count(logits, targets, total_count, ignore_index, n, cores)
    kernel_mean = int(reduction == 'mean' and not post_mean)
    if uses_small:
        _batched_small_kernel(dtype, small_width, small_rows, _active_cores((n + small_rows - 1) // small_rows, cores))(
            logits, targets, loss, total_count, ignore_index, kernel_mean
        )
    elif n >= cores or k == 1 or row_work <= (n * k + cores - 1) // cores * block + block:
        _launch_rows(logits, targets, ignore_index, kernel_mean, dtype, block, cores, inline_count, loss, total_count)
    else:
        _launch_chunks(logits, targets, ignore_index, kernel_mean, dtype, block, cores, loss, total_count)
    if reduction == 'none':
        return loss
    if n == 1:
        return loss.reshape(())
    return _finish_loss(logits, loss, targets, total_count, ignore_index, reduction, n, cores, post_mean)


def _empty_loss(logits, reduction):
    if reduction == 'none':
        return torch.empty((0,), device=logits.device, dtype=torch.float32)
    return torch.full((), float('nan') if reduction == 'mean' else 0.0, device=logits.device, dtype=torch.float32)


def _inline_mean(n, v, k, cores, waves):
    if k != 1 or not 1 < n <= 32:
        return False
    return n < 2 * cores and waves <= SIMD_ROW_MAX_WAVES and v <= 2147483647


def _post_mean(logits, count_tiles, cores, n, v):
    if not 8 < count_tiles <= cores:
        return False
    return n * (2 * torch.finfo(logits.dtype).max + v + 1) < torch.finfo(torch.float32).max / 4


def _count_cores(count_tiles, cores):
    return _active_cores(count_tiles, cores) if count_tiles > 8 else 1


def _mean_plan(logits, dtype, block, k, cores, waves, uses_small, n, v, element_size):
    inline_count = _inline_mean(n, v, k, cores, waves)
    if inline_count:
        return inline_count, False, 0, 0
    if k != 1 or n > 2**24 or dtype == 'bfloat16':
        return inline_count, False, 0, 0
    count_tiles = (n + ROW_BLOCK - 1) // ROW_BLOCK
    if dtype == 'float16':
        return inline_count, _post_mean(logits, count_tiles, cores, n, v), 0, 0
    if uses_small or waves <= SIMD_ROW_MAX_WAVES:
        return inline_count, False, 0, 0
    if _count_cores(count_tiles, cores) == 1:
        return inline_count, False, 0, 0
    row_cores = _active_cores(n, cores)
    row_slots = tilelang.next_power_of_2((n + row_cores - 1) // row_cores)
    row_ub_bytes = 2 * (block + 128) * element_size + 2 * max(8, row_slots) * 4
    if row_slots > 256 or row_ub_bytes > FUSED_UB_BUDGET:
        return inline_count, False, 0, 0
    return inline_count, False, row_cores, row_slots


def _launch_fused(logits, targets, ignore_index, reduction, dtype, n, cores, fused_block, slots):
    workspace = torch.empty((cores, slots * 8), device=logits.device, dtype=torch.float32)
    output = torch.empty((n,) if reduction == 'none' else (1,), device=logits.device, dtype=torch.float32)
    _fully_simd_fused_kernel(dtype, fused_block, cores, slots)(
        logits,
        targets.view(torch.int32),
        workspace,
        output,
        ignore_index,
        {'none': 0, 'sum': 1, 'mean': 2}[reduction],
    )
    return output if reduction == 'none' else output.reshape(())


def _launch_rows(logits, targets, ignore_index, kernel_mean, dtype, block, cores, inline_count, loss, total_count):
    n, v = logits.shape
    single_block = v <= block
    if n <= 2147483647 and v <= 2147483647 and ((n + cores - 1) // cores <= SIMD_ROW_MAX_WAVES):
        _simd_row_kernel(dtype, block, _active_cores(n, cores), inline_count)(
            logits,
            targets.view(torch.int32),
            loss,
            total_count.view(torch.int32),
            ignore_index,
            kernel_mean,
        )
    else:
        row_cores = _active_cores(n, cores)
        row_slots = tilelang.next_power_of_2((n + row_cores - 1) // row_cores)
        row_ub_bytes = 2 * (block + 128) * logits.element_size() + 2 * max(8, row_slots) * 4
        batch_slots = row_slots if single_block and row_slots <= 256 and (row_ub_bytes <= FUSED_UB_BUDGET) else 0
        _row_kernel(
            dtype, block, row_cores, single_block, batch_slots, static_v=v if single_block and batch_slots else None
        )(logits, targets, loss, total_count, ignore_index, kernel_mean)


def _launch_chunks(logits, targets, ignore_index, kernel_mean, dtype, block, cores, loss, total_count):
    n, v = logits.shape
    k = (v + block - 1) // block
    maxima = torch.empty((n, k), device=logits.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    _chunk_kernel(dtype, block, _active_cores(n * k, cores))(logits, maxima, sums)
    _merge_kernel(dtype, max(32, tilelang.next_power_of_2(k)), _active_cores(n, cores))(
        logits, targets, maxima, sums, loss, ignore_index, total_count, kernel_mean
    )


def _whole_row_route(n, v, block, cores, reduction, waves):
    if reduction == 'none' or n > 2**24:
        return 0
    if n <= SIMD_ROW_MAX_WAVES * cores or v <= block:
        return 0
    if reduction == 'sum':
        return 1
    return 2 if v > 2 * block and waves >= ROW_BLOCK else 1


def _whole_row_mean(logits, targets, ignore_index, dtype, block, cores, route):
    n = logits.shape[0]
    active = _active_cores(n, cores)
    if route == 2:
        values = torch.empty((active,), device=logits.device, dtype=torch.float32)
        counts = torch.empty((active,), device=logits.device, dtype=torch.int32)
        output = torch.empty((1,), device=logits.device, dtype=torch.float32)
        _partial_launch_kernel(dtype, block, active, True)(logits, targets, values, counts, output, ignore_index)
        return output.reshape(())
    values = torch.empty((n,), device=logits.device, dtype=torch.float32)
    counts = torch.empty((n,), device=logits.device, dtype=torch.int32)
    _counted_row_kernel(dtype, block, active, True)(logits, targets, values, counts, ignore_index)
    remaining = n
    while True:
        capacity = 2 * ROW_BLOCK if ROW_BLOCK < remaining <= 2 * ROW_BLOCK else ROW_BLOCK
        blocks = (remaining + capacity - 1) // capacity
        output = torch.empty((blocks,), device=logits.device, dtype=torch.float32)
        output_count = torch.empty((blocks,), device=logits.device, dtype=torch.int32)
        _joint_reduce_kernel(_active_cores(blocks, cores), capacity)(
            values, counts, output, output_count, n, int(blocks == 1), 1
        )
        if blocks == 1:
            return output.reshape(())
        (values, counts, remaining) = (output, output_count, blocks)


def _single_block_scaled_mean(logits, targets, ignore_index, dtype, block, cores, slots):
    n = logits.shape[0]
    values = torch.empty((n,), device=logits.device, dtype=torch.float32)
    counts = torch.empty((n,), device=logits.device, dtype=torch.int32)
    _scaled_row_kernel(dtype, block, cores, True, slots, static_v=logits.shape[1])(
        logits, targets, values, counts, ignore_index, 1
    )
    remaining = n
    while True:
        blocks = (remaining + ROW_BLOCK - 1) // ROW_BLOCK
        output = torch.empty((blocks,), device=logits.device, dtype=torch.float32)
        output_count = torch.empty((blocks,), device=logits.device, dtype=torch.int32)
        _joint_reduce_kernel(_active_cores(blocks, cores), ROW_BLOCK)(
            values, counts, output, output_count, n, int(blocks == 1), 1
        )
        if blocks == 1:
            return output.reshape(())
        (values, counts, remaining) = (output, output_count, blocks)


def _prepare_count(logits, targets, total_count, ignore_index, n, cores):
    if n <= 1:
        return
    count_tiles = (n + ROW_BLOCK - 1) // ROW_BLOCK
    count_cores = _count_cores(count_tiles, cores)
    if count_cores == 1:
        _valid_count_kernel(n > 2147483647)(targets, total_count, ignore_index)
    else:
        partial_count = torch.empty((count_cores,), device=logits.device, dtype=torch.int64)
        _valid_count_kernel(n > 2147483647, count_cores)(targets, partial_count, ignore_index)
        _valid_count_merge_kernel()(partial_count, total_count)


def _finish_loss(logits, loss, targets, total_count, ignore_index, reduction, n, cores, post_mean):
    if post_mean:
        counts = targets
        first = True
        while True:
            capacity = 2 * ROW_BLOCK if ROW_BLOCK < n <= 2 * ROW_BLOCK else ROW_BLOCK
            blocks = (n + capacity - 1) // capacity
            output = torch.empty((blocks,), device=logits.device, dtype=torch.float32)
            output_counts = torch.empty((blocks,), device=logits.device, dtype=torch.int64)
            _reduce_sum_count_kernel(_active_cores(blocks, cores), capacity, first, blocks == 1)(
                loss, counts, output, output_counts, ignore_index
            )
            if blocks == 1:
                return output.reshape(())
            (loss, counts, n, first) = (output, output_counts, blocks, False)
    while True:
        capacity = 2 * ROW_BLOCK if ROW_BLOCK < n <= 2 * ROW_BLOCK else ROW_BLOCK
        blocks = (n + capacity - 1) // capacity
        output = torch.empty((blocks,), device=logits.device, dtype=torch.float32)
        _reduce_kernel(_active_cores(blocks, cores), capacity)(
            loss, output, total_count, int(reduction == 'mean' and blocks == 1)
        )
        if blocks == 1:
            return output.reshape(())
        (loss, n) = (output, blocks)


def _partial_sum(logits, targets, ignore_index, dtype, block, route_cores):
    n = logits.shape[0]
    active = _active_cores(n, route_cores)
    values = torch.empty((active,), device=logits.device, dtype=torch.float32)
    counts = torch.empty((active,), device=logits.device, dtype=torch.int32)
    output = torch.empty((1,), device=logits.device, dtype=torch.float32)
    _partial_launch_kernel(dtype, block, active, False)(logits, targets, values, counts, output, ignore_index)
    return output.reshape(())


def _fused_route(v, k, n, cores, row_work, element_size):
    fused_block = ((v + k - 1) // k + 127) // 128 * 128
    fused_k = (v + fused_block - 1) // fused_block
    slots = (n * fused_k + cores - 1) // cores
    fused = (
        cores <= n <= 2 * cores
        and n % cores != 0
        and (2 <= fused_k <= 32)
        and (slots <= 16)
        and (slots * fused_block * 8 <= row_work * 7)
    )
    fused_ub = (
        2 * (fused_block + 128) * element_size
        + (slots * 8 + 4 * cores + 64 + cores * slots * 8 + (2 * cores + 63) // 64 * 64) * 4
    )
    fused = fused and fused_ub <= FUSED_UB_BUDGET
    return fused, fused_block, slots


@torch.library.custom_op('cann_ops_tilelang::cross_entropy_forward', mutates_args=())
def cross_entropy_forward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = 'none',
) -> torch.Tensor:
    return _cross_entropy_forward_impl(logits, targets, ignore_index=ignore_index, reduction=reduction)


@cross_entropy_forward.register_fake
def _cross_entropy_forward_fake(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = 'none',
) -> torch.Tensor:
    _validate(logits, targets, ignore_index=ignore_index, reduction=reduction)
    return logits.new_empty(
        (logits.shape[0],) if reduction == 'none' else (),
        dtype=torch.float32,
    )
