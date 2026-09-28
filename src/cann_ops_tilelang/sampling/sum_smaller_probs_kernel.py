from typing import Any

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

_UB_SIZE = 240000


def _align32(value: int) -> int:
    return (value + 31) // 32 * 32


def _sum_smaller_probs_ub_bytes(vocab_size: int, pipeline_stages: int, max_rows_per_core: int) -> int:
    vector_lanes = 64
    max_block_size = 24576 if pipeline_stages == 1 else 16384
    block_size = min(
        max_block_size,
        max(vector_lanes, (vocab_size + vector_lanes - 1) // vector_lanes * vector_lanes),
    )
    n_chunks = (vocab_size + block_size - 1) // block_size
    n_chunk_sum_vectors = (n_chunks + vector_lanes - 1) // vector_lanes
    has_tail = vocab_size % block_size != 0
    full_chunks = n_chunks - int(has_tail)
    tail_size = vocab_size - full_chunks * block_size if has_tail else 0
    tail_vectors = (tail_size + vector_lanes - 1) // vector_lanes
    block_vectors = block_size // vector_lanes
    tail_buffer_size = max(1, tail_vectors * vector_lanes) if has_tail else 1
    max_vector_sums = max(block_vectors, tail_vectors)
    vector_sum_slots = (max_vector_sums + vector_lanes - 1) // vector_lanes * vector_lanes
    wave_rows = min(max_rows_per_core, 32)
    row_stride = n_chunk_sum_vectors * vector_lanes
    return (
        pipeline_stages * _align32(4 * block_size)
        + _align32(4 * vector_sum_slots)
        + _align32(4 * wave_rows * row_stride)
        + _align32(4 * wave_rows)
        + _align32(32 * wave_rows)
        + _align32(4 * max_rows_per_core)
        + (_align32(4 * tail_buffer_size) if has_tail else 0)
    )


@tilelang.jit(out_idx=None, target='ascend')
def _sum_smaller_probs_tl_ascend(
    vocab_size: int,
    num_cores: int,
    pipeline_stages: int,
    max_rows_per_core: int,
) -> Any:
    assert _sum_smaller_probs_ub_bytes(vocab_size, pipeline_stages, max_rows_per_core) <= _UB_SIZE
    num_tokens = T.dynamic('num_tokens')
    probs_stride = T.dynamic('probs_stride')

    vector_lanes = 64
    # max_block_size is deduced from pipeline_stages with an approximate 256KB UB budget.
    # The tail block uses an independent single-version tail_probs_ub. It must coexist with
    # probs_ub in UB because it is outside the pipeline.
    # pipeline_stages=1: max_block_size 24576, probs_ub ≈ 98KB + tail_probs_ub ≈ 98KB ≈ 200KB
    # pipeline_stages=2: max_block_size 16384, probs_ub ≈ 131KB + tail_probs_ub ≈ 65KB ≈ 200KB
    max_block_size = 24576 if pipeline_stages == 1 else 16384
    block_size = min(
        max_block_size,
        max(vector_lanes, (vocab_size + vector_lanes - 1) // vector_lanes * vector_lanes),
    )
    n_chunks = (vocab_size + block_size - 1) // block_size
    n_chunk_sum_vectors = (n_chunks + vector_lanes - 1) // vector_lanes
    has_tail = vocab_size % block_size != 0
    full_chunks = n_chunks - int(has_tail)
    tail_size = vocab_size - full_chunks * block_size if has_tail else 0
    tail_full_vectors = tail_size // vector_lanes
    tail_remainder = tail_size % vector_lanes
    block_vectors = block_size // vector_lanes
    tail_vectors = tail_full_vectors + int(tail_remainder > 0)
    tail_buffer_size = max(1, tail_vectors * vector_lanes) if has_tail else 1
    max_vector_sums = max(block_vectors, tail_vectors)
    vector_sum_slots = (max_vector_sums + vector_lanes - 1) // vector_lanes * vector_lanes

    @T.macro
    def finish_chunk(vector_sums_ub, chunk_sums_ub, chunk, num_vector_sums):
        with T.SimdVF():
            full = S.pset(32, 'PAT_ALL')
            zero = S.vdup(0.0, T.float32, full)
            total = S.alloc_var(T.float32)
            total = zero
            for group in T.serial((num_vector_sums + vector_lanes - 1) // vector_lanes):
                indices = S.vci(T.int32(group * vector_lanes), T.int32)
                valid = S.vcmps(indices, num_vector_sums, full, 'lt')
                values = S.vsel(
                    S.vld(vector_sums_ub[group * vector_lanes]),
                    zero,
                    valid,
                )
                total = S.vadd(total, values, full)
            S.vsts(
                chunk_sums_ub[chunk],
                S.vcadd(total, full),
                full,
                'ONEPT_B32',
            )

    @T.macro
    def reduce_compare_chunk(probs_ub, vector_sums_ub, probs_limit, op):
        with T.SimdVF():
            full = S.pset(32, 'PAT_ALL')
            zero = S.vdup(0.0, T.float32, full)
            for vector in T.serial(block_vectors):
                values = S.vld(probs_ub[vector * vector_lanes])
                selected = S.vsel(values, zero, S.vcmps(values, probs_limit, full, op))
                S.vsts(
                    vector_sums_ub[vector],
                    S.vcadd(selected, full),
                    full,
                    'ONEPT_B32',
                )

    @T.macro
    def reduce_compare_tail(probs_ub, vector_sums_ub, probs_limit, op):
        with T.SimdVF():
            full = S.pset(32, 'PAT_ALL')
            zero = S.vdup(0.0, T.float32, full)
            for vector in T.serial(tail_full_vectors):
                values = S.vld(probs_ub[vector * vector_lanes])
                selected = S.vsel(values, zero, S.vcmps(values, probs_limit, full, op))
                S.vsts(
                    vector_sums_ub[vector],
                    S.vcadd(selected, full),
                    full,
                    'ONEPT_B32',
                )
            # Handle tail remainder with a validity mask.
            if tail_remainder > 0:
                offset = tail_full_vectors * vector_lanes
                tail_indices = S.vci(T.int32(offset), T.int32)
                tail_values = S.vld(probs_ub[offset])
                cond = S.vcmps(tail_values, probs_limit, full, op)
                selected = S.vsel(tail_values, zero, cond)
                selected = S.vsel(
                    selected,
                    zero,
                    S.vcmps(tail_indices, tail_size, full, 'lt'),
                )
                S.vsts(
                    vector_sums_ub[tail_full_vectors],
                    S.vcadd(selected, full),
                    full,
                    'ONEPT_B32',
                )

    @T.macro
    def reduce_tie(probs_ub, vector_sums_ub, base, probs_limit, sample_result, is_tail):
        with T.SimdVF():
            full = S.pset(32, 'PAT_ALL')
            zero = S.vdup(0.0, T.float32, full)
            num_vectors = tail_vectors if is_tail else block_vectors
            for vector in T.serial(num_vectors):
                col = base + vector * vector_lanes
                indices = S.vci(T.int32(col), T.int32)
                values = S.vld(probs_ub[vector * vector_lanes])
                less = S.vcmps(values, probs_limit, full, 'lt')
                equal = S.vcmps(values, probs_limit, full, 'eq')
                equal_before = S.vsel(values, zero, equal)
                equal_before = S.vsel(
                    equal_before,
                    zero,
                    S.vcmps(indices, sample_result, full, 'le'),
                )
                selected = S.vsel(values, equal_before, less)
                if is_tail:
                    # Ensure that vocab_size is not exceeded
                    valid = S.vcmps(indices, vocab_size, full, 'lt')
                    selected = S.vsel(selected, zero, valid)
                S.vsts(
                    vector_sums_ub[vector],
                    S.vcadd(selected, full),
                    full,
                    'ONEPT_B32',
                )

    @T.prim_func
    def _sum_smaller_probs(
        probs: T.StridedTensor((num_tokens, vocab_size), (probs_stride, 1), T.float32),  # type: ignore
        sample_results: T.Tensor((num_tokens,), T.int32),  # type: ignore
        output: T.Tensor((num_tokens,), T.float32),  # type: ignore
    ) -> None:
        with T.Kernel(num_cores) as core_id:
            # Static row allocation, with each core processing consecutive rows
            row_start = core_id * max_rows_per_core
            row_end = T.min(row_start + max_rows_per_core, num_tokens)
            row_count = T.max(row_end - row_start, 0)
            # Limit rows per wave to bound chunk_sums_ub/pivot_win_ub UB usage.
            max_wave_rows = 32
            wave_rows = min(max_rows_per_core, max_wave_rows)
            row_stride = n_chunk_sum_vectors * vector_lanes

            probs_ub = T.alloc_shared((block_size,), T.float32)
            # Independent vector sums avoid a long loop-carried register dependency.
            vector_sums_ub = T.alloc_shared((vector_sum_slots,), T.float32)
            # Each chunk is reduced to one scalar; wave_rows rows' chunks are
            # accumulated here. Round up for the final SIMD load.
            chunk_sums_ub = T.alloc_shared((wave_rows * row_stride,), T.float32)
            sr_ub = T.alloc_shared((wave_rows,), T.int32)
            pivot_win_ub = T.alloc_shared((wave_rows * 8,), T.float32)
            output_accum_ub = T.alloc_shared((max_rows_per_core,), T.float32)
            tail_probs_ub = T.alloc_shared((tail_buffer_size,), T.float32) if has_tail else None
            T.annotate_buffer_versions(
                {
                    probs_ub: pipeline_stages,
                    vector_sums_ub: 1,
                    chunk_sums_ub: 1,
                    pivot_win_ub: 1,
                }
            )

            if row_count > 0:
                num_waves = (row_count + wave_rows - 1) // wave_rows
                for w in T.serial(num_waves):
                    w_start = row_start + w * wave_rows
                    w_rows = T.min(wave_rows, row_count - w * wave_rows)

                    T.copy(sample_results[w_start : w_start + w_rows], sr_ub[:w_rows])

                    if full_chunks > 0:
                        for it in T.Pipelined(
                            w_rows * full_chunks,
                            num_stages=pipeline_stages,
                            annotations={'multi_buffer_eligible': [probs_ub]},
                        ):
                            local_row = it // full_chunks
                            chunk = it % full_chunks
                            row = w_start + local_row
                            base = chunk * block_size
                            T.copy(
                                probs[row, base : base + block_size],
                                probs_ub,
                                l2_cache_ctrl='NOTALLOC_KEEP',
                            )
                            sr = T.alloc_var(T.int32, sr_ub[local_row])
                            if chunk == 0:
                                if sr < 0:
                                    pivot_win_ub[local_row * 8] = -T.infinity(T.float32)
                                else:
                                    T.device_assert(
                                        sr < vocab_size, msg='sample result must be smaller than vocab_size'
                                    )
                                    pivot_win_ub[local_row * 8] = probs[row, sr]
                            probs_limit = T.alloc_var(T.float32, pivot_win_ub[local_row * 8])
                            # Only the chunk containing the sampled index needs the
                            # equality and index tie-break. Whole chunks on either
                            # side collapse to <= pivot or < pivot.
                            if sr >= base + block_size:
                                reduce_compare_chunk(probs_ub, vector_sums_ub, probs_limit, 'le')
                            elif sr < base:
                                reduce_compare_chunk(probs_ub, vector_sums_ub, probs_limit, 'lt')
                            else:
                                reduce_tie(probs_ub, vector_sums_ub, base, probs_limit, sr, False)
                            finish_chunk(
                                vector_sums_ub,
                                chunk_sums_ub,
                                local_row * row_stride + chunk,
                                block_vectors,
                            )

                    # The only partial DMA and validity mask are isolated here.
                    if has_tail:
                        for r in T.serial(w_rows):
                            row = w_start + r
                            tail_base = full_chunks * block_size
                            T.copy(
                                probs[row, tail_base : tail_base + tail_size],
                                tail_probs_ub[:tail_size],
                                l2_cache_ctrl='NOTALLOC_KEEP',
                            )
                            sr = T.alloc_var(T.int32, sr_ub[r])
                            if full_chunks == 0:
                                # Load the pivot here when there are no full blocks
                                if sr < 0:
                                    pivot_win_ub[r * 8] = -T.infinity(T.float32)
                                else:
                                    T.device_assert(
                                        sr < vocab_size, msg='sample result must be smaller than vocab_size'
                                    )
                                    pivot_win_ub[r * 8] = probs[row, sr]
                            probs_limit = T.alloc_var(T.float32, pivot_win_ub[r * 8])
                            if sr >= tail_base + tail_size:
                                reduce_compare_tail(tail_probs_ub, vector_sums_ub, probs_limit, 'le')
                            elif sr < tail_base:
                                reduce_compare_tail(tail_probs_ub, vector_sums_ub, probs_limit, 'lt')
                            else:
                                reduce_tie(tail_probs_ub, vector_sums_ub, tail_base, probs_limit, sr, True)
                            finish_chunk(
                                vector_sums_ub,
                                chunk_sums_ub,
                                r * row_stride + full_chunks,
                                tail_vectors,
                            )

                    for r in T.serial(w_rows):
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            zero = S.vdup(0.0, T.float32, full)
                            total = S.alloc_var(T.float32)
                            total = zero
                            for vector in T.serial(n_chunk_sum_vectors):
                                chunk_indices = S.vci(T.int32(vector * vector_lanes), T.int32)
                                valid = S.vcmps(chunk_indices, n_chunks, full, 'lt')
                                total = S.vadd(
                                    total,
                                    S.vsel(
                                        S.vld(chunk_sums_ub[r * row_stride + vector * vector_lanes]),
                                        zero,
                                        valid,
                                    ),
                                    full,
                                )
                            S.vsts(
                                output_accum_ub[w_start + r - row_start],
                                S.vcadd(total, full),
                                full,
                                'ONEPT_B32',
                            )

            # Batch DMA to GM
            if row_count > 0:
                T.copy(
                    output_accum_ub[:row_count],
                    output[row_start:row_end],
                    l2_cache_ctrl='NOTALLOC_PW',
                )

    return _sum_smaller_probs
