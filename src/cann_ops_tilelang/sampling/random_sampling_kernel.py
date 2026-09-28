from typing import Any

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(
    out_idx=None,
    target='ascend',
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
    },
)
def _random_sampling_tl_ascend(
    vocab_size: int,
    is_nextn_predict: bool,
) -> Any:
    """Sample one token per row using Ascend vector reductions."""
    num_tokens = T.dynamic('num_tokens')
    probs_stride = T.dynamic('probs_stride')

    vl = 64
    # Chunk adaptive: align to vl for small vocab to avoid zero-padding waste
    chunk = 16384 if vocab_size > 4096 else (vocab_size + vl - 1) // vl * vl
    vectors_per_chunk = chunk // vl
    num_chunks = (vocab_size + chunk - 1) // chunk
    num_vectors = (vocab_size + vl - 1) // vl
    stored_num_vectors = num_chunks * vectors_per_chunk
    padded_stored_vectors = ((stored_num_vectors + vl - 1) // vl) * vl
    # Two-level search: chunk_sums aggregates every 64 group_sums into one value
    num_group_chunks = (num_vectors + vl - 1) // vl
    fine_seg_size = vl // 8  # 8 elements per fine search segment
    sample_case = 1 if is_nextn_predict else 0
    inv_2pow53 = 1.1102230246251565e-16
    num_cores = get_num_vector_cores()

    total_slot = 0
    target_slot = 1
    group_prefix_slot = 2
    selected_group_slot = 0
    sample_slot = 1

    @T.macro
    def init_input_db() -> None:
        if num_chunks > 1:
            T.ascend_set_flag('V_MTE2', 0)
            T.ascend_set_flag('V_MTE2', 1)
            if num_chunks > 2:
                T.ascend_set_flag('V_MTE2', 2)

    @T.macro
    def drain_input_db() -> None:
        if num_chunks > 1:
            T.ascend_wait_flag('V_MTE2', 0)
            T.ascend_wait_flag('V_MTE2', 1)
            if num_chunks > 2:
                T.ascend_wait_flag('V_MTE2', 2)

    @T.macro
    def sync_v_to_s() -> None:
        T.ascend_set_flag('V_S', 0)
        T.ascend_wait_flag('V_S', 0)

    # VEC processing extracted into macro to avoid code duplication across
    # prologue / main loop / epilogue (3 sites).
    @T.macro
    def process_chunk(
        buf: T.Tensor,  # type: ignore
        gsum: T.Tensor,  # type: ignore
        slot: T.int32,
        chunk_idx: T.int32,
        base_val: T.int32,
    ) -> None:
        with T.SimdVF():
            full = S.pset(32, 'PAT_ALL')
            zero_vector = S.vdup(0.0, T.float32, full)
            for vector in T.serial(vectors_per_chunk):
                col = base_val + vector * vl
                indices = S.vci(T.int32(col), T.int32)
                valid = S.vcmps(indices, vocab_size, full, 'lt')
                values = S.vsel(
                    S.vld(buf[slot, vector * vl]),
                    zero_vector,
                    valid,
                )
                S.vsts(
                    gsum[chunk_idx * vectors_per_chunk + vector],
                    S.vcadd(values, full),
                    full,
                    'ONEPT_B32',
                )

    @T.macro
    def random_uniform53(seed: T.uint64, sequence: T.int32) -> T.float32:  # type: ignore
        T.rng_init(seed, sequence)
        counter0 = T.alloc_var(T.uint64, T.cast(T.rng_rand(), T.uint64))
        counter1 = T.alloc_var(T.uint64, T.cast(T.rng_rand(), T.uint64))
        random53 = T.alloc_var(T.uint64, counter0 ^ (counter1 << 21))
        return T.cast(random53, T.float32) * T.float32(inv_2pow53) + T.float32(inv_2pow53 / 2.0)

    @T.prim_func
    def _random_sampling(
        probs: T.StridedTensor((num_tokens, vocab_size), (probs_stride, 1), T.float32),  # type: ignore
        seq_seeds: T.Tensor((num_tokens,), T.uint64),  # type: ignore
        sample_positions: T.Tensor((num_tokens,), T.int32),  # type: ignore
        out: T.Tensor((num_tokens,), T.int32),  # type: ignore
    ) -> None:
        with T.Kernel(T.min(num_tokens, num_cores)) as core_id:
            # 3-slot pingpong: allows MTE2 to prefetch 2 chunks ahead of VEC
            probs_ub = T.alloc_shared((3, chunk), T.float32)
            group_sum_ub = T.alloc_shared((padded_stored_vectors,), T.float32)
            scalar_f_ub = T.alloc_shared((vl,), T.float32)
            scalar_i_ub = T.alloc_shared((vl,), T.int32)
            # Per-64-group block sums for two-level search
            chunk_sums_ub = T.alloc_shared((num_group_chunks,), T.float32)

            rows_per_core = T.ceildiv(num_tokens, num_cores)
            row_begin = core_id * rows_per_core
            row_end = T.min(row_begin + rows_per_core, num_tokens)
            for row in T.Pipelined(row_begin, row_end, num_stages=2):
                init_input_db()
                prologue_valid = T.min(chunk, vocab_size)
                # MTE2 chunk0 issued first so its latency overlaps the RNG +
                # group_sum_ub zero-init below (different UB regions).
                if num_chunks > 1:
                    T.ascend_wait_flag('V_MTE2', 0)
                T.copy(
                    probs[row, 0:prologue_valid],
                    probs_ub[0, :prologue_valid],
                )
                T.ascend_set_flag('MTE2_V', 0)
                # RNG (scalar, overlaps MTE2 load)
                sequence = T.alloc_var(T.int32, sample_positions[row] * 3 + sample_case)
                with T.SimtVF(threads=1):
                    scalar_f_ub[target_slot] = random_uniform53(seq_seeds[row], sequence)
                # VEC zero-init (overlaps MTE2 load)
                with T.SimdVF():
                    zero = S.vdup(0.0, T.float32)
                    for vector in T.serial(padded_stored_vectors // vl):
                        S.vsts(group_sum_ub[vector * vl], zero)
                if num_chunks > 1:
                    prologue2_valid = T.min(chunk, vocab_size - chunk)
                    T.ascend_wait_flag('V_MTE2', 1)
                    T.copy(
                        probs[row, chunk : chunk + prologue2_valid],
                        probs_ub[1, :prologue2_valid],
                    )
                    T.ascend_set_flag('MTE2_V', 1)
                for chunk_id in T.serial(max(num_chunks - 2, 0)):
                    slot = chunk_id % 3
                    prefetch_slot = (chunk_id + 2) % 3
                    base = chunk_id * chunk
                    prefetch_base = (chunk_id + 2) * chunk
                    prefetch_valid = T.min(chunk, vocab_size - prefetch_base)
                    T.ascend_wait_flag('MTE2_V', slot)
                    process_chunk(probs_ub, group_sum_ub, slot, chunk_id, base)
                    T.ascend_set_flag('V_MTE2', slot)
                    T.ascend_wait_flag('V_MTE2', prefetch_slot)
                    T.copy(
                        probs[row, prefetch_base : prefetch_base + prefetch_valid],
                        probs_ub[prefetch_slot, :prefetch_valid],
                    )
                    T.ascend_set_flag('MTE2_V', prefetch_slot)
                if num_chunks >= 2:
                    epi1_slot = (num_chunks - 2) % 3
                    epi1_base = (num_chunks - 2) * chunk
                    T.ascend_wait_flag('MTE2_V', epi1_slot)
                    process_chunk(probs_ub, group_sum_ub, epi1_slot, num_chunks - 2, epi1_base)
                    T.ascend_set_flag('V_MTE2', epi1_slot)
                last_slot = (num_chunks - 1) % 3
                last_base = (num_chunks - 1) * chunk
                T.ascend_wait_flag('MTE2_V', last_slot)
                process_chunk(probs_ub, group_sum_ub, last_slot, num_chunks - 1, last_base)
                if num_chunks > 1:
                    T.ascend_set_flag('V_MTE2', last_slot)
                drain_input_db()

                # Two-level VEC reduction: total sum + per-64-group block sums
                with T.SimdVF():
                    full = S.pset(32, 'PAT_ALL')
                    total = S.alloc_var(T.float32)
                    total = S.vdup(0.0, T.float32, full)
                    for gc in T.serial(num_group_chunks):
                        vec = S.vld(group_sum_ub[gc * vl])
                        total = S.vadd(total, vec, full)
                        # vcadd reduces 64 lanes to 1 scalar (block sum)
                        S.vsts(
                            chunk_sums_ub[gc],
                            S.vcadd(vec, full),
                            full,
                            'ONEPT_B32',
                        )
                    S.vsts(
                        scalar_f_ub[total_slot],
                        S.vcadd(total, full),
                        full,
                        'ONEPT_B32',
                    )
                sync_v_to_s()

                # target = random * total (SCALAR, no extra sync needed)
                scalar_f_ub[target_slot] = scalar_f_ub[target_slot] * scalar_f_ub[total_slot]

                # ── Level-1 search: find target chunk (pure SCALAR) ──
                prefix = T.alloc_var(T.float32, 0.0)
                found = T.alloc_var(T.bool, False)
                scalar_i_ub[selected_group_slot] = num_vectors - 1
                scalar_i_ub[sample_slot] = vocab_size - 1
                scalar_f_ub[group_prefix_slot] = 0.0
                target_gc = T.alloc_var(T.int32, 0)
                chunk_prefix = T.alloc_var(T.float32, 0.0)
                gc = T.alloc_var(T.int32, 0)
                while gc < num_group_chunks and not found:
                    chunk_total = T.alloc_var(T.float32, prefix + chunk_sums_ub[gc])
                    if scalar_f_ub[target_slot] > prefix and scalar_f_ub[target_slot] <= chunk_total:
                        found = True
                        target_gc = gc
                        chunk_prefix = prefix
                    prefix = chunk_total
                    gc += 1
                if not found:
                    scalar_f_ub[group_prefix_slot] = prefix - group_sum_ub[num_vectors - 1]

                fine_start = T.alloc_var(T.int32, target_gc * vl)
                fine_end = T.alloc_var(T.int32, T.min(vl, num_vectors - fine_start))

                # ── Level-2 search: find target group within chunk ──
                if num_chunks > 1:
                    with T.SimdVF():
                        full2 = S.pset(32, 'PAT_ALL')
                        zero2 = S.vdup(0.0, T.float32, full2)
                        idx2 = S.vci(T.int32(0), T.int32)
                        gv = S.vld(group_sum_ub[fine_start])
                        m_gcnt = S.vcmps(idx2, fine_end, full2, 'lt')
                        gv_cnt = S.vsel(gv, zero2, m_gcnt)
                        S.vsts(
                            scalar_f_ub[8],
                            S.vcadd(S.vsel(gv_cnt, zero2, S.vcmps(idx2, fine_seg_size, full2, 'lt')), full2),
                            full2,
                            'ONEPT_B32',
                        )
                        S.vsts(
                            scalar_f_ub[9],
                            S.vcadd(S.vsel(gv_cnt, zero2, S.vcmps(idx2, fine_seg_size * 2, full2, 'lt')), full2),
                            full2,
                            'ONEPT_B32',
                        )
                        S.vsts(
                            scalar_f_ub[10],
                            S.vcadd(S.vsel(gv_cnt, zero2, S.vcmps(idx2, fine_seg_size * 3, full2, 'lt')), full2),
                            full2,
                            'ONEPT_B32',
                        )
                        S.vsts(
                            scalar_f_ub[11],
                            S.vcadd(S.vsel(gv_cnt, zero2, S.vcmps(idx2, fine_seg_size * 4, full2, 'lt')), full2),
                            full2,
                            'ONEPT_B32',
                        )
                        S.vsts(
                            scalar_f_ub[12],
                            S.vcadd(S.vsel(gv_cnt, zero2, S.vcmps(idx2, fine_seg_size * 5, full2, 'lt')), full2),
                            full2,
                            'ONEPT_B32',
                        )
                        S.vsts(
                            scalar_f_ub[13],
                            S.vcadd(S.vsel(gv_cnt, zero2, S.vcmps(idx2, fine_seg_size * 6, full2, 'lt')), full2),
                            full2,
                            'ONEPT_B32',
                        )
                        S.vsts(
                            scalar_f_ub[14],
                            S.vcadd(S.vsel(gv_cnt, zero2, S.vcmps(idx2, fine_seg_size * 7, full2, 'lt')), full2),
                            full2,
                            'ONEPT_B32',
                        )
                        S.vsts(scalar_f_ub[15], S.vcadd(gv_cnt, full2), full2, 'ONEPT_B32')
                    sync_v_to_s()
                    g_lo = T.alloc_var(T.int32, 0)
                    g_hi = T.alloc_var(T.int32, 7)
                    g_seg = T.alloc_var(T.int32, 7)
                    while g_lo <= g_hi:
                        g_mid = T.alloc_var(T.int32, (g_lo + g_hi) // 2)
                        g_mid_sum = T.alloc_var(T.float32, chunk_prefix + scalar_f_ub[8 + g_mid])
                        if scalar_f_ub[target_slot] <= g_mid_sum:
                            g_seg = g_mid
                            g_hi = g_mid - 1
                        else:
                            g_lo = g_mid + 1
                    g_seg_start = T.alloc_var(T.int32, g_seg * fine_seg_size)
                    g_seg_end = T.alloc_var(T.int32, T.min((g_seg + 1) * fine_seg_size, fine_end))
                    g_prefix = T.alloc_var(T.float32, chunk_prefix)
                    if g_seg > 0:
                        g_prefix = chunk_prefix + scalar_f_ub[8 + g_seg - 1]
                    g_found = T.alloc_var(T.bool, False)
                    gv_idx = T.alloc_var(T.int32, 0)
                    while gv_idx < (g_seg_end - g_seg_start) and not g_found:
                        g_next = T.alloc_var(
                            T.float32,
                            g_prefix + group_sum_ub[fine_start + g_seg_start + gv_idx],
                        )
                        if scalar_f_ub[target_slot] > g_prefix and scalar_f_ub[target_slot] <= g_next:
                            scalar_i_ub[selected_group_slot] = fine_start + g_seg_start + gv_idx
                            scalar_f_ub[group_prefix_slot] = g_prefix
                            g_found = True
                        g_prefix = g_next
                        gv_idx += 1
                    if not g_found:
                        scalar_i_ub[selected_group_slot] = fine_start + fine_end - 1
                        scalar_f_ub[group_prefix_slot] = g_prefix - group_sum_ub[fine_start + fine_end - 1]
                else:
                    fine_prefix = T.alloc_var(T.float32, chunk_prefix)
                    fv = T.alloc_var(T.int32, 0)
                    fine_found = T.alloc_var(T.bool, False)
                    while fv < fine_end and not fine_found:
                        fine_next = T.alloc_var(
                            T.float32,
                            fine_prefix + group_sum_ub[fine_start + fv],
                        )
                        if scalar_f_ub[target_slot] > fine_prefix and scalar_f_ub[target_slot] <= fine_next:
                            scalar_i_ub[selected_group_slot] = fine_start + fv
                            scalar_f_ub[group_prefix_slot] = fine_prefix
                            fine_found = True
                        fine_prefix = fine_next
                        fv += 1

                selected_base = T.alloc_var(
                    T.int32,
                    scalar_i_ub[selected_group_slot] * vl,
                )
                selected_count = T.alloc_var(
                    T.int32,
                    T.min(vl, vocab_size - selected_base),
                )
                # S2: Skip GM reload for single-chunk — data already in probs_ub[0]
                if num_chunks > 1:
                    T.copy(
                        probs[row, selected_base : selected_base + selected_count],
                        probs_ub[0, :selected_count],
                    )
                    T.ascend_set_flag('MTE2_S', 0)
                    T.ascend_wait_flag('MTE2_S', 0)

                group_target = T.alloc_var(
                    T.float32,
                    scalar_f_ub[target_slot] - scalar_f_ub[group_prefix_slot],
                )

                # ── S1: VEC 8-segment reduction + scalar binary search ──
                # Compute 8 partial prefix sums at 8,16,24,32,40,48,56,64
                # Binary search (3 iterations) narrows to 8-element range
                # Linear search within 8 (avg 4, max 8). Total avg ~7 (vs 13.5 for 3-seg)
                with T.SimdVF():
                    full = S.pset(32, 'PAT_ALL')
                    zero = S.vdup(0.0, T.float32, full)
                    idx = S.vci(T.int32(0), T.int32)
                    pv = S.vld(probs_ub[0, selected_base]) if num_chunks == 1 else S.vld(probs_ub[0, 0])
                    m_cnt = S.vcmps(idx, selected_count, full, 'lt')
                    pv_cnt = S.vsel(pv, zero, m_cnt)
                    S.vsts(
                        scalar_f_ub[0],
                        S.vcadd(S.vsel(pv_cnt, zero, S.vcmps(idx, fine_seg_size, full, 'lt')), full),
                        full,
                        'ONEPT_B32',
                    )
                    S.vsts(
                        scalar_f_ub[1],
                        S.vcadd(S.vsel(pv_cnt, zero, S.vcmps(idx, fine_seg_size * 2, full, 'lt')), full),
                        full,
                        'ONEPT_B32',
                    )
                    S.vsts(
                        scalar_f_ub[2],
                        S.vcadd(S.vsel(pv_cnt, zero, S.vcmps(idx, fine_seg_size * 3, full, 'lt')), full),
                        full,
                        'ONEPT_B32',
                    )
                    S.vsts(
                        scalar_f_ub[3],
                        S.vcadd(S.vsel(pv_cnt, zero, S.vcmps(idx, fine_seg_size * 4, full, 'lt')), full),
                        full,
                        'ONEPT_B32',
                    )
                    S.vsts(
                        scalar_f_ub[4],
                        S.vcadd(S.vsel(pv_cnt, zero, S.vcmps(idx, fine_seg_size * 5, full, 'lt')), full),
                        full,
                        'ONEPT_B32',
                    )
                    S.vsts(
                        scalar_f_ub[5],
                        S.vcadd(S.vsel(pv_cnt, zero, S.vcmps(idx, fine_seg_size * 6, full, 'lt')), full),
                        full,
                        'ONEPT_B32',
                    )
                    S.vsts(
                        scalar_f_ub[6],
                        S.vcadd(S.vsel(pv_cnt, zero, S.vcmps(idx, fine_seg_size * 7, full, 'lt')), full),
                        full,
                        'ONEPT_B32',
                    )
                    S.vsts(scalar_f_ub[7], S.vcadd(pv_cnt, full), full, 'ONEPT_B32')
                sync_v_to_s()

                # Binary search over 8 prefix sums
                f_lo = T.alloc_var(T.int32, 0)
                f_hi = T.alloc_var(T.int32, 7)
                f_seg = T.alloc_var(T.int32, 7)
                while f_lo <= f_hi:
                    f_mid = T.alloc_var(T.int32, (f_lo + f_hi) // 2)
                    f_mid_sum = T.alloc_var(T.float32, scalar_f_ub[f_mid])
                    if group_target <= f_mid_sum:
                        f_seg = f_mid
                        f_hi = f_mid - 1
                    else:
                        f_lo = f_mid + 1

                # Linear search within found 8-element segment
                f_seg_start = T.alloc_var(T.int32, f_seg * fine_seg_size)
                f_seg_end = T.alloc_var(T.int32, T.min((f_seg + 1) * fine_seg_size, selected_count))
                f_prefix = T.alloc_var(T.float32, 0.0)
                if f_seg > 0:
                    f_prefix = scalar_f_ub[f_seg - 1]
                found = T.alloc_var(T.bool, False)
                fl = T.alloc_var(T.int32, 0)
                f_se = T.alloc_var(T.int32, f_seg_end - f_seg_start)
                while fl < f_se and not found:
                    c = T.alloc_var(T.int32, selected_base + f_seg_start + fl)
                    if num_chunks == 1:
                        fn = T.alloc_var(T.float32, f_prefix + probs_ub[0, selected_base + f_seg_start + fl])
                    else:
                        fn = T.alloc_var(T.float32, f_prefix + probs_ub[0, f_seg_start + fl])
                    if group_target > f_prefix and group_target <= fn:
                        scalar_i_ub[sample_slot] = c
                        found = True
                    f_prefix = fn
                    fl += 1

                if not found:
                    scalar_i_ub[sample_slot] = T.min(selected_base + selected_count - 1, vocab_size - 1)
                out[row] = scalar_i_ub[sample_slot]

    return _random_sampling
