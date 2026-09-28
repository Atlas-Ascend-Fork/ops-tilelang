import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _compute_seqlens_q_by_confidence_rates_tl_ascend_fallback(nextn: int):
    assert nextn <= 19792
    num_seqs = T.dynamic('num_seqs')
    n_prev_seq = T.dynamic('n_prev_seq')
    # Ascend has no grid-wide barrier here, so keep both dependent phases
    # within one AIV and one kernel launch.
    num_cores = 1
    vector_lanes = 64
    block_size = 4096
    num_vectors = block_size // vector_lanes

    @T.prim_func
    def _compute_seqlens_q_by_confidence_rates(
        remap: T.Tensor[(num_seqs,), T.int32],  # type: ignore
        prev_confidence_rates: T.Tensor[(n_prev_seq, nextn), T.float32],  # type: ignore
        cum_rates: T.Tensor[(num_seqs * nextn,), T.float32],  # type: ignore
        seqlens_q: T.Tensor[(num_seqs,), T.int32],  # type: ignore
        total_q: T.int32,  # type: ignore
    ):
        # Phase 1: compute cumprod, reading confidence rates via remap (1.0 for fresh seqs)
        with T.Kernel(num_cores) as pid:
            for seq_idx in T.Persistent([num_seqs], num_cores, pid, group_size=1):
                acc = T.alloc_var(T.float32)
                acc = 1.0
                prev_idx = remap[seq_idx]
                for j in T.serial(nextn):
                    if prev_idx != -1:
                        acc = acc * prev_confidence_rates[prev_idx, j]
                    cum_rates[seq_idx * nextn + j] = acc

            # Phase 2: process flattened rates in SIMD-sized UB tiles.
            total_rates = num_seqs * nextn
            rates_ub = T.alloc_shared((block_size,), T.float32)
            my_rates_ub = T.alloc_shared((nextn,), T.float32)
            chunk_rank_ub = T.alloc_shared((nextn,), T.int32)
            rank_ub = T.alloc_shared((nextn,), T.int32)

            for seq_idx in T.Persistent([num_seqs], num_cores, pid, group_size=1):
                for j in T.serial(nextn):
                    my_rates_ub[j] = cum_rates[seq_idx * nextn + j]
                    rank_ub[j] = 0

                for block_idx in T.serial(T.ceildiv(total_rates, block_size)):
                    flat_base = block_idx * block_size
                    valid_count = T.min(block_size, total_rates - flat_base)
                    T.copy(cum_rates[flat_base : flat_base + valid_count], rates_ub[:valid_count])

                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        zero = S.vdup(0.0, T.float32, full)
                        one = S.vdup(1.0, T.float32, full)

                        for j in T.serial(nextn):
                            lane_sums = S.alloc_var(T.float32)
                            lane_sums = zero
                            my_flat_idx = seq_idx * nextn + j

                            for vector in T.serial(num_vectors):
                                vector_base = flat_base + vector * vector_lanes
                                flat_indices = S.vci(T.int32(vector_base), T.int32)
                                valid = S.vcmps(flat_indices, total_rates, full, 'lt')
                                other_rates = S.vld(rates_ub[vector * vector_lanes])
                                greater = S.vcmps(other_rates, my_rates_ub[j], valid, 'gt')
                                equal = S.vcmps(other_rates, my_rates_ub[j], valid, 'eq')
                                before = S.vcmps(flat_indices, my_flat_idx, valid, 'lt')
                                beats = S.por(
                                    greater,
                                    S.pand(equal, before, full),
                                    full,
                                )
                                lane_sums = S.vadd(
                                    lane_sums,
                                    S.vsel(one, zero, beats),
                                    full,
                                )

                            S.vsts(
                                chunk_rank_ub[j],
                                S.vcvt(S.vcadd(lane_sums, full), T.int32, round='ROUND_Z'),
                                full,
                                'ONEPT_B32',
                            )

                    for j in T.serial(nextn):
                        rank_ub[j] += chunk_rank_ub[j]

                extra_budget = total_q - num_seqs
                num_tokens = T.alloc_var(T.int32)
                num_tokens = 0
                for j in T.serial(nextn):
                    if rank_ub[j] < extra_budget:
                        num_tokens = num_tokens + 1

                seqlens_q[seq_idx] = 1 + num_tokens

    return _compute_seqlens_q_by_confidence_rates


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _compute_seqlens_q_by_confidence_rates_tl_ascend(nextn: int, block_size: int):
    """Fused cumulative-rate generation and exact rank selection.

    Every vector core redundantly builds the complete cumulative-rate array in
    its private UB.  Rank work is then distributed by sequence across cores, so
    the producer/consumer dependency is core-local and needs no grid barrier.
    """
    num_seqs = T.dynamic('num_seqs')
    n_prev_seq = T.dynamic('n_prev_seq')
    num_cores = get_num_vector_cores()
    simt_threads = 1024
    vector_lanes = 64
    num_vectors = block_size // vector_lanes
    search_steps = nextn.bit_length()

    @T.prim_func
    def _compute_seqlens_q_by_confidence_rates_fused(
        remap: T.Tensor[(num_seqs,), T.int32],  # type: ignore
        prev_confidence_rates: T.Tensor[(n_prev_seq, nextn), T.float32],  # type: ignore
        seqlens_q: T.Tensor[(num_seqs,), T.int32],  # type: ignore
        total_q: T.int32,  # type: ignore
    ):
        with T.Kernel(num_cores) as core_id:
            rates_ub = T.alloc_shared((block_size,), T.float32)
            my_rate_ub = T.alloc_shared((1,), T.float32)
            chunk_rank_ub = T.alloc_shared((1,), T.int32)
            total_rates = num_seqs * nextn
            extra_budget = total_q - num_seqs

            # Each core creates its own complete rate table.  This deliberately
            # duplicates the small cumprod phase to avoid a cross-core barrier.
            with T.SimtVF(threads=simt_threads):
                tid = T.get_thread_binding()
                for work_id in T.serial(T.ceildiv(num_seqs, simt_threads)):
                    rate_seq_idx = work_id * simt_threads + tid
                    if rate_seq_idx < num_seqs:
                        acc = T.alloc_var(T.float32)
                        acc = 1.0
                        prev_idx = remap[rate_seq_idx]
                        for j in T.serial(nextn):
                            if prev_idx != -1:
                                acc = acc * prev_confidence_rates[prev_idx, j]
                            rates_ub[rate_seq_idx * nextn + j] = acc

            for seq_idx in T.Persistent([num_seqs], num_cores, core_id, group_size=1):
                if extra_budget <= 0:
                    seqlens_q[seq_idx] = 1
                else:
                    if extra_budget >= total_rates:
                        seqlens_q[seq_idx] = 1 + nextn
                    else:
                        lo = T.alloc_var(T.int32)
                        hi = T.alloc_var(T.int32)
                        mid = T.alloc_var(T.int32)
                        rank = T.alloc_var(T.int32)
                        lo = 0
                        hi = nextn

                        for _ in T.serial(search_steps):
                            if lo < hi:
                                mid = (lo + hi) // 2
                                my_rate_ub[0] = rates_ub[seq_idx * nextn + mid]
                                rank = 0

                                with T.SimdVF():
                                    full = S.pset(32, 'PAT_ALL')
                                    zero = S.vdup(0.0, T.float32, full)
                                    one = S.vdup(1.0, T.float32, full)
                                    lane_sums = S.alloc_var(T.float32)
                                    lane_sums = zero
                                    my_flat_idx = seq_idx * nextn + mid

                                    for vector in T.serial(num_vectors):
                                        vector_base = vector * vector_lanes
                                        flat_indices = S.vci(T.int32(vector_base), T.int32)
                                        valid = S.vcmps(flat_indices, total_rates, full, 'lt')
                                        other_rates = S.vld(rates_ub[vector_base])
                                        greater = S.vcmps(other_rates, my_rate_ub[0], valid, 'gt')
                                        equal = S.vcmps(other_rates, my_rate_ub[0], valid, 'eq')
                                        before = S.vcmps(flat_indices, my_flat_idx, valid, 'lt')
                                        beats = S.por(
                                            greater,
                                            S.pand(equal, before, full),
                                            full,
                                        )
                                        lane_sums = S.vadd(
                                            lane_sums,
                                            S.vsel(one, zero, beats),
                                            full,
                                        )

                                    S.vsts(
                                        chunk_rank_ub[0],
                                        S.vcvt(S.vcadd(lane_sums, full), T.int32, round='ROUND_Z'),
                                        full,
                                        'ONEPT_B32',
                                    )

                                rank = chunk_rank_ub[0]
                                if rank < extra_budget:
                                    lo = mid + 1
                                else:
                                    hi = mid

                        seqlens_q[seq_idx] = 1 + lo

    return _compute_seqlens_q_by_confidence_rates_fused
