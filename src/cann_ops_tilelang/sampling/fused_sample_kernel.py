from typing import Any

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

DRAFT_POS_HASH_MUL = 10000019


DRAFT_PROB_PENALTY = 1e-6  # 防止 reject sampling 采不出东西


@tilelang.jit(
    out_idx=None,
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
    },
)
def _fused_sample_tl_ascend(
    vocab_size: int,
    do_top_p: bool,
    return_batch_format: bool,
    do_sample: bool,
    do_reject_sampling: bool,
    num_cores: int,
) -> Any:
    """Ascend SIMD implementation with SIMT restricted to Philox RNG and
    the scalar reject-accept decision.

    The vocabulary-wide stages follow the CANN ApplyTopP and RandomSample
    kernels: 64-lane SIMD reductions, a three-threshold top-p pass, and
    hierarchical categorical sampling.  Probabilities are kept in the output
    buffer until sampling is complete, then converted to log probabilities.
    Two-slot UB rings use directional MTE2/V/MTE3 events; V/S events protect
    only the scalar control points that consume SIMD summaries.
    """

    req_cnt = T.dynamic('req_cnt')
    total_q = T.dynamic('total_q')
    num_blocks = T.dynamic('num_blocks')
    logits_stride = T.dynamic('logits_stride')
    p_stride = T.dynamic('p_stride')
    temp_stride = T.dynamic('temp_stride')
    logprobs_stride = T.dynamic('logprobs_stride')
    reject_steps = T.dynamic('reject_steps')
    last_nextn_llp_stride0 = T.dynamic('last_nextn_llp_stride0')
    last_nextn_llp_stride1 = T.dynamic('last_nextn_llp_stride1')

    vector_lanes = 64
    # Reject sampling keeps target and draft double buffers live together.
    # An 8192-token chunk would exceed the Ascend UB launch budget.
    chunk_size = 4096 if do_reject_sampling or vocab_size <= 4096 else 8192
    vregs_per_chunk = chunk_size // vector_lanes
    num_chunks = (vocab_size + chunk_size - 1) // chunk_size
    num_vectors = (vocab_size + vector_lanes - 1) // vector_lanes
    num_vectors_aligned = ((num_vectors + vector_lanes - 1) // vector_lanes) * vector_lanes
    sample_ub_fastpath = num_chunks == 1 and do_sample and not do_top_p and not do_reject_sampling
    top_p_search_rounds = 15
    inv_2pow53 = 1.1102230246251565e-16

    f_max = 0
    f_sum = 1
    f_top_p_target = 2
    f_mass1 = 3
    f_mass2 = 4
    f_mass3 = 5
    f_threshold = 6
    f_kept_sum = 7
    f_random_target = 8
    f_sample_total = 9
    f_inv_temperature = 10
    f_group_prefix = 11

    i_cut = 0
    i_selected_vec = 1
    i_sample = 2

    @T.macro
    def init_input_db() -> None:
        T.ascend_set_flag('V_MTE2', 0)
        T.ascend_set_flag('V_MTE2', 1)

    @T.macro
    def drain_input_db() -> None:
        T.ascend_wait_flag('V_MTE2', 0)
        T.ascend_wait_flag('V_MTE2', 1)

    @T.macro
    def init_output_db() -> None:
        T.ascend_set_flag('MTE3_V', 0)
        T.ascend_set_flag('MTE3_V', 1)

    @T.macro
    def drain_output_db() -> None:
        T.ascend_wait_flag('MTE3_V', 0)
        T.ascend_wait_flag('MTE3_V', 1)
        # MTE3_V waits retire on the vector consumer and do not stall scalar
        # issue.  Close the writeback phase explicitly before another phase
        # reads the same GM buffer or the kernel advances to the next row.
        T.ascend_set_flag('MTE3_S', 0)
        T.ascend_wait_flag('MTE3_S', 0)
        T.ascend_set_flag('V_S', 0)
        T.ascend_wait_flag('V_S', 0)

    @T.macro
    def sync_v_to_s() -> None:
        T.ascend_set_flag('V_S', 0)
        T.ascend_wait_flag('V_S', 0)

    @T.macro
    def sync_s_to_v() -> None:
        T.ascend_set_flag('S_V', 0)
        T.ascend_wait_flag('S_V', 0)

    @T.macro
    def rng_uniform53(seed: T.uint64, sequence: T.int32) -> T.float32:  # type: ignore
        T.rng_init(seed, sequence)
        counter0 = T.alloc_var(T.uint64, T.cast(T.rng_rand(), T.uint64))
        counter1 = T.alloc_var(T.uint64, T.cast(T.rng_rand(), T.uint64))
        random53 = T.alloc_var(T.uint64, counter0 ^ (counter1 << 21))
        return T.cast(random53, T.float32) * T.float32(inv_2pow53) + T.float32(inv_2pow53 / 2.0)

    @T.macro
    def clear_accumulators(
        acc1_ub: T.Tensor,  # type: ignore
        acc2_ub: T.Tensor,  # type: ignore
        acc3_ub: T.Tensor,  # type: ignore
    ) -> None:
        with T.SimdVF():
            zero = S.vdup(0.0, T.float32)
            S.vsts(acc1_ub[0], zero)
            S.vsts(acc2_ub[0], zero)
            S.vsts(acc3_ub[0], zero)

    @T.macro
    def select_from_probability_groups(
        probabilities: T.Tensor,  # type: ignore
        probability_row: T.int32,  # type: ignore
        seed: T.uint64,  # type: ignore
        sequence: T.int32,  # type: ignore
        default_token: T.int32,  # type: ignore
        probs_ub: T.Tensor,  # type: ignore
        group_sum_ub: T.Tensor,  # type: ignore
        scalar_f_ub: T.Tensor,  # type: ignore
        scalar_i_ub: T.Tensor,  # type: ignore
    ) -> None:
        with T.SimtVF(threads=1):
            scalar_f_ub[f_random_target] = rng_uniform53(seed, sequence) * scalar_f_ub[f_sample_total]
        sync_v_to_s()

        prefix = T.alloc_var(T.float32, 0.0)
        found = T.alloc_var(T.bool, False)
        scalar_i_ub[i_selected_vec] = num_vectors - 1
        scalar_i_ub[i_sample] = default_token
        scalar_f_ub[f_group_prefix] = 0.0
        for vector in T.serial(num_vectors):
            next_prefix = T.alloc_var(T.float32, prefix + group_sum_ub[vector])
            if not found and scalar_f_ub[f_random_target] > prefix and scalar_f_ub[f_random_target] <= next_prefix:
                scalar_i_ub[i_selected_vec] = vector
                scalar_f_ub[f_group_prefix] = prefix
                found = True
            prefix = next_prefix
        if not found:
            scalar_f_ub[f_group_prefix] = prefix - group_sum_ub[num_vectors - 1]

        selected_base = T.alloc_var(T.int32, scalar_i_ub[i_selected_vec] * vector_lanes)
        selected_count = T.alloc_var(T.int32, T.min(vector_lanes, vocab_size - selected_base))
        if num_chunks > 1:
            T.copy(
                probabilities[probability_row, selected_base : selected_base + selected_count],
                probs_ub[0, :selected_count],
            )
            T.ascend_set_flag('MTE2_S', 0)
            T.ascend_wait_flag('MTE2_S', 0)
        group_target = T.alloc_var(
            T.float32,
            scalar_f_ub[f_random_target] - scalar_f_ub[f_group_prefix],
        )
        prefix = 0.0
        found = False
        for lane in T.serial(vector_lanes):
            col = selected_base + lane
            if col < vocab_size:
                if num_chunks > 1:
                    probability = T.alloc_var(T.float32, probs_ub[0, lane])
                else:
                    probability = T.alloc_var(T.float32, probs_ub[0, selected_base + lane])
                next_prefix = T.alloc_var(T.float32, prefix + probability)
                if (
                    not found
                    and scalar_f_ub[f_sample_total] > 0.0
                    and group_target > prefix
                    and group_target <= next_prefix
                ):
                    scalar_i_ub[i_sample] = col
                    found = True
                prefix = next_prefix

    @T.macro
    def build_residual_group_sums(
        probabilities: T.Tensor,  # type: ignore
        draft_logprobs: T.Tensor,  # type: ignore
        row: T.int32,  # type: ignore
        sx: T.int32,  # type: ignore
        bx: T.int32,  # type: ignore
        probs_ub: T.Tensor,  # type: ignore
        draft_ub: T.Tensor,  # type: ignore
        group_sum_ub: T.Tensor,  # type: ignore
        scalar_f_ub: T.Tensor,  # type: ignore
    ) -> None:
        with T.SimdVF():
            zero = S.vdup(0.0, T.float32)
            for vector in T.serial(num_vectors_aligned // vector_lanes):
                S.vsts(group_sum_ub[vector * vector_lanes], zero)
        init_input_db()
        for chunk in T.serial(num_chunks):
            slot = chunk % 2
            base = chunk * chunk_size
            valid_count = T.min(chunk_size, vocab_size - base)
            T.ascend_wait_flag('V_MTE2', slot)
            T.copy(probabilities[row, base : base + valid_count], probs_ub[slot, :valid_count])
            T.copy(
                draft_logprobs[sx, bx, base : base + valid_count],
                draft_ub[slot, :valid_count],
            )
            T.ascend_set_flag('MTE2_V', slot)
            T.ascend_wait_flag('MTE2_V', slot)
            with T.SimdVF():
                full = S.pset(32, 'PAT_ALL')
                zero = S.vdup(0.0, T.float32, full)
                for vector in T.serial(vregs_per_chunk):
                    col = base + vector * vector_lanes
                    indices = S.vci(T.int32(col), T.int32)
                    valid = S.vcmps(indices, vocab_size, full, 'lt')
                    target_prob = S.vld(probs_ub[slot, vector * vector_lanes])
                    draft_prob = S.vexp(
                        S.vadds(
                            S.vld(draft_ub[slot, vector * vector_lanes]),
                            -DRAFT_PROB_PENALTY,
                            full,
                        ),
                        full,
                    )
                    residual = S.vsel(
                        S.vrelu(S.vsub(target_prob, draft_prob, full), full),
                        zero,
                        valid,
                    )
                    S.vsts(
                        group_sum_ub[chunk * vregs_per_chunk + vector],
                        S.vcadd(residual, full),
                        full,
                        'ONEPT_B32',
                    )
            T.ascend_set_flag('V_MTE2', slot)
        drain_input_db()

        with T.SimdVF():
            full = S.pset(32, 'PAT_ALL')
            total = S.alloc_var(T.float32)
            total = S.vdup(0.0, T.float32, full)
            for vector in T.serial(num_vectors_aligned // vector_lanes):
                total = S.vadd(total, S.vld(group_sum_ub[vector * vector_lanes]), full)
            S.vsts(scalar_f_ub[f_sample_total], S.vcadd(total, full), full, 'ONEPT_B32')
        sync_v_to_s()

    @T.prim_func
    def _fused_sample(
        logits: T.StridedTensor((total_q, vocab_size), (logits_stride, 1), T.float32),  # type: ignore
        temperatures: T.StridedTensor((req_cnt,), (temp_stride,), T.float32),  # type: ignore
        top_p: T.StridedTensor((req_cnt,), (p_stride,), T.float32),  # type: ignore
        ftp: T.Tensor((req_cnt,), T.int32),  # type: ignore
        sample_positions: T.Tensor((req_cnt,), T.int32),  # type: ignore
        seq_seeds: T.Tensor((req_cnt,), T.uint64),  # type: ignore
        logprobs: T.StridedTensor((num_blocks, vocab_size), (logprobs_stride, 1), T.float32),  # type: ignore
        sample_result: T.Tensor((num_blocks,), T.int32),  # type: ignore
        last_nextn_token_logprobs: T.StridedTensor(
            (reject_steps, req_cnt, vocab_size),
            (last_nextn_llp_stride0, last_nextn_llp_stride1, 1),
            T.float32,
        ),  # type: ignore
        last_next_n_token_ids: T.Tensor((reject_steps, req_cnt), T.int32),  # type: ignore
        accept_result_verify: T.Tensor((reject_steps, req_cnt), T.bool),  # type: ignore
        sample_reject_cur: T.Tensor((reject_steps, req_cnt), T.int32),  # type: ignore
        seed_offset: T.int32,
    ) -> None:
        with T.Kernel(num_cores) as core_id:
            logits_ub = T.alloc_shared((2, chunk_size), T.float32)
            probs_ub = T.alloc_shared((2, chunk_size), T.float32)
            draft_ub = T.alloc_shared((2, chunk_size), T.float32)
            work_ub = T.alloc_shared((chunk_size,), T.float32)
            group_sum_ub = T.alloc_shared((num_vectors_aligned,), T.float32)
            acc1_ub = T.alloc_shared((vector_lanes,), T.float32)
            acc2_ub = T.alloc_shared((vector_lanes,), T.float32)
            acc3_ub = T.alloc_shared((vector_lanes,), T.float32)
            scalar_f_ub = T.alloc_shared((vector_lanes,), T.float32)
            scalar_i_ub = T.alloc_shared((vector_lanes,), T.int32)
            for pid in T.Persistent([num_blocks], num_cores, core_id, group_size=1):
                if return_batch_format:
                    sx = T.alloc_var(T.int32, pid // req_cnt)
                    bx = T.alloc_var(T.int32, pid % req_cnt)
                    row = T.alloc_var(T.int32, ftp[bx] + sx)
                    upper = T.alloc_var(T.int32, T.Select(bx < req_cnt - 1, ftp[bx + 1], total_q))
                    valid_row = T.alloc_var(T.bool, row < upper)
                else:
                    sx = T.alloc_var(T.int32, 0)
                    bx = T.alloc_var(T.int32, pid)
                    row = T.alloc_var(T.int32, pid)
                    valid_row = T.alloc_var(T.bool, True)

                if valid_row:
                    T.device_assert(temperatures[bx] > 0, msg='temperatures must be positive')
                    if do_top_p:
                        T.device_assert(top_p[bx] > 0 and top_p[bx] <= 1, msg='top_p must be in (0, 1]')
                    scalar_f_ub[f_inv_temperature] = 1.0 / temperatures[bx]
                    sync_s_to_v()

                    # Softmax max: one SIMD reduction per GM chunk, followed by
                    # a SIMD reduction over the small chunk summary.
                    init_input_db()
                    for chunk in T.serial(num_chunks):
                        slot = chunk % 2
                        base = chunk * chunk_size
                        valid_count = T.min(chunk_size, vocab_size - base)
                        T.ascend_wait_flag('V_MTE2', slot)
                        T.copy(logits[row, base : base + valid_count], logits_ub[slot, :valid_count])
                        T.ascend_set_flag('MTE2_V', slot)
                        T.ascend_wait_flag('MTE2_V', slot)
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            neg_inf = S.vdup(-T.infinity(T.float32), T.float32, full)
                            chunk_max = S.alloc_var(T.float32)
                            chunk_max = neg_inf
                            for vector in T.serial(vregs_per_chunk):
                                col = base + vector * vector_lanes
                                indices = S.vci(T.int32(col), T.int32)
                                valid = S.vcmps(indices, vocab_size, full, 'lt')
                                values = S.vsel(S.vld(logits_ub[slot, vector * vector_lanes]), neg_inf, valid)
                                chunk_max = S.vmax(chunk_max, values, full)
                            S.vsts(
                                group_sum_ub[chunk],
                                S.vcmax(chunk_max, full),
                                full,
                                'ONEPT_B32',
                            )
                        T.ascend_set_flag('V_MTE2', slot)
                    drain_input_db()

                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        neg_inf = S.vdup(-T.infinity(T.float32), T.float32, full)
                        chunk_indices = S.vci(0, T.int32)
                        chunk_values = S.vsel(
                            S.vld(group_sum_ub[0]),
                            neg_inf,
                            S.vcmps(chunk_indices, num_chunks, full, 'lt'),
                        )
                        S.vsts(
                            scalar_f_ub[f_max],
                            S.vcmax(chunk_values, full),
                            full,
                            'ONEPT_B32',
                        )

                    # Top-p and multi-chunk paths materialize unnormalized
                    # probabilities in GM. A single-chunk sampling row stays in
                    # UB through normalization and token selection.
                    if not sample_ub_fastpath:
                        init_input_db()
                        init_output_db()
                    for chunk in T.serial(num_chunks):
                        slot = chunk % 2
                        base = chunk * chunk_size
                        valid_count = T.min(chunk_size, vocab_size - base)
                        if not sample_ub_fastpath:
                            T.ascend_wait_flag('V_MTE2', slot)
                            T.copy(logits[row, base : base + valid_count], logits_ub[slot, :valid_count])
                            T.ascend_set_flag('MTE2_V', slot)
                            T.ascend_wait_flag('MTE2_V', slot)
                            T.ascend_wait_flag('MTE3_V', slot)
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            zero = S.vdup(0.0, T.float32, full)
                            max_vec = S.vld(scalar_f_ub[f_max], dist='BRC_B32')
                            inv_temp = S.vld(scalar_f_ub[f_inv_temperature], dist='BRC_B32')
                            chunk_sum = S.alloc_var(T.float32)
                            chunk_sum = zero
                            for vector in T.serial(vregs_per_chunk):
                                col = base + vector * vector_lanes
                                indices = S.vci(T.int32(col), T.int32)
                                valid = S.vcmps(indices, vocab_size, full, 'lt')
                                values = S.vld(logits_ub[slot, vector * vector_lanes])
                                values = S.vmul(S.vsub(values, max_vec, full), inv_temp, full)
                                probabilities = S.vsel(S.vexp(values, full), zero, valid)
                                S.vsts(probs_ub[slot, vector * vector_lanes], probabilities, full)
                                chunk_sum = S.vadd(chunk_sum, probabilities, full)
                            S.vsts(
                                group_sum_ub[chunk],
                                S.vcadd(chunk_sum, full),
                                full,
                                'ONEPT_B32',
                            )
                        if not sample_ub_fastpath:
                            T.ascend_set_flag('V_MTE2', slot)
                            T.ascend_set_flag('V_MTE3', slot)
                            T.ascend_wait_flag('V_MTE3', slot)
                            T.copy(probs_ub[slot, :valid_count], logprobs[pid, base : base + valid_count])
                            T.ascend_set_flag('MTE3_V', slot)
                    if not sample_ub_fastpath:
                        drain_input_db()
                        drain_output_db()

                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        zero = S.vdup(0.0, T.float32, full)
                        chunk_indices = S.vci(0, T.int32)
                        chunk_values = S.vsel(
                            S.vld(group_sum_ub[0]),
                            zero,
                            S.vcmps(chunk_indices, num_chunks, full, 'lt'),
                        )
                        S.vsts(
                            scalar_f_ub[f_sum],
                            S.vcadd(chunk_values, full),
                            full,
                            'ONEPT_B32',
                        )
                    sync_v_to_s()

                    T.device_assert(
                        scalar_f_ub[f_sum] > 0,
                        msg='Invalid logits: contains NaN or +inf, or all entries are -inf',
                    )
                    scalar_f_ub[f_kept_sum] = scalar_f_ub[f_sum]
                    scalar_i_ub[i_cut] = vocab_size - 1

                    if do_top_p:
                        scalar_f_ub[f_top_p_target] = top_p[bx] * scalar_f_ub[f_sum]
                        if top_p[bx] < 1.0:
                            lo_bits = T.alloc_var(T.uint32, 0)
                            hi_bits = T.alloc_var(T.uint32, 0x40000000)
                            for _search_round in T.serial(top_p_search_rounds):
                                span = T.alloc_var(T.uint32, hi_bits - lo_bits)
                                q1_bits = T.alloc_var(T.uint32, lo_bits + span // 4)
                                q2_bits = T.alloc_var(T.uint32, lo_bits + span // 2)
                                q3_bits = T.alloc_var(T.uint32, lo_bits + (span // 4) * 3)
                                q1 = T.alloc_var(T.float32, T.reinterpret(T.float32, q1_bits))
                                q2 = T.alloc_var(T.float32, T.reinterpret(T.float32, q2_bits))
                                q3 = T.alloc_var(T.float32, T.reinterpret(T.float32, q3_bits))
                                clear_accumulators(acc1_ub, acc2_ub, acc3_ub)

                                init_input_db()
                                for chunk in T.serial(num_chunks):
                                    slot = chunk % 2
                                    base = chunk * chunk_size
                                    valid_count = T.min(chunk_size, vocab_size - base)
                                    T.ascend_wait_flag('V_MTE2', slot)
                                    T.copy(
                                        logprobs[pid, base : base + valid_count],
                                        probs_ub[slot, :valid_count],
                                    )
                                    T.ascend_set_flag('MTE2_V', slot)
                                    T.ascend_wait_flag('MTE2_V', slot)
                                    with T.SimdVF():
                                        full = S.pset(32, 'PAT_ALL')
                                        zero = S.vdup(0.0, T.float32, full)
                                        mass1 = S.alloc_var(T.float32)
                                        mass2 = S.alloc_var(T.float32)
                                        mass3 = S.alloc_var(T.float32)
                                        mass1 = S.vld(acc1_ub[0])
                                        mass2 = S.vld(acc2_ub[0])
                                        mass3 = S.vld(acc3_ub[0])
                                        for vector in T.serial(vregs_per_chunk):
                                            col = base + vector * vector_lanes
                                            indices = S.vci(T.int32(col), T.int32)
                                            valid = S.vcmps(indices, vocab_size, full, 'lt')
                                            probability = S.vsel(
                                                S.vld(probs_ub[slot, vector * vector_lanes]),
                                                zero,
                                                valid,
                                            )
                                            mass1 = S.vadd(
                                                mass1,
                                                S.vsel(probability, zero, S.vcmps(probability, q1, full, 'ge')),
                                                full,
                                            )
                                            mass2 = S.vadd(
                                                mass2,
                                                S.vsel(probability, zero, S.vcmps(probability, q2, full, 'ge')),
                                                full,
                                            )
                                            mass3 = S.vadd(
                                                mass3,
                                                S.vsel(probability, zero, S.vcmps(probability, q3, full, 'ge')),
                                                full,
                                            )
                                        S.vsts(acc1_ub[0], mass1, full)
                                        S.vsts(acc2_ub[0], mass2, full)
                                        S.vsts(acc3_ub[0], mass3, full)
                                    T.ascend_set_flag('V_MTE2', slot)
                                drain_input_db()

                                with T.SimdVF():
                                    full = S.pset(32, 'PAT_ALL')
                                    S.vsts(scalar_f_ub[f_mass1], S.vcadd(S.vld(acc1_ub[0]), full), full, 'ONEPT_B32')
                                    S.vsts(scalar_f_ub[f_mass2], S.vcadd(S.vld(acc2_ub[0]), full), full, 'ONEPT_B32')
                                    S.vsts(scalar_f_ub[f_mass3], S.vcadd(S.vld(acc3_ub[0]), full), full, 'ONEPT_B32')
                                sync_v_to_s()

                                if scalar_f_ub[f_mass3] >= scalar_f_ub[f_top_p_target]:
                                    lo_bits = q3_bits
                                else:
                                    if scalar_f_ub[f_mass2] >= scalar_f_ub[f_top_p_target]:
                                        lo_bits = q2_bits
                                        hi_bits = q3_bits
                                    else:
                                        if scalar_f_ub[f_mass1] >= scalar_f_ub[f_top_p_target]:
                                            lo_bits = q1_bits
                                            hi_bits = q2_bits
                                        else:
                                            hi_bits = q1_bits

                            scalar_f_ub[f_threshold] = T.reinterpret(T.float32, lo_bits)
                            clear_accumulators(acc1_ub, acc2_ub, acc3_ub)
                            sync_s_to_v()
                            init_input_db()
                            for chunk in T.serial(num_chunks):
                                slot = chunk % 2
                                base = chunk * chunk_size
                                valid_count = T.min(chunk_size, vocab_size - base)
                                T.ascend_wait_flag('V_MTE2', slot)
                                T.copy(
                                    logprobs[pid, base : base + valid_count],
                                    probs_ub[slot, :valid_count],
                                )
                                T.ascend_set_flag('MTE2_V', slot)
                                T.ascend_wait_flag('MTE2_V', slot)
                                with T.SimdVF():
                                    full = S.pset(32, 'PAT_ALL')
                                    one = S.vdup(1.0, T.float32, full)
                                    zero = S.vdup(0.0, T.float32, full)
                                    mass_ge = S.alloc_var(T.float32)
                                    mass_gt = S.alloc_var(T.float32)
                                    mass_ge = S.vld(acc1_ub[0])
                                    mass_gt = S.vld(acc2_ub[0])
                                    for vector in T.serial(vregs_per_chunk):
                                        col = base + vector * vector_lanes
                                        indices = S.vci(T.int32(col), T.int32)
                                        valid = S.vcmps(indices, vocab_size, full, 'lt')
                                        probability = S.vsel(
                                            S.vld(probs_ub[slot, vector * vector_lanes]),
                                            zero,
                                            valid,
                                        )
                                        mass_ge = S.vadd(
                                            mass_ge,
                                            S.vsel(
                                                probability,
                                                zero,
                                                S.vcmps(probability, scalar_f_ub[f_threshold], full, 'ge'),
                                            ),
                                            full,
                                        )
                                        mass_gt = S.vadd(
                                            mass_gt,
                                            S.vsel(
                                                probability,
                                                zero,
                                                S.vcmps(probability, scalar_f_ub[f_threshold], full, 'gt'),
                                            ),
                                            full,
                                        )
                                        equal = S.pand(
                                            S.vcmps(probability, scalar_f_ub[f_threshold], full, 'eq'),
                                            valid,
                                            full,
                                        )
                                        S.vsts(
                                            group_sum_ub[chunk * vregs_per_chunk + vector],
                                            S.vcadd(S.vsel(one, zero, equal), full),
                                            full,
                                            'ONEPT_B32',
                                        )
                                    S.vsts(acc1_ub[0], mass_ge, full)
                                    S.vsts(acc2_ub[0], mass_gt, full)
                                T.ascend_set_flag('V_MTE2', slot)
                            drain_input_db()

                            with T.SimdVF():
                                full = S.pset(32, 'PAT_ALL')
                                S.vsts(scalar_f_ub[f_mass1], S.vcadd(S.vld(acc1_ub[0]), full), full, 'ONEPT_B32')
                                S.vsts(scalar_f_ub[f_mass2], S.vcadd(S.vld(acc2_ub[0]), full), full, 'ONEPT_B32')
                            sync_v_to_s()

                            equal_ratio = T.alloc_var(
                                T.float32,
                                T.Select(
                                    scalar_f_ub[f_threshold] > 0,
                                    (scalar_f_ub[f_top_p_target] - scalar_f_ub[f_mass2]) / scalar_f_ub[f_threshold],
                                    1.0,
                                ),
                            )
                            selected_equal = T.alloc_var(
                                T.int32,
                                T.max(T.cast(equal_ratio, T.int32), 1),
                            )
                            if T.cast(selected_equal, T.float32) < equal_ratio:
                                selected_equal += 1
                            equal_total = T.alloc_var(
                                T.int32,
                                T.Select(
                                    scalar_f_ub[f_threshold] > 0,
                                    T.max(
                                        T.cast(
                                            (scalar_f_ub[f_mass1] - scalar_f_ub[f_mass2]) / scalar_f_ub[f_threshold]
                                            + 0.5,
                                            T.int32,
                                        ),
                                        1,
                                    ),
                                    1,
                                ),
                            )
                            selected_equal = T.min(selected_equal, equal_total)
                            remaining = T.alloc_var(T.int32, selected_equal)

                            # SIMD produced one equal-count summary per
                            # 64-token group in the S_ge/S_gt pass. Scalar
                            # control locates the only boundary group, then
                            # inspects at most 64 probabilities in token order.
                            found_equal_group = T.alloc_var(T.bool, False)
                            scalar_i_ub[i_selected_vec] = num_vectors - 1
                            for vector in T.serial(num_vectors):
                                equal_count = T.alloc_var(T.int32, T.cast(group_sum_ub[vector], T.int32))
                                inspect_group = T.alloc_var(
                                    T.bool,
                                    not found_equal_group and remaining > 0 and remaining <= equal_count,
                                )
                                if inspect_group:
                                    scalar_i_ub[i_selected_vec] = vector
                                    found_equal_group = True
                                if not found_equal_group and remaining > equal_count:
                                    remaining -= equal_count

                            selected_base = T.alloc_var(T.int32, scalar_i_ub[i_selected_vec] * vector_lanes)
                            selected_count = T.alloc_var(T.int32, T.min(vector_lanes, vocab_size - selected_base))
                            T.copy(
                                logprobs[pid, selected_base : selected_base + selected_count],
                                probs_ub[0, :selected_count],
                            )
                            T.ascend_set_flag('MTE2_S', 0)
                            T.ascend_wait_flag('MTE2_S', 0)
                            for lane in T.serial(vector_lanes):
                                col = selected_base + lane
                                if (
                                    lane < selected_count
                                    and remaining > 0
                                    and probs_ub[0, lane] == scalar_f_ub[f_threshold]
                                ):
                                    remaining -= 1
                                    scalar_i_ub[i_cut] = T.Select(
                                        remaining == 0,
                                        col,
                                        scalar_i_ub[i_cut],
                                    )

                            # Apply the exact tie cutoff and accumulate the kept
                            # mass in persistent SIMD lane accumulators.
                            clear_accumulators(acc1_ub, acc2_ub, acc3_ub)
                            sync_s_to_v()
                            init_input_db()
                            init_output_db()
                            for chunk in T.serial(num_chunks):
                                slot = chunk % 2
                                base = chunk * chunk_size
                                valid_count = T.min(chunk_size, vocab_size - base)
                                # Keep the previous MTE3 write for this slot out
                                # of flight before the next filter iteration reuses it.
                                if chunk >= 2:
                                    T.ascend_wait_flag('MTE3_S', slot)
                                T.ascend_wait_flag('V_MTE2', slot)
                                T.ascend_wait_flag('MTE3_V', slot)
                                T.copy(
                                    logprobs[pid, base : base + valid_count],
                                    probs_ub[slot, :valid_count],
                                )
                                T.ascend_set_flag('MTE2_V', slot)
                                T.ascend_wait_flag('MTE2_V', slot)
                                with T.SimdVF():
                                    full = S.pset(32, 'PAT_ALL')
                                    zero = S.vdup(0.0, T.float32, full)
                                    kept_sum = S.alloc_var(T.float32)
                                    kept_sum = S.vld(acc1_ub[0])
                                    for vector in T.serial(vregs_per_chunk):
                                        col = base + vector * vector_lanes
                                        indices = S.vci(T.int32(col), T.int32)
                                        valid = S.vcmps(indices, vocab_size, full, 'lt')
                                        probability = S.vsel(
                                            S.vld(probs_ub[slot, vector * vector_lanes]),
                                            zero,
                                            valid,
                                        )
                                        greater = S.vcmps(probability, scalar_f_ub[f_threshold], full, 'gt')
                                        equal = S.vcmps(probability, scalar_f_ub[f_threshold], full, 'eq')
                                        before_cut = S.vcmps(indices, scalar_i_ub[i_cut], full, 'le')
                                        keep = S.por(greater, S.pand(equal, before_cut, full), full)
                                        probability = S.vsel(probability, zero, keep)
                                        S.vsts(probs_ub[slot, vector * vector_lanes], probability, full)
                                        if do_sample and not do_reject_sampling:
                                            # Plain top-p sampling can select from
                                            # unnormalized masses. Reuse this
                                            # already-required filter pass to
                                            # retain its 64-token group sums.
                                            S.vsts(
                                                group_sum_ub[chunk * vregs_per_chunk + vector],
                                                S.vcadd(probability, full),
                                                full,
                                                'ONEPT_B32',
                                            )
                                        kept_sum = S.vadd(kept_sum, probability, full)
                                    S.vsts(acc1_ub[0], kept_sum, full)
                                T.ascend_set_flag('V_MTE2', slot)
                                T.ascend_set_flag('V_MTE3', slot)
                                T.ascend_wait_flag('V_MTE3', slot)
                                T.copy(
                                    probs_ub[slot, :valid_count],
                                    logprobs[pid, base : base + valid_count],
                                )
                                T.ascend_set_flag('MTE3_S', slot)
                                T.ascend_set_flag('MTE3_V', slot)
                            T.ascend_wait_flag('MTE3_S', 0)
                            if num_chunks > 1:
                                T.ascend_wait_flag('MTE3_S', 1)
                            drain_input_db()
                            drain_output_db()

                            with T.SimdVF():
                                full = S.pset(32, 'PAT_ALL')
                                S.vsts(
                                    scalar_f_ub[f_kept_sum],
                                    S.vcadd(S.vld(acc1_ub[0]), full),
                                    full,
                                    'ONEPT_B32',
                                )
                            sync_v_to_s()

                    # Plain top-p sampling can use the unnormalized filtered
                    # mass directly. This avoids a GM read/normalize/write pass;
                    # final log-probability conversion folds in the denominator.
                    if do_sample:
                        if do_top_p and not do_reject_sampling and top_p[bx] < 1.0:
                            scalar_f_ub[f_sample_total] = scalar_f_ub[f_kept_sum]
                        else:
                            if not do_top_p:
                                sync_s_to_v()
                            with T.SimdVF():
                                zero = S.vdup(0.0, T.float32)
                                for vector in T.serial(num_vectors_aligned // vector_lanes):
                                    S.vsts(group_sum_ub[vector * vector_lanes], zero)
                            if not sample_ub_fastpath:
                                init_input_db()
                                init_output_db()
                            for chunk in T.serial(num_chunks):
                                slot = chunk % 2
                                base = chunk * chunk_size
                                valid_count = T.min(chunk_size, vocab_size - base)
                                if chunk >= 2:
                                    if chunk % 2 == 0:
                                        T.ascend_wait_flag('MTE3_S', 0)
                                    else:
                                        T.ascend_wait_flag('MTE3_S', 1)
                                if not sample_ub_fastpath:
                                    T.ascend_wait_flag('V_MTE2', slot)
                                    if chunk % 2 == 0:
                                        T.ascend_wait_flag('MTE3_V', 0)
                                    else:
                                        T.ascend_wait_flag('MTE3_V', 1)
                                    T.copy(
                                        logprobs[pid, base : base + valid_count],
                                        probs_ub[slot, :valid_count],
                                    )
                                    T.ascend_set_flag('MTE2_V', slot)
                                    T.ascend_wait_flag('MTE2_V', slot)
                                with T.SimdVF():
                                    full = S.pset(32, 'PAT_ALL')
                                    zero = S.vdup(0.0, T.float32, full)
                                    denominator = S.vld(scalar_f_ub[f_kept_sum], dist='BRC_B32')
                                    for vector in T.serial(vregs_per_chunk):
                                        col = base + vector * vector_lanes
                                        indices = S.vci(T.int32(col), T.int32)
                                        valid = S.vcmps(indices, vocab_size, full, 'lt')
                                        probability = S.vsel(
                                            S.vdiv(S.vld(probs_ub[slot, vector * vector_lanes]), denominator, full),
                                            zero,
                                            valid,
                                        )
                                        S.vsts(
                                            probs_ub[slot, vector * vector_lanes],
                                            probability,
                                            full,
                                        )
                                        S.vsts(
                                            group_sum_ub[chunk * vregs_per_chunk + vector],
                                            S.vcadd(probability, full),
                                            full,
                                            'ONEPT_B32',
                                        )
                                if not sample_ub_fastpath:
                                    T.ascend_set_flag('V_MTE2', slot)
                                    if chunk % 2 == 0:
                                        T.ascend_set_flag('V_MTE3', 0)
                                        T.ascend_wait_flag('V_MTE3', 0)
                                    else:
                                        T.ascend_set_flag('V_MTE3', 1)
                                        T.ascend_wait_flag('V_MTE3', 1)
                                    T.copy(
                                        probs_ub[slot, :valid_count],
                                        logprobs[pid, base : base + valid_count],
                                    )
                                    if chunk % 2 == 0:
                                        T.ascend_set_flag('MTE3_S', 0)
                                        T.ascend_set_flag('MTE3_V', 0)
                                    else:
                                        T.ascend_set_flag('MTE3_S', 1)
                                        T.ascend_set_flag('MTE3_V', 1)
                            if not sample_ub_fastpath:
                                T.ascend_wait_flag('MTE3_S', 0)
                                if num_chunks > 1:
                                    T.ascend_wait_flag('MTE3_S', 1)
                                drain_input_db()
                                drain_output_db()
                            with T.SimdVF():
                                full = S.pset(32, 'PAT_ALL')
                                total = S.alloc_var(T.float32)
                                total = S.vdup(0.0, T.float32, full)
                                for vector in T.serial(num_vectors_aligned // vector_lanes):
                                    total = S.vadd(total, S.vld(group_sum_ub[vector * vector_lanes]), full)
                                S.vsts(
                                    scalar_f_ub[f_sample_total],
                                    S.vcadd(total, full),
                                    full,
                                    'ONEPT_B32',
                                )
                            sync_v_to_s()

                    if do_sample:
                        seed_val = T.alloc_var(T.uint64, seq_seeds[bx] + T.cast(seed_offset, T.uint64))
                        position = T.alloc_var(T.int32, sample_positions[bx] + sx * DRAFT_POS_HASH_MUL)
                        select_from_probability_groups(
                            logprobs,
                            pid,
                            seed_val,
                            position * 3,
                            vocab_size - 1,
                            probs_ub,
                            group_sum_ub,
                            scalar_f_ub,
                            scalar_i_ub,
                        )
                        sample_result[pid] = scalar_i_ub[i_sample]
                        if sample_ub_fastpath:
                            sync_s_to_v()

                    if do_reject_sampling and sx < reject_steps:
                        seed_val = T.alloc_var(T.uint64, seq_seeds[bx] + T.cast(seed_offset, T.uint64))
                        position = T.alloc_var(T.int32, sample_positions[bx] + sx * DRAFT_POS_HASH_MUL)
                        draft_tid = T.alloc_var(T.int32, last_next_n_token_ids[sx, bx])
                        if draft_tid < 0:
                            accept_result_verify[sx, bx] = False
                        else:
                            draft_lp = T.alloc_var(T.float32, last_nextn_token_logprobs[sx, bx, draft_tid])
                            target_prob = T.alloc_var(T.float32, logprobs[pid, draft_tid])
                            with T.SimtVF(threads=1):
                                accept_probability = T.alloc_var(
                                    T.float32,
                                    T.Select(
                                        draft_lp > 0.5,
                                        1.0,
                                        T.min(target_prob / T.exp(draft_lp), 1.0),
                                    ),
                                )
                                accept_result_verify[sx, bx] = (
                                    rng_uniform53(seed_val, position * 3 + 2) < accept_probability
                                )

                        build_residual_group_sums(
                            logprobs,
                            last_nextn_token_logprobs,
                            pid,
                            sx,
                            bx,
                            probs_ub,
                            draft_ub,
                            group_sum_ub,
                            scalar_f_ub,
                        )
                        with T.SimtVF(threads=1):
                            scalar_f_ub[f_random_target] = (
                                rng_uniform53(seed_val, position * 3 + 1) * scalar_f_ub[f_sample_total]
                            )
                        sync_v_to_s()

                        prefix = T.alloc_var(T.float32, 0.0)
                        found = T.alloc_var(T.bool, False)
                        scalar_i_ub[i_selected_vec] = num_vectors - 1
                        scalar_i_ub[i_sample] = 0
                        scalar_f_ub[f_group_prefix] = 0.0
                        for vector in T.serial(num_vectors):
                            next_prefix = T.alloc_var(T.float32, prefix + group_sum_ub[vector])
                            if (
                                not found
                                and scalar_f_ub[f_random_target] > prefix
                                and scalar_f_ub[f_random_target] <= next_prefix
                            ):
                                scalar_i_ub[i_selected_vec] = vector
                                scalar_f_ub[f_group_prefix] = prefix
                                found = True
                            prefix = next_prefix
                        if not found:
                            scalar_f_ub[f_group_prefix] = prefix - group_sum_ub[num_vectors - 1]

                        selected_base = T.alloc_var(T.int32, scalar_i_ub[i_selected_vec] * vector_lanes)
                        selected_count = T.alloc_var(T.int32, T.min(vector_lanes, vocab_size - selected_base))
                        T.copy(
                            logprobs[pid, selected_base : selected_base + selected_count],
                            probs_ub[0, :selected_count],
                        )
                        T.copy(
                            last_nextn_token_logprobs[sx, bx, selected_base : selected_base + selected_count],
                            draft_ub[0, :selected_count],
                        )
                        T.ascend_set_flag('MTE2_V', 0)
                        T.ascend_wait_flag('MTE2_V', 0)
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            target_prob_vec = S.vld(probs_ub[0, 0])
                            draft_prob_vec = S.vexp(
                                S.vadds(S.vld(draft_ub[0, 0]), -DRAFT_PROB_PENALTY, full),
                                full,
                            )
                            S.vsts(
                                work_ub[0],
                                S.vrelu(S.vsub(target_prob_vec, draft_prob_vec, full), full),
                                full,
                            )
                        sync_v_to_s()

                        group_target = T.alloc_var(
                            T.float32,
                            scalar_f_ub[f_random_target] - scalar_f_ub[f_group_prefix],
                        )
                        prefix = 0.0
                        found = False
                        for lane in T.serial(vector_lanes):
                            col = selected_base + lane
                            if col < vocab_size:
                                next_prefix = T.alloc_var(T.float32, prefix + work_ub[lane])
                                if (
                                    not found
                                    and scalar_f_ub[f_sample_total] > 0.0
                                    and group_target > prefix
                                    and group_target <= next_prefix
                                ):
                                    scalar_i_ub[i_sample] = col
                                    found = True
                                prefix = next_prefix
                        sample_reject_cur[sx, bx] = scalar_i_ub[i_sample]

                    # Public output is log probability. Use one final explicit
                    # denominator: plain top-p sampling kept unnormalized mass,
                    # while all paths that already normalized use one.
                    if not do_sample:
                        scalar_f_ub[f_sample_total] = scalar_f_ub[f_kept_sum]
                    elif not do_top_p or do_reject_sampling or top_p[bx] >= 1.0:
                        scalar_f_ub[f_sample_total] = 1.0
                    sync_s_to_v()

                    # vln(0) yields -inf for top-p-pruned entries.
                    if not sample_ub_fastpath:
                        init_input_db()
                    init_output_db()
                    for chunk in T.serial(num_chunks):
                        slot = chunk % 2
                        base = chunk * chunk_size
                        valid_count = T.min(chunk_size, vocab_size - base)
                        if chunk >= 2:
                            if chunk % 2 == 0:
                                T.ascend_wait_flag('MTE3_S', 0)
                            else:
                                T.ascend_wait_flag('MTE3_S', 1)
                        if chunk % 2 == 0:
                            T.ascend_wait_flag('MTE3_V', 0)
                        else:
                            T.ascend_wait_flag('MTE3_V', 1)
                        if not sample_ub_fastpath:
                            T.ascend_wait_flag('V_MTE2', slot)
                            T.copy(
                                logprobs[pid, base : base + valid_count],
                                probs_ub[slot, :valid_count],
                            )
                            T.ascend_set_flag('MTE2_V', slot)
                            T.ascend_wait_flag('MTE2_V', slot)
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            log_denominator = S.vln(
                                S.vld(scalar_f_ub[f_sample_total], dist='BRC_B32'),
                                full,
                            )
                            for vector in T.serial(vregs_per_chunk):
                                log_probability = S.vln(S.vld(probs_ub[slot, vector * vector_lanes]), full)
                                log_probability = S.vsub(log_probability, log_denominator, full)
                                S.vsts(
                                    probs_ub[slot, vector * vector_lanes],
                                    log_probability,
                                    full,
                                )
                        if not sample_ub_fastpath:
                            T.ascend_set_flag('V_MTE2', slot)
                        if chunk % 2 == 0:
                            T.ascend_set_flag('V_MTE3', 0)
                            T.ascend_wait_flag('V_MTE3', 0)
                        else:
                            T.ascend_set_flag('V_MTE3', 1)
                            T.ascend_wait_flag('V_MTE3', 1)
                        T.copy(
                            probs_ub[slot, :valid_count],
                            logprobs[pid, base : base + valid_count],
                        )
                        if chunk % 2 == 0:
                            T.ascend_set_flag('MTE3_S', 0)
                            T.ascend_set_flag('MTE3_V', 0)
                        else:
                            T.ascend_set_flag('MTE3_S', 1)
                            T.ascend_set_flag('MTE3_V', 1)
                    T.ascend_wait_flag('MTE3_S', 0)
                    if num_chunks > 1:
                        T.ascend_wait_flag('MTE3_S', 1)
                    if not sample_ub_fastpath:
                        drain_input_db()
                    drain_output_db()
                else:
                    init_output_db()
                    for chunk in T.serial(num_chunks):
                        slot = chunk % 2
                        base = chunk * chunk_size
                        valid_count = T.min(chunk_size, vocab_size - base)
                        T.ascend_wait_flag('MTE3_V', slot)
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            zero = S.vdup(0.0, T.float32, full)
                            neg_inf = S.vdup(-T.infinity(T.float32), T.float32, full)
                            for vector in T.serial(vregs_per_chunk):
                                indices = S.vci(T.int32(base + vector * vector_lanes), T.int32)
                                is_zero = S.vcmps(indices, 0, full, 'eq')
                                S.vsts(
                                    probs_ub[slot, vector * vector_lanes],
                                    S.vsel(zero, neg_inf, is_zero),
                                    full,
                                )
                        T.ascend_set_flag('V_MTE3', slot)
                        T.ascend_wait_flag('V_MTE3', slot)
                        T.copy(
                            probs_ub[slot, :valid_count],
                            logprobs[pid, base : base + valid_count],
                        )
                        T.ascend_set_flag('MTE3_V', slot)
                    drain_output_db()
                    if do_sample:
                        sample_result[pid] = 0
                    if do_reject_sampling and sx < reject_steps:
                        accept_result_verify[sx, bx] = True
                        sample_reject_cur[sx, bx] = 0

    return _fused_sample


@tilelang.jit(
    out_idx=None,
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
    },
)
def _fused_sample_tl_ascend_inrow(
    vocab_size: int,
    do_reject_sampling: bool,
    groups_per_row: int,
    num_cores: int,
) -> Any:
    """Large-vocabulary path with multiple AIV cores cooperating per row."""

    req_cnt = T.dynamic('req_cnt')
    total_q = T.dynamic('total_q')
    num_blocks = T.dynamic('num_blocks')
    logits_stride = T.dynamic('logits_stride')
    p_stride = T.dynamic('p_stride')
    temp_stride = T.dynamic('temp_stride')
    logprobs_stride = T.dynamic('logprobs_stride')
    reject_steps = T.dynamic('reject_steps')
    last_nextn_llp_stride0 = T.dynamic('last_nextn_llp_stride0')
    last_nextn_llp_stride1 = T.dynamic('last_nextn_llp_stride1')

    vector_lanes = 64
    chunk_size = 4096
    vregs_per_chunk = chunk_size // vector_lanes
    num_chunks = (vocab_size + chunk_size - 1) // chunk_size
    segment_size = (
        ((vocab_size + groups_per_row - 1) // groups_per_row + vector_lanes - 1) // vector_lanes * vector_lanes
    )
    num_segment_chunks = (segment_size + chunk_size - 1) // chunk_size
    num_segment_vectors = (segment_size + vector_lanes - 1) // vector_lanes
    num_segment_vectors_aligned = ((num_segment_vectors + vector_lanes - 1) // vector_lanes) * vector_lanes
    init_rounds = (num_blocks * num_chunks + num_cores - 1) // num_cores
    top_p_search_rounds = 15
    inv_2pow53 = 1.1102230246251565e-16

    # Per-row control workspace slots.
    rf_max = 0
    rf_sum = 1
    rf_target = 2
    rf_threshold = 3
    rf_kept_sum = 4
    rf_random_target = 5
    rf_group_prefix = 6
    ri_lo_bits = 0
    ri_hi_bits = 1
    ri_cut_group = 2
    ri_cut_remaining = 3
    ri_cut = 4
    ri_sample_group = 5

    # Per-(row, segment) partial workspace slots.
    pf_max = 0
    pf_sum = 1
    pf_alt_mass1 = 5
    pf_alt_mass2 = 6
    pf_alt_mass3 = 7
    pf_residual = 5

    # Core-local UB control slots used by sampling.
    lf_sample_total = 0
    lf_random_target = 1
    lf_group_prefix = 2
    lf_threshold = 3
    lf_kept_sum = 4
    li_selected_vec = 0
    li_sample = 1

    @T.macro
    def sync_all() -> None:
        # A5 AIV-only hard SYNCALL: drain every pipe, publish FFTS event 14,
        # then wait for the device-side barrier completion.
        T.ascend_pipe_barrier('PIPE_ALL')
        T.ascend_sync_inter_arrive('PIPE_MTE3', 14)
        T.ascend_sync_inter_wait('PIPE_S', 14)

    @T.macro
    def init_input_db() -> None:
        T.ascend_set_flag('V_MTE2', 0)
        T.ascend_set_flag('V_MTE2', 1)

    @T.macro
    def drain_input_db() -> None:
        T.ascend_wait_flag('V_MTE2', 0)
        T.ascend_wait_flag('V_MTE2', 1)

    @T.macro
    def init_output_db() -> None:
        T.ascend_set_flag('MTE3_V', 0)
        T.ascend_set_flag('MTE3_V', 1)

    @T.macro
    def drain_output_db() -> None:
        T.ascend_wait_flag('MTE3_V', 0)
        T.ascend_wait_flag('MTE3_V', 1)
        T.ascend_set_flag('MTE3_S', 0)
        T.ascend_wait_flag('MTE3_S', 0)
        T.ascend_set_flag('V_S', 0)
        T.ascend_wait_flag('V_S', 0)

    @T.macro
    def sync_v_to_s() -> None:
        T.ascend_set_flag('V_S', 0)
        T.ascend_wait_flag('V_S', 0)

    @T.macro
    def sync_s_to_v() -> None:
        T.ascend_set_flag('S_V', 0)
        T.ascend_wait_flag('S_V', 0)

    @T.macro
    def rng_uniform53(seed: T.uint64, sequence: T.int32) -> T.float32:  # type: ignore
        T.rng_init(seed, sequence)
        counter0 = T.alloc_var(T.uint64, T.cast(T.rng_rand(), T.uint64))
        counter1 = T.alloc_var(T.uint64, T.cast(T.rng_rand(), T.uint64))
        random53 = T.alloc_var(T.uint64, counter0 ^ (counter1 << 21))
        return T.cast(random53, T.float32) * T.float32(inv_2pow53) + T.float32(inv_2pow53 / 2.0)

    @T.prim_func
    def _fused_sample(
        logits: T.StridedTensor((total_q, vocab_size), (logits_stride, 1), T.float32),  # type: ignore
        temperatures: T.StridedTensor((req_cnt,), (temp_stride,), T.float32),  # type: ignore
        top_p: T.StridedTensor((req_cnt,), (p_stride,), T.float32),  # type: ignore
        ftp: T.Tensor((req_cnt,), T.int32),  # type: ignore
        sample_positions: T.Tensor((req_cnt,), T.int32),  # type: ignore
        seq_seeds: T.Tensor((req_cnt,), T.uint64),  # type: ignore
        logprobs: T.StridedTensor((num_blocks, vocab_size), (logprobs_stride, 1), T.float32),  # type: ignore
        sample_result: T.Tensor((num_blocks,), T.int32),  # type: ignore
        last_nextn_token_logprobs: T.StridedTensor(
            (reject_steps, req_cnt, vocab_size),
            (last_nextn_llp_stride0, last_nextn_llp_stride1, 1),
            T.float32,
        ),  # type: ignore
        last_next_n_token_ids: T.Tensor((reject_steps, req_cnt), T.int32),  # type: ignore
        accept_result_verify: T.Tensor((reject_steps, req_cnt), T.bool),  # type: ignore
        sample_reject_cur: T.Tensor((reject_steps, req_cnt), T.int32),  # type: ignore
        partial_f: T.Tensor((total_q, groups_per_row, 8), T.float32),  # type: ignore
        search_f: T.Tensor((total_q, 2, groups_per_row * 3), T.float32),  # type: ignore
        row_f: T.Tensor((total_q, 8), T.float32),  # type: ignore
        row_i: T.Tensor((total_q, 8), T.int32),  # type: ignore
        seed_offset: T.int32,
    ) -> None:
        with T.Kernel(num_cores) as core_id:
            packed_row = T.alloc_var(T.int32, core_id // groups_per_row)
            segment_id = T.alloc_var(T.int32, core_id % groups_per_row)
            segment_base = T.alloc_var(T.int32, segment_id * segment_size)
            segment_end = T.alloc_var(T.int32, T.min(segment_base + segment_size, vocab_size))

            bx = T.alloc_var(T.int32, 0)
            for request in T.serial(req_cnt):
                if packed_row >= ftp[request]:
                    bx = request
            sx = T.alloc_var(T.int32, packed_row - ftp[bx])
            pid = T.alloc_var(T.int32, sx * req_cnt + bx)

            logits_ub = T.alloc_shared((2, chunk_size), T.float32)
            probs_ub = T.alloc_shared((2, chunk_size), T.float32)
            draft_ub = T.alloc_shared((2, chunk_size), T.float32)
            work_ub = T.alloc_shared((chunk_size,), T.float32)
            group_sum_ub = T.alloc_shared((num_segment_vectors_aligned,), T.float32)
            acc1_ub = T.alloc_shared((vector_lanes,), T.float32)
            acc2_ub = T.alloc_shared((vector_lanes,), T.float32)
            acc3_ub = T.alloc_shared((vector_lanes,), T.float32)
            scalar_f_ub = T.alloc_shared((vector_lanes,), T.float32)
            scalar_i_ub = T.alloc_shared((vector_lanes,), T.int32)

            # Initialize padding outputs while all cores are available. Valid
            # packed rows overwrite their dense output positions below.
            init_output_db()
            for init_round in T.serial(init_rounds):
                slot = init_round % 2
                init_task = init_round * num_cores + core_id
                if init_task < num_blocks * num_chunks:
                    init_row = init_task // num_chunks
                    init_chunk = init_task % num_chunks
                    init_base = init_chunk * chunk_size
                    valid_count = T.min(chunk_size, vocab_size - init_base)
                    T.ascend_wait_flag('MTE3_V', slot)
                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        zero = S.vdup(0.0, T.float32, full)
                        neg_inf = S.vdup(-T.infinity(T.float32), T.float32, full)
                        for vector in T.serial(vregs_per_chunk):
                            indices = S.vci(T.int32(init_base + vector * vector_lanes), T.int32)
                            is_token_zero = S.vcmps(indices, 0, full, 'eq')
                            S.vsts(probs_ub[slot, vector * vector_lanes], S.vsel(zero, neg_inf, is_token_zero), full)
                    T.ascend_set_flag('V_MTE3', slot)
                    T.ascend_wait_flag('V_MTE3', slot)
                    T.copy(probs_ub[slot, :valid_count], logprobs[init_row, init_base : init_base + valid_count])
                    T.ascend_set_flag('MTE3_V', slot)
            drain_output_db()
            if core_id < num_blocks:
                sample_result[core_id] = 0
            if do_reject_sampling and core_id < reject_steps * req_cnt:
                accept_result_verify[core_id // req_cnt, core_id % req_cnt] = True
                sample_reject_cur[core_id // req_cnt, core_id % req_cnt] = 0
            sync_all()

            if segment_id == 0:
                T.device_assert(temperatures[bx] > 0, msg='temperatures must be positive')
                T.device_assert(top_p[bx] > 0 and top_p[bx] <= 1, msg='top_p must be in (0, 1]')

            # Local maximum over this vocabulary segment.
            with T.SimdVF():
                S.vsts(acc1_ub[0], S.vdup(-T.infinity(T.float32), T.float32))
            init_input_db()
            for chunk in T.serial(num_segment_chunks):
                slot = chunk % 2
                base = segment_base + chunk * chunk_size
                valid_count = T.max(T.min(chunk_size, segment_end - base), 0)
                if valid_count > 0:
                    T.ascend_wait_flag('V_MTE2', slot)
                    T.copy(logits[packed_row, base : base + valid_count], logits_ub[slot, :valid_count])
                    T.ascend_set_flag('MTE2_V', slot)
                    T.ascend_wait_flag('MTE2_V', slot)
                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        neg_inf = S.vdup(-T.infinity(T.float32), T.float32, full)
                        local_max = S.alloc_var(T.float32)
                        local_max = S.vld(acc1_ub[0])
                        for vector in T.serial(vregs_per_chunk):
                            indices = S.vci(T.int32(base + vector * vector_lanes), T.int32)
                            valid = S.vcmps(indices, segment_end, full, 'lt')
                            values = S.vsel(S.vld(logits_ub[slot, vector * vector_lanes]), neg_inf, valid)
                            local_max = S.vmax(local_max, values, full)
                        S.vsts(acc1_ub[0], local_max, full)
                    T.ascend_set_flag('V_MTE2', slot)
            drain_input_db()
            with T.SimdVF():
                full = S.pset(32, 'PAT_ALL')
                S.vsts(scalar_f_ub[0], S.vcmax(S.vld(acc1_ub[0]), full), full, 'ONEPT_B32')
            sync_v_to_s()
            partial_f[packed_row, segment_id, pf_max] = scalar_f_ub[0]
            sync_all()
            if segment_id == 0:
                row_max = T.alloc_var(T.float32, -T.infinity(T.float32))
                for group in T.serial(groups_per_row):
                    row_max = T.max(row_max, partial_f[packed_row, group, pf_max])
                row_f[packed_row, rf_max] = row_max
            sync_all()

            # Write unnormalized softmax probabilities and reduce their sum.
            scalar_f_ub[0] = row_f[packed_row, rf_max]
            scalar_f_ub[1] = 1.0 / temperatures[bx]
            sync_s_to_v()
            with T.SimdVF():
                S.vsts(acc1_ub[0], S.vdup(0.0, T.float32))
            init_input_db()
            init_output_db()
            for chunk in T.serial(num_segment_chunks):
                slot = chunk % 2
                base = segment_base + chunk * chunk_size
                valid_count = T.max(T.min(chunk_size, segment_end - base), 0)
                if valid_count > 0:
                    T.ascend_wait_flag('V_MTE2', slot)
                    T.ascend_wait_flag('MTE3_V', slot)
                    T.copy(logits[packed_row, base : base + valid_count], logits_ub[slot, :valid_count])
                    T.ascend_set_flag('MTE2_V', slot)
                    T.ascend_wait_flag('MTE2_V', slot)
                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        zero = S.vdup(0.0, T.float32, full)
                        max_vec = S.vld(scalar_f_ub[0], dist='BRC_B32')
                        inv_temp = S.vld(scalar_f_ub[1], dist='BRC_B32')
                        local_sum = S.alloc_var(T.float32)
                        local_sum = S.vld(acc1_ub[0])
                        for vector in T.serial(vregs_per_chunk):
                            indices = S.vci(T.int32(base + vector * vector_lanes), T.int32)
                            valid = S.vcmps(indices, segment_end, full, 'lt')
                            values = S.vmul(
                                S.vsub(S.vld(logits_ub[slot, vector * vector_lanes]), max_vec, full), inv_temp, full
                            )
                            probabilities = S.vsel(S.vexp(values, full), zero, valid)
                            S.vsts(probs_ub[slot, vector * vector_lanes], probabilities, full)
                            local_sum = S.vadd(local_sum, probabilities, full)
                        S.vsts(acc1_ub[0], local_sum, full)
                    T.ascend_set_flag('V_MTE2', slot)
                    T.ascend_set_flag('V_MTE3', slot)
                    T.ascend_wait_flag('V_MTE3', slot)
                    T.copy(probs_ub[slot, :valid_count], logprobs[pid, base : base + valid_count])
                    T.ascend_set_flag('MTE3_V', slot)
            drain_input_db()
            drain_output_db()
            with T.SimdVF():
                full = S.pset(32, 'PAT_ALL')
                S.vsts(scalar_f_ub[0], S.vcadd(S.vld(acc1_ub[0]), full), full, 'ONEPT_B32')
            sync_v_to_s()
            partial_f[packed_row, segment_id, pf_sum] = scalar_f_ub[0]
            sync_all()
            if segment_id == 0:
                row_sum = T.alloc_var(T.float32, 0.0)
                for group in T.serial(groups_per_row):
                    row_sum += partial_f[packed_row, group, pf_sum]
                row_f[packed_row, rf_sum] = row_sum
                row_f[packed_row, rf_kept_sum] = row_sum
                row_f[packed_row, rf_target] = top_p[bx] * row_sum
                row_i[packed_row, ri_lo_bits] = 0
                row_i[packed_row, ri_hi_bits] = 0x40000000
                row_i[packed_row, ri_cut] = vocab_size - 1
            sync_all()

            # The AscendC recipe uses the same 4-way radix search, but splits
            # each row over all available AIV cores and synchronizes partials.
            lo_bits = T.alloc_var(T.uint32, 0)
            hi_bits = T.alloc_var(T.uint32, 0x40000000)
            for _search_round in T.serial(top_p_search_rounds):
                span = T.alloc_var(T.uint32, hi_bits - lo_bits)
                q1_bits = T.alloc_var(T.uint32, lo_bits + span // 4)
                q2_bits = T.alloc_var(T.uint32, lo_bits + span // 2)
                q3_bits = T.alloc_var(T.uint32, lo_bits + (span // 4) * 3)
                q1 = T.alloc_var(T.float32, T.reinterpret(T.float32, q1_bits))
                q2 = T.alloc_var(T.float32, T.reinterpret(T.float32, q2_bits))
                q3 = T.alloc_var(T.float32, T.reinterpret(T.float32, q3_bits))
                with T.SimdVF():
                    zero = S.vdup(0.0, T.float32)
                    S.vsts(acc1_ub[0], zero)
                    S.vsts(acc2_ub[0], zero)
                    S.vsts(acc3_ub[0], zero)
                if top_p[bx] < 1.0:
                    init_input_db()
                    for chunk in T.serial(num_segment_chunks):
                        slot = chunk % 2
                        base = segment_base + chunk * chunk_size
                        valid_count = T.max(T.min(chunk_size, segment_end - base), 0)
                        if valid_count > 0:
                            T.ascend_wait_flag('V_MTE2', slot)
                            T.copy(logprobs[pid, base : base + valid_count], probs_ub[slot, :valid_count])
                            T.ascend_set_flag('MTE2_V', slot)
                            T.ascend_wait_flag('MTE2_V', slot)
                            with T.SimdVF():
                                full = S.pset(32, 'PAT_ALL')
                                zero = S.vdup(0.0, T.float32, full)
                                mass1 = S.alloc_var(T.float32)
                                mass2 = S.alloc_var(T.float32)
                                mass3 = S.alloc_var(T.float32)
                                mass1 = S.vld(acc1_ub[0])
                                mass2 = S.vld(acc2_ub[0])
                                mass3 = S.vld(acc3_ub[0])
                                for vector in T.serial(vregs_per_chunk):
                                    indices = S.vci(T.int32(base + vector * vector_lanes), T.int32)
                                    valid = S.vcmps(indices, segment_end, full, 'lt')
                                    probability = S.vsel(S.vld(probs_ub[slot, vector * vector_lanes]), zero, valid)
                                    mass1 = S.vadd(
                                        mass1,
                                        S.vsel(probability, zero, S.vcmps(probability, q1, full, 'ge')),
                                        full,
                                    )
                                    mass2 = S.vadd(
                                        mass2,
                                        S.vsel(probability, zero, S.vcmps(probability, q2, full, 'ge')),
                                        full,
                                    )
                                    mass3 = S.vadd(
                                        mass3,
                                        S.vsel(probability, zero, S.vcmps(probability, q3, full, 'ge')),
                                        full,
                                    )
                                S.vsts(acc1_ub[0], mass1, full)
                                S.vsts(acc2_ub[0], mass2, full)
                                S.vsts(acc3_ub[0], mass3, full)
                            T.ascend_set_flag('V_MTE2', slot)
                    drain_input_db()
                with T.SimdVF():
                    full = S.pset(32, 'PAT_ALL')
                    S.vsts(scalar_f_ub[0], S.vcadd(S.vld(acc1_ub[0]), full), full, 'ONEPT_B32')
                    S.vsts(scalar_f_ub[1], S.vcadd(S.vld(acc2_ub[0]), full), full, 'ONEPT_B32')
                    S.vsts(scalar_f_ub[2], S.vcadd(S.vld(acc3_ub[0]), full), full, 'ONEPT_B32')
                search_bank = _search_round % 2
                T.ascend_set_flag('V_MTE3', 0)
                T.ascend_wait_flag('V_MTE3', 0)
                T.copy(
                    scalar_f_ub[:3],
                    search_f[
                        packed_row,
                        search_bank,
                        segment_id * 3 : segment_id * 3 + 3,
                    ],
                )
                T.ascend_set_flag('MTE3_S', 0)
                T.ascend_wait_flag('MTE3_S', 0)
                sync_all()
                if top_p[bx] < 1.0:
                    T.copy(
                        search_f[packed_row, search_bank, : groups_per_row * 3],
                        work_ub[: groups_per_row * 3],
                    )
                    T.ascend_set_flag('MTE2_S', 0)
                    T.ascend_wait_flag('MTE2_S', 0)
                    mass1 = T.alloc_var(T.float32, 0.0)
                    mass2 = T.alloc_var(T.float32, 0.0)
                    mass3 = T.alloc_var(T.float32, 0.0)
                    for group in T.serial(groups_per_row):
                        mass1 += work_ub[group * 3]
                        mass2 += work_ub[group * 3 + 1]
                        mass3 += work_ub[group * 3 + 2]
                    if mass3 >= row_f[packed_row, rf_target]:
                        lo_bits = q3_bits
                    else:
                        if mass2 >= row_f[packed_row, rf_target]:
                            lo_bits = q2_bits
                            hi_bits = q3_bits
                        else:
                            if mass1 >= row_f[packed_row, rf_target]:
                                lo_bits = q1_bits
                                hi_bits = q2_bits
                            else:
                                hi_bits = q1_bits

            # Reduce exact cutoff masses and locate the segment containing the
            # stable token-id tie boundary.
            scalar_f_ub[lf_threshold] = T.reinterpret(T.float32, lo_bits)
            sync_s_to_v()
            with T.SimdVF():
                zero = S.vdup(0.0, T.float32)
                S.vsts(acc1_ub[0], zero)
                S.vsts(acc2_ub[0], zero)
                S.vsts(acc3_ub[0], zero)
            if top_p[bx] < 1.0:
                init_input_db()
                for chunk in T.serial(num_segment_chunks):
                    slot = chunk % 2
                    base = segment_base + chunk * chunk_size
                    valid_count = T.max(T.min(chunk_size, segment_end - base), 0)
                    if valid_count > 0:
                        T.ascend_wait_flag('V_MTE2', slot)
                        T.copy(logprobs[pid, base : base + valid_count], probs_ub[slot, :valid_count])
                        T.ascend_set_flag('MTE2_V', slot)
                        T.ascend_wait_flag('MTE2_V', slot)
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            zero = S.vdup(0.0, T.float32, full)
                            one = S.vdup(1.0, T.float32, full)
                            mass_ge = S.alloc_var(T.float32)
                            mass_gt = S.alloc_var(T.float32)
                            equal_count = S.alloc_var(T.float32)
                            mass_ge = S.vld(acc1_ub[0])
                            mass_gt = S.vld(acc2_ub[0])
                            equal_count = S.vld(acc3_ub[0])
                            threshold_vec = S.vld(scalar_f_ub[lf_threshold], dist='BRC_B32')
                            for vector in T.serial(vregs_per_chunk):
                                indices = S.vci(T.int32(base + vector * vector_lanes), T.int32)
                                valid = S.vcmps(indices, segment_end, full, 'lt')
                                probability = S.vsel(S.vld(probs_ub[slot, vector * vector_lanes]), zero, valid)
                                mass_ge = S.vadd(
                                    mass_ge,
                                    S.vsel(probability, zero, S.vcmp(probability, threshold_vec, full, 'ge')),
                                    full,
                                )
                                mass_gt = S.vadd(
                                    mass_gt,
                                    S.vsel(probability, zero, S.vcmp(probability, threshold_vec, full, 'gt')),
                                    full,
                                )
                                equal = S.pand(S.vcmp(probability, threshold_vec, full, 'eq'), valid, full)
                                equal_count = S.vadd(equal_count, S.vsel(one, zero, equal), full)
                            S.vsts(acc1_ub[0], mass_ge, full)
                            S.vsts(acc2_ub[0], mass_gt, full)
                            S.vsts(acc3_ub[0], equal_count, full)
                        T.ascend_set_flag('V_MTE2', slot)
                drain_input_db()
            with T.SimdVF():
                full = S.pset(32, 'PAT_ALL')
                S.vsts(scalar_f_ub[0], S.vcadd(S.vld(acc1_ub[0]), full), full, 'ONEPT_B32')
                S.vsts(scalar_f_ub[1], S.vcadd(S.vld(acc2_ub[0]), full), full, 'ONEPT_B32')
                S.vsts(scalar_f_ub[2], S.vcadd(S.vld(acc3_ub[0]), full), full, 'ONEPT_B32')
            sync_v_to_s()
            partial_f[packed_row, segment_id, pf_alt_mass1] = scalar_f_ub[0]
            partial_f[packed_row, segment_id, pf_alt_mass2] = scalar_f_ub[1]
            partial_f[packed_row, segment_id, pf_alt_mass3] = scalar_f_ub[2]
            sync_all()
            if segment_id == 0 and top_p[bx] < 1.0:
                mass_ge = T.alloc_var(T.float32, 0.0)
                mass_gt = T.alloc_var(T.float32, 0.0)
                for group in T.serial(groups_per_row):
                    mass_ge += partial_f[packed_row, group, pf_alt_mass1]
                    mass_gt += partial_f[packed_row, group, pf_alt_mass2]
                threshold = T.alloc_var(T.float32, scalar_f_ub[lf_threshold])
                equal_ratio = T.alloc_var(
                    T.float32,
                    T.Select(threshold > 0, (row_f[packed_row, rf_target] - mass_gt) / threshold, 1.0),
                )
                selected_equal = T.alloc_var(T.int32, T.max(T.cast(equal_ratio, T.int32), 1))
                if T.cast(selected_equal, T.float32) < equal_ratio:
                    selected_equal += 1
                equal_total = T.alloc_var(
                    T.int32,
                    T.Select(
                        threshold > 0,
                        T.max(T.cast((mass_ge - mass_gt) / threshold + 0.5, T.int32), 1),
                        1,
                    ),
                )
                selected_equal = T.min(selected_equal, equal_total)
                remaining = T.alloc_var(T.int32, selected_equal)
                found_group = T.alloc_var(T.bool, False)
                row_i[packed_row, ri_cut_group] = groups_per_row - 1
                for group in T.serial(groups_per_row):
                    equal_count = T.alloc_var(T.int32, T.cast(partial_f[packed_row, group, pf_alt_mass3], T.int32))
                    if not found_group and remaining > 0 and remaining <= equal_count:
                        row_i[packed_row, ri_cut_group] = group
                        found_group = True
                    if not found_group and remaining > equal_count:
                        remaining -= equal_count
                row_i[packed_row, ri_cut_remaining] = remaining
                row_f[packed_row, rf_threshold] = threshold
            sync_all()

            if segment_id == row_i[packed_row, ri_cut_group] and top_p[bx] < 1.0:
                threshold = T.alloc_var(T.float32, row_f[packed_row, rf_threshold])
                with T.SimdVF():
                    zero = S.vdup(0.0, T.float32)
                    for vector in T.serial(num_segment_vectors_aligned // vector_lanes):
                        S.vsts(group_sum_ub[vector * vector_lanes], zero)
                init_input_db()
                for chunk in T.serial(num_segment_chunks):
                    slot = chunk % 2
                    base = segment_base + chunk * chunk_size
                    valid_count = T.max(T.min(chunk_size, segment_end - base), 0)
                    if valid_count > 0:
                        T.ascend_wait_flag('V_MTE2', slot)
                        T.copy(logprobs[pid, base : base + valid_count], probs_ub[slot, :valid_count])
                        T.ascend_set_flag('MTE2_V', slot)
                        T.ascend_wait_flag('MTE2_V', slot)
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            zero = S.vdup(0.0, T.float32, full)
                            one = S.vdup(1.0, T.float32, full)
                            for vector in T.serial(vregs_per_chunk):
                                indices = S.vci(T.int32(base + vector * vector_lanes), T.int32)
                                valid = S.vcmps(indices, segment_end, full, 'lt')
                                equal = S.pand(
                                    S.vcmps(S.vld(probs_ub[slot, vector * vector_lanes]), threshold, full, 'eq'),
                                    valid,
                                    full,
                                )
                                S.vsts(
                                    group_sum_ub[chunk * vregs_per_chunk + vector],
                                    S.vcadd(S.vsel(one, zero, equal), full),
                                    full,
                                    'ONEPT_B32',
                                )
                        T.ascend_set_flag('V_MTE2', slot)
                drain_input_db()
                sync_v_to_s()
                remaining = T.alloc_var(T.int32, row_i[packed_row, ri_cut_remaining])
                selected_vector = T.alloc_var(T.int32, num_segment_vectors - 1)
                found_vector = T.alloc_var(T.bool, False)
                for vector in T.serial(num_segment_vectors):
                    equal_count = T.alloc_var(T.int32, T.cast(group_sum_ub[vector], T.int32))
                    if not found_vector and remaining > 0 and remaining <= equal_count:
                        selected_vector = vector
                        found_vector = True
                    if not found_vector and remaining > equal_count:
                        remaining -= equal_count
                selected_base = T.alloc_var(T.int32, segment_base + selected_vector * vector_lanes)
                selected_count = T.alloc_var(T.int32, T.min(vector_lanes, segment_end - selected_base))
                T.copy(logprobs[pid, selected_base : selected_base + selected_count], probs_ub[0, :selected_count])
                T.ascend_set_flag('MTE2_S', 0)
                T.ascend_wait_flag('MTE2_S', 0)
                for lane in T.serial(vector_lanes):
                    if lane < selected_count and remaining > 0 and probs_ub[0, lane] == threshold:
                        remaining -= 1
                        if remaining == 0:
                            row_i[packed_row, ri_cut] = selected_base + lane
            sync_all()

            # Apply the exact tie cutoff and reduce kept mass. Rows with p=1
            # already have the correct unfiltered probabilities in GM.
            with T.SimdVF():
                S.vsts(acc1_ub[0], S.vdup(0.0, T.float32))
            if top_p[bx] < 1.0:
                scalar_f_ub[lf_threshold] = row_f[packed_row, rf_threshold]
                scalar_i_ub[0] = row_i[packed_row, ri_cut]
                sync_s_to_v()
                init_input_db()
                init_output_db()
                for chunk in T.serial(num_segment_chunks):
                    slot = chunk % 2
                    base = segment_base + chunk * chunk_size
                    valid_count = T.max(T.min(chunk_size, segment_end - base), 0)
                    if valid_count > 0:
                        T.ascend_wait_flag('V_MTE2', slot)
                        T.ascend_wait_flag('MTE3_V', slot)
                        T.copy(logprobs[pid, base : base + valid_count], probs_ub[slot, :valid_count])
                        T.ascend_set_flag('MTE2_V', slot)
                        T.ascend_wait_flag('MTE2_V', slot)
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            zero = S.vdup(0.0, T.float32, full)
                            threshold_vec = S.vld(scalar_f_ub[lf_threshold], dist='BRC_B32')
                            kept_sum = S.alloc_var(T.float32)
                            kept_sum = S.vld(acc1_ub[0])
                            for vector in T.serial(vregs_per_chunk):
                                indices = S.vci(T.int32(base + vector * vector_lanes), T.int32)
                                valid = S.vcmps(indices, segment_end, full, 'lt')
                                probability = S.vsel(S.vld(probs_ub[slot, vector * vector_lanes]), zero, valid)
                                greater = S.vcmp(probability, threshold_vec, full, 'gt')
                                equal = S.vcmp(probability, threshold_vec, full, 'eq')
                                before_cut = S.vcmps(indices, scalar_i_ub[0], full, 'le')
                                keep = S.por(greater, S.pand(equal, before_cut, full), full)
                                probability = S.vsel(probability, zero, keep)
                                S.vsts(probs_ub[slot, vector * vector_lanes], probability, full)
                                kept_sum = S.vadd(kept_sum, probability, full)
                            S.vsts(acc1_ub[0], kept_sum, full)
                        T.ascend_set_flag('V_MTE2', slot)
                        T.ascend_set_flag('V_MTE3', slot)
                        T.ascend_wait_flag('V_MTE3', slot)
                        T.copy(probs_ub[slot, :valid_count], logprobs[pid, base : base + valid_count])
                        T.ascend_set_flag('MTE3_V', slot)
                drain_input_db()
                drain_output_db()
                with T.SimdVF():
                    full = S.pset(32, 'PAT_ALL')
                    S.vsts(scalar_f_ub[0], S.vcadd(S.vld(acc1_ub[0]), full), full, 'ONEPT_B32')
                sync_v_to_s()
                partial_f[packed_row, segment_id, pf_sum] = scalar_f_ub[0]
            sync_all()
            if segment_id == 0 and top_p[bx] < 1.0:
                kept_sum = T.alloc_var(T.float32, 0.0)
                for group in T.serial(groups_per_row):
                    kept_sum += partial_f[packed_row, group, pf_sum]
                row_f[packed_row, rf_kept_sum] = kept_sum
            sync_all()

            # Select a vocabulary segment using its already reduced kept mass.
            # Only the selected segment scans local probabilities.
            if segment_id == 0:
                seed_val = T.alloc_var(T.uint64, seq_seeds[bx] + T.cast(seed_offset, T.uint64))
                position = T.alloc_var(T.int32, sample_positions[bx] + sx * DRAFT_POS_HASH_MUL)
                with T.SimtVF(threads=1):
                    scalar_f_ub[lf_random_target] = (
                        rng_uniform53(seed_val, position * 3) * row_f[packed_row, rf_kept_sum]
                    )
                sync_v_to_s()
                target = T.alloc_var(T.float32, scalar_f_ub[lf_random_target])
                prefix = T.alloc_var(T.float32, 0.0)
                found = T.alloc_var(T.bool, False)
                row_i[packed_row, ri_sample_group] = groups_per_row - 1
                row_f[packed_row, rf_group_prefix] = 0.0
                row_f[packed_row, rf_random_target] = target
                for group in T.serial(groups_per_row):
                    next_prefix = T.alloc_var(T.float32, prefix + partial_f[packed_row, group, pf_sum])
                    if not found and target > prefix and target <= next_prefix:
                        row_i[packed_row, ri_sample_group] = group
                        row_f[packed_row, rf_group_prefix] = prefix
                        found = True
                    prefix = next_prefix
            sync_all()

            if segment_id == row_i[packed_row, ri_sample_group]:
                with T.SimdVF():
                    zero = S.vdup(0.0, T.float32)
                    for vector in T.serial(num_segment_vectors_aligned // vector_lanes):
                        S.vsts(group_sum_ub[vector * vector_lanes], zero)
                init_input_db()
                for chunk in T.serial(num_segment_chunks):
                    slot = chunk % 2
                    base = segment_base + chunk * chunk_size
                    valid_count = T.max(T.min(chunk_size, segment_end - base), 0)
                    if valid_count > 0:
                        T.ascend_wait_flag('V_MTE2', slot)
                        T.copy(logprobs[pid, base : base + valid_count], probs_ub[slot, :valid_count])
                        T.ascend_set_flag('MTE2_V', slot)
                        T.ascend_wait_flag('MTE2_V', slot)
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            zero = S.vdup(0.0, T.float32, full)
                            for vector in T.serial(vregs_per_chunk):
                                indices = S.vci(T.int32(base + vector * vector_lanes), T.int32)
                                valid = S.vcmps(indices, segment_end, full, 'lt')
                                values = S.vsel(S.vld(probs_ub[slot, vector * vector_lanes]), zero, valid)
                                S.vsts(
                                    group_sum_ub[chunk * vregs_per_chunk + vector],
                                    S.vcadd(values, full),
                                    full,
                                    'ONEPT_B32',
                                )
                        T.ascend_set_flag('V_MTE2', slot)
                drain_input_db()
                sync_v_to_s()
                group_target = T.alloc_var(
                    T.float32,
                    row_f[packed_row, rf_random_target] - row_f[packed_row, rf_group_prefix],
                )
                prefix = T.alloc_var(T.float32, 0.0)
                found = T.alloc_var(T.bool, False)
                scalar_i_ub[li_selected_vec] = num_segment_vectors - 1
                scalar_f_ub[lf_group_prefix] = 0.0
                for vector in T.serial(num_segment_vectors):
                    next_prefix = T.alloc_var(T.float32, prefix + group_sum_ub[vector])
                    if not found and group_target > prefix and group_target <= next_prefix:
                        scalar_i_ub[li_selected_vec] = vector
                        scalar_f_ub[lf_group_prefix] = prefix
                        found = True
                    prefix = next_prefix
                selected_base = T.alloc_var(T.int32, segment_base + scalar_i_ub[li_selected_vec] * vector_lanes)
                selected_count = T.alloc_var(T.int32, T.min(vector_lanes, segment_end - selected_base))
                T.copy(logprobs[pid, selected_base : selected_base + selected_count], probs_ub[0, :selected_count])
                T.ascend_set_flag('MTE2_S', 0)
                T.ascend_wait_flag('MTE2_S', 0)
                lane_target = T.alloc_var(T.float32, group_target - scalar_f_ub[lf_group_prefix])
                prefix = 0.0
                found = False
                scalar_i_ub[li_sample] = T.min(selected_base, vocab_size - 1)
                for lane in T.serial(vector_lanes):
                    col = selected_base + lane
                    if col < segment_end:
                        next_prefix = T.alloc_var(T.float32, prefix + probs_ub[0, lane])
                        if not found and lane_target > prefix and lane_target <= next_prefix:
                            scalar_i_ub[li_sample] = col
                            found = True
                        prefix = next_prefix
                sample_result[pid] = scalar_i_ub[li_sample]
            sync_all()

            if do_reject_sampling:
                # Build residual masses on all row segments concurrently. Keep
                # global barriers outside the ragged-row validity branches.
                if sx < reject_steps:
                    scalar_f_ub[lf_kept_sum] = row_f[packed_row, rf_kept_sum]
                    sync_s_to_v()
                    with T.SimdVF():
                        zero = S.vdup(0.0, T.float32)
                        for vector in T.serial(num_segment_vectors_aligned // vector_lanes):
                            S.vsts(group_sum_ub[vector * vector_lanes], zero)
                    init_input_db()
                    for chunk in T.serial(num_segment_chunks):
                        slot = chunk % 2
                        base = segment_base + chunk * chunk_size
                        valid_count = T.max(T.min(chunk_size, segment_end - base), 0)
                        if valid_count > 0:
                            T.ascend_wait_flag('V_MTE2', slot)
                            T.copy(logprobs[pid, base : base + valid_count], probs_ub[slot, :valid_count])
                            T.copy(
                                last_nextn_token_logprobs[sx, bx, base : base + valid_count],
                                draft_ub[slot, :valid_count],
                            )
                            T.ascend_set_flag('MTE2_V', slot)
                            T.ascend_wait_flag('MTE2_V', slot)
                            with T.SimdVF():
                                full = S.pset(32, 'PAT_ALL')
                                zero = S.vdup(0.0, T.float32, full)
                                denominator = S.vld(scalar_f_ub[lf_kept_sum], dist='BRC_B32')
                                for vector in T.serial(vregs_per_chunk):
                                    indices = S.vci(T.int32(base + vector * vector_lanes), T.int32)
                                    valid = S.vcmps(indices, segment_end, full, 'lt')
                                    target_prob = S.vdiv(
                                        S.vld(probs_ub[slot, vector * vector_lanes]), denominator, full
                                    )
                                    draft_prob = S.vexp(
                                        S.vadds(
                                            S.vld(draft_ub[slot, vector * vector_lanes]),
                                            -DRAFT_PROB_PENALTY,
                                            full,
                                        ),
                                        full,
                                    )
                                    residual = S.vsel(
                                        S.vrelu(S.vsub(target_prob, draft_prob, full), full),
                                        zero,
                                        valid,
                                    )
                                    S.vsts(
                                        group_sum_ub[chunk * vregs_per_chunk + vector],
                                        S.vcadd(residual, full),
                                        full,
                                        'ONEPT_B32',
                                    )
                            T.ascend_set_flag('V_MTE2', slot)
                    drain_input_db()
                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        total = S.alloc_var(T.float32)
                        total = S.vdup(0.0, T.float32, full)
                        for vector in T.serial(num_segment_vectors_aligned // vector_lanes):
                            total = S.vadd(total, S.vld(group_sum_ub[vector * vector_lanes]), full)
                        S.vsts(scalar_f_ub[lf_sample_total], S.vcadd(total, full), full, 'ONEPT_B32')
                    sync_v_to_s()
                    partial_f[packed_row, segment_id, pf_residual] = scalar_f_ub[lf_sample_total]
                else:
                    partial_f[packed_row, segment_id, pf_residual] = 0.0
                sync_all()

                if segment_id == 0 and sx < reject_steps:
                    seed_val = T.alloc_var(T.uint64, seq_seeds[bx] + T.cast(seed_offset, T.uint64))
                    position = T.alloc_var(T.int32, sample_positions[bx] + sx * DRAFT_POS_HASH_MUL)
                    draft_tid = T.alloc_var(T.int32, last_next_n_token_ids[sx, bx])
                    if draft_tid < 0:
                        accept_result_verify[sx, bx] = False
                    else:
                        draft_lp = T.alloc_var(T.float32, last_nextn_token_logprobs[sx, bx, draft_tid])
                        target_prob = T.alloc_var(
                            T.float32,
                            logprobs[pid, draft_tid] / row_f[packed_row, rf_kept_sum],
                        )
                        with T.SimtVF(threads=1):
                            accept_probability = T.alloc_var(
                                T.float32,
                                T.Select(draft_lp > 0.5, 1.0, T.min(target_prob / T.exp(draft_lp), 1.0)),
                            )
                            accept_result_verify[sx, bx] = (
                                rng_uniform53(seed_val, position * 3 + 2) < accept_probability
                            )

                    residual_total = T.alloc_var(T.float32, 0.0)
                    for group in T.serial(groups_per_row):
                        residual_total += partial_f[packed_row, group, pf_residual]
                    with T.SimtVF(threads=1):
                        scalar_f_ub[lf_random_target] = rng_uniform53(seed_val, position * 3 + 1) * residual_total
                    sync_v_to_s()
                    target = T.alloc_var(T.float32, scalar_f_ub[lf_random_target])
                    prefix = T.alloc_var(T.float32, 0.0)
                    found = T.alloc_var(T.bool, False)
                    row_i[packed_row, ri_sample_group] = groups_per_row - 1
                    row_f[packed_row, rf_group_prefix] = 0.0
                    row_f[packed_row, rf_random_target] = target
                    for group in T.serial(groups_per_row):
                        next_prefix = T.alloc_var(T.float32, prefix + partial_f[packed_row, group, pf_residual])
                        if not found and target > prefix and target <= next_prefix:
                            row_i[packed_row, ri_sample_group] = group
                            row_f[packed_row, rf_group_prefix] = prefix
                            found = True
                        prefix = next_prefix
                sync_all()

                if sx < reject_steps and segment_id == row_i[packed_row, ri_sample_group]:
                    group_target = T.alloc_var(
                        T.float32,
                        row_f[packed_row, rf_random_target] - row_f[packed_row, rf_group_prefix],
                    )
                    prefix = T.alloc_var(T.float32, 0.0)
                    found = T.alloc_var(T.bool, False)
                    scalar_i_ub[li_selected_vec] = num_segment_vectors - 1
                    scalar_i_ub[li_sample] = 0
                    scalar_f_ub[lf_group_prefix] = 0.0
                    for vector in T.serial(num_segment_vectors):
                        next_prefix = T.alloc_var(T.float32, prefix + group_sum_ub[vector])
                        if not found and group_target > prefix and group_target <= next_prefix:
                            scalar_i_ub[li_selected_vec] = vector
                            scalar_f_ub[lf_group_prefix] = prefix
                            found = True
                        prefix = next_prefix

                    selected_base = T.alloc_var(T.int32, segment_base + scalar_i_ub[li_selected_vec] * vector_lanes)
                    selected_count = T.alloc_var(T.int32, T.min(vector_lanes, segment_end - selected_base))
                    T.copy(
                        logprobs[pid, selected_base : selected_base + selected_count],
                        probs_ub[0, :selected_count],
                    )
                    T.copy(
                        last_nextn_token_logprobs[sx, bx, selected_base : selected_base + selected_count],
                        draft_ub[0, :selected_count],
                    )
                    T.ascend_set_flag('MTE2_V', 0)
                    T.ascend_wait_flag('MTE2_V', 0)
                    scalar_f_ub[lf_kept_sum] = row_f[packed_row, rf_kept_sum]
                    sync_s_to_v()
                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        denominator = S.vld(scalar_f_ub[lf_kept_sum], dist='BRC_B32')
                        target_prob_vec = S.vdiv(S.vld(probs_ub[0, 0]), denominator, full)
                        draft_prob_vec = S.vexp(S.vadds(S.vld(draft_ub[0, 0]), -DRAFT_PROB_PENALTY, full), full)
                        S.vsts(work_ub[0], S.vrelu(S.vsub(target_prob_vec, draft_prob_vec, full), full))
                    sync_v_to_s()
                    lane_target = T.alloc_var(T.float32, group_target - scalar_f_ub[lf_group_prefix])
                    prefix = 0.0
                    found = False
                    for lane in T.serial(vector_lanes):
                        col = selected_base + lane
                        if col < segment_end:
                            next_prefix = T.alloc_var(T.float32, prefix + work_ub[lane])
                            if not found and lane_target > prefix and lane_target <= next_prefix:
                                scalar_i_ub[li_sample] = col
                                found = True
                            prefix = next_prefix
                    sample_reject_cur[sx, bx] = scalar_i_ub[li_sample]
                sync_all()

            # Normalize and convert to public log-probabilities in one pass.
            scalar_f_ub[lf_kept_sum] = row_f[packed_row, rf_kept_sum]
            sync_s_to_v()
            init_input_db()
            init_output_db()
            for chunk in T.serial(num_segment_chunks):
                slot = chunk % 2
                base = segment_base + chunk * chunk_size
                valid_count = T.max(T.min(chunk_size, segment_end - base), 0)
                if valid_count > 0:
                    T.ascend_wait_flag('V_MTE2', slot)
                    T.ascend_wait_flag('MTE3_V', slot)
                    T.copy(logprobs[pid, base : base + valid_count], probs_ub[slot, :valid_count])
                    T.ascend_set_flag('MTE2_V', slot)
                    T.ascend_wait_flag('MTE2_V', slot)
                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        denominator = S.vld(scalar_f_ub[lf_kept_sum], dist='BRC_B32')
                        for vector in T.serial(vregs_per_chunk):
                            S.vsts(
                                probs_ub[slot, vector * vector_lanes],
                                S.vln(
                                    S.vdiv(S.vld(probs_ub[slot, vector * vector_lanes]), denominator, full),
                                    full,
                                ),
                                full,
                            )
                    T.ascend_set_flag('V_MTE2', slot)
                    T.ascend_set_flag('V_MTE3', slot)
                    T.ascend_wait_flag('V_MTE3', slot)
                    T.copy(probs_ub[slot, :valid_count], logprobs[pid, base : base + valid_count])
                    T.ascend_set_flag('MTE3_V', slot)
            drain_input_db()
            drain_output_db()

    return _fused_sample
