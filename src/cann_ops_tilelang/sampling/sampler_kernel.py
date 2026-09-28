from typing import Any

import tilelang
import torch
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    }
)
def _find_value_by_index_tl_ascend(
    force_topk: int,
) -> Any:
    max_seqlen_q = T.dynamic('max_seqlen_q')
    req_cnt = T.dynamic('req_cnt')
    index_n0_stride = T.dynamic('index_n0_stride')
    index_n1_stride = T.dynamic('index_n1_stride')
    value_n0_stride = T.dynamic('value_n0_stride')
    value_n1_stride = T.dynamic('value_n1_stride')
    q_index_n0_stride = T.dynamic('q_index_n0_stride')
    q_index_n1_stride = T.dynamic('q_index_n1_stride')
    num_cores = get_num_vector_cores()
    vector_lanes = 64
    assert force_topk <= 4096 and force_topk % vector_lanes == 0
    ub_size = 240000
    block_size = 4 if force_topk > 2048 else 8
    per_stage_bytes = force_topk * block_size * 8 + block_size * 32 + block_size * 4
    num_stages = min(8, ub_size // per_stage_bytes)

    @T.prim_func
    def _find_value_by_index(
        index: T.StridedTensor[(max_seqlen_q, req_cnt, force_topk), (index_n0_stride, index_n1_stride, 1), T.int32],
        value: T.StridedTensor[(max_seqlen_q, req_cnt, force_topk), (value_n0_stride, value_n1_stride, 1), T.float32],
        q_index: T.StridedTensor[(max_seqlen_q, req_cnt, 1), (q_index_n0_stride, q_index_n1_stride, 1), T.int32],
        out: T.Tensor[(max_seqlen_q, req_cnt), T.float32],
    ) -> None:
        with T.Kernel(num_cores) as pid:
            index_ub = T.alloc_shared((block_size, force_topk), T.int32)
            value_ub = T.alloc_shared((block_size, force_topk), T.float32)
            q_index_ub = T.alloc_shared((block_size, 8), T.int32)
            out_ub = T.alloc_shared((block_size,), T.float32)
            T.annotate_buffer_versions(
                {index_ub: num_stages, value_ub: num_stages, q_index_ub: num_stages, out_ub: num_stages}
            )
            for bx, by_group in T.Persistent(
                [max_seqlen_q, T.ceildiv(req_cnt, block_size)],
                num_cores,
                pid,
                group_size=1,
                num_stages=num_stages,
            ):
                by_base = by_group * block_size
                valid_block_size = T.min(block_size, req_cnt - by_base)
                for r in T.serial(valid_block_size):
                    T.copy(index[bx, by_base + r, :], index_ub[r, :force_topk])
                    T.copy(value[bx, by_base + r, :], value_ub[r, :force_topk])
                T.copy(q_index[bx, by_base : by_base + valid_block_size, :], q_index_ub[:valid_block_size, :1])

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for r in T.serial(valid_block_size):
                        target = T.simd.vld(q_index_ub[r, 0], dist='BRC_B32')
                        result = T.simd.alloc_var('float32')
                        result = T.simd.vdup(float('-inf'), 'float32', mask)
                        for i in T.serial(T.ceildiv(force_topk, vector_lanes)):
                            idx_chunk = T.simd.vld(index_ub[r, i * vector_lanes])
                            val_chunk = T.simd.vld(value_ub[r, i * vector_lanes])
                            cmp_mask = T.simd.vcmp(idx_chunk, target, mask, op='eq')
                            result = T.simd.vsel(val_chunk, result, cmp_mask)
                        result = T.simd.vcmax(result, mask)
                        T.simd.vsts(out_ub[r], result, mask, dist='ONEPT_B32')

                T.copy(out_ub[:valid_block_size], out[bx, by_base : by_base + valid_block_size])

    return _find_value_by_index


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    }
)
def _batch_find_value_by_index_tl_ascend(
    force_topk: int,
    vocab_size: int,
    use_pdl: bool,
    qcap: int | None = None,
) -> Any:
    max_seqlen_q = T.dynamic('max_seqlen_q')
    req_cnt = T.dynamic('req_cnt')
    query_dim = T.dynamic('query_dim')
    index_n0_stride = T.dynamic('index_n0_stride')
    index_n1_stride = T.dynamic('index_n1_stride')
    value_n0_stride = T.dynamic('value_n0_stride')
    value_n1_stride = T.dynamic('value_n1_stride')
    num_cores = get_num_vector_cores()
    pool_size = 43136
    vector_lanes = 64
    force_topk_size = T.ceildiv(force_topk, vector_lanes) * vector_lanes
    query_capacity = force_topk_size if qcap is None else qcap
    num_stages = min(
        4, max(1, (240000 - pool_size * 4 - query_capacity * 4) // (force_topk * 4 * 2 + query_capacity * 4))
    )

    @T.prim_func
    def _batch_find_value_by_index(
        index: T.StridedTensor[(max_seqlen_q, req_cnt, force_topk), (index_n0_stride, index_n1_stride, 1), T.int32],
        value: T.StridedTensor[(max_seqlen_q, req_cnt, force_topk), (value_n0_stride, value_n1_stride, 1), T.float32],
        q_index: T.Tensor[(max_seqlen_q, req_cnt, query_dim), T.int32],
        out: T.Tensor[(max_seqlen_q, req_cnt, query_dim), T.float32],
    ) -> None:
        num_q_vec = T.ceildiv(query_dim, vector_lanes)
        with T.Kernel(num_cores) as pid:
            pool = T.alloc_shared((pool_size,), T.float32)
            index_ub = T.alloc_shared((force_topk_size,), T.int32)
            value_ub = T.alloc_shared((force_topk_size,), T.float32)
            q_index_ub = T.alloc_shared((query_capacity,), T.int32)
            out_ub = T.alloc_shared((query_capacity,), T.float32)
            T.annotate_buffer_versions(
                {
                    index_ub: num_stages,
                    value_ub: num_stages,
                    q_index_ub: num_stages,
                }
            )
            for bx, by in T.Persistent(
                [max_seqlen_q, req_cnt],
                num_cores,
                pid,
                group_size=1,
                num_stages=num_stages,
            ):
                T.copy(index[bx, by, :], index_ub[:force_topk])
                T.copy(value[bx, by, :], value_ub[:force_topk])
                T.copy(q_index[bx, by, :], q_index_ub[:query_dim])

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    neg_inf = T.simd.vdup(float('-inf'), 'float32', mask)

                    for _range_l in T.serial(0, vocab_size, pool_size):
                        if _range_l > 0:
                            T.simd.mem_bar('VLD_VST')
                        for v in T.serial(num_q_vec):
                            q_idx = T.simd.vld(q_index_ub[v * vector_lanes])
                            q_adj = T.simd.vadds(q_idx, -_range_l, mask)
                            mask_pool = T.simd.vcmps(T.reinterpret('uint32x64', q_adj), pool_size, mask, 'lt')
                            T.simd.vscatter(neg_inf, pool[0], T.reinterpret('uint32x64', q_adj), mask_pool)
                        T.simd.mem_bar('VST_VST')
                        for v in T.serial(T.ceildiv(force_topk, vector_lanes)):
                            idx = T.simd.vld(index_ub[v * vector_lanes])
                            val = T.simd.vld(value_ub[v * vector_lanes])
                            idx_adj = T.simd.vadds(idx, -_range_l, mask)
                            mask_pool = T.simd.vcmps(T.reinterpret('uint32x64', idx_adj), pool_size, mask, 'lt')
                            T.simd.vscatter(val, pool[0], T.reinterpret('uint32x64', idx_adj), mask_pool)
                        T.simd.mem_bar('VST_VLD')

                        for v in T.serial(num_q_vec):
                            q_idx = T.simd.vld(q_index_ub[v * vector_lanes])
                            q_adj = T.simd.vadds(q_idx, -_range_l, mask)
                            mask_pool = T.simd.vcmps(T.reinterpret('uint32x64', q_adj), pool_size, mask, 'lt')
                            out_val = T.simd.vgather2(pool[0], T.reinterpret('uint32x64', q_adj), mask_pool)
                            T.simd.vsts(out_ub[v * vector_lanes], out_val, mask_pool)

                T.copy(out_ub[:query_dim], out[bx, by, :])

    return _batch_find_value_by_index


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    }
)
def _get_accept_result_tl_ascend(
    has_so_valid: bool,
    has_golden_output: bool,
) -> Any:
    num_steps = T.dynamic('M')
    max_seqlen_q = T.dynamic('max_seqlen_q')
    num_requests = T.dynamic('N')
    stride_col = T.dynamic('stride_col')

    stride_1 = T.dynamic('stride_1')
    golden_output_shape0 = T.dynamic('golden_output_shape0')
    gold_accepted_shape0 = T.dynamic('gold_accepted_shape0')
    golden_output_stride0 = T.dynamic('golden_output_stride0')
    gold_accepted_stride0 = T.dynamic('gold_accepted_stride0')

    num_cores = get_num_vector_cores()
    vector_lanes = 64

    num_stages = 4

    @T.prim_func
    def _get_accept_result(
        accept_result_verify: T.Tensor[(num_steps, num_requests), T.bool],
        so_valid: T.Tensor[(num_steps, num_requests), T.bool],
        golden_output: T.StridedTensor[
            (golden_output_shape0, num_requests), (stride_1, golden_output_stride0), T.int32
        ],
        last_next_n_token_ids: T.StridedTensor[(num_steps, num_requests), (stride_1, stride_col), T.int32],
        gold_accepted: T.StridedTensor[(gold_accepted_shape0, num_requests), (stride_1, gold_accepted_stride0), T.bool],
        is_verify: T.Tensor[(max_seqlen_q, num_requests), T.bool],
        sample_reject_cur: T.Tensor[(num_steps, num_requests), T.int32],
        fresh_sample_final: T.Tensor[(max_seqlen_q, num_requests), T.int32],
        sample_results_cur: T.Tensor[(max_seqlen_q, num_requests), T.int32],
        accept_result: T.Tensor[(max_seqlen_q, num_requests), T.bool],
    ) -> None:
        with T.Kernel(num_cores) as pid:
            for bx, bn in T.Persistent(
                [max_seqlen_q, T.ceildiv(num_requests, vector_lanes)], num_cores, pid, num_stages=num_stages
            ):
                offset_n = bn * vector_lanes
                cnt_n = T.min(num_requests - offset_n, vector_lanes)

                is_verify_ub = T.alloc_shared((vector_lanes,), T.int8)
                fresh_final_ub = T.alloc_shared((vector_lanes,), T.int32)
                acc_verify_ub = T.alloc_shared((vector_lanes,), T.int8)
                reject_cur_ub = T.alloc_shared((vector_lanes,), T.int32)
                last_next_ub = T.alloc_shared((vector_lanes,), T.int32)
                if has_so_valid:
                    so_valid_ub = T.alloc_shared((vector_lanes,), T.int8)
                    T.annotate_buffer_versions(
                        {
                            so_valid_ub: num_stages,
                        }
                    )
                if has_golden_output:
                    golden_output_ub = T.alloc_shared((vector_lanes,), T.int32)
                    gold_accepted_i8_ub = T.alloc_shared((vector_lanes,), T.int8)
                    T.annotate_buffer_versions(
                        {
                            golden_output_ub: num_stages,
                            gold_accepted_i8_ub: num_stages,
                        }
                    )
                result_i32_ub = T.alloc_shared((vector_lanes,), T.int32)
                accept_i8_ub = T.alloc_shared((vector_lanes,), T.int8)

                T.annotate_buffer_versions(
                    {
                        is_verify_ub: num_stages,
                        fresh_final_ub: num_stages,
                        acc_verify_ub: num_stages,
                        reject_cur_ub: num_stages,
                        last_next_ub: num_stages,
                        result_i32_ub: num_stages,
                        accept_i8_ub: num_stages,
                    }
                )

                is_verify_i8 = T.view(is_verify, dtype=T.int8)
                T.copy(is_verify_i8[bx, offset_n : offset_n + cnt_n], is_verify_ub[:cnt_n])
                T.copy(fresh_sample_final[bx, offset_n : offset_n + cnt_n], fresh_final_ub[:cnt_n])

                if bx < num_steps:
                    accept_result_verify_i8 = T.view(accept_result_verify, dtype=T.int8)
                    T.copy(accept_result_verify_i8[bx, offset_n : offset_n + cnt_n], acc_verify_ub[:cnt_n])
                    T.copy(sample_reject_cur[bx, offset_n : offset_n + cnt_n], reject_cur_ub[:cnt_n])
                    if has_so_valid:
                        so_valid_i8 = T.view(so_valid, dtype=T.int8)
                        T.copy(so_valid_i8[bx, offset_n : offset_n + cnt_n], so_valid_ub[:cnt_n])

                    # SIMT: read strided tensors → UB
                    with T.SimtVF(threads=vector_lanes):
                        for j in T.Parallel(vector_lanes):
                            if j < cnt_n:
                                last_next_ub[j] = last_next_n_token_ids[bx, offset_n + j]
                        if has_golden_output and bx < golden_output_shape0:
                            for j in T.Parallel(vector_lanes):
                                if j < cnt_n:
                                    golden_output_ub[j] = golden_output[bx, offset_n + j]
                                    gold_accepted_i8_ub[j] = T.Select(
                                        gold_accepted[bx, offset_n + j], T.int8(1), T.int8(0)
                                    )

                        for j in T.Parallel(vector_lanes):
                            if j < cnt_n:
                                is_verify_j = T.cast(is_verify_ub[j], T.int32)
                                fresh_final_j = fresh_final_ub[j]
                                acc_verify_j = T.cast(acc_verify_ub[j], T.int32)
                                reject_cur_j = reject_cur_ub[j]
                                last_next_j = last_next_ub[j]

                                if has_so_valid:
                                    so_valid_j = T.cast(so_valid_ub[j], T.int32)
                                    acc_verify_j = T.if_then_else(so_valid_j != 0, acc_verify_j, 0)

                                if has_golden_output:
                                    gold_out_j = golden_output_ub[j]
                                    gold_acc_j = T.cast(gold_accepted_i8_ub[j], T.int32)
                                    acc_verify_j = T.if_then_else(
                                        bx < golden_output_shape0,
                                        T.if_then_else(
                                            gold_out_j == T.cast(-1, T.int32),
                                            acc_verify_j,
                                            T.if_then_else(gold_out_j == last_next_j, gold_acc_j, 0),
                                        ),
                                        acc_verify_j,
                                    )
                                    reject_cur_j = T.if_then_else(
                                        bx < golden_output_shape0,
                                        T.if_then_else(gold_out_j == T.cast(-1, T.int32), reject_cur_j, gold_out_j),
                                        reject_cur_j,
                                    )

                                sample_verify = T.if_then_else(acc_verify_j != 0, last_next_j, reject_cur_j)
                                result_j = T.if_then_else(is_verify_j != 0, sample_verify, fresh_final_j)
                                accept_j = T.if_then_else(is_verify_j != 0, T.if_then_else(acc_verify_j != 0, 1, 0), 1)

                                result_i32_ub[j] = result_j
                                accept_i8_ub[j] = T.Select(accept_j != 0, T.int8(1), T.int8(0))
                else:
                    with T.SimtVF(threads=vector_lanes):
                        for j in T.Parallel(vector_lanes):
                            if j < cnt_n:
                                result_i32_ub[j] = fresh_final_ub[j]
                                accept_i8_ub[j] = T.int8(1)

                T.copy(result_i32_ub[:cnt_n], sample_results_cur[bx, offset_n : offset_n + cnt_n])
                T.copy(T.view(accept_i8_ub, dtype=T.bool)[:cnt_n], accept_result[bx, offset_n : offset_n + cnt_n])

    return _get_accept_result


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _get_logits_cur_by_step_tl_ascend(
    vocab_size: int,
) -> Any:
    total_q = T.dynamic('total_q')
    logits_stride_0 = T.dynamic('logits_stride_0')
    n = T.dynamic('n')
    block_dim = 4096
    num_dim_blocks = T.ceildiv(vocab_size, block_dim)
    num_cores = get_num_vector_cores()

    @T.prim_func
    def _get_logits_cur_by_step(
        logits: T.StridedTensor[(total_q, vocab_size), (logits_stride_0, 1), T.float],
        ftp: T.Tensor[n, T.int32],
        has_token: T.Tensor[n, T.bool],
        out: T.Tensor[(n, vocab_size), T.float],
        step: T.int32,
    ) -> None:
        T.assume(logits_stride_0 % 16 == 0)
        with T.Kernel(num_cores) as core_id:
            logits_ub = T.alloc_shared((block_dim,), T.float32)
            for row, dim_block in T.Persistent(
                [n, num_dim_blocks],
                num_cores,
                core_id,
                group_size=1,
                num_stages=2,
            ):
                dim_start = dim_block * block_dim
                copy_length = T.min(block_dim, vocab_size - dim_start)
                if has_token[row]:
                    gather_idx = ftp[row] + step
                    T.assume(0 <= gather_idx < total_q)
                    T.copy(
                        logits[gather_idx, dim_start : dim_start + copy_length],
                        logits_ub[:copy_length],
                    )
                else:
                    T.fill(logits_ub[:copy_length], 0)
                T.copy(
                    logits_ub[:copy_length],
                    out[row, dim_start : dim_start + copy_length],
                )

    return _get_logits_cur_by_step


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _update_verify_states_tl_ascend() -> Any:
    num_steps = T.dynamic('N')
    num_requests = T.dynamic('R')
    num_cores = get_num_vector_cores()
    vector_lanes = 64
    block_size = vector_lanes

    @T.prim_func
    def _update_verify_states(
        accept_result_all: T.Tensor[(num_steps, num_requests), T.bool],
        has_token_all: T.Tensor[(num_steps, num_requests), T.bool],
        seqlen_q: T.Tensor[num_requests, T.int32],
        sample_results_cur_all: T.Tensor[(num_steps, num_requests), T.int32],
        count_steps: T.int32,
        accepted_tokens_all: T.Tensor[(num_steps, num_requests), T.int32],
        is_final_all: T.Tensor[(num_steps, num_requests), T.bool],
        last_valid_offset: T.Tensor[num_requests, T.int32],
        num_generated_tokens: T.Tensor[num_steps, T.int32],
    ) -> None:
        with T.Kernel(num_cores) as pid:
            for bx in T.Persistent([T.ceildiv(num_requests, block_size)], num_cores, pid, num_stages=2):
                offset = bx * block_size
                block_end = T.min(offset + block_size, num_requests)
                rem = block_end - offset

                seqlen_state_ub = T.alloc_shared((block_size,), T.int32)
                accept_i8_ub = T.alloc_shared((block_size,), T.int8)
                has_i8_ub = T.alloc_shared((block_size,), T.int8)
                accept_ub = T.alloc_shared((block_size,), T.int32)
                has_ub = T.alloc_shared((block_size,), T.int32)
                sample_ub = T.alloc_shared((block_size,), T.int32)
                accepted_ub = T.alloc_shared((block_size,), T.int32)
                final_ub = T.alloc_shared((block_size,), T.int32)
                final_i8_ub = T.alloc_shared((block_size,), T.int8)
                last_state_ub = T.alloc_shared((block_size,), T.int32)
                valid_state_ub = T.alloc_shared((block_size,), T.int32)
                valid_sum_ub = T.alloc_shared((2,), T.int32)

                T.annotate_buffer_versions(
                    {
                        last_state_ub: 2,
                        valid_state_ub: 2,
                        accept_ub: 2,
                        has_ub: 2,
                        final_ub: 2,
                        accept_i8_ub: 2,
                        has_i8_ub: 2,
                        final_i8_ub: 2,
                    }
                )

                T.copy(seqlen_q[offset : offset + rem], seqlen_state_ub[:rem])

                with T.SimdVF():
                    full = T.simd.pset(32)
                    lane_idx = T.simd.vci(0, 'int32', 'INC_ORDER')
                    rem_v = T.simd.vdup(rem, 'int32', full)
                    tail_mask_init = T.simd.vcmp(lane_idx, rem_v, full, 'lt')

                    seq_v = T.simd.vld(seqlen_state_ub[0])
                    last_init = T.simd.vadds(seq_v, -1, full)
                    valid_init = T.simd.vdup(1, 'int32', full)
                    T.simd.vsts(last_state_ub[0], last_init, tail_mask_init)
                    T.simd.vsts(valid_state_ub[0], valid_init, tail_mask_init)

                for step_i in T.Serial(num_steps):
                    accept_result_all_i8 = T.view(accept_result_all, dtype=T.int8)
                    has_token_all_i8 = T.view(has_token_all, dtype=T.int8)
                    T.copy(accept_result_all_i8[step_i, offset : offset + rem], accept_i8_ub[:rem])
                    T.copy(has_token_all_i8[step_i, offset : offset + rem], has_i8_ub[:rem])
                    T.copy(sample_results_cur_all[step_i, offset : offset + rem], sample_ub[:rem])

                    with T.SimtVF(threads=block_size):
                        for j in T.Parallel(block_size):
                            accept_ub[j] = T.cast(accept_i8_ub[j], T.int32)
                            has_ub[j] = T.cast(has_i8_ub[j], T.int32)

                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, 'PAT_VL1')
                        zero_i32 = T.simd.vdup(0, 'int32', full)
                        one_i32 = T.simd.vdup(1, 'int32', full)

                        lane_idx = T.simd.vci(0, 'int32', 'INC_ORDER')
                        rem_v = T.simd.vdup(rem, 'int32', full)
                        tail_mask = T.simd.vcmp(lane_idx, rem_v, full, 'lt')

                        seq_v = T.simd.vld(seqlen_state_ub[0])
                        acc_v = T.simd.vld(accept_ub[0])
                        has_v = T.simd.vld(has_ub[0])
                        smp_v = T.simd.vld(sample_ub[0])

                        last_v = T.simd.vld(last_state_ub[0])
                        valid_v = T.simd.vld(valid_state_ub[0])

                        step_1 = T.simd.vdup(step_i + 1, 'int32', full)
                        is_last = T.simd.vsel(one_i32, zero_i32, T.simd.vcmp(seq_v, step_1, full, 'eq'))
                        not_is_last = T.simd.vsel(one_i32, zero_i32, T.simd.vcmp(seq_v, step_1, full, 'ne'))
                        vh = T.simd.vand(valid_v, has_v, full)
                        vha = T.simd.vand(vh, acc_v, full)
                        not_acc = T.simd.vsel(one_i32, zero_i32, T.simd.vcmps(acc_v, 0, full, 'eq'))
                        is_final_v = T.simd.vand(vh, T.simd.vor(is_last, not_acc, full), full)
                        valid_v = T.simd.vand(vha, not_is_last, full)

                        cond_lv = T.simd.vand(vh, not_acc, full)
                        cond_pred = T.simd.vcmp(cond_lv, zero_i32, full, 'ne')
                        last_v = T.simd.vsel(T.simd.vdup(step_i, 'int32', full), last_v, cond_pred)

                        valid_pred = T.simd.vcmp(valid_v, zero_i32, full, 'ne')
                        acc_out_v = T.simd.vsel(smp_v, T.simd.vdup(-1, 'int32', full), valid_pred)

                        valid_16_0 = T.simd.vcvt(valid_v, 'int16', tail_mask, part=0)
                        valid_sum_0 = T.simd.vcadd(valid_16_0, tail_mask)
                        T.simd.vsts(valid_sum_ub[0], valid_sum_0, one, 'ONEPT_B32')

                        valid_16_1 = T.simd.vcvt(valid_v, 'int16', tail_mask, part=1)
                        valid_sum_1 = T.simd.vcadd(valid_16_1, tail_mask)
                        T.simd.vsts(valid_sum_ub[1], valid_sum_1, one, 'ONEPT_B32')

                        T.simd.vsts(accepted_ub[0], acc_out_v, tail_mask)
                        T.simd.vsts(final_ub[0], is_final_v, tail_mask)
                        T.simd.vsts(last_state_ub[0], last_v, tail_mask)
                        T.simd.vsts(valid_state_ub[0], valid_v, tail_mask)

                    with T.SimtVF(threads=block_size):
                        for j in T.Parallel(block_size):
                            if j < rem:
                                final_i8_ub[j] = T.Select(final_ub[j] != 0, T.int8(1), T.int8(0))
                        if T.get_thread_binding() == 0 and step_i < count_steps:
                            T.atomic_add(num_generated_tokens[step_i], valid_sum_ub[0] + valid_sum_ub[1])

                    T.copy(accepted_ub[:rem], accepted_tokens_all[step_i, offset : offset + rem])
                    T.copy(T.view(final_i8_ub, dtype=T.bool)[:rem], is_final_all[step_i, offset : offset + rem])

                T.copy(last_state_ub[:rem], last_valid_offset[offset : offset + rem])

    return _update_verify_states


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _scatter_verified_tl_ascend(
    is_rl_or_eval: bool,
    has_topk: bool,
    topk_n: int,
    top_token_ids_cur_dtype: torch.dtype,
) -> Any:
    assert not has_topk or 8 * ((topk_n + 15) // 16 * 16) <= 240000
    topk_n_aligned = T.max(T.ceildiv(topk_n, 16) * 16, 16)
    num_steps = T.dynamic('N')
    req_cnt = T.dynamic('req_cnt')
    total_q = T.dynamic('total_q')
    top_logprobs_stride_0 = T.dynamic('top_logprobs_stride_0')
    top_logprobs_stride_1 = T.dynamic('top_logprobs_stride_1')
    top_token_ids_stride_0 = T.dynamic('top_token_ids_stride_0')
    top_token_ids_stride_1 = T.dynamic('top_token_ids_stride_1')
    num_cores = get_num_vector_cores()

    @T.prim_func
    def _scatter_verified(
        is_verify: T.Tensor[(num_steps, req_cnt), T.bool],
        ftp: T.Tensor[req_cnt, T.int32],
        accepted_tokens: T.Tensor[(num_steps, req_cnt), T.int32],
        token_logprobs_cur: T.Tensor[(num_steps, req_cnt), T.float32],
        sum_smaller_probs_cur: T.Tensor[(num_steps, req_cnt), T.float32],
        last_nextn_token_logprobs: T.Tensor[(num_steps, req_cnt), T.float32],
        verified_token_ids: T.Tensor[total_q, T.int32],
        verified_token_logprobs: T.Tensor[total_q, T.float32],
        verified_sum_smaller_probs: T.Tensor[total_q, T.float32],
        accepted_token_last_logprobs: T.Tensor[total_q, T.float32],
        top_logprobs_cur: T.StridedTensor[
            (num_steps, req_cnt, topk_n), (top_logprobs_stride_0, top_logprobs_stride_1, 1), T.float32
        ],
        top_token_ids_cur: T.StridedTensor[
            (num_steps, req_cnt, topk_n), (top_token_ids_stride_0, top_token_ids_stride_1, 1), top_token_ids_cur_dtype
        ],
        verified_topk_logps: T.Tensor[(total_q, topk_n), T.float32],
        verified_topk_tokens: T.Tensor[(total_q, topk_n), T.int32],
    ) -> None:
        with T.Kernel(num_cores) as pid:
            for step_i, req_idx in T.Persistent([num_steps, req_cnt], num_cores, pid, num_stages=2):
                is_verify_var = T.alloc_var(T.bool, is_verify[step_i, req_idx])
                if is_verify_var:
                    dest = T.alloc_var(T.int32, ftp[req_idx] + step_i)
                    if 0 <= dest < total_q:
                        _tmp_id = accepted_tokens[step_i, req_idx]
                        _tmp_lp = token_logprobs_cur[step_i, req_idx]
                        verified_token_ids[dest] = _tmp_id
                        verified_token_logprobs[dest] = _tmp_lp
                        if is_rl_or_eval:
                            _tmp_ss = sum_smaller_probs_cur[step_i, req_idx]
                            _tmp_ll = last_nextn_token_logprobs[step_i, req_idx]
                            verified_sum_smaller_probs[dest] = _tmp_ss
                            accepted_token_last_logprobs[dest] = _tmp_ll

                        if has_topk:
                            ub_lp = T.alloc_shared((topk_n,), T.float32)
                            if topk_n_aligned > topk_n:
                                ub_lp_pad = T.alloc_shared(  # noqa: F841
                                    (topk_n_aligned - topk_n,), T.float32
                                )
                            ub_tk = T.alloc_shared((topk_n,), T.int32)
                            if topk_n_aligned > topk_n:
                                ub_tk_pad = T.alloc_shared(  # noqa: F841
                                    (topk_n_aligned - topk_n,), T.int32
                                )
                            T.copy(top_logprobs_cur[step_i, req_idx, :], ub_lp)
                            T.copy(top_token_ids_cur[step_i, req_idx, :], ub_tk)
                            T.copy(ub_lp, verified_topk_logps[dest, :])
                            T.copy(ub_tk, verified_topk_tokens[dest, :])

    return _scatter_verified


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _sampled_tokens_update_tl_ascend(
    has_sum_smaller: bool,
    has_top_logprobs: bool,
    has_top_token_ids: bool,
    topk_n: int,
    top_token_ids_cur_dtype: torch.dtype,
) -> Any:
    stage_bytes = 576 + (256 if has_sum_smaller else 0) + 4 * topk_n * (int(has_top_logprobs) + int(has_top_token_ids))
    assert stage_bytes <= 240000
    topk_n_aligned = T.max(T.ceildiv(topk_n, 16) * 16, 16)  # noqa: F841
    max_seqlen_q = T.dynamic('max_seqlen_q')
    req_cnt = T.dynamic('req_cnt')
    top_logprobs_stride_0 = T.dynamic('top_logprobs_stride_0')
    top_logprobs_stride_1 = T.dynamic('top_logprobs_stride_1')
    top_token_ids_stride_0 = T.dynamic('top_token_ids_stride_0')
    top_token_ids_stride_1 = T.dynamic('top_token_ids_stride_1')
    num_cores = get_num_vector_cores()
    block_size = 16
    num_stages = 2

    @T.prim_func
    def _sampled_tokens_update(
        is_final: T.Tensor[(max_seqlen_q, req_cnt), T.bool],
        sample_results_cur: T.Tensor[(max_seqlen_q, req_cnt), T.int32],
        token_logprobs_cur: T.Tensor[(max_seqlen_q, req_cnt), T.float32],
        sum_smaller_probs_cur: T.Tensor[(max_seqlen_q, req_cnt), T.float32],
        top_logprobs_cur: T.StridedTensor[
            (max_seqlen_q, req_cnt, topk_n), (top_logprobs_stride_0, top_logprobs_stride_1, 1), T.float32
        ],
        top_token_ids_cur: T.StridedTensor[
            (max_seqlen_q, req_cnt, topk_n),
            (top_token_ids_stride_0, top_token_ids_stride_1, 1),
            top_token_ids_cur_dtype,
        ],
        token_ids_out: T.Tensor[req_cnt, T.int32],
        token_logprobs_out: T.Tensor[req_cnt, T.float32],
        sum_smaller_probs_out: T.Tensor[req_cnt, T.float32],
        top_logps_out: T.Tensor[(req_cnt, topk_n), T.float32],
        top_tokens_out: T.Tensor[(req_cnt, topk_n), T.int32],
    ) -> None:
        with T.Kernel(num_cores) as pid:
            for step_i, req_idx_ in T.Persistent(
                [max_seqlen_q, T.ceildiv(req_cnt, block_size)], num_cores, pid, group_size=1, num_stages=num_stages
            ):
                req_cnt_local = T.min(block_size, req_cnt - req_idx_ * block_size)
                is_final_ub = T.alloc_shared((32,), T.int8)
                sample_results_cur_ub = T.alloc_shared((32,), T.int32)
                token_logprobs_cur_ub = T.alloc_shared((32,), T.float32)
                T.annotate_buffer_versions(
                    {
                        is_final_ub: num_stages,
                        sample_results_cur_ub: num_stages,
                        token_logprobs_cur_ub: num_stages,
                    }
                )
                is_final_i8 = T.view(is_final, dtype=T.int8)
                T.copy(
                    is_final_i8[step_i, req_idx_ * block_size : req_idx_ * block_size + req_cnt_local],
                    is_final_ub[:req_cnt_local],
                )
                T.copy(
                    sample_results_cur[step_i, req_idx_ * block_size : req_idx_ * block_size + req_cnt_local],
                    sample_results_cur_ub[:req_cnt_local],
                )
                T.copy(
                    token_logprobs_cur[step_i, req_idx_ * block_size : req_idx_ * block_size + req_cnt_local],
                    token_logprobs_cur_ub[:req_cnt_local],
                )
                if has_sum_smaller:
                    sum_smaller_probs_cur_ub = T.alloc_shared((32,), T.float32)
                    T.annotate_buffer_versions({sum_smaller_probs_cur_ub: num_stages})
                    T.copy(
                        sum_smaller_probs_cur[step_i, req_idx_ * block_size : req_idx_ * block_size + req_cnt_local],
                        sum_smaller_probs_cur_ub[:req_cnt_local],
                    )
                for i in T.serial(block_size):
                    req_idx = i + req_idx_ * block_size
                    if req_idx < req_cnt and is_final_ub[i] != 0:
                        T.assume(0 <= step_i < max_seqlen_q)
                        T.assume(0 <= req_idx < req_cnt)
                        token_ids_out[req_idx] = sample_results_cur_ub[i]
                        token_logprobs_out[req_idx] = token_logprobs_cur_ub[i]
                        if has_sum_smaller:
                            sum_smaller_probs_out[req_idx] = sum_smaller_probs_cur_ub[i]
                        if has_top_logprobs:
                            ub_lp = T.alloc_shared((topk_n,), T.float32)
                            T.copy(top_logprobs_cur[step_i, req_idx, :], ub_lp)
                            T.copy(ub_lp, top_logps_out[req_idx, :])
                        if has_top_token_ids:
                            ub_tk = T.alloc_shared((topk_n,), T.int32)
                            T.copy(top_token_ids_cur[step_i, req_idx, :], ub_tk)
                            T.copy(ub_tk, top_tokens_out[req_idx, :])

    return _sampled_tokens_update


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    },
)
def _compute_confidence_error_tl_ascend() -> Any:
    num_steps = T.dynamic('M')
    num_requests = T.dynamic('N')
    confidence_stride = T.dynamic('confidence_stride')
    confidence_stride0 = T.dynamic('confidence_stride0')
    num_cores = get_num_vector_cores()
    vector_lanes = 64
    num_stages = 4

    @T.prim_func
    def _compute_confidence_error(
        last_logprobs: T.Tensor[(num_steps, num_requests), T.float32],
        next_logprobs: T.Tensor[(num_steps, num_requests), T.float32],
        confidence: T.StridedTensor[(num_steps, num_requests), (confidence_stride0, confidence_stride), T.float32],
        verify_masks: T.Tensor[(num_steps, num_requests), T.bool],
        out: T.Tensor[1, T.float32],
        temp: T.Tensor[num_steps, T.float32],
    ) -> None:
        with T.Kernel(num_cores) as pid:
            for bx, bn in T.Persistent(
                [num_steps, T.ceildiv(num_requests, vector_lanes)], num_cores, pid, num_stages=num_stages
            ):
                offset = bn * vector_lanes
                rem = T.min(num_requests - offset, vector_lanes)

                last_ub = T.alloc_shared((vector_lanes,), T.float32)
                next_ub = T.alloc_shared((vector_lanes,), T.float32)
                conf_ub = T.alloc_shared((vector_lanes,), T.float32)
                mask_i32_ub = T.alloc_shared((vector_lanes,), T.int32)
                mask_ub = T.alloc_shared((vector_lanes,), T.int8)
                total_ub = T.alloc_shared((8,), T.float32)
                T.annotate_buffer_versions(
                    {
                        last_ub: num_stages,
                        next_ub: num_stages,
                        conf_ub: num_stages,
                        mask_i32_ub: num_stages,
                        mask_ub: num_stages,
                        total_ub: num_stages,
                    }
                )

                verify_masks_i8 = T.view(verify_masks, dtype=T.int8)
                T.copy(last_logprobs[bx, offset : offset + rem], last_ub[:rem])
                T.copy(next_logprobs[bx, offset : offset + rem], next_ub[:rem])
                T.copy(verify_masks_i8[bx, offset : offset + rem], mask_ub[:rem])

                with T.SimtVF(threads=vector_lanes):
                    for j in T.Parallel(vector_lanes):
                        if offset + j < num_requests:
                            conf_ub[j] = confidence[bx, offset + j]
                            mask_i32_ub[j] = T.cast(mask_ub[j], T.int32)

                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, 'PAT_VL1')
                    zero_f32 = T.simd.vdup(0.0, 'float32', full)

                    lane_idx = T.simd.vci(0, 'int32', 'INC_ORDER')
                    rem_v = T.simd.vdup(rem, 'int32', full)
                    tail_mask = T.simd.vcmp(lane_idx, rem_v, full, 'lt')

                    last_v = T.simd.vld(last_ub[0])
                    next_v = T.simd.vld(next_ub[0])
                    conf_v = T.simd.vld(conf_ub[0])
                    mask_v = T.simd.vld(mask_i32_ub[0])

                    pos_pred = T.simd.vcmp(last_v, zero_f32, full, 'gt')
                    logp_diff = T.simd.vsub(next_v, last_v, full)
                    clamped = T.simd.vmin(logp_diff, zero_f32, full)
                    accept_logp = T.simd.vsel(zero_f32, clamped, pos_pred)
                    accept_prob = T.simd.vexp(accept_logp, full)

                    prob_diff = T.simd.vsub(accept_prob, conf_v, full)
                    abs_diff = T.simd.vabs(prob_diff, full)

                    zero_i32 = T.simd.vdup(0, 'int32', full)
                    mask_pred = T.simd.vcmp(mask_v, zero_i32, full, 'ne')
                    error_v = T.simd.vsel(abs_diff, zero_f32, mask_pred)

                    error_sum = T.simd.vcadd(error_v, tail_mask)
                    T.simd.vsts(total_ub[0], error_sum, one, 'ONEPT_B32')

                with T.SimtVF(threads=1):
                    T.atomic_add(out[0], total_ub[0])

    return _compute_confidence_error
