import tilelang
import torch
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.config import get_num_vector_cores


def get_block_size(vocab_size: int) -> int:
    # A larger AIV tile helps production vocabularies; 8192 avoids excess tail work on smaller shapes.
    block_size = 16384 if vocab_size >= 98304 else 8192
    if vocab_size <= block_size:
        return tilelang.next_power_of_2(vocab_size)
    return block_size


@tilelang.jit(
    out_idx=None,
    target='ascend',
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _softmax_tl_ascend(
    vocab_size: int,
    vocab_block_size: int,
    dtype: torch.dtype,
    do_softmax: bool,
    do_logsoftmax: bool,
    inplace_softmax: bool,
    inplace_logsoftmax: bool,
) -> None:
    """Ascend AIV implementation shared by the softmax family."""
    num_tokens, logits_stride0 = T.dynamic('num_tokens'), T.dynamic('logits_stride0')
    num_cores = get_num_vector_cores()
    num_vocab_blocks = (vocab_size + vocab_block_size - 1) // vocab_block_size

    vector_lanes = 64
    negative_limit = -3.0e38  # not float32's minimum: the literal check compares in double and rejects it
    simd_dtype_map = {torch.float32: T.float32, torch.bfloat16: T.bfloat16, torch.float16: T.float16}
    ub_block_size = -(-vocab_block_size // vector_lanes) * vector_lanes

    narrow = dtype != torch.float32
    store_dist = 'PK_B32' if narrow else 'NORM_B32'

    def from_ub(buf, offset):
        if narrow:
            return S.vcvt(S.vld(buf[offset], dist='UNPK_B16'), T.float32, part=0)
        return S.vld(buf[offset])

    def to_ub(value):
        if narrow:
            return S.vcvt(value, simd_dtype_map[dtype])
        return value

    assert dtype in simd_dtype_map, f'softmax on Ascend covers {list(simd_dtype_map)}, got {dtype}'
    assert vocab_size > 0
    assert vocab_block_size <= 16384
    assert int(do_softmax) + int(do_logsoftmax) >= 1
    assert int(inplace_softmax) + int(inplace_logsoftmax) <= 1
    assert not inplace_softmax or do_softmax
    assert not inplace_logsoftmax or do_logsoftmax

    # Both passes stream the row through logits_ub one block at a time, reading and computing in turn.
    # A second version of logits_ub lets the next block's copy run under this block's vector work.
    # The passes are sibling loops, so each can rotate the same buffer: only one more block is needed,
    # and it is taken only when UB holds it next to every other block buffer and the running state.
    ub_capacity_bytes = 248 * 1024
    ub_align = 32  # state_ub and lse_ub each take one aligned unit
    num_block_buffers = (
        1
        + int(do_softmax and not inplace_softmax)
        + int(do_logsoftmax and not inplace_logsoftmax)
        + int(inplace_softmax or inplace_logsoftmax)
    )
    block_bytes = ub_block_size * torch.finfo(dtype).bits // 8
    double_buffer = num_vocab_blocks > 1 and (num_block_buffers + 1) * block_bytes + 2 * ub_align <= ub_capacity_bytes
    block_stages = 2 if double_buffer else 0

    @T.macro
    def simd_fold_block(logits_ub, state_ub, n_full, n_tail):
        with T.SimdVF():
            full = S.pset(32, 'PAT_ALL')
            neg_big = S.vdup(negative_limit, T.float32, full)
            lane_idx = S.vci(T.int32(0), T.int32)
            if double_buffer:
                # The previous block's region stored the running state; with the blocks pipelined,
                # nothing else orders that store before this load.
                S.mem_bar('VST_VLD')
            run_max = S.vld(state_ub[0], dist='BRC_B32')
            run_sum = S.vld(state_ub[1], dist='BRC_B32')

            lane_max = S.alloc_var(T.float32)
            lane_max = run_max
            for vec in T.serial(n_full):
                lane_max = S.vmax(lane_max, from_ub(logits_ub, vec * vector_lanes), full)
            tail = S.vsel(
                from_ub(logits_ub, n_full * vector_lanes),
                neg_big,
                S.vcmps(lane_idx, n_tail, full, 'lt'),
            )
            lane_max = S.vmax(lane_max, tail, full)
            new_max = S.vdupv(S.vcmax(lane_max, full), full)

            lane_sum = S.alloc_var(T.float32)
            lane_sum = S.vdup(0.0, T.float32, full)
            for vec in T.serial(n_full):
                lane_sum = S.vadd(
                    lane_sum,
                    S.vexp(S.vsub(from_ub(logits_ub, vec * vector_lanes), new_max, full), full),
                    full,
                )
            tail2 = S.vsel(
                from_ub(logits_ub, n_full * vector_lanes),
                neg_big,
                S.vcmps(lane_idx, n_tail, full, 'lt'),
            )
            lane_sum = S.vadd(lane_sum, S.vexp(S.vsub(tail2, new_max, full), full), full)
            block_sum = S.vdupv(S.vcadd(lane_sum, full), full)

            merged = S.vadd(
                S.vmul(S.vexp(S.vsub(run_max, new_max, full), full), run_sum, full),
                block_sum,
                full,
            )
            S.vsts(state_ub[0], new_max, full, 'ONEPT_B32')
            S.vsts(state_ub[1], merged, full, 'ONEPT_B32')

    @T.macro
    def simd_apply_block(logits_ub, lse_ub, out_ub, n_vectors, take_exp):
        with T.SimdVF():
            full = S.pset(32, 'PAT_ALL')
            if double_buffer:
                S.mem_bar('VST_VLD')
            lse = S.vld(lse_ub[0], dist='BRC_B32')
            for vec in T.serial(n_vectors):
                logprob = S.vsub(from_ub(logits_ub, vec * vector_lanes), lse, full)
                if take_exp:
                    S.vsts(out_ub[vec * vector_lanes], to_ub(S.vexp(logprob, full)), None, store_dist)
                else:
                    S.vsts(out_ub[vec * vector_lanes], to_ub(logprob), None, store_dist)

    @T.prim_func
    def _softmax_ascend(
        logits: T.StridedTensor[[num_tokens, vocab_size], [logits_stride0, 1], dtype],  # type: ignore
        softmax: T.Tensor[[num_tokens, vocab_size], dtype],
        logsoftmax: T.Tensor[[num_tokens, vocab_size], dtype],
    ):  # type: ignore
        with T.Kernel(num_cores) as core_id:
            logits_ub = T.alloc_shared((ub_block_size,), dtype)
            state_ub = T.alloc_shared((2,), T.float32)
            lse_ub = T.alloc_shared((1,), T.float32)
            if do_softmax and not inplace_softmax:
                softmax_ub = T.alloc_shared((ub_block_size,), dtype)
            if do_logsoftmax and not inplace_logsoftmax:
                logsoftmax_ub = T.alloc_shared((ub_block_size,), dtype)
            if inplace_softmax or inplace_logsoftmax:
                inplace_ub = T.alloc_shared((ub_block_size,), dtype)
            # Filled in place rather than rebound: a name bound under the if would not be seen after it.
            versions = {}
            if double_buffer:
                versions[logits_ub] = 2
                T.annotate_buffer_versions(versions)
            rotated = list(versions)

            for token_idx in T.Persistent([num_tokens], num_cores, core_id, group_size=1, num_stages=0):
                with T.SimdVF():
                    S.vsts(state_ub[0], S.vdup(negative_limit, T.float32), dist='ONEPT_B32')
                    S.vsts(state_ub[1], S.vdup(0.0, T.float32), dist='ONEPT_B32')

                # Online log-sum-exp keeps only one vocabulary block in UB.
                for vocab_block in T.Pipelined(
                    num_vocab_blocks, num_stages=block_stages, annotations={'multi_buffer_eligible': rotated}
                ):
                    block_offset = vocab_block * vocab_block_size
                    valid_cols = T.min(vocab_block_size, vocab_size - block_offset)
                    T.copy(logits[token_idx, block_offset : block_offset + valid_cols], logits_ub[:valid_cols])
                    n_full = T.ceildiv(valid_cols, vector_lanes) - 1
                    simd_fold_block(logits_ub, state_ub, n_full, valid_cols - n_full * vector_lanes)

                with T.SimdVF():
                    full = S.pset(32, 'PAT_ALL')
                    if double_buffer:
                        S.mem_bar('VST_VLD')
                    S.vsts(
                        lse_ub[0],
                        S.vadd(
                            S.vln(S.vld(state_ub[1], dist='BRC_B32'), full),
                            S.vld(state_ub[0], dist='BRC_B32'),
                            full,
                        ),
                        full,
                        'ONEPT_B32',
                    )

                # The second pass reuses the same LSE for either or both outputs.
                for vocab_block in T.Pipelined(
                    num_vocab_blocks, num_stages=block_stages, annotations={'multi_buffer_eligible': rotated}
                ):
                    block_offset = vocab_block * vocab_block_size
                    valid_cols = T.min(vocab_block_size, vocab_size - block_offset)
                    T.copy(logits[token_idx, block_offset : block_offset + valid_cols], logits_ub[:valid_cols])

                    n_vec = T.ceildiv(valid_cols, vector_lanes)
                    if do_logsoftmax:
                        if inplace_logsoftmax:
                            simd_apply_block(logits_ub, lse_ub, inplace_ub, n_vec, False)
                        else:
                            simd_apply_block(logits_ub, lse_ub, logsoftmax_ub, n_vec, False)
                    if do_softmax:
                        if inplace_softmax:
                            simd_apply_block(logits_ub, lse_ub, inplace_ub, n_vec, True)
                        else:
                            simd_apply_block(logits_ub, lse_ub, softmax_ub, n_vec, True)

                    if do_logsoftmax:
                        if inplace_logsoftmax:
                            T.copy(inplace_ub[:valid_cols], logits[token_idx, block_offset : block_offset + valid_cols])
                        else:
                            T.copy(
                                logsoftmax_ub[:valid_cols],
                                logsoftmax[token_idx, block_offset : block_offset + valid_cols],
                            )
                    if do_softmax:
                        if inplace_softmax:
                            T.copy(inplace_ub[:valid_cols], logits[token_idx, block_offset : block_offset + valid_cols])
                        else:
                            T.copy(
                                softmax_ub[:valid_cols], softmax[token_idx, block_offset : block_offset + valid_cols]
                            )

    return _softmax_ascend
