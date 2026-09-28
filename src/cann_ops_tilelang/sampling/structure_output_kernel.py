"""TileLang Ascend kernel for applying packed token masks."""

from typing import Any

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.config import get_num_vector_cores

BITS_PER_BLOCK = 32


@tilelang.jit(out_idx=None)
def _apply_token_bitmask_inplace_tl_ascend(
    logits_vocab_size: int,
    bitmask_size: int,
    max_block_size: int,
    num_stages: int,
) -> Any:
    """Apply a packed per-token bitmask to selected logits rows on Ascend."""
    num_rows = T.dynamic('num_rows')
    logits_rows = T.dynamic('logits_rows')
    bitmask_rows = T.dynamic('bitmask_rows')
    logits_stride = T.dynamic('logits_stride')
    bitmask_stride = T.dynamic('bitmask_stride')

    num_cores = get_num_vector_cores()
    masked_vocab_size = min(logits_vocab_size, bitmask_size * BITS_PER_BLOCK)
    block_size = min(max_block_size, ((masked_vocab_size + 63) // 64) * 64)
    words_per_block = block_size // BITS_PER_BLOCK
    vec_size = 64
    num_vocab_tiles = (masked_vocab_size + block_size - 1) // block_size
    use_single_row_direct = max_block_size == 4096 and num_stages == 1
    use_input_l2_bypass = max_block_size in (8192, 10816) and num_stages == 1

    @T.prim_func
    def _apply_token_bitmask_inplace_ascend(
        logits: T.StridedTensor[(logits_rows, logits_vocab_size), (logits_stride, 1), T.float32],  # type: ignore
        bitmask: T.StridedTensor[(bitmask_rows, bitmask_size), (bitmask_stride, 1), T.int32],  # type: ignore
        indices: T.Tensor[(num_rows,), T.int32],
    ) -> None:
        if use_single_row_direct:
            # The dispatcher selects this specialization only for num_rows == 1.
            # Launch exactly one core per vocab tile and avoid Persistent setup.
            with T.Kernel(num_vocab_tiles) as pid_v:
                logits_ub = T.alloc_shared((block_size,), T.float32)
                bitmask_ub = T.alloc_shared((words_per_block,), T.int32)

                row = T.alloc_var(T.int32, init=indices[0])
                T.device_assert(
                    row >= 0 and row < logits_rows and row < bitmask_rows,
                    msg='indices must refer to valid logits and bitmask rows',
                )
                token_offset = pid_v * block_size
                word_offset = pid_v * words_per_block

                if token_offset + block_size <= masked_vocab_size:
                    T.copy(logits[row, token_offset], logits_ub)
                    T.copy(bitmask[row, word_offset], bitmask_ub)
                else:
                    valid_tokens = masked_vocab_size - token_offset
                    valid_words = T.ceildiv(valid_tokens, BITS_PER_BLOCK)
                    T.copy(logits[row, token_offset : token_offset + valid_tokens], logits_ub[:valid_tokens])
                    T.copy(bitmask[row, word_offset : word_offset + valid_words], bitmask_ub[:valid_words])

                with T.SimdVF():
                    lane_bit = S.vand(S.vci(0, T.int32), S.vdup(BITS_PER_BLOCK - 1, T.int32))
                    one = S.vdup(1, T.uint32)
                    mask_group_low = S.pset(32, 'PAT_VL32')
                    negative_infinity = S.vdup(float('-inf'), T.float32)

                    for vector in T.unroll(block_size // vec_size, explicit=True):
                        word_low = S.vld(bitmask_ub[vector * 2], dist='BRC_B32')
                        word_high = S.vld(bitmask_ub[vector * 2 + 1], dist='BRC_B32')
                        words = T.reinterpret(S.vsel(word_low, word_high, mask_group_low), 'uint32x64')
                        token_bits = S.vand(S.vshr(words, lane_bit), one)
                        allowed = S.vcmps(token_bits, 0, op='ne')
                        values = S.vld(logits_ub[vector * vec_size])
                        S.vsts(logits_ub[vector * vec_size], S.vsel(values, negative_infinity, allowed))

                if token_offset + block_size <= masked_vocab_size:
                    T.copy(logits_ub, logits[row, token_offset])
                else:
                    valid_tokens = masked_vocab_size - token_offset
                    T.copy(logits_ub[:valid_tokens], logits[row, token_offset : token_offset + valid_tokens])
        else:
            with T.Kernel(num_cores) as core_id:
                logits_ub = T.alloc_shared((block_size,), T.float32)
                bitmask_ub = T.alloc_shared((words_per_block,), T.int32)
                T.annotate_buffer_versions(
                    {
                        logits_ub: num_stages,
                        bitmask_ub: num_stages,
                    }
                )

                # Keep rows as the fastest-varying dimension. With group_size=1,
                # successive persistent workers receive different rows for a
                # fixed vocab tile, distributing every tail tile across cores.
                for pid_v, pid_n in T.Persistent(
                    [num_vocab_tiles, num_rows],
                    num_cores,
                    core_id,
                    group_size=1,
                    num_stages=num_stages,
                ):
                    row = T.alloc_var(T.int32, init=indices[pid_n])
                    T.device_assert(
                        row >= 0 and row < logits_rows and row < bitmask_rows,
                        msg='indices must refer to valid logits and bitmask rows',
                    )
                    token_offset = pid_v * block_size
                    word_offset = pid_v * words_per_block

                    if token_offset + block_size <= masked_vocab_size:
                        if use_input_l2_bypass:
                            T.copy(logits[row, token_offset], logits_ub, l2_cache_ctrl='NOTALLOC_KEEP')
                            T.copy(bitmask[row, word_offset], bitmask_ub, l2_cache_ctrl='NOTALLOC_KEEP')
                        else:
                            T.copy(logits[row, token_offset], logits_ub)
                            T.copy(bitmask[row, word_offset], bitmask_ub)
                    else:
                        valid_tokens = masked_vocab_size - token_offset
                        valid_words = T.ceildiv(valid_tokens, BITS_PER_BLOCK)
                        if use_input_l2_bypass:
                            T.copy(
                                logits[row, token_offset : token_offset + valid_tokens],
                                logits_ub[:valid_tokens],
                                l2_cache_ctrl='NOTALLOC_KEEP',
                            )
                            T.copy(
                                bitmask[row, word_offset : word_offset + valid_words],
                                bitmask_ub[:valid_words],
                                l2_cache_ctrl='NOTALLOC_KEEP',
                            )
                        else:
                            T.copy(logits[row, token_offset : token_offset + valid_tokens], logits_ub[:valid_tokens])
                            T.copy(bitmask[row, word_offset : word_offset + valid_words], bitmask_ub[:valid_words])

                    with T.SimdVF():
                        lane_bit = S.vand(S.vci(0, T.int32), S.vdup(BITS_PER_BLOCK - 1, T.int32))
                        one = S.vdup(1, T.uint32)
                        mask_group_low = S.pset(32, 'PAT_VL32')
                        negative_infinity = S.vdup(float('-inf'), T.float32)

                        for vector in T.unroll(block_size // vec_size, explicit=True):
                            word_low = S.vld(bitmask_ub[vector * 2], dist='BRC_B32')
                            word_high = S.vld(bitmask_ub[vector * 2 + 1], dist='BRC_B32')
                            words = T.reinterpret(S.vsel(word_low, word_high, mask_group_low), 'uint32x64')
                            token_bits = S.vand(S.vshr(words, lane_bit), one)
                            allowed = S.vcmps(token_bits, 0, op='ne')
                            values = S.vld(logits_ub[vector * vec_size])
                            S.vsts(logits_ub[vector * vec_size], S.vsel(values, negative_infinity, allowed))

                    if token_offset + block_size <= masked_vocab_size:
                        T.copy(logits_ub, logits[row, token_offset])
                    else:
                        valid_tokens = masked_vocab_size - token_offset
                        T.copy(logits_ub[:valid_tokens], logits[row, token_offset : token_offset + valid_tokens])

    return _apply_token_bitmask_inplace_ascend
