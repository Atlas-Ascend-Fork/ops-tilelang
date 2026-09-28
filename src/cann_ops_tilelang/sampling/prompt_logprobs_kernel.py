"""TileLang Ascend kernel for prompt token log probabilities."""

from typing import Any

import tilelang
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores

_UB_SIZE_950DT = 188416


def _align32(value: int) -> int:
    return (value + 31) // 32 * 32


def _rows_per_block(topk_cnt: int) -> int:
    ub_cols = max(8, (topk_cnt + 7) // 8 * 8)
    cap = 12 if topk_cnt <= 2048 else 6
    for rows in range(cap, 0, -1):
        ub_bytes = 4 * rows * ub_cols + 2 * _align32(4 * rows) + 32 * rows + 256
        if ub_bytes <= _UB_SIZE_950DT:
            return rows
    raise AssertionError('topk count exceeds the prompt logprobs kernel UB capacity.')


@tilelang.jit(
    out_idx=None,
    target='ascend',
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _get_prompt_logprobs_tensor_tl_ascend(
    topk_cnt: int,
) -> Any:
    assert 0 < topk_cnt <= 47016, 'topk count must be in the range [1, 47016].'
    num_tokens = T.dynamic('num_tokens')
    logprobs_stride = T.dynamic('logprobs_stride')
    token_index_stride = T.dynamic('token_index_stride')
    num_cores = get_num_vector_cores()
    vector_length = 64
    rows_per_block = _rows_per_block(topk_cnt)
    tail_base = (topk_cnt // vector_length) * vector_length
    ub_cols = max(8, (topk_cnt + 7) // 8 * 8)

    @T.prim_func
    def _get_prompt_logprobs_tensor_ascend(
        prompt_tokens: T.Tensor[(num_tokens,), T.int32],
        logprobs: T.StridedTensor[(num_tokens, topk_cnt), (logprobs_stride, 1), T.float32],
        token_index: T.StridedTensor[(num_tokens, topk_cnt), (token_index_stride, 1), T.int32],
        output: T.Tensor[(num_tokens,), T.float32],
    ) -> None:
        with T.Kernel(T.min(T.ceildiv(num_tokens, rows_per_block), num_cores)) as core_id:
            token_index_ub = T.alloc_shared((rows_per_block, ub_cols), T.int32)
            match_ub = T.alloc_shared((rows_per_block,), T.float32)
            prompt_tokens_ub = T.alloc_shared((rows_per_block,), T.int32)
            output_ub = T.alloc_shared((rows_per_block, 8), T.float32)
            ramp_ub = T.alloc_shared((vector_length,), T.float32)  # [0..63]
            for i in T.serial(vector_length):
                ramp_ub[i] = T.cast(i, 'float32')
            total_blocks = T.ceildiv(num_tokens, rows_per_block)
            blocks_per_core = T.ceildiv(total_blocks, T.min(total_blocks, num_cores))
            block_begin = core_id * blocks_per_core
            block_end = T.min(block_begin + blocks_per_core, total_blocks)
            for block_id in T.serial(block_begin, block_end):
                row_base = block_id * rows_per_block
                valid_rows = T.min(rows_per_block, num_tokens - row_base)
                if row_base + rows_per_block <= num_tokens and topk_cnt % 8 == 0:
                    T.copy(token_index[row_base, 0], token_index_ub)
                else:
                    for row_offset in T.serial(valid_rows):
                        row = row_base + row_offset
                        T.copy(token_index[row, :], token_index_ub[row_offset, :topk_cnt])
                if row_base + rows_per_block <= num_tokens:
                    T.copy(prompt_tokens[row_base : row_base + rows_per_block], prompt_tokens_ub)
                else:
                    T.copy(prompt_tokens[row_base : row_base + valid_rows], prompt_tokens_ub[:valid_rows])
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    step_vec = T.simd.vdup(float(vector_length), 'float32', mask)
                    for row_offset in T.serial(valid_rows):
                        target = T.simd.vld(prompt_tokens_ub[row_offset], dist='BRC_B32')
                        matched = T.simd.alloc_var('float32')
                        matched = T.simd.vdup(float('-1.0'), 'float32', mask)
                        pos_vec = T.simd.alloc_var('float32')
                        pos_vec = T.simd.vld(ramp_ub[0])
                        for col_group in T.serial(topk_cnt // vector_length):
                            col = col_group * vector_length
                            index_chunk = T.simd.vld(token_index_ub[row_offset, col])
                            compare_mask = T.simd.vcmp(index_chunk, target, mask, op='eq')
                            matched = T.simd.vsel(pos_vec, matched, compare_mask)
                            pos_vec = T.simd.vadd(pos_vec, step_vec, mask)
                        matched = T.simd.vcmax(matched, mask)
                        T.simd.vsts(match_ub[row_offset], matched, mask, dist='ONEPT_B32')

                with T.SimtVF(threads=rows_per_block):
                    row_offset = T.get_thread_binding()
                    if row_offset < valid_rows:
                        for col in T.serial(topk_cnt % vector_length):
                            tc = tail_base + col
                            if token_index_ub[row_offset, tc] == prompt_tokens_ub[row_offset]:
                                match_ub[row_offset] = T.max(match_ub[row_offset], T.cast(tc, 'float32'))
                        row = row_base + row_offset
                        match_col = T.cast(match_ub[row_offset], 'int32')
                        if match_col >= 0:
                            output_ub[row_offset, 0] = logprobs[row, match_col]
                        else:
                            output_ub[row_offset, 0] = float('-inf')
                if row_base + rows_per_block <= num_tokens:
                    T.copy(output_ub[:, :1], output[row_base : row_base + rows_per_block])
                else:
                    T.copy(output_ub[:valid_rows, :1], output[row_base : row_base + valid_rows])

    return _get_prompt_logprobs_tensor_ascend
