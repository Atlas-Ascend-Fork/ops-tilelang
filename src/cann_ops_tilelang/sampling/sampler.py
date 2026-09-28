"""Sampling and speculative-decoding helpers for Ascend."""

import torch

from cann_ops_tilelang.config import require_ascend

from .sampler_kernel import (
    _batch_find_value_by_index_tl_ascend,
    _compute_confidence_error_tl_ascend,
    _find_value_by_index_tl_ascend,
    _get_accept_result_tl_ascend,
    _get_logits_cur_by_step_tl_ascend,
    _sampled_tokens_update_tl_ascend,
    _scatter_verified_tl_ascend,
    _update_verify_states_tl_ascend,
)


def _require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f'{name} must have shape {expected}, got {tuple(tensor.shape)}')


def find_value_by_index(index: torch.Tensor, value: torch.Tensor, q_index: torch.Tensor) -> torch.Tensor:
    """Find one value for each requested token index."""
    require_ascend()
    force_topk = index.shape[-1]
    assert index.stride(0) % force_topk == 0 and index.stride(1) % force_topk == 0
    assert index.stride(-1) == 1 and value.stride(-1) == 1 and (q_index.stride(-1) == 1)
    out = torch.empty((index.shape[0], index.shape[1]), dtype=torch.float32, device=index.device)
    assert index.shape[-1] % 512 == 0
    get_kernel = _find_value_by_index_tl_ascend
    kernel = get_kernel(force_topk)
    kernel(index, value, q_index, out)
    return out


def batchfind_value_by_index_ref(
    index: torch.Tensor, value: torch.Tensor, q_index: torch.Tensor, vocab_size: int, use_pdl: bool = False
) -> torch.Tensor:
    require_ascend()
    num_steps, num_requests = (index.shape[0], index.shape[1])
    num_queries = q_index.shape[-1]
    index_flat = index.reshape(-1, index.shape[-1])
    value_flat = value.reshape(-1, index.shape[-1])
    q_index_flat = q_index.reshape(-1, num_queries)
    pool = torch.full((num_steps * num_requests, vocab_size), float('-inf'), dtype=torch.float32, device=index.device)
    pool.scatter_(1, q_index_flat, float('-inf'))
    pool.scatter_(1, index_flat, value_flat)
    out_flat = pool.gather(-1, q_index_flat)
    return out_flat.view(num_steps, num_requests, num_queries)


def batch_find_value_by_index(
    index: torch.Tensor, value: torch.Tensor, q_index: torch.Tensor, vocab_size: int, use_pdl: bool = False
) -> torch.Tensor:
    """Find values for a batch of requested token indices."""
    require_ascend()
    assert index.stride(-1) == 1 and value.stride(-1) == 1
    ft, qd = (index.shape[-1], q_index.shape[-1])
    qcap = None
    qcap = 2048 if qd > ft else None
    use_shared_memory = (
        ft > 0
        and ft % 512 == 0
        and vocab_size <= 2**20
        and (qd <= ft <= 4096 or (qcap is not None and ft in (512, 1024) and (qd <= 2048)))
    )
    if use_shared_memory:
        num_steps, num_requests = (index.shape[0], index.shape[1])
        out = torch.empty((num_steps, num_requests, qd), dtype=torch.float32, device=index.device)
        kernel = _batch_find_value_by_index_tl_ascend(ft, vocab_size, use_pdl, qcap)
        kernel(index, value, q_index, out)
        return out
    return batchfind_value_by_index_ref(index, value, q_index, vocab_size)


def get_accept_result(
    accept_result_verify: torch.Tensor,
    so_valid: torch.Tensor | None,
    golden_output: torch.Tensor | None,
    last_next_n_token_ids: torch.Tensor,
    gold_accepted: torch.Tensor | None,
    is_verify: torch.Tensor,
    sample_reject_cur: torch.Tensor,
    fresh_sample_final: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    require_ascend()
    if accept_result_verify.ndim != 2:
        raise ValueError('accept_result_verify must be two-dimensional')
    m, n = accept_result_verify.shape
    for name, tensor in (
        ('last_next_n_token_ids', last_next_n_token_ids),
        ('sample_reject_cur', sample_reject_cur),
    ):
        _require_shape(name, tensor, (m, n))
    if is_verify.ndim != 2 or is_verify.shape[1] != n or is_verify.shape[0] < m:
        raise ValueError('is_verify must have shape (max_seqlen_q, N) with max_seqlen_q >= M')
    _require_shape('fresh_sample_final', fresh_sample_final, tuple(is_verify.shape))
    if so_valid is not None:
        _require_shape('so_valid', so_valid, (m, n))
    if (golden_output is None) != (gold_accepted is None):
        raise ValueError('golden_output and gold_accepted must be provided together')
    if golden_output is not None:
        if golden_output.ndim != 2 or golden_output.shape[1] != n or golden_output.shape[0] > m:
            raise ValueError('golden_output must have shape (golden_steps, N) with golden_steps <= M')
        _require_shape('gold_accepted', gold_accepted, tuple(golden_output.shape))
    max_seqlen_q = is_verify.shape[0]
    sample_results_cur = torch.empty((max_seqlen_q, n), dtype=torch.int32, device=accept_result_verify.device)
    accept_result = torch.empty((max_seqlen_q, n), dtype=torch.bool, device=accept_result_verify.device)
    has_so_valid_flag = so_valid is not None
    has_golden_output_flag = golden_output is not None
    get_kernel = _get_accept_result_tl_ascend
    kernel = get_kernel(has_so_valid_flag, has_golden_output_flag)
    kernel(
        accept_result_verify,
        so_valid,
        golden_output,
        last_next_n_token_ids,
        gold_accepted,
        is_verify,
        sample_reject_cur,
        fresh_sample_final,
        sample_results_cur,
        accept_result,
    )
    return (sample_results_cur, accept_result)


def get_logits_cur_by_step(logits: torch.Tensor, ftp: torch.Tensor, has_token: torch.Tensor, step: int) -> torch.Tensor:
    require_ascend()
    assert logits.stride(1) == 1 and logits.stride(0) % 16 == 0
    n = ftp.shape[0]
    vocab_size = logits.shape[1]
    out = torch.empty((n, vocab_size), dtype=logits.dtype, device=logits.device)
    kernel = _get_logits_cur_by_step_tl_ascend(vocab_size)
    kernel(logits, ftp, has_token, out, step)
    return out


def update_verify_states(
    accept_result_all: torch.Tensor,
    has_token_all: torch.Tensor,
    seqlen_q: torch.Tensor,
    sample_results_cur_all: torch.Tensor,
    cur_step_mtp_value: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    require_ascend()
    num_steps, num_requests = accept_result_all.shape
    device = accept_result_all.device
    accepted_all = torch.empty((num_steps, num_requests), dtype=torch.int32, device=device)
    is_final_all = torch.empty((num_steps, num_requests), dtype=torch.bool, device=device)
    last_offset = torch.empty(num_requests, dtype=torch.int32, device=device)
    num_gen = torch.empty(num_steps, dtype=torch.int32, device=device)
    num_gen.zero_()
    kernel = _update_verify_states_tl_ascend()
    kernel(
        accept_result_all,
        has_token_all,
        seqlen_q,
        sample_results_cur_all,
        cur_step_mtp_value,
        accepted_all,
        is_final_all,
        last_offset,
        num_gen,
    )
    return (accepted_all, is_final_all, last_offset, num_gen[:cur_step_mtp_value])


def scatter_verified(
    is_verify: torch.Tensor,
    ftp: torch.Tensor,
    accepted_tokens: torch.Tensor,
    token_logprobs_cur: torch.Tensor,
    sum_smaller_probs_cur: torch.Tensor | None,
    last_nextn_token_logprobs: torch.Tensor | None,
    top_logprobs_cur: torch.Tensor | None,
    top_token_ids_cur: torch.Tensor | None,
    total_q: int,
) -> tuple:
    """Scatter verified tokens and their optional statistics."""
    require_ascend()
    if is_verify.ndim != 2:
        raise ValueError('is_verify must be two-dimensional')
    steps, req_cnt = is_verify.shape
    _require_shape('ftp', ftp, (req_cnt,))
    _require_shape('accepted_tokens', accepted_tokens, (steps, req_cnt))
    _require_shape('token_logprobs_cur', token_logprobs_cur, (steps, req_cnt))
    if (sum_smaller_probs_cur is None) != (last_nextn_token_logprobs is None):
        raise ValueError('sum_smaller_probs_cur and last_nextn_token_logprobs must be provided together')
    if sum_smaller_probs_cur is not None:
        _require_shape('sum_smaller_probs_cur', sum_smaller_probs_cur, (steps, req_cnt))
        _require_shape('last_nextn_token_logprobs', last_nextn_token_logprobs, (steps, req_cnt))
    if (top_logprobs_cur is None) != (top_token_ids_cur is None):
        raise ValueError('top_logprobs_cur and top_token_ids_cur must be provided together')
    if top_logprobs_cur is not None:
        if top_logprobs_cur.ndim != 3 or top_logprobs_cur.shape[:2] != (steps, req_cnt):
            raise ValueError('top_logprobs_cur must have shape (steps, req_cnt, topk_n)')
        if top_logprobs_cur.shape[2] <= 0:
            raise ValueError('topk_n must be positive when top-k tensors are provided')
        _require_shape('top_token_ids_cur', top_token_ids_cur, tuple(top_logprobs_cur.shape))
    device = is_verify.device
    is_rl_or_eval = sum_smaller_probs_cur is not None
    has_topk_n = top_logprobs_cur is not None
    topk_n = top_logprobs_cur.shape[2] if top_logprobs_cur is not None else 0
    topk_stage_bytes = 8 * ((topk_n + 15) // 16 * 16)
    if topk_stage_bytes > 240000:
        raise ValueError('topk_n exceeds the scatter_verified kernel UB capacity.')
    verified_token_ids = torch.empty(total_q, dtype=torch.int32, device=device)
    verified_token_logprobs = torch.empty(total_q, dtype=torch.float32, device=device)
    verified_sum_smaller_probs = torch.empty(total_q, dtype=torch.float32, device=device) if is_rl_or_eval else None
    accepted_token_last_logprobs = torch.empty(total_q, dtype=torch.float32, device=device) if is_rl_or_eval else None
    verified_topk_logps = torch.empty(total_q, topk_n, dtype=torch.float32, device=device) if has_topk_n else None
    verified_topk_tokens = torch.empty(total_q, topk_n, dtype=torch.int32, device=device) if has_topk_n else None
    top_token_ids_cur_dtype = torch.int32 if top_token_ids_cur is None else top_token_ids_cur.dtype
    kernel = _scatter_verified_tl_ascend(is_rl_or_eval, has_topk_n, topk_n, top_token_ids_cur_dtype)
    kernel(
        is_verify,
        ftp,
        accepted_tokens,
        token_logprobs_cur,
        sum_smaller_probs_cur,
        last_nextn_token_logprobs,
        verified_token_ids,
        verified_token_logprobs,
        verified_sum_smaller_probs,
        accepted_token_last_logprobs,
        top_logprobs_cur,
        top_token_ids_cur,
        verified_topk_logps,
        verified_topk_tokens,
    )
    return (
        verified_token_ids,
        verified_token_logprobs,
        verified_sum_smaller_probs,
        accepted_token_last_logprobs,
        verified_topk_logps,
        verified_topk_tokens,
    )


def sampled_tokens_update(
    is_final: torch.Tensor,
    sample_results_cur: torch.Tensor,
    token_logprobs_cur: torch.Tensor,
    sum_smaller_probs_cur: torch.Tensor | None,
    top_logprobs_cur: torch.Tensor | None,
    top_token_ids_cur: torch.Tensor | None,
    token_ids_out: torch.Tensor,
    token_logprobs_out: torch.Tensor,
    sum_smaller_probs_out: torch.Tensor | None,
    top_logps_out: torch.Tensor | None,
    top_tokens_out: torch.Tensor | None,
) -> None:
    """Scatter the last final step for each request into sampled-token outputs."""
    require_ascend()
    if is_final.ndim != 2:
        raise ValueError('is_final must be two-dimensional')
    steps, req_cnt = is_final.shape
    _require_shape('sample_results_cur', sample_results_cur, (steps, req_cnt))
    _require_shape('token_logprobs_cur', token_logprobs_cur, (steps, req_cnt))
    _require_shape('token_ids_out', token_ids_out, (req_cnt,))
    _require_shape('token_logprobs_out', token_logprobs_out, (req_cnt,))
    if (sum_smaller_probs_cur is None) != (sum_smaller_probs_out is None):
        raise ValueError('sum_smaller_probs_cur and sum_smaller_probs_out must be provided together')
    if sum_smaller_probs_cur is not None:
        _require_shape('sum_smaller_probs_cur', sum_smaller_probs_cur, (steps, req_cnt))
        _require_shape('sum_smaller_probs_out', sum_smaller_probs_out, (req_cnt,))
    if (top_logprobs_cur is None) != (top_logps_out is None):
        raise ValueError('top_logprobs_cur and top_logps_out must be provided together')
    if (top_token_ids_cur is None) != (top_tokens_out is None):
        raise ValueError('top_token_ids_cur and top_tokens_out must be provided together')
    has_sum_smaller = sum_smaller_probs_cur is not None and sum_smaller_probs_out is not None
    has_top_logprobs = top_logprobs_cur is not None and top_logps_out is not None
    has_top_token_ids = top_token_ids_cur is not None and top_tokens_out is not None
    topk_n = top_logprobs_cur.shape[-1] if has_top_logprobs else 0
    if has_top_logprobs:
        if top_logprobs_cur.ndim != 3 or top_logprobs_cur.shape[:2] != (steps, req_cnt) or topk_n <= 0:
            raise ValueError('top_logprobs_cur must have shape (steps, req_cnt, positive topk_n)')
        _require_shape('top_logps_out', top_logps_out, (req_cnt, topk_n))
    if has_top_token_ids:
        if top_token_ids_cur.ndim != 3 or top_token_ids_cur.shape[:2] != (steps, req_cnt):
            raise ValueError('top_token_ids_cur must have shape (steps, req_cnt, topk_n)')
        token_topk_n = top_token_ids_cur.shape[-1]
        if token_topk_n <= 0:
            raise ValueError('topk_n must be positive when top-token tensors are provided')
        _require_shape('top_tokens_out', top_tokens_out, (req_cnt, token_topk_n))
        if has_top_logprobs and token_topk_n != topk_n:
            raise ValueError('top-logprob and top-token tensors must use the same top-k size')
        topk_n = token_topk_n
    stage_bytes = 576 + (256 if has_sum_smaller else 0) + 4 * topk_n * (int(has_top_logprobs) + int(has_top_token_ids))
    if stage_bytes > 240000:
        raise ValueError('topk_n exceeds the sampled_tokens_update kernel UB capacity.')
    top_token_ids_cur_dtype = torch.int32 if top_token_ids_cur is None else top_token_ids_cur.dtype
    kernel = _sampled_tokens_update_tl_ascend(
        has_sum_smaller,
        has_top_logprobs,
        has_top_token_ids,
        topk_n,
        top_token_ids_cur_dtype,
    )
    kernel(
        is_final,
        sample_results_cur,
        token_logprobs_cur,
        sum_smaller_probs_cur,
        top_logprobs_cur,
        top_token_ids_cur,
        token_ids_out,
        token_logprobs_out,
        sum_smaller_probs_out,
        top_logps_out,
        top_tokens_out,
    )


def compute_confidence_error(
    last_nextn_token_logprobs: torch.Tensor,
    nextn_token_logprobs: torch.Tensor,
    confidence: torch.Tensor,
    verify_masks: torch.Tensor,
) -> torch.Tensor:
    require_ascend()
    device = last_nextn_token_logprobs.device
    num_steps = confidence.shape[0]
    out = torch.empty(1, dtype=torch.float32, device=device)
    temp = torch.empty(num_steps, dtype=torch.float32, device=device)
    out.zero_()
    kernel = _compute_confidence_error_tl_ascend()
    kernel(last_nextn_token_logprobs, nextn_token_logprobs, confidence, verify_masks, out, temp)
    return out.view(())
