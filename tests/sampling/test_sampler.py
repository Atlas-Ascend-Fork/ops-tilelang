from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.sampling.sampler import (
    batch_find_value_by_index,
    batchfind_value_by_index_ref,
    compute_confidence_error,
    find_value_by_index,
    get_accept_result,
    get_logits_cur_by_step,
    sampled_tokens_update,
    scatter_verified,
    update_verify_states,
)

pytestmark = pytest.mark.npu


def find_value_by_index_ref(
    index: torch.Tensor,
    value: torch.Tensor,
    q_index: torch.Tensor,
) -> torch.Tensor:
    mask = index == q_index
    v = torch.where(mask, value, float('-inf'))
    return torch.max(v, dim=-1).values


@pytest.mark.parametrize(
    'max_seqlen_q,req_cnt',
    [
        (1, 1),
        (1, 3),
        (1, 7),
        (1, 233),
        (1, 256),
        (1, 512),
        (1, 1024),
        (3, 7),
        (3, 233),
        (5, 233),
    ],
)
@pytest.mark.parametrize('force_topk', [512, 1024, 4096])
def test_find_value_by_index(max_seqlen_q: int, req_cnt: int, force_topk: int) -> None:
    n = max_seqlen_q * req_cnt
    steps = torch.randint(1, 10000, (n, force_topk), device=get_device(), dtype=torch.int32)
    index_flat = torch.cumsum(steps, dim=1, dtype=torch.int32)
    for i in range(n):
        perm = torch.randperm(force_topk, device=get_device())
        index_flat[i] = index_flat[i][perm]
    value_flat = torch.randn((n, force_topk), dtype=torch.float32, device=get_device())

    mask = torch.rand(n, device=index_flat.device) < 0.9
    force_topk_val = index_flat.shape[1]
    indices_idx = torch.randint(0, force_topk_val, (n,), device=index_flat.device)
    from_index = index_flat[torch.arange(n), indices_idx]
    random_vals = torch.randint(0, 10**9, (n,), dtype=torch.int32, device=index_flat.device)
    q_index_flat = torch.where(mask, from_index, random_vals).unsqueeze(-1)

    num_steps, num_requests = max_seqlen_q, req_cnt
    force_topk_size = force_topk
    stride_factor = 7
    big_index = torch.zeros(
        num_steps * stride_factor, num_requests * stride_factor, force_topk_size, dtype=torch.int32, device=get_device()
    )
    big_value = torch.zeros(
        num_steps * stride_factor,
        num_requests * stride_factor,
        force_topk_size,
        dtype=torch.float32,
        device=get_device(),
    )
    big_qindex = torch.zeros(
        num_steps * stride_factor, num_requests * stride_factor, 1, dtype=torch.int32, device=get_device()
    )
    big_index[::stride_factor, ::stride_factor, :] = index_flat.view(num_steps, num_requests, force_topk_size)
    big_value[::stride_factor, ::stride_factor, :] = value_flat.view(num_steps, num_requests, force_topk_size)
    big_qindex[::stride_factor, ::stride_factor, :] = q_index_flat.view(num_steps, num_requests, 1)
    index = big_index[: num_steps * stride_factor : stride_factor, : num_requests * stride_factor : stride_factor, :]
    value = big_value[: num_steps * stride_factor : stride_factor, : num_requests * stride_factor : stride_factor, :]
    q_index = big_qindex[: num_steps * stride_factor : stride_factor, : num_requests * stride_factor : stride_factor, :]

    out = find_value_by_index(index, value, q_index)
    ref = find_value_by_index_ref(index, value, q_index)
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    'max_seqlen_q,req_cnt',
    [
        (1, 1),
        (1, 3),
        (1, 7),
        (1, 233),
        (1, 256),
        (1, 512),
        (1, 1024),
        (3, 7),
        (3, 233),
        (5, 233),
    ],
)
@pytest.mark.parametrize('force_topk', [512, 1024, 4096])
@pytest.mark.parametrize('query_dim', [512, 800, 1024, 2048])
def test_batch_find_value_by_index(
    max_seqlen_q: int,
    req_cnt: int,
    force_topk: int,
    query_dim: int,
    vocab_size: int = 129280,
) -> None:
    n = max_seqlen_q * req_cnt
    index_flat = torch.empty((n, force_topk), dtype=torch.int32, device=get_device())
    for i in range(n):
        index_flat[i] = torch.randperm(vocab_size, device=get_device())[:force_topk]
    value_flat = torch.randn((n, force_topk), dtype=torch.float32, device=get_device())

    mask = torch.rand((n, query_dim), device=index_flat.device) < 0.9
    indices_idx = torch.randint(0, force_topk, (n, query_dim), device=index_flat.device)
    from_index = index_flat[torch.arange(n, device=get_device()).reshape(-1, 1).repeat(1, query_dim), indices_idx]
    random_vals = torch.randint(0, vocab_size, (n, query_dim), dtype=torch.int32, device=index_flat.device)
    q_index_flat = torch.where(mask, from_index, random_vals)

    index = index_flat.unflatten(0, (max_seqlen_q, req_cnt))
    value = value_flat.unflatten(0, (max_seqlen_q, req_cnt))
    q_index = q_index_flat.unflatten(0, (max_seqlen_q, req_cnt))

    out = batch_find_value_by_index(index, value, q_index, vocab_size)
    ref = batchfind_value_by_index_ref(index, value, q_index, vocab_size)
    assert torch.equal(out, ref)


@pytest.mark.parametrize('n,force_topk,query_dim,vocab_size', [(512, 1024, 1024, 10**6)])
def test_batch_find_value_by_index_special(n: int, force_topk: int, query_dim: int, vocab_size: int) -> None:
    test_batch_find_value_by_index(1, n, force_topk, query_dim, vocab_size)


def get_accept_result_ref(
    accept_result_verify: torch.Tensor,
    so_valid: torch.Tensor | None,
    golden_output: torch.Tensor | None,
    last_next_n_token_ids: torch.Tensor,
    gold_accepted: torch.Tensor | None,
    is_verify: torch.Tensor,
    sample_reject_cur: torch.Tensor,
    fresh_sample_final: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = is_verify.device
    max_seqlen_q = is_verify.shape[0]
    num_steps = accept_result_verify.shape[0]
    if so_valid is not None:
        accept_result_verify = torch.where(so_valid, accept_result_verify, False)
    if golden_output is not None:
        gs = min(golden_output.shape[0], accept_result_verify.shape[0])
        accept_result_verify[:gs] = torch.where(
            golden_output[:gs] == -1,
            accept_result_verify[:gs],
            torch.where(golden_output[:gs] == last_next_n_token_ids[:gs], gold_accepted[:gs], False),
        )
        sample_reject_cur[:gs] = torch.where(golden_output[:gs] == -1, sample_reject_cur[:gs], golden_output[:gs])

    sample_verify = torch.where(accept_result_verify, last_next_n_token_ids, sample_reject_cur)
    sample_verify = torch.cat(
        [
            sample_verify,
            torch.zeros(max_seqlen_q - num_steps, sample_verify.shape[1], dtype=sample_verify.dtype, device=device),
        ]
    )
    accept_result_verify = torch.cat(
        [
            accept_result_verify,
            torch.zeros(max_seqlen_q - num_steps, accept_result_verify.shape[1], dtype=torch.bool, device=device),
        ]
    )
    sample_results_cur = torch.where(is_verify, sample_verify, fresh_sample_final)
    accept_result = torch.where(is_verify, accept_result_verify, True)

    return sample_results_cur, accept_result


@pytest.mark.parametrize('m', [1, 3, 5])
@pytest.mark.parametrize('n', [1, 7, 4096, 10077])
@pytest.mark.parametrize('has_so_valid', [True, False])
@pytest.mark.parametrize('has_golden_output,partial_golden', [(False, False), (True, False), (True, True)])
def test_get_accept_result(m: int, n: int, has_so_valid: bool, has_golden_output: bool, partial_golden: bool) -> None:
    max_seqlen_q = m + 1
    so_valid = torch.rand(m, n, device=get_device()) < 0.8 if has_so_valid else None
    accept_result_verify = torch.rand(m, n, device=get_device()) < 0.8
    is_verify = torch.rand(max_seqlen_q, n, device=get_device()) < 0.6
    is_verify[max_seqlen_q - 1, :] = False  # 确实是这样
    sample_reject_cur = torch.randint(0, 129280, (m, n), dtype=torch.int32, device=get_device())
    fresh_sample_final = torch.randint(0, 129280, (max_seqlen_q, n), dtype=torch.int32, device=get_device())
    if has_golden_output:
        golden_steps = m // 2 + 1 if partial_golden else m
        golden_output = torch.where(
            torch.rand(n, golden_steps, device=get_device()) < 0.3,
            torch.tensor(-1, dtype=torch.int32, device=get_device()),
            torch.randint(0, 10000, (n, golden_steps), dtype=torch.int32, device=get_device()),
        ).t()
        last_next_n_token_ids = torch.randint(0, 129280, (n, m), dtype=torch.int32, device=get_device()).t()
        gold_accepted = (torch.rand(n, golden_steps, device=get_device()) < 0.8).t()
    else:
        golden_output = None
        last_next_n_token_ids = torch.randint(0, 129280, (n, m), dtype=torch.int32, device=get_device()).t()
        gold_accepted = None

    def fn() -> tuple[torch.Tensor, torch.Tensor]:
        return get_accept_result(
            accept_result_verify,
            so_valid,
            golden_output,
            last_next_n_token_ids,
            gold_accepted,
            is_verify,
            sample_reject_cur,
            fresh_sample_final,
        )

    def fn_ref() -> tuple[torch.Tensor, torch.Tensor]:
        return get_accept_result_ref(
            accept_result_verify,
            so_valid,
            golden_output,
            last_next_n_token_ids,
            gold_accepted,
            is_verify,
            sample_reject_cur,
            fresh_sample_final,
        )

    sample_results_cur0, accept_result0 = fn()
    sample_results_cur1, accept_result1 = fn_ref()
    assert torch.equal(accept_result0, accept_result1), f'accept_result mismatch\n{accept_result0}\n{accept_result1}'
    assert torch.equal(sample_results_cur0, sample_results_cur1), (
        f'sample_results_cur mismatch\n{sample_results_cur0}\n{sample_results_cur1}'
    )


def get_logits_cur_by_step_ref(
    logits: torch.Tensor,
    ftp: torch.Tensor,
    has_token: torch.Tensor,
    step: int,
) -> torch.Tensor:
    gather_idx = torch.where(has_token, ftp + step, ftp)
    logits = logits[gather_idx].clone()
    logits.masked_fill_(torch.logical_not(has_token).unsqueeze(-1), 0)
    return logits


@pytest.mark.parametrize('n', [1, 3, 5, 256, 512, 1024])
@pytest.mark.parametrize('maxlen,step', [(8, 0), (8, 1), (8, 3), (5, 0), (5, 1), (5, 3)])
def test_get_logits_cur_by_step(
    n: int,
    maxlen: int,
    step: int,
    vocab_size: int = 129280,
) -> None:
    qlen = torch.randint(1, maxlen, (n,), device=get_device())
    ftp = torch.cat([torch.tensor([0], device=get_device()), torch.cumsum(qlen, 0)[:-1]]).int()
    has_token = step < qlen
    logits = torch.randn((torch.sum(qlen).item(), vocab_size), dtype=torch.float, device=get_device())

    def fn() -> torch.Tensor:
        return get_logits_cur_by_step(logits, ftp, has_token, step)

    def fn_torch() -> torch.Tensor:
        return get_logits_cur_by_step_ref(logits, ftp, has_token, step)

    out0 = fn_torch()
    out1 = fn()
    assert torch.equal(out0, out1), f'Mismatch!\n{out0}\n{out1}'


def update_verify_states_ref(
    accept_result_all: torch.Tensor,
    has_token_all: torch.Tensor,
    seqlen_q: torch.Tensor,
    sample_results_cur_all: torch.Tensor,
    cur_step_mtp_value: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_steps, num_requests = accept_result_all.shape
    device = accept_result_all.device
    accepted_all = torch.empty((num_steps, num_requests), dtype=torch.int32, device=device)
    is_final_all = torch.empty((num_steps, num_requests), dtype=torch.bool, device=device)
    valid_req = torch.ones(num_requests, dtype=torch.bool, device=device)
    last_offset = seqlen_q.clone() - 1
    num_gen = torch.zeros(num_steps, dtype=torch.int32, device=device)
    for step_i in range(num_steps):
        is_last = step_i + 1 == seqlen_q
        not_acc = torch.logical_not(accept_result_all[step_i])
        vh = valid_req & has_token_all[step_i]
        is_final_all[step_i] = vh & (is_last | not_acc)
        last_offset[:] = torch.where(vh & not_acc, step_i, last_offset)
        valid_req[:] = vh & accept_result_all[step_i] & torch.logical_not(is_last)
        accepted_all[step_i] = torch.where(valid_req, sample_results_cur_all[step_i], -1)
        if step_i < cur_step_mtp_value:
            num_gen[step_i] = valid_req.sum().to(torch.int32)
    return accepted_all, is_final_all, last_offset, num_gen[:cur_step_mtp_value]


@pytest.mark.parametrize('req_cnt,max_seqlen_q', [(1, 2), (7, 3), (32, 5), (64, 3), (256, 5), (512, 7)])
def test_update_verify_states(req_cnt: int, max_seqlen_q: int) -> None:
    device = get_device()
    cur_step_mtp_value = max_seqlen_q - 1
    accept_result_all = torch.rand(max_seqlen_q, req_cnt, device=device) > 0.4
    has_token_all = torch.rand(max_seqlen_q, req_cnt, device=device) > 0.2
    seqlen_q = torch.randint(max_seqlen_q, max_seqlen_q + 3, (req_cnt,), dtype=torch.int32, device=device)
    sample_results_cur_all = torch.randint(0, 129280, (max_seqlen_q, req_cnt), dtype=torch.int32, device=device)

    accepted_tl, is_final_tl, last_off_tl, num_gen_tl = update_verify_states(
        accept_result_all,
        has_token_all,
        seqlen_q,
        sample_results_cur_all,
        cur_step_mtp_value,
    )

    accepted_ref, is_final_ref, last_off_ref, num_gen_ref = update_verify_states_ref(
        accept_result_all,
        has_token_all,
        seqlen_q,
        sample_results_cur_all,
        cur_step_mtp_value,
    )

    assert torch.equal(accepted_tl, accepted_ref), f'accepted_tokens mismatch\n{accepted_tl}\n{accepted_ref}'
    assert torch.equal(is_final_tl, is_final_ref), f'is_final mismatch\n{is_final_tl}\n{is_final_ref}'
    assert torch.equal(last_off_tl, last_off_ref), f'last_valid_offset mismatch\n{last_off_tl}\n{last_off_ref}'
    assert torch.equal(num_gen_tl, num_gen_ref), f'num_generated mismatch\n{num_gen_tl}\n{num_gen_ref}'


@pytest.mark.benchmark
@pytest.mark.parametrize('req_cnt,max_seqlen_q', [(64, 3), (256, 5), (512, 7)])
def test_update_verify_states_benchmark(
    req_cnt: int,
    max_seqlen_q: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    device = get_device()
    cur_step_mtp_value = max_seqlen_q - 1
    accept_result_all = torch.rand(max_seqlen_q, req_cnt, device=device) > 0.4
    has_token_all = torch.rand(max_seqlen_q, req_cnt, device=device) > 0.2
    seqlen_q = torch.randint(max_seqlen_q, max_seqlen_q + 3, (req_cnt,), dtype=torch.int32, device=device)
    sample_results_cur_all = torch.randint(0, 129280, (max_seqlen_q, req_cnt), dtype=torch.int32, device=device)

    t_us = benchmark_timer(
        lambda: update_verify_states(
            accept_result_all,
            has_token_all,
            seqlen_q,
            sample_results_cur_all,
            cur_step_mtp_value,
        )
    )

    num_steps, num_requests = max_seqlen_q, req_cnt
    total_bytes = (
        num_steps * num_requests * (1 + 1 + 8 + 8 + 1) + num_requests * 4 + num_requests * 4 + cur_step_mtp_value * 4
    )

    benchmark_record(
        kernel='sampler',
        operation='update_verify_states',
        params={'max_seqlen_q': max_seqlen_q, 'req_cnt': req_cnt},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1e9 / (t_us / 1e6)},
    )


def scatter_verified_ref(
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
    """PyTorch reference implementation for multi-step scatter.

    Same signal convention as scatter_verified:
      sum_smaller_probs_cur is not None -> has_sum_smaller
      last_nextn_token_logprobs is not None -> has_last_logprobs
      top_logprobs_cur is not None -> has_topk_n

    Returns a 6-tuple (same order as scatter_verified).
    """
    device = is_verify.device
    is_rl_or_eval = sum_smaller_probs_cur is not None
    has_topk_n = top_logprobs_cur is not None

    topk_n = top_logprobs_cur.shape[2] if top_logprobs_cur is not None else 0

    verified_token_ids = torch.full((total_q,), -1, dtype=torch.int32, device=device)
    verified_token_logprobs = torch.zeros(total_q, dtype=torch.float32, device=device)
    verified_sum_smaller_probs = torch.zeros(total_q, dtype=torch.float32, device=device) if is_rl_or_eval else None
    accepted_token_last_logprobs = torch.zeros(total_q, dtype=torch.float32, device=device) if is_rl_or_eval else None
    verified_topk_logps = torch.zeros(total_q, topk_n, dtype=torch.float32, device=device) if has_topk_n else None
    verified_topk_tokens = torch.zeros(total_q, topk_n, dtype=torch.int32, device=device) if has_topk_n else None

    steps = is_verify.shape[0]
    for step_i in range(steps):
        is_verify_step = is_verify[step_i]  # [req_cnt]
        fill_idx = torch.where(is_verify_step, ftp + step_i, ftp)
        verified_token_ids[fill_idx] = torch.where(
            is_verify_step, accepted_tokens[step_i], verified_token_ids[fill_idx]
        )
        verified_token_logprobs[fill_idx] = torch.where(
            is_verify_step, token_logprobs_cur[step_i], verified_token_logprobs[fill_idx]
        )
        if is_rl_or_eval:
            verified_sum_smaller_probs[fill_idx] = torch.where(
                is_verify_step, sum_smaller_probs_cur[step_i], verified_sum_smaller_probs[fill_idx]
            )
            accepted_token_last_logprobs[fill_idx] = torch.where(
                is_verify_step, last_nextn_token_logprobs[step_i], accepted_token_last_logprobs[fill_idx]
            )
        if has_topk_n:
            verified_topk_logps[fill_idx] = torch.where(
                is_verify_step.unsqueeze(-1), top_logprobs_cur[step_i], verified_topk_logps[fill_idx]
            )
            verified_topk_tokens[fill_idx] = torch.where(
                is_verify_step.unsqueeze(-1), top_token_ids_cur[step_i], verified_topk_tokens[fill_idx]
            )

    return (
        verified_token_ids,
        verified_token_logprobs,
        verified_sum_smaller_probs,
        accepted_token_last_logprobs,
        verified_topk_logps,
        verified_topk_tokens,
    )


@pytest.mark.parametrize('req_cnt', [1, 7, 512])
@pytest.mark.parametrize('has_topk_n', [True, False])
@pytest.mark.parametrize('is_rl_or_eval', [True, False])
@pytest.mark.parametrize('max_seqlen_q', [2, 6])
def test_scatter_verified(
    req_cnt: int,
    has_topk_n: bool,
    is_rl_or_eval: bool,
    max_seqlen_q: int,
) -> None:
    device = get_device()
    # qlen: seqlen_q analog, bounded by max_seqlen_q (each req has 1..max_seqlen_q tokens)
    qlen = torch.randint(1, max_seqlen_q + 1, (req_cnt,), device=get_device())
    total_q = torch.sum(qlen).item()
    ftp = torch.cat([torch.tensor([0], device=get_device()), torch.cumsum(qlen, 0)[:-1]]).int()

    is_verify = torch.arange(0, max_seqlen_q, device=device).unsqueeze(1) < qlen.unsqueeze(0)

    accepted_tokens = torch.randint(0, 129280, (max_seqlen_q, req_cnt), dtype=torch.int32, device=device)
    token_logprobs_cur = torch.randn(max_seqlen_q, req_cnt, dtype=torch.float32, device=device)

    topk_n = 10 if has_topk_n else 0

    sum_smaller_probs_cur = (
        torch.randn(max_seqlen_q, req_cnt, dtype=torch.float32, device=device) if is_rl_or_eval else None
    )
    last_nextn_token_logprobs = (
        torch.randn(max_seqlen_q, req_cnt, dtype=torch.float32, device=device) if is_rl_or_eval else None
    )
    top_logprobs_cur = (
        torch.randn(max_seqlen_q, req_cnt, topk_n, dtype=torch.float32, device=device) if has_topk_n else None
    )
    top_token_ids_cur = (
        torch.randint(0, 129280, (max_seqlen_q, req_cnt, topk_n), dtype=torch.int32, device=device)
        if has_topk_n
        else None
    )

    # Run reference
    (
        verified_token_ids_ref,
        verified_token_logprobs_ref,
        verified_sum_smaller_probs_ref,
        accepted_token_last_logprobs_ref,
        verified_topk_logps_ref,
        verified_topk_tokens_ref,
    ) = scatter_verified_ref(
        is_verify,
        ftp,
        accepted_tokens,
        token_logprobs_cur,
        sum_smaller_probs_cur,
        last_nextn_token_logprobs,
        top_logprobs_cur,
        top_token_ids_cur,
        total_q=total_q,
    )

    # Run tilelang
    (
        verified_token_ids_tl,
        verified_token_logprobs_tl,
        verified_sum_smaller_probs_tl,
        accepted_token_last_logprobs_tl,
        verified_topk_logps_tl,
        verified_topk_tokens_tl,
    ) = scatter_verified(
        is_verify,
        ftp,
        accepted_tokens,
        token_logprobs_cur,
        sum_smaller_probs_cur,
        last_nextn_token_logprobs,
        top_logprobs_cur,
        top_token_ids_cur,
        total_q=total_q,
    )

    # Verify
    assert torch.equal(verified_token_ids_ref, verified_token_ids_tl), (
        f'token_ids mismatch\n{verified_token_ids_ref}\n{verified_token_ids_tl}\n{is_verify}\n{ftp}'
    )
    assert torch.equal(verified_token_logprobs_ref, verified_token_logprobs_tl), 'token_logprobs mismatch'
    if is_rl_or_eval:
        assert torch.equal(verified_sum_smaller_probs_ref, verified_sum_smaller_probs_tl), 'sum_smaller_probs mismatch'
        assert torch.equal(accepted_token_last_logprobs_ref, accepted_token_last_logprobs_tl), 'last_logprobs mismatch'
    if has_topk_n:
        assert torch.equal(verified_topk_logps_ref, verified_topk_logps_tl), 'topk_logps mismatch'
        assert torch.equal(verified_topk_tokens_ref, verified_topk_tokens_tl), 'topk_tokens mismatch'


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'req_cnt,max_seqlen_q,topk_n,is_rl_or_eval',
    [
        (64, 2, 0, False),
        (64, 2, 0, True),
        (256, 4, 0, True),
        (256, 4, 10, False),
        (256, 4, 10, True),
        (512, 4, 10, True),
    ],
)
def test_scatter_verified_benchmark(
    req_cnt: int,
    max_seqlen_q: int,
    topk_n: int,
    is_rl_or_eval: bool,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    device = get_device()

    qlen = torch.randint(1, max_seqlen_q + 1, (req_cnt,), device=get_device())
    total_q = torch.sum(qlen).item()
    ftp = torch.cat([torch.tensor([0], device=get_device()), torch.cumsum(qlen, 0)[:-1]]).int()

    is_verify = (torch.arange(1, max_seqlen_q + 1, device=device).unsqueeze(1) < qlen.unsqueeze(0)) & (
        (ftp.unsqueeze(0) + torch.arange(max_seqlen_q, device=device).unsqueeze(1)) < total_q
    )

    accepted_tokens = torch.randint(0, 129280, (max_seqlen_q, req_cnt), dtype=torch.int32, device=device)
    token_logprobs_cur = torch.randn(max_seqlen_q, req_cnt, dtype=torch.float32, device=device)
    sum_smaller_probs_cur = (
        torch.randn(max_seqlen_q, req_cnt, dtype=torch.float32, device=device) if is_rl_or_eval else None
    )
    last_nextn_token_logprobs = (
        torch.randn(max_seqlen_q, req_cnt, dtype=torch.float32, device=device) if is_rl_or_eval else None
    )

    top_logprobs_cur = (
        torch.randn(max_seqlen_q, req_cnt, topk_n, dtype=torch.float32, device=device) if topk_n > 0 else None
    )
    top_token_ids_cur = (
        torch.randint(0, 129280, (max_seqlen_q, req_cnt, topk_n), dtype=torch.int32, device=device)
        if topk_n > 0
        else None
    )

    cur_step_mtp_value = max_seqlen_q - 1

    t_us = benchmark_timer(
        lambda: scatter_verified(
            is_verify[:cur_step_mtp_value],
            ftp,
            accepted_tokens[:cur_step_mtp_value],
            token_logprobs_cur[:cur_step_mtp_value],
            sum_smaller_probs_cur[:cur_step_mtp_value] if sum_smaller_probs_cur is not None else None,
            last_nextn_token_logprobs[:cur_step_mtp_value] if last_nextn_token_logprobs is not None else None,
            top_logprobs_cur[:cur_step_mtp_value] if top_logprobs_cur is not None else None,
            top_token_ids_cur[:cur_step_mtp_value] if top_token_ids_cur is not None else None,
            total_q=total_q,
        )
    )

    num_steps, num_requests = cur_step_mtp_value, req_cnt
    total_bytes = (
        num_steps * num_requests * (1 + 8 + 4) + num_requests * 8
    )  # is_verify + accepted_tokens + token_logprobs_cur + ftp
    total_bytes += total_q * (8 + 4)  # verified_token_ids + verified_token_logprobs
    if is_rl_or_eval:
        total_bytes += num_steps * num_requests * (4 + 4)  # sum_smaller_probs_cur + last_nextn_token_logprobs
        total_bytes += total_q * (4 + 4)  # verified outputs
    if topk_n > 0:
        total_bytes += num_steps * num_requests * topk_n * (4 + 8) + total_q * topk_n * (4 + 8)

    benchmark_record(
        kernel='sampler',
        operation='scatter_verified',
        params={
            'is_rl_or_eval': int(is_rl_or_eval),
            'max_seqlen_q': max_seqlen_q,
            'req_cnt': req_cnt,
            'topk_n': topk_n,
        },
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1e9 / (t_us / 1e6)},
    )


def sampled_tokens_update_ref(
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
    """PyTorch reference implementation (in-place). Mirrors the original per-step SampledInfo.update loop."""
    max_seqlen_q = is_final.shape[0]
    for step_i in range(max_seqlen_q):
        mask = is_final[step_i]
        token_ids_out.copy_(torch.where(mask, sample_results_cur[step_i].to(token_ids_out.dtype), token_ids_out))
        token_logprobs_out.copy_(torch.where(mask, token_logprobs_cur[step_i], token_logprobs_out))
        if sum_smaller_probs_cur is not None and sum_smaller_probs_out is not None:
            sum_smaller_probs_out.copy_(torch.where(mask, sum_smaller_probs_cur[step_i], sum_smaller_probs_out))
        if top_logprobs_cur is not None and top_logps_out is not None:
            top_logps_out.copy_(torch.where(mask.unsqueeze(-1), top_logprobs_cur[step_i], top_logps_out))
        if top_token_ids_cur is not None and top_tokens_out is not None:
            top_tokens_out.copy_(
                torch.where(mask.unsqueeze(-1), top_token_ids_cur[step_i].to(top_tokens_out.dtype), top_tokens_out)
            )


@pytest.mark.parametrize('req_cnt,max_seqlen_q', [(1, 2), (256, 4)])
@pytest.mark.parametrize('topk_n', [0, 10])
@pytest.mark.parametrize('has_sum_smaller', [True, False])
def test_sampled_tokens_update(
    req_cnt: int,
    max_seqlen_q: int,
    topk_n: int,
    has_sum_smaller: bool,
) -> None:
    device = get_device()

    is_final = torch.zeros(max_seqlen_q, req_cnt, dtype=torch.bool, device=device)
    final_steps = torch.randint(0, max_seqlen_q, (req_cnt,), device=device)
    is_final[final_steps, torch.arange(req_cnt, device=device)] = True

    sample_results_cur = torch.randint(0, 10000, (max_seqlen_q, req_cnt), dtype=torch.int32, device=device)
    token_logprobs_cur = torch.randn(max_seqlen_q, req_cnt, dtype=torch.float32, device=device)

    sum_smaller_probs_cur = (
        torch.randn(max_seqlen_q, req_cnt, dtype=torch.float32, device=device) if has_sum_smaller else None
    )
    top_logprobs_cur = (
        torch.randn(max_seqlen_q, req_cnt, topk_n, dtype=torch.float32, device=device) if topk_n > 0 else None
    )
    top_token_ids_cur = (
        torch.randint(0, 10000, (max_seqlen_q, req_cnt, topk_n), dtype=torch.int32, device=device)
        if topk_n > 0
        else None
    )

    token_ids_ref = torch.full((req_cnt,), -1, dtype=torch.int32, device=device)
    token_logprobs_ref = torch.zeros(req_cnt, dtype=torch.float32, device=device)
    sum_smaller_probs_ref = torch.zeros(req_cnt, dtype=torch.float32, device=device) if has_sum_smaller else None
    top_logps_ref = torch.zeros(req_cnt, topk_n, dtype=torch.float32, device=device) if topk_n > 0 else None
    top_tokens_ref = torch.zeros(req_cnt, topk_n, dtype=torch.int32, device=device) if topk_n > 0 else None

    token_ids_tl = token_ids_ref.clone()
    token_logprobs_tl = token_logprobs_ref.clone()
    sum_smaller_probs_tl = sum_smaller_probs_ref.clone() if has_sum_smaller else None
    top_logps_tl = top_logps_ref.clone() if topk_n > 0 else None
    top_tokens_tl = top_tokens_ref.clone() if topk_n > 0 else None

    sampled_tokens_update_ref(
        is_final,
        sample_results_cur,
        token_logprobs_cur,
        sum_smaller_probs_cur,
        top_logprobs_cur,
        top_token_ids_cur,
        token_ids_ref,
        token_logprobs_ref,
        sum_smaller_probs_ref,
        top_logps_ref,
        top_tokens_ref,
    )

    sampled_tokens_update(
        is_final,
        sample_results_cur,
        token_logprobs_cur,
        sum_smaller_probs_cur,
        top_logprobs_cur,
        top_token_ids_cur,
        token_ids_tl,
        token_logprobs_tl,
        sum_smaller_probs_tl,
        top_logps_tl,
        top_tokens_tl,
    )

    assert torch.equal(token_ids_ref, token_ids_tl), f'token_ids mismatch\n{token_ids_ref}\n{token_ids_tl}\n{is_final}'
    assert torch.equal(token_logprobs_ref, token_logprobs_tl), 'token_logprobs mismatch'
    if has_sum_smaller:
        assert torch.equal(sum_smaller_probs_ref, sum_smaller_probs_tl), 'sum_smaller_probs mismatch'
    if topk_n > 0:
        assert torch.equal(top_logps_ref, top_logps_tl), 'topk_logps mismatch'
        assert torch.equal(top_tokens_ref, top_tokens_tl), 'topk_tokens mismatch'


def compute_confidence_error_ref(
    last_nextn_token_logprobs: torch.Tensor,
    nextn_token_logprobs: torch.Tensor,
    confidence: torch.Tensor,
    verify_masks: torch.Tensor,
) -> torch.Tensor:
    device = last_nextn_token_logprobs.device
    accept_logprob = torch.where(
        last_nextn_token_logprobs > 0,
        torch.zeros((), device=device, dtype=torch.float32),
        torch.clamp(nextn_token_logprobs - last_nextn_token_logprobs, max=0),
    )
    accept_prob = torch.exp(accept_logprob)
    error = torch.where(verify_masks, torch.abs(accept_prob - confidence), 0.0)
    return error.sum()


@pytest.mark.parametrize('num_steps,num_requests', [(1, 1), (7, 7), (3, 256), (5, 1024)])
def test_compute_confidence_error(num_steps: int, num_requests: int) -> None:
    device = get_device()
    last_logprobs = torch.randn(num_steps, num_requests, dtype=torch.float32, device=device)
    next_logprobs = torch.randn(num_steps, num_requests, dtype=torch.float32, device=device)
    confidence = torch.sigmoid(torch.randn(num_requests, num_steps, dtype=torch.float32, device=device)).t()
    verify_masks = torch.rand(num_steps, num_requests, device=device) > 0.5

    out = compute_confidence_error(last_logprobs, next_logprobs, confidence, verify_masks)
    ref = compute_confidence_error_ref(last_logprobs, next_logprobs, confidence, verify_masks)
    assert torch.allclose(out, ref)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'max_seqlen_q,req_cnt,force_topk',
    [
        (1, 1024, 2048),
        (3, 233, 1024),
        (3, 512, 1024),
        (5, 512, 1024),
    ],
)
def test_find_value_by_index_benchmark(
    max_seqlen_q: int,
    req_cnt: int,
    force_topk: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    n = max_seqlen_q * req_cnt
    steps = torch.randint(1, 10000, (n, force_topk), device=get_device(), dtype=torch.int32)
    index_flat = torch.cumsum(steps, dim=1, dtype=torch.int32)
    for i in range(n):
        perm = torch.randperm(force_topk, device=get_device())
        index_flat[i] = index_flat[i][perm]
    value_flat = torch.randn((n, force_topk), dtype=torch.float32, device=get_device())

    indices_idx = torch.randint(0, force_topk, (n,), device=get_device())
    q_index_flat = index_flat[torch.arange(n), indices_idx].unsqueeze(-1)

    index = index_flat.unflatten(0, (max_seqlen_q, req_cnt))
    value = value_flat.unflatten(0, (max_seqlen_q, req_cnt))
    q_index = q_index_flat.unflatten(0, (max_seqlen_q, req_cnt))

    t_us = benchmark_timer(lambda: find_value_by_index(index, value, q_index))
    benchmark_record(
        kernel='sampler',
        operation='find_value_by_index',
        params={'force_topk': force_topk, 'max_seqlen_q': max_seqlen_q, 'req_cnt': req_cnt},
        time_us=t_us,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'max_seqlen_q,req_cnt,force_topk,query_dim',
    [
        (1, 256, 1024, 1024),
        (1, 512, 1024, 1024),
        (3, 233, 1024, 1024),
        (3, 256, 1024, 234),
        (5, 512, 1024, 1024),
        (1, 512, 4096, 4096),
    ],
)
def test_batch_find_value_by_index_benchmark(
    max_seqlen_q: int,
    req_cnt: int,
    force_topk: int,
    query_dim: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    vocab_size = 129280
    n = max_seqlen_q * req_cnt
    index_flat = torch.empty((n, force_topk), dtype=torch.int32, device=get_device())
    for i in range(n):
        index_flat[i] = torch.randperm(vocab_size, device=get_device())[:force_topk]
    value_flat = torch.randn((n, force_topk), dtype=torch.float32, device=get_device())

    indices_idx = torch.randint(0, force_topk, (n, query_dim), device=get_device())
    q_index_flat = index_flat[torch.arange(n, device=get_device()).reshape(-1, 1).expand(-1, query_dim), indices_idx]

    index = index_flat.unflatten(0, (max_seqlen_q, req_cnt))
    value = value_flat.unflatten(0, (max_seqlen_q, req_cnt))
    q_index = q_index_flat.unflatten(0, (max_seqlen_q, req_cnt))

    t_us = benchmark_timer(lambda: batch_find_value_by_index(index, value, q_index, vocab_size))
    benchmark_record(
        kernel='sampler',
        operation='batch_find_value_by_index',
        params={'force_topk': force_topk, 'max_seqlen_q': max_seqlen_q, 'query_dim': query_dim, 'req_cnt': req_cnt},
        time_us=t_us,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'm,n,has_so_valid,has_golden_output,partial_golden',
    [
        (1, 1024, True, True, False),
        (3, 1024, True, True, True),
        (5, 1024, True, False, False),
        (5, 4096, True, False, False),
    ],
)
def test_get_accept_result_benchmark(
    m: int,
    n: int,
    has_so_valid: bool,
    has_golden_output: bool,
    partial_golden: bool,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    max_seqlen_q = m + 1
    so_valid = torch.randint(0, 2, (m, n), dtype=torch.bool, device=get_device()) if has_so_valid else None
    accept_result_verify = torch.rand(m, n, device=get_device()) < 0.8
    if has_golden_output:
        golden_steps = m // 2 + 1 if partial_golden else m
        golden_output = torch.randint(0, 129280, (n, golden_steps), dtype=torch.int32, device=get_device()).t()
        gold_accepted = torch.randint(0, 2, (n, golden_steps), dtype=torch.bool, device=get_device()).t()
    else:
        golden_steps = 0
        golden_output = None
        gold_accepted = None
    last_next_n_token_ids = torch.randint(0, 129280, (n, m), dtype=torch.int32, device=get_device()).t()
    is_verify = torch.rand(max_seqlen_q, n, device=get_device()) < 0.6
    sample_reject_cur = torch.randint(0, 129280, (m, n), dtype=torch.int32, device=get_device())
    fresh_sample_final = torch.randint(0, 129280, (max_seqlen_q, n), dtype=torch.int32, device=get_device())

    t_us = benchmark_timer(
        lambda: get_accept_result(
            accept_result_verify,
            so_valid,
            golden_output,
            last_next_n_token_ids,
            gold_accepted,
            is_verify,
            sample_reject_cur,
            fresh_sample_final,
        )
    )
    total_bytes = m * n * 1 + (m * n if has_so_valid else 0) + (golden_steps * n * (8 + 1) if has_golden_output else 0)
    benchmark_record(
        kernel='sampler',
        operation='get_accept_result',
        params={'golden': int(has_golden_output), 'm': m, 'n': n, 'so_valid': int(has_so_valid)},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1e9 / (t_us / 1e6)},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'req_cnt,max_seqlen_q,topk_n,has_sum_smaller',
    [
        (1, 2, 0, False),
        (32, 3, 10, True),
        (64, 4, 10, True),
        (256, 4, 10, True),
        (512, 2, 0, False),
        (512, 4, 10, True),
    ],
)
def test_sampled_tokens_update_benchmark(
    req_cnt: int,
    max_seqlen_q: int,
    topk_n: int,
    has_sum_smaller: bool,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    device = get_device()

    is_final = torch.zeros(max_seqlen_q, req_cnt, dtype=torch.bool, device=device)
    final_steps = torch.randint(0, max_seqlen_q, (req_cnt,), device=device)
    is_final[final_steps, torch.arange(req_cnt, device=device)] = True

    sample_results_cur = torch.randint(0, 10000, (max_seqlen_q, req_cnt), dtype=torch.int32, device=device)
    token_logprobs_cur = torch.randn(max_seqlen_q, req_cnt, dtype=torch.float32, device=device)
    sum_smaller_probs_cur = (
        torch.randn(max_seqlen_q, req_cnt, dtype=torch.float32, device=device) if has_sum_smaller else None
    )
    top_logprobs_cur = (
        torch.randn(max_seqlen_q, req_cnt, topk_n, dtype=torch.float32, device=device) if topk_n > 0 else None
    )
    top_token_ids_cur = (
        torch.randint(0, 10000, (max_seqlen_q, req_cnt, topk_n), dtype=torch.int32, device=device)
        if topk_n > 0
        else None
    )

    token_ids_out = torch.full((req_cnt,), -1, dtype=torch.int32, device=device)
    token_logprobs_out = torch.zeros(req_cnt, dtype=torch.float32, device=device)
    sum_smaller_probs_out = torch.zeros(req_cnt, dtype=torch.float32, device=device) if has_sum_smaller else None
    top_logps_out = torch.zeros(req_cnt, topk_n, dtype=torch.float32, device=device) if topk_n > 0 else None
    top_tokens_out = torch.zeros(req_cnt, topk_n, dtype=torch.int32, device=device) if topk_n > 0 else None

    t_us = benchmark_timer(
        lambda: sampled_tokens_update(
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
    )
    total_bytes = (
        req_cnt * max_seqlen_q * (1 + 8 + 4)  # is_final + sample_results + logprobs
        + (req_cnt * max_seqlen_q * 4 if has_sum_smaller else 0)  # sum_smaller_probs
        + (req_cnt * max_seqlen_q * topk_n * (4 + 8) if topk_n > 0 else 0)  # topk
    )
    benchmark_record(
        kernel='sampler',
        operation='sampled_tokens_update',
        params={
            'req_cnt': req_cnt,
            'max_seqlen_q': max_seqlen_q,
            'topk_n': topk_n,
            'has_sum_smaller': has_sum_smaller,
        },
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1e9 / (t_us / 1e6)},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('num_steps,num_requests', [(1, 256), (1, 1024), (3, 512), (5, 512)])
def test_compute_confidence_error_benchmark(
    num_steps: int,
    num_requests: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    device = get_device()
    last_logprobs = torch.randn(num_steps, num_requests, dtype=torch.float32, device=device)
    next_logprobs = torch.randn(num_steps, num_requests, dtype=torch.float32, device=device)
    confidence = torch.sigmoid(torch.randn(num_requests, num_steps, dtype=torch.float32, device=device)).t()
    verify_masks = torch.rand(num_steps, num_requests, device=device) > 0.5

    def fn() -> torch.Tensor:
        return compute_confidence_error(last_logprobs, next_logprobs, confidence, verify_masks)

    t_us = benchmark_timer(fn)
    total_bytes = num_steps * num_requests * 13  # 4 × float32 + 1 × bool per element
    benchmark_record(
        kernel='sampler',
        operation='compute_confidence_error',
        params={'M': num_steps, 'N': num_requests},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1e9 / (t_us / 1e6)},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'n,vocab_size,maxlen,step',
    [
        (256, 129280, 4, 1),
        (1024, 129280, 4, 1),
        (256, 129280, 8, 3),
    ],
)
def test_get_logits_cur_by_step_benchmark(
    n: int,
    vocab_size: int,
    maxlen: int,
    step: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    qlen = torch.randint(step + 1, maxlen, (n,), device=get_device())
    has_token = step < qlen
    ftp = torch.cat([torch.tensor([0], device=get_device()), torch.cumsum(qlen, 0)[:-1]]).int()
    logits = torch.randn((torch.sum(qlen).item(), vocab_size), dtype=torch.float, device=get_device())

    t_us = benchmark_timer(lambda: get_logits_cur_by_step(logits, ftp, has_token, step))
    total_bytes = n * vocab_size * 8  # read + write × float32
    benchmark_record(
        kernel='sampler',
        operation='get_logits_cur_by_step',
        params={'n': n, 'step': step, 'vocab_size': vocab_size},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1e9 / (t_us / 1e6)},
    )
