from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.sampling.fused_sample import fused_sample
from cann_ops_tilelang.sampling.fused_sample_kernel import DRAFT_POS_HASH_MUL, DRAFT_PROB_PENALTY
from cann_ops_tilelang.sampling.random_sampling import random_sampling
from cann_ops_tilelang.sampling.softmax import log_softmax

pytestmark = pytest.mark.npu


def generate_logits(
    num_tokens: int,
    vocab_size: int,
    *,
    dist: str = 'randn',
    device: str = 'npu',
) -> torch.Tensor:
    if dist == 'ternary':
        return (torch.randint(0, 3, (num_tokens, vocab_size), device=device) - 1).float()
    elif dist == 'randn':
        return torch.randn((num_tokens, vocab_size), dtype=torch.float32, device=device)
    else:
        logits = torch.full((num_tokens, vocab_size), -float('inf'), dtype=torch.float32, device=device)
        num_valid = 16
        idx = torch.argsort(torch.rand((num_tokens, vocab_size), device=device), dim=1)[:, :num_valid]
        rows = torch.arange(num_tokens, device=device).unsqueeze(1)
        logits[rows, idx] = torch.randn((num_tokens, num_valid), dtype=torch.float32, device=device)
        return logits


def _check_top_p(
    prob: torch.Tensor,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_p: torch.Tensor,
    do_top_p: bool,
    *,
    label: str = '',
    pruning_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Comprehensive top-p validation for a single prob output.

    Returns (sorted_desc, sorted_indices, n_pos) for cross-validation.
    """
    eps = 1e-4

    mask = prob > 0
    n_survive = mask.sum(dim=-1)

    if not do_top_p:
        surv_mass = (prob * mask).sum(dim=-1)
        assert (surv_mass >= 1 - eps).all(), (
            f'[{label}] no top-p but survive mass < 1-eps, min={surv_mass.min().item()}'
        )
    assert (n_survive > 0).all(), f'[{label}] row has no surviving tokens'
    assert (prob.sum(dim=-1) - 1).abs().max() < 1e-5, (
        f'[{label}] prob sum != 1, max_dev={((prob.sum(dim=-1) - 1).abs().max().item())}'
    )

    logits_scaled = logits / temperatures.unsqueeze(1)
    full_softmax = logits_scaled.double().softmax(dim=-1)

    # 1) max(logits[pruned]) <= min(logits[surviving]) per row.
    pruned_max = logits_scaled.masked_fill(mask, float('-inf')).max(dim=-1).values
    survive_min = logits_scaled.masked_fill(~mask, float('inf')).min(dim=-1).values
    chk = (~mask).any(dim=-1)
    assert (pruned_max[chk] <= survive_min[chk] + pruning_eps).all(), f'[{label}] pruned_max > survive_min+eps'

    # 2) surviving probs equal renormalized full softmax over same set.
    surv = full_softmax * mask
    surv_norm = surv / surv.sum(dim=-1, keepdim=True)
    diff = (prob - surv_norm).abs() * mask
    diff_sum = diff.sum(dim=-1)
    threshold = 1e-4 + logits_scaled.max(dim=-1).values * 1e-7
    assert (diff_sum <= threshold).all(), f'[{label}] renormalized softmax mismatch, max_sum={diff_sum.max().item()}'

    # 3) top-p semantic.
    if do_top_p:
        retained_mass = surv.sum(dim=-1)
        assert (retained_mass >= top_p - eps).all(), (
            f'[{label}] retained_mass < p-eps, max gap: {(top_p - retained_mass).max().item()}'
        )
        min_surv_val = full_softmax.masked_fill(~mask, float('inf')).min(dim=-1).values
        assert (retained_mass - min_surv_val <= top_p + eps).all(), f'[{label}] retained_mass-min > p+eps'

    sorted_desc, sorted_indices = prob.sort(dim=-1, descending=True)
    return sorted_desc, sorted_indices, n_survive


def generate_test_params(is_benchmark: bool = False) -> list[dict]:
    params = [
        {
            'b': b,
            'h': h,
            'dist': dist,
            'do_top_p': do_top_p,
            'return_batch_format': return_batch_format,
            'do_sample': ds,
            'do_reject_sampling': dr,
        }
        for b in ([256, 1536] if is_benchmark else [4096])
        for h in [1024, 129280]
        for dist in ['randn', 'ternary', '-inf']
        for do_top_p in [True, False]
        for return_batch_format in [True, False]
        for ds in ([True] if is_benchmark else [True, False])
        for dr in [True, False]
    ]

    def include_test_params(d: dict) -> bool:
        if is_benchmark and d['dist'] == '-inf':
            return False
        if d['dist'] == '-inf' and (not d['do_top_p'] or not d['do_reject_sampling']):
            return False
        if d['do_reject_sampling'] and (not d['return_batch_format'] or not d['do_sample']):
            return False
        if is_benchmark and d['return_batch_format'] and not d['do_reject_sampling']:
            return False
        return not (is_benchmark and d['return_batch_format'] and d['b'] != 256)

    return [x for x in params if include_test_params(x)]


def generate_ascend_test_params() -> list[dict]:
    """Cover each correctness mode at practical Ascend batch sizes.

    Keep one copy of each mode, then add focused shape cases for
    multi-wave execution, the 4096-token chunk boundary, and the production
    vocabulary size.
    """
    mode_keys = ('dist', 'do_top_p', 'return_batch_format', 'do_sample', 'do_reject_sampling')
    params = []
    seen = set()
    for source_params in generate_test_params(is_benchmark=False):
        mode = tuple(source_params[key] for key in mode_keys)
        if mode in seen:
            continue
        seen.add(mode)
        params.append({'b': 4, 'h': 128, **dict(zip(mode_keys, mode, strict=True))})

    params.extend(
        [
            {
                'b': 65,
                'h': 1024,
                'dist': 'randn',
                'do_top_p': False,
                'return_batch_format': False,
                'do_sample': False,
                'do_reject_sampling': False,
            },
            {
                'b': 3,
                'h': 4096,
                'dist': 'ternary',
                'do_top_p': True,
                'return_batch_format': True,
                'do_sample': True,
                'do_reject_sampling': True,
            },
            {
                'b': 3,
                'h': 4097,
                'dist': 'ternary',
                'do_top_p': True,
                'return_batch_format': False,
                'do_sample': True,
                'do_reject_sampling': False,
            },
            {
                'b': 2,
                'h': 129280,
                'dist': 'randn',
                'do_top_p': True,
                'return_batch_format': True,
                'do_sample': True,
                'do_reject_sampling': False,
            },
        ]
    )
    params.append(
        {
            'b': 2,
            'h': 129280,
            'dist': 'randn',
            'do_top_p': True,
            'return_batch_format': True,
            'do_sample': True,
            'do_reject_sampling': True,
        }
    )
    return params


def make_param_id(params: dict) -> str:
    param_str = ','.join(f'{k}={v}' for k, v in params.items() if v is not None)
    return f'{param_str}'


def test_fused_sample_npu() -> None:
    b, vocab_size, max_seqlen_q = 2, 128, 3
    torch.manual_seed(42)
    logits_cpu = torch.randn((4, vocab_size), dtype=torch.float32)
    temperatures_cpu = torch.tensor([0.8, 1.2], dtype=torch.float32)
    top_p_cpu = torch.tensor([torch.finfo(torch.float64).eps, 0.75], dtype=torch.float32)
    ftp_cpu = torch.tensor([0, 3], dtype=torch.int32)
    positions_cpu = torch.tensor([5, 9], dtype=torch.int32)
    seeds_cpu = torch.tensor([123456789, 987654321], dtype=torch.int64).view(torch.uint64)
    draft_cpu = torch.log_softmax(torch.randn((max_seqlen_q - 1, b, vocab_size), dtype=torch.float32), dim=-1)
    draft_ids_cpu = torch.tensor([[3, -1], [5, 7]], dtype=torch.int32)

    logits = logits_cpu.npu()
    temperatures = temperatures_cpu.npu()
    kwargs = dict(
        top_p=top_p_cpu.npu(),
        ftp=ftp_cpu.npu(),
        max_seqlen_q=max_seqlen_q,
        sample_positions=positions_cpu.npu(),
        seq_seeds=seeds_cpu.npu(),
        last_nextn_token_logprobs=draft_cpu.npu(),
        last_next_n_token_ids=draft_ids_cpu.npu(),
    )

    outputs = fused_sample(logits, temperatures, **kwargs)
    outputs_again = fused_sample(logits, temperatures, **kwargs)
    torch.npu.synchronize()
    logprobs, samples, accepts, rejects = (output.cpu() for output in outputs)
    logprobs_again, samples_again, accepts_again, rejects_again = (output.cpu() for output in outputs_again)

    assert torch.equal(logits.cpu(), logits_cpu), 'logits must remain read-only'
    assert torch.equal(logprobs, logprobs_again)
    assert torch.equal(samples, samples_again)
    assert torch.equal(accepts, accepts_again)
    assert torch.equal(rejects, rejects_again)

    valid = torch.tensor([[True, True], [True, False], [True, False]])
    valid_logprobs = logprobs[valid]
    packed_logits = logits_cpu[[0, 3, 1, 2]]
    row_temperatures = temperatures_cpu[[0, 1, 0, 0]]
    row_top_p = top_p_cpu[[0, 1, 0, 0]]
    full_probs = torch.softmax(packed_logits / row_temperatures.unsqueeze(1), dim=-1)
    keep = torch.isfinite(valid_logprobs)
    kept_mass = (full_probs * keep).sum(dim=-1)
    min_kept = full_probs.masked_fill(~keep, float('inf')).min(dim=-1).values
    max_pruned = full_probs.masked_fill(keep, float('-inf')).max(dim=-1).values
    renormalized = full_probs * keep
    renormalized /= renormalized.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(valid_logprobs.exp(), renormalized, rtol=1e-5, atol=1e-6)
    assert (max_pruned <= min_kept).all()
    assert (kept_mass >= row_top_p - 1e-6).all()
    assert ((kept_mass - min_kept <= row_top_p + 1e-6) | (row_top_p == 1)).all()
    assert (keep[[0, 2, 3]].sum(dim=-1) == 1).all(), 'a tiny positive top_p must retain one token'

    invalid_logprobs = logprobs[~valid]
    assert (invalid_logprobs[:, 0] == 0).all()
    assert torch.isneginf(invalid_logprobs[:, 1:]).all()
    assert (samples[~valid] == 0).all()
    assert accepts[1, 1]
    assert rejects[1, 1] == 0
    assert not accepts[0, 1], 'draft token -1 must force rejection'
    assert ((rejects >= 0) & (rejects < vocab_size)).all()
    for sx, bx in valid.nonzero().tolist():
        assert torch.isfinite(logprobs[sx, bx, samples[sx, bx]])


@pytest.mark.parametrize('params', generate_ascend_test_params(), ids=make_param_id)
def test_fused_sample_ascend_cases(params: dict) -> None:
    """Exercise the source correctness modes through the Ascend implementation."""
    b = params['b']
    h = params['h']
    dist = params['dist']
    do_top_p = params['do_top_p']
    return_batch_format = params['return_batch_format']
    do_sample = params['do_sample']
    do_reject_sampling = params['do_reject_sampling']
    max_seqlen_q = 3

    logits_cpu = generate_logits(b, h, dist=dist, device='cpu')
    temperatures_cpu = torch.linspace(0.25, 1.5, b, dtype=torch.float32)
    temperatures_cpu[0] = 1e-5  # SAMPLING_EPS
    top_p_values = torch.tensor([torch.finfo(torch.float64).eps, 1.0, 0.5, 0.9], dtype=torch.float32)
    top_p_cpu = top_p_values.repeat((b + top_p_values.numel() - 1) // top_p_values.numel())[:b].clone()
    positions_cpu = torch.arange(b, dtype=torch.int32) * 17 + 5
    seeds_cpu = (torch.arange(b, dtype=torch.int64) + 123456789).view(torch.uint64)

    seqlens_cpu = None
    ftp_cpu = None
    if return_batch_format:
        seqlens_cpu = torch.arange(b, dtype=torch.int32) % max_seqlen_q + 1
        ftp_cpu = torch.zeros(b, dtype=torch.int32)
        ftp_cpu[1:] = torch.cumsum(seqlens_cpu, dim=0)[:-1]
        logits_cpu = logits_cpu.repeat_interleave(seqlens_cpu, dim=0)

    draft_logprobs_cpu = None
    draft_ids_cpu = None
    if do_reject_sampling:
        reject_steps = max_seqlen_q - 1
        draft_logprobs_cpu = torch.log_softmax(torch.randn((reject_steps, b, h), dtype=torch.float32), dim=-1)
        draft_ids_cpu = draft_logprobs_cpu.argmax(dim=-1).to(torch.int32)
        draft_ids_cpu[0, 0] = -1

    logits = logits_cpu.npu()
    temperatures = temperatures_cpu.npu()
    kwargs = {}
    if return_batch_format:
        kwargs.update(ftp=ftp_cpu.npu(), max_seqlen_q=max_seqlen_q)
    if do_sample:
        kwargs.update(sample_positions=positions_cpu.npu(), seq_seeds=seeds_cpu.npu())
    if do_reject_sampling:
        kwargs.update(
            last_nextn_token_logprobs=draft_logprobs_cpu.npu(),
            last_next_n_token_ids=draft_ids_cpu.npu(),
        )

    outputs = fused_sample(logits, temperatures, top_p=top_p_cpu.npu() if do_top_p else None, **kwargs)
    outputs_again = fused_sample(logits, temperatures, top_p=top_p_cpu.npu() if do_top_p else None, **kwargs)
    torch.npu.synchronize()
    outputs_cpu = tuple(output.cpu() if output is not None else None for output in outputs)
    outputs_again_cpu = tuple(output.cpu() if output is not None else None for output in outputs_again)
    logprobs, samples, accepts, rejects = outputs_cpu

    assert torch.equal(logits.cpu(), logits_cpu), 'logits must remain read-only'
    for output, output_again in zip(outputs_cpu, outputs_again_cpu, strict=True):
        assert (output is None) == (output_again is None)
        if output is not None:
            assert torch.equal(output, output_again)

    if return_batch_format:
        assert logprobs.shape == (max_seqlen_q, b, h)
        steps = torch.arange(max_seqlen_q, dtype=torch.int32).unsqueeze(1)
        valid = steps < seqlens_cpu.unsqueeze(0)
        packed_rows = (ftp_cpu.unsqueeze(0) + steps)[valid].long()
        row_to_req = torch.repeat_interleave(torch.arange(b), seqlens_cpu)
        probs_to_check = logprobs[valid].exp()
        logits_to_check = logits_cpu[packed_rows]
        temperatures_to_check = temperatures_cpu[row_to_req[packed_rows]]
        top_p_to_check = top_p_cpu[row_to_req[packed_rows]] if do_top_p else None

        invalid_logprobs = logprobs[~valid]
        assert (invalid_logprobs[:, 0] == 0).all()
        assert torch.isneginf(invalid_logprobs[:, 1:]).all()
    else:
        assert logprobs.shape == (b, h)
        valid = torch.ones((b,), dtype=torch.bool)
        probs_to_check = logprobs.exp()
        logits_to_check = logits_cpu
        temperatures_to_check = temperatures_cpu
        top_p_to_check = top_p_cpu if do_top_p else None

    _check_top_p(
        probs_to_check,
        logits_to_check,
        temperatures_to_check,
        top_p_to_check,
        do_top_p,
        label='npu',
        pruning_eps=1e-4,
    )

    if do_sample:
        assert samples is not None and samples.dtype == torch.int32
        assert ((samples >= 0) & (samples < h)).all()
        if return_batch_format:
            assert samples.shape == (max_seqlen_q, b)
            assert (samples[~valid] == 0).all()
            sampled_logprobs = logprobs.gather(-1, samples.long().unsqueeze(-1)).squeeze(-1)[valid]
        else:
            assert samples.shape == (b,)
            sampled_logprobs = logprobs.gather(-1, samples.long().unsqueeze(-1)).squeeze(-1)
        assert torch.isfinite(sampled_logprobs).all()
    else:
        assert samples is None

    if do_reject_sampling:
        reject_steps = max_seqlen_q - 1
        assert accepts is not None and rejects is not None
        assert accepts.shape == (reject_steps, b)
        assert rejects.shape == (reject_steps, b)
        assert rejects.dtype == torch.int32
        assert ((rejects >= 0) & (rejects < h)).all()
        assert not accepts[draft_ids_cpu == -1].any(), 'sentinel -1 must force reject'

        valid_reject = valid[:-1]
        assert accepts[~valid_reject].all()
        assert (rejects[~valid_reject] == 0).all()
        target_selected = logprobs[:-1].gather(-1, rejects.long().unsqueeze(-1)).squeeze(-1)
        draft_selected = draft_logprobs_cpu.gather(-1, rejects.long().unsqueeze(-1)).squeeze(-1)
        selected_residual = target_selected.exp() - (draft_selected - DRAFT_PROB_PENALTY).exp()
        assert (selected_residual[valid_reject] > 0).all()
    else:
        assert accepts is None
        assert rejects is None


def generate_ascend_benchmark_params() -> list[dict]:
    """Return the 24 Ascend benchmark parameter combinations."""
    return generate_test_params(is_benchmark=True)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_ascend_benchmark_params(), ids=make_param_id)
def test_fused_sample_ascend_benchmark(
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
    params: dict,
) -> None:
    """Benchmark the fused sampling operator on Ascend."""
    b = params['b']
    h = params['h']
    do_top_p = params['do_top_p']
    return_batch_format = params['return_batch_format']
    do_sample = params['do_sample']
    do_reject_sampling = params['do_reject_sampling']
    max_seqlen_q = 6

    torch.manual_seed(42)
    logits = generate_logits(b, h, dist=params['dist'], device='npu')
    top_p = torch.full((b,), 0.9, dtype=torch.float32, device='npu') if do_top_p else None
    temperatures = torch.full((b,), 1.0, dtype=torch.float32, device='npu')
    sample_positions = torch.randint(0, 10**8, (b,), dtype=torch.int32, device='npu')
    seq_seeds = torch.randint(0, 10**18, (b,), dtype=torch.int64, device='npu').view(torch.uint64)

    kwargs = {}
    if return_batch_format:
        seqlens = torch.full((b,), max_seqlen_q, dtype=torch.int32, device='npu')
        ftp = torch.zeros(b, dtype=torch.int32, device='npu')
        ftp[1:] = torch.cumsum(seqlens, dim=0)[:-1]
        logits = logits.repeat_interleave(seqlens, dim=0)
        kwargs.update(ftp=ftp, max_seqlen_q=max_seqlen_q)
        num_blocks = max_seqlen_q * b
    else:
        seqlens = None
        num_blocks = b

    if do_sample:
        kwargs.update(sample_positions=sample_positions, seq_seeds=seq_seeds)
    if do_reject_sampling:
        reject_steps = max_seqlen_q - 1
        steps = torch.arange(max_seqlen_q, device='npu', dtype=torch.int32)
        pos_block = (sample_positions.unsqueeze(0) + DRAFT_POS_HASH_MUL * steps.unsqueeze(1)).reshape(-1)
        seed_block = seq_seeds.unsqueeze(0).expand(max_seqlen_q, b).reshape(-1)
        reject_positions = pos_block.reshape(max_seqlen_q, b)[:-1].reshape(-1)
        reject_seeds = seed_block.reshape(max_seqlen_q, b)[:-1].reshape(-1)
        draft_logits = torch.randn((reject_steps * b, h), dtype=torch.float32, device='npu')
        draft_logprobs = log_softmax(draft_logits.clone())
        draft_ids = random_sampling(
            draft_logprobs.exp(),
            reject_seeds,
            reject_positions,
        )
        kwargs.update(
            last_nextn_token_logprobs=draft_logprobs.reshape(reject_steps, b, h),
            last_next_n_token_ids=draft_ids.reshape(reject_steps, b),
        )

    def fn() -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        return fused_sample(logits, temperatures, top_p=top_p if do_top_p else None, **kwargs)

    fn()
    torch.npu.synchronize()
    time_us = benchmark_timer(fn)

    n_extra = 1 + (1 if do_sample else 0)
    b_extra = 1 if do_top_p else 0
    if return_batch_format:
        total_q = int(seqlens.sum().item())
        total_bytes = total_q * h * 4 + b * 4 * (1 + b_extra) + b * 8 + b * 8 + max_seqlen_q * b * h * 4 * n_extra
    else:
        total_bytes = b * h * 4 * (1 + n_extra) + b * 4 * (1 + b_extra) + b * 8
    if do_reject_sampling:
        total_bytes += num_blocks * h * 4
        total_bytes += num_blocks * 8
        total_bytes += num_blocks
        total_bytes += num_blocks * 8

    record_params = {
        'b': b,
        'h': h,
        'dist': params['dist'],
        'do_top_p': do_top_p,
        'return_batch_format': int(return_batch_format),
        'do_reject_sampling': int(do_reject_sampling),
    }
    benchmark_record(
        kernel='fused_sample',
        operation='fused_sample',
        params=record_params,
        time_us=time_us,
        extras={'bandwidth_gbs': total_bytes / 1e9 / (time_us / 1e6)},
    )
