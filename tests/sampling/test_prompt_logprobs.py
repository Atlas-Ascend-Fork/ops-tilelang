from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.sampling.prompt_logprobs import get_prompt_logprobs_tensor

pytestmark = pytest.mark.npu


def get_prompt_logprobs_tensor_ref(
    prompt_tokens: torch.Tensor,
    logprobs: torch.Tensor,
    token_index: torch.Tensor,
) -> torch.Tensor:
    mask = token_index == prompt_tokens.unsqueeze(-1)
    return torch.where(mask, logprobs, torch.full_like(logprobs, float('-inf'))).max(dim=-1).values


@pytest.mark.parametrize('num_tokens', [1, 257])
@pytest.mark.parametrize('topk_cnt', [1, 233, 1024, 4096])
def test_get_prompt_logprobs_tensor(num_tokens: int, topk_cnt: int) -> None:
    device = get_device()
    vocab_size = 129280
    token_index = torch.empty((num_tokens, topk_cnt), dtype=torch.int32, device=device)
    for i in range(num_tokens):
        token_index[i] = torch.randperm(vocab_size, device=device)[:topk_cnt]
    logprobs = torch.randn((num_tokens, topk_cnt), dtype=torch.float32, device=device)

    row_idx = torch.arange(num_tokens, device=device)
    selected_col = torch.randint(0, topk_cnt, (num_tokens,), device=device)
    in_topk = torch.rand(num_tokens, device=device) < 0.8
    prompt_tokens = torch.where(
        in_topk,
        token_index[row_idx, selected_col],
        torch.full((num_tokens,), vocab_size + 1, dtype=torch.int32, device=device),
    )

    out = get_prompt_logprobs_tensor(prompt_tokens, logprobs, token_index)
    ref = get_prompt_logprobs_tensor_ref(prompt_tokens, logprobs, token_index)
    assert torch.equal(out, ref)


GENERALIZED_TOPK_PROFILES = [
    ('topk-1', 1),
    ('topk-2', 2),
    ('topk-3', 3),
    ('topk-7', 7),
    ('topk-16', 16),
    ('topk-31', 31),
    ('topk-64', 64),
    ('topk-127', 127),
    ('topk-233', 233),
    ('topk-1024', 1024),
]


GENERALIZED_RUNTIME_PROFILES = [
    ('single-found', 1, 'all_found', 0),
    ('two-partial', 2, 'partial', 3),
    ('tiny-missing', 3, 'all_missing', 0),
    ('seven-found-padded', 7, 'all_found', 5),
    ('prime-partial', 17, 'partial', 0),
    ('below-core-count', 31, 'all_missing', 1),
    ('core-tail-minus-one', 47, 'all_found', 0),
    ('core-count', 48, 'partial', 7),
    ('core-tail-plus-one', 49, 'all_missing', 0),
    ('multi-core-rounds', 97, 'all_found', 2),
    ('large-prime', 129, 'partial', 0),
    ('persistent-tail', 257, 'all_missing', 11),
]


GENERALIZED_CASES = [
    pytest.param(num_tokens, topk_cnt, match_mode, row_padding, id=f'{topk_name}-{runtime_name}')
    for topk_name, topk_cnt in GENERALIZED_TOPK_PROFILES
    for runtime_name, num_tokens, match_mode, row_padding in GENERALIZED_RUNTIME_PROFILES
]


@pytest.mark.parametrize('num_tokens,topk_cnt,match_mode,row_padding', GENERALIZED_CASES)
def test_get_prompt_logprobs_tensor_generalized(
    num_tokens: int,
    topk_cnt: int,
    match_mode: str,
    row_padding: int,
) -> None:
    device = get_device()
    vocab_size = 129280
    row_ids = torch.arange(num_tokens, dtype=torch.int32, device=device).unsqueeze(1)
    col_ids = torch.arange(topk_cnt, dtype=torch.int32, device=device).unsqueeze(0)
    token_values = (row_ids * 4099 + col_ids * 17 + 11) % vocab_size
    logprob_values = ((row_ids * 37 - col_ids * 13) % 2001).to(torch.float32) / 37.0 - 30.0

    if row_padding:
        token_storage = torch.empty((num_tokens, topk_cnt + row_padding), dtype=torch.int32, device=device)
        logprob_storage = torch.empty((num_tokens, topk_cnt + row_padding), dtype=torch.float32, device=device)
        token_index = token_storage[:, :topk_cnt]
        logprobs = logprob_storage[:, :topk_cnt]
        token_index.copy_(token_values)
        logprobs.copy_(logprob_values)
    else:
        token_index = token_values.contiguous()
        logprobs = logprob_values.contiguous()

    row_idx = torch.arange(num_tokens, device=device)
    selected_col = (row_idx * 7 + 3) % topk_cnt
    prompt_tokens = token_index[row_idx, selected_col].contiguous()
    if match_mode == 'partial':
        prompt_tokens[1::2] = vocab_size + row_idx[1::2].to(torch.int32) + 1
    elif match_mode == 'all_missing':
        prompt_tokens = (vocab_size + row_idx.to(torch.int32) + 1).contiguous()

    out = get_prompt_logprobs_tensor(prompt_tokens, logprobs, token_index)
    ref = get_prompt_logprobs_tensor_ref(prompt_tokens, logprobs, token_index)
    assert torch.equal(out, ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('num_tokens', [256, 4096])
@pytest.mark.parametrize('topk_cnt', [1024, 4096])
def test_get_prompt_logprobs_tensor_benchmark(
    num_tokens: int,
    topk_cnt: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    device = get_device()
    vocab_size = 129280
    token_index = torch.randint(0, vocab_size, (num_tokens, topk_cnt), dtype=torch.int32, device=device)
    logprobs = torch.randn((num_tokens, topk_cnt), dtype=torch.float32, device=device)
    selected_col = torch.randint(0, topk_cnt, (num_tokens,), device=device)
    prompt_tokens = token_index[torch.arange(num_tokens, device=device), selected_col].contiguous()

    t_us = benchmark_timer(lambda: get_prompt_logprobs_tensor(prompt_tokens, logprobs, token_index))
    total_bytes = prompt_tokens.numel() * prompt_tokens.element_size() + logprobs.numel() * logprobs.element_size()
    total_bytes += token_index.numel() * token_index.element_size() + prompt_tokens.numel() * logprobs.element_size()
    benchmark_record(
        kernel='prompt_logprobs',
        operation='get_prompt_logprobs_tensor',
        params={'num_tokens': num_tokens, 'topk_cnt': topk_cnt},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1e9 / (t_us / 1e6)},
    )
