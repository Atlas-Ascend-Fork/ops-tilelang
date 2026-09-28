from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.sampling.sum_smaller_probs import get_sum_smaller_probs

pytestmark = pytest.mark.npu


def sum_smaller_probs_ref(
    probs: torch.Tensor,
    sample_results: torch.Tensor,
) -> torch.Tensor:
    m = sample_results < 0
    sample_results = sample_results.clone()
    sample_results.masked_fill_(m, 0)
    pivot = probs.gather(1, sample_results.to(torch.long).unsqueeze(1))
    mask_less = probs < pivot
    col_idx = torch.arange(probs.shape[1], device=probs.device).unsqueeze(0)
    mask_equal_before = (probs == pivot) & (col_idx <= sample_results.unsqueeze(1))
    out = (probs * (mask_less | mask_equal_before)).sum(dim=1)
    out.masked_fill_(m, 0.0)
    return out


@pytest.mark.parametrize('init_type', ['rand', 'ones'])
@pytest.mark.parametrize('has_reject', [False, True])
@pytest.mark.parametrize('vocab_size', [4000, 129280])
def test_sum_smaller_probs(init_type: str, has_reject: bool, vocab_size: int) -> None:
    batch_size = 256
    device = get_device()
    probs = getattr(torch, init_type)((batch_size, vocab_size), dtype=torch.float32, device=device)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    sample_results = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device=device)
    if has_reject:
        sample_results[::7] = -1

    ref = sum_smaller_probs_ref(probs, sample_results)
    out = get_sum_smaller_probs(probs, sample_results)
    assert torch.allclose(out, ref, atol=1e-5, rtol=0)


@pytest.mark.benchmark
@pytest.mark.parametrize('batch_size', [1, 256, 512])
def test_sum_smaller_probs_benchmark(
    batch_size: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    vocab_size = 129280
    device = get_device()
    probs = torch.rand((batch_size, vocab_size), dtype=torch.float32, device=device)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    sample_results = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device=device)

    io_bytes = batch_size * vocab_size * probs.element_size() + batch_size * (
        sample_results.element_size() + probs.element_size()
    )
    t_us = benchmark_timer(lambda: get_sum_smaller_probs(probs, sample_results))
    benchmark_record(
        kernel='sum_smaller_probs',
        operation='sum_smaller_probs',
        params={'batch_size': batch_size},
        time_us=t_us,
        extras={'bandwidth_gbs': io_bytes / 1e9 / (t_us / 1e6)},
    )
