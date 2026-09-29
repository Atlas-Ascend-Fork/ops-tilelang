from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.sampling.sampler import get_logits_cur_by_step

pytestmark = pytest.mark.npu


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
