from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.tensor.gather import gather

pytestmark = pytest.mark.npu

_GATHER_BRANCH_PARAMS = [
    (131072, 4, 131072),
    (131072, 128, 131072),
    (131072, 656, 131072),
    (131072, 4096, 131072),
    (65536, 4096, 65536),
    (262144, 4096, 262144),
    (65536, 8192, 65536),
    (262144, 8192, 262144),
    (131072, 7168 * 2, 131072),
    (129280, 7168 * 2, 131072 // 8),
]


@pytest.mark.parametrize('n,d,m', _GATHER_BRANCH_PARAMS)
def test_gather(n: int, d: int, m: int) -> None:
    device = get_device()
    x = torch.randn(n, d, device=device)
    index = torch.randint(0, n, (m,), dtype=torch.int32, device=device)
    actual = gather(x, index)
    expected = x[index]
    assert torch.equal(actual, expected)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'n,d,m',
    [
        (131072, 4, 131072),
        (131072, 128, 131072),
        (131072, 656, 131072),
        (131072, 7168 * 2, 131072),
    ],
)
def test_gather_benchmark(
    n: int,
    d: int,
    m: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    device = get_device()
    x = torch.randn(n, d, device=device)
    index = torch.randint(0, n, (m,), dtype=torch.int32, device=device)

    time_us = benchmark_timer(lambda: gather(x, index))
    total_bytes = m * d * x.element_size() * 2 + index.element_size() * m
    benchmark_record(
        kernel='gather',
        operation='gather',
        params={'n': n, 'd': d, 'm': m},
        time_us=time_us,
        extras={'bandwidth_gbs': total_bytes / 1e9 / (time_us / 1e6)},
    )
