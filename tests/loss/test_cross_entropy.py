# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

import pytest
import torch

from cann_ops_tilelang.loss import cross_entropy_forward

from .cases import generate_test_params, get_test_level
from .checks import run_precision_case
from .reference import cross_entropy_golden


@pytest.mark.npu
@pytest.mark.parametrize('case', generate_test_params(get_test_level()), ids=lambda case: case.id)
def test_cross_entropy_precision(case):
    run_precision_case(cross_entropy_forward, case, device='npu:0')


@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_cross_entropy_reference(reduction):
    x = torch.tensor([[2.0, 1.0, 0.0], [1.0, 3.0, 2.0]])
    y = torch.tensor([0, 1])
    actual = cross_entropy_golden(x, y, reduction=reduction)
    expected = torch.nn.functional.cross_entropy(x, y, reduction=reduction)
    torch.testing.assert_close(actual, expected)


@pytest.mark.npu
@pytest.mark.parametrize(
    'kind',
    ['dtype_x', 'dtype_y', 'rank_x', 'rank_y', 'length', 'stride', 'reduction', 'ignore_type', 'ignore_overflow'],
)
def test_cross_entropy_validation(kind):
    x = torch.zeros((3, 7), device='npu:0')
    y = torch.zeros((3,), device='npu:0', dtype=torch.int64)
    kwargs = {}
    if kind == 'dtype_x':
        x = x.to(torch.int32)
    if kind == 'dtype_y':
        y = y.to(torch.int32)
    if kind == 'rank_x':
        x = x.flatten()
    if kind == 'rank_y':
        y = y[:, None]
    if kind == 'length':
        y = y[:2]
    if kind == 'stride':
        x = x[:, ::2]
    if kind == 'reduction':
        kwargs['reduction'] = 'invalid'
    if kind == 'ignore_type':
        kwargs['ignore_index'] = 1.5
    if kind == 'ignore_overflow':
        kwargs['ignore_index'] = 2**63
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        cross_entropy_forward(x, y, **kwargs)


@pytest.mark.npu
@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
@pytest.mark.parametrize('n', [64, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
def test_cross_entropy_benchmark(benchmark_timer, benchmark_record, n, dtype, reduction):
    x = torch.randn((n, 129280), device='npu:0', dtype=getattr(torch, dtype))
    y = torch.randint(129280, (n,), device='npu:0')
    result = cross_entropy_forward(x, y, reduction=reduction)
    time_us = benchmark_timer(lambda: cross_entropy_forward(x, y, reduction=reduction))
    count = x.numel() * x.element_size() + y.numel() * y.element_size() + result.numel() * result.element_size()
    benchmark_record(
        kernel='cross_entropy_forward',
        operation='fwd',
        params=dict(n=n, v=129280, dtype=dtype, reduction=reduction),
        time_us=time_us,
        bandwidth_gbs=count / time_us / 1000.0,
    )
