# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

from itertools import product

import pytest
import torch

from cann_ops_tilelang.config import clear_runtime_caches, get_num_vector_cores
from cann_ops_tilelang.loss import cross_entropy_forward

from .cases import Case
from .checks import assert_forward_matches, run_precision_case

pytestmark = pytest.mark.npu


@pytest.mark.parametrize('dtype,reduction', list(product(['float32', 'float16', 'bfloat16'], ['none', 'sum', 'mean'])))
@pytest.mark.parametrize('route', ['remainder', 'whole_row', 'single_block'])
def test_cross_entropy_routes(dtype, reduction, route):
    cores = get_num_vector_cores()
    if route == 'remainder':
        n, v = cores + max(1, cores // 7), 129280
    elif route == 'whole_row':
        n, v = 2 * cores + 1, 129280
    else:
        n, v = 16 * 256 + 1, 8193
    case = Case(route, n, v, dtype=dtype, reduction=reduction, ignore='partial')
    run_precision_case(cross_entropy_forward, case, device='npu:0')


@pytest.fixture
def single_vector_core(monkeypatch):
    monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', '1')
    clear_runtime_caches()
    assert get_num_vector_cores() == 1
    yield
    clear_runtime_caches()


@pytest.mark.usefixtures('single_vector_core')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('reduction', ['sum', 'mean'])
@pytest.mark.parametrize('pattern', ['overflow', 'after_overflow', 'finite', 'all_ignored'])
def test_partial_accumulation_extremes(dtype, reduction, pattern):
    n = 256 if reduction == 'mean' else (7 if pattern == 'after_overflow' else 5)
    v = 129280 if reduction == 'mean' else (24577 if dtype == torch.float32 else 49153)
    magnitude = 1e38 if pattern == 'finite' else 2.5e38
    logits = torch.zeros((n, v), dtype=dtype)
    logits[:, 0] = magnitude
    logits[:, 1] = -magnitude
    targets = torch.ones(n, dtype=torch.int64)
    if pattern == 'after_overflow':
        targets[-2] = -100
        targets[-1] = 0
    if pattern == 'all_ignored':
        targets.fill_(-100)
    logits_npu = logits.to('npu:0')
    targets_npu = targets.to('npu:0')
    output = cross_entropy_forward(logits_npu, targets_npu, reduction=reduction)
    assert_forward_matches(output, logits, targets, reduction=reduction, expected_device=logits_npu.device)
