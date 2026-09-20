import itertools
import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_device
from tests.moe.topk_ref import stable_topk as torch_stable_topk
from tests.testing.bench import make_param_id
from tests.testing.generator import generate_num_tokens, get_test_level
from tests.testing.numeric import assert_equal, count_bytes

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


_EXPERT_CONFIGS = [
    (72, 6),   # oss-ignore-note: V3 lite
    (32, 6),   # oss-ignore-note: V3 MM 0
    (64, 6),   # oss-ignore-note: V3 MM 1
    (96, 6),   # oss-ignore-note: V3 MM 2
    (16, 6),   # oss-ignore-note: V3 MM 3
    (36, 6),   # oss-ignore-note: V3 MM 4
    (108, 6),  # oss-ignore-note: V3 MM 5
    (128, 6),  # oss-ignore-note: V3 MM 6
    (144, 6),  # oss-ignore-note: V3 MM 7
    (256, 8),  # oss-ignore-note: V3 full
    (8, 2),    # oss-ignore-line: V33 lite
    (128, 4),  # oss-ignore-line: V33 mid
    (384, 6),  # oss-ignore-line: V33 full
    (256, 6),  # oss-ignore-line: V33 infra V1
]  # fmt: off


def generate_test_data(params):
    num_tokens = params['num_tokens']
    num_experts = params['num_experts']
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device=get_device())
    return scores


def generate_test_params(level: int) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'num_experts': num_experts,
            'num_topk': num_topk,
        }
        for num_tokens in generate_num_tokens(level)
        # oss-ignore-note: Researcher may use various combination of num_experts and num_topk, so we try to support a wide range of configurations  # noqa: E501
        for num_experts, num_topk in itertools.product([1, 35, 1025], [1, 9, 32])
        if num_experts >= num_topk
    ]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_topk_gate(params):
    scores = generate_test_data(params)
    num_topk = params['num_topk']

    topk_idx_ref = torch_stable_topk(scores, num_topk)
    topk_idx = cann_ops_tilelang.topk_gate(scores, num_topk)
    assert_equal(topk_idx, topk_idx_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'params',
    [
        {'num_tokens': num_tokens, 'num_experts': num_experts, 'num_topk': num_topk}
        for num_tokens in generate_num_tokens(0)
        for num_experts, num_topk in _EXPERT_CONFIGS
    ],
    ids=make_param_id,
)
def test_topk_gate_benchmark(benchmark_timer, benchmark_record, params):
    scores = generate_test_data(params)
    num_topk = params['num_topk']

    topk_idx = cann_ops_tilelang.topk_gate(scores, num_topk)

    t_us = benchmark_timer(lambda: cann_ops_tilelang.topk_gate(scores, num_topk))
    num_bytes = count_bytes(scores, topk_idx)
    bandwidth_gbs = num_bytes / t_us / 1e3

    benchmark_record(
        kernel='topk_gate',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
