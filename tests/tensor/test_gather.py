# MIT License
#
# Copyright (c) 2026 ops-tilelang contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.tensor.gather import _gather_fake, gather
from tests.tensor import gather_graph_utils as graph
from tests.tensor.gather_graph_utils import reset_compile_cache as reset_compile_cache

_LOGGER = logging.getLogger(__name__)

pytestmark = [pytest.mark.npu, pytest.mark.usefixtures('reset_compile_cache')]


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


def _make_compile_inputs(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    offset: int = 0,
):
    device = get_device()
    count = 1
    for size in shape:
        count *= size
    x = (torch.arange(count, device=device).reshape(shape) % 97 + offset * 3).to(dtype)
    index = torch.tensor([shape[0] - 1, 0, -1, shape[0], 2, 2, 1], dtype=index_dtype, device=device)
    if offset:
        index = index.roll(offset)
    return x, index


def _expected_gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    # Compute the oracle on CPU; negative indices are invalid rather than
    # Python-style indexing from the end.
    source = x.cpu()
    indices = index.cpu().long()
    expected = torch.zeros((indices.numel(), *x.shape[1:]), dtype=x.dtype)
    valid = (indices >= 0) & (indices < x.shape[0])
    expected[valid] = source[indices[valid]]
    return expected.to(x.device)


@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize(
    ('shape', 'dtype'),
    [
        ((19,), torch.int32),
        ((19, 3), torch.bfloat16),
        ((19, 7, 5), torch.float32),
        ((19, 9001), torch.int8),
    ],
)
@torch.inference_mode()
def test_gather_compile(shape: tuple[int, ...], dtype: torch.dtype, index_dtype: torch.dtype) -> None:
    inputs = _make_compile_inputs(shape, dtype, index_dtype)
    compiled = torch.compile(gather, backend='npugraph_ex', fullgraph=True, dynamic=False)
    for step in range(3):
        if step:
            replacement = _make_compile_inputs(shape, dtype, index_dtype, offset=step)
            inputs = graph.replace_tensor_inputs(inputs, replacement, inplace=step == 1)
        before = tuple(tensor.clone() for tensor in inputs)
        expected = _expected_gather(*inputs)
        if step == 0:
            assert torch.equal(gather(*inputs), expected)
        actual = compiled(*inputs)
        assert actual.shape == expected.shape
        assert actual.dtype == inputs[0].dtype and actual.device == inputs[0].device
        assert actual.is_contiguous()
        assert actual.data_ptr() != inputs[0].data_ptr()
        assert torch.equal(actual, expected), f'gather mismatch at step {step}'
        for tensor, original in zip(inputs, before, strict=True):
            assert torch.equal(tensor, original), f'read-only input changed at step {step}'
        del actual


@torch.inference_mode()
def test_gather_graph_replay(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=__name__)

    def model(x, index):
        return gather(x, index) + 1

    inputs = _make_compile_inputs((23, 9), torch.float32, torch.int32)
    compiled = torch.compile(model, backend='npugraph_ex', fullgraph=True, dynamic=False)
    events_by_call = []
    for step in range(3):
        inputs[0].add_(step)
        expected = _expected_gather(*inputs) + 1
        actual, events = graph.profile_npu_call(compiled, *inputs)
        events_by_call.append(events)
        assert torch.equal(actual, expected)
        del actual
    _LOGGER.info('gather ACL Graph events by call: %s', events_by_call)
    graph.assert_acl_graph_replay(events_by_call)


@torch.inference_mode()
def test_gather_compile_core_config(monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
    inputs = _make_compile_inputs((23, 9), torch.float32, torch.int32)
    expected = _expected_gather(*inputs)
    graph.check_core_config(gather, inputs, expected, monkeypatch, caplog)


def test_gather_registration() -> None:
    inputs = _make_compile_inputs((19, 7, 5), torch.float32, torch.int64)
    torch.library.opcheck(gather, inputs, test_utils=('test_schema', 'test_faketensor'))


@pytest.mark.parametrize(('shape', 'count'), [((0, 3), 4), ((4, 3), 0), ((4, 0), 3)])
@torch.inference_mode()
def test_gather_empty_compile(shape: tuple[int, ...], count: int) -> None:
    device = get_device()
    x = torch.empty(shape, dtype=torch.float32, device=device)
    index = torch.zeros(count, dtype=torch.int32, device=device)
    compiled = torch.compile(gather, backend='npugraph_ex', fullgraph=True, dynamic=False)
    expected = _expected_gather(x, index)
    for implementation in (gather, compiled):
        actual = implementation(x, index)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(actual, expected)


@pytest.mark.parametrize('invalid', ['scalar', 'index_rank', 'index_dtype', 'layout'])
def test_gather_invalid_metadata(invalid: str) -> None:
    x, index = _make_compile_inputs((19, 3), torch.float32, torch.int32)
    error = ValueError
    if invalid == 'scalar':
        x = x[0, 0]
    elif invalid == 'index_rank':
        index = index.unsqueeze(0)
    elif invalid == 'index_dtype':
        index = index.float()
        error = TypeError
    else:
        x = x.t()
    for implementation, kwargs in ((gather, {}), (_gather_fake, {})):
        with pytest.raises(error):
            implementation(x, index, **kwargs)


def test_gather_rejects_optional_out() -> None:
    x, index = _make_compile_inputs((19, 3), torch.float32, torch.int32)
    with pytest.raises(RuntimeError, match='expected at most 2 argument'):
        gather(x, index, out=torch.empty((index.numel(), 3), device=x.device))


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
