from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.sampling.softmax import log_softmax, softmax, softmax_and_logsoftmax

pytestmark = pytest.mark.npu


def assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str, eps: float) -> None:
    """Compare an NPU result with its high-precision reference."""
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu().to(actual_cpu.dtype)
    rtol = max(eps, torch.finfo(actual_cpu.dtype).eps)
    torch.testing.assert_close(actual_cpu, expected_cpu, atol=eps, rtol=rtol)


def _check_softmax(logits: torch.Tensor, prob: torch.Tensor, eps: float) -> None:
    prob_ref = torch.softmax(logits.cpu().double(), dim=-1, dtype=torch.double)
    assert_close(prob, prob_ref, 'softmax', eps)


def _check_logsoftmax(logits: torch.Tensor, logprob: torch.Tensor, eps: float) -> None:
    block_size = 8192

    logits_f32 = logits.float()
    vocab_size = logits_f32.shape[-1]
    lse = torch.zeros(logits_f32.shape[0], dtype=torch.float32, device=logits_f32.device)
    max_val = torch.full_like(lse, float('-inf'))
    for i in range(0, vocab_size, block_size):
        block = logits_f32[:, i : i + block_size]
        block_max = block.max(dim=-1).values
        new_max = torch.maximum(max_val, block_max)
        lse = torch.exp(max_val - new_max) * lse + torch.exp(block - new_max.unsqueeze(-1)).sum(dim=-1)
        max_val = new_max
    logprob_online = logits_f32 - (torch.log(lse) + max_val).unsqueeze(-1)
    logprob_online = logprob_online.to(logits.dtype)
    assert_close(logprob, logprob_online, 'log_softmax_online', eps)

    logprob_ref = torch.log_softmax(logits.cpu().double(), dim=-1, dtype=torch.double)
    assert_close(logprob_online, logprob_ref, 'log_softmax_sim', eps)
    assert_close(logprob, logprob_ref, 'log_softmax', eps)


@pytest.mark.parametrize(
    'b,h,padding_size,float_type,eps',
    [
        (4096, 129280, 0, torch.float32, 1e-5),
        (512, 1024, 0, torch.float32, 1e-5),
        (1024, 4000, 0, torch.float32, 1e-5),
        (4096, 129280, 0, torch.bfloat16, 1e-5),
        (512, 1024, 0, torch.bfloat16, 1e-5),
        (1024, 4000, 0, torch.bfloat16, 1e-5),
        (4096, 129280, 0, torch.float16, 1e-5),
        (512, 1024, 0, torch.float16, 1e-5),
        (1024, 4000, 0, torch.float16, 1e-5),
    ],
)
def test_logsoftmax(
    b: int,
    h: int,
    padding_size: int,
    float_type: torch.dtype,
    eps: float,
) -> None:
    logits = torch.randn((b, h + padding_size), dtype=float_type, device=get_device())[:, :h]

    prob, logprob = softmax_and_logsoftmax(logits)
    _check_logsoftmax(logits, logprob, eps)
    _check_softmax(logits, prob, eps)

    # Single-kernel variants must produce identical results to the fused kernel
    prob_single = softmax(logits)
    logprob_single = log_softmax(logits)
    assert (prob_single == prob).all()
    assert (logprob_single == logprob).all()


@pytest.mark.parametrize(
    'num_tokens,vocab_size,padding_size,dtype',
    [
        pytest.param(113, 127, 1, torch.float32, id='persistent-multi-wave'),
        pytest.param(3, 16383, 1, torch.bfloat16, id='bf16-block-tail'),
        pytest.param(3, 16384, 8, torch.float16, id='fp16-full-block'),
        pytest.param(3, 16385, 7, torch.float32, id='cross-block-single-tail'),
        pytest.param(2, 129280, 0, torch.bfloat16, id='production-vocab-multi-block'),
    ],
)
def test_softmax_ascend_generalized(num_tokens: int, vocab_size: int, padding_size: int, dtype: torch.dtype) -> None:
    """Cover persistent scheduling, dtype variants, row strides, and UB tails."""
    storage = torch.randn((num_tokens, vocab_size + padding_size), dtype=dtype, device='npu')
    logits = storage[:, :vocab_size]
    assert logits.stride(0) == vocab_size + padding_size

    prob, logprob = softmax_and_logsoftmax(logits)
    _check_softmax(logits, prob, 1e-5)
    _check_logsoftmax(logits, logprob, 1e-5)
    assert prob.is_contiguous()
    assert logprob.is_contiguous()


def test_softmax_ascend_public_variants_and_stability() -> None:
    """All public wrappers share numerics and preserve the in-place contract."""
    logits = torch.tensor(
        [[1000.0, 999.0, -1000.0, 0.0], [-1000.0, -999.0, -998.0, -997.0], [7.0, 7.0, 7.0, 7.0]],
        dtype=torch.float32,
        device='npu',
    )
    prob, logprob = softmax_and_logsoftmax(logits)
    prob_single, logprob_single = softmax(logits), log_softmax(logits)
    assert torch.equal(prob_single, prob)
    assert torch.equal(logprob_single, logprob)
    _check_softmax(logits, prob, 5e-5)
    _check_logsoftmax(logits, logprob, 5e-5)

    softmax_input = logits.clone()
    softmax_inplace = softmax(softmax_input, inplace=True)
    assert softmax_inplace.data_ptr() == softmax_input.data_ptr()
    assert torch.equal(softmax_inplace, prob)

    logsoftmax_input = logits.clone()
    logsoftmax_inplace = log_softmax(logsoftmax_input, inplace=True)
    assert logsoftmax_inplace.data_ptr() == logsoftmax_input.data_ptr()
    assert torch.equal(logsoftmax_inplace, logprob)

    wide_logits = torch.randn((2, 16385), dtype=torch.float32, device='npu')
    wide_prob = softmax(wide_logits)
    wide_logprob = log_softmax(wide_logits)
    wide_softmax_input = wide_logits.clone()
    assert torch.equal(softmax(wide_softmax_input, inplace=True), wide_prob)
    wide_logsoftmax_input = wide_logits.clone()
    assert torch.equal(log_softmax(wide_logsoftmax_input, inplace=True), wide_logprob)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'b,h',
    [
        (4096, 129280),
    ],
)
def test_softmax_and_logsoftmax_benchmark(
    b: int,
    h: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    logits = torch.randn((b, h), dtype=torch.float32, device=get_device())
    elem_bytes = logits.numel() * logits.element_size()

    t_us = benchmark_timer(lambda: softmax_and_logsoftmax(logits))

    # I/O: read logits + write softmax + write logsoftmax = 3x
    benchmark_record(
        kernel='softmax',
        operation='softmax_and_logsoftmax',
        params={'b': b, 'h': h},
        time_us=t_us,
        extras={'bandwidth_gbs': elem_bytes * 3 / 1e9 / (t_us / 1e6)},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'b,h',
    [
        (4096, 129280),
    ],
)
def test_softmax_benchmark(
    b: int,
    h: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    logits = torch.randn((b, h), dtype=torch.float32, device=get_device())
    elem_bytes = logits.numel() * logits.element_size()

    t_us = benchmark_timer(lambda: softmax(logits))

    # I/O: read logits + write softmax = 2x
    benchmark_record(
        kernel='softmax',
        operation='softmax',
        params={'b': b, 'h': h},
        time_us=t_us,
        extras={'bandwidth_gbs': elem_bytes * 2 / 1e9 / (t_us / 1e6)},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'b,h',
    [
        (4096, 129280),
    ],
)
def test_log_softmax_benchmark(
    b: int,
    h: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    logits = torch.randn((b, h), dtype=torch.float32, device=get_device())
    elem_bytes = logits.numel() * logits.element_size()

    t_us = benchmark_timer(lambda: log_softmax(logits))

    # I/O: read logits + write logsoftmax = 2x
    benchmark_record(
        kernel='softmax',
        operation='log_softmax',
        params={'b': b, 'h': h},
        time_us=t_us,
        extras={'bandwidth_gbs': elem_bytes * 2 / 1e9 / (t_us / 1e6)},
    )
