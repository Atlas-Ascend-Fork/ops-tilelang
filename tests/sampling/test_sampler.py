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


def _update_sampling_graph_inputs(args, step):
    if step == 2:
        args = tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in args)
    if step:
        for value in args:
            if not isinstance(value, torch.Tensor):
                continue
            if value.is_floating_point():
                value.mul_(-0.5).add_(0.125)
            elif value.dtype == torch.bool:
                value.logical_not_()
            else:
                value.copy_(value.roll(1, dims=-1))
    return args


def check_sampling_graph(op, reference, args):
    from tests.testing.quant import call_quant_graph, check_quant_graph_replay, compile_quant_graph

    compiled = compile_quant_graph(op)
    records = []
    for step in range(3):
        args = _update_sampling_graph_inputs(args, step)
        inputs = [value for value in args if isinstance(value, torch.Tensor)]
        snapshots = [value.clone() for value in inputs]
        expected = reference(*args)
        result, events = call_quant_graph(compiled, *args)
        records.append(events)
        _check_sampling_output(result, expected)
        for value, before in zip(inputs, snapshots, strict=True):
            torch.testing.assert_close(value, before, rtol=0, atol=0)
        del result
    check_quant_graph_replay(records)
    torch.library.opcheck(op, args, test_utils=('test_schema', 'test_faketensor'))


def _check_sampling_output(result, expected):
    assert result.shape == expected.shape and result.stride() == expected.stride()
    assert result.dtype == expected.dtype and result.device == expected.device
    assert result.storage_offset() == 0
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_get_logits_cur_by_step_compile():
    args = (
        torch.randn((6, 256), device=get_device()),
        torch.tensor([0, 2, 4], dtype=torch.int32, device=get_device()),
        torch.tensor([True, False, True], device=get_device()),
        1,
    )
    check_sampling_graph(get_logits_cur_by_step, get_logits_cur_by_step_ref, args)


def _sampling_config_case(entry, rows=97, device=None):
    from cann_ops_tilelang.sampling.prompt_logprobs import get_prompt_logprobs_tensor
    from cann_ops_tilelang.sampling.sum_smaller_probs import get_sum_smaller_probs
    from tests.sampling.test_prompt_logprobs import get_prompt_logprobs_tensor_ref
    from tests.sampling.test_sum_smaller_probs import sum_smaller_probs_ref

    device = get_device() if device is None else device
    if entry == 'sampler':
        args = (
            torch.randn((rows * 2, 256), device=device),
            torch.arange(rows, dtype=torch.int32, device=device) * 2,
            torch.arange(rows, device=device) % 2 == 0,
            1,
        )
        return get_logits_cur_by_step, get_logits_cur_by_step_ref, args, '_get_logits_cur_by_step_tl_ascend'
    if entry == 'prompt_logprobs':
        args = (
            torch.arange(rows, dtype=torch.int32, device=device) % 17,
            torch.randn((rows, 16), device=device),
            torch.arange(16, dtype=torch.int32, device=device).repeat(rows, 1),
        )
        return get_prompt_logprobs_tensor, get_prompt_logprobs_tensor_ref, args, '_get_prompt_logprobs_tensor_tl_ascend'
    args = (
        torch.rand((rows, 256), device=device),
        torch.arange(rows, dtype=torch.int32, device=device) % 257 - 1,
    )
    return get_sum_smaller_probs, sum_smaller_probs_ref, args, '_sum_smaller_probs_tl_ascend'


def _check_sampling_config_call(compiled, reference, args, calls, recapture, entry):
    from cann_ops_tilelang import config
    from tests.testing.quant import call_quant_graph

    before = len(calls)
    snapshots = [value.clone() for value in args if isinstance(value, torch.Tensor)]
    expected = reference(*args)
    result, events = call_quant_graph(compiled, *args)
    assert bool(events['acl_graph_capture']) is recapture
    assert events['acl_graph_replay'] > 0 and events['fx_run_eagerly'] == 0
    assert (len(calls) > before) is recapture
    cores = config.get_num_vector_cores()
    expected_cores = min(args[0].shape[0], cores) if entry == 'sum_smaller_probs' else cores
    assert calls[-1] == expected_cores
    _check_sampling_output(result, expected)
    inputs = [value for value in args if isinstance(value, torch.Tensor)]
    for value, snapshot in zip(inputs, snapshots, strict=True):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
    del result


def _set_sampling_core_cap(patch, cap):
    from cann_ops_tilelang import config

    if cap is None:
        patch.delenv('OPS_TILELANG_NUM_VECTOR_CORES')
    else:
        patch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', str(cap))
    config.clear_runtime_caches()


@pytest.mark.parametrize('entry', ['sampler', 'prompt_logprobs', 'sum_smaller_probs'])
@torch.inference_mode()
def test_sampling_compile_core_config(entry, monkeypatch):
    from importlib import import_module

    from cann_ops_tilelang import config
    from tests.testing.quant import compile_quant_graph

    with monkeypatch.context() as patch:
        patch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
        config.reset_runtime_config()
        try:
            op, reference, args, factory_name = _sampling_config_case(entry)
            module = import_module('cann_ops_tilelang.sampling.' + entry)
            factory = getattr(module, factory_name)
            calls = []

            def record(*factory_args, **kwargs):
                calls.append(factory_args[1])
                return factory(*factory_args, **kwargs)

            patch.setattr(module, factory_name, record)
            compiled = compile_quant_graph(op)

            def check(recapture):
                _check_sampling_config_call(compiled, reference, args, calls, recapture, entry)

            check(True)
            half_ai = max(1, config.get_num_ai_cores() // 2)
            config.set_num_ai_cores(half_ai)
            check(True)
            config.set_num_ai_cores(half_ai)
            check(False)
            with pytest.raises(ValueError):
                config.set_num_ai_cores(0)
            check(False)
            config.reset_runtime_config()
            check(True)
            cores = config.get_num_vector_cores()
            for cap in (max(1, cores // 2), max(1, cores // 3), None):
                _set_sampling_core_cap(patch, cap)
                check(True)
                config.clear_runtime_caches()
                check(False)
        finally:
            config.reset_runtime_config()
            patch.undo()
            config.clear_runtime_caches()


@pytest.mark.parametrize('entry', ['sampler', 'prompt_logprobs', 'sum_smaller_probs'])
def test_sampling_fake_without_runtime_queries(entry, monkeypatch):
    from importlib import import_module

    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    module = import_module('cann_ops_tilelang.sampling.' + entry)

    def reject_query(*args, **kwargs):
        raise AssertionError('Fake execution must not query runtime tuning or build kernels.')

    with FakeTensorMode():
        op, _, args, factory_name = _sampling_config_case(entry, device='npu')
        for name in ('get_num_vector_cores', 'require_ascend', factory_name):
            monkeypatch.setattr(module, name, reject_query)
        result = op(*args)
        assert isinstance(result, FakeTensor)
        assert result.shape == ((97, 256) if entry == 'sampler' else (97,))
