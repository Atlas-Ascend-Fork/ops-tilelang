# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

from importlib import import_module

import pytest
import torch
from tilelang import next_power_of_2
from torch._subclasses.fake_tensor import FakeTensorMode

from cann_ops_tilelang import cross_entropy_forward
from cann_ops_tilelang.config import clear_runtime_caches, get_num_vector_cores, set_num_ai_cores
from cann_ops_tilelang.loss.cross_entropy_kernel import _batched_small_kernel

from .reference import cross_entropy_golden

pytestmark = pytest.mark.npu


@pytest.fixture
def isolated_graph_rng():
    device_index = torch.npu.current_device()
    generator = torch.npu.default_generators[device_index]
    original = generator.graphsafe_get_state()
    generator.graphsafe_set_state(generator.clone_state())
    try:
        yield
    finally:
        generator.graphsafe_set_state(original)


@pytest.fixture(autouse=True)
def reset_compile_cache():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _inputs(dtype, n=7):
    logits = torch.randn((n, 41), device='npu', dtype=dtype)[:, :33]
    targets = torch.arange(n, device=logits.device, dtype=torch.int64) % 33
    targets[::3] = -100
    return logits, targets


def _check_forward(forward, logits, targets, reduction):
    caller_mode = torch.is_grad_enabled()
    before_logits = logits.detach().cpu().clone()
    before_targets = targets.cpu().clone()
    expected = cross_entropy_golden(before_logits, before_targets, ignore_index=-100, reduction=reduction)
    with torch.no_grad():
        output = forward(logits, targets, ignore_index=-100, reduction=reduction)
        assert not torch.is_grad_enabled()
        assert output.shape == expected.shape
        assert output.dtype == torch.float32
        assert output.device == logits.device
        assert not output.requires_grad
        assert output.grad_fn is None
        assert output.untyped_storage() is not logits.untyped_storage()
        assert output.untyped_storage() is not targets.untyped_storage()
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-4, atol=1e-5, equal_nan=True)
        torch.testing.assert_close(logits.detach().cpu(), before_logits, rtol=0, atol=0)
        torch.testing.assert_close(targets.cpu(), before_targets, rtol=0, atol=0)
    assert torch.is_grad_enabled() == caller_mode


@pytest.mark.usefixtures('isolated_graph_rng')
@pytest.mark.parametrize('dtype', (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize('reduction', ('none', 'sum', 'mean'))
def test_compile_replay(dtype, reduction):
    logits, targets = _inputs(dtype)
    logits.requires_grad_()
    direct = torch.ops.cann_ops_tilelang.cross_entropy_forward.default
    compiled = torch.compile(cross_entropy_forward, backend='npugraph_ex', fullgraph=True, dynamic=False)
    original_logits = logits
    original_targets = targets
    for iteration in range(3):
        if iteration == 1:
            with torch.no_grad():
                logits.mul_(0.5).add_(torch.randn_like(logits))
                targets.copy_(
                    torch.tensor(
                        [32, -100, 5, 4, -100, 9, 1],
                        device=targets.device,
                        dtype=torch.int64,
                    )
                )
            assert logits is original_logits
            assert targets is original_targets
        elif iteration == 2:
            logits, targets = _inputs(dtype)
            logits.requires_grad_()
            assert logits is not original_logits
            assert targets is not original_targets
            assert logits.untyped_storage() is not original_logits.untyped_storage()
            assert targets.untyped_storage() is not original_targets.untyped_storage()
        assert logits.shape == (7, 33)
        assert logits.stride() == (41, 1)
        assert targets.shape == (7,)
        assert targets.stride() == (1,)
        for forward in (cross_entropy_forward, direct, compiled):
            _check_forward(forward, logits, targets, reduction)


@pytest.mark.usefixtures('isolated_graph_rng')
@pytest.mark.parametrize('reduction', ('none', 'sum', 'mean'))
@pytest.mark.parametrize('n', (0, 7), ids=('empty', 'all_ignored'))
def test_compile_empty_and_all_ignored(n, reduction):
    logits, targets = _inputs(torch.float32, n)
    targets.fill_(-100)
    direct = torch.ops.cann_ops_tilelang.cross_entropy_forward.default
    compiled = torch.compile(
        cross_entropy_forward,
        backend='eager' if n == 0 else 'npugraph_ex',
        fullgraph=True,
        dynamic=False,
    )
    for forward in (cross_entropy_forward, direct, compiled):
        _check_forward(forward, logits, targets, reduction)


@pytest.mark.parametrize('dtype', (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize('reduction', ('none', 'sum', 'mean'))
@pytest.mark.parametrize('n', (0, 7))
def test_custom_op_fake_metadata(monkeypatch, dtype, reduction, n):
    host = import_module('cann_ops_tilelang.loss.cross_entropy')

    def fail(*args, **kwargs):
        raise AssertionError('fake execution must not enter the host implementation, compiler, or device query')

    monkeypatch.setattr(host, '_cross_entropy_forward_impl', fail)
    monkeypatch.setattr(host, 'get_num_vector_cores', fail)
    monkeypatch.setattr(host.tilelang, 'compile', fail)
    caller_mode = torch.is_grad_enabled()
    with FakeTensorMode(), torch.no_grad():
        logits = torch.empty_strided((n, 33), (41, 1), device='npu', dtype=dtype)
        targets = torch.empty((n,), device=logits.device, dtype=torch.int64)
        output = cross_entropy_forward(logits, targets, ignore_index=-100, reduction=reduction)
        second = torch.ops.cann_ops_tilelang.cross_entropy_forward.default(
            logits, targets, ignore_index=-100, reduction=reduction
        )
        assert not torch.is_grad_enabled()
        for result in (output, second):
            assert result.shape == ((n,) if reduction == 'none' else ())
            assert result.dtype == torch.float32
            assert result.device == logits.device
            assert not result.requires_grad
            assert result.grad_fn is None
            assert result is not logits
            assert result is not targets
            assert result.untyped_storage() is not logits.untyped_storage()
            assert result.untyped_storage() is not targets.untyped_storage()
        assert output is not second
        assert output.untyped_storage() is not second.untyped_storage()
        invalid_targets = torch.empty((n,), device=logits.device, dtype=torch.int32)
        with pytest.raises(TypeError, match='int64 targets'):
            cross_entropy_forward(logits, invalid_targets, reduction=reduction)
    assert torch.is_grad_enabled() == caller_mode


@pytest.mark.parametrize('reduction', ('none', 'sum', 'mean'))
def test_custom_op_opcheck(reduction):
    logits, targets = _inputs(torch.float32)
    caller_mode = torch.is_grad_enabled()
    with torch.no_grad():
        torch.library.opcheck(
            torch.ops.cann_ops_tilelang.cross_entropy_forward.default,
            (logits, targets),
            kwargs={'ignore_index': -100, 'reduction': reduction},
            test_utils=('test_schema', 'test_faketensor'),
        )
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled() == caller_mode


@pytest.mark.parametrize('invalid_ignore', (1.5, 2**63))
def test_custom_op_rejects_invalid_ignore_index(invalid_ignore):
    logits, targets = _inputs(torch.float32)
    _check_forward(cross_entropy_forward, logits, targets, 'none')
    caller_mode = torch.is_grad_enabled()
    with torch.no_grad():
        before_logits = logits.cpu().clone()
        before_targets = targets.cpu().clone()
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            cross_entropy_forward(logits, targets, ignore_index=invalid_ignore)
        assert not torch.is_grad_enabled()
        torch.testing.assert_close(logits.cpu(), before_logits, rtol=0, atol=0)
        torch.testing.assert_close(targets.cpu(), before_targets, rtol=0, atol=0)
    assert torch.is_grad_enabled() == caller_mode


@pytest.mark.usefixtures('isolated_graph_rng')
@pytest.mark.parametrize('grad_enabled', (False, True))
@pytest.mark.parametrize('requires_grad', (False, True))
def test_compile_preserves_grad_mode(grad_enabled, requires_grad):
    logits, targets = _inputs(torch.float32)
    logits.requires_grad_(requires_grad)
    direct = torch.ops.cann_ops_tilelang.cross_entropy_forward.default
    compiled = torch.compile(cross_entropy_forward, backend='npugraph_ex', fullgraph=True, dynamic=False)
    caller_mode = torch.is_grad_enabled()
    with torch.set_grad_enabled(grad_enabled):
        for forward in (cross_entropy_forward, direct, compiled, compiled):
            _check_forward(forward, logits, targets, 'mean')
            assert torch.is_grad_enabled() == grad_enabled
            assert logits.requires_grad == requires_grad
    assert torch.is_grad_enabled() == caller_mode


@pytest.mark.usefixtures('isolated_graph_rng')
def test_compile_recaptures_core_config(monkeypatch):
    host = import_module('cann_ops_tilelang.loss.cross_entropy')
    factory = _batched_small_kernel
    launched_cores = []

    def record_factory(*args):
        launched_cores.append(args[-1])
        return factory(*args)

    monkeypatch.setattr(host, '_batched_small_kernel', record_factory)
    monkeypatch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
    clear_runtime_caches()
    available = get_num_vector_cores()
    n = 2 * max(64, available)
    logits = torch.randn((n, 257), device='npu', dtype=torch.float32)
    targets = torch.arange(n, device=logits.device, dtype=torch.int64) % 257
    targets[::3] = -100
    initial_cores = min(available, next_power_of_2((n + 3) // 4))
    compiled = torch.compile(cross_entropy_forward, backend='npugraph_ex', fullgraph=True, dynamic=False)
    _check_forward(compiled, logits, targets, 'none')
    assert launched_cores and set(launched_cores) == {initial_cores}
    del compiled
    set_num_ai_cores(1)
    clear_runtime_caches()
    assert get_num_vector_cores() == 2
    torch.compiler.reset()
    launched_cores.clear()
    compiled = torch.compile(cross_entropy_forward, backend='npugraph_ex', fullgraph=True, dynamic=False)
    _check_forward(compiled, logits, targets, 'none')
    assert launched_cores and set(launched_cores) == {2}
    del compiled
    monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', '0')
    clear_runtime_caches()
    with pytest.raises(ValueError, match='OPS_TILELANG_NUM_VECTOR_CORES'):
        get_num_vector_cores()
