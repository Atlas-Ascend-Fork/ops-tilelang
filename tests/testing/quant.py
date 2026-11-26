import importlib
import logging
import math
from contextlib import contextmanager
from functools import partial

import pytest
import torch
from torch.utils._pytree import tree_leaves

from cann_ops_tilelang import config
from cann_ops_tilelang.quant.common import get_packed_ue8m0_pack_factor, unpack_from_e2m1fn_x2
from cann_ops_tilelang.utils import align_up, ceil_div
from tests.testing.numeric import assert_equal


def quant_level_rank(xc: torch.Tensor, fmt: str) -> torch.Tensor:
    """Map each fp8/fp4 code to its rank in the value-sorted list of representable levels.

    Two outputs are 'numerically adjacent' iff their ranks differ by at most 1, which holds
    across zero and exponent boundaries (unlike raw bit-pattern subtraction).
    """
    if fmt == 'e4m3':
        codes = torch.arange(256, dtype=torch.uint8, device=xc.device)
        values = codes.view(torch.float8_e4m3fn).float()
        q = xc.float()
    else:
        codes = torch.arange(16, dtype=torch.uint8, device=xc.device)
        values = unpack_from_e2m1fn_x2(codes.view(-1, 1)).flatten()
        q = unpack_from_e2m1fn_x2(xc).float()
    finite = torch.isfinite(values)
    levels = torch.unique(values[finite])  # sorted ascending
    return torch.searchsorted(levels, q.flatten())


def clear_unused_sf(sf: torch.Tensor, hidden: int, num_per_channels: int) -> torch.Tensor:
    """Zero out unused sf entries beyond the actual channel block count.

    Args:
        sf: Scale-factor tensor to clean up.
        hidden: Number of hidden channels in the original tensor.
        num_per_channels: Number of channels per scaling block.

    Returns:
        Flattened sf tensor with unused trailing entries set to zero.
    """
    num_channel_blocks = ceil_div(hidden, num_per_channels)
    aligned_num_channel_blocks = align_up(num_channel_blocks, get_packed_ue8m0_pack_factor())
    sf_flattened = sf.contiguous().flatten()
    sf_flattened = (sf_flattened.as_strided(sf_flattened.shape, (1,)).view(torch.uint8)).view(
        -1, aligned_num_channel_blocks
    )
    sf_flattened[:, num_channel_blocks:] = 0
    return sf_flattened


def clear_unused_sf_col_pack(sf: torch.Tensor, num_tokens: int, num_per_tokens: int) -> torch.Tensor:
    """Zero out unused sf entries beyond the actual token block count.

    Args:
        sf: Scale-factor tensor to clean up.
        num_tokens: Number of tokens in the original tensor.
        num_per_tokens: Number of tokens per scaling block.

    Returns:
        uint8 view of sf with unused trailing entries set to zero.
    """
    num_token_blocks = ceil_div(num_tokens, num_per_tokens)
    pack_factor = get_packed_ue8m0_pack_factor()
    aligned_num_token_blocks = align_up(num_token_blocks, pack_factor)
    if sf.numel() == 0:
        return sf.view(torch.uint8)
    sf_flattened = sf.contiguous().flatten()
    sf_uint8 = sf_flattened.as_strided(sf_flattened.shape, (1,)).view(torch.uint8)
    sf_unpacked = sf_uint8.view(aligned_num_token_blocks // pack_factor, -1, pack_factor)
    sf_unpacked[num_token_blocks // pack_factor :, :, num_token_blocks % pack_factor :] = 0
    return sf_uint8.view(aligned_num_token_blocks // pack_factor, -1)


def compile_quant_graph(op):
    """Build an independent fixed-shape graph for a small correctness test."""
    torch.compiler.reset()
    return torch.compile(op, backend='npugraph_ex', fullgraph=True, dynamic=False)


def call_quant_graph(compiled, *args, **kwargs):
    """Return one call's output and evidence of capture/replay, not just Dynamo tracing."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        result = compiled(*args, **kwargs)
        torch.npu.synchronize()
    events = {event.key: event.count for event in profile.key_averages()}
    names = ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly')
    counts = {name: events.get(name, 0) for name in names}
    logging.getLogger(__name__).info('NPU graph events: %s', counts)
    return result, counts


def check_quant_graph_replay(records):
    """Same-storage input updates must replay; mutated state may recapture after replacement."""
    if not any(record['acl_graph_capture'] for record in records):
        raise AssertionError(f'No NPU graph capture recorded: {records}')
    if records[1]['acl_graph_replay'] <= 0:
        raise AssertionError(f'The second call did not replay the NPU graph: {records}')
    if records[1]['acl_graph_capture'] != 0:
        raise AssertionError(f'The second call recaptured the NPU graph: {records}')
    if not all(record['fx_run_eagerly'] == 0 for record in records):
        raise AssertionError(f'Unexpected eager fallback: {records}')


@contextmanager
def quant_config_test_context(monkeypatch):
    """Isolate tuning changes and refresh queries after restoring the environment."""
    with monkeypatch.context() as patch:
        patch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
        config.reset_runtime_config()
        try:
            yield patch
        finally:
            config.reset_runtime_config()
            patch.undo()
            config.clear_runtime_caches()


def record_quant_config_factory(factory_path, patch):
    """Record actual JIT arguments and returned kernels without replacing execution."""
    module_name, factory_name = factory_path.rsplit('.', 1)
    module = importlib.import_module('cann_ops_tilelang.quant.' + module_name)
    factory = getattr(module, factory_name)
    calls = []

    def record(*args, **kwargs):
        kernel = factory(*args, **kwargs)
        cores = kwargs.get('n_cores', kwargs.get('num_cores'))
        calls.append((cores, kwargs['out_config'].clamp_min_value, kernel))
        return kernel

    patch.setattr(module, factory_name, record)
    return calls


def _set_quant_test_vector_cap(patch, cap):
    if cap is None:
        patch.delenv('OPS_TILELANG_NUM_VECTOR_CORES')
    else:
        patch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', str(cap))
    config.clear_runtime_caches()


def _quant_core_update_steps(patch):
    half_ai = max(1, config.get_num_ai_cores() // 2)
    vector_cores = config.get_num_vector_cores()
    return [
        (partial(config.set_num_ai_cores, half_ai), True),
        (partial(config.set_num_ai_cores, half_ai), False),
        (config.reset_runtime_config, True),
        (partial(_set_quant_test_vector_cap, patch, max(1, vector_cores // 2)), True),
        (partial(_set_quant_test_vector_cap, patch, max(1, vector_cores // 3)), True),
        (partial(_set_quant_test_vector_cap, patch, None), True),
    ]


def check_quant_config_updates(check, patch):
    """Exercise one compiled callable; only the configuration API invalidates it."""
    core_steps = _quant_core_update_steps(patch)
    baseline = check(True)
    for update, recapture in core_steps:
        update()
        kernel = check(recapture)
    if kernel is not baseline:
        raise AssertionError('Restoring core settings did not reuse the original JIT kernel.')
    with pytest.raises(ValueError):
        config.set_num_ai_cores(0)
    check(False)
    clamp = config.get_amax_clamp_for_quant('e4m3')
    clamp_steps = (
        (partial(config.set_amax_clamp_for_quant, 'e4m3', clamp * 100), True),
        (partial(config.set_amax_clamp_for_quant, 'e4m3', clamp * 100), False),
        (config.reset_runtime_config, True),
    )
    for update, recapture in clamp_steps:
        update()
        kernel = check(recapture)
    if kernel is not baseline:
        raise AssertionError('Restoring clamp settings did not reuse the original JIT kernel.')
    with pytest.raises(ValueError):
        config.set_amax_clamp_for_quant('e4m3', float('nan'))
    check(False)


def _check_quant_config_capture(events, recapture):
    if bool(events['acl_graph_capture']) is not recapture:
        raise AssertionError(f'Unexpected graph capture after configuration update: {events}')
    if events['acl_graph_replay'] <= 0 or events['fx_run_eagerly']:
        raise AssertionError(f'Expected graph replay without eager fallback: {events}')


def check_quant_config_result(compiled, args, expected, calls, recapture):
    """Check capture, JIT configuration, independent results and read-only inputs."""
    inputs = [value for value in tree_leaves(args) if isinstance(value, torch.Tensor)]
    snapshots = [value.clone() for value in inputs]
    previous_calls = len(calls)
    result, events = call_quant_graph(compiled, *args)
    _check_quant_config_capture(events, recapture)
    if (len(calls) > previous_calls) is not recapture:
        raise AssertionError('The JIT factory was not called exactly when a new capture was expected.')
    cores, clamp, kernel = calls[-1]
    if cores != config.get_num_vector_cores():
        raise AssertionError(f'JIT factory used stale vector core count: {cores}')
    if not math.isclose(clamp, config.get_amax_clamp_for_quant('e4m3'), rel_tol=0.0, abs_tol=0.0):
        raise AssertionError(f'JIT factory used stale clamp: {clamp}')
    outputs, references = tree_leaves(result), tree_leaves(expected)
    if len(outputs) != len(references):
        raise AssertionError('The compiled result has an unexpected output structure.')
    for actual, reference in zip(outputs, references, strict=True):
        assert_equal(actual, reference)
    for value, snapshot in zip(inputs, snapshots, strict=True):
        assert_equal(value, snapshot)
    return kernel
