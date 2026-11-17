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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cann_ops_tilelang import config
from cann_ops_tilelang.config import (
    get_amax_clamp_for_quant,
    reset_runtime_config,
    set_amax_clamp_for_quant,
)


@pytest.fixture(autouse=True)
def reset_quant_config():
    reset_runtime_config()
    yield
    reset_runtime_config()


@pytest.mark.parametrize('fmt', ['e4m3', 'e2m1'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 0.0, -1.0])
def test_quant_clamp_rejects_invalid_values_without_changing_config(fmt, value):
    previous = get_amax_clamp_for_quant(fmt)
    with pytest.raises(ValueError, match='positive'):
        set_amax_clamp_for_quant(fmt, value)
    assert get_amax_clamp_for_quant(fmt) == previous


@pytest.mark.parametrize('fmt', ['e4m3', 'e2m1'])
@pytest.mark.parametrize('value', [1e-8, 1.0, 1e6])
def test_quant_clamp_accepts_positive_finite_values(fmt, value):
    set_amax_clamp_for_quant(fmt, value)
    assert get_amax_clamp_for_quant(fmt) == value


def test_quant_clamp_rejects_unsupported_format():
    with pytest.raises(ValueError, match='Unsupported quantization format'):
        set_amax_clamp_for_quant('unknown', 1.0)


@pytest.fixture
def core_config_runtime(monkeypatch):
    monkeypatch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
    config.reset_runtime_config()
    monkeypatch.setattr(config, 'require_ascend', lambda: None)
    monkeypatch.setattr(
        torch,
        'npu',
        SimpleNamespace(
            current_device=lambda: 0,
            get_device_properties=lambda index: SimpleNamespace(cube_core_num=28),
        ),
        raising=False,
    )
    reset_compiler = Mock()
    monkeypatch.setattr(torch.compiler, 'reset', reset_compiler)
    return reset_compiler


def test_set_num_ai_cores_invalidates_only_changed_configuration(core_config_runtime):
    assert config.get_num_vector_cores() == 56
    config.set_num_ai_cores(24)
    assert config.get_num_ai_cores() == 24
    assert config.get_num_vector_cores() == 48
    core_config_runtime.assert_called_once_with()
    config.set_num_ai_cores(24)
    core_config_runtime.assert_called_once_with()


@pytest.mark.parametrize('value, error', [(True, TypeError), (1.5, TypeError), (0, ValueError), (29, ValueError)])
def test_invalid_core_configuration_preserves_compiled_cache(core_config_runtime, value, error):
    config.set_num_ai_cores(24)
    assert config.get_num_vector_cores() == 48
    core_config_runtime.reset_mock()
    with pytest.raises(error):
        config.set_num_ai_cores(value)
    assert config.get_num_ai_cores() == 24
    assert config.get_num_vector_cores() == 48
    core_config_runtime.assert_not_called()


def test_reset_runtime_config_invalidates_core_override(core_config_runtime):
    config.set_num_ai_cores(24)
    core_config_runtime.reset_mock()
    config.reset_runtime_config()
    assert config.get_num_vector_cores() == 56
    core_config_runtime.assert_called_once_with()
    config.reset_runtime_config()
    core_config_runtime.assert_called_once_with()


def test_clear_runtime_caches_invalidates_vector_core_environment(core_config_runtime, monkeypatch):
    assert config.get_num_vector_cores() == 56
    config.clear_runtime_caches()
    core_config_runtime.assert_not_called()
    for cap, expected, reset_count in [('48', 48, 1), ('48', 48, 1), ('44', 44, 2), (None, 56, 3)]:
        if cap is None:
            monkeypatch.delenv('OPS_TILELANG_NUM_VECTOR_CORES')
        else:
            monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', cap)
        config.clear_runtime_caches()
        assert config.get_num_vector_cores() == expected
        assert core_config_runtime.call_count == reset_count
