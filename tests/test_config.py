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

import pytest

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
