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

"""PR77 configuration notification with directly registered MHC public APIs."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import cann_ops_tilelang as ops
from cann_ops_tilelang import config
from cann_ops_tilelang.mhc import expand, head_compute_mix
from tests.mhc.test_mhc_graph_utils import (
    check_round,
    clone_args,
    compile_case,
    make_case,
    registered_case,
)
from tests.mhc.test_mhc_graph_utils import isolate_graphs as isolate_graphs

pytestmark = pytest.mark.npu
CONFIG_NAMES = ('expand_to_mhc_forward', 'expand_to_mhc_backward', 'mhc_head_compute_mix_backward')
ALL_NAMES = CONFIG_NAMES + ('mhc_head_compute_mix_forward',)


@pytest.mark.parametrize('name', ALL_NAMES)
def test_direct_registration_and_metadata_only_fake(name, monkeypatch):
    case = make_case(name)
    op, args, kwargs = registered_case(case)
    public = getattr(ops, name)
    module = head_compute_mix if name.startswith('mhc_head') else expand
    assert isinstance(public, torch.library.CustomOpDef)
    assert public is getattr(module, name)

    def forbidden(*args, **kwargs):
        raise AssertionError('Fake queried hardware/configuration unrelated to output metadata.')

    monkeypatch.setattr(module, 'resolve_vector_cores', forbidden, raising=False)
    with FakeTensorMode() as mode:
        values = tuple(mode.from_tensor(v) if isinstance(v, torch.Tensor) else v for v in args)
        assert op(*values, **kwargs) is None


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_head_runtime_core_limit(compiled, empty, monkeypatch):
    monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', '2')
    config.clear_runtime_caches()
    try:
        case = make_case('mhc_head_compute_mix_backward', n=0 if empty else 7, cores=4)
        fn = compile_case(case) if compiled else ops.mhc_head_compute_mix_backward
        before = clone_args(case.args)
        with pytest.raises((ValueError, RuntimeError)), torch.inference_mode():
            fn(*case.args)
        for actual, expected in zip(case.args, before, strict=True):
            if isinstance(actual, torch.Tensor):
                assert torch.equal(actual, expected)
    finally:
        monkeypatch.undo()
        config.clear_runtime_caches()


@pytest.mark.parametrize('name', CONFIG_NAMES[:2])
@pytest.mark.parametrize('compiled', [False, True])
def test_empty_expand_ignores_invalid_configuration(name, compiled, monkeypatch):
    monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', 'invalid')
    config.clear_runtime_caches()
    try:
        case = make_case(name, n=0)
        fn = compile_case(case) if compiled else getattr(ops, name)
        check_round(case, fn)
    finally:
        monkeypatch.undo()
        config.clear_runtime_caches()
