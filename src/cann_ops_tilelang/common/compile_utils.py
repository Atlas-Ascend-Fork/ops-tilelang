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

"""Typed launch configuration shared by operator adapters."""

import os

import torch

from cann_ops_tilelang.config import get_num_ai_cores


def resolve_vector_cores(tensor: torch.Tensor) -> int:
    # Read public AI-core configuration; avoid torch_npu's logging os.getenv patch.
    index = tensor.device.index
    ai_cores = get_num_ai_cores() if index == torch.npu.current_device() else get_num_ai_cores(index)
    available = ai_cores * 2
    key = 'OPS_TILELANG_NUM_VECTOR_CORES'
    override = os.environ[key] if key in tuple(os.environ) else None
    requested = available if override is None else int(override)
    if not 0 < requested <= available:
        raise ValueError(f'{key} must be in [1, {available}], got {requested}.')
    return requested


def validate_vector_cores(vector_cores: int) -> None:
    if isinstance(vector_cores, bool) or not isinstance(vector_cores, int):
        raise TypeError('vector_cores must be an integer.')
    if vector_cores <= 0:
        raise ValueError('vector_cores must be positive.')
