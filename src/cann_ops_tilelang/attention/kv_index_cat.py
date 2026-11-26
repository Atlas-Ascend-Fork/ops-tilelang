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

import torch

from cann_ops_tilelang.attention.kv_index_cat_kernel import _kv_index_cat_tl_ascend
from cann_ops_tilelang.config import get_num_vector_cores, require_ascend


def _validate_kv_index_cat_inputs(kv_index, kv_ca_index) -> tuple[int, int, int]:
    assert kv_index.ndim == 2, 'kv_index must be a 2D tensor'
    assert kv_ca_index.ndim == 2, 'kv_ca_index must be a 2D tensor'
    total_q, window_size = kv_index.shape
    _, topk_tokens = kv_ca_index.shape
    assert total_q > 0, 'total_q must be positive'
    assert window_size > 0, 'window_size must be positive'
    assert topk_tokens > 0, 'topk_tokens must be positive'
    assert kv_ca_index.shape[0] == total_q, 'both inputs must have the same total_q'
    assert kv_index.is_contiguous(), 'kv_index must be contiguous'
    assert kv_ca_index.is_contiguous(), 'kv_ca_index must be contiguous'
    assert kv_index.dtype == torch.int32, 'kv_index must use int32'
    assert kv_ca_index.dtype == torch.int32, 'kv_ca_index must use int32'
    assert kv_index.device == kv_ca_index.device, 'both inputs must be on the same device'
    if kv_index.device.type != 'npu':
        raise ValueError(f'input tensors must be on an NPU, got {kv_index.device}.')
    assert kv_index.stride(0) % 4 == 0, 'window_size must be aligned to 4 elements'
    return total_q, window_size, topk_tokens


def _kv_index_cat_impl(
    kv_index: torch.Tensor, kv_ca_index: torch.Tensor, total_k_swa: int, total_k: int
) -> torch.Tensor:
    require_ascend()
    total_q, window_size, topk_tokens = _validate_kv_index_cat_inputs(kv_index, kv_ca_index)
    out_shape = window_size + topk_tokens
    out_stride = (out_shape + 3) // 4 * 4
    out = torch.empty((total_q, out_stride), dtype=kv_index.dtype, device=kv_index.device)[:, :out_shape]
    kernel = _kv_index_cat_tl_ascend('int32', window_size, topk_tokens, out_stride, get_num_vector_cores())
    kernel(
        kv_index,
        kv_ca_index,
        out[:, :window_size],
        out[:, window_size : window_size + topk_tokens],
        total_k_swa,
        total_k,
    )
    return out


@torch.library.custom_op('cann_ops_tilelang::kv_index_cat', mutates_args=())
def kv_index_cat(kv_index: torch.Tensor, kv_ca_index: torch.Tensor, total_k_swa: int, total_k: int) -> torch.Tensor:
    """
    Args:
        kv_index: [total_q, window_size], int32
        kv_ca_index: [total_q, topk_tokens], int32
        total_k_swa: scalar int32
        total_k: scalar int32

    Returns:
        out: [total_q, window_size + topk_tokens], int32
    """
    return _kv_index_cat_impl(
        kv_index,
        kv_ca_index,
        total_k_swa,
        total_k,
    )


@kv_index_cat.register_fake
def _kv_index_cat_fake(kv_index, kv_ca_index, total_k_swa, total_k):
    total_q, window_size, topk_tokens = _validate_kv_index_cat_inputs(kv_index, kv_ca_index)
    out_shape = window_size + topk_tokens
    out_stride = (out_shape + 3) // 4 * 4
    return torch.empty((total_q, out_stride), dtype=kv_index.dtype, device=kv_index.device)[:, :out_shape]
