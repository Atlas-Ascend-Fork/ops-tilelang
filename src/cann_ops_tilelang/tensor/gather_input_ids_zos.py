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

"""Public interface for gathering ZOS input token IDs."""

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .gather_input_ids_zos_kernel import get_gather_input_ids_zos_kernel


def _validate_max_seqlen_q(max_seqlen_q: int) -> None:
    if not isinstance(max_seqlen_q, int) or isinstance(max_seqlen_q, bool):
        raise TypeError('max_seqlen_q must be an integer.')
    if max_seqlen_q <= 0 or max_seqlen_q > 128:
        raise ValueError(f'max_seqlen_q must be in [1, 128], got {max_seqlen_q}.')


def _validate_inputs(tensors: tuple[torch.Tensor, ...], *, max_seqlen_q: int) -> None:
    fresh_tokens, prev_sampled_token_ids, prev_predict_token_ids = tensors[:3]
    remap, seqlens_q, cu_seqlens_q = tensors[3:6]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('all inputs and outputs must be torch.Tensor instances.')
    _validate_max_seqlen_q(max_seqlen_q)
    if any(tensor.device.type != 'npu' for tensor in tensors):
        raise ValueError('all inputs and outputs must be NPU tensors.')
    if any(tensor.device != fresh_tokens.device for tensor in tensors):
        raise ValueError('all inputs and outputs must be on the same device.')
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise TypeError('all inputs and outputs must use torch.int32.')
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError('all inputs and outputs must be contiguous.')

    expected_ranks = (2, 1, 2, 1, 1, 1, 1)
    if any(tensor.ndim != rank for tensor, rank in zip(tensors, expected_ranks, strict=True)):
        raise ValueError('input and output tensor ranks do not match the operator contract.')

    if fresh_tokens.shape[1] != max_seqlen_q:
        raise ValueError(f'fresh_tokens must have shape (n_seq, {max_seqlen_q}).')
    n_seq = fresh_tokens.shape[0]
    prev_n_seq = prev_sampled_token_ids.shape[0]
    nextn = max_seqlen_q - 1
    if prev_predict_token_ids.shape != (prev_n_seq, nextn):
        raise ValueError(f'prev_predict_token_ids must have shape ({prev_n_seq}, {nextn}).')
    if remap.shape != (n_seq,) or seqlens_q.shape != (n_seq,) or cu_seqlens_q.shape != (n_seq,):
        raise ValueError('remap, seqlens_q, and cu_seqlens_q must have shape (n_seq,).')


def _gather_input_ids_zos_impl(*tensors: torch.Tensor, max_seqlen_q: int) -> None:
    _validate_inputs(tensors, max_seqlen_q=max_seqlen_q)
    require_ascend()
    if tensors[0].shape[0] == 0 or tensors[-1].numel() == 0:
        return
    kernel = get_gather_input_ids_zos_kernel(max_seqlen_q)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(*tensors)


@torch.library.custom_op('cann_ops_tilelang::gather_input_ids_zos', mutates_args=('out_input_ids',))
def gather_input_ids_zos(
    fresh_tokens: torch.Tensor,
    prev_sampled_token_ids: torch.Tensor,
    prev_predict_token_ids: torch.Tensor,
    remap: torch.Tensor,
    seqlens_q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    out_input_ids: torch.Tensor,
    *,
    max_seqlen_q: int,
) -> None:
    """Gather ZOS input IDs into the required output buffer and return None.

    Only ``out_input_ids`` is written. All tensors must be contiguous int32
    tensors on the same NPU. Remap indices and cumulative sequence lengths must
    describe valid accesses; validation does not read device data. Output
    elements outside the described queries retain their previous value.

    Supports fixed-shape ``npugraph_ex`` inference with ``fullgraph=True`` and
    ``dynamic=False``. Fake execution validates metadata. TileLang JIT and
    launch run in the real implementation. No autograd formula is registered.
    """
    return _gather_input_ids_zos_impl(
        fresh_tokens,
        prev_sampled_token_ids,
        prev_predict_token_ids,
        remap,
        seqlens_q,
        cu_seqlens_q,
        out_input_ids,
        max_seqlen_q=max_seqlen_q,
    )


@gather_input_ids_zos.register_fake
def _gather_input_ids_zos_fake(
    fresh_tokens: torch.Tensor,
    prev_sampled_token_ids: torch.Tensor,
    prev_predict_token_ids: torch.Tensor,
    remap: torch.Tensor,
    seqlens_q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    out_input_ids: torch.Tensor,
    *,
    max_seqlen_q: int,
) -> None:
    _validate_inputs(
        (
            fresh_tokens,
            prev_sampled_token_ids,
            prev_predict_token_ids,
            remap,
            seqlens_q,
            cu_seqlens_q,
            out_input_ids,
        ),
        max_seqlen_q=max_seqlen_q,
    )
    return None


def gather_input_ids_zos_ref(
    fresh_tokens: torch.Tensor,
    prev_sampled_token_ids: torch.Tensor,
    prev_predict_token_ids: torch.Tensor,
    remap: torch.Tensor,
    seqlens_q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    out_input_ids: torch.Tensor,
    *,
    max_seqlen_q: int,
) -> None:
    """PyTorch reference for ZOS input token gathering."""
    n_seq = fresh_tokens.shape[0]
    token_idx = torch.arange(max_seqlen_q, device=fresh_tokens.device, dtype=torch.int32)
    valid_mask = token_idx.unsqueeze(0) < seqlens_q.unsqueeze(1)

    q_base = torch.zeros(n_seq, device=fresh_tokens.device, dtype=torch.int32)
    if n_seq > 1:
        q_base[1:] = cu_seqlens_q[:-1]
    out_indices = q_base.unsqueeze(1) + token_idx.unsqueeze(0)

    is_old = (remap >= 0).unsqueeze(1)
    remap_clamped = remap.clamp(min=0)
    sampled_values = prev_sampled_token_ids[remap_clamped.long()]
    previous_idx = (token_idx - 1).clamp(min=0)
    predicted_values = prev_predict_token_ids[
        remap_clamped.long().unsqueeze(1).expand(-1, max_seqlen_q),
        previous_idx.unsqueeze(0).long().expand(n_seq, -1),
    ]

    is_first = (token_idx == 0).unsqueeze(0)
    old_values = torch.where(is_first, sampled_values.unsqueeze(1), predicted_values)
    values = torch.where(is_old, old_values, fresh_tokens)

    flat_indices = out_indices.reshape(-1).long()
    flat_valid = valid_mask.reshape(-1)
    out_input_ids.scatter_(0, flat_indices[flat_valid], values.reshape(-1)[flat_valid])
