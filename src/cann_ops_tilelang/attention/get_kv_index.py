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

from cann_ops_tilelang.attention.get_kv_index_kernel import _get_fused_dflash_get_input_tl_ascend
from cann_ops_tilelang.config import get_num_vector_cores, require_ascend


def _validate_int32_npu_tensor(tensor: torch.Tensor, name: str, ndim: int | None = None) -> None:
    if tensor.device.type != 'npu':
        raise ValueError(f'{name} must be on an NPU, got {tensor.device}.')
    if tensor.dtype != torch.int32:
        raise TypeError(f'{name} must have dtype torch.int32, got {tensor.dtype}.')
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f'{name} must be {ndim}D, got shape {tuple(tensor.shape)}.')


def _validate_dflash_inputs(
    slot_mapping_swa,
    positions,
    ftp,
    accepted_pos_real,
    swa_block_size,
    dflash_block_size,
    window_size,
) -> int:
    for name, tensor in (
        ('slot_mapping_swa', slot_mapping_swa),
        ('positions', positions),
        ('ftp', ftp),
        ('accepted_pos_real', accepted_pos_real),
    ):
        _validate_int32_npu_tensor(tensor, name, ndim=1)
    devices = {tensor.device for tensor in (slot_mapping_swa, positions, ftp, accepted_pos_real)}
    if len(devices) != 1:
        raise ValueError('all inputs must be on the same NPU device.')
    if not 0 < dflash_block_size <= 32:
        raise ValueError(f'dflash_block_size must be in [1, 32], got {dflash_block_size}.')
    if swa_block_size <= 0 or window_size <= 0:
        raise ValueError('swa_block_size and window_size must be positive.')
    real_cnt = accepted_pos_real.shape[0]
    if real_cnt > ftp.numel():
        raise ValueError('accepted_pos_real cannot contain more entries than ftp.')
    return real_cnt


def _get_fused_dflash_get_input_impl(
    slot_mapping_swa: torch.Tensor,
    positions: torch.Tensor,
    ftp: torch.Tensor,
    accepted_pos_real: torch.Tensor,
    swa_block_size: int,
    dflash_block_size: int,
    window_size: int,
    has_valid_token_mask: bool,
    has_gate_force_random: bool,
    input_ids_dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    require_ascend()
    real_cnt = _validate_dflash_inputs(
        slot_mapping_swa, positions, ftp, accepted_pos_real, swa_block_size, dflash_block_size, window_size
    )
    kv_len = window_size + dflash_block_size
    padded_kv_len = (kv_len + 63) // 64 * 64
    device = accepted_pos_real.device
    newx_slot_mapping_out = torch.empty((real_cnt, dflash_block_size), dtype=torch.int32, device=device)
    positions_out = torch.empty((real_cnt, dflash_block_size), dtype=torch.int32, device=device)
    input_ids_out = torch.empty((real_cnt, dflash_block_size), dtype=torch.int32, device=device)
    valid_token_mask_out = (
        torch.empty((real_cnt, dflash_block_size), dtype=torch.bool, device=device) if has_valid_token_mask else None
    )
    gate_force_random_out = (
        torch.empty((real_cnt, dflash_block_size), dtype=torch.bool, device=device) if has_gate_force_random else None
    )
    kv_index_out = torch.empty((real_cnt * dflash_block_size, padded_kv_len), dtype=torch.int32, device=device)
    kernel = _get_fused_dflash_get_input_tl_ascend(
        swa_block_size,
        padded_kv_len,
        dflash_block_size,
        has_valid_token_mask,
        has_gate_force_random,
        get_num_vector_cores(),
    )
    kernel(
        slot_mapping_swa,
        positions,
        ftp[:real_cnt],
        accepted_pos_real,
        newx_slot_mapping_out,
        positions_out,
        input_ids_out,
        valid_token_mask_out,
        gate_force_random_out,
        kv_index_out,
        kv_len,
        window_size,
    )
    valid_token_mask = valid_token_mask_out.view(-1) if has_valid_token_mask else None
    gate_force_random = gate_force_random_out.view(-1) if has_gate_force_random else None
    return (
        newx_slot_mapping_out.view(-1),
        positions_out.view(-1),
        input_ids_out.to(input_ids_dtype).view(-1),
        valid_token_mask,
        gate_force_random,
        kv_index_out,
    )


@torch.library.custom_op(
    'cann_ops_tilelang::get_fused_dflash_get_input',
    mutates_args=(),
    schema='(Tensor slot_mapping_swa, Tensor positions, Tensor ftp, Tensor accepted_pos_real, '
    'int swa_block_size, int dflash_block_size, int window_size, bool has_valid_token_mask=False, '
    'bool has_gate_force_random=False, ScalarType input_ids_dtype=3) '
    '-> (Tensor, Tensor, Tensor, Tensor?, Tensor?, Tensor)',
)
def get_fused_dflash_get_input(
    slot_mapping_swa: torch.Tensor,
    positions: torch.Tensor,
    ftp: torch.Tensor,
    accepted_pos_real: torch.Tensor,
    swa_block_size: int,
    dflash_block_size: int,
    window_size: int,
    has_valid_token_mask: bool = False,
    has_gate_force_random: bool = False,
    input_ids_dtype: torch.dtype = torch.int32,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    """Return slot mapping, positions, IDs, optional masks, and KV indices."""
    return _get_fused_dflash_get_input_impl(
        slot_mapping_swa,
        positions,
        ftp,
        accepted_pos_real,
        swa_block_size,
        dflash_block_size,
        window_size,
        has_valid_token_mask,
        has_gate_force_random,
        input_ids_dtype,
    )


@get_fused_dflash_get_input.register_fake
def _fused_dflash_get_input_fake(
    slot_mapping_swa,
    positions,
    ftp,
    accepted_pos_real,
    swa_block_size,
    dflash_block_size,
    window_size,
    has_valid_token_mask=False,
    has_gate_force_random=False,
    input_ids_dtype=torch.int32,
):
    real_cnt = _validate_dflash_inputs(
        slot_mapping_swa, positions, ftp, accepted_pos_real, swa_block_size, dflash_block_size, window_size
    )
    total_q = real_cnt * dflash_block_size
    padded_kv_len = (window_size + dflash_block_size + 63) // 64 * 64
    device = accepted_pos_real.device
    return (
        torch.empty(total_q, dtype=torch.int32, device=device),
        torch.empty(total_q, dtype=torch.int32, device=device),
        torch.empty(total_q, dtype=input_ids_dtype, device=device),
        torch.empty(total_q, dtype=torch.bool, device=device) if has_valid_token_mask else None,
        torch.empty(total_q, dtype=torch.bool, device=device) if has_gate_force_random else None,
        torch.empty((total_q, padded_kv_len), dtype=torch.int32, device=device),
    )
