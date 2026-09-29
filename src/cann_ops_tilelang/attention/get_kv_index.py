import torch

from cann_ops_tilelang.attention.get_kv_index_kernel import _get_fused_dflash_get_input_tl_ascend
from cann_ops_tilelang.config import require_ascend


def _validate_int32_npu_tensor(tensor: torch.Tensor, name: str, ndim: int | None = None) -> None:
    if tensor.device.type != 'npu':
        raise ValueError(f'{name} must be on an NPU, got {tensor.device}.')
    if tensor.dtype != torch.int32:
        raise TypeError(f'{name} must have dtype torch.int32, got {tensor.dtype}.')
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f'{name} must be {ndim}D, got shape {tuple(tensor.shape)}.')


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
    require_ascend()
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
        swa_block_size, padded_kv_len, dflash_block_size, has_valid_token_mask, has_gate_force_random
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
