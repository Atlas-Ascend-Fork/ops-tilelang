import torch

from cann_ops_tilelang.attention.state_cache_kernel import _load_state_to_cache_tl_ascend, _select_state_slice_tl_ascend
from cann_ops_tilelang.config import require_ascend


def select_state_slice(state_tensor: torch.Tensor, state_cache: torch.Tensor, last_hit_position: int) -> None:
    require_ascend()
    _validate_state_tensors(state_tensor, state_cache, last_hit_position)
    num_layers, block_size_kvstore, feature_dim = state_cache.shape
    _, block_size_inference, _ = state_tensor.shape
    contiguous_tokens = (
        feature_dim > 1 and state_tensor.stride(1) == feature_dim and (state_cache.stride(1) == feature_dim)
    )
    kernel = _select_state_slice_tl_ascend(
        block_size_kvstore, block_size_inference, num_layers, feature_dim, contiguous_tokens
    )
    return kernel(state_tensor, state_cache, last_hit_position)


def load_state_to_cache(state_tensor: torch.Tensor, state_cache: torch.Tensor, last_hit_position: int) -> None:
    require_ascend()
    _validate_state_tensors(state_tensor, state_cache, last_hit_position)
    return _load_state_to_cache_impl(state_tensor, state_cache, last_hit_position)


def _validate_state_tensors(
    state_tensor: torch.Tensor,
    state_cache: torch.Tensor,
    last_hit_position: int,
) -> None:
    if state_tensor.ndim != 3 or state_cache.ndim != 3:
        raise ValueError('state_tensor and state_cache must both be 3D tensors')
    if state_tensor.dtype != torch.uint8 or state_cache.dtype != torch.uint8:
        raise TypeError('state_tensor and state_cache must both have dtype torch.uint8')
    if state_tensor.device != state_cache.device:
        raise ValueError('state_tensor and state_cache must be on the same device')
    if state_tensor.device.type != 'npu':
        raise ValueError(f'state tensors must be on an NPU, got {state_tensor.device}.')
    num_layers, block_size_kvstore, feature_dim = state_cache.shape
    tensor_num_layers, block_size_inference, tensor_feature_dim = state_tensor.shape
    if num_layers <= 0 or block_size_kvstore <= 0 or block_size_inference <= 0 or (feature_dim <= 0):
        raise ValueError('all state tensor dimensions must be positive')
    if tensor_num_layers != num_layers or tensor_feature_dim != feature_dim:
        raise ValueError('state_tensor and state_cache must have matching layer and feature dimensions')
    for name, tensor in (('state_tensor', state_tensor), ('state_cache', state_cache)):
        if tensor.stride(2) != 1:
            raise ValueError(f'{name} must be contiguous in the feature dimension')
        if tensor.stride(1) < feature_dim or tensor.stride(0) < tensor.shape[1] * tensor.stride(1):
            raise ValueError(f'{name} must have a non-overlapping row-major layout')
    if not isinstance(last_hit_position, int) or not 0 <= last_hit_position <= torch.iinfo(torch.int32).max:
        raise ValueError('last_hit_position must be a non-negative int32 value')
    return None


def _load_state_to_cache_impl(
    state_tensor: torch.Tensor,
    state_cache: torch.Tensor,
    last_hit_position: int,
) -> None:
    num_layers, block_size_kvstore, feature_dim = state_cache.shape
    _, block_size_inference, _ = state_tensor.shape
    contiguous_tokens = (
        feature_dim > 1 and state_tensor.stride(1) == feature_dim and state_cache.stride(1) == feature_dim
    )
    kernel = _load_state_to_cache_tl_ascend(
        block_size_kvstore, block_size_inference, num_layers, feature_dim, contiguous_tokens
    )
    return kernel(state_tensor, state_cache, last_hit_position)
