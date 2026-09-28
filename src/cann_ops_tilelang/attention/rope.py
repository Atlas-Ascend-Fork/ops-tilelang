"""Public Ascend interface for rotary embeddings."""

from __future__ import annotations

import torch

from cann_ops_tilelang.config import require_ascend

from .rope_kernel import _rotary_embedding_tl_ascend


def rotary_embedding(
    positions: torch.Tensor,  # [seqlen]
    query: torch.Tensor,  # [seqlen, num_heads * head_size] or [seqlen, num_heads, head_size]
    key: torch.Tensor | None,  # [seqlen, num_kv_heads * head_size] or [seqlen, num_kv_heads, head_size]
    query_out: torch.Tensor | None,  # The same shape as `query`
    key_out: torch.Tensor | None,  # The same shape as `key`
    head_size: int,
    cos_sin_cache: torch.Tensor,  # [seqlen_ro, rot_dim]
    is_neox: bool,
    conjugate: bool = False,
) -> None:
    require_ascend()
    assert positions.ndim == 1
    assert query.ndim == 2 or query.ndim == 3
    if key is not None:
        assert key.ndim == 2 or key.ndim == 3
    if query_out is not None:
        assert query_out.shape == query.shape
    if key_out is not None:
        assert key is not None and key_out.shape == key.shape
    assert cos_sin_cache.ndim == 2

    assert head_size > 0
    assert positions.is_contiguous(), 'positions must be contiguous'
    assert positions.dtype in (torch.int32, torch.int64), 'positions must use int32 or int64'
    assert query.dtype in (torch.bfloat16, torch.float32), 'query must use bfloat16 or float32'
    assert cos_sin_cache.dtype == torch.float32, 'cos_sin_cache must use float32'
    assert cos_sin_cache.stride(-1) == 1, 'cos_sin_cache must be contiguous in its last dimension'
    assert positions.device == query.device == cos_sin_cache.device, 'inputs must be on the same device'

    seqlen = positions.shape[0]
    _, rot_dim = cos_sin_cache.shape
    assert rot_dim > 0 and rot_dim % 2 == 0, 'rotary_dim must be a positive even number'
    assert rot_dim <= head_size, 'rotary_dim must not exceed head_size'

    assert query.shape[0] == seqlen
    if key is not None:
        assert key.shape[0] == seqlen

    if query.ndim == 2:
        assert query.shape[1] % head_size == 0
        num_heads = query.shape[1] // head_size
        query = query.view(seqlen, num_heads, head_size)
        if query_out is not None:
            query_out = query_out.view(seqlen, num_heads, head_size)
    else:
        assert query.shape[2] == head_size
        num_heads = query.shape[1]
    if key is not None:
        if key.ndim == 2:
            assert key.shape[1] % head_size == 0
            num_kv_heads = key.shape[1] // head_size
            key = key.view(seqlen, num_kv_heads, head_size)
            if key_out is not None:
                key_out = key_out.view(seqlen, num_kv_heads, head_size)
        else:
            assert key.shape[2] == head_size
            num_kv_heads = key.shape[1]

    positions_dtype = 'int32'
    if positions is not None:
        if positions.dtype == torch.int32:
            positions_dtype = 'int32'
        elif positions.dtype == torch.int64:
            positions_dtype = 'int64'
        else:
            raise AssertionError('Unsupported dtype')

    dtype_str = ''
    if query.dtype == torch.bfloat16:
        dtype_str = 'bfloat16'
    elif query.dtype == torch.float:
        dtype_str = 'float32'
    else:
        raise AssertionError('Unsupported dtype')
    if key is not None:
        assert key.dtype == query.dtype and key.device == query.device
    if query_out is not None:
        assert query_out.dtype == query.dtype, 'query_out must have the same dtype as query'
        assert query_out.device == query.device
    if key_out is not None:
        assert key_out.dtype == key.dtype and key_out.device == key.device
    for tensor in (query, key, query_out, key_out):
        assert tensor is None or tensor.stride(-1) == 1

    assert query.device.type == 'npu', 'inputs must be NPU tensors'

    cos = cos_sin_cache[:, : rot_dim // 2]
    sin = cos_sin_cache[:, rot_dim // 2 :]
    assert cos.stride(0) == sin.stride(0)
    query_rope_kernel = _rotary_embedding_tl_ascend(
        dtype_str,
        positions_dtype,
        num_heads,
        head_size,
        rot_dim,
        True,
        not is_neox,
        conjugate,
        cos_sin_cache.stride(0),
        query.stride(0),
        query.stride(1),
        query.stride(0) if query_out is None else query_out.stride(0),
        query.stride(1) if query_out is None else query_out.stride(1),
    )
    query_rope_kernel(
        query,
        cos_sin_cache,
        positions,
        query if query_out is None else query_out,
        0,
    )

    if key is not None:
        key_rope_kernel = _rotary_embedding_tl_ascend(
            dtype_str,
            positions_dtype,
            num_kv_heads,
            head_size,
            rot_dim,
            True,
            not is_neox,
            conjugate,
            cos_sin_cache.stride(0),
            key.stride(0),
            key.stride(1),
            key.stride(0) if key_out is None else key_out.stride(0),
            key.stride(1) if key_out is None else key_out.stride(1),
        )
        key_rope_kernel(
            key,
            cos_sin_cache,
            positions,
            key if key_out is None else key_out,
            0,
        )
