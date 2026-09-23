"""Quantization operators for Ascend."""

__all__ = [
    'cast_back',
    'per_block_cast',
    'per_block_cast_lossless',
    'per_block_cast_with_precomputed_sf',
    'per_block_cast_with_sf_only',
    'per_channel_cast',
    'per_token_cast',
    'per_token_cast_and_cast_back',
    'per_token_cast_back',
    'per_token_cast_with_precomputed_sf',
    'per_token_cast_with_sf_only',
    'swiglu_backward',
    'swiglu_forward',
    'swiglu_forward_and_per_token_cast',
    'unpack_from_e2m1fn_x2',
]

from .cast_back import cast_back, per_token_cast_back
from .common import unpack_from_e2m1fn_x2
from .per_block_cast import per_block_cast, per_block_cast_with_precomputed_sf, per_block_cast_with_sf_only
from .per_block_cast_lossless import per_block_cast_lossless
from .per_channel_cast import per_channel_cast
from .per_token_cast import per_token_cast, per_token_cast_with_precomputed_sf, per_token_cast_with_sf_only
from .per_token_cast_and_cast_back import per_token_cast_and_cast_back
from .swiglu_backward import swiglu_backward
from .swiglu_forward import swiglu_forward, swiglu_forward_and_per_token_cast
