"""Compression operators for Ascend."""

__all__ = ['compress_and_update_state_decode', 'compress_and_update_state_prefill']

from .compress import compress_and_update_state_decode, compress_and_update_state_prefill
