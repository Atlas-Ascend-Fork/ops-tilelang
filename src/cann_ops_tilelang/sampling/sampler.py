"""Gather logits for the current decoding step on Ascend."""

import torch

from cann_ops_tilelang.config import require_ascend

from .sampler_kernel import _get_logits_cur_by_step_tl_ascend


def get_logits_cur_by_step(logits: torch.Tensor, ftp: torch.Tensor, has_token: torch.Tensor, step: int) -> torch.Tensor:
    require_ascend()
    assert logits.stride(1) == 1 and logits.stride(0) % 16 == 0
    n = ftp.shape[0]
    vocab_size = logits.shape[1]
    out = torch.empty((n, vocab_size), dtype=logits.dtype, device=logits.device)
    kernel = _get_logits_cur_by_step_tl_ascend(vocab_size)
    kernel(logits, ftp, has_token, out, step)
    return out
