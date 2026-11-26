"""Gather logits for the current decoding step on Ascend."""

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend

from .sampler_kernel import _get_logits_cur_by_step_tl_ascend


@torch.library.custom_op('cann_ops_tilelang::get_logits_cur_by_step', mutates_args=())
def get_logits_cur_by_step(logits: torch.Tensor, ftp: torch.Tensor, has_token: torch.Tensor, step: int) -> torch.Tensor:
    """Gather logits for the selected decoding step."""
    return _get_logits_cur_by_step_impl(logits, ftp, has_token, step)


def _validate_get_logits_cur_by_step(logits, ftp, has_token, step):
    assert logits.stride(1) == 1 and logits.stride(0) % 16 == 0


def _get_logits_cur_by_step_impl(
    logits: torch.Tensor, ftp: torch.Tensor, has_token: torch.Tensor, step: int
) -> torch.Tensor:
    require_ascend()
    _validate_get_logits_cur_by_step(logits, ftp, has_token, step)
    n = ftp.shape[0]
    vocab_size = logits.shape[1]
    out = torch.empty((n, vocab_size), dtype=logits.dtype, device=logits.device)
    num_cores = get_num_vector_cores()
    kernel = _get_logits_cur_by_step_tl_ascend(vocab_size, num_cores)
    kernel(logits, ftp, has_token, out, step)
    return out


@get_logits_cur_by_step.register_fake
def _get_logits_cur_by_step_fake(logits, ftp, has_token, step):
    _validate_get_logits_cur_by_step(logits, ftp, has_token, step)
    return torch.empty((ftp.shape[0], logits.shape[1]), dtype=logits.dtype, device=logits.device)
