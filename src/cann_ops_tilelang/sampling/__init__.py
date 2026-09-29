"""Sampling and speculative-decoding operators for Ascend."""

__all__ = [
    'apply_token_bitmask_inplace',
    'compute_seqlens_q_by_confidence_rates',
    'fused_sample',
    'get_logits_cur_by_step',
    'get_prompt_logprobs_tensor',
    'get_sum_smaller_probs',
    'log_softmax',
    'random_sampling',
    'softmax',
    'softmax_and_logsoftmax',
]

from .fused_sample import fused_sample
from .prompt_logprobs import get_prompt_logprobs_tensor
from .random_sampling import random_sampling
from .sampler import get_logits_cur_by_step
from .schedule_mtp_varlen import compute_seqlens_q_by_confidence_rates
from .softmax import log_softmax, softmax, softmax_and_logsoftmax
from .structure_output import apply_token_bitmask_inplace
from .sum_smaller_probs import get_sum_smaller_probs
