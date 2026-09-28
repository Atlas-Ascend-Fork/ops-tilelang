"""Sampling and speculative-decoding operators for Ascend."""

__all__ = [
    'apply_token_bitmask_inplace',
    'batch_find_value_by_index',
    'compute_confidence_error',
    'compute_seqlens_q_by_confidence_rates',
    'confidence_head_fused',
    'dflash_input_store',
    'find_value_by_index',
    'fused_sample',
    'get_accept_result',
    'get_logits_cur_by_step',
    'get_prompt_logprobs_tensor',
    'get_sum_smaller_probs',
    'log_softmax',
    'random_sampling',
    'sampled_tokens_update',
    'scatter_verified',
    'softmax',
    'softmax_and_logsoftmax',
    'update_verify_states',
]

from .dflash import confidence_head_fused, dflash_input_store
from .fused_sample import fused_sample
from .prompt_logprobs import get_prompt_logprobs_tensor
from .random_sampling import random_sampling
from .sampler import (
    batch_find_value_by_index,
    compute_confidence_error,
    find_value_by_index,
    get_accept_result,
    get_logits_cur_by_step,
    sampled_tokens_update,
    scatter_verified,
    update_verify_states,
)
from .schedule_mtp_varlen import compute_seqlens_q_by_confidence_rates
from .softmax import log_softmax, softmax, softmax_and_logsoftmax
from .structure_output import apply_token_bitmask_inplace
from .sum_smaller_probs import get_sum_smaller_probs
