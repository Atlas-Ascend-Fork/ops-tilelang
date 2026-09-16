"""Pure PyTorch reference for the Engram gate operator."""

from __future__ import annotations

import torch


def engram_gate_ref(
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    clamp_value: float,
    eps: float,
    save_for_backward: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Engram gate with eager PyTorch operations.

    This implementation is intentionally device-agnostic and supports
    autograd, making it the numerical oracle for both forward and backward
    TileLang kernels.
    """

    num_tokens, hc_mult, hidden_size = hidden_states.shape
    scalar = hidden_size**-0.5
    k, v = kv.split([hc_mult * hidden_size, hidden_size], dim=-1)
    k = k.view(num_tokens, hc_mult, hidden_size)

    x = hidden_states.float()
    k_float = k.float()
    weight_hidden_float = weight_hidden.float().unsqueeze(0)
    weight_embed_float = weight_embed.float().unsqueeze(0)

    rstd_x = torch.rsqrt(x.square().mean(-1) + eps)
    rstd_k = torch.rsqrt(k_float.square().mean(-1) + eps)
    raw_dot = torch.einsum(
        '...d,...d->...',
        x * weight_hidden_float,
        k_float * weight_embed_float,
    )
    normalized_dot = raw_dot * rstd_x * rstd_k * scalar
    signed_sqrt = normalized_dot.abs().clamp_min(clamp_value).sqrt() * normalized_dot.sign()
    gate_score = signed_sqrt.sigmoid()
    output = (x + gate_score.unsqueeze(-1) * v.unsqueeze(-2)).bfloat16()

    if save_for_backward:
        return output, raw_dot, gate_score, rstd_x, rstd_k
    return output
