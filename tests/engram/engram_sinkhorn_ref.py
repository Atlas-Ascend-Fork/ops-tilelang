"""Pure PyTorch references for the Engram Sinkhorn operator family."""

from __future__ import annotations

import torch


def engram_sinkhorn_momentum_update_ref(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta1: float,
    nesterov: bool = True,
) -> torch.Tensor:
    """Update momentum in place and return the FP32 Sinkhorn input."""

    grad_fp32 = grad.float()
    momentum_fp32 = momentum_buffer.float()
    momentum_fp32.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
    momentum_buffer.copy_(momentum_fp32)
    if nesterov:
        return torch.lerp(momentum_fp32, grad_fp32, 1 - beta1)
    return momentum_fp32


def engram_sinkhorn_step_ref(
    matrix: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update ``row_scale`` and return row norms and column squared sums."""

    iterate = row_scale.unsqueeze(1) * matrix * col_scale
    row_norm = torch.linalg.vector_norm(iterate, dim=1)
    row_scale.div_(row_norm + eps)
    normalized = row_scale.unsqueeze(1) * matrix * col_scale
    return row_norm, normalized.square().sum(dim=0)


def engram_sinkhorn_finalize_ref(
    matrix: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    eps: float,
    alpha: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the final row-normalized Sinkhorn matrix."""

    iterate = row_scale.unsqueeze(1) * matrix * col_scale
    row_norm = torch.linalg.vector_norm(iterate, dim=1)
    normalized_row_scale = row_scale / (row_norm + eps)
    result = alpha * normalized_row_scale.unsqueeze(1) * matrix * col_scale
    if out is None:
        return result
    out.copy_(result)
    return out
