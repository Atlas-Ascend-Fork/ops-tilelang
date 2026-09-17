"""Pure PyTorch reference for one Engram Sinkhorn normalization step."""

from __future__ import annotations

import torch


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
