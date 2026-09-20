import torch


def normalize_weight(topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each token's top-k weights so they sum to one.

    Args:
        topk_weights: Float32 tensor of shape ``(num_tokens, num_topk)``.

    Returns:
        Tuple of ``(denominator, normalized_weights)`` where *denominator* has
        shape ``(num_tokens,)`` and *normalized_weights* has the same shape as
        the input.
    """
    denominator = torch.sum(topk_weights, dim=1)
    normalized_weights = topk_weights / denominator.unsqueeze(1)
    return denominator, normalized_weights
