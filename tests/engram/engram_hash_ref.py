import torch


def make_offsets(vocab_sizes: torch.Tensor) -> torch.Tensor:
    """Compute exclusive prefix-sum offsets from vocab_sizes.

    Args:
        vocab_sizes: Per-layer per-ngram embedding table sizes of shape
            (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.

    Returns:
        Offsets of shape (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """
    num_ngram_layers = vocab_sizes.shape[0]
    offsets_list = []
    for layer_idx in range(num_ngram_layers):
        flat = vocab_sizes[layer_idx].view(-1)
        prefix = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=flat.device),
                flat[:-1].cumsum(0, dtype=torch.int32),
            ]
        )
        offsets_list.append(prefix)
    return torch.stack(offsets_list, dim=0)


def engram_hash_ref(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of engram hash.

    Args:
        ngram_token_ids: N-gram token IDs of shape (num_tokens, max_ngram_size), int32.
        multipliers: Per-layer hash multipliers of shape (num_ngram_layers, max_ngram_size), int64.
        vocab_sizes: Per-layer per-ngram embedding table sizes of shape
            (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.
        offsets: Per-layer embedding table offsets of shape
            (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.

    Returns:
        Embedding indices of shape
            (num_ngram_layers, num_tokens, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """

    num_ngram_layers = multipliers.shape[0]
    max_ngram_size = multipliers.shape[1]

    prod = ngram_token_ids.to(torch.int64).unsqueeze(0) * multipliers.unsqueeze(1)

    ans = [[] for _ in range(num_ngram_layers)]
    hashes = prod[:, :, 0].clone()
    for i in range(1, max_ngram_size):
        hashes.bitwise_xor_(prod[:, :, i])
        for layer_idx in range(num_ngram_layers):
            ans[layer_idx].append(
                (hashes[layer_idx].unsqueeze(-1) % vocab_sizes[layer_idx, i - 1].to(torch.int64).unsqueeze(0)).to(
                    torch.int32
                )
            )

    for layer_idx in range(num_ngram_layers):
        ans[layer_idx] = torch.cat(ans[layer_idx], dim=-1)

    output = torch.stack(ans, dim=0)
    return output + offsets.unsqueeze(1)
