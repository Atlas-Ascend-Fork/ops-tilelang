import math

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.multimodal.dsocr import attn_rel_h_rel_w

pytestmark = pytest.mark.npu


def attn_rel_h_rel_w_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_h: torch.Tensor,
    rel_w: torch.Tensor,
    query_chunk_size: int = 128,
) -> torch.Tensor:
    batch, sequence, heads, dim = q.shape
    side = math.isqrt(sequence)
    assert side * side == sequence

    q_head = q.permute(0, 2, 1, 3).float()
    k_head = k.permute(0, 2, 1, 3).float()
    v_head = v.permute(0, 2, 1, 3).float()
    key_idx = torch.arange(sequence, device=q.device)
    key_h = key_idx // side
    key_w = key_idx % side
    output = torch.empty((batch, heads, sequence, dim), dtype=torch.float32, device=q.device)

    for query_start in range(0, sequence, query_chunk_size):
        query_end = min(query_start + query_chunk_size, sequence)
        scores = torch.matmul(
            q_head[:, :, query_start:query_end, :],
            k_head.transpose(-1, -2),
        ) * math.sqrt(1.0 / dim)
        rel_h_chunk = rel_h[:, query_start:query_end, :, :side].permute(0, 2, 1, 3).float()
        rel_w_chunk = rel_w[:, query_start:query_end, :, :side].permute(0, 2, 1, 3).float()
        scores += rel_h_chunk.index_select(-1, key_h)
        scores += rel_w_chunk.index_select(-1, key_w)
        probs = torch.softmax(scores, dim=-1).to(torch.bfloat16).float()
        output[:, :, query_start:query_end, :] = torch.matmul(probs, v_head)

    return output.permute(0, 2, 1, 3).to(torch.bfloat16)


@pytest.mark.parametrize('batch,side', [(2, 14), (1, 40), (1, 64)])
def test_attn_rel_h_rel_w(batch: int, side: int) -> None:
    torch.manual_seed(42)
    device = get_device()
    sequence = side * side
    heads = 12
    dim = 64
    width_aligned = ((side + 7) // 8) * 8

    qkv = torch.randn((batch, sequence, 3, heads, dim), dtype=torch.bfloat16, device=device)
    q, k, v = qkv.unbind(2)
    rel_h = torch.randn((batch, sequence, heads, width_aligned), dtype=torch.bfloat16, device=device)
    rel_w = torch.randn((batch, sequence, heads, width_aligned), dtype=torch.bfloat16, device=device)

    actual = attn_rel_h_rel_w(q, k, v, rel_h, rel_w)
    expected = attn_rel_h_rel_w_ref(q, k, v, rel_h, rel_w)
    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.005)
