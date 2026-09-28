import pytest
import torch

from cann_ops_tilelang.attention.prepare_cp_mirror import prepare_for_mirror_cp_tilelang
from cann_ops_tilelang.config import get_device

pytestmark = pytest.mark.npu


def prepare_for_mirror_cp_ref(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    compress_ratio: int,
    total_q: int,
    output_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch (CPU loop) implementation of prepare_for_mirror_cp.
    Translates the TileLang kernels back to native Python/PyTorch.

    Args:
        cu_seqlens_q: [batch_size + 1], int32, accelerator
        cu_seqlens_k: [batch_size + 1], int32, accelerator
        cp_size: context parallelism size
        cp_rank: current rank
        compress_ratio: KV compression ratio
        total_q: total number of query tokens
        output_len: total output length for cp_rank

    Returns:
        mirror_cp_index_tensor,
        mirror_cp_reverse_index_tensor,
        ks_tensor,
        ke_tensor,
        cu_seqlens_q_cp_tensor,
        cu_seqlens_k_cp_tensor
    """
    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_k.dtype == torch.int32

    device = cu_seqlens_q.device
    batch_size = cu_seqlens_q.numel() - 1
    assert cu_seqlens_k.numel() == batch_size + 1

    if batch_size == 0 or output_len == 0 or total_q == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        z1 = torch.zeros(1, dtype=torch.int32, device=device)
        return (
            empty,
            torch.zeros(total_q, dtype=torch.int32, device=device),
            empty,
            empty,
            z1,
            z1,
        )

    # Truncating division toward zero, matching C++ integer division.
    def truncdiv(a: int, b: int) -> int:
        if a >= 0:
            return a // b
        return -((-a) // b)

    # Move to CPU for scalar operations
    cu_q = cu_seqlens_q.cpu().numpy().astype(int)
    cu_k = cu_seqlens_k.cpu().numpy().astype(int)

    # ---- Phase 1: Precompute per-batch quantities ---- #
    q_c4_lens = []
    k_c4_lens = []
    partial_seqlens_list = []
    elems_per_batch_list = []
    delta_sq = []  # batch_size x 4
    delta_sk = []  # batch_size x 4

    for i in range(batch_size):
        sq = int(cu_q[i + 1] - cu_q[i])
        sk = int(cu_k[i + 1] - cu_k[i])

        # q_c4, k_c4
        q_c4_val = sk // compress_ratio - (sk - sq) // compress_ratio
        k_c4_val = sk // compress_ratio
        q_c4_lens.append(max(1, q_c4_val))
        k_c4_lens.append(max(1, k_c4_val))

        # partial_seqlen
        ps = (sq + cp_size * 2 - 1) // (cp_size * 2)
        partial_seqlens_list.append(ps)
        elems_per_batch_list.append(2 * ps)

        # num_valid for half0 and half1 at cp_rank
        nv_h0 = min(ps, max(0, sq - ps * cp_rank))
        mirror_rank = cp_size * 2 - 1 - cp_rank
        nv_h1 = min(ps, max(0, sq - ps * mirror_rank))

        kl_h0 = sk - sq + ps * cp_rank + nv_h0
        kl_h1 = sk - sq + ps * mirror_rank + nv_h1

        # delta_sq and delta_sk for 4 append_qk calls
        d_sq_i = [0, 0, 0, 0]
        d_sk_i = [0, 0, 0, 0]

        # --- call 0: sk_arg = kl_h0 >= 0, sq_arg = nv_h0 >= 0 ---
        d_sk_i[0] = kl_h0 // compress_ratio
        d_sq_i[0] = kl_h0 // compress_ratio - (kl_h0 - nv_h0) // compress_ratio

        # --- call 1: sk_arg = -kl_h0 <= 0, sq_arg = ps - nv_h0 >= 0 ---
        d_sk_i[1] = -(kl_h0 // compress_ratio)
        d_sq_i[1] = -(kl_h0 // compress_ratio) - (-((kl_h0 + ps - nv_h0) // compress_ratio))

        # --- call 2: sk_arg = kl_h1 >= 0, sq_arg = nv_h1 >= 0 ---
        d_sk_i[2] = kl_h1 // compress_ratio
        d_sq_i[2] = kl_h1 // compress_ratio - (kl_h1 - nv_h1) // compress_ratio

        # --- call 3: sk_arg = sq - ps*mirror_rank - nv_h1, can be negative ---
        sk_arg3 = sq - ps * mirror_rank - nv_h1
        sq_arg3 = ps - nv_h1
        diff3 = sk_arg3 - sq_arg3  # = sq - ps*(mirror_rank + 1)

        td_sk3 = truncdiv(sk_arg3, compress_ratio)
        td_diff3 = truncdiv(diff3, compress_ratio)
        d_sk_i[3] = td_sk3
        d_sq_i[3] = td_sk3 - td_diff3

        delta_sq.append(d_sq_i)
        delta_sk.append(d_sk_i)

    # ---- Phase 2: Cumulative sums ---- #
    q_c4_lens_t = torch.tensor(q_c4_lens, dtype=torch.int32, device=device)
    k_c4_lens_t = torch.tensor(k_c4_lens, dtype=torch.int32, device=device)

    cu_seqlens_q_c4 = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k_c4 = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q_c4[1:] = torch.cumsum(q_c4_lens_t, dim=0)
    cu_seqlens_k_c4[1:] = torch.cumsum(k_c4_lens_t, dim=0)

    elems_per_batch_t = torch.tensor(elems_per_batch_list, dtype=torch.int32, device=device)
    cu_elems = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_elems[1:] = torch.cumsum(elems_per_batch_t, dim=0)

    total_per_rank = int(cu_elems[batch_size].item())

    # ---- Phase 3: Main scatter ---- #
    mirror_cp_reverse_index = torch.zeros(total_q, dtype=torch.int32, device=device)
    mirror_cp_index = torch.empty(output_len, dtype=torch.int32, device=device)
    ks_out = torch.empty(output_len, dtype=torch.int32, device=device)
    ke_out = torch.empty(output_len, dtype=torch.int32, device=device)

    # Convert cumulative arrays to CPU for indexing
    cu_elems_cpu = cu_elems.cpu().numpy().astype(int)
    cu_k_c4_cpu = cu_seqlens_k_c4.cpu().numpy().astype(int)

    for i in range(batch_size):
        sq = int(cu_q[i + 1] - cu_q[i])
        sk = int(cu_k[i + 1] - cu_k[i])
        cu_q_i = int(cu_q[i])
        cu_k_c4_i = int(cu_k_c4_cpu[i])
        ps = partial_seqlens_list[i]
        cu_elem_i = int(cu_elems_cpu[i])

        for rank in range(cp_size):
            # ---- half 0: local_idx = ps * rank + j ----
            for j in range(ps):
                local_idx = ps * rank + j
                cnt = rank * total_per_rank + cu_elem_i + j
                if local_idx < sq:
                    mirror_cp_reverse_index[cu_q_i + local_idx] = cnt
                if rank == cp_rank:
                    out_pos = cu_elem_i + j
                    if local_idx < sq:
                        mirror_cp_index[out_pos] = cu_q_i + local_idx
                        ks_out[out_pos] = cu_k_c4_i
                        ke_out[out_pos] = cu_k_c4_i + (sk - sq + local_idx + 1) // compress_ratio
                    else:
                        mirror_cp_index[out_pos] = -1
                        ks_out[out_pos] = cu_k_c4_i
                        ke_out[out_pos] = cu_k_c4_i

            # ---- half 1: local_idx = ps * (cp_size*2-1-rank) + j ----
            for j in range(ps):
                mirror_rank = cp_size * 2 - 1 - rank
                local_idx = ps * mirror_rank + j
                cnt = rank * total_per_rank + cu_elem_i + ps + j
                if local_idx < sq:
                    mirror_cp_reverse_index[cu_q_i + local_idx] = cnt
                if rank == cp_rank:
                    out_pos = cu_elem_i + ps + j
                    if local_idx < sq:
                        mirror_cp_index[out_pos] = cu_q_i + local_idx
                        ks_out[out_pos] = cu_k_c4_i
                        ke_out[out_pos] = cu_k_c4_i + (sk - sq + local_idx + 1) // compress_ratio
                    else:
                        mirror_cp_index[out_pos] = -1
                        ks_out[out_pos] = cu_k_c4_i
                        ke_out[out_pos] = cu_k_c4_i

    # ---- Phase 4: cu_seqlens_q_cp / cu_seqlens_k_cp ---- #
    delta_sq_flat = torch.tensor(delta_sq, dtype=torch.int32, device=device).reshape(-1)
    delta_sk_flat = torch.tensor(delta_sk, dtype=torch.int32, device=device).reshape(-1)

    cu_seqlens_q_cp = torch.zeros(batch_size * 4 + 1, dtype=torch.int32, device=device)
    cu_seqlens_k_cp = torch.zeros(batch_size * 4 + 1, dtype=torch.int32, device=device)
    cu_seqlens_q_cp[1:] = torch.cumsum(delta_sq_flat, dim=0)
    cu_seqlens_k_cp[1:] = torch.cumsum(delta_sk_flat, dim=0)

    return (
        mirror_cp_index,
        mirror_cp_reverse_index,
        ks_out,
        ke_out,
        cu_seqlens_q_cp,
        cu_seqlens_k_cp,
    )


def _prefix_sum(values: tuple[int, ...]) -> list[int]:
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return result


def _assert_prepare_for_mirror_cp_matches_ref(
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    cp_size: int,
    cp_rank: int,
    compress_ratio: int,
) -> None:
    device = get_device()
    cu_seqlens_q = torch.tensor(_prefix_sum(q_lens), dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(_prefix_sum(k_lens), dtype=torch.int32, device=device)
    total_q = sum(q_lens)
    output_len = sum(2 * ((sq + cp_size * 2 - 1) // (cp_size * 2)) for sq in q_lens)

    actual = prepare_for_mirror_cp_tilelang(
        cu_seqlens_q,
        cu_seqlens_k,
        cp_size,
        cp_rank,
        compress_ratio,
        total_q,
        output_len,
    )
    expected = prepare_for_mirror_cp_ref(
        cu_seqlens_q,
        cu_seqlens_k,
        cp_size,
        cp_rank,
        compress_ratio,
        total_q,
        output_len,
    )

    output_names = (
        'mirror_cp_index',
        'mirror_cp_reverse_index',
        'ks',
        'ke',
        'cu_seqlens_q_cp',
        'cu_seqlens_k_cp',
    )
    for name, actual_tensor, expected_tensor in zip(output_names, actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor), f'{name} mismatch'


@pytest.mark.parametrize(
    'q_lens,k_lens,cp_size,cp_rank,compress_ratio',
    [
        pytest.param((), (), 1, 0, 4, id='empty'),
        pytest.param((1,), (4,), 1, 0, 2, id='single-token'),
        pytest.param((7, 13), (10, 21), 2, 0, 4, id='two-ranks-rank0'),
        pytest.param((7, 13), (10, 21), 2, 1, 4, id='two-ranks-rank1'),
        pytest.param((1, 17, 33), (4, 24, 50), 4, 0, 8, id='padding-negative-delta'),
        pytest.param((1, 17, 33), (4, 24, 50), 4, 3, 8, id='last-rank'),
        pytest.param((17001,), (20000,), 1, 0, 128, id='multi-block-tail'),
    ],
)
def test_prepare_for_mirror_cp_tilelang_edge_cases(
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    cp_size: int,
    cp_rank: int,
    compress_ratio: int,
) -> None:
    _assert_prepare_for_mirror_cp_matches_ref(
        q_lens,
        k_lens,
        cp_size,
        cp_rank,
        compress_ratio,
    )


@pytest.mark.parametrize(
    'batch_size,max_seqlen_q,max_context_len,cp_size,cp_rank,compress_ratio',
    [
        (1, 1, 0, 1, 0, 1),
        (64, 257, 1024, 2, 0, 4),
        (64, 257, 1024, 2, 1, 4),
        (128, 1024, 4096, 4, 2, 16),
        (512, 128, 512, 8, 7, 32),
    ],
)
def test_prepare_for_mirror_cp_tilelang(
    batch_size: int,
    max_seqlen_q: int,
    max_context_len: int,
    cp_size: int,
    cp_rank: int,
    compress_ratio: int,
) -> None:
    q_lens = tuple((batch_idx * 37 + 1) % max_seqlen_q + 1 for batch_idx in range(batch_size))
    k_lens = tuple(sq + (batch_idx * 29 + 3) % (max_context_len + 1) for batch_idx, sq in enumerate(q_lens))
    _assert_prepare_for_mirror_cp_matches_ref(
        q_lens,
        k_lens,
        cp_size,
        cp_rank,
        compress_ratio,
    )
