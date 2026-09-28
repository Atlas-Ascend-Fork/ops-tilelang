from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.attention.prepare_llama_decode_metadata import prepare_llama_decode_metadata
from cann_ops_tilelang.config import get_device

pytestmark = pytest.mark.npu


def prepare_llama_decode_metadata_ref(
    seqlens_q: torch.Tensor,
    seqlens_k: torch.Tensor,
    block_tables_in: torch.Tensor,
    block_tables_offsets: torch.Tensor,
    is_placeholder: torch.Tensor,
    total_q: int,
    *,
    max_num_blocks: int,
    block_size: int,
    force_load_balance: bool = False,
) -> dict:
    """CPU reference translated from csrc/engine/input_metadata/prepare_tensor_input_LLaMA.cpp
    (prepare_decode_LLaMA + update_common_LLaMA + prepare_decode_base)."""
    n_seq = seqlens_q.shape[0]

    sq_cpu = seqlens_q.cpu()
    sk_cpu = seqlens_k.cpu()
    bt_cpu = block_tables_in.cpu()
    off_cpu = block_tables_offsets.cpu()
    ip_cpu = is_placeholder.cpu()

    actual_max_blocks_cpu = max(int(off_cpu[i + 1].item() - off_cpu[i].item()) for i in range(n_seq))

    positions = torch.zeros(total_q, dtype=torch.int32, device='cpu')
    valid_token_mask = torch.zeros(total_q, dtype=torch.bool, device='cpu')
    gate_force_random = torch.zeros(total_q, dtype=torch.bool, device='cpu')
    fix_routing_mask = torch.zeros(total_q, dtype=torch.bool, device='cpu')
    context_lens = torch.zeros(total_q, dtype=torch.int32, device='cpu')
    slot_mapping_store = torch.zeros(total_q, dtype=torch.int32, device='cpu')
    query_indices = torch.zeros(total_q, dtype=torch.int32, device='cpu')
    block_tables_out = torch.zeros((total_q, actual_max_blocks_cpu), dtype=torch.int32, device='cpu')

    i_q = 0
    for i_seq in range(n_seq):
        sk = sk_cpu[i_seq].item()
        sq = sq_cpu[i_seq].item()
        is_pad = bool(ip_cpu[i_seq].item())
        indices = 998244353 if is_pad else i_seq
        bt_start = int(off_cpu[i_seq].item())
        bt_end = int(off_cpu[i_seq + 1].item())
        num_blocks = bt_end - bt_start
        for j in range(sq):
            pos = sk - sq + j
            positions[i_q] = pos
            valid_token_mask[i_q] = not is_pad
            gate_force_random[i_q] = force_load_balance
            fix_routing_mask[i_q] = False
            context_lens[i_q] = pos + 1
            query_indices[i_q] = indices
            block_number = bt_cpu[bt_start + pos // block_size].item()
            slot_mapping_store[i_q] = block_number * block_size + pos % block_size
            block_tables_out[i_q, :num_blocks] = bt_cpu[bt_start:bt_end]
            i_q += 1

    return {
        'positions': positions,
        'valid_token_mask': valid_token_mask,
        'gate_force_random': gate_force_random,
        'fix_routing_mask': fix_routing_mask,
        'context_lens': context_lens,
        'slot_mapping_store': slot_mapping_store,
        'query_indices': query_indices,
        'block_tables_out': block_tables_out,
    }


def _run_prepare_llama_decode_metadata_case(
    *,
    n_seq: int,
    max_seqlen_q: int,
    max_num_blocks: int,
    seqlen_k_base: int,
    block_size: int,
    has_placeholder: bool,
    force_load_balance: bool,
) -> None:
    seqlens_q = torch.arange(n_seq, dtype=torch.int32, device=get_device()) % max_seqlen_q + 1
    seqlens_k = seqlens_q + seqlen_k_base
    cu_seqlens_q = torch.cumsum(seqlens_q, dim=0, dtype=torch.int32)
    total_q = int(seqlens_q.sum().item())

    # Build compact block tables: num_blocks = ceil(seqlen_k / block_size).
    num_blocks_per_seq = [max(1, (sk.item() + block_size - 1) // block_size) for sk in seqlens_k]
    assert max(num_blocks_per_seq) <= max_num_blocks
    total_blocks = sum(num_blocks_per_seq)
    block_tables_in = torch.empty(total_blocks, dtype=torch.int32, device=get_device())
    block_tables_offsets = torch.empty(n_seq + 1, dtype=torch.int32, device=get_device())
    off = 0
    for i, nb in enumerate(num_blocks_per_seq):
        block_tables_offsets[i] = off
        block_tables_in[off : off + nb] = torch.randint(0, 1000, (nb,), dtype=torch.int32, device=get_device())
        off += nb
    block_tables_offsets[n_seq] = off
    is_placeholder = torch.zeros(n_seq, dtype=torch.bool, device=get_device())
    if has_placeholder:
        is_placeholder[n_seq // 2 :] = True

    positions = torch.zeros(total_q, dtype=torch.int32, device=get_device())
    valid_token_mask = torch.zeros(total_q, dtype=torch.bool, device=get_device())
    gate_force_random = torch.zeros(total_q, dtype=torch.bool, device=get_device())
    fix_routing_mask = torch.zeros(total_q, dtype=torch.bool, device=get_device())
    context_lens = torch.zeros(total_q, dtype=torch.int32, device=get_device())
    slot_mapping_store = torch.zeros(total_q, dtype=torch.int32, device=get_device())
    query_indices = torch.zeros(total_q, dtype=torch.int32, device=get_device())
    block_tables_out = torch.zeros((total_q, max_num_blocks), dtype=torch.int32, device=get_device())

    prepare_llama_decode_metadata(
        seqlens_q,
        seqlens_k,
        cu_seqlens_q,
        block_tables_in,
        block_tables_offsets,
        is_placeholder,
        positions,
        valid_token_mask,
        gate_force_random,
        fix_routing_mask,
        context_lens,
        slot_mapping_store,
        query_indices,
        block_tables_out,
        max_seqlen_q=max_seqlen_q,
        max_num_blocks=max_num_blocks,
        block_size=block_size,
        force_load_balance=force_load_balance,
    )

    ref = prepare_llama_decode_metadata_ref(
        seqlens_q,
        seqlens_k,
        block_tables_in,
        block_tables_offsets,
        is_placeholder,
        total_q,
        max_num_blocks=max_num_blocks,
        block_size=block_size,
        force_load_balance=force_load_balance,
    )

    actual_max_blocks = ref['block_tables_out'].shape[1]
    assert torch.equal(positions.cpu(), ref['positions'])
    assert torch.equal(valid_token_mask.cpu(), ref['valid_token_mask'])
    assert torch.equal(gate_force_random.cpu(), ref['gate_force_random'])
    assert torch.equal(fix_routing_mask.cpu(), ref['fix_routing_mask'])
    assert torch.equal(context_lens.cpu(), ref['context_lens'])
    assert torch.equal(slot_mapping_store.cpu(), ref['slot_mapping_store'])
    assert torch.equal(query_indices.cpu(), ref['query_indices'])
    assert torch.equal(block_tables_out[:, :actual_max_blocks].cpu(), ref['block_tables_out'])


@pytest.mark.parametrize('has_placeholder', [True, False])
@pytest.mark.parametrize('max_seqlen_q', [1, 5, 8])
@pytest.mark.parametrize('max_num_blocks', [256, 1024])
@pytest.mark.parametrize('seqlen_k_base', [30, 127, 200, 255])
@pytest.mark.parametrize('force_load_balance', [True, False])
def test_prepare_llama_decode_metadata(
    has_placeholder: bool, max_seqlen_q: int, max_num_blocks: int, seqlen_k_base: int, force_load_balance: bool
) -> None:
    _run_prepare_llama_decode_metadata_case(
        n_seq=512,
        max_seqlen_q=max_seqlen_q,
        max_num_blocks=max_num_blocks,
        seqlen_k_base=seqlen_k_base,
        block_size=128,
        has_placeholder=has_placeholder,
        force_load_balance=force_load_balance,
    )


GENERALIZED_CASES = [
    pytest.param(1, 1, 3, 1, 16, False, False, id='edge-single-sequence'),
    pytest.param(47, 3, 17, 63, 64, True, False, id='below-vector-core-count'),
    pytest.param(48, 7, 19, 127, 64, False, True, id='at-vector-core-count'),
    pytest.param(49, 2, 23, 128, 64, True, True, id='above-vector-core-count'),
    pytest.param(53, 6, 37, 200, 128, False, False, id='prime-sequence-count'),
    pytest.param(127, 7, 65, 511, 256, True, True, id='prime-nonaligned-long-kv'),
    pytest.param(257, 5, 31, 383, 32, False, True, id='small-cache-blocks'),
    pytest.param(511, 9, 29, 1023, 128, True, False, id='non-power-of-two-multi-round'),
    pytest.param(513, 3, 33, 1023, 128, True, False, id='persistent-tail-one'),
    pytest.param(1025, 8, 41, 2047, 256, False, True, id='persistent-many-rounds'),
]


@pytest.mark.parametrize(
    (
        'n_seq',
        'max_seqlen_q',
        'max_num_blocks',
        'seqlen_k_base',
        'block_size',
        'has_placeholder',
        'force_load_balance',
    ),
    GENERALIZED_CASES,
)
def test_prepare_llama_decode_metadata_generalized(
    n_seq: int,
    max_seqlen_q: int,
    max_num_blocks: int,
    seqlen_k_base: int,
    block_size: int,
    has_placeholder: bool,
    force_load_balance: bool,
) -> None:
    _run_prepare_llama_decode_metadata_case(
        n_seq=n_seq,
        max_seqlen_q=max_seqlen_q,
        max_num_blocks=max_num_blocks,
        seqlen_k_base=seqlen_k_base,
        block_size=block_size,
        has_placeholder=has_placeholder,
        force_load_balance=force_load_balance,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('n_seq', [64, 128, 256, 512, 1024])
def test_prepare_llama_decode_metadata_bench(
    n_seq: int, benchmark_record: Callable[..., Any], benchmark_timer: Callable[..., Any]
) -> None:
    max_seqlen_q = 8
    block_size = 128
    max_num_blocks = 1024

    seqlens_q = torch.full((n_seq,), max_seqlen_q, dtype=torch.int32, device=get_device())
    seqlens_k = seqlens_q + 256
    cu_seqlens_q = torch.cumsum(seqlens_q, dim=0, dtype=torch.int32)
    total_q = n_seq * max_seqlen_q

    num_blocks_per_seq = [max(1, (sk.item() + block_size - 1) // block_size) for sk in seqlens_k]
    total_blocks = sum(num_blocks_per_seq)
    block_tables_in = torch.empty(total_blocks, dtype=torch.int32, device=get_device())
    block_tables_offsets = torch.empty(n_seq + 1, dtype=torch.int32, device=get_device())
    off = 0
    for i, nb in enumerate(num_blocks_per_seq):
        block_tables_offsets[i] = off
        block_tables_in[off : off + nb] = torch.randint(0, 1000, (nb,), dtype=torch.int32, device=get_device())
        off += nb
    block_tables_offsets[n_seq] = off
    is_placeholder = torch.zeros(n_seq, dtype=torch.bool, device=get_device())

    positions = torch.zeros(total_q, dtype=torch.int32, device=get_device())
    valid_token_mask = torch.zeros(total_q, dtype=torch.bool, device=get_device())
    gate_force_random = torch.zeros(total_q, dtype=torch.bool, device=get_device())
    fix_routing_mask = torch.zeros(total_q, dtype=torch.bool, device=get_device())
    context_lens = torch.zeros(total_q, dtype=torch.int32, device=get_device())
    slot_mapping_store = torch.zeros(total_q, dtype=torch.int32, device=get_device())
    query_indices = torch.zeros(total_q, dtype=torch.int32, device=get_device())
    block_tables_out = torch.zeros((total_q, max_num_blocks), dtype=torch.int32, device=get_device())

    t_us = benchmark_timer(
        lambda: prepare_llama_decode_metadata(
            seqlens_q,
            seqlens_k,
            cu_seqlens_q,
            block_tables_in,
            block_tables_offsets,
            is_placeholder,
            positions,
            valid_token_mask,
            gate_force_random,
            fix_routing_mask,
            context_lens,
            slot_mapping_store,
            query_indices,
            block_tables_out,
            max_seqlen_q=max_seqlen_q,
            max_num_blocks=max_num_blocks,
            block_size=block_size,
        )
    )
    benchmark_record(
        kernel='prepare_llama_decode_metadata',
        operation='fwd',
        params={'max_seqlen_q': max_seqlen_q, 'n_seq': n_seq},
        time_us=t_us,
    )
