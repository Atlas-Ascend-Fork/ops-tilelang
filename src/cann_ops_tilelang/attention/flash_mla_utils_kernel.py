# Ruff's simplification rules do not model TileLang's compile-time branches.
# ruff: noqa: SIM108

import tilelang
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores

_UB_BUDGET_BYTES = 200 * 1024


@tilelang.jit(
    out_idx=None,
    target='ascend',
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _convert_kv_index_tl_ascend(
    s: int,
    k: int,
    n: int,
    block_size: int,
) -> None:
    """Single kernel with compile-time specialization: layouts where k is a
    multiple of 64 and n a multiple of 8 use BR-task batches with flat
    contiguous DMAs (the batch stride B == k, so DMA and compute offsets
    match); anything else uses BR=1 per-task 2D copies (unaligned-safe).
    Both share the same SIMD vgather2 core; blk_ub is padded to >=VL
    elements so vgather2's 64-lane span stays inside one version slot."""
    b = T.dynamic('b')
    num_cores = get_num_vector_cores()
    vector_length = 64
    index_tile_size = (k + vector_length - 1) // vector_length * vector_length
    table_tile_size = max(n, vector_length)
    batched = k % vector_length == 0 and n % 8 == 0
    chunked = False
    rows_per_batch = 5
    while (
        rows_per_batch > 1
        and (rows_per_batch * index_tile_size + rows_per_batch * table_tile_size) * 4 * 2 > _UB_BUDGET_BYTES
    ):
        rows_per_batch -= 1
    if batched and (rows_per_batch * index_tile_size + rows_per_batch * table_tile_size) * 4 * 2 > _UB_BUDGET_BYTES:
        batched = False
    if batched:
        num_stages = 2
    else:
        rows_per_batch = 1
        one_stage_bytes = (index_tile_size + table_tile_size) * 4
        if one_stage_bytes > _UB_BUDGET_BYTES:
            chunked = True
            num_stages = 1
        else:
            num_stages = min(4, _UB_BUDGET_BYTES // one_stage_bytes)
    is_pow2 = (block_size & (block_size - 1)) == 0
    block_bits = block_size.bit_length() - 1 if is_pow2 else 0

    if chunked:
        index_tile = 4096
        block_table_tile = 32768
        num_index_blocks = (k + index_tile - 1) // index_tile

        @T.prim_func
        def _convert_kv_index_chunked(
            topk_idxs: T.Tensor([b, s, k], T.int32),  # type: ignore
            context_lens: T.Tensor([b, s], T.int32),  # type: ignore
            block_tables: T.Tensor([b, n], T.int32),  # type: ignore
        ):
            with T.Kernel(num_cores) as pid:
                idx_ub = T.alloc_shared((index_tile,), 'int32')
                out_ub = T.alloc_shared((index_tile,), 'int32')
                blk_ub = T.alloc_shared((block_table_tile,), 'int32')

                for task in T.Persistent([b * s * num_index_blocks], num_cores, pid, group_size=1, num_stages=1):
                    row_task = task // num_index_blocks
                    index_block = task % num_index_blocks
                    bid = row_task // s
                    sid = row_task % s
                    cl = context_lens[bid, sid]
                    index_begin = index_block * index_tile
                    index_count = T.min(index_tile, k - index_begin)
                    T.copy(
                        topk_idxs[bid, sid, index_begin : index_begin + index_count],
                        idx_ub[:index_count],
                    )

                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        neg1 = T.simd.vdup(-1, 'int32', mask)
                        for vector in T.serial(T.ceildiv(index_count, vector_length)):
                            T.simd.vsts(out_ub[vector * vector_length], neg1, mask)

                    for table_block in T.serial(T.ceildiv(n, block_table_tile)):
                        table_begin = table_block * block_table_tile
                        table_count = T.min(block_table_tile, n - table_begin)
                        T.copy(
                            block_tables[bid, table_begin : table_begin + table_count],
                            blk_ub[:table_count],
                        )

                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            zero = T.simd.vdup(0, 'int32', mask)
                            table_begin_vec = T.simd.vdup(table_begin, 'int32', mask)
                            table_end_vec = T.simd.vdup(table_begin + table_count, 'int32', mask)
                            if is_pow2:
                                bs_mask = T.simd.vdup(block_size - 1, 'int32', mask)
                            else:
                                bs_vec = T.simd.vdup(block_size, 'int32', mask)

                            for vector in T.serial(T.ceildiv(index_count, vector_length)):
                                global_index = T.simd.vci(index_begin + vector * vector_length, 'int32')
                                x_vec = T.simd.vld(idx_ub[vector * vector_length])
                                if is_pow2:
                                    blk_idx = T.simd.vshrs(x_vec, block_bits, mask)
                                    rem = T.simd.vand(x_vec, bs_mask, mask)
                                else:
                                    blk_idx = T.simd.vdiv(x_vec, bs_vec, mask)
                                    rem = T.simd.vsub(x_vec, T.simd.vmul(blk_idx, bs_vec, mask), mask)

                                valid_index = T.simd.vcmps(global_index, k, mask, 'lt')
                                valid_context = T.simd.vcmps(global_index, cl, mask, 'lt')
                                in_table_lower = T.simd.vcmp(blk_idx, table_begin_vec, mask, 'ge')
                                in_table_upper = T.simd.vcmp(blk_idx, table_end_vec, mask, 'lt')
                                valid = T.simd.pand(valid_index, valid_context, mask)
                                valid = T.simd.pand(valid, in_table_lower, mask)
                                valid = T.simd.pand(valid, in_table_upper, mask)

                                local_blk_idx = T.simd.vsub(blk_idx, table_begin_vec, mask)
                                local_blk_idx = T.simd.vsel(local_blk_idx, zero, valid)
                                blk_val = T.simd.vgather2(
                                    blk_ub[0],
                                    T.reinterpret('uint32x64', local_blk_idx),
                                    valid,
                                )
                                if is_pow2:
                                    converted = T.simd.vadd(T.simd.vshls(blk_val, block_bits, mask), rem, mask)
                                else:
                                    converted = T.simd.vadd(T.simd.vmul(blk_val, bs_vec, mask), rem, mask)
                                previous = T.simd.vld(out_ub[vector * vector_length])
                                T.simd.vsts(
                                    out_ub[vector * vector_length],
                                    T.simd.vsel(converted, previous, valid),
                                    mask,
                                )

                    T.copy(
                        out_ub[:index_count],
                        topk_idxs[bid, sid, index_begin : index_begin + index_count],
                    )

        return _convert_kv_index_chunked

    @T.prim_func
    def _convert_kv_index(
        topk_idxs: T.Tensor([b, s, k], T.int32),  # type: ignore
        context_lens: T.Tensor([b, s], T.int32),  # type: ignore
        block_tables: T.Tensor([b, n], T.int32),  # type: ignore
    ):
        topk_flat = T.reshape(topk_idxs, (b * s * k,))
        cl_flat = T.reshape(context_lens, (b * s,))
        bt_flat = T.reshape(block_tables, (b * n,))
        ntask = b * s
        with T.Kernel(num_cores) as pid:
            idx_ub = T.alloc_shared((rows_per_batch * index_tile_size,), 'int32')
            blk_ub = T.alloc_shared((rows_per_batch * table_tile_size,), 'int32')
            T.annotate_buffer_versions({idx_ub: num_stages, blk_ub: num_stages})

            t0 = (pid * ntask) // num_cores
            t1 = ((pid + 1) * ntask) // num_cores
            my = t1 - t0

            for j in T.Pipelined(T.ceildiv(my, rows_per_batch), num_stages=num_stages):
                lo = t0 + j * rows_per_batch
                hi = T.min(lo + rows_per_batch, t1)
                ln = hi - lo
                if ln > 0:
                    bid_lo = lo // s

                    if batched:
                        bid_hi = (hi - 1) // s
                        nrows = bid_hi - bid_lo + 1

                        T.copy(topk_flat[lo * k : hi * k], idx_ub[0 : ln * k])
                        T.copy(bt_flat[bid_lo * n : bid_lo * n + nrows * n], blk_ub[0 : nrows * n])
                    else:
                        for i in T.serial(rows_per_batch):
                            if i < ln:
                                task = lo + i
                                bid = task // s
                                sid = task % s
                                T.copy(topk_idxs[bid, sid, 0:k], idx_ub[i * index_tile_size : i * index_tile_size + k])
                                T.copy(block_tables[bid, 0:n], blk_ub[i * table_tile_size : i * table_tile_size + n])

                    for i in T.serial(rows_per_batch):
                        if i < ln:
                            task = lo + i
                            bid = task // s
                            cl = cl_flat[task]
                            if batched:
                                row = (bid - bid_lo) * n
                            else:
                                row = i * table_tile_size

                            with T.SimdVF():
                                mask = T.simd.pset(32)
                                neg1 = T.simd.vdup(-1, 'int32', mask)

                                if is_pow2:
                                    bs_mask = T.simd.vdup(block_size - 1, 'int32', mask)
                                    for v in T.serial(index_tile_size // vector_length):
                                        gidx = T.simd.vci(v * vector_length, 'int32')
                                        x_vec = T.simd.vld(idx_ub[i * index_tile_size + v * vector_length])
                                        blk_idx = T.simd.vshrs(x_vec, block_bits, mask)
                                        rem = T.simd.vand(x_vec, bs_mask, mask)

                                        mask_k = T.simd.vcmps(gidx, k, mask, 'lt')
                                        mask_cl = T.simd.vcmps(gidx, cl, mask, 'lt')
                                        mask_gather = T.simd.pand(mask_k, mask_cl, mask)

                                        blk_val = T.simd.vgather2(
                                            blk_ub[row],
                                            T.reinterpret('uint32x64', blk_idx),
                                            mask_gather,
                                        )
                                        val = T.simd.vadd(T.simd.vshls(blk_val, block_bits, mask), rem, mask)
                                        res = T.simd.vsel(val, neg1, mask_gather)
                                        T.simd.vsts(idx_ub[i * index_tile_size + v * vector_length], res, mask)
                                else:
                                    bs_vec = T.simd.vdup(block_size, 'int32', mask)
                                    for v in T.serial(index_tile_size // vector_length):
                                        gidx = T.simd.vci(v * vector_length, 'int32')
                                        x_vec = T.simd.vld(idx_ub[i * index_tile_size + v * vector_length])
                                        blk_idx = T.simd.vdiv(x_vec, bs_vec, mask)
                                        rem = T.simd.vsub(x_vec, T.simd.vmul(blk_idx, bs_vec, mask), mask)

                                        mask_k = T.simd.vcmps(gidx, k, mask, 'lt')
                                        mask_cl = T.simd.vcmps(gidx, cl, mask, 'lt')
                                        mask_gather = T.simd.pand(mask_k, mask_cl, mask)

                                        blk_val = T.simd.vgather2(
                                            blk_ub[row],
                                            T.reinterpret('uint32x64', blk_idx),
                                            mask_gather,
                                        )
                                        val = T.simd.vadd(T.simd.vmul(blk_val, bs_vec, mask), rem, mask)
                                        res = T.simd.vsel(val, neg1, mask_gather)
                                        T.simd.vsts(idx_ub[i * index_tile_size + v * vector_length], res, mask)

                    if batched:
                        T.copy(idx_ub[0 : ln * k], topk_flat[lo * k : hi * k])
                    else:
                        for i in T.serial(rows_per_batch):
                            if i < ln:
                                task = lo + i
                                bid = task // s
                                sid = task % s
                                T.copy(idx_ub[i * index_tile_size : i * index_tile_size + k], topk_idxs[bid, sid, 0:k])

    return _convert_kv_index
