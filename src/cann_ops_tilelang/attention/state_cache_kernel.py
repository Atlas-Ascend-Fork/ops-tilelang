import tilelang
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(out_idx=None, target='ascend')
def _select_state_slice_tl_ascend(
    block_size_kvstore: int, block_size_inference: int, num_layers: int, feature_dim: int, contiguous_tokens: bool
) -> None:
    """Create an Ascend AIV kernel for selecting a state-cache slice."""
    cache_layer_stride = T.dynamic('cache_layer_stride')
    cache_token_stride = T.dynamic('cache_token_stride')
    tensor_layer_stride = T.dynamic('tensor_layer_stride')
    tensor_token_stride = T.dynamic('tensor_token_stride')

    num_tokens = block_size_inference
    num_cores = get_num_vector_cores()
    num_stages = 1
    tokens_per_chunk = max(1, min(num_tokens, block_size_kvstore, block_size_inference, 96 * 1024 // feature_dim))
    num_chunks = (num_tokens + tokens_per_chunk - 1) // tokens_per_chunk
    total_tasks = num_layers * num_chunks
    kernel_cores = min(num_cores, total_tasks if contiguous_tokens else num_layers * num_tokens)

    @T.macro
    def copy_rows(state_cache, state_tensor, state_ub, layer_idx, source_idx, target_idx, num_tokens):
        T.copy(
            state_cache[layer_idx, source_idx : source_idx + num_tokens, :],
            state_ub[:num_tokens, :],
            l2_cache_ctrl='NOTALLOC_KEEP',
        )
        T.copy(state_ub[:num_tokens, :], state_tensor[layer_idx, target_idx : target_idx + num_tokens, :])

    @T.macro
    def copy_chunk(state_cache, state_tensor, state_ub, flat_idx, last_hit_position):
        layer_idx = flat_idx // num_chunks
        chunk_offset = flat_idx % num_chunks * tokens_per_chunk
        chunk_tokens = T.min(tokens_per_chunk, num_tokens - chunk_offset)
        position = T.cast(last_hit_position, 'int64') - num_tokens + chunk_offset
        source_token_idx = T.floormod(position, block_size_kvstore)
        target_token_idx = T.floormod(position, block_size_inference)
        first_tokens = T.min(
            chunk_tokens, T.min(block_size_kvstore - source_token_idx, block_size_inference - target_token_idx)
        )
        copy_rows(state_cache, state_tensor, state_ub, layer_idx, source_token_idx, target_token_idx, first_tokens)

        remaining_tokens = chunk_tokens - first_tokens
        if remaining_tokens > 0:
            source_token_idx_1 = T.floormod(source_token_idx + first_tokens, block_size_kvstore)
            target_token_idx_1 = T.floormod(target_token_idx + first_tokens, block_size_inference)
            second_tokens = T.min(
                remaining_tokens,
                T.min(block_size_kvstore - source_token_idx_1, block_size_inference - target_token_idx_1),
            )
            copy_rows(
                state_cache, state_tensor, state_ub, layer_idx, source_token_idx_1, target_token_idx_1, second_tokens
            )

            remaining_tokens_1 = remaining_tokens - second_tokens
            if remaining_tokens_1 > 0:
                source_token_idx_2 = T.floormod(source_token_idx_1 + second_tokens, block_size_kvstore)
                target_token_idx_2 = T.floormod(target_token_idx_1 + second_tokens, block_size_inference)
                copy_rows(
                    state_cache,
                    state_tensor,
                    state_ub,
                    layer_idx,
                    source_token_idx_2,
                    target_token_idx_2,
                    remaining_tokens_1,
                )

    @T.prim_func
    def _select_state_slice_ascend(
        state_tensor: T.StridedTensor(
            (num_layers, block_size_inference, feature_dim), (tensor_layer_stride, tensor_token_stride, 1), 'uint8'
        ),  # pyright: ignore[reportInvalidTypeForm]
        state_cache: T.StridedTensor(
            (num_layers, block_size_kvstore, feature_dim), (cache_layer_stride, cache_token_stride, 1), 'uint8'
        ),  # pyright: ignore[reportInvalidTypeForm]
        last_hit_position: T.int32,
    ) -> None:
        T.assume(cache_layer_stride % 8 == 0)
        T.assume(cache_token_stride % 8 == 0)
        T.assume(tensor_layer_stride % 8 == 0)
        T.assume(tensor_token_stride % 8 == 0)

        with T.Kernel(kernel_cores) as core_id:
            if contiguous_tokens:
                state_ub = T.alloc_shared((tokens_per_chunk, feature_dim), 'uint8')
                T.annotate_buffer_versions({state_ub: num_stages})
                if total_tasks <= num_cores:
                    copy_chunk(state_cache, state_tensor, state_ub, core_id, last_hit_position)
                else:
                    for flat_idx in T.Persistent(
                        [total_tasks], kernel_cores, core_id, group_size=1, num_stages=num_stages
                    ):
                        copy_chunk(state_cache, state_tensor, state_ub, flat_idx, last_hit_position)
            else:
                state_ub = T.alloc_shared((feature_dim,), 'uint8')
                T.annotate_buffer_versions({state_ub: num_stages})
                for flat_idx in T.Persistent(
                    [num_layers * num_tokens], kernel_cores, core_id, group_size=1, num_stages=num_stages
                ):
                    layer_idx = flat_idx // num_tokens
                    token_idx = flat_idx % num_tokens
                    position = T.cast(last_hit_position, 'int64') - num_tokens + token_idx
                    source_token_idx = T.floormod(position, block_size_kvstore)
                    target_token_idx = T.floormod(position, block_size_inference)
                    T.copy(state_cache[layer_idx, source_token_idx, 0], state_ub)
                    T.copy(state_ub, state_tensor[layer_idx, target_token_idx, 0])

    return _select_state_slice_ascend


ASCEND_910B_UB_BYTES = 192 * 1024


_MAX_COPY_BYTES = 128 * 1024


_RRWW_MAX_UB_BYTES = 40 * 1024


@tilelang.jit(out_idx=None, target='ascend')
def _load_state_to_cache_tl_ascend(
    block_size_kvstore: int,
    block_size_inference: int,
    num_layers: int,
    feature_dim: int,
    contiguous_tokens: bool,
) -> None:
    """Create an Ascend AIV kernel for loading state into the cache."""
    cache_layer_stride = T.dynamic('cache_layer_stride')
    cache_token_stride = T.dynamic('cache_token_stride')
    tensor_layer_stride = T.dynamic('tensor_layer_stride')
    tensor_token_stride = T.dynamic('tensor_token_stride')

    num_tokens_to_store = min(block_size_inference, block_size_kvstore)
    num_cores = get_num_vector_cores()
    # Double buffering triggers a Bisheng 15.0.5 UB dead-store pass crash.
    num_stages = 1

    # Reduce idle core overhead for contiguous copies.
    # When work items (num_layers * num_chunks) < num_cores, excess cores sit
    # idle but still participate in barrier sync and incur mte2_wait.  Capping
    # num_cores to the actual work items eliminates this overhead.
    # All params below are JIT build-time constants, so the if/cap logic is
    # resolved at compile time with zero runtime cost.
    if contiguous_tokens:
        _tokens_per_chunk = max(1, min(num_tokens_to_store, _MAX_COPY_BYTES // feature_dim))
        _ub_half = ASCEND_910B_UB_BYTES // 2
        _tokens_per_chunk = max(1, min(_tokens_per_chunk, _ub_half // feature_dim))
        _use_rrww = _tokens_per_chunk * feature_dim <= _RRWW_MAX_UB_BYTES
        _num_chunks = (num_tokens_to_store + _tokens_per_chunk - 1) // _tokens_per_chunk
        _work_items = num_layers * _num_chunks
        if _work_items < num_cores:
            num_cores = _work_items

    # L2 cache hint — stream data without allocating L2 cache lines.
    # Since each datum is read once and written once (no temporal reuse),
    # NOTALLOC_KEEP prevents polluting the L2 cache, improving hit rate for
    # other concurrent kernels that benefit from L2 caching.
    l2_ctrl = 'NOTALLOC_KEEP'

    # Factor load/store into parameterized macros for reuse in both RRWW
    # and RWRW paths, accepting an ub_offset to address disjoint UB regions.
    @T.macro
    def load_seg(state_tensor, state_ub, layer_idx, src_start, n_tokens, ub_offset):
        """Load one segment: GM source → UB[ub_offset : ub_offset+n]."""
        T.copy(
            state_tensor[layer_idx, src_start : src_start + n_tokens, :],
            state_ub[ub_offset : ub_offset + n_tokens, :],
            l2_cache_ctrl=l2_ctrl,
        )

    @T.macro
    def store_seg(state_cache, state_ub, layer_idx, tgt_start, n_tokens, ub_offset):
        """Store one segment: UB[ub_offset : ub_offset+n] → GM sink."""
        T.copy(
            state_ub[ub_offset : ub_offset + n_tokens, :],
            state_cache[layer_idx, tgt_start : tgt_start + n_tokens, :],
        )

    # Refactored single `copy_rows` inline loop into a dedicated macro.
    # Dispatches to two copy-order variants at JIT compile time (use_rrww is
    # a build-time constant), so there is zero runtime branch divergence.
    @T.macro
    def _contiguous_dispatch(
        state_tensor,
        state_cache,
        core_id,
        num_cores,
        num_stages,
        num_tokens_to_store,
        block_size_inference,
        block_size_kvstore,
        tokens_per_chunk,
        num_chunks,
        last_hit_position,
        use_rrww,
    ):
        """Single Persistent loop body; diverges on use_rrww at the copy stage.

        Copy order differs:
        * use_rrww=True  (UB ≤ 40KB):  two independent UBs, copy order R-R-W-W.
        * use_rrww=False (UB > 40KB):  one UB,          copy order R-W-R-W.
        """
        if use_rrww:
            ub0 = T.alloc_shared((tokens_per_chunk, feature_dim), 'uint8')
            ub1 = T.alloc_shared((tokens_per_chunk, feature_dim), 'uint8')
            T.annotate_buffer_versions({ub0: num_stages, ub1: num_stages})

            for flat_idx in T.Persistent(
                [num_layers * num_chunks],
                num_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                layer_idx = flat_idx // num_chunks
                chunk_idx = flat_idx % num_chunks
                chunk_offset = chunk_idx * tokens_per_chunk
                chunk_tokens = T.min(tokens_per_chunk, num_tokens_to_store - chunk_offset)
                position = T.cast(last_hit_position, 'int64') - num_tokens_to_store + chunk_offset
                src0 = T.truncmod(position + block_size_inference, block_size_inference)
                tgt0 = T.truncmod(position + block_size_kvstore, block_size_kvstore)

                n1 = T.min(chunk_tokens, T.min(block_size_inference - src0, block_size_kvstore - tgt0))
                rem = chunk_tokens - n1
                s1 = T.truncmod(src0 + n1, block_size_inference)
                t1 = T.truncmod(tgt0 + n1, block_size_kvstore)
                n2 = T.min(rem, T.min(block_size_inference - s1, block_size_kvstore - t1))

                # read-read: MTE2 issues back-to-back into disjoint UB regions
                load_seg(state_tensor, ub0, layer_idx, src0, n1, 0)
                if rem > 0:
                    load_seg(state_tensor, ub1, layer_idx, s1, n2, 0)

                # write-write: MTE3 issues back-to-back
                store_seg(state_cache, ub0, layer_idx, tgt0, n1, 0)
                if rem > 0:
                    store_seg(state_cache, ub1, layer_idx, t1, n2, 0)

                # optional third segment (same-buffer round-trip is fine)
                rem2 = rem - n2
                if rem2 > 0:
                    s2 = T.truncmod(s1 + n2, block_size_inference)
                    t2 = T.truncmod(t1 + n2, block_size_kvstore)
                    load_seg(state_tensor, ub0, layer_idx, s2, rem2, 0)
                    store_seg(state_cache, ub0, layer_idx, t2, rem2, 0)

        else:
            ub = T.alloc_shared((tokens_per_chunk, feature_dim), 'uint8')
            T.annotate_buffer_versions({ub: num_stages})

            for flat_idx in T.Persistent(
                [num_layers * num_chunks],
                num_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                layer_idx = flat_idx // num_chunks
                chunk_idx = flat_idx % num_chunks
                chunk_offset = chunk_idx * tokens_per_chunk
                chunk_tokens = T.min(tokens_per_chunk, num_tokens_to_store - chunk_offset)
                position = T.cast(last_hit_position, 'int64') - num_tokens_to_store + chunk_offset
                src0 = T.truncmod(position + block_size_inference, block_size_inference)
                tgt0 = T.truncmod(position + block_size_kvstore, block_size_kvstore)

                n1 = T.min(chunk_tokens, T.min(block_size_inference - src0, block_size_kvstore - tgt0))

                # segment 1  (R-W serial)
                load_seg(state_tensor, ub, layer_idx, src0, n1, 0)
                store_seg(state_cache, ub, layer_idx, tgt0, n1, 0)

                rem = chunk_tokens - n1
                if rem > 0:
                    s1 = T.truncmod(src0 + n1, block_size_inference)
                    t1 = T.truncmod(tgt0 + n1, block_size_kvstore)
                    n2 = T.min(rem, T.min(block_size_inference - s1, block_size_kvstore - t1))
                    load_seg(state_tensor, ub, layer_idx, s1, n2, 0)
                    store_seg(state_cache, ub, layer_idx, t1, n2, 0)

                    rem2 = rem - n2
                    if rem2 > 0:
                        s2 = T.truncmod(s1 + n2, block_size_inference)
                        t2 = T.truncmod(t1 + n2, block_size_kvstore)
                        load_seg(state_tensor, ub, layer_idx, s2, rem2, 0)
                        store_seg(state_cache, ub, layer_idx, t2, rem2, 0)

    # Changed to T.Persistent scheduler for better load balancing under
    # dynamic workloads.  prim_func body now delegates contiguous path to
    # keeping only thin glue logic here for readability.
    @T.prim_func
    def _load_state_to_cache_ascend(
        state_tensor: T.StridedTensor(
            (num_layers, block_size_inference, feature_dim),
            (tensor_layer_stride, tensor_token_stride, 1),
            'uint8',
        ),  # pyright: ignore[reportInvalidTypeForm]
        state_cache: T.StridedTensor(
            (num_layers, block_size_kvstore, feature_dim),
            (cache_layer_stride, cache_token_stride, 1),
            'uint8',
        ),  # pyright: ignore[reportInvalidTypeForm]
        last_hit_position: T.int32,
    ) -> None:
        with T.Kernel(num_cores) as core_id:
            if contiguous_tokens:
                # _MAX_COPY_BYTES (128 KB) instead of master's hardcoded 32 KB.
                # Additional UB-half cap guarantees each UB buffer fits in 96 KB,
                # leaving headroom for compiler metadata.
                tokens_per_chunk = max(1, min(num_tokens_to_store, _MAX_COPY_BYTES // feature_dim))
                tokens_per_chunk = max(1, min(tokens_per_chunk, (ASCEND_910B_UB_BYTES // 2) // feature_dim))
                num_chunks = (num_tokens_to_store + tokens_per_chunk - 1) // tokens_per_chunk

                # Delegates to _contiguous_dispatch (RRWW or RWRW variant)
                # instead of master's inline copy_rows with single UB.
                _contiguous_dispatch(
                    state_tensor,
                    state_cache,
                    core_id,
                    num_cores,
                    num_stages,
                    num_tokens_to_store,
                    block_size_inference,
                    block_size_kvstore,
                    tokens_per_chunk,
                    num_chunks,
                    last_hit_position,
                    _use_rrww,
                )
            else:
                state_ub = T.alloc_shared((feature_dim,), 'uint8')
                T.annotate_buffer_versions({state_ub: num_stages})

                for flat_idx in T.Persistent(
                    [num_layers * num_tokens_to_store], num_cores, core_id, group_size=1, num_stages=num_stages
                ):
                    layer_idx = flat_idx // num_tokens_to_store
                    token_idx = flat_idx % num_tokens_to_store
                    position = T.cast(last_hit_position, 'int64') - num_tokens_to_store + token_idx
                    target_token_idx = T.truncmod(position + block_size_kvstore, block_size_kvstore)
                    source_token_idx = T.truncmod(position + block_size_inference, block_size_inference)

                    T.copy(state_tensor[layer_idx, source_token_idx, 0], state_ub)
                    T.copy(state_ub, state_cache[layer_idx, target_token_idx, 0])

    return _load_state_to_cache_ascend
