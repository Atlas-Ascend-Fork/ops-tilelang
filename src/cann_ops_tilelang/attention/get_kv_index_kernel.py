# Ruff's simplification rules do not model TileLang's compile-time branches.
# ruff: noqa: SIM102

import tilelang
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores


def get_ascend_disable_smem_reuse_config() -> dict:
    pass_configs = {tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True}
    key = getattr(tilelang.PassConfigKey, 'TL_DISABLE_SHARED_MEMORY_REUSE', None)
    if key is not None:
        pass_configs[key] = True
    return pass_configs


@tilelang.jit(
    out_idx=None,
    pass_configs=get_ascend_disable_smem_reuse_config(),
)
def _get_fused_dflash_get_input_tl_ascend(
    swa_block_size: int,
    padded_kv_len: int,
    dflash_block_size: int,
    has_valid_token_mask: bool,
    has_gate_force_random: bool,
) -> None:
    real_cnt = T.dynamic('real_cnt')
    total_q = T.dynamic('total_q')
    kv_index_out_shape0 = T.dynamic('kv_index_out_shape0')
    num_cores = get_num_vector_cores()
    vector_length = 64
    bs = dflash_block_size
    compact_kv = bs * padded_kv_len * 4 > 64 * 1024
    num_stages = 1 if compact_kv and padded_kv_len * 4 > 96 * 1024 else 2
    # requests per persistent iteration: keep the kv chunk UB under 64KB (x2 for double buffering)
    kv_elems_per_request = padded_kv_len if compact_kv else bs * padded_kv_len
    requests_per_group = max(1, min(8, (64 * 1024) // (kv_elems_per_request * 4)))
    rows = requests_per_group * bs
    token_span = 1024  # token-span staging window (elements) for the per-request scalar gather
    is_pow2 = (swa_block_size & (swa_block_size - 1)) == 0

    @T.prim_func
    def _get_fused_dflash_get_input(
        slot_mapping_swa: T.Tensor((total_q,), 'int32'),  # type: ignore
        positions: T.Tensor((total_q,), 'int32'),  # type: ignore
        ftp: T.Tensor((real_cnt,), 'int32'),  # type: ignore
        accepted_pos_real: T.Tensor((real_cnt,), 'int32'),  # type: ignore
        newx_slot_mapping_out: T.Tensor((real_cnt, bs), 'int32'),  # type: ignore
        positions_out: T.Tensor((real_cnt, bs), 'int32'),  # type: ignore
        input_ids_out: T.Tensor((real_cnt, bs), 'int32'),  # type: ignore
        valid_token_mask_out: T.Tensor((real_cnt, bs), 'bool'),  # type: ignore
        gate_force_random_out: T.Tensor((real_cnt, bs), 'bool'),  # type: ignore
        kv_index_out: T.Tensor((kv_index_out_shape0, padded_kv_len), 'int32'),  # type: ignore
        kv_len: T.int32,
        window_size: T.int32,
    ):
        total_rows = real_cnt * bs
        ids_1d = T.reshape(input_ids_out, (total_rows,))
        vtm_1d = T.reshape(valid_token_mask_out, (total_rows,))
        gfr_1d = T.reshape(gate_force_random_out, (total_rows,))
        newx_1d = T.reshape(newx_slot_mapping_out, (total_rows,))
        pos_1d = T.reshape(positions_out, (total_rows,))
        kvout_1d = T.reshape(kv_index_out, (kv_index_out_shape0 * padded_kv_len,))

        with T.Kernel(num_cores) as pid:
            # constant-fill sources: vector-filled int32 patterns (z: 0, t: 0x01010101)
            z_ub = T.alloc_shared((256,), 'int32')  # type: ignore
            t32_ub = T.alloc_shared((256,), 'int32')
            z8 = T.view(z_ub, (1024,), T.bool)  # type: ignore  # bool view for gfr fill
            t_ub = T.view(t32_ub, (1024,), T.bool)  # type: ignore  # bool view for vtm fill
            with T.SimdVF():
                mask = T.simd.pset(32)
                z_vec = T.simd.vdup(0, 'int32', mask)
                t_vec = T.simd.vdup(0x01010101, 'int32', mask)
                for c in T.serial(4):
                    T.simd.vsts(z_ub[c * 64], z_vec, mask)
                    T.simd.vsts(t32_ub[c * 64], t_vec, mask)

            ftp_ub = T.alloc_shared((requests_per_group + 1,), 'int32')  # type: ignore
            ap_ub = T.alloc_shared((64,), 'int32')  # type: ignore
            so_ub = T.alloc_shared((64,), 'int32')  # type: ignore
            start_ub = T.alloc_shared((64,), 'int32')  # type: ignore
            pos_span = T.alloc_shared((token_span,), 'int32')  # type: ignore
            slot_span = T.alloc_shared((token_span,), 'int32')  # type: ignore
            kv2 = T.alloc_shared((requests_per_group * kv_elems_per_request,), 'int32')  # type: ignore
            nx_ub = T.alloc_shared(((rows + 63) // 64 * 64,), 'int32')  # type: ignore
            pos_out_ub = T.alloc_shared(((rows + 63) // 64 * 64,), 'int32')  # type: ignore
            T.annotate_buffer_versions(
                {
                    ftp_ub: num_stages,
                    ap_ub: num_stages,
                    so_ub: num_stages,
                    start_ub: num_stages,
                    pos_span: num_stages,
                    slot_span: num_stages,
                    kv2: num_stages,
                    nx_ub: num_stages,
                    pos_out_ub: num_stages,
                }
            )

            for blk in T.Persistent(
                [T.ceildiv(real_cnt, requests_per_group)], num_cores, pid, group_size=1, num_stages=num_stages
            ):
                g0 = blk * requests_per_group
                glen = T.min(requests_per_group, real_cnt - g0)
                rows_g = glen * bs

                # 1) request metadata: ftp (+1 boundary), ap
                load_len = T.min(requests_per_group + 1, real_cnt - g0)
                T.copy(ftp[g0 : g0 + load_len], ftp_ub[:load_len])
                T.copy(accepted_pos_real[g0 : g0 + glen], ap_ub[:glen])
                s0 = ftp_ub[0]
                s1 = T.Select(g0 + glen < real_cnt, ftp_ub[glen], total_q)

                # 2) per-request so/start via span windows (few large MTE2 instead of 2G tiny ones)
                nwin = T.ceildiv(s1 - s0, token_span)
                for w in T.serial(nwin):
                    w0 = s0 + w * token_span
                    wlen = T.min(token_span, s1 - w0)
                    T.copy(positions[w0 : w0 + wlen], pos_span[:wlen])
                    T.copy(slot_mapping_swa[w0 : w0 + wlen], slot_span[:wlen])
                    for r in T.serial(requests_per_group):
                        if r < glen:
                            if ftp_ub[r] >= w0:
                                if ftp_ub[r] < w0 + wlen:
                                    off = ftp_ub[r] - w0
                                    so_ub[r] = slot_span[off] - pos_span[off] % swa_block_size
                                    start_ub[r] = ap_ub[r] - (window_size - 1)

                # 3) kv rows: compute one row per request, replicate to bs rows, one contiguous write
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    neg1 = T.simd.vdup(-1, 'int32', mask)
                    swa_vec = T.simd.vdup(swa_block_size, 'int32', mask)
                    swa_mask_vec = T.simd.vdup(swa_block_size - 1, 'int32', mask)
                    kv_len_vec = T.simd.vdup(kv_len, 'int32', mask)
                    col = T.simd.alloc_var('int32')
                    for r in T.serial(requests_per_group):
                        if r < glen:
                            so_vec = T.simd.vdup(so_ub[r], 'int32', mask)
                            start_vec = T.simd.vdup(start_ub[r], 'int32', mask)
                            col = T.simd.vci(0, 'int32')
                            for v in T.serial(T.ceildiv(padded_kv_len, vector_length)):
                                pos = T.simd.vadd(col, start_vec, mask)
                                if is_pow2:
                                    rem = T.simd.vand(pos, swa_mask_vec, mask)
                                else:
                                    pos_q = T.simd.vdiv(pos, swa_vec, mask)
                                    rem = T.simd.vsub(pos, T.simd.vmul(pos_q, swa_vec, mask), mask)
                                val = T.simd.vadd(rem, so_vec, mask)
                                # static-operand compares only (avoid scalar->vector syncs)
                                mask_len = T.simd.vcmps(T.simd.vsub(col, kv_len_vec, mask), 0, mask, 'lt')
                                mask_ge = T.simd.vcmps(pos, 0, mask, 'ge')
                                res = T.simd.vsel(val, neg1, T.simd.pand(mask_len, mask_ge, mask))
                                if compact_kv:
                                    T.simd.vsts(kv2[r * padded_kv_len + v * vector_length], res, mask)
                                else:
                                    for bx in T.serial(bs):
                                        T.simd.vsts(kv2[(r * bs + bx) * padded_kv_len + v * vector_length], res, mask)
                                col = T.simd.vadds(col, 64, mask)

                # 4) newx / positions: gather pass over the group's rows, 64-lane chunks
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    swa_vec = T.simd.vdup(swa_block_size, 'int32', mask)
                    swa_mask_vec = T.simd.vdup(swa_block_size - 1, 'int32', mask)
                    bs_vec = T.simd.vdup(bs, 'int32', mask)
                    for ch in T.serial((rows + 63) // 64):
                        lane = T.simd.vci(ch * 64, 'int32')
                        r_lane = T.simd.vdiv(lane, bs_vec, mask)
                        r_lane = T.simd.vmins(r_lane, requests_per_group - 1, mask)
                        bx_lane = T.simd.vsub(lane, T.simd.vmul(r_lane, bs_vec, mask), mask)
                        r_idx = T.reinterpret('uint32x64', r_lane)
                        ap_g = T.simd.vgather2(ap_ub[0], r_idx, mask)
                        so_g = T.simd.vgather2(so_ub[0], r_idx, mask)
                        posv = T.simd.vadd(T.simd.vadds(bx_lane, 1, mask), ap_g, mask)
                        if is_pow2:
                            rem2 = T.simd.vand(posv, swa_mask_vec, mask)
                        else:
                            q2 = T.simd.vdiv(posv, swa_vec, mask)
                            rem2 = T.simd.vsub(posv, T.simd.vmul(q2, swa_vec, mask), mask)
                        newx = T.simd.vadd(rem2, so_g, mask)
                        mact = T.simd.vcmps(lane, rows, mask, 'lt')
                        T.simd.vsts(nx_ub[ch * 64], newx, mact)
                        T.simd.vsts(pos_out_ub[ch * 64], posv, mact)
                T.copy(nx_ub[:rows_g], newx_1d[g0 * bs : g0 * bs + rows_g])
                T.copy(pos_out_ub[:rows_g], pos_1d[g0 * bs : g0 * bs + rows_g])

                # 5) constant fills for the group's rows
                off1 = g0 * bs
                T.copy(z_ub[:rows_g], ids_1d[off1 : off1 + rows_g])
                if has_valid_token_mask:
                    T.copy(t_ub[:rows_g], vtm_1d[off1 : off1 + rows_g])
                if has_gate_force_random:
                    T.copy(z8[:rows_g], gfr_1d[off1 : off1 + rows_g])

                # 6) kv_index: keep the one-copy path for normal shapes and
                # replicate from one staged row when the expanded group would
                # exceed UB capacity.
                if compact_kv:
                    for r in T.serial(requests_per_group):
                        if r < glen:
                            for bx in T.serial(bs):
                                out_row = (g0 + r) * bs + bx
                                T.copy(
                                    kv2[r * padded_kv_len : (r + 1) * padded_kv_len],
                                    kvout_1d[out_row * padded_kv_len : (out_row + 1) * padded_kv_len],
                                )
                else:
                    T.copy(
                        kv2[: rows_g * padded_kv_len],
                        kvout_1d[g0 * bs * padded_kv_len : g0 * bs * padded_kv_len + rows_g * padded_kv_len],
                    )

    return _get_fused_dflash_get_input
