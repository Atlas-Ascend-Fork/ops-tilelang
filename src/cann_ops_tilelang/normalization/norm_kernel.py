"""TileLang Ascend kernel for normalization operations."""

from typing import Any

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(
    out_idx=None,
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _norm_fwd_tl_ascend(
    d: int,
    dtype: str,
    return_dtype: str,
    has_residual: bool,
    has_residual_out: bool,
    has_bias: bool,
    is_rms_norm: bool,
    eps: float,
    out_scale: float,
    down_cast_before_norm: bool,
    return_origin_out: bool,
    has_x2: bool,
    x_scale: float,
    x_stride_s: int,
    y_stride_s: int,
    residual_stride_s: int,
    residual_out_stride_s: int,
    x2_stride_s: int,
    y_origin_stride_s: int,
    do_fp8_cast: bool = False,
    num_per_channels: int = 1,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> Any:
    batch = T.dynamic('batch')
    seqlen = T.dynamic('seqlen')

    x_stride_b = T.dynamic('x_stride_b')
    y_stride_b = T.dynamic('y_stride_b')
    residual_stride_b = T.dynamic('residual_stride_b')
    residual_out_stride_b = T.dynamic('residual_out_stride_b')
    x2_stride_b = T.dynamic('x2_stride_b')
    y_origin_stride_b = T.dynamic('y_origin_stride_b')
    sf_stride_b = T.dynamic('sf_stride_b')
    sf_stride_s = T.dynamic('sf_stride_s')
    sf_stride_g = T.dynamic('sf_stride_g')

    num_cores = get_num_vector_cores()
    num_groups = (d + num_per_channels - 1) // num_per_channels
    padded_num_groups = ((num_groups + 63) // 64) * 64  # ceil64 for batch scale computation
    num_group_vectors = padded_num_groups // 64
    sf_type = 'uint8' if use_packed_ue8m0 else ('float32' if do_fp8_cast else dtype)
    assert not use_packed_ue8m0 or round_sf, 'packed ue8m0 scale requires round_sf'

    is_bf16 = dtype == 'bfloat16'
    dtype_bytes = 4 if dtype in ('float', 'float32') else 2
    return_dtype_bytes = 1 if return_dtype == T.float8_e4m3fn else (4 if return_dtype in ('float', 'float32') else 2)
    sf_bytes = 1 if use_packed_ue8m0 else (4 if do_fp8_cast else dtype_bytes)
    per_row_bytes = d * return_dtype_bytes
    per_row_bytes += (
        d
        * dtype_bytes
        * (1 + int(has_x2) + int(has_residual) + int(has_residual_out) + int(return_origin_out) + int(do_fp8_cast))
    )
    if do_fp8_cast:
        per_row_bytes += padded_num_groups * sf_bytes + 2 * padded_num_groups * 4
    resident_bytes = d * dtype_bytes * (1 + int(has_bias))
    resident_full = resident_bytes + d * 4
    if has_bias:
        resident_full += d * 4
    resident_stream = resident_bytes
    ub_budget_bytes = 240 * 1024  # 950DT 248KB actual, 8KB margin

    def _derive_block_m(res_bytes):
        if res_bytes + per_row_bytes > ub_budget_bytes:
            return None
        max_bm = (ub_budget_bytes - res_bytes) // (2 * per_row_bytes)
        if max_bm >= 1:
            return 2, min(4, max_bm)
        return 1, min(4, max(1, (ub_budget_bytes - res_bytes) // per_row_bytes))

    full_tiling = _derive_block_m(resident_full)
    stream_tiling = _derive_block_m(resident_stream)
    if full_tiling is None and stream_tiling is None:
        raise ValueError(
            'Ascend norm row does not fit UB budget: '
            f'resident_full={resident_full} resident_stream={resident_stream} '
            f'per_row={per_row_bytes} budget={ub_budget_bytes}'
        )

    stream_affine = full_tiling is None or stream_tiling[1] > full_tiling[1]
    if stream_affine:
        num_stages, block_m = stream_tiling
    else:
        num_stages, block_m = full_tiling

    if block_m == 1:
        resident = resident_stream if stream_affine else resident_full
        if resident + 3 * per_row_bytes <= ub_budget_bytes:
            num_stages = 3

    if use_packed_ue8m0:
        sf_shape = (batch * ((num_groups + 1) // 2), 2 * seqlen)
        sf_stride = (sf_stride_s, 1)
    else:
        sf_shape = (batch, seqlen, num_groups)
        sf_stride = (sf_stride_b, sf_stride_s, sf_stride_g)

    @T.macro
    def load_norm_as_fp32(norm_ub, r, offset: int):
        if is_bf16:
            return S.vcvt(S.vld(norm_ub[r, offset], dist='UNPK_B16'), T.float32, part=0)
        return S.vld(norm_ub[r, offset])

    @T.macro
    def reconstruct_v(x_buf, x2_buf, res_buf, r, off):
        v_vec = S.alloc_var(T.float32)
        v_vec = load_norm_as_fp32(x_buf, r, off)
        if has_x2:
            x2_vec = load_norm_as_fp32(x2_buf, r, off)
            v_vec = S.vadd(v_vec, x2_vec)
            if is_bf16:
                v_vec = S.vcvt(S.vcvt(v_vec, T.bfloat16), T.float32, part=0)
        if x_scale != 1.0:
            v_vec = S.vmuls(v_vec, x_scale)
            if is_bf16:
                v_vec = S.vcvt(S.vcvt(v_vec, T.bfloat16), T.float32, part=0)
        if has_residual:
            r_vec = load_norm_as_fp32(res_buf, r, off)
            v_vec = S.vadd(v_vec, r_vec)
            if down_cast_before_norm and is_bf16:
                v_vec = S.vcvt(S.vcvt(v_vec, T.bfloat16), T.float32, part=0)
        return v_vec

    @T.macro
    def load_affine_as_fp32(native_ub, fp32_ub, off):
        if stream_affine:
            if is_bf16:
                return S.vcvt(S.vld(native_ub[off], dist='UNPK_B16'), T.float32, part=0)
            return S.vld(native_ub[off])
        return S.vld(fp32_ub[off])

    @T.macro
    def store_v_as_dtype(buf, r, off, v_vec):
        if is_bf16:
            S.vsts(buf[r, off], S.vcvt(v_vec, T.bfloat16), dist='PK_B32')
        else:
            S.vsts(buf[r, off], v_vec)

    @T.macro
    def store_scale_row(sf_ub, sf, r, b_id, s_id):
        if use_packed_ue8m0:
            if num_groups % 2 != 0:
                sf_ub[r, num_groups] = 0
            for g in T.serial((num_groups + 1) // 2 * 2):
                sf[
                    b_id * ((num_groups + 1) // 2) + g // 2,
                    s_id * 2 + g % 2,
                ] = sf_ub[r, g]
        else:
            T.copy(sf_ub[r, :num_groups], sf[b_id, s_id, :num_groups])

    @T.prim_func
    def _norm_fwd_ascend(
        x: T.StridedTensor((batch, seqlen, d), (x_stride_b, x_stride_s, 1), dtype),  # type: ignore
        y: T.StridedTensor((batch, seqlen, d), (y_stride_b, y_stride_s, 1), return_dtype),  # type: ignore
        w: T.Tensor((d,), dtype),  # type: ignore
        bias: T.Tensor((d,), dtype),  # type: ignore
        residual: T.StridedTensor((batch, seqlen, d), (residual_stride_b, residual_stride_s, 1), dtype),  # type: ignore
        residual_out: T.StridedTensor(  # type: ignore
            (batch, seqlen, d), (residual_out_stride_b, residual_out_stride_s, 1), dtype
        ),
        x2: T.StridedTensor((batch, seqlen, d), (x2_stride_b, x2_stride_s, 1), dtype),  # type: ignore
        sf: T.StridedTensor(sf_shape, sf_stride, sf_type),  # type: ignore
        y_origin: T.StridedTensor((batch, seqlen, d), (y_origin_stride_b, y_origin_stride_s, 1), dtype),  # type: ignore
    ) -> None:
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((block_m, d), dtype)
            weight_ub = T.alloc_shared((d,), dtype)
            y_ub = T.alloc_shared((block_m, d), return_dtype)
            if do_fp8_cast:
                norm_ub = T.alloc_shared((block_m, d), dtype)
            if has_x2:
                x2_ub = T.alloc_shared((block_m, d), dtype)
            if has_residual:
                residual_ub = T.alloc_shared((block_m, d), dtype)
            if has_residual_out:
                residual_out_ub = T.alloc_shared((block_m, d), dtype)
            if has_bias:
                bias_ub = T.alloc_shared((d,), dtype)
            if return_origin_out:
                y_origin_ub = T.alloc_shared((block_m, d), dtype)
            if do_fp8_cast:
                sf_ub = T.alloc_shared((block_m, padded_num_groups), sf_type)
                amax_ub = T.alloc_shared((block_m, padded_num_groups), 'float32')
                sf_inv_ub = T.alloc_shared((block_m, padded_num_groups), 'float32')
            if not stream_affine:
                w32_ub = T.alloc_shared((d,), 'float32')
                if has_bias:
                    b32_ub = T.alloc_shared((d,), 'float32')
            stats_slots = 1 if is_rms_norm else 2
            stats_ub = T.alloc_shared((block_m, stats_slots), 'float32')
            x2_arg = x2_ub if has_x2 else x_ub
            residual_arg = residual_ub if has_residual else x_ub
            w32_arg = w32_ub if not stream_affine else weight_ub
            b32_arg = b32_ub if (not stream_affine and has_bias) else weight_ub

            buffer_versions = {
                x_ub: num_stages,
                y_ub: num_stages,
            }
            if do_fp8_cast:
                buffer_versions[norm_ub] = num_stages
            if has_x2:
                buffer_versions[x2_ub] = num_stages
            if has_residual:
                buffer_versions[residual_ub] = num_stages
            if has_residual_out:
                buffer_versions[residual_out_ub] = num_stages
            if return_origin_out:
                buffer_versions[y_origin_ub] = num_stages
            if do_fp8_cast:
                buffer_versions[sf_ub] = num_stages
                # Epilogue-local workspaces: rotating them corrupts the scales.
                buffer_versions[amax_ub] = 1
                buffer_versions[sf_inv_ub] = 1
            T.annotate_buffer_versions(buffer_versions)

            if do_fp8_cast:
                with T.SimdVF():
                    zero_vec = S.vdup(0.0, T.float32)
                    for r in range(block_m):
                        for off in range(0, padded_num_groups, 64):
                            S.vsts(amax_ub[r, off], zero_vec)

            T.copy(w[:d], weight_ub[:d])
            if has_bias:
                T.copy(bias[:d], bias_ub[:d])

            if not stream_affine:
                if is_bf16:
                    with T.SimdVF():
                        for vec in range(d // 64):
                            off = vec * 64
                            w_vec = S.vcvt(S.vld(weight_ub[off], dist='UNPK_B16'), T.float32, part=0)
                            S.vsts(w32_ub[off], w_vec)
                            if has_bias:
                                b_vec = S.vcvt(S.vld(bias_ub[off], dist='UNPK_B16'), T.float32, part=0)
                                S.vsts(b32_ub[off], b_vec)
                else:
                    T.copy(weight_ub[:d], w32_ub[:d])
                    if has_bias:
                        T.copy(bias_ub[:d], b32_ub[:d])

            total_rows = batch * seqlen
            for task_id in T.Persistent(
                [T.ceildiv(total_rows, block_m)],
                num_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
                annotations={'enable_offset': True},
            ):
                row_base = task_id * block_m
                batch_id = row_base // seqlen
                seq_id0 = row_base % seqlen

                if seq_id0 + block_m <= seqlen:
                    T.copy(x[batch_id, seq_id0 : seq_id0 + block_m, :d], x_ub[:block_m, :d])
                    if has_x2:
                        T.copy(x2[batch_id, seq_id0 : seq_id0 + block_m, :d], x2_ub[:block_m, :d])
                    if has_residual:
                        T.copy(residual[batch_id, seq_id0 : seq_id0 + block_m, :d], residual_ub[:block_m, :d])
                else:
                    for r in T.serial(block_m):
                        row = row_base + r
                        if row < total_rows:
                            b_id = row // seqlen
                            s_id = row % seqlen
                            T.copy(x[b_id, s_id, :d], x_ub[r, :d])
                            if has_x2:
                                T.copy(x2[b_id, s_id, :d], x2_ub[r, :d])
                            if has_residual:
                                T.copy(residual[b_id, s_id, :d], residual_ub[r, :d])

                with T.SimdVF():
                    vec_full = S.pset(32, 'PAT_ALL')
                    zero = S.vdup(0.0, T.float32, vec_full)

                    for r in T.serial(block_m):
                        row = row_base + r
                        if row < total_rows:
                            acc = S.alloc_var(T.float32)
                            acc = zero
                            for vec in range(d // 64):
                                off = vec * 64
                                v_vec = reconstruct_v(x_ub, x2_arg, residual_arg, r, off)
                                if has_residual_out:
                                    store_v_as_dtype(residual_out_ub, r, off, v_vec)
                                if is_rms_norm:
                                    acc = S.vadd(acc, S.vmul(v_vec, v_vec), vec_full)
                                else:
                                    acc = S.vadd(acc, v_vec, vec_full)
                            S.vsts(stats_ub[r, 0], S.vdupv(S.vcadd(acc), vec_full), dist='ONEPT_B32')
                    # Same-scope VST->VLD: drain before the stats loads below.
                    S.mem_bar('VST_VLD')
                    if not is_rms_norm:
                        for r in T.serial(block_m):
                            row = row_base + r
                            if row < total_rows:
                                mean = S.vmuls(S.vld(stats_ub[r, 0], dist='BRC_B32'), 1.0 / d)
                                acc_sq = S.alloc_var(T.float32)
                                acc_sq = zero
                                for vec in range(d // 64):
                                    off = vec * 64
                                    v_vec = reconstruct_v(x_ub, x2_arg, residual_arg, r, off)
                                    v_vec = S.vsub(v_vec, mean)
                                    acc_sq = S.vadd(acc_sq, S.vmul(v_vec, v_vec), vec_full)
                                S.vsts(stats_ub[r, 1], S.vdupv(S.vcadd(acc_sq), vec_full), dist='ONEPT_B32')
                        # Same-scope VST->VLD hazard as the pass 1 boundary.
                        S.mem_bar('VST_VLD')
                    for r in T.serial(block_m):
                        row = row_base + r
                        if row < total_rows:
                            if not is_rms_norm:
                                mean = S.vmuls(S.vld(stats_ub[r, 0], dist='BRC_B32'), 1.0 / d)
                                total_sq = S.vld(stats_ub[r, 1], dist='BRC_B32')
                            else:
                                total_sq = S.vld(stats_ub[r, 0], dist='BRC_B32')
                            rstd = S.vdiv(
                                S.vdup(1.0, T.float32, vec_full),
                                S.vsqrt(S.vadds(S.vmuls(total_sq, 1.0 / d), eps)),
                            )
                            for vec in range(d // 64):
                                off = vec * 64
                                v_vec = reconstruct_v(x_ub, x2_arg, residual_arg, r, off)
                                if not is_rms_norm:
                                    v_vec = S.vsub(v_vec, mean)
                                w_vec = load_affine_as_fp32(weight_ub, w32_arg, off)
                                y_vec = S.vmul(S.vmul(v_vec, rstd), S.vmuls(w_vec, out_scale))
                                if has_bias:
                                    b_vec = load_affine_as_fp32(bias_ub, b32_arg, off)
                                    y_vec = S.vadd(y_vec, b_vec)
                                if return_origin_out:
                                    store_v_as_dtype(y_origin_ub, r, off, y_vec)
                                if do_fp8_cast:
                                    store_v_as_dtype(norm_ub, r, off, y_vec)
                                elif return_dtype == 'bfloat16':
                                    S.vsts(y_ub[r, off], S.vcvt(y_vec, T.bfloat16), dist='PK_B32')
                                else:
                                    S.vsts(y_ub[r, off], y_vec)

                if do_fp8_cast:
                    with T.SimdVF():
                        if num_per_channels == 32:
                            low_mask = S.pset(32, 'PAT_VL32')
                            high_mask = S.vcmps(S.vci(0, T.int32), 31, op='gt')
                        for r in T.serial(block_m):
                            row = row_base + r
                            if row < total_rows:
                                if num_per_channels == 32:
                                    for vec in T.serial(d // 64):
                                        values = load_norm_as_fp32(norm_ub, r, vec * 64)
                                        abs_values = S.vabs(values)
                                        S.vsts(amax_ub[r, vec * 2], S.vcmax(abs_values, low_mask), dist='ONEPT_B32')
                                        S.vsts(
                                            amax_ub[r, vec * 2 + 1], S.vcmax(abs_values, high_mask), dist='ONEPT_B32'
                                        )
                                else:
                                    for group in T.serial(num_groups):
                                        values0 = load_norm_as_fp32(norm_ub, r, group * 128)
                                        values1 = load_norm_as_fp32(norm_ub, r, group * 128 + 64)
                                        S.vsts(
                                            amax_ub[r, group],
                                            S.vcmax(S.vmax(S.vabs(values0), S.vabs(values1))),
                                            dist='ONEPT_B32',
                                        )
                        # ONEPT amax stores must drain before the vld below.
                        S.mem_bar('VST_VLD')
                        for r in T.serial(block_m):
                            row = row_base + r
                            if row < total_rows:
                                for b in range(num_group_vectors):
                                    off = b * 64
                                    amax = S.vmaxs(S.vld(amax_ub[r, off]), 1e-4)
                                    if round_sf:
                                        bits = T.reinterpret(S.vmul(amax, S.vdup(1.0 / 448.0, T.float32)), 'uint32x64')
                                        exponent = S.vadds(S.vshrs(S.vsub(bits, S.vdup(1, T.uint32)), 23), 1)
                                        scale_inv = T.reinterpret(
                                            S.vshls(S.vsub(S.vdup(254, T.uint32), exponent), 23), 'float32x64'
                                        )
                                        if use_packed_ue8m0:
                                            S.vsts(sf_ub[r, off], exponent, dist='PK4_B32')
                                        else:
                                            scale = T.reinterpret(S.vshls(exponent, 23), 'float32x64')
                                            S.vsts(sf_ub[r, off], scale)
                                    else:
                                        scale = S.vdiv(amax, S.vdup(448.0, T.float32))
                                        scale_inv = S.vdiv(S.vdup(448.0, T.float32), amax)
                                        S.vsts(sf_ub[r, off], scale)
                                    S.vsts(sf_inv_ub[r, off], scale_inv)
                        # Same-scope VST->VLD: stale inverses without the drain.
                        S.mem_bar('VST_VLD')
                        for r in T.serial(block_m):
                            row = row_base + r
                            if row < total_rows:
                                if num_per_channels == 32:
                                    for vec in T.serial(d // 64):
                                        values = load_norm_as_fp32(norm_ub, r, vec * 64)
                                        inv0 = S.vld(sf_inv_ub[r, vec * 2], dist='BRC_B32')
                                        inv1 = S.vld(sf_inv_ub[r, vec * 2 + 1], dist='BRC_B32')
                                        inverse = S.vsel(inv0, inv1, low_mask)
                                        S.vsts(
                                            y_ub[r, vec * 64],
                                            S.vcvt(S.vmul(values, inverse), T.float8_e4m3fn),
                                            dist='PK4_B32',
                                        )
                                else:
                                    for group in T.serial(num_groups):
                                        values0 = load_norm_as_fp32(norm_ub, r, group * 128)
                                        values1 = load_norm_as_fp32(norm_ub, r, group * 128 + 64)
                                        inverse = S.vld(sf_inv_ub[r, group], dist='BRC_B32')
                                        S.vsts(
                                            y_ub[r, group * 128],
                                            S.vcvt(S.vmul(values0, inverse), T.float8_e4m3fn),
                                            dist='PK4_B32',
                                        )
                                        S.vsts(
                                            y_ub[r, group * 128 + 64],
                                            S.vcvt(S.vmul(values1, inverse), T.float8_e4m3fn),
                                            dist='PK4_B32',
                                        )

                if seq_id0 + block_m <= seqlen:
                    T.copy(y_ub[:block_m, :d], y[batch_id, seq_id0 : seq_id0 + block_m, :d])
                    if do_fp8_cast:
                        for r in T.serial(block_m):
                            store_scale_row(sf_ub, sf, r, batch_id, seq_id0 + r)
                    if has_residual_out:
                        T.copy(residual_out_ub[:block_m, :d], residual_out[batch_id, seq_id0 : seq_id0 + block_m, :d])
                    if return_origin_out:
                        T.copy(y_origin_ub[:block_m, :d], y_origin[batch_id, seq_id0 : seq_id0 + block_m, :d])
                else:
                    for r in T.serial(block_m):
                        row = row_base + r
                        if row < total_rows:
                            b_id = row // seqlen
                            s_id = row % seqlen
                            T.copy(y_ub[r, :d], y[b_id, s_id, :d])
                            if do_fp8_cast:
                                store_scale_row(sf_ub, sf, r, b_id, s_id)
                            if has_residual_out:
                                T.copy(residual_out_ub[r, :d], residual_out[b_id, s_id, :d])
                            if return_origin_out:
                                T.copy(y_origin_ub[r, :d], y_origin[b_id, s_id, :d])

    return _norm_fwd_ascend
