"""Improvement summary pytest plugin for ops-tilelang.

Computes per-kernel geometric-mean latency, bandwidth, and speedup
from benchmark results collected during the session, and prints
a summary table comparing ops-tilelang against legacy kernels.

This file is deliberately NOT named ``conftest.py`` -- it is loaded via
``pytest_plugins`` in the root ``conftest.py``.  A non-conftest name
prevents pluggy's duplicate-registration error.
"""

import json
import os
import statistics

import pytest

from tests.testing.bench import make_param_id
from tests.testing.stash import BENCHMARK_RESULTS

_IMPROVEMENT_SUMMARY_KEY = pytest.StashKey()


def pytest_addoption(parser):
    parser.addoption(
        '--legacy-table',
        action='store_true',
        default=False,
        help='Print a performance summary table comparing ops-tilelang vs legacy kernels',
    )


def _build_key(rec):
    kernel, operation = rec['kernel'], rec['operation']
    prefix = f'{rec["cluster"]}/{kernel}/{operation}'
    params = rec.get('params')
    if params:
        return f'{prefix}[{make_param_id(params)}]'
    return prefix


def _compute_improvement_summary(config):
    if not config.getoption('--legacy-table', default=False):
        return None

    results = config.stash.get(BENCHMARK_RESULTS, [])
    if not results:
        output_path = config.getoption('--benchmark-output', default=None)
        if output_path and os.path.exists(output_path):
            with open(output_path) as f:
                results = [json.loads(line) for line in f if line.strip()]
    if not results:
        return None

    deduped = {}
    for rec in results:
        key = _build_key(rec)
        if key in deduped:
            existing = deduped[key]
            if existing.get('extras', {}).get('speedup') is not None:
                continue
        deduped[key] = rec

    kernel_records = {}
    for rec in deduped.values():
        kernel = rec['kernel']
        kernel_records.setdefault(kernel, []).append(rec)

    summary = []
    for kernel in sorted(kernel_records.keys()):
        records = kernel_records[kernel]
        latencies = [r['time_us'] for r in records]
        bws = [r.get('bandwidth_gbs') for r in records if r.get('bandwidth_gbs') is not None]
        speedups = [
            r.get('extras', {}).get('speedup') for r in records if r.get('extras', {}).get('speedup') is not None
        ]

        if not latencies:
            continue

        avg_latency = statistics.geometric_mean(latencies)
        avg_bw = statistics.geometric_mean(bws) if bws else None
        avg_speedup = statistics.geometric_mean(speedups) if speedups else None

        summary.append(
            {
                'kernel': kernel,
                'latency_us': avg_latency,
                'bw_gbs': avg_bw,
                'speedup': avg_speedup,
            }
        )

    return summary


def pytest_sessionfinish(session, exitstatus):
    summary = _compute_improvement_summary(session.config)
    session.config.stash[_IMPROVEMENT_SUMMARY_KEY] = summary


def pytest_terminal_summary(terminalreporter, config):
    summary = config.stash.get(_IMPROVEMENT_SUMMARY_KEY, None)
    if summary is None:
        return

    tr = terminalreporter

    kw = max(max(len(s['kernel']) for s in summary), 20) + 2
    lat_w = 12
    bw_w = 10
    sp_w = 10
    total_w = kw + lat_w + bw_w + sp_w

    tr.write_line('=' * total_w)
    tr.write_line('  ops-tilelang vs legacy  --  Performance Summary')
    tr.write_line('=' * total_w)
    hdr = f'{"Kernel":<{kw}} {"Latency(us)":>{lat_w}} {"BW(GB/s)":>{bw_w}} {"Speedup":>{sp_w}}'
    tr.write_line(hdr)
    tr.write_line('-' * total_w)

    speedups_for_overall = []
    for s in summary:
        latency_str = f'{s["latency_us"]:.1f}'
        bw_str = f'{s["bw_gbs"]:.0f}' if s['bw_gbs'] is not None else '-'
        if s['speedup'] is not None:
            speedup_str = f'{s["speedup"]:.2f}x'
            speedups_for_overall.append(s['speedup'])
        else:
            speedup_str = 'N/A'
        line = f'{s["kernel"]:<{kw}} {latency_str:>{lat_w}} {bw_str:>{bw_w}} {speedup_str:>{sp_w}}'
        tr.write_line(line)

    tr.write_line('-' * total_w)
    if speedups_for_overall:
        overall = statistics.geometric_mean(speedups_for_overall)
        tr.write_line(f'  Geometric-mean speedup ({len(speedups_for_overall)} kernels): {overall:.2f}x')
    tr.write_line('')
