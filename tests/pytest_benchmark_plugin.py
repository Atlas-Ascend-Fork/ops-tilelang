"""Benchmark pytest plugin for ops-tilelang.

CLI options, markers, fixtures, and regression reporting for kernel
benchmarks.

This file is deliberately NOT named ``conftest.py`` — it is loaded via
``pytest_plugins`` in the root ``conftest.py``.  A non-conftest name
prevents pluggy's duplicate-registration error.
"""

import functools
import json
import math
import os
import re
import shlex
import tempfile
import threading
import warnings

import pytest
from tilelang.profiler.bench import do_bench

from tests.testing.bench import format_param_id, make_param_id
from tests.testing.stash import BENCHMARK_RESULTS

_KEY_FILTER_MATCHERS_KEY = pytest.StashKey()
_BENCHMARK_RESULTS_KEY = BENCHMARK_RESULTS
_BENCHMARK_RESULTS_LOCK_KEY = pytest.StashKey()
_BENCHMARK_DETECTION_KEY = pytest.StashKey()
_BENCHMARK_UPDATE_SUMMARY_KEY = pytest.StashKey()

_TESTS_DIR = os.path.dirname(__file__)

# Each test shard is owned by its operator directory:
# tests/<operator>/benchmark/<test_file>.jsonl.
_BENCHMARK_DIR_NAME = 'benchmark'
_BASELINES_LAYOUT = os.path.join(_TESTS_DIR, '<operator>', _BENCHMARK_DIR_NAME)

# Temporary file naming for auto-set --benchmark-output
_BENCHMARK_TMP_PREFIX = 'ops_tilelang_bench_'
_BENCHMARK_TMP_SUFFIX = '.jsonl'

# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        '--run-benchmark',
        action='store_true',
        default=False,
        help='Run benchmark tests (skipped by default)',
    )
    parser.addoption(
        '--benchmark-output',
        default=None,
        help='Path to write benchmark results as JSONL (one JSON object per line)',
    )
    parser.addoption(
        '--benchmark-regression-threshold',
        default=0.05,
        type=float,
        help='Fraction of slowdown that triggers a regression warning (default: 0.05 = 5%%)',
    )
    parser.addoption(
        '--benchmark-min-delta-us',
        default=0.8,
        type=float,
        help='Minimum absolute latency change to classify benchmark changes (default: 0.8 us)',
    )
    parser.addoption(
        '--benchmark-verbose',
        action='store_true',
        default=False,
        help='Show extras columns (e.g., speedup, …) in the benchmark regression report',
    )
    parser.addoption(
        '-K',
        '--key-filter',
        action='append',
        default=[],
        help='Run only benchmark node IDs matching this regex boolean expression. Can be passed multiple times.',
    )
    parser.addoption(
        '--refresh-baseline',
        action='store_true',
        default=False,
        help='Update tests/<operator>/benchmark/ shards from this benchmark run',
    )


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.stash[_KEY_FILTER_MATCHERS_KEY] = _compile_key_filters(config)

    config.addinivalue_line('markers', 'benchmark: mark test as benchmark (skip by default)')

    if config.getoption('--refresh-baseline', default=False) and not config.getoption('--run-benchmark', default=False):
        raise pytest.UsageError('--refresh-baseline requires --run-benchmark')

    # When --run-benchmark is used without an explicit --benchmark-output,
    # auto-create a shared temp file.  The master process creates it and
    # records the path in an env var; xdist workers inherit the env var
    # and reuse the same path, so all processes write to one file.
    if config.getoption('--run-benchmark', default=False):
        output_path = config.getoption('--benchmark-output', default=None)
        if output_path is None:
            path = os.environ.get('_OPS_TILELANG_BENCHMARK_OUTPUT')
            if path is None:
                fd, path = tempfile.mkstemp(
                    prefix=_BENCHMARK_TMP_PREFIX,
                    suffix=_BENCHMARK_TMP_SUFFIX,
                )
                os.close(fd)
                os.environ['_OPS_TILELANG_BENCHMARK_OUTPUT'] = path
            config.option.benchmark_output = path

    # Shared state for collecting benchmark results across this session
    config.stash[_BENCHMARK_RESULTS_KEY] = []
    config.stash[_BENCHMARK_RESULTS_LOCK_KEY] = threading.Lock()


def pytest_collection_modifyitems(config, items):
    matchers = config.stash.get(_KEY_FILTER_MATCHERS_KEY, [])
    if matchers:
        deselected = [item for item in items if not _matches_key_filter(matchers, item.nodeid)]
        if deselected:
            deselected_set = set(deselected)
            items[:] = [item for item in items if item not in deselected_set]
            config.hook.pytest_deselected(items=deselected)

    if not config.getoption('--run-benchmark'):
        # Without --run-benchmark, skip all benchmark tests
        skip_bench = pytest.mark.skip(reason='need --run-benchmark to run')
        for item in items:
            if 'benchmark' in item.keywords:
                item.add_marker(skip_bench)
    # With --run-benchmark, benchmark tests run alongside correctness tests
    # (e.g. `pytest kernel.py --run-benchmark`).
    # Use `-m benchmark` explicitly if you want ONLY benchmarks.


def _compile_key_filter(expression):
    tokens = _split_key_filter_expression(expression)
    if not tokens:
        raise ValueError('-K/--key-filter expression cannot be empty')
    parser = _ParamFilterParser(tokens, expression)
    matcher = parser.parse()
    if parser.has_tokens():
        raise ValueError(f'Unexpected token in -K/--key-filter expression: {parser.peek()!r}')
    return matcher


def _compile_key_filters(config):
    key_filters = config.getoption('--key-filter')
    if not key_filters:
        return []
    try:
        return [_compile_key_filter(key_filter) for key_filter in key_filters]
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc


def _matches_key_filter(matchers, text):
    return not matchers or any(matcher(text) for matcher in matchers)


def _split_key_filter_expression(expression):
    lexer = shlex.shlex(expression, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = ''
    return [_strip_key_filter_quotes(token) for token in lexer]


def _strip_key_filter_quotes(token):
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


class _ParamFilterParser:
    def __init__(self, tokens, expression):
        self.tokens = tokens
        self.expression = expression
        self.pos = 0

    def has_tokens(self):
        return self.pos < len(self.tokens)

    def peek(self):
        if self.has_tokens():
            return self.tokens[self.pos]
        return None

    def take(self):
        token = self.peek()
        self.pos += 1
        return token

    def parse(self):
        return self.parse_or()

    def parse_or(self):
        matcher = self.parse_and()
        while self.peek() == 'or':
            self.take()
            rhs = self.parse_and()
            lhs = matcher

            def matcher(text, lhs=lhs, rhs=rhs):
                return lhs(text) or rhs(text)

        return matcher

    def parse_and(self):
        matcher = self.parse_not()
        while self.peek() == 'and' or self.starts_atom(self.peek()):
            if self.peek() == 'and':
                self.take()
            rhs = self.parse_not()
            lhs = matcher

            def matcher(text, lhs=lhs, rhs=rhs):
                return lhs(text) and rhs(text)

        return matcher

    def starts_atom(self, token):
        return token is not None and token != 'or'

    def parse_not(self):
        if self.peek() == 'not':
            self.take()
            matcher = self.parse_not()
            return lambda text, matcher=matcher: not matcher(text)
        return self.parse_atom()

    def parse_atom(self):
        token = self.peek()
        if token is None:
            raise ValueError(f'Unexpected end of -K/--key-filter expression: {self.expression!r}')
        if token in ('and', 'or'):
            raise ValueError(f'Expected regex before {token!r} in -K/--key-filter expression')
        token = self.take()
        try:
            pattern = re.compile(token)
        except re.error as exc:
            raise ValueError(f'Invalid regex in -K/--key-filter expression {token!r}: {exc}') from exc
        return lambda text, pattern=pattern: pattern.search(text) is not None


# ---------------------------------------------------------------------------
# Regression detection & exit code
# ---------------------------------------------------------------------------


def _detect_regressions(config):
    """Check benchmark results against baselines and return regressions.

    Returns:
        A tuple ``(results, baselines, regressions, improvements, missing)``
        or ``None`` if no results were collected.
    """
    results = config.stash.get(_BENCHMARK_RESULTS_KEY, [])
    if not results:
        output_path = config.getoption('--benchmark-output', default=None)
        if output_path and os.path.exists(output_path):
            with open(output_path) as f:
                results = [json.loads(line) for line in f if line.strip()]
    if not results:
        return None

    threshold = config.getoption('--benchmark-regression-threshold')
    min_delta_us = config.getoption('--benchmark-min-delta-us')
    baselines = _load_baselines()

    regressions = []
    improvements = []
    missing = []

    for rec in results:
        key = _make_key(rec)
        if key not in baselines:
            missing.append((key, rec['time_us']))
            continue
        baseline_us = baselines[key]['time_us']
        current_us = rec['time_us']
        ratio = current_us / baseline_us
        delta_us = current_us - baseline_us
        if ratio > 1.0 + threshold and delta_us >= min_delta_us:
            regressions.append((key, baseline_us, current_us, ratio))
        elif ratio < 1.0 - threshold and -delta_us >= min_delta_us:
            improvements.append((key, baseline_us, current_us, ratio))

    return results, baselines, regressions, improvements, missing


def _update_baselines(config, new_records):
    """Update benchmark baselines from records collected in this pytest run."""
    if not config.getoption('--refresh-baseline'):
        return None
    if not new_records:
        return None

    baseline_shards = _load_baseline_shards()
    num_updated = 0
    num_added = 0
    touched_shards = set()
    for record in new_records:
        key = _make_key(record)
        shard_path = record.get('source_file')
        if shard_path is None:
            continue
        touched_shards.add(shard_path)

        if key in baseline_shards.get(shard_path, {}):
            num_updated += 1
        else:
            num_added += 1

        baseline_shards.setdefault(shard_path, {})[key] = record

    _write_baseline_shards(baseline_shards, touched_shards)
    baselines = _flatten_baseline_shards(baseline_shards)
    return {
        'num_records': len(baselines),
        'num_updated': num_updated,
        'num_added': num_added,
        'num_shards': len(touched_shards),
        'path': _BASELINES_LAYOUT,
    }


def pytest_sessionfinish(session, exitstatus):
    """Set non-zero exit code when benchmark regressions are detected.

    Runs before ``pytest_terminal_summary``, so regression detection is
    performed here and stashed on ``config`` for the terminal report.
    """
    config = session.config
    if hasattr(config, 'workerinput'):
        return

    result = _detect_regressions(config)
    if result is None:
        return
    results, baselines, regressions, improvements, missing = result
    # Stash for pytest_terminal_summary
    config.stash[_BENCHMARK_DETECTION_KEY] = result

    update_summary = _update_baselines(config, results) if exitstatus == 0 else None
    if update_summary is not None:
        config.stash[_BENCHMARK_UPDATE_SUMMARY_KEY] = update_summary

    is_refreshing_baseline = config.getoption('--refresh-baseline')
    if regressions and exitstatus == 0 and not is_refreshing_baseline:
        session.exitstatus = 1


# ---------------------------------------------------------------------------
# Terminal summary: regression report
# ---------------------------------------------------------------------------


def pytest_terminal_summary(terminalreporter, config):
    """Print a benchmark regression report at the end of the pytest session."""
    # Use pre-computed results from pytest_sessionfinish if available,
    # otherwise compute now
    detection = config.stash.get(_BENCHMARK_DETECTION_KEY, None)
    if detection is None:
        detection = _detect_regressions(config)
    if detection is None:
        return

    results, baselines, regressions, improvements, missing = detection
    threshold = config.getoption('--benchmark-regression-threshold')
    min_delta_us = config.getoption('--benchmark-min-delta-us')
    verbose = config.getoption('--benchmark-verbose')

    tr = terminalreporter

    update_summary = config.stash.get(_BENCHMARK_UPDATE_SUMMARY_KEY, None)
    if update_summary is not None:
        tr.section('Benchmark Baseline Update')
        tr.write_line(f'Output: {update_summary["path"]}')
        tr.write_line(
            f'Mode: merge, {update_summary["num_updated"]} updated, '
            f'{update_summary["num_added"]} new, '
            f'{update_summary["num_records"]} total benchmarks, '
            f'{update_summary["num_shards"]} shard(s) touched'
        )
        tr.write_line('')

    tr.section('Benchmark Regression Report')

    if baselines:
        # Collect extras column names when verbose
        extras_keys = []
        if verbose:
            extras_keys = _collect_extras_keys(results, baselines)

        matched_records = sorted(
            (r for r in results if _make_key(r) in baselines),
            key=_benchmark_sort_key,
        )

        # Compute dynamic Kernel column width
        matched_keys = [_make_display_key(r) for r in matched_records]
        kw = max((len(k) for k in matched_keys), default=20) + 2

        # Extras column widths: fit header label or widest value
        ek_widths = {}
        for ek in extras_keys:
            cur_label = ek + '(cur)'
            ref_label = ek + '(ref)'
            w = max(len(cur_label), len(ref_label), 8)
            for rec in results:
                rk = _make_key(rec)
                if rk not in baselines:
                    continue
                for src in (rec, baselines[rk]):
                    v = (src.get('extras') or {}).get(ek)
                    w = max(w, len(_fmt_extra(v)))
            ek_widths[ek] = w

        # Header
        hdr = f'{"Kernel":<{kw}} {"Latency":>11} {"Bandwidth":>11} {"Ratio":>8} {"Stat":>4}'
        for ek in extras_keys:
            w = ek_widths[ek]
            hdr += f'  {(ek + "(cur)"):>{w}}  {(ek + "(ref)"):>{w}}'
        tr.write_line(hdr)
        tr.write_line('-' * len(hdr))

        for rec in matched_records:
            key = _make_key(rec)
            display_key = _make_display_key(rec)
            baseline_rec = baselines[key]
            baseline_us = baseline_rec['time_us']
            current_us = rec['time_us']
            ratio = current_us / baseline_us
            delta_us = current_us - baseline_us
            if ratio > 1.0 + threshold and delta_us >= min_delta_us:
                status = '--'
            elif ratio < 1.0 - threshold and -delta_us >= min_delta_us:
                status = '++'
            else:
                status = '='
            cur_bw = rec.get('bandwidth_gbs')
            line = f'{display_key:<{kw}} {current_us:>8.1f} us {_fmt_bw(cur_bw):>11} {ratio:>7.2f}x {status:>4}'
            for ek in extras_keys:
                w = ek_widths[ek]
                cur_v = (rec.get('extras') or {}).get(ek)
                ref_v = (baseline_rec.get('extras') or {}).get(ek)
                line += f'  {_fmt_extra(cur_v):>{w}}  {_fmt_extra(ref_v):>{w}}'
            tr.write_line(line)

        geomean_latency = _geomean(r['time_us'] for r in matched_records)
        geomean_bandwidth = _geomean(r.get('bandwidth_gbs') for r in matched_records)
        geomean_ratio = _geomean(r['time_us'] / baselines[_make_key(r)]['time_us'] for r in matched_records)
        if geomean_ratio is None:
            geomean_status = '-'
        elif geomean_ratio > 1.0 + threshold:
            geomean_status = '--'
        elif geomean_ratio < 1.0 - threshold:
            geomean_status = '++'
        else:
            geomean_status = '='
        tr.write_line('-' * len(hdr))
        tr.write_line(
            f'{"GEOMEAN":<{kw}} {_fmt_latency(geomean_latency):>11} '
            f'{_fmt_bw(geomean_bandwidth):>11} {_fmt_ratio(geomean_ratio):>8} '
            f'{geomean_status:>4}'
        )
    else:
        tr.write_line('No baseline file found — skipping regression comparison.')
        tr.write_line(f'  (looked at: {_BASELINES_LAYOUT})')

    # New benchmarks without baselines
    if missing:
        new_recs = sorted(
            (r for r in results if _make_key(r) not in baselines),
            key=_benchmark_sort_key,
        )
        tr.write_line('')
        tr.write_line('!! MISSING BASELINES — run refresh script before merging !!')  # oss-ignore-line

        # Dynamic column widths
        new_keys = [_make_display_key(r) for r in new_recs]
        nkw = max((len(k) for k in new_keys), default=20) + 2

        # Bandwidth column width for new-benchmarks table
        new_bw_col_w = 9
        for r in new_recs:
            v = r.get('bandwidth_gbs', None)
            new_bw_col_w = max(new_bw_col_w, len(_fmt_bw(v)))

        new_extras_keys = []
        new_ek_widths = {}
        if verbose:
            ek_set = set()
            for r in new_recs:
                ek_set.update((r.get('extras') or {}).keys())
            new_extras_keys = sorted(ek_set)
            for ek in new_extras_keys:
                w = len(ek)
                for r in new_recs:
                    v = (r.get('extras') or {}).get(ek)
                    w = max(w, len(_fmt_extra(v)))
                new_ek_widths[ek] = max(w, 8)

        # Header
        nhdr = f'{"Kernel":<{nkw}} {"Current":>11}  {"Bandwidth":>{new_bw_col_w}}'
        for ek in new_extras_keys:
            nhdr += f'  {ek:>{new_ek_widths[ek]}}'
        tr.write_line(nhdr)
        tr.write_line('-' * len(nhdr))

        for r in new_recs:
            key = _make_display_key(r)
            bw = r.get('bandwidth_gbs', None)
            line = f'{key:<{nkw}} {r["time_us"]:>8.1f} us  {_fmt_bw(bw):>{new_bw_col_w}}'
            for ek in new_extras_keys:
                w = new_ek_widths[ek]
                v = (r.get('extras') or {}).get(ek)
                line += f'  {_fmt_extra(v):>{w}}'
            tr.write_line(line)

        geomean_latency = _geomean(r['time_us'] for r in new_recs)
        geomean_bandwidth = _geomean(r.get('bandwidth_gbs') for r in new_recs)
        tr.write_line('-' * len(nhdr))
        tr.write_line(
            f'{"GEOMEAN":<{nkw}} {_fmt_latency(geomean_latency):>11}  {_fmt_bw(geomean_bandwidth):>{new_bw_col_w}}'
        )

    # Summary
    matched = sum(1 for r in results if baselines and _make_key(r) in baselines)
    tr.write_line('')
    tr.write_line(
        f'Total: {len(results)} benchmarks, {matched} with baselines, '
        f'{len(missing)} missing, '
        f'{len(regressions)} regressions, {len(improvements)} improvements '
        f'(threshold: {threshold:.0%})'
    )

    if improvements:
        tr.write_line('')
        tr.write_line('!! IMPROVEMENTS DETECTED !!')
        for key, baseline_us, current_us, ratio in improvements:
            tr.write_line(f'  {key}: {current_us:.1f} us vs baseline {baseline_us:.1f} us ({1.0 / ratio:.2f}x faster)')

    if regressions:
        tr.write_line('')
        tr.write_line('!! REGRESSIONS DETECTED !!')
        for key, baseline_us, current_us, ratio in regressions:
            tr.write_line(f'  {key}: {current_us:.1f} us vs baseline {baseline_us:.1f} us ({ratio:.2f}x slower)')


def _fmt_extra(v):
    """Format an extras value for display."""
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.2f}'
    return str(v)


def _benchmark_sort_key(record):
    """Sort benchmarks by kernel, operation, then parameters.

    Parameters are compared name by name (sorted), with numeric values
    ordered numerically (e.g. num_tokens=128 before num_tokens=1024) and
    other values by string.  No parameter names are hardcoded, so the
    ordering works for any kernel's parametrization.
    """
    params = record.get('params') or {}
    param_key = tuple((name, *_sort_value(params[name])) for name in sorted(params))
    return (
        record.get('kernel', ''),
        record.get('operation', ''),
        param_key,
        _make_display_key(record),
    )


def _sort_value(value):
    # Rank numbers before other types so mixed int/str values for the same
    # parameter name compare without TypeError.
    if isinstance(value, bool) or not isinstance(value, int | float):
        return (1, str(value))
    return (0, value)


def _fmt_bw(v):
    """Format a bandwidth_gbs value for display (e.g. '1234.56 GB/s')."""
    if v is None:
        return '-'
    return f'{v:6.1f} GB/s'


def _fmt_latency(v):
    """Format a latency value in microseconds for display."""
    if v is None:
        return '-'
    return f'{v:8.1f} us'


def _fmt_ratio(v):
    """Format a benchmark ratio for display."""
    if v is None:
        return '-'
    return f'{v:7.2f}x'


def _geomean(values):
    """Return the geometric mean of positive values, ignoring missing values."""
    logs = []
    for value in values:
        if value is None or value <= 0:
            continue
        logs.append(math.log(value))
    if not logs:
        return None
    return math.exp(sum(logs) / len(logs))


def _collect_extras_keys(results, baselines):
    """Return a sorted list of extras keys across results and baselines,
    excluding bandwidth_gbs (reported as a dedicated column)."""
    keys = set()
    for rec in results:
        key = _make_key(rec)
        if key not in baselines:
            continue
        for e in (rec.get('extras') or {}, (baselines[key].get('extras') or {})):
            keys.update(e.keys())
    return sorted(keys)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Lock for concurrent JSONL writes from xdist workers
_jsonl_write_lock = threading.Lock()


@pytest.fixture
def benchmark_record(request):
    """Record a benchmark result for regression tracking.

    Prints a human-readable summary, appends a JSONL record to
    ``--benchmark-output`` (if given), collects the result for the terminal
    regression report, and emits a pytest warning on regressions.

    JSONL schema::

        {
            "kernel":         str,
            "operation":      str,
            "params":         dict,
            "param_id":       str | None,   # pytest parametrize ID
            "cluster":        str,          # baseline cluster target ("a5")
            "source_file":    str | None,   # tests/-relative test file (shard key)
            "time_us":        float,
            "extra_params":   dict,         # optional; params not in the node ID
            "bandwidth_gbs":  float,        # optional
            "extras":         dict,         # optional
        }
    """
    output_path = request.config.getoption('--benchmark-output')
    # oss-ignore-begin
    threshold = request.config.getoption('--benchmark-regression-threshold')
    min_delta_us = request.config.getoption('--benchmark-min-delta-us')
    baselines = _load_baselines()
    # oss-ignore-end

    def _record(*, kernel, operation, params, time_us, bandwidth_gbs=None, extras=None):
        # Write JSONL
        param_id = _node_param_id(request.node)
        record = {
            'kernel': kernel,
            'operation': operation,
            'params': dict(sorted(params.items())) if params else params,
            'param_id': param_id,
            'cluster': _get_cluster(),
            'source_file': _test_source_file(request.node),
            'time_us': round(time_us, 2),
        }
        extra_params = _extra_params(request.node, params)
        if extra_params:
            record['extra_params'] = extra_params
        if bandwidth_gbs is not None:
            record['bandwidth_gbs'] = round(bandwidth_gbs, 4)
        if extras:
            record['extras'] = {k: round(v, 4) if isinstance(v, float) else v for k, v in extras.items()}
        key = _make_key(record)
        display_key = _make_display_key(record)

        # Human-readable print
        parts = [f'  BENCH {display_key}: {time_us:.1f} us']
        if bandwidth_gbs is not None:
            parts.append(f', bandwidth_gbs={bandwidth_gbs:.2f}')
        if extras:
            for ek, ev in extras.items():
                if isinstance(ev, float):
                    parts.append(f', {ek}={ev:.2f}')
                else:
                    parts.append(f', {ek}={ev}')
        print(''.join(parts))

        if output_path:
            line = json.dumps(record, ensure_ascii=False)
            with _jsonl_write_lock, open(output_path, 'a') as f:
                f.write(line + '\n')

        # Collect for terminal summary
        with request.config.stash[_BENCHMARK_RESULTS_LOCK_KEY]:
            request.config.stash[_BENCHMARK_RESULTS_KEY].append(record)

        # oss-ignore-begin
        # Per-test regression warning
        if baselines and key in baselines:
            baseline_us = baselines[key]['time_us']
            ratio = time_us / baseline_us
            delta_us = time_us - baseline_us
            if ratio > 1.0 + threshold and delta_us >= min_delta_us:
                warnings.warn(
                    f'PERFORMANCE REGRESSION: {key} is {ratio:.2f}x slower than baseline '
                    f'({time_us:.1f} us vs {baseline_us:.1f} us, threshold={threshold:.0%})',
                    stacklevel=2,
                )
        # oss-ignore-end

    return _record


@pytest.fixture
def benchmark_timer():
    """Return a callable that measures kernel execution time in microseconds.

    Wraps ``tilelang.profiler.bench.do_bench`` with the Ascend msprof backend.
    Keyword arguments are forwarded to ``do_bench``, allowing per-test
    overrides (e.g. ``benchmark_timer(fn, rep=30)``).

    Returns:
        A callable ``(fn, **overrides) -> float`` returning time in
        microseconds.
    """

    def _timer(fn, **overrides):
        kwargs = dict(backend='msprof', warmup=0, rep=30)
        kwargs.update(overrides)
        return do_bench(fn, **kwargs) * 1e3  # ms → us

    return _timer


def _make_key(rec):
    """Build the benchmark baseline key from a benchmark record."""
    kernel, operation = rec['kernel'], rec['operation']
    prefix = f'{rec["cluster"]}/{kernel}/{operation}'
    return f'{prefix}[{rec["param_id"]}]'


def _make_display_key(rec):
    kernel, operation = rec['kernel'], rec['operation']
    param_id = rec.get('param_id')
    key = f'{kernel}/{operation}[{format_param_id(param_id, display=True)}]' if param_id else _make_key(rec)
    extra_params = rec.get('extra_params')
    if extra_params:
        extra_param_id = make_param_id(extra_params, display=True).rstrip().rstrip(',')
        return f'{key} {{{extra_param_id}}}'
    return key


def _extra_params(node, params):
    node_params = _node_params(node)
    if not node_params or not params:
        return None
    extra_params = {key: value for key, value in params.items() if key not in node_params or node_params[key] != value}
    return extra_params or None


def _node_params(node):
    callspec = getattr(node, 'callspec', None)
    if callspec is None:
        return None
    params = callspec.params.get('params')
    if isinstance(params, dict):
        return params
    return None


def _node_param_id(node):
    callspec = getattr(node, 'callspec', None)
    if callspec is not None:
        return callspec.id
    return None


def _test_source_file(item):
    """Return the tests/-relative source file for a pytest item."""
    path = getattr(item, 'path', None)
    if path is None:
        path = getattr(item, 'fspath', None)
    if path is None:
        return None
    return _source_file_from_path(str(path))


def _source_file_from_path(path):
    relpath = os.path.relpath(os.path.abspath(path), _TESTS_DIR)
    if relpath.startswith(os.pardir + os.sep) or relpath == os.pardir:
        return None
    return relpath


def _get_cluster():
    """Return the Ascend A5 benchmark baseline cluster target."""

    return 'a5'


@functools.lru_cache(maxsize=1)
def _load_baselines():
    """Load sharded benchmark baselines into a ``{key: record}`` dict."""
    return _flatten_baseline_shards(_load_baseline_shards())


def _load_baseline_shards():
    """Load benchmark baseline shards into ``{source_file: {key: record}}``."""
    baseline_shards = {}
    for dirpath, dirnames, filenames in os.walk(_TESTS_DIR):
        dirnames[:] = sorted(name for name in dirnames if name != '__pycache__')
        rel_dir = os.path.relpath(dirpath, _TESTS_DIR)
        dir_parts = [] if rel_dir == os.curdir else rel_dir.split(os.sep)
        if _BENCHMARK_DIR_NAME not in dir_parts:
            continue

        marker_index = dir_parts.index(_BENCHMARK_DIR_NAME)
        source_dir_parts = dir_parts[:marker_index] + dir_parts[marker_index + 1 :]
        for filename in sorted(filenames):
            if not filename.endswith('.jsonl'):
                continue
            source_filename = filename[: -len('.jsonl')] + '.py'
            shard_path = os.path.join(*source_dir_parts, source_filename)
            path = os.path.join(dirpath, filename)
            baseline_shards[shard_path] = _load_baseline_records(path, shard_path)

    return baseline_shards


def _load_baseline_records(path, default_source_file):
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record['source_file'] = default_source_file
            records[_make_key(record)] = record
    return records


def _flatten_baseline_shards(baseline_shards):
    baselines = {}
    for shard_baselines in baseline_shards.values():
        baselines.update(shard_baselines)
    return baselines


def _write_baseline_shards(baseline_shards, touched_shards):
    """Write touched benchmark baseline shards as sorted JSONL records."""
    for shard_path in touched_shards:
        records = list(baseline_shards.get(shard_path, {}).values())
        path = _baseline_shard_path(shard_path)
        if not records:
            if os.path.exists(path):
                os.remove(path)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _write_baseline_records(path, records)


def _write_baseline_records(path, records):
    for record in records:
        record['time_us'] = round(record['time_us'], 2)

    records.sort(key=_make_key)

    with open(path, 'w') as f:
        for record in records:
            serialized_record = dict(record)
            serialized_record.pop('source_file', None)
            f.write(json.dumps(serialized_record, ensure_ascii=False) + '\n')


def _baseline_shard_path(source_file):
    parts = [part for part in source_file.replace('\\', '/').split('/') if part not in ('', '.')]
    if not parts or '..' in parts:
        raise ValueError(f'Invalid tests-relative benchmark source file: {source_file!r}')

    # Insert the artifact directory directly below the owning operator. Root
    # tests (which normally do not benchmark kernels) fall back to
    # tests/benchmark/.
    artifact_parts = (
        parts[:1] + [_BENCHMARK_DIR_NAME] + parts[1:]
        if len(parts) > 1
        else [
            _BENCHMARK_DIR_NAME,
            *parts,
        ]
    )
    path = os.path.join(_TESTS_DIR, *artifact_parts)
    if path.endswith('.py'):
        return path[: -len('.py')] + '.jsonl'
    return path
