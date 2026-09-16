"""NPU memory profiling pytest plugin for ops-tilelang."""

import gc
import json
import logging
import os
import signal
import tempfile
import threading

import pytest
import torch

_LOGGER = logging.getLogger(__name__)

_TESTS_DIR = os.path.dirname(__file__)
_NPU_MEM_DIR_NAME = 'npu_mem_profiles'
_NPU_MEM_PROFILES_LAYOUT = os.path.join(_TESTS_DIR, '<operator>', _NPU_MEM_DIR_NAME)
_NPU_MEM_TMP_ENV = '_OPS_TILELANG_NPU_MEM_OUTPUT'
_OPS_TILELANG_PREFIX = os.path.join('tests', '')
_npu_mem_profile_update_lock = threading.Lock()
_previous_sigterm_handler = None

_NPU_MEM_MISSING_ONLY_DESELECTED_KEY = pytest.StashKey()
_NPU_MEM_NUM_UNKNOWN_KEY = pytest.StashKey()
_NPU_MEM_UPDATE_SUMMARY_KEY = pytest.StashKey()
_NPU_MEM_FLUSH_STOP_EVENT_KEY = pytest.StashKey()
_NPU_MEM_FLUSH_THREAD_KEY = pytest.StashKey()
_NPU_MEM_RECORD_KEY = pytest.StashKey()
_NPU_MEM_TEST_FAILED_KEY = pytest.StashKey()


def pytest_addoption(parser):
    parser.addoption(
        '--npu-mem-output',
        default=None,
        help='Path to write per-test NPU peak memory as JSONL (for refreshing tests/<operator>/npu_mem_profiles/)',
    )
    parser.addoption(
        '--refresh-mem',
        action='store_true',
        default=False,
        help='Refresh tests/<operator>/npu_mem_profiles/ shards from NPU memory records collected in this pytest run',
    )
    parser.addoption(
        '--refresh-mem-missing-only',
        action='store_true',
        default=False,
        help='With --refresh-mem, only run tests missing from tests/<operator>/npu_mem_profiles/',
    )
    parser.addoption(
        '--npu-mem-threshold',
        default=15000.0,
        type=float,
        help=(
            'Peak NPU memory threshold (MB). Tests above this run in Stage 2 '
            'with fewer workers (default: 15000 = ~15 GB)'
        ),
    )
    parser.addoption(
        '--npu-mem-flush-interval',
        default=300.0,
        type=float,
        help='Seconds between periodic NPU memory profile flushes during --refresh-mem (default: 300)',
    )


def pytest_configure(config):
    global _previous_sigterm_handler

    config.addinivalue_line(
        'markers', 'large_npu_mem: mark test as requiring large NPU memory (run separately with fewer workers)'
    )

    if config.getoption('--refresh-mem-missing-only', default=False) and not config.getoption(
        '--refresh-mem', default=False
    ):
        raise pytest.UsageError('--refresh-mem-missing-only requires --refresh-mem')

    if not config.getoption('--refresh-mem', default=False):
        return
    if config.getoption('--npu-mem-output', default=None) is not None:
        return

    # Share one JSONL path with xdist workers through the environment.
    path = os.environ.get(_NPU_MEM_TMP_ENV)
    if path is None:
        fd, path = tempfile.mkstemp(prefix='npu_mem_', suffix='.jsonl')
        os.close(fd)
        os.environ[_NPU_MEM_TMP_ENV] = path
    config.option.npu_mem_output = path

    if not hasattr(config, 'workerinput') and _previous_sigterm_handler is None:
        _previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _make_sigterm_handler(config))
        _start_periodic_npu_mem_flush(config)


def pytest_collection_modifyitems(config, items):
    profiles = _load_npu_mem_profiles()

    if config.getoption('--refresh-mem-missing-only', default=False):
        selected = []
        deselected = []
        for item in items:
            (deselected if _make_mem_key(item.nodeid) in profiles else selected).append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = selected
            config.stash[_NPU_MEM_MISSING_ONLY_DESELECTED_KEY] = len(deselected)

    threshold = config.getoption('--npu-mem-threshold')
    num_unknown = 0
    for item in items:
        if 'large_npu_mem' in item.keywords:
            continue
        profile = profiles.get(_make_mem_key(item.nodeid))
        if profile is None:
            item.add_marker(pytest.mark.large_npu_mem)
            num_unknown += 1
        elif profile['peak_npu_mb'] > threshold:
            item.add_marker(pytest.mark.large_npu_mem)
    config.stash[_NPU_MEM_NUM_UNKNOWN_KEY] = num_unknown


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if hasattr(config, 'workerinput'):
        return

    npu_mem_update_summary = _update_npu_mem_profiles(config)
    if npu_mem_update_summary is not None:
        config.stash[_NPU_MEM_UPDATE_SUMMARY_KEY] = npu_mem_update_summary

    _stop_periodic_npu_mem_flush(config)
    _restore_sigterm_handler()


def pytest_terminal_summary(terminalreporter, config):
    _print_npu_mem_summary(terminalreporter, config)


def _load_npu_mem_profiles():
    profiles = {}
    for dirpath, dirnames, filenames in os.walk(_TESTS_DIR):
        dirnames[:] = sorted(name for name in dirnames if name != '__pycache__')
        rel_dir = os.path.relpath(dirpath, _TESTS_DIR)
        dir_parts = [] if rel_dir == os.curdir else rel_dir.split(os.sep)
        if _NPU_MEM_DIR_NAME not in dir_parts:
            continue
        for filename in sorted(filenames):
            if filename.endswith('.json'):
                with open(os.path.join(dirpath, filename)) as f:
                    _merge_npu_mem_profiles(profiles, json.load(f))
    return profiles


def _load_npu_mem_records(config):
    output_path = config.getoption('--npu-mem-output', default=None)
    if not output_path or not os.path.exists(output_path):
        return []
    records = []
    with open(output_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _update_npu_mem_profiles(config, blocking=True):
    if not config.getoption('--refresh-mem', default=False):
        return None

    # The SIGTERM handler runs on the main thread and may interrupt code that
    # already holds this non-reentrant lock (e.g. pytest_sessionfinish writing
    # shards).  A blocking acquire would self-deadlock there, so the handler
    # passes blocking=False and skips the merge; this is safe because the
    # current lock holder is already performing the same merge.
    if not _npu_mem_profile_update_lock.acquire(blocking=blocking):
        return None
    try:
        return _update_npu_mem_profiles_unlocked(config)
    finally:
        _npu_mem_profile_update_lock.release()


def _update_npu_mem_profiles_unlocked(config):
    records = _load_npu_mem_records(config)
    if not records:
        return {
            'updated': False,
            'reason': 'no NPU memory records were collected',
            'path': _NPU_MEM_PROFILES_LAYOUT,
        }

    # Merge with max everywhere: records from different num_sms variants share
    # one normalized key, and xdist record order is arbitrary, so last-write-
    # wins could let a low-num_sms (smaller-peak) record overwrite a larger
    # one and misclassify the test into Stage 1.
    new_profiles = {}
    for record in records:
        _merge_npu_mem_profile(new_profiles, record['test_id'], {'peak_npu_mb': record['peak_npu_mb']})

    profiles = _load_npu_mem_profiles()
    num_updated = sum(1 for test_id in new_profiles if test_id in profiles)
    for test_id, value in new_profiles.items():
        _merge_npu_mem_profile(profiles, test_id, value)

    touched_shards = {_npu_mem_source_file_from_test_id(test_id) for test_id in new_profiles}
    touched_shards.discard(None)
    _write_npu_mem_profile_shards(dict(sorted(profiles.items())), touched_shards)

    threshold = config.getoption('--npu-mem-threshold')
    num_large = sum(1 for value in profiles.values() if value['peak_npu_mb'] > threshold)
    return {
        'updated': True,
        'num_profiles': len(profiles),
        'num_collected': len(new_profiles),
        'num_updated': num_updated,
        'num_added': len(new_profiles) - num_updated,
        'num_large': num_large,
        'num_small': len(profiles) - num_large,
        'threshold': threshold,
        'path': _NPU_MEM_PROFILES_LAYOUT,
    }


def _start_periodic_npu_mem_flush(config):
    interval = config.getoption('--npu-mem-flush-interval')
    if interval <= 0:
        return

    stop_event = threading.Event()
    config.stash[_NPU_MEM_FLUSH_STOP_EVENT_KEY] = stop_event

    def _flush_periodically():
        while not stop_event.wait(interval):
            _update_npu_mem_profiles(config)

    thread = threading.Thread(target=_flush_periodically, name='npu-mem-profile-flush', daemon=True)
    config.stash[_NPU_MEM_FLUSH_THREAD_KEY] = thread
    thread.start()


def _stop_periodic_npu_mem_flush(config):
    stop_event = config.stash.get(_NPU_MEM_FLUSH_STOP_EVENT_KEY, None)
    if stop_event is not None:
        stop_event.set()
    thread = config.stash.get(_NPU_MEM_FLUSH_THREAD_KEY, None)
    if thread is not None:
        thread.join(timeout=1.0)


def _make_sigterm_handler(config):
    def _handle_sigterm(signum, frame):
        # blocking=False: this handler runs on the main thread and must not
        # block on the update lock if the interrupted code already holds it.
        update_summary = _update_npu_mem_profiles(config, blocking=False)
        if update_summary is not None:
            lines = _format_update_summary_lines(update_summary)
            _LOGGER.warning(
                '\nNPU memory profile refresh on SIGTERM:\n%s',
                '\n'.join(lines),
            )
        _restore_sigterm_handler()
        # The previous handler is intentionally not invoked: it is normally
        # SIG_DFL (not callable), and chaining a custom Python handler here
        # could raise twice.  Translate SIGTERM into KeyboardInterrupt so
        # pytest unwinds like on Ctrl+C.
        raise KeyboardInterrupt

    return _handle_sigterm


def _format_update_summary_lines(update_summary):
    """Format a refresh summary as lines shared by terminal and SIGTERM output."""
    if not update_summary['updated']:
        return [
            f'Skipped: {update_summary["reason"]}',
            f'Output: {update_summary["path"]}',
        ]
    return [
        f'Output: {update_summary["path"]}',
        f'Mode: merge, {update_summary["num_collected"]} collected, '
        f'{update_summary["num_updated"]} updated, {update_summary["num_added"]} new, '
        f'{update_summary["num_profiles"]} total profiles',
    ]


def _restore_sigterm_handler():
    global _previous_sigterm_handler

    if _previous_sigterm_handler is None:
        return
    signal.signal(signal.SIGTERM, _previous_sigterm_handler)
    _previous_sigterm_handler = None


def _make_mem_key(nodeid: str) -> str:
    idx = nodeid.find(_OPS_TILELANG_PREFIX)
    if idx != -1:
        nodeid = nodeid[idx + len(_OPS_TILELANG_PREFIX) :]
    return _normalize_mem_key(nodeid)


def _merge_npu_mem_profiles(profiles, loaded_profiles):
    for key, value in loaded_profiles.items():
        _merge_npu_mem_profile(profiles, _make_mem_key(key), value)


def _merge_npu_mem_profile(profiles, key, value):
    """Merge one profile entry, keeping the larger peak on key collisions.

    Collisions are expected: num_sms variants of a test share one normalized
    key (see _normalize_mem_key), and taking the max keeps the classification
    conservative.
    """
    existing = profiles.get(key)
    if existing is None or value['peak_npu_mb'] > existing['peak_npu_mb']:
        profiles[key] = value


def _normalize_mem_key(key: str) -> str:
    """Strip num_sms=... from a parametrized node ID.

    Peak NPU memory is assumed to be insensitive to the num_sms parameter, so
    all num_sms variants of a test share one profile entry (merged with max).
    If a kernel's memory usage ever depends on num_sms, this assumption and
    the merge strategy must be revisited.
    """
    prefix, separator, param_id = key.rpartition('[')
    if not separator or not param_id.endswith(']'):
        return key

    fields = [field for field in param_id[:-1].split(',') if field]
    kept_fields = [field for field in fields if not field.startswith('num_sms=')]
    if not kept_fields:
        return prefix
    return f'{prefix}[{",".join(kept_fields)},]'


def _npu_mem_source_file_from_test_id(test_id: str):
    source_file = test_id.split('::', 1)[0]
    if not source_file.endswith('.py'):
        return None
    return source_file


def _write_npu_mem_profile_shards(profiles, touched_shards):
    for source_file in touched_shards:
        shard_profiles = {
            test_id: value
            for test_id, value in profiles.items()
            if _npu_mem_source_file_from_test_id(test_id) == source_file
        }
        path = _npu_mem_profile_shard_path(source_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write to a temp file and rename so an interrupt (e.g. SIGTERM
        # raising KeyboardInterrupt) cannot leave a truncated shard behind.
        tmp_path = path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(shard_profiles, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.replace(tmp_path, path)


def _npu_mem_profile_shard_path(source_file):
    parts = [part for part in source_file.replace('\\', '/').split('/') if part not in ('', '.')]
    if not parts or '..' in parts:
        raise ValueError(f'Invalid tests-relative NPU memory source file: {source_file!r}')

    artifact_parts = (
        parts[:1] + [_NPU_MEM_DIR_NAME] + parts[1:]
        if len(parts) > 1
        else [
            _NPU_MEM_DIR_NAME,
            *parts,
        ]
    )
    path = os.path.join(_TESTS_DIR, *artifact_parts)
    if path.endswith('.py'):
        return path[: -len('.py')] + '.json'
    return path


def _print_npu_mem_summary(terminalreporter, config):
    num_missing_only_deselected = config.stash.get(_NPU_MEM_MISSING_ONLY_DESELECTED_KEY, 0)
    if num_missing_only_deselected:
        terminalreporter.section('NPU Memory Missing-Only Refresh')
        terminalreporter.write_line(
            f'Skipped {num_missing_only_deselected} test(s) already present in tests/<operator>/npu_mem_profiles/.'
        )
        terminalreporter.write_line('')

    update_summary = config.stash.get(_NPU_MEM_UPDATE_SUMMARY_KEY, None)
    if update_summary is not None:
        terminalreporter.section('NPU Memory Profile Refresh')
        for line in _format_update_summary_lines(update_summary):
            terminalreporter.write_line(line)
        if update_summary['updated']:
            terminalreporter.write_line(
                f'Stage 1 (parallel): {update_summary["num_small"]} test(s) <= '
                f'{update_summary["threshold"]:.0f} MB; Stage 2 (isolated): '
                f'{update_summary["num_large"]} test(s) > {update_summary["threshold"]:.0f} MB'
            )
        terminalreporter.write_line('')

    if config.getoption('--npu-mem-output', default=None) is None:
        num_unknown = config.stash.get(_NPU_MEM_NUM_UNKNOWN_KEY, 0)
        if num_unknown > 0:
            terminalreporter.section('NPU Memory Auto-Classification')
            terminalreporter.write_line(
                f'{num_unknown} test(s) have no NPU memory profile and were '
                f'conservatively classified as large_npu_mem (Stage 2).'
            )
            terminalreporter.write_line(
                "Run pytest with '--refresh-mem' to measure and record their actual NPU memory usage."
            )
        return

    records = _load_npu_mem_records(config)
    if not records:
        return

    threshold = config.getoption('--npu-mem-threshold')
    terminalreporter.section('NPU Memory Profile Report')
    terminalreporter.write_line(f'Threshold: {threshold:.0f} MB ({threshold / 1024:.1f} GB)')
    terminalreporter.write_line('')

    records.sort(key=lambda record: record['peak_npu_mb'], reverse=True)
    display_names = [
        record['test_id'] if len(record['test_id']) <= 78 else '...' + record['test_id'][-75:] for record in records
    ]
    test_width = max(len('Test'), *(len(name) for name in display_names))
    header = f'{"Test":<{test_width}} {"Peak MB":>10} {"Class":>8}'
    terminalreporter.write_line(header)
    terminalreporter.write_line('-' * len(header))

    num_large = 0
    for record, test_id in zip(records, display_names, strict=False):
        peak = record['peak_npu_mb']
        cls = 'LARGE' if peak > threshold else 'ok'
        if peak > threshold:
            num_large += 1
        terminalreporter.write_line(f'{test_id:<{test_width}} {peak:>9.1f} {cls:>8}')

    terminalreporter.write_line('')
    terminalreporter.write_line(
        f'Total: {len(records)} tests profiled, {num_large} above threshold '
        f'({threshold:.0f} MB), {len(records) - num_large} below.'
    )


@pytest.fixture(autouse=True)
def _track_npu_peak_memory(request):
    if request.config.getoption('--npu-mem-output', default=None) is None:
        yield
        return

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    mem_before = torch.npu.memory_allocated()

    yield

    peak_mb = max(torch.npu.max_memory_allocated() - mem_before, 0) / (1024 * 1024)
    request.node.stash[_NPU_MEM_RECORD_KEY] = {
        'test_id': _make_mem_key(request.node.nodeid),
        'peak_npu_mb': round(peak_mb, 1),
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    output_path = item.config.getoption('--npu-mem-output', default=None)
    if output_path is None:
        return

    if outcome.get_result().failed:
        item.stash[_NPU_MEM_TEST_FAILED_KEY] = True
    if call.when != 'teardown' or item.stash.get(_NPU_MEM_TEST_FAILED_KEY, False):
        return

    # Only record tests whose setup, call, and teardown all succeeded.
    record = item.stash.get(_NPU_MEM_RECORD_KEY, None)
    if record is not None:
        with open(output_path, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
