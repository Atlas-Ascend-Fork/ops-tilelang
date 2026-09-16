"""Root pytest configuration and Ascend xdist worker binding."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

_NPU_SMI_PATH = '/usr/local/bin/npu-smi'

# Source checkouts work without an editable install. This must precede plugin
# imports because the plugins import cann_ops_tilelang during pytest startup.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / 'src'
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _ignore_sigint_in_xdist_workers() -> None:
    """Leave Ctrl+C handling to the xdist controller process."""

    if os.environ.get('PYTEST_XDIST_WORKER') is not None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _visible_npus() -> list[str]:
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    if visible and visible.strip():
        return [device.strip() for device in visible.split(',') if device.strip()]

    try:
        result = subprocess.run(
            [_NPU_SMI_PATH, 'info', '-l'],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ['0']
    if result.returncode != 0:
        return ['0']
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [str(index) for index in range(max(len(lines) - 1, 1))]


def _bind_xdist_worker_to_npu() -> None:
    """Bind each xdist worker before TileLang can initialize the runtime."""

    worker_id = os.environ.get('PYTEST_XDIST_WORKER')
    if worker_id is None or not os.path.exists('/dev/davinci_manager'):
        return
    worker_index = int(worker_id.removeprefix('gw'))
    devices = _visible_npus()
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = devices[worker_index % len(devices)]


_ignore_sigint_in_xdist_workers()
_bind_xdist_worker_to_npu()


@pytest.fixture(autouse=True)
def reset_cann_ops_tilelang_runtime_config():
    """Prevent cached device configuration from leaking between tests."""

    from cann_ops_tilelang.config import clear_runtime_caches

    clear_runtime_caches()
    yield
    clear_runtime_caches()


@pytest.fixture(scope='session')
def npu_available() -> bool:
    from cann_ops_tilelang.config import is_ascend_available

    return is_ascend_available()


pytest_plugins = [
    'tests.pytest_improvement_plugin',
    'tests.pytest_random_plugin',
    'tests.pytest_xdist_failfast_plugin',
    'tests.pytest_npu_mem_plugin',
    'tests.pytest_benchmark_plugin',
    'tests.pytest_tilelang_warning_plugin',
]
