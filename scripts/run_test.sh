#!/usr/bin/env bash
#
# 并行运行 ops-tilelang 正确性测试。编译瓶颈主要在 CPU，因此每张 NPU
# 可使用多个 worker 并行编译，而大显存测试使用更低并发度。
#
# 分两阶段：
#   1. 普通测试：默认每张 NPU 2 个 worker，排除 large_npu_mem。
#   2. 大/未知显存测试：默认每张 NPU 1 个 worker。
#
# OPS_TILELANG_RUN_BENCHMARK=1 时只运行 benchmark 回归。
#

set -euo pipefail
set -x

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -e /dev/davinci_manager ]]; then
  echo 'No Ascend NPU runtime was detected (/dev/davinci_manager is absent).' >&2
  exit 1
fi

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info
else
  echo 'Warning: npu-smi is unavailable; continuing with torch_npu device detection.' >&2
fi

NUM_NPUS=$(python -c 'import torch; import torch_npu; print(torch.npu.device_count())')
if [[ "${NUM_NPUS}" -lt 1 ]]; then
  echo 'No Ascend NPU is visible to torch_npu.' >&2
  exit 1
fi

pytest_args=("$@")

handle_error() {
  local exit_code=$?
  if [[ "${exit_code}" -eq 5 ]]; then
    echo 'No tests found, ignoring...'
    return 0
  fi
  echo "Command failed with exit code ${exit_code}" >&2
  exit "${exit_code}"
}

# Benchmark 测速时每张 NPU 只允许一个 worker。
if [[ "${OPS_TILELANG_RUN_BENCHMARK:-0}" == '1' ]]; then
  echo "=== Benchmark regression tests with ${NUM_NPUS} workers (1 per NPU) ==="
  python -m pytest "${pytest_args[@]}" \
    -n "${NUM_NPUS}" \
    --dist load \
    -m benchmark \
    --run-benchmark \
    --benchmark-regression-threshold 0.10 \
    --benchmark-min-delta-us 1.0 || handle_error
  exit 0
fi

WORKERS_PER_NPU=${TILELANG_TEST_WORKERS_PER_NPU:-${OPS_TILELANG_TEST_WORKERS_PER_NPU:-2}}
NUM_WORKERS=$((NUM_NPUS * WORKERS_PER_NPU))
echo "=== Stage 1: Regular tests with ${NUM_WORKERS} workers (${NUM_NPUS} NPUs x ${WORKERS_PER_NPU} workers/NPU) ==="
python -m pytest "${pytest_args[@]}" \
  -n "${NUM_WORKERS}" \
  --dist load \
  -m 'not large_npu_mem and not benchmark' \
  -x || handle_error

WORKERS_PER_NPU_LARGE=${TILELANG_TEST_WORKERS_PER_NPU_LARGE:-${OPS_TILELANG_TEST_LARGE_WORKERS_PER_NPU:-1}}
NUM_WORKERS_LARGE=$((NUM_NPUS * WORKERS_PER_NPU_LARGE))
echo "=== Stage 2: Large NPU memory tests with ${NUM_WORKERS_LARGE} workers (${NUM_NPUS} NPUs x ${WORKERS_PER_NPU_LARGE} workers/NPU) ==="
python -m pytest "${pytest_args[@]}" \
  -n "${NUM_WORKERS_LARGE}" \
  --dist load \
  -m 'large_npu_mem and not benchmark' \
  -x || handle_error
