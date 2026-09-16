#!/usr/bin/env bash
# Portable CI entry point. Select a task with OPS_TILELANG_CI_TASK.

set -euo pipefail
set -x

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CI_TASK=${OPS_TILELANG_CI_TASK:-all}

run_lint() {
  python -m ruff check .
  python -m ruff format --check .
}

run_cpu_tests() {
  python -m pytest -m 'not npu and not benchmark'
}

run_wheel() {
  "${REPO_ROOT}/scripts/build_wheel.sh"
}

run_npu_tests() {
  "${REPO_ROOT}/scripts/run_test.sh" "$@"
}

case "${CI_TASK}" in
  lint)
    run_lint
    ;;
  cpu)
    run_cpu_tests
    ;;
  wheel)
    run_wheel
    ;;
  npu)
    run_npu_tests "$@"
    ;;
  benchmark)
    export OPS_TILELANG_RUN_BENCHMARK=1
    run_npu_tests "$@"
    ;;
  all)
    run_lint
    run_cpu_tests
    run_wheel
    run_npu_tests "$@"
    ;;
  *)
    echo "Unknown OPS_TILELANG_CI_TASK=${CI_TASK}. Expected: lint, cpu, wheel, npu, benchmark, or all." >&2
    exit 2
    ;;
esac
