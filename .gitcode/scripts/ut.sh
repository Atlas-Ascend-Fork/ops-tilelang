#!/usr/bin/env bash

# Run regular TileKernels pytest cases on one or more devices (xdist workers
# are bound round-robin to the device list by tests/conftest.py), on both
# Ascend codegen backends: "auto" (default, AscendC) then "pto" (PTO codegen).
# Override the backend list with UT_TARGETS, e.g. UT_TARGETS="auto" or
# UT_TARGETS="pto".
#
# Incremental mode (default when pr_filelist.txt exists): statically discover
# tests affected by changed Python modules, including transitive imports.
# Shared-file changes or uncertain dependencies run the complete L0 scope.
# No operator-to-test mapping is required. Override the changed-file list with
# PR_FILELIST and global full-test rules with FULL_TEST_TRIGGER_CONFIG.
#
# Optional overrides:
#   DEVICE_ID=1,3 PYTEST_WORKERS=6 bash scripts/ut.sh
#   bash scripts/ut.sh tests/quant/test_swiglu_forward.py -k case_name
#
# Benchmark regression mode (opt-in): TK_RUN_BENCHMARK=1 runs only the
# benchmark-marked cases (--run-benchmark) once, with 1 worker on the
# default backend, comparing against tests/benchmark_baselines/
# (comparison only; never refreshes baselines). Targets follow the same
# selection rules as functional tests (args > pr_filelist > tests/).
#   TK_RUN_BENCHMARK=1 bash scripts/ut.sh tests

set -Eeo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-tl_tilelang0922ci}"
DEVICE_ID="${DEVICE_ID:-${ASCEND_RT_VISIBLE_DEVICES:-2,3}}"
PYTEST_WORKERS="${PYTEST_WORKERS:-6}"
PR_FILELIST="${PR_FILELIST:-${ROOT_DIR}/pr_filelist.txt}"
FULL_TEST_TRIGGER_CONFIG="${FULL_TEST_TRIGGER_CONFIG:-${ROOT_DIR}/.gitcode/config/full_test_triggers.conf}"

if [[ ! "${PYTEST_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PYTEST_WORKERS must be a positive integer, got: ${PYTEST_WORKERS}" >&2
    exit 2
fi

if [[ -z "${DEVICE_ID}" || ! "${DEVICE_ID}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "DEVICE_ID must be one or more comma-separated device ids, got: ${DEVICE_ID}" >&2
    exit 2
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
source /data/z00520135/weekly0922/cann/set_env.sh

cd "${ROOT_DIR}"

export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
export TILELANG_DISABLE_CACHE=1

echo "Running regular pytest cases on NPU ${ASCEND_RT_VISIBLE_DEVICES} with ${PYTEST_WORKERS} workers"

if (( $# > 0 )); then
    pytest_targets=("$@")
elif [[ -f "${PR_FILELIST}" ]]; then
    python -m unittest discover -s "${ROOT_DIR}/.gitcode/scripts/tests"
    # Capture the command first so selector failures propagate under set -e.
    selection_output="$(python "${ROOT_DIR}/.gitcode/scripts/select_tests.py" \
        --root "${ROOT_DIR}" \
        --changed-files "${PR_FILELIST}" \
        --full-test-triggers "${FULL_TEST_TRIGGER_CONFIG}")"
    mapfile -t selection <<< "${selection_output}"
    case "${selection[0]}" in
        full)
            pytest_targets=(tests)
            export OPS_TILELANG_TEST_LEVEL=0
            echo "Full L0 UT selected by changed-file analysis."
            ;;
        incremental)
            pytest_targets=("${selection[@]:1}")
            if (( ${#pytest_targets[@]} == 0 )); then
                echo "Test selector returned an empty incremental scope." >&2
                exit 2
            fi
            echo "Incremental UT on ${#pytest_targets[@]} affected pytest file(s):"
            printf '  %s\n' "${pytest_targets[@]}"
            ;;
        skip)
            echo "No affected executable files; skip UT."
            exit 0
            ;;
        *)
            echo "Invalid test selector result: ${selection[0]}" >&2
            exit 2
            ;;
    esac
else
    pytest_targets=(tests)
fi

# Benchmark regression mode: mirrors scripts/run_test.sh's TK_RUN_BENCHMARK
# branch — single pytest invocation, 1 worker (timing must not share the
# device), default backend (no TILELANG_DEFAULT_TARGET export, so auto and
# pto never compare against the same baseline). Exit 5 ("no tests
# collected") is tolerated, e.g. when the incremental scope has no
# benchmark cases; any regression makes pytest exit non-zero, which is
# propagated to CI.
if [[ "${TK_RUN_BENCHMARK:-0}" == "1" ]]; then
    echo ">>> Benchmark regression on NPU ${ASCEND_RT_VISIBLE_DEVICES} (1 worker, default backend)"
    rc=0
    pytest "${pytest_targets[@]}" -v \
        -n 1 \
        --dist load \
        -m "benchmark" \
        --run-benchmark \
        --benchmark-regression-threshold 0.10 \
        --benchmark-min-delta-us 1.0 || rc=$?
    if (( rc == 5 )); then
        echo "No benchmark case collected in the given scope, skip."
        rc=0
    fi
    exit "${rc}"
fi

run_pytest_for_target() {
    local target="$1"
    local rc=0
    export TILELANG_DEFAULT_TARGET="${target}"
    echo ""
    echo ">>> Running pytest with TILELANG_DEFAULT_TARGET=${TILELANG_DEFAULT_TARGET}"
    pytest "${pytest_targets[@]}" -v \
        -n "${PYTEST_WORKERS}" \
        --dist load \
        -m "not benchmark" || rc=$?
    if (( rc == 5 )); then
        echo "No functional test collected for target: ${target}, skip."
        return 0
    fi
    return "${rc}"
}

UT_TARGETS="${UT_TARGETS:-auto pto}"
overall_status=0
for target in ${UT_TARGETS}; do
    if ! run_pytest_for_target "${target}"; then
        echo ">>> UT FAILED for target: ${target}" >&2
        overall_status=1
    fi
done

exit ${overall_status}
