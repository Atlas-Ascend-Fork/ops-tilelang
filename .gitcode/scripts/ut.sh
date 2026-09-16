#!/usr/bin/env bash

# Run regular TileKernels pytest cases on one device, on both Ascend codegen
# backends: "auto" (default, AscendC) then "pto" (PTO codegen). Override the
# backend list with UT_TARGETS, e.g. UT_TARGETS="auto" or UT_TARGETS="pto".
#
# Incremental mode (default when pr_filelist.txt exists): run changed
# pytest files directly and map changed operator sources to their pytest files;
# exit 0 if no changed file qualifies. Override with PR_FILELIST env var.
# Mapping rules live in .gitcode/config/operator_tests.conf; override its path
# with OPERATOR_TEST_MAP. Rules are loaded once in incremental mode.
#
# Optional overrides:
#   DEVICE_ID=0 PYTEST_WORKERS=4 bash scripts/ut.sh
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
CONDA_ENV="${CONDA_ENV:-tl_tilelang0909ci}"
DEVICE_ID="${DEVICE_ID:-${ASCEND_RT_VISIBLE_DEVICES:-3}}"
PYTEST_WORKERS="${PYTEST_WORKERS:-2}"
PR_FILELIST="${PR_FILELIST:-${ROOT_DIR}/pr_filelist.txt}"
OPERATOR_TEST_MAP="${OPERATOR_TEST_MAP:-${ROOT_DIR}/.gitcode/config/operator_tests.conf}"
CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward}"

if [[ ! "${PYTEST_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PYTEST_WORKERS must be a positive integer, got: ${PYTEST_WORKERS}" >&2
    exit 2
fi

if [[ -z "${DEVICE_ID}" || "${DEVICE_ID}" == *,* ]]; then
    echo "DEVICE_ID must select exactly one device, got: ${DEVICE_ID}" >&2
    exit 2
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
source /data/zw/Ascend/cann/set_env.sh

cd "${ROOT_DIR}"

export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
export TILELANG_DISABLE_CACHE=1
export CPLUS_INCLUDE_PATH

echo "Running regular pytest cases on NPU ${ASCEND_RT_VISIBLE_DEVICES} with ${PYTEST_WORKERS} workers"

if (( $# > 0 )); then
    pytest_targets=("$@")
elif [[ -f "${PR_FILELIST}" ]]; then
    # Incremental mode: pick changed pytest files and tests mapped from changed
    # operator sources in pr_filelist.txt.
    pytest_targets=()

    add_pytest_target() {
        local test_path="$1"
        local existing

        # Deleted or not-yet-migrated tests cannot be passed to pytest.
        if [[ ! -f "${ROOT_DIR}/${test_path}" ]]; then
            return
        fi
        for existing in "${pytest_targets[@]}"; do
            if [[ "${existing}" == "${test_path}" ]]; then
                return
            fi
        done
        pytest_targets+=("${test_path}")
    }

    load_operator_test_map() {
        local map_file="$1"
        local line line_number=0 test_path
        local -a fields=()
        operator_source_patterns=()
        operator_test_targets=()

        if [[ ! -f "${map_file}" || ! -r "${map_file}" ]]; then
            echo "Operator test map is missing or unreadable: ${map_file}" >&2
            return 2
        fi

        while IFS= read -r line || [[ -n "${line}" ]]; do
            ((line_number += 1))
            line="${line%$'\r'}"
            read -r -a fields <<< "${line}"
            if (( ${#fields[@]} == 0 )) || [[ "${fields[0]}" == \#* ]]; then
                continue
            fi
            if (( ${#fields[@]} < 2 )); then
                echo "${map_file}:${line_number}: expected a source pattern and test paths" >&2
                return 2
            fi
            for test_path in "${fields[@]:1}"; do
                if [[ "${test_path}" != tests/*.py || "${test_path}" == */../* ]]; then
                    echo "${map_file}:${line_number}: invalid test path: ${test_path}" >&2
                    return 2
                fi
            done
            operator_source_patterns+=("${fields[0]}")
            operator_test_targets+=("${fields[*]:1}")
        done < "${map_file}"
    }

    map_operator_tests() {
        local file_path="$1"
        local index test_path
        local -a targets=()

        for ((index = 0; index < ${#operator_source_patterns[@]}; index++)); do
            # Leave the right-hand side unquoted to apply the configured Bash glob.
            if [[ "${file_path}" == ${operator_source_patterns[index]} ]]; then
                read -r -a targets <<< "${operator_test_targets[index]}"
                for test_path in "${targets[@]}"; do
                    add_pytest_target "${test_path}"
                done
                # Match the original case statement: the first matching rule wins.
                return 0
            fi
        done
        return 0
    }

    load_operator_test_map "${OPERATOR_TEST_MAP}"

    mapfile -t changed_lines < "${PR_FILELIST}"
    for file_path in "${changed_lines[@]}"
    do
        # Keep the committed name-only file-list parsing behavior.
        file_path="$(echo "${file_path}" | xargs)"
        if [[ -z "${file_path}" ]]; then
            continue
        fi

        # Keep the existing behavior for directly changed pytest files.
        if [[ "${file_path}" == tests/* ]]; then
            base_name="$(basename -- "${file_path}")"
            if [[ "${base_name}" == test_*.py || "${base_name}" == *_test.py ]]; then
                add_pytest_target "${file_path}"
                continue
            fi
        fi

        map_operator_tests "${file_path}"
    done
    if (( ${#pytest_targets[@]} == 0 )); then
        echo "No changed pytest file or mapped operator test found, skip UT."
        exit 0
    fi
    echo "Incremental UT on ${#pytest_targets[@]} changed or mapped pytest file(s):"
    printf '  %s\n' "${pytest_targets[@]}"
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
