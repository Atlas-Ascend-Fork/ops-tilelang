#!/usr/bin/env bash

# Run regular TileKernels pytest cases on one device, on both Ascend codegen
# backends: "auto" (default, AscendC) then "pto" (PTO codegen). Override the
# backend list with UT_TARGETS, e.g. UT_TARGETS="auto" or UT_TARGETS="pto".
#
# Incremental mode (default when pr_filelist.txt exists): only run changed
# files under tests/ that match pytest naming (test_*.py / *_test.py); exit 0
# if no changed file qualifies. Override with PR_FILELIST env var.
#
# Optional overrides:
#   DEVICE_ID=0 PYTEST_WORKERS=4 bash scripts/ut.sh
#   bash scripts/ut.sh tests/quant/test_swiglu_forward.py -k case_name

set -Eeo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-tl_tilelang0909ci}"
DEVICE_ID="${DEVICE_ID:-${ASCEND_RT_VISIBLE_DEVICES:-2}}"
PYTEST_WORKERS="${PYTEST_WORKERS:-2}"
PR_FILELIST="${PR_FILELIST:-${ROOT_DIR}/pr_filelist.txt}"
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
    # Incremental mode: pick changed pytest files from pr_filelist.txt
    pytest_targets=()
    mapfile -t changed_lines < "${PR_FILELIST}"
    for file_path in "${changed_lines[@]}"
    do
        # trim leading/trailing whitespace and skip empty lines
        file_path="$(echo "${file_path}" | xargs)"
        if [[ -z "${file_path}" ]]; then
            continue
        fi
        # keep only files under tests/
        if [[ "${file_path}" != tests/* ]]; then
            continue
        fi
        # keep only pytest-collectable names: test_*.py or *_test.py
        base_name="$(basename -- "${file_path}")"
        if [[ "${base_name}" != test_*.py && "${base_name}" != *_test.py ]]; then
            continue
        fi
        # skip files deleted in this PR
        if [[ ! -f "${ROOT_DIR}/${file_path}" ]]; then
            continue
        fi
        # dedupe
        if [[ ! " ${pytest_targets[*]:-} " =~ " ${file_path} " ]]; then
            pytest_targets+=("${file_path}")
        fi
    done
    if (( ${#pytest_targets[@]} == 0 )); then
        echo "No changed file under tests/ matches test_*.py or *_test.py, skip UT."
        exit 0
    fi
    echo "Incremental UT on ${#pytest_targets[@]} changed pytest file(s):"
    printf '  %s\n' "${pytest_targets[@]}"
else
    pytest_targets=(tests)
fi

run_pytest_for_target() {
    local target="$1"
    export TILELANG_DEFAULT_TARGET="${target}"
    echo ""
    echo ">>> Running pytest with TILELANG_DEFAULT_TARGET=${TILELANG_DEFAULT_TARGET}"
    pytest "${pytest_targets[@]}" \
        -n "${PYTEST_WORKERS}" \
        --dist load \
        -m "not large_gpu_mem and not benchmark"
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