#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

python -m build --wheel --no-isolation

echo 'Wheel artifacts:'
find dist -maxdepth 1 -type f -name 'ops_tilelang-*.whl' -print | sort
