#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

nero_require_uv
nero_print_paths

if [[ ! -d "${NERO_MISSION7_DATASET_ROOT}" ]]; then
    echo "ERROR: generated dataset does not exist: ${NERO_MISSION7_DATASET_ROOT}" >&2
    exit 1
fi

cd "${NERO_REPO_ROOT}"
uv run python scripts/nero/mission7_training_views.py validate \
    --dataset "${NERO_MISSION7_DATASET_ROOT}"
