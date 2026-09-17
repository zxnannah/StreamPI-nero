#!/usr/bin/env bash

set -euo pipefail

NERO_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NERO_REPO_ROOT="$(cd -- "${NERO_SCRIPT_DIR}/../.." && pwd)"

export NERO_MISSION7_SOURCE_ROOT="${NERO_MISSION7_SOURCE_ROOT:-/mnt/nero_nas/missions/nero/mission7/smooth}"
export NERO_MISSION7_DATASET_ROOT="${NERO_MISSION7_DATASET_ROOT:-${NERO_REPO_ROOT}/data/nero/mission7_training_views_lerobot_v2_0}"
export NERO_MISSION7_CONFIG_NAME="${NERO_MISSION7_CONFIG_NAME:-pi05_nero_stream5_mission7_views}"
export NERO_MISSION7_VIEW_TYPES="${NERO_MISSION7_VIEW_TYPES:-full phase transition}"

NERO_MISSION7_NORM_PATH="${NERO_REPO_ROOT}/assets/${NERO_MISSION7_CONFIG_NAME}/nero_mission7_views/norm_stats.json"
NERO_MISSION7_LOG_DIR="${NERO_REPO_ROOT}/logs"

nero_require_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv is not available in PATH." >&2
        exit 1
    fi
}

nero_print_paths() {
    echo "Repository: ${NERO_REPO_ROOT}"
    echo "Read-only source: ${NERO_MISSION7_SOURCE_ROOT}"
    echo "Generated dataset: ${NERO_MISSION7_DATASET_ROOT}"
    echo "View types: ${NERO_MISSION7_VIEW_TYPES}"
    echo "Training config: ${NERO_MISSION7_CONFIG_NAME}"
}
