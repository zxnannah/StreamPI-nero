#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

nero_require_uv
nero_print_paths

if [[ ! -d "${NERO_MISSION7_DATASET_ROOT}" ]]; then
    echo "ERROR: validate and build the generated dataset first: ${NERO_MISSION7_DATASET_ROOT}" >&2
    exit 1
fi
if [[ -e "${NERO_MISSION7_NORM_PATH}" ]]; then
    echo "ERROR: normalization output already exists; refusing to overwrite it:" >&2
    echo "  ${NERO_MISSION7_NORM_PATH}" >&2
    exit 1
fi

mkdir -p "${NERO_MISSION7_LOG_DIR}"
log_path="${NERO_MISSION7_LOG_DIR}/mission7_training_views_norm_stats.log"

cd "${NERO_REPO_ROOT}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
uv run python scripts/nero/compute_mission7_norm_stats_fast.py \
    --config-name "${NERO_MISSION7_CONFIG_NAME}" \
    2>&1 | tee "${log_path}"

if [[ ! -f "${NERO_MISSION7_NORM_PATH}" ]]; then
    echo "ERROR: normalization command ended without creating ${NERO_MISSION7_NORM_PATH}" >&2
    exit 1
fi
echo "Normalization statistics: ${NERO_MISSION7_NORM_PATH}"
