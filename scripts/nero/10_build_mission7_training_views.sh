#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

nero_require_uv
nero_print_paths
read -r -a view_types <<<"${NERO_MISSION7_VIEW_TYPES}"

if [[ -e "${NERO_MISSION7_DATASET_ROOT}" ]]; then
    echo "ERROR: destination already exists; refusing to overwrite it:" >&2
    echo "  ${NERO_MISSION7_DATASET_ROOT}" >&2
    exit 1
fi

cd "${NERO_REPO_ROOT}"
uv run python scripts/nero/mission7_training_views.py build \
    --source "${NERO_MISSION7_SOURCE_ROOT}" \
    --destination "${NERO_MISSION7_DATASET_ROOT}" \
    --view-types "${view_types[@]}"
