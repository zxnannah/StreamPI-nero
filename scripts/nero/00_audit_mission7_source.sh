#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

nero_require_uv
nero_print_paths
read -r -a view_types <<<"${NERO_MISSION7_VIEW_TYPES}"

cd "${NERO_REPO_ROOT}"
uv run python scripts/nero/mission7_training_views.py audit \
    --source "${NERO_MISSION7_SOURCE_ROOT}" \
    --view-types "${view_types[@]}"
