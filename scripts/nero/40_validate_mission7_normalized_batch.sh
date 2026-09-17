#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

nero_require_uv
nero_print_paths

if [[ ! -f "${NERO_MISSION7_NORM_PATH}" ]]; then
    echo "ERROR: normalization statistics do not exist: ${NERO_MISSION7_NORM_PATH}" >&2
    exit 1
fi

cd "${NERO_REPO_ROOT}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
uv run python scripts/nero/validate_normalized_batch.py \
    --config-name "${NERO_MISSION7_CONFIG_NAME}"
