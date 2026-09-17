#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${script_dir}/00_audit_mission7_source.sh"
"${script_dir}/10_build_mission7_training_views.sh"
"${script_dir}/20_validate_mission7_training_views.sh"
"${script_dir}/30_compute_mission7_norm_stats.sh"
"${script_dir}/40_validate_mission7_normalized_batch.sh"
