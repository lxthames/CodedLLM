#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_build_dir="${repo_root}/build-ninja"
if [[ -f /.dockerenv ]]; then
  default_build_dir="${repo_root}/build-ninja-container"
fi
build_dir="${BUILD_DIR:-"${default_build_dir}"}"
artifacts_dir="${ARTIFACTS_DIR:-"${repo_root}/artifacts"}"
base_seed="${BASE_SEED:-20260901}"
seed_count="${SEED_COUNT:-2}"
num_requests="${NUM_REQUESTS:-100}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${artifacts_dir}"
run_dir="$(mktemp -d "${artifacts_dir}/phase7-pilot-${timestamp}-XXXXXX")"

BUILD_DIR="${build_dir}" bash "${repo_root}/commands/build.sh"

{
  printf 'timestamp_utc=%s\n' "${timestamp}"
  printf 'git_commit='
  git -C "${repo_root}" rev-parse HEAD
  printf 'git_status:\n'
  git -C "${repo_root}" status --short
  printf '\nbase_seed=%s\nseed_count=%s\nnum_requests=%s\n' \
    "${base_seed}" "${seed_count}" "${num_requests}"
} > "${run_dir}/metadata.txt"

"${build_dir}/phase7_parameter_sweep" \
  --base-seed "${base_seed}" \
  --seed-count "${seed_count}" \
  --num-requests "${num_requests}" \
  > "${run_dir}/phase7_raw.csv"
python3 "${repo_root}/scripts/analyze_phase7.py" \
  "${run_dir}/phase7_raw.csv" \
  -o "${run_dir}/phase7_summary.csv"

printf 'Phase 7 pilot results: %s\n' "${run_dir}"
