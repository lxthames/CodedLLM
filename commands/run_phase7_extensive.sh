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
seed_count="${SEED_COUNT:-30}"
num_requests="${NUM_REQUESTS:-10000}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  printf 'run_phase7_extensive.sh requires a clean tracked and untracked worktree\n' >&2
  exit 1
fi

mkdir -p "${artifacts_dir}"
run_dir="$(mktemp -d "${artifacts_dir}/phase7-extensive-${timestamp}-XXXXXX")"

BUILD_DIR="${build_dir}" bash "${repo_root}/commands/build.sh" >/dev/null

{
  printf 'timestamp_utc=%s\n' "${timestamp}"
  printf 'git_commit='
  git -C "${repo_root}" rev-parse HEAD
  printf 'base_seed=%s\nseed_count=%s\nnum_requests=%s\n' \
    "${base_seed}" "${seed_count}" "${num_requests}"
  printf 'matrix=primary,queue_robustness,coding_sensitivity\n'
  printf '\nnvcc:\n'
  nvcc --version
  printf '\ncmake:\n'
  cmake --version
  printf '\nnvidia_smi:\n'
  nvidia-smi
} > "${run_dir}/metadata.txt"

"${build_dir}/device_info" > "${run_dir}/device_info.txt"
"${build_dir}/bench_decoder" --benchmark_format=json > "${run_dir}/bench_decoder.json"
"${build_dir}/bench_decode_plan" --benchmark_format=json \
  > "${run_dir}/bench_decode_plan.json"
python3 "${repo_root}/scripts/extract_phase7_decode_costs.py" \
  "${run_dir}/bench_decoder.json" \
  -o "${run_dir}/decode_costs.csv"

{
  printf 'scenario=primary\n'
  printf 'arrival_rates_hz=500,1000,2000,4000,8000\n'
  printf 'straggler_probabilities=0.01,0.05,0.10\n'
  printf 'parity_counts=4,8\n'
  printf 'degrees=3\nshard_sizes_mib=16\nqueue_depths=4\ngpu_concurrencies=2\n\n'
  printf 'scenario=queue_robustness\n'
  printf 'arrival_rates_hz=2000\nstraggler_probabilities=0.05\n'
  printf 'parity_counts=8\ndegrees=3\nshard_sizes_mib=16\n'
  printf 'queue_depths=1,4,16\ngpu_concurrencies=2,4\n\n'
  printf 'scenario=coding_sensitivity\n'
  printf 'arrival_rates_hz=2000\nstraggler_probabilities=0.05\n'
  printf 'parity_counts=4,8\ndegrees=2,3,5,8\nshard_sizes_mib=1,16\n'
  printf 'queue_depths=4\ngpu_concurrencies=2\n'
} > "${run_dir}/manifest.txt"

run_sweep() {
  local scenario="$1"
  local arrival_rates="$2"
  local stragglers="$3"
  local parity_counts="$4"
  local degrees="$5"
  local shard_sizes="$6"
  local queue_depths="$7"
  local concurrencies="$8"
  "${build_dir}/phase7_parameter_sweep" \
    --scenario "${scenario}" \
    --base-seed "${base_seed}" \
    --seed-count "${seed_count}" \
    --num-requests "${num_requests}" \
    --arrival-rates-hz "${arrival_rates}" \
    --straggler-probabilities "${stragglers}" \
    --parity-counts "${parity_counts}" \
    --degrees "${degrees}" \
    --shard-sizes-mib "${shard_sizes}" \
    --queue-depths "${queue_depths}" \
    --gpu-concurrencies "${concurrencies}" \
    --calibration-csv "${run_dir}/decode_costs.csv" \
    > "${run_dir}/${scenario}_raw.csv"
}

run_sweep primary 500,1000,2000,4000,8000 0.01,0.05,0.10 4,8 3 16 4 2
run_sweep queue_robustness 2000 0.05 8 3 16 1,4,16 2,4
run_sweep coding_sensitivity 2000 0.05 4,8 2,3,5,8 1,16 4 2

python3 - "${run_dir}" <<'PY'
import csv
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
inputs = [
    run_dir / "primary_raw.csv",
    run_dir / "queue_robustness_raw.csv",
    run_dir / "coding_sensitivity_raw.csv",
]
with (run_dir / "phase7_raw.csv").open("w", newline="", encoding="utf-8") as output:
    writer = None
    for path in inputs:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                writer.writeheader()
            writer.writerows(reader)
PY
python3 "${repo_root}/scripts/analyze_phase7.py" \
  "${run_dir}/phase7_raw.csv" \
  -o "${run_dir}/phase7_summary.csv" \
  --conclusions "${run_dir}/conclusions.csv"

printf '%s\n' "${run_dir}"
