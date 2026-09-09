#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_build_dir="${repo_root}/build-ninja"
if [[ -f /.dockerenv ]]; then
  default_build_dir="${repo_root}/build-ninja-container"
fi
build_dir="${BUILD_DIR:-"${default_build_dir}"}"
artifacts_dir="${ARTIFACTS_DIR:-"${repo_root}/artifacts"}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${artifacts_dir}"
run_dir="$(mktemp -d "${artifacts_dir}/gpu-benchmarks-${timestamp}-XXXXXX")"

BUILD_DIR="${build_dir}" bash "${repo_root}/commands/build.sh"

{
  printf 'timestamp_utc=%s\n' "${timestamp}"
  printf 'git_commit='
  git -C "${repo_root}" rev-parse HEAD
  printf 'git_status:\n'
  git -C "${repo_root}" status --short
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

printf 'GPU benchmark results: %s\n' "${run_dir}"
