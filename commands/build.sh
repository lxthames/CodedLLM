#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_build_dir="${repo_root}/build-ninja"
if [[ -f /.dockerenv ]]; then
  default_build_dir="${repo_root}/build-ninja-container"
fi
build_dir="${BUILD_DIR:-"${default_build_dir}"}"

cmake -S "${repo_root}" -B "${build_dir}" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel
