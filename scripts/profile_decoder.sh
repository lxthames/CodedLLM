#!/usr/bin/env bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
build_dir="${BUILD_DIR:-/tmp/codedllm-profile-build}"

cd "${repo_root}"

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
  echo "Build directory not configured; creating a Release build at ${build_dir}"
  cmake -S "${repo_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
fi

# Always rebuild so changes to the CUDA architecture or decoder are reflected
# in the binary selected for profiling.
cmake --build "${build_dir}" --target bench_decoder --parallel

benchmark_binary="${build_dir}/bench_decoder"
if [[ ! -x "${benchmark_binary}" ]]; then
  echo "Benchmark executable not found: ${benchmark_binary}" >&2
  echo "Reconfigure with: cmake -S ${repo_root} -B ${build_dir} -DCMAKE_BUILD_TYPE=Release" >&2
  exit 1
fi

# The registered benchmark uses /2/ for degree 2; there is no benchmark name
# containing the literal string "Degree2". The launch limit ensures Nsight
# Compute profiles exactly one matching XOR kernel invocation.
ncu --section MemoryWorkloadAnalysis -c 1 \
  "${benchmark_binary}" \
  --benchmark_filter="^SparseDecoderBenchmark/KernelOnlyLatency/2/16777216/.*$" \
  --benchmark_min_time=0.01s
