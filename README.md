# CodedLLM

CodedLLM is a CUDA systems research project investigating whether sparse erasure
coding can recover slow or unavailable KV-cache shards faster than waiting for
stragglers in disaggregated LLM serving.

## Why CodedLLM?

Disaggregated LLM serving may transfer KV-cache data between prefill and decode
workers. A decode request can be delayed by its slowest required shard. Full
replication reduces this risk, but duplicates large KV tensors and consumes GPU
memory that could otherwise support larger models, batches, or contexts.

CodedLLM explores a different trade-off:

1. Keep the original KV shards in systematic form.
2. Store a smaller number of sparse XOR parity shards.
3. Detect a missing or excessively slow shard.
4. Build a recovery plan from the available shards.
5. Reconstruct the exact missing bytes with a bandwidth-oriented CUDA kernel.

```text
Available data and parity shards
              |
              v
      CPU decode plan
              |
              v
 Device-resident CUDA executor
              |
              v
   Reconstructed KV-cache shard
              |
              v
       Ordinary attention
```

The project reconstructs raw fixed-width words with XOR. It does not approximate
KV values, modify attention, or perform floating-point operations on the encoded
representation.

## Current status

The repository currently provides the foundational coding and CUDA execution
platform:

- validated `ShardLayout`, immutable `CodeGraph`, and `DecodePlan` contracts;
- CPU parity generation and bit-exact decode-plan execution;
- CUDA XOR kernels using grid-stride loops and safe `uint4` vectorization;
- compile-time degree specialization for degrees 2 through 6;
- a dynamic-degree fallback and safe scalar tail handling;
- hardware-aware launch configuration using CUDA occupancy APIs;
- reusable device-resident buffers, source tables, streams, and timing events;
- sequential execution of dependent multi-operation decode plans;
- GoogleTest CPU/GPU differential tests;
- Google Benchmark kernel, end-to-end, roofline, and dependent-plan benchmarks;
- a CUDA device information utility; and
- a targeted Nsight Compute profiling script.

The sparse graph generator, peeling decoder, recovery policy, straggler simulator,
and serving-engine integration are future research stages. This is currently a
research prototype, not a production serving library.

## Repository layout

```text
include/codedllm/       Core, coding, and CUDA tuning interfaces
include/kernels/        Host-facing decoder interface
src/                    Pure C++ graph, codec, and CPU reference code
kernels/                CUDA kernels and resource-management utilities
tests/                  GoogleTest contract and CPU/GPU correctness tests
benchmarks/             Kernel, roofline, and decode-plan microbenchmarks
tools/                  CUDA device capability reporting
scripts/                Reproducible profiling commands
.devcontainer/          CUDA 12.4 development container
HelpingMatrials/        Research overview, roadmap, and reference material
```

## Requirements

- NVIDIA GPU with a working CUDA driver
- CUDA Toolkit 12.x or compatible NVCC
- C++17 compiler
- CMake 3.20 or newer
- Docker Desktop with WSL2 GPU support, or a native Linux CUDA environment

CMake 3.24 and newer use the native GPU architecture automatically. The supplied
Ubuntu 22.04 container uses an `sm_75` fallback for its older CMake version and the
project's current RTX 2080 Ti development GPU.

## Reproducible Docker setup

On Windows, install Docker Desktop, enable its WSL2 backend, and verify that this
command can see the GPU:

```cmd
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Build the development image from Command Prompt:

```cmd
docker build -t codedllm-cuda-dev:local -f .devcontainer/Dockerfile .devcontainer
```

Configure, build, and test the project:

```cmd
docker run --rm --gpus all -v "%cd%:/workspace" -w /workspace codedllm-cuda-dev:local bash -lc "cmake -S . -B /tmp/codedllm-build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/codedllm-build --parallel && ctest --test-dir /tmp/codedllm-build --output-on-failure"
```

You can also open the repository in VS Code or Cursor and choose **Reopen in
Container**. The DevContainer installs the C++, CMake, and formatting extensions
and exposes the host GPU to the container.

## Native Linux build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The test executable is available at `build/tests/test_decoder`.

## Benchmarks

After building, run:

```bash
./build/bench_decoder
./build/bench_decode_plan
./build/device_info
```

`bench_decoder` reports:

- end-to-end allocation, transfer, kernel, and return latency;
- true device-resident kernel-only latency;
- effective XOR bandwidth; and
- a 16 MiB device-to-device copy roofline.

`bench_decode_plan` measures dependent plans containing 1, 2, 4, and 8 recovery
operations for 1 MiB and 16 MiB shards.

On the current RTX 2080 Ti development system, large device-resident decode plans
sustain approximately 512-527 GB/s. These are machine-specific microbenchmark
results, not an end-to-end serving claim.

## Nsight Compute profiling

The profiling script builds an architecture-correct Release binary and profiles
one degree-2, 16 MiB kernel launch:

```bash
bash scripts/profile_decoder.sh
```

From Windows Command Prompt with Docker:

```cmd
docker run --rm --gpus all --cap-add=SYS_ADMIN -v "%cd%:/workspace" -w /workspace codedllm-cuda-dev:local bash scripts/profile_decoder.sh
```

NVIDIA GPU performance-counter access must be enabled on the Windows host before
using Nsight Compute through WSL2.

## Research roadmap

The next major milestones are:

1. deterministic sparse graph construction and multi-erasure peeling;
2. randomized recoverability and CPU/GPU differential testing;
3. deterministic shard-arrival and straggler simulation;
4. wait-versus-recover latency policy and bounded recovery queue;
5. KV-shaped end-to-end experiments; and
6. integration with a disaggregated serving or transport prototype.

The central research question is not merely whether XOR is fast. CodedLLM must show
that complete recovery, including planning, data movement, queueing, and handoff,
beats waiting or replication in a reproducible operating region.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
