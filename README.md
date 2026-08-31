# CodedLLM

CodedLLM explores sparse-graph erasure coding to reconstruct straggling KV-cache shards in disaggregated LLM serving.

## Build and test

On a Linux system with NVCC, CMake 3.20+, and a C++17 compiler installed:

```bash
mkdir build && cd build && cmake .. && make && ./tests/test_decoder
```

## Development Environment Setup

Install Docker Desktop for Windows and enable its WSL2 backend. Open this project
folder in VS Code or Cursor, then select **Reopen in Container** when prompted.

The DevContainer builds a Linux CUDA development environment with CMake and NVCC
pre-configured, giving every developer a reproducible build and test environment.
