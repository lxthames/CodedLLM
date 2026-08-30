# CodedLLM

CodedLLM explores sparse-graph erasure coding to reconstruct straggling KV-cache shards in disaggregated LLM serving.

## Build and test

On a Linux system with NVCC, CMake 3.20+, and a C++17 compiler installed:

```bash
mkdir build && cd build && cmake .. && make && ./tests/test_decoder.
```
