#include "codedllm/kernels/cuda_utils.cuh"
#include "codedllm/kernels/tuning.cuh"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

namespace {

__global__ void empty_kernel() {}

}  // namespace

int main() {
  constexpr int kDeviceId = 0;
  CUDA_CHECK(cudaSetDevice(kDeviceId));

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, kDeviceId));
  int memory_clock_rate_khz = 0;
  int memory_bus_width_bits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_rate_khz,
                                    cudaDevAttrMemoryClockRate, kDeviceId));
  CUDA_CHECK(cudaDeviceGetAttribute(&memory_bus_width_bits,
                                    cudaDevAttrGlobalMemoryBusWidth, kDeviceId));
  const double l2_cache_mib =
      static_cast<double>(properties.l2CacheSize) / (1024.0 * 1024.0);
  const double theoretical_bandwidth_gbps =
      (memory_clock_rate_khz * 1000.0) *
      (memory_bus_width_bits / 8.0) * 2.0 / 1.0e9;
  const codedllm::kernels::DeviceTuningConfig config =
      codedllm::kernels::GetOptimalLaunchConfig(empty_kernel, kDeviceId);

  std::cout << "CodedLLM CUDA Device Report\n"
            << "----------------------------\n"
            << "Device: " << properties.name << '\n'
            << "Compute capability: " << properties.major << '.'
            << properties.minor << '\n'
            << "Streaming multiprocessors: " << properties.multiProcessorCount
            << '\n'
            << "Maximum threads per block: " << properties.maxThreadsPerBlock
            << '\n'
            << "Warp size: " << properties.warpSize << '\n'
            << std::fixed << std::setprecision(2)
            << "L2 cache: " << l2_cache_mib << " MiB\n"
            << "Theoretical memory bandwidth: "
            << theoretical_bandwidth_gbps << " GB/s\n\n"
            << "Occupancy recommendation for empty_kernel\n"
            << "Threads per block: " << config.threads_per_block << '\n'
            << "Resident blocks per SM: " << config.blocks_per_sm << '\n'
            << "Grid block cap: " << config.max_grid_blocks << '\n'
            << "Vectorization threshold: "
            << config.vectorization_threshold_words << " uint32 words\n";

  return 0;
}
